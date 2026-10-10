"""TCH-03 Tap Bus: 층 8/15/23 훅, 토큰 슬라이스·격자, 통과 불변, 정확히 한 번 발화, 오류 복구."""
import dataclasses

import pytest
import torch

import tch03_fakes as F
from buc.teacher import taps as T

GH, GW = F.GRID
N_IMG = GH * GW


def cap_run(bus, model, x, prov=F.ENGINE, **kw):
    kw.setdefault("grid", F.GRID)
    kw.setdefault("token_offset", F.OFF)
    with bus.capture(prov, **kw) as cap:
        with torch.no_grad():
            y = model(x)
    return cap.tapset, y


def hooked_layers(model):
    return [i for i, b in enumerate(model.layers) if len(b._forward_hooks) > 0]


def test_attach_registers_exactly_the_three_hooks_and_detach_is_clean(basis, model):
    bus = T.TapBus(basis)
    assert not bus.attached and bus.hook_count == 0
    assert bus.attach(model.layers) is bus
    assert bus.attached and bus.hook_count == 3
    assert hooked_layers(model) == [8, 15, 23]
    with pytest.raises(T.TapError):
        bus.attach(model.layers)                         # 이중 부착 금지
    assert bus.hook_count == 3
    bus.detach()
    bus.detach()                                         # 멱등
    assert hooked_layers(model) == [] and not bus.attached and bus.hook_count == 0
    bus.attach(model.layers)                             # 재부착 가능
    assert hooked_layers(model) == [8, 15, 23]
    bus.detach()


def test_ctor_and_attach_validation(basis, model):
    with pytest.raises(T.TapSpecError):
        T.TapBus(basis, layers=(8, 9))                   # 기저에 없는 층
    with pytest.raises(T.TapSpecError):
        T.TapBus(basis, layers=(8, 8))
    with pytest.raises(T.TapSpecError):
        T.TapBus(basis, layers=())
    short = F.FakeDiT(n=20)
    bus = T.TapBus(basis)
    with pytest.raises(T.TapSpecError):
        bus.attach(short.layers)                         # 층 23이 없다
    assert hooked_layers(short) == [] and not bus.attached   # 부분 부착 흔적 없음
    # 기저가 변조됐으면 부착을 거부한다
    b2 = T.PCABasis(basis.layers, basis.mean, basis.comps, basis.scale)
    b2.comps[0, 0, 0] += 0.01
    with pytest.raises(T.TapBasisError):
        T.TapBus(b2).attach(model.layers)
    assert hooked_layers(model) == []


def test_capture_shapes_and_metadata(bus, basis, model):
    x = F.make_input(1)
    ts, _ = cap_run(bus, model, x, key="k1")
    assert tuple(sorted(ts.layers)) == T.TAP_LAYERS
    for l in T.TAP_LAYERS:
        q = ts.layers[l]
        assert q.dtype == torch.int8 and q.shape == (2, F.K, GH, GW) and not q.requires_grad
    assert ts.grid == F.GRID and ts.basis_fp == basis.fingerprint and ts.key == "k1"
    assert ts.prov == F.ENGINE and ts.ephemeral is False
    assert ts.nbytes == 3 * 2 * F.K * N_IMG                    # int8: 층당 B*k*gh*gw 바이트
    with pytest.raises(dataclasses.FrozenInstanceError):
        ts.key = "other"


def test_taps_match_float64_oracle_with_rowmajor_token_layout(bus, basis, model):
    x = F.make_input(2)
    acts = F.collect_acts(model, x)
    ts, _ = cap_run(bus, model, x)
    for l in T.TAP_LAYERS:
        img = acts[l][:, F.OFF:F.OFF + N_IMG]                          # 이미지 토큰 구간
        want = F.oracle_quant(basis, l, img).reshape(2, GH, GW, F.K).permute(0, 3, 1, 2)   # 행 우선 (gh, gw)
        F.assert_quant_close(ts.layers[l], want)
    # 토큰 순서가 실제로 행 우선인지: 격자를 전치하면 달라야 한다
    l = 8
    img = acts[l][:, F.OFF:F.OFF + N_IMG]
    swapped = F.oracle_quant(basis, l, img).reshape(2, GW, GH, F.K).permute(0, 3, 2, 1)
    assert (swapped != ts.layers[l].to(torch.int64)).any()


def test_only_the_image_token_slice_is_tapped(bus, model):
    x = F.make_input(3)
    ts1, _ = cap_run(bus, model, x)
    x2 = x.clone()
    x2[:, :F.OFF] += 5.0                                   # 앞 비이미지 토큰
    x2[:, F.OFF + N_IMG:] -= 5.0                           # 뒤 비이미지 토큰
    ts2, _ = cap_run(bus, model, x2)                       # 블록이 토큰별 독립이라 영향이 없어야 한다
    for l in T.TAP_LAYERS:
        assert torch.equal(ts1.layers[l], ts2.layers[l])
    x3 = x.clone()
    x3[:, F.OFF + 5] += 5.0                                # 이미지 토큰 하나를 바꾸면 해당 격자 위치만 바뀐다
    ts3, _ = cap_run(bus, model, x3)
    diff = (ts1.layers[8] != ts3.layers[8]).any(dim=1)     # [B, gh, gw]
    assert diff.any() and int(diff.sum()) == 2 and bool(diff[:, 5 // GW, 5 % GW].all())


def test_token_offset_zero_and_other_grids(bus, model):
    x = F.make_input(4, seq=1 + 6 * 2 + 1)
    ts, _ = cap_run(bus, model, x, grid=(6, 2), token_offset=1)
    assert ts.layers[15].shape == (2, F.K, 6, 2) and ts.grid == (6, 2)
    ts, _ = cap_run(bus, model, F.make_input(4, seq=12), grid=(3, 4), token_offset=0)
    assert ts.layers[23].shape == (2, F.K, 3, 4)


def test_hooks_do_not_alter_model_output_or_autograd(basis):
    m = F.FakeDiT().eval()
    x = F.make_input(5)
    ref = m(x.clone())
    bus = T.TapBus(basis).attach(m.layers)
    try:
        assert torch.equal(m(x.clone()), ref)                      # 부착만, 캡처 밖
        xin = x.clone().requires_grad_(True)
        with bus.capture(F.ENGINE, grid=F.GRID, token_offset=F.OFF) as cap:
            y = m(xin)
        assert torch.equal(y.detach(), ref)                        # 캡처 중에도 출력 비트 동일
        assert y.requires_grad and y.grad_fn is not None           # 하류 그래프는 그대로
        y.sum().backward()
        assert xin.grad is not None and torch.isfinite(xin.grad).all()
        assert all(not q.requires_grad and q.grad_fn is None for q in cap.tapset.layers.values())
    finally:
        bus.detach()


def test_outside_capture_nothing_is_recorded(bus, model):
    model(F.make_input(6))
    assert bus.stats["captures"] == 0
    ts, _ = cap_run(bus, model, F.make_input(6))
    model(F.make_input(7))                                          # 캡처 밖 forward가 이전 결과를 바꾸지 않는다
    assert bus.stats["captures"] == 1
    assert ts.layers[8].shape[0] == 2


def test_tapset_unavailable_until_capture_finishes(bus, model):
    with bus.capture(F.ENGINE, grid=F.GRID, token_offset=F.OFF) as cap:
        with pytest.raises(T.TapError):
            cap.tapset
        model(F.make_input(8))
    assert cap.tapset is not None


def test_each_tapped_layer_must_fire_exactly_once(bus, model):
    x = F.make_input(9)
    with pytest.raises(T.TapError):
        with bus.capture(F.ENGINE, grid=F.GRID, token_offset=F.OFF):
            model(x, twice=(8,))                                    # 층 8이 두 번
    with pytest.raises(T.TapError):
        with bus.capture(F.ENGINE, grid=F.GRID, token_offset=F.OFF):
            model(x, upto=20)                                       # 층 23이 안 불렸다
    ts, _ = cap_run(bus, model, x)                                  # 오류 뒤에도 버스는 재사용 가능
    assert tuple(sorted(ts.layers)) == T.TAP_LAYERS
    assert bus.stats["captures"] == 1                               # 실패한 캡처는 세지 않는다


def test_exception_in_body_propagates_and_resets(bus, model):
    with pytest.raises(KeyError):
        with bus.capture(F.ENGINE, grid=F.GRID, token_offset=F.OFF):
            model(F.make_input(1), upto=10)
            raise KeyError("boom")
    ts, _ = cap_run(bus, model, F.make_input(1))
    assert ts.layers[8].shape == (2, F.K, GH, GW)                   # 이전 부분 결과가 섞이지 않는다


def test_capture_state_errors(basis, model):
    bus = T.TapBus(basis)
    with pytest.raises(T.TapError):
        with bus.capture(F.ENGINE, grid=F.GRID):
            pass                                                    # 부착 전
    bus.attach(model.layers)
    with bus.capture(F.ENGINE, grid=F.GRID, token_offset=F.OFF):
        with pytest.raises(T.TapError):
            with bus.capture(F.ENGINE, grid=F.GRID, token_offset=F.OFF):
                pass                                                # 중첩 금지
        model(F.make_input(1))
    bus.detach()
    with pytest.raises(T.TapError):
        with bus.capture(F.ENGINE, grid=F.GRID):
            pass                                                    # 분리 뒤


@pytest.mark.parametrize("grid,off", [((0, 3), 0), ((3, 0), 0), ((-1, 2), 0), ((3,), 0), ((3.0, 4), 0),
                                      ((3, 4), -1), ((3, 4), True)])
def test_invalid_grid_or_offset(bus, grid, off):
    with pytest.raises(T.TapSpecError):
        with bus.capture(F.ENGINE, grid=grid, token_offset=off):
            pass


def test_tuple_output_blocks_are_supported(basis):
    plain, tup = F.FakeDiT(), F.FakeDiT(tuple_layers=(8, 15))
    x = F.make_input(10)
    b1 = T.TapBus(basis).attach(plain.layers)
    b2 = T.TapBus(basis).attach(tup.layers)
    t1, _ = cap_run(b1, plain, x)
    t2, _ = cap_run(b2, tup, x)
    for l in T.TAP_LAYERS:
        assert torch.equal(t1.layers[l], t2.layers[l])
    b1.detach()
    b2.detach()


@pytest.mark.parametrize("case", ["dim", "ndim", "short"])
def test_bad_layer_output_layout_raises_and_recovers(basis, case):
    m = F.FakeDiT(n=24)
    # 버스 훅보다 먼저 등록해 버스가 '망가진' 출력을 보게 한다 (훅은 등록 순서대로 실행)
    if case == "dim":
        x = F.make_input(1)
        m.layers[8].register_forward_hook(lambda mod, a, out: torch.cat([out, out], dim=-1))
    elif case == "ndim":
        x = F.make_input(1)
        m.layers[8].register_forward_hook(lambda mod, a, out: out[0])
    else:
        x = F.make_input(1, seq=F.OFF + N_IMG - 1)
    bus = T.TapBus(basis).attach(m.layers)
    with pytest.raises(T.TapSpecError):
        with bus.capture(F.ENGINE, grid=F.GRID, token_offset=F.OFF):
            m(x)
    bus.detach()
    m2 = F.FakeDiT()
    b2 = T.TapBus(basis).attach(m2.layers)
    ts, _ = cap_run(b2, m2, F.make_input(1))                      # 같은 기저로 새 모델은 정상
    assert ts.layers[8].shape == (2, F.K, GH, GW)
    b2.detach()


def test_nonfinite_activation_is_rejected(bus, model):
    x = F.make_input(1)
    x[0, F.OFF + 1, 0] = float("nan")
    with pytest.raises(T.TapNumericError):
        cap_run(bus, model, x)
    ts, _ = cap_run(bus, model, F.make_input(1))
    assert ts.layers[8].shape[0] == 2


def test_bf16_activations_are_projected_in_float32(basis):
    m = F.FakeDiT().to(torch.bfloat16).eval()
    bus = T.TapBus(basis).attach(m.layers)
    x = F.make_input(11).to(torch.bfloat16)
    acts = F.collect_acts(m, x)
    ts, _ = cap_run(bus, m, x)
    for l in T.TAP_LAYERS:
        assert ts.layers[l].dtype == torch.int8
        want = F.oracle_quant(basis, l, acts[l][:, F.OFF:F.OFF + N_IMG].float())
        want = want.reshape(2, GH, GW, F.K).permute(0, 3, 1, 2)
        F.assert_quant_close(ts.layers[l], want)
    bus.detach()


def test_capture_is_deterministic_and_digest_is_sensitive(bus, model):
    x = F.make_input(12)
    a, _ = cap_run(bus, model, x, key="a")
    b, _ = cap_run(bus, model, x, key="a")
    assert a.digest() == b.digest() and len(a.digest()) == 32
    assert all(torch.equal(a.layers[l], b.layers[l]) for l in T.TAP_LAYERS)
    assert cap_run(bus, model, x, key="b")[0].digest() != a.digest()          # key
    assert cap_run(bus, model, x + 0.5, key="a")[0].digest() != a.digest()    # 입력
    assert cap_run(bus, model, x, key="a", prov=["HOLDOUT"])[0].digest() != a.digest()   # prov 비트·ns (merkle_parts)
    assert cap_run(bus, model, x, key="a", grid=(4, 3))[0].digest() != a.digest()        # 격자


def test_features_dequantize_with_matching_basis_only(bus, basis, model):
    ts, _ = cap_run(bus, model, F.make_input(13))
    feats = ts.features(basis)
    assert tuple(sorted(feats)) == T.TAP_LAYERS
    for i, l in enumerate(basis.layers):
        f, q = feats[l], ts.layers[l]
        assert f.dtype == torch.float32 and f.shape == q.shape
        want = q.float() * basis.scale[i].view(1, -1, 1, 1)
        assert torch.allclose(f, want, rtol=1e-6, atol=1e-7)
    other = F.make_basis(seed=99)
    with pytest.raises(T.TapBasisError):
        ts.features(other)
