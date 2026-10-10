"""RD-07 MrhReader: 가짜 DiT·VAE + 진짜 load_teacher/TeacherSampler/TapBus + 소형 MRH. 탭은 독립 오라클과 비트 일치."""
import pytest
import torch

import rd07_helpers as H
from buc.gov import prov as P
from buc.read import mrh_call as M
from buc.teacher import taps as T
from buc.teacher.sample import DEFAULT_TABLE

IMG = H.canvas_display(H.make_image(64, 96, seed=31))        # [1,3,64,96] ∈ [-1,1]
SIG = DEFAULT_TABLE.points[-1]                               # 0.3 (잠정 표의 마지막 점)


@pytest.fixture()
def stack(tmp_path):
    return H.make_stack(tmp_path)


# ----------------------------------------------------------------------------------------- 프로토콜

def test_snap_uses_teacher_table(stack):
    r = stack.reader
    assert r.snap(0.2) == pytest.approx(0.3) and r.snap(0.35) == pytest.approx(0.3) and r.snap(0.25) == pytest.approx(0.3)
    assert r.snap(0.9) == pytest.approx(0.9) and r.snap(0.55) == pytest.approx(0.5)


def test_single_forward_with_snapped_sigma_and_t_equals_one_minus_sigma(stack):
    v = stack.reader(IMG, 0.2, prov=H.ENGINE, key="a")
    assert isinstance(v, M.ViewRead) and v.sigma == pytest.approx(SIG)
    calls = stack.dit.calls
    assert len(calls) == 1, "뷰당 변환기 forward는 정확히 1회"
    c = calls[0]
    assert c.n == 1 and c.return_dict is False and c.t_dtype == torch.float32
    assert c.t.tolist() == pytest.approx([1.0 - SIG]) and c.x_shapes == [(16, 1, 8, 12)]
    assert c.grad_enabled is False
    assert [float(x.t) for x in stack.spy.calls] == [pytest.approx(SIG)], "MRH t = 스냅한 σ"
    assert stack.spy.calls[0].grad is False


def test_taps_match_independent_oracle_bit_exactly_and_mrh_gets_dequantized_features(stack):
    v = stack.reader(IMG, 0.35, prov=H.ENGINE, key="a")
    q, z = H.oracle_taps(stack, IMG, SIG, "a")
    assert tuple(q[8].shape) == (1, H.K, 4, 6) and q[8].dtype == torch.int8
    ts = T.TapSet(layers=q, grid=(4, 6), basis_fp=stack.basis.fingerprint, prov=P.as_record(H.ENGINE),
                  ephemeral=False, key="a")
    assert v.digest == ts.digest() and len(v.digest) == 32 and v.ephemeral is False
    feats = ts.features(stack.basis)
    seen = stack.spy.calls[0].taps
    assert set(seen) == set(H.LAYERS)
    for l in H.LAYERS:
        assert torch.equal(seen[l], feats[l])                  # MRH 입력 = TapSet.features(basis)
    with torch.no_grad():
        want = stack.mrh(feats, SIG)
    assert set(v.outputs) == set(want)
    for k in want:
        torch.testing.assert_close(v.outputs[k], want[k], atol=1e-6, rtol=0)
        assert v.outputs[k].dtype == torch.float32 and tuple(v.outputs[k].shape[-2:]) == (16, 24)
        assert not v.outputs[k].requires_grad
    assert tuple(v.outputs["sem"].shape) == (1, 8, 16, 24)


def test_noise_follows_key_and_seed_but_is_otherwise_deterministic(tmp_path):
    s = H.make_stack(tmp_path)
    a1 = s.reader(IMG, 0.3, prov=H.ENGINE, key="a")
    a2 = s.reader(IMG, 0.3, prov=H.ENGINE, key="a")
    b = s.reader(IMG, 0.3, prov=H.ENGINE, key="b")
    assert a1.digest == a2.digest and b.digest != a1.digest
    for k in a1.outputs:
        assert torch.equal(a1.outputs[k], a2.outputs[k])
    s2 = H.make_stack(tmp_path / "x", seed=5)
    assert s2.reader(IMG, 0.3, prov=H.ENGINE, key="a").digest != a1.digest      # 시드가 다르면 노이즈가 다르다


def test_token_offset_selects_image_tokens(tmp_path):
    s0 = H.make_stack(tmp_path / "o0", token_offset=0)
    s2 = H.make_stack(tmp_path / "o2", token_offset=H.OFF)
    d0 = s0.reader(IMG, 0.3, prov=H.ENGINE, key="a").digest
    d2 = s2.reader(IMG, 0.3, prov=H.ENGINE, key="a").digest
    assert d0 != d2
    q, _ = H.oracle_taps(s0, IMG, SIG, "a", token_offset=0)
    ts = T.TapSet(layers=q, grid=(4, 6), basis_fp=s0.basis.fingerprint, prov=P.as_record(H.ENGINE),
                  ephemeral=False, key="a")
    assert d0 == ts.digest()


# ----------------------------------------------------------------------------------------- 부작용 없음

def test_mrh_weights_and_modes_untouched_no_grad_and_no_tapset_kept(stack):
    h0 = H.weights_hash(stack.mrh)
    tr0 = stack.mrh.training
    stack.reader(IMG, 0.3, prov=H.ENGINE, key="a")
    assert H.weights_hash(stack.mrh) == h0 and stack.mrh.training == tr0
    assert all(p.grad is None for p in stack.mrh.parameters())
    for obj in (stack.reader, stack.bus):
        assert not any(isinstance(v, T.TapSet) for v in vars(obj).values())
    stack.reader(IMG, 0.3, prov=H.ENGINE, key="a2")             # 버스가 여전히 쓸 수 있다


def test_global_rng_untouched(stack):
    stack.reader(IMG, 0.3, prov=H.ENGINE, key="warm")          # 가짜 텍스트 인코더의 지연 생성(가중치 초기화)을 먼저 치른다
    t0 = torch.get_rng_state()
    stack.reader(IMG, 0.3, prov=H.ENGINE, key="a")
    assert torch.equal(torch.get_rng_state(), t0)


# ----------------------------------------------------------------------------------------- 검증은 교사 호출 전

def _forbid_teacher(stack):
    def boom(*a, **k):
        raise AssertionError("검증보다 먼저 교사가 호출됐다")
    stack.teacher.encode_image = boom
    stack.teacher.make_noise = boom


@pytest.mark.parametrize("img", [
    torch.zeros(1, 3, 40, 48),                                 # 16 배수 아님
    torch.zeros(1, 3, 48, 100),
    torch.full((1, 3, 48, 48), 1.5),                           # 범위 밖
    torch.full((1, 3, 48, 48), float("nan")),
    torch.zeros(3, 48, 48),                                    # 3차원
    torch.zeros(2, 3, 48, 48),                                 # 배치 2
    torch.zeros(1, 1, 48, 48),                                 # 채널 1
], ids=["h40", "w100", "range", "nan", "3d", "batch2", "ch1"])
def test_bad_image_rejected_before_teacher(stack, img):
    _forbid_teacher(stack)
    with pytest.raises((ValueError, TypeError)):
        stack.reader(img, 0.3, prov=H.ENGINE, key="a")
    assert stack.dit.calls == [] and stack.spy.calls == [] and stack.bus.stats["captures"] == 0


def test_bad_prov_rejected_before_teacher(stack):
    _forbid_teacher(stack)
    for bad in (None, "USER"):
        with pytest.raises((TypeError, ValueError)):
            stack.reader(IMG, 0.3, prov=bad, key="a")
    assert stack.dit.calls == []


def test_detached_bus_is_an_error_before_teacher(stack):
    stack.bus.detach()
    with pytest.raises((M.MrhCallError, T.TapError, ValueError)):
        stack.reader(IMG, 0.3, prov=H.ENGINE, key="a")
    assert stack.dit.calls == []


def test_constructor_validation(tmp_path):
    from buc.models.mrh.heads import MRH
    s = H.make_stack(tmp_path)
    bad_layers = MRH(H.tiny_mrh_cfg(tap_layers=(7, 15, 23)), seed=0)
    bad_dim = MRH(H.tiny_mrh_cfg(tap_dim=H.K + 1), seed=0)
    for mrh in (bad_layers, bad_dim):
        with pytest.raises(M.MrhCallSpecError):
            M.MrhReader(s.sampler, s.bus, mrh)
    good = MRH(H.tiny_mrh_cfg(), seed=0)
    for kw in (dict(seed=-1), dict(seed=1 << 64), dict(token_offset=-1), dict(caption=3)):
        with pytest.raises((M.MrhCallSpecError, TypeError, ValueError)):
            M.MrhReader(s.sampler, s.bus, good, **kw)
    with pytest.raises((TypeError, M.MrhCallSpecError)):
        M.MrhReader(s.sampler, s.bus, good, ring="ring")


# ----------------------------------------------------------------------------------------- 클린룸 레인(탭)

def test_engine_lane_pushes_to_ring_ext_lane_never_does(tmp_path):
    ring = T.TapRing(H.make_basis(), max_entries=16)
    s = H.make_stack(tmp_path, ring=ring)
    v1 = s.reader(IMG, 0.3, prov=H.ENGINE, key="e1")
    assert len(ring) == 1 and s.bus.stats["pushed"] == 1 and v1.ephemeral is False
    for tag in (["USER"], ["USER-DERIVED"], ["EXTGEN:test"], ["BIO"]):
        v = s.reader(IMG, 0.3, prov=P.as_record(tag), key="x" + tag[0])
        assert v.ephemeral is True, tag
    assert len(ring) == 1, "ext 호출은 링에 들어가지 않는다"
    assert s.bus.stats["pushed"] == 1 and s.bus.stats["bypassed"] == 4 and s.bus.stats["captures"] == 5
    # ext 호출의 탭도 값은 같다: 같은 이미지·키라도 prov만 다른 호출은 digest가 다르다(Merkle prov 비트)
    a = s.reader(IMG, 0.3, prov=H.USER, key="same")
    b = s.reader(IMG, 0.3, prov=H.ENGINE, key="same")
    assert a.digest != b.digest
    for k in a.outputs:
        torch.testing.assert_close(a.outputs[k], b.outputs[k], atol=0, rtol=0)   # 판독 값 자체는 prov와 무관


def test_not_train_ok_non_ext_is_bypassed_too(tmp_path):
    ring = T.TapRing(H.make_basis(), max_entries=4)
    s = H.make_stack(tmp_path, ring=ring)
    v = s.reader(IMG, 0.3, prov=P.as_record(["HOLDOUT"]), key="h")
    assert len(ring) == 0 and s.bus.stats["bypassed"] == 1 and v.ephemeral is False
