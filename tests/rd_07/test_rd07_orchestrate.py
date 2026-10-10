"""RD-07 오케스트레이터 read_mrh: 호출 순서·입력 크롭·혼합 항등·게이지 보정·모순 검출·σ 쌍·캔버스·검증 우선."""
import json

import numpy as np
import pytest
import torch

import rd07_helpers as H
from buc.gov import prov as P
from buc.read import mrh_call as M

IMG_HW = (96, 128)
# 전역 = 캔버스 그대로(업스케일 없음 → 예산이 크면 축소 안 함), 타일 64 → 2x2, 페더 16 px
CFG = dict(tile_px=64, feather_px=16, global_pixels=10 ** 9)


def _cfg(**kw):
    return M.ReadConfig(**{**CFG, **kw})


def _run(reader, img=None, *, cfg=None, prov=H.ENGINE, key="k", **kw):
    img = H.make_image(*IMG_HW, seed=7) if img is None else img
    return img, M.read_mrh(img, reader, prov=prov, cfg=_cfg() if cfg is None else cfg, key=key, **kw)


def _stub_full(img_lin, gauge=(1.0, 0.0)):
    return H.stub_outputs(H.canvas_display(img_lin), gauge)


# ----------------------------------------------------------------------------------------- ReadConfig

def test_read_config_defaults_and_validation():
    c = M.ReadConfig()
    assert tuple(c.global_sigmas) == (0.20, 0.35) and c.tile_sigma == 0.25 and c.tiles is True
    assert (c.tile_px, c.global_pixels, c.read_long_max, c.feather_px) == (1024, 589_824, 3840, 64)
    from buc.models.mrh.heads import HEAD_NAMES
    assert tuple(c.heads) == tuple(HEAD_NAMES)
    with pytest.raises(Exception):
        c.tile_px = 64                                          # frozen
    with pytest.raises(TypeError):
        M.ReadConfig(0.2)                                       # kw_only
    bad = [dict(global_sigmas=()), dict(global_sigmas=(0.1, 0.2, 0.3)), dict(global_sigmas=(0.0,)),
           dict(global_sigmas=(1.5,)), dict(global_sigmas=(float("nan"),)), dict(tile_sigma=0.0),
           dict(tile_px=100), dict(tile_px=0), dict(feather_px=-4), dict(feather_px=6), dict(global_pixels=10),
           dict(read_long_max=100), dict(heads=()), dict(heads=("normal",)), dict(heads=("log_depth", "bogus")),
           dict(depth_floor=0.0), dict(unc_tau=-1.0)]
    for kw in bad:
        with pytest.raises(M.MrhCallSpecError):
            M.ReadConfig(**kw)


# ----------------------------------------------------------------------------------------- 호출 순서·입력

def test_call_sequence_inputs_and_bookkeeping():
    stub = H.StubReader()
    img, r = _run(stub, cfg=_cfg())
    x = H.canvas_display(img)
    assert [c.key for c in stub.calls] == ["k/g0", "k/g1", "k/t0", "k/t1", "k/t2", "k/t3"]
    assert [c.sigma for c in stub.calls][:2] == pytest.approx([0.20, 0.35])
    assert [c.sigma for c in stub.calls][2:] == pytest.approx([0.25] * 4)
    assert r.n_forward == 6 == len(r.digests)
    for c in stub.calls[:2]:                                    # 전역 = 캔버스(예산이 커서 축소 없음)
        assert tuple(c.image.shape) == (1, 3, 96, 128) and c.image.dtype == torch.float32
        torch.testing.assert_close(c.image, x, atol=1e-6, rtol=0)
    boxes = M.plan_tiles(96, 128, tile=64)
    assert tuple(r.tiles) == tuple(boxes) and len(boxes) == 4
    for c, b in zip(stub.calls[2:], boxes):                     # 타일 = 캔버스 크롭 그대로
        torch.testing.assert_close(c.image, x[:, :, b.y0:b.y1, b.x0:b.x1], atol=1e-6, rtol=0)
    assert all(float(c.image.min()) >= -1.0 and float(c.image.max()) <= 1.0 for c in stub.calls)
    assert all(c.prov == P.as_record(H.ENGINE) for c in stub.calls)
    assert r.global_size == (96, 128) and r.canvas_hw == (96, 128) and r.read_hw == r.src_hw == (96, 128)
    assert tuple(r.global_sigmas) == pytest.approx((0.20, 0.35)) and r.tile_sigma == pytest.approx(0.25)
    assert r.sigma_pair_degenerate is False and r.key == "k"
    assert len(set(r.digests)) >= 1 and all(isinstance(d, bytes) for d in r.digests)


def test_global_view_is_downscaled_to_budget():
    stub = H.StubReader()
    img = H.make_image(*IMG_HW, seed=8)
    r = M.read_mrh(img, stub, prov=H.ENGINE, cfg=_cfg(global_pixels=3072, tiles=False), key="k")
    for c in stub.calls:
        assert tuple(c.image.shape) == (1, 3, 48, 64)
    assert r.global_size == (48, 64) and r.n_forward == 2 and r.tiles == ()
    # 출력은 항상 캔버스 격자(24x32)
    assert tuple(r.outputs["log_depth"].shape) == (1, 1, 24, 32) and tuple(r.outputs["sem"].shape) == (1, 8, 24, 32)


def test_identity_blend_equals_global_for_every_head():
    stub = H.StubReader()
    img, r = _run(stub)
    want = _stub_full(img)
    assert set(r.outputs) == set(want) and tuple(r.outputs["normal"].shape) == (1, 3, 24, 32)
    for k in want:
        torch.testing.assert_close(r.outputs[k], want[k], atol=2e-5, rtol=0)
        assert r.outputs[k].dtype == torch.float32 and bool(torch.isfinite(r.outputs[k]).all())
    for k in ("normal", "sem"):
        assert float((r.outputs[k].norm(dim=1) - 1).abs().max()) < 1e-5
    torch.testing.assert_close(r.depth_global, want["log_depth"], atol=2e-5, rtol=0)
    assert r.r_dis < 1e-3 and r.r_dis_tiles < 1e-3 and r.r_dis_sigma < 1e-3
    assert not bool(r.uncertain.any()) and r.uncertain_frac == 0.0
    assert tuple(r.coverage.shape) == (24, 32) and float(r.coverage.min()) > 0.0
    assert tuple(r.r_dis_map.shape) == (24, 32) and r.uncertain.dtype == torch.bool
    assert len(r.tile_affine) == 4


def test_gauge_mismatch_between_tiles_and_sigmas_is_absorbed_into_first_sigma_frame():
    gauge = {"k/g0": (1.7, 0.4), "k/g1": (0.6, -2.0), "k/t0": (0.5, 3.0), "k/t1": (2.0, -1.0),
             "k/t2": (1.3, 0.2), "k/t3": (0.8, 0.0)}
    stub = H.StubReader(gauge=gauge)
    img, r = _run(stub)
    want = _stub_full(img, (1.7, 0.4))
    torch.testing.assert_close(r.outputs["log_depth"], want["log_depth"], atol=2e-4, rtol=0)
    torch.testing.assert_close(r.depth_global, want["log_depth"], atol=2e-4, rtol=0)
    assert r.r_dis < 2e-3
    for (a, b), key in zip(r.tile_affine, ("k/t0", "k/t1", "k/t2", "k/t3")):
        A, B = gauge[key]
        assert a == pytest.approx(1.7 / A, rel=2e-3) and b == pytest.approx(0.4 - 1.7 * B / A, abs=5e-3)
    for k in want:                                              # 깊이 외 헤드는 게이지와 무관
        if k != "log_depth":
            torch.testing.assert_close(r.outputs[k], want[k], atol=2e-5, rtol=0)


def test_contradictory_tile_raises_r_dis_and_marks_region_uncertain():
    stub = H.StubReader(depth_fn={"k/t1": lambda g4: 2.0 * g4.flip(-1)})
    img, r = _run(stub)
    assert r.r_dis_tiles > 0.08 and r.r_dis >= r.r_dis_tiles - 1e-9
    assert r.r_dis_sigma < 1e-3
    assert r.r_dis == pytest.approx(max(r.r_dis_tiles, r.r_dis_sigma), rel=1e-6)
    hot, cold = r.r_dis_map[1:7, 20:32], r.r_dis_map[0:6, 0:12]
    assert float(hot.mean()) > 0.10 and float(cold.max()) < 1e-3
    assert bool(r.uncertain[1:7, 20:32].any()) and not bool(r.uncertain[0:6, 0:12].any())
    assert 0.0 < r.uncertain_frac < 0.5
    assert r.uncertain_frac == pytest.approx(float(r.uncertain.float().mean()), abs=1e-9)
    # 같은 입력에서 unc_tau를 올리면 불확실 영역이 줄어든다(r_dis 자체는 불변)
    r2 = M.read_mrh(img, H.StubReader(depth_fn={"k/t1": lambda g4: 2.0 * g4.flip(-1)}), prov=H.ENGINE,
                    cfg=_cfg(unc_tau=5.0), key="k")
    assert r2.uncertain_frac == 0.0 and r2.r_dis == pytest.approx(r.r_dis, rel=1e-6)


def test_sigma_pair_disagreement_gives_r_dis_sigma_matching_oracle():
    pert = (np.sin(np.linspace(0, 9, 24))[:, None] * np.cos(np.linspace(0, 7, 32))[None, :]) * 0.12
    pert_t = torch.from_numpy(pert).float()[None, None]
    stub = H.StubReader(gauge={"k/g0": (1.0, 0.0), "k/g1": (2.0, 1.0)},
                        post={"k/g1": lambda o: dict(o, log_depth=o["log_depth"] + pert_t)})
    img = H.make_image(*IMG_HW, seed=9)
    r = M.read_mrh(img, stub, prov=H.ENGINE, cfg=_cfg(tiles=False), key="k")
    full = _stub_full(img)
    d1 = full["log_depth"]
    d2 = 2.0 * d1 + 1.0 + pert_t
    assert r.r_dis_sigma == pytest.approx(H.oracle_rel(d2.numpy(), d1.numpy()), rel=3e-3, abs=3e-4)
    assert r.r_dis_sigma > 0.03 and r.r_dis_tiles == 0.0 and r.n_forward == 2
    assert r.r_dis == pytest.approx(r.r_dis_sigma, rel=1e-6)
    a, b = H.oracle_affine(d2.numpy(), d1.numpy())
    want = 0.5 * (d1.double() + a * d2.double() + b).float()
    torch.testing.assert_close(r.outputs["log_depth"], want, atol=2e-4, rtol=0)
    assert float(r.r_dis_map.mean()) > 0.0


def test_degenerate_sigma_pair_calls_global_once():
    stub = H.StubReader(points=(1.0, 0.5, 0.3))                 # snap(0.2) == snap(0.35) == 0.3
    img, r = _run(stub, cfg=_cfg(tiles=False))
    assert [c.key for c in stub.calls] == ["k/g0"] and stub.calls[0].sigma == pytest.approx(0.3)
    assert r.sigma_pair_degenerate is True and r.r_dis_sigma == 0.0 and r.n_forward == 1
    assert tuple(r.global_sigmas) == pytest.approx((0.3,))
    # 한 개만 달라고 하면 퇴화가 아니다
    stub2 = H.StubReader(points=(1.0, 0.5, 0.3))
    _, r2 = _run(stub2, cfg=_cfg(tiles=False, global_sigmas=(0.2,)))
    assert r2.sigma_pair_degenerate is False and r2.n_forward == 1


def test_no_tiles_when_flag_off_or_image_fits_one_tile():
    stub = H.StubReader()
    img, r = _run(stub, cfg=_cfg(tiles=False))
    assert r.n_forward == 2 and r.tiles == () and tuple(r.tile_affine) == () and r.tile_sigma is None
    assert r.r_dis_tiles == 0.0 and float(r.coverage.abs().max()) == 0.0
    for k, v in _stub_full(img).items():
        torch.testing.assert_close(r.outputs[k], v, atol=2e-5, rtol=0)
    stub2 = H.StubReader()
    small = H.make_image(48, 64, seed=3)
    r2 = M.read_mrh(small, stub2, prov=H.ENGINE, cfg=_cfg(), key="k")   # 긴 변 64 ≤ tile_px 64 → 타일 없음
    assert r2.n_forward == 2 and r2.tiles == ()


def test_odd_size_is_padded_to_canvas_and_cropped_back():
    stub = H.StubReader()
    img = H.make_image(90, 122, seed=10)
    r = M.read_mrh(img, stub, prov=H.ENGINE, cfg=_cfg(tiles=False), key="k")
    assert r.src_hw == r.read_hw == (90, 122) and r.canvas_hw == (96, 128)
    assert tuple(stub.calls[0].image.shape) == (1, 3, 96, 128)
    full = H.stub_outputs(H.canvas_display(img))                # edge 패딩 캔버스
    for k, v in full.items():
        assert tuple(r.outputs[k].shape[-2:]) == (23, 31)       # ceil(90/4), ceil(122/4)
        torch.testing.assert_close(r.outputs[k], v[..., :23, :31], atol=2e-5, rtol=0)
    assert tuple(r.coverage.shape) == (23, 31) and tuple(r.r_dis_map.shape) == (23, 31)


def test_long_side_is_area_downscaled_before_reading():
    stub = H.StubReader()
    img = H.make_image(192, 256, seed=11)
    r = M.read_mrh(img, stub, prov=H.ENGINE, cfg=_cfg(read_long_max=128, tiles=False), key="k")
    assert r.src_hw == (192, 256) and r.read_hw == (96, 128)
    half = img.reshape(96, 2, 128, 2, 3).mean(axis=(1, 3))      # 2x2 면적 평균(선형 공간)
    torch.testing.assert_close(stub.calls[0].image, H.canvas_display(half.astype(np.float32)), atol=2e-5, rtol=0)
    assert tuple(r.outputs["log_depth"].shape[-2:]) == (24, 32)


def test_heads_subset_controls_output_keys():
    stub = H.StubReader()
    _, r = _run(stub, cfg=_cfg(heads=("log_depth", "normal", "defocus")))
    assert set(r.outputs) == {"log_depth", "normal", "defocus"}


# ----------------------------------------------------------------------------------------- 검증 우선·실패

@pytest.mark.parametrize("mut", [
    lambda a: a.astype(np.int32),                               # 정수 dtype
    lambda a: a[..., 0],                                        # 2차원
    lambda a: a[..., :2],                                       # 채널 2
    lambda a: np.where(a > 0.5, np.nan, a),                     # 비유한
    lambda a: a - 1.0,                                          # 음수
    lambda a: a[:8, :],                                         # 짧은 변 < 16
    lambda a: a[:0],                                            # 빈 배열
], ids=["int", "2d", "2ch", "nan", "neg", "tiny", "empty"])
def test_bad_image_fails_before_any_reader_call(mut):
    stub = H.StubReader()
    img = H.make_image(*IMG_HW, seed=12)
    with pytest.raises((TypeError, ValueError)):
        M.read_mrh(mut(img), stub, prov=H.ENGINE, cfg=_cfg(), key="k")
    assert stub.calls == []


def test_bad_prov_key_lane_fail_before_any_reader_call():
    img = H.make_image(*IMG_HW, seed=13)
    for kw in (dict(prov=None), dict(prov="USER"), dict(prov=H.ENGINE, key=""), dict(prov=H.ENGINE, key=3)):
        stub = H.StubReader()
        with pytest.raises((TypeError, ValueError)):
            M.read_mrh(img, stub, cfg=_cfg(), **{"key": "k", **kw})
        assert stub.calls == []
    stub = H.StubReader()
    with pytest.raises(M.CleanroomError):
        M.read_mrh(img, stub, prov=H.USER, cfg=_cfg(), key="k", lane="engine")
    assert stub.calls == []


def test_bad_reader_views_are_rejected():
    img = H.make_image(*IMG_HW, seed=14)
    short = H.StubReader(post={"k/g0": lambda o: {k: v for k, v in o.items() if k != "sem"}})
    with pytest.raises(M.MrhCallError):
        M.read_mrh(img, short, prov=H.ENGINE, cfg=_cfg(), key="k")
    wrong = H.StubReader(post={"k/g0": lambda o: {k: v[..., :-1] for k, v in o.items()}})
    with pytest.raises(M.MrhCallError):
        M.read_mrh(img, wrong, prov=H.ENGINE, cfg=_cfg(), key="k")
    nan = H.StubReader(post={"k/t2": lambda o: dict(o, defocus=o["defocus"] * float("nan"))})
    with pytest.raises(M.MrhCallError):
        M.read_mrh(img, nan, prov=H.ENGINE, cfg=_cfg(), key="k")


# ----------------------------------------------------------------------------------------- 결정론·부작용

def test_deterministic_input_immutable_meta_json_and_global_rng_untouched():
    img = H.make_image(*IMG_HW, seed=15)
    keep = img.copy()
    t0, n0 = torch.get_rng_state(), np.random.get_state()
    r1 = M.read_mrh(img, H.StubReader(), prov=H.ENGINE, cfg=_cfg(), key="k")
    r2 = M.read_mrh(img, H.StubReader(), prov=H.ENGINE, cfg=_cfg(), key="k")
    assert torch.equal(torch.get_rng_state(), t0)
    n1 = np.random.get_state()
    assert n1[0] == n0[0] and np.array_equal(n1[1], n0[1]) and n1[2:] == n0[2:]
    assert np.array_equal(img, keep), "입력 배열을 바꾸면 안 된다"
    for k in r1.outputs:
        assert torch.equal(r1.outputs[k], r2.outputs[k])
    assert torch.equal(r1.r_dis_map, r2.r_dis_map) and r1.r_dis == r2.r_dis
    m = r1.meta()
    json.dumps(m, allow_nan=False)
    need = {"lane", "ext", "ns", "key", "n_forward", "n_tiles", "global_sigmas", "sigma_pair_degenerate", "r_dis",
            "r_dis_tiles", "r_dis_sigma", "uncertain_frac", "cleanroom_binary_hash"}
    assert need <= set(m)
    assert m["n_forward"] == 6 and m["n_tiles"] == 4 and m["key"] == "k" and m["lane"] == "engine"
    assert m["ext"] is False and m["global_sigmas"] == pytest.approx([0.20, 0.35])
    assert json.loads(json.dumps(m)) == m
    with pytest.raises(Exception):
        r1.r_dis = 0.0                                          # frozen


def test_float64_and_noncontiguous_inputs_are_accepted():
    img = H.make_image(*IMG_HW, seed=16)
    a = M.read_mrh(img, H.StubReader(), prov=H.ENGINE, cfg=_cfg(), key="k")
    b = M.read_mrh(np.asfortranarray(img.astype(np.float64)), H.StubReader(), prov=H.ENGINE, cfg=_cfg(), key="k")
    for k in a.outputs:
        torch.testing.assert_close(a.outputs[k], b.outputs[k], atol=2e-5, rtol=0)
