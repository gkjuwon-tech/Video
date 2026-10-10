"""RD-07 순수 수치 2: affine 정렬, 전역 σ 융합(r_dis_sigma), 타일 혼합(r_dis_tiles)."""
import numpy as np
import pytest
import torch

import rd07_helpers as H
from buc.read import mrh_call as M


def _field(seed, h, w, std=0.5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, h, w, generator=g)
    k = torch.ones(1, 1, 3, 3) / 9.0
    x = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"), k)
    return (x - x.mean()) / x.std() * std


# ----------------------------------------------------------------------------------------- affine_fit

def test_affine_fit_recovers_exact_map_even_with_zero_weight_outliers():
    src = _field(1, 12, 16)
    ref = 2.5 * src - 1.2
    a, b = M.affine_fit(src, ref)
    assert float(a) == pytest.approx(2.5, abs=1e-5) and float(b) == pytest.approx(-1.2, abs=1e-5)
    w = torch.ones_like(src)
    w[..., :4, :] = 0.0
    bad = ref.clone()
    bad[..., :4, :] = 99.0                                   # 가중치 0인 곳의 이상치는 무시된다
    a2, b2 = M.affine_fit(src, bad, w)
    assert float(a2) == pytest.approx(2.5, abs=1e-4) and float(b2) == pytest.approx(-1.2, abs=1e-4)


def test_affine_fit_matches_weighted_lstsq_oracle():
    src, ref = _field(2, 10, 14), _field(3, 10, 14) + 0.7 * _field(2, 10, 14)
    w = torch.rand(1, 1, 10, 14, generator=torch.Generator().manual_seed(4)) + 0.05
    a, b = M.affine_fit(src, ref, w)
    oa, ob = H.oracle_affine(src.numpy(), ref.numpy(), w.numpy())
    assert float(a) == pytest.approx(oa, rel=1e-4, abs=1e-5) and float(b) == pytest.approx(ob, abs=1e-5)


def test_affine_fit_rejects_negative_slope_and_constant_source():
    src = _field(5, 8, 8)
    a, b = M.affine_fit(src, -src + 0.4)                     # 뒤집힌 깊이는 정렬로 숨기지 않는다: a = 0
    assert float(a) == 0.0 and float(b) == pytest.approx(0.4 + float((-src).mean()), abs=1e-5)
    a, b = M.affine_fit(torch.full((1, 1, 8, 8), 3.0), src + 1.0)
    assert float(a) == 0.0 and float(b) == pytest.approx(float((src + 1.0).mean()), abs=1e-5)


def test_affine_fit_input_validation():
    x = _field(6, 6, 6)
    with pytest.raises(ValueError):
        M.affine_fit(x, _field(7, 6, 7))                     # 모양 불일치
    with pytest.raises(ValueError):
        M.affine_fit(x, x, -torch.ones_like(x))              # 음수 가중치
    with pytest.raises(ValueError):
        M.affine_fit(x, x, torch.zeros_like(x))              # 가중치 합 0


# ----------------------------------------------------------------------------------------- affine_residual

def test_affine_residual_is_gauge_invariant_and_matches_oracle():
    ref = _field(8, 16, 20, std=0.4)
    noise = _field(9, 16, 20, std=0.03)
    src = (ref - 0.3) / 1.7 + noise
    rel, rmap = M.affine_residual(src, ref)
    assert float(rel) == pytest.approx(H.oracle_rel(src.numpy(), ref.numpy()), rel=1e-3, abs=1e-4)
    # 게이지(ref에 ×4 +9)를 바꿔도 상대 잔차는 같다
    rel2, _ = M.affine_residual(src, 4.0 * ref + 9.0)
    assert float(rel2) == pytest.approx(float(rel), rel=1e-3, abs=1e-4)
    assert tuple(rmap.shape[-2:]) == (16, 20) and float(rmap.min()) >= 0.0
    # rms(resid_map) == rel (같은 분모)
    assert float(torch.sqrt((rmap.double() ** 2).mean())) == pytest.approx(float(rel), rel=1e-3, abs=1e-4)
    rel0, rmap0 = M.affine_residual(2.0 * ref + 1.0, ref)
    assert float(rel0) < 1e-5 and float(rmap0.max()) < 1e-4


def test_affine_residual_floor_and_flip():
    flat = _field(10, 12, 12, std=0.002)                     # 퍼짐 < floor
    src = flat + _field(11, 12, 12, std=0.01)
    rel, _ = M.affine_residual(src, flat)
    assert float(rel) == pytest.approx(H.oracle_rel(src.numpy(), flat.numpy(), floor=0.05), rel=1e-3, abs=1e-4)
    assert float(rel) < 0.35                                 # floor가 없으면 분모가 0.002라 rel ≈ 1이 된다
    rel_big, _ = M.affine_residual(src, flat, floor=1.0)     # floor 인자가 분모에 쓰인다
    assert float(rel_big) == pytest.approx(H.oracle_rel(src.numpy(), flat.numpy(), floor=1.0), rel=1e-3, abs=1e-4)
    assert float(rel_big) < 0.01
    ref = _field(12, 12, 12, std=0.5)
    rel_flip, _ = M.affine_residual(-ref, ref)               # 부호 뒤집힘 = 모순(a=0 → rms = std)
    assert float(rel_flip) == pytest.approx(1.0, abs=2e-2)


# ----------------------------------------------------------------------------------------- fuse_global

def _read(seed, h=12, w=16, gauge=(1.0, 0.0)):
    img = H.canvas_display(H.make_image(4 * h, 4 * w, seed))
    return H.stub_outputs(img, gauge)


def test_fuse_global_single_read_is_a_copy():
    r = _read(1)
    f = M.fuse_global([r])
    assert f.r_dis_sigma == 0.0 and float(f.sigma_map.abs().max()) == 0.0
    for k in r:
        torch.testing.assert_close(f.outputs[k], r[k], rtol=0, atol=0)
    f.outputs["log_depth"].add_(1.0)
    assert not torch.equal(f.outputs["log_depth"], r["log_depth"]), "입력과 메모리를 공유하면 안 된다"


def test_fuse_global_gauge_difference_only_gives_zero_r_dis_and_first_sigma_gauge():
    r1 = _read(2, gauge=(1.3, 0.2))
    r2 = _read(2, gauge=(0.4, -3.0))                         # 같은 장면, 다른 affine 게이지
    f = M.fuse_global([r1, r2])
    assert f.r_dis_sigma < 1e-4
    torch.testing.assert_close(f.outputs["log_depth"], r1["log_depth"], atol=1e-4, rtol=0)   # 게이지 = 첫 σ
    for k in ("defocus", "albedo", "shading", "sigma_hat"):
        torch.testing.assert_close(f.outputs[k], r1[k], atol=1e-6, rtol=0)
    assert tuple(f.sigma_map.shape) == (12, 16) and float(f.sigma_map.max()) < 1e-3


def test_fuse_global_perturbed_second_read_matches_oracle():
    r1 = _read(3)
    r2 = _read(3, gauge=(2.0, 1.0))
    pert = _field(13, 12, 16, std=0.08)
    r2 = dict(r2)
    r2["log_depth"] = r2["log_depth"] + pert
    f = M.fuse_global([r1, r2])
    d1, d2 = r1["log_depth"], r2["log_depth"]
    assert f.r_dis_sigma == pytest.approx(H.oracle_rel(d2.numpy(), d1.numpy()), rel=2e-3, abs=2e-4)
    a, b = H.oracle_affine(d2.numpy(), d1.numpy())
    want = 0.5 * (d1.double() + a * d2.double() + b).float()
    torch.testing.assert_close(f.outputs["log_depth"], want, atol=1e-4, rtol=0)
    assert f.r_dis_sigma > 0.05
    # 지도는 정렬 후 |잔차|/분모, 평균 제곱근이 스칼라와 같다
    assert float(torch.sqrt((f.sigma_map.double() ** 2).mean())) == pytest.approx(f.r_dis_sigma, rel=2e-3, abs=2e-4)


def test_fuse_global_normal_and_sem_are_renormalized_means():
    r1, r2 = _read(4), _read(5)
    f = M.fuse_global([r1, r2])
    for k in ("normal", "sem"):
        want = H.unit(0.5 * (r1[k] + r2[k]))
        torch.testing.assert_close(f.outputs[k], want, atol=1e-5, rtol=0)
        assert float((f.outputs[k].norm(dim=1) - 1).abs().max()) < 1e-5
    torch.testing.assert_close(f.outputs["defocus"], 0.5 * (r1["defocus"] + r2["defocus"]), atol=1e-6, rtol=0)


# ----------------------------------------------------------------------------------------- blend_reads

HW = (64, 96)                                               # 캔버스 px → 출력 격자 16x24
TILE = 64


def _glob_and_crops(gauges, extra=None):
    """전역 G(캔버스 전체)와, 타일마다 G를 잘라 게이지를 먹인 타일 출력."""
    img = H.canvas_display(H.make_image(*HW, seed=21))
    G = H.stub_outputs(img)
    boxes = M.plan_tiles(*HW, tile=TILE)
    tiles = []
    for b, gz in zip(boxes, gauges):
        crop = img[:, :, b.y0:b.y1, b.x0:b.x1]
        t = H.stub_outputs(crop, gz)
        if extra:
            t = extra(b, t)
        tiles.append((b, t))
    return G, boxes, tiles


def test_blend_identity_when_tiles_agree_exactly():
    G, boxes, tiles = _glob_and_crops([(1.0, 0.0)] * 2)
    assert len(boxes) == 2
    res = M.blend_reads(G, tiles, canvas_hw=HW, feather_px=16)
    for k in G:
        torch.testing.assert_close(res.outputs[k], G[k], atol=1e-5, rtol=0)
    assert res.r_dis_tiles < 1e-4 and all(r < 1e-4 for r in res.rel)
    for (a, b) in res.affine:
        assert float(a) == pytest.approx(1.0, abs=1e-4) and float(b) == pytest.approx(0.0, abs=1e-4)
    assert float(res.tile_map.max()) < 1e-3


def test_blend_corrects_inconsistent_affine_gauges_and_naive_mean_would_not():
    gauges = [(0.5, 3.0), (2.0, -1.0)]
    G, boxes, tiles = _glob_and_crops(gauges)
    res = M.blend_reads(G, tiles, canvas_hw=HW, feather_px=16)
    torch.testing.assert_close(res.outputs["log_depth"], G["log_depth"], atol=1e-4, rtol=0)
    for (a, b), (A, B) in zip(res.affine, gauges):           # 정렬 맵 = 게이지의 역
        assert float(a) == pytest.approx(1.0 / A, rel=1e-3) and float(b) == pytest.approx(-B / A, abs=2e-3)
    assert res.r_dis_tiles < 1e-3
    # 비어 있지 않은 검사: 정렬 없이 단순 가중 평균이면 크게 어긋난다
    w = M.tile_weights(boxes, HW, feather_px=16)
    num = torch.zeros_like(G["log_depth"])
    for t, (b, out) in enumerate(tiles):
        full = torch.zeros_like(num)
        full[..., b.y0 // 4:b.y1 // 4, b.x0 // 4:b.x1 // 4] = out["log_depth"]
        num += w[t] * full
    naive = num / w.sum(0).clamp_min(1.0)
    assert float((naive - G["log_depth"]).abs().max()) > 0.5


def test_blend_weights_partition_and_coverage():
    G, boxes, tiles = _glob_and_crops([(1.0, 0.0)] * 2)
    res = M.blend_reads(G, tiles, canvas_hw=HW, feather_px=16)
    assert tuple(res.weights.shape) == (2, 16, 24) and tuple(res.coverage.shape) == (16, 24)
    ref = H.oracle_tile_weights(boxes, HW, 16)
    np.testing.assert_allclose(res.weights.numpy(), ref, atol=2e-6)
    np.testing.assert_allclose(res.coverage.numpy(), np.clip(ref.sum(0), 0, 1), atol=2e-6)
    assert float(res.coverage.min()) > 0.0                    # 겹침이 있어 전 셀이 덮인다


def test_blend_formula_and_global_fills_zero_overlap_seam():
    # 가로로 틈 없이 맞닿는 두 타일(겹침 0): 비정렬 채널(defocus)에 ±0.5 오프셋을 줘 이음매를 본다
    hw = (64, 128)
    img = H.canvas_display(H.make_image(*hw, seed=22))
    G = H.stub_outputs(img)
    boxes = M.plan_tiles(*hw, tile=64)
    assert [(b.x0, b.x1) for b in boxes] == [(0, 64), (64, 128)]
    tiles = []
    for b in boxes:
        t = H.stub_outputs(img[:, :, b.y0:b.y1, b.x0:b.x1])
        t["defocus"] = t["defocus"] + (0.5 if b.x0 == 0 else -0.5)
        tiles.append((b, t))
    res = M.blend_reads(G, tiles, canvas_hw=hw, feather_px=16)
    # 계약서 수식 오라클: out = Σ w T / max(W,1) + max(0,1-W) G  (깊이 외 채널은 정렬 없음)
    w = torch.from_numpy(H.oracle_tile_weights(boxes, hw, 16)).float()
    full = []
    for (b, t) in tiles:
        f = torch.zeros_like(G["defocus"])
        f[..., b.y0 // 4:b.y1 // 4, b.x0 // 4:b.x1 // 4] = t["defocus"]
        full.append(f)
    W = w.sum(0)
    want = sum(w[i] * full[i] for i in range(2)) / W.clamp_min(1.0) + (1.0 - W).clamp_min(0.0) * G["defocus"]
    torch.testing.assert_close(res.outputs["defocus"], want, atol=1e-5, rtol=0)
    out, g = res.outputs["defocus"], G["defocus"]
    # 타일 안쪽은 타일 값 그대로, 이음매 양쪽 셀은 전역 값에 붙는다(편차 ≪ 오프셋 0.5)
    assert float(res.coverage[8, 0]) == pytest.approx(1.0, abs=1e-6)
    assert out[0, 0, 8, 0].item() == pytest.approx(g[0, 0, 8, 0].item() + 0.5, abs=1e-5)
    assert out[0, 0, 8, 31].item() == pytest.approx(g[0, 0, 8, 31].item() - 0.5, abs=1e-5)
    d15 = out[0, 0, 8, 15].item() - g[0, 0, 8, 15].item()
    d16 = out[0, 0, 8, 16].item() - g[0, 0, 8, 16].item()
    assert 0.0 < d15 < 0.05 and -0.05 < d16 < 0.0


def test_blend_no_tiles_returns_global_and_zero_stats():
    G, _, _ = _glob_and_crops([])
    res = M.blend_reads(G, [], canvas_hw=HW, feather_px=16)
    for k in G:
        torch.testing.assert_close(res.outputs[k], G[k], atol=1e-6, rtol=0)
    assert res.r_dis_tiles == 0.0 and tuple(res.affine) == () and tuple(res.rel) == ()
    assert float(res.coverage.abs().max()) == 0.0 and float(res.tile_map.abs().max()) == 0.0
    assert res.weights.shape[0] == 0


def test_blend_detects_contradictory_tile_and_map_is_local():
    flip = {1: lambda b, t: dict(t, log_depth=2.0 * t["log_depth"].flip(-1))}
    hw = (96, 128)                                            # 타일 4개(2x2), 타일 1 = (y0=0, x0=64)
    img = H.canvas_display(H.make_image(*hw, seed=23))
    G = H.stub_outputs(img)
    boxes = M.plan_tiles(*hw, tile=64)
    assert len(boxes) == 4 and (boxes[1].y0, boxes[1].x0) == (0, 64)
    tiles = []
    for j, b in enumerate(boxes):
        t = H.stub_outputs(img[:, :, b.y0:b.y1, b.x0:b.x1])
        if j in flip:
            t = flip[j](b, t)
        tiles.append((b, t))
    res = M.blend_reads(G, tiles, canvas_hw=hw, feather_px=16)
    assert res.rel[1] > 0.3 and max(res.rel[0], res.rel[2], res.rel[3]) < 1e-3
    assert res.r_dis_tiles == pytest.approx(float(np.mean(res.rel)), rel=1e-5)
    m = res.tile_map[1:7, 20:32]                              # 타일 1만 덮는 안쪽 셀
    ok = res.tile_map[0:6, 0:12]                              # 정상 타일 0만 덮는 셀
    assert float(m.mean()) > 0.10 and float(ok.max()) < 1e-3
    # 정상 타일만 덮는 곳은 모순 타일의 영향을 받지 않는다
    d = (res.outputs["log_depth"] - G["log_depth"]).abs()
    assert float(d[..., 0:6, 0:12].max()) < 1e-4


def test_blend_input_validation():
    G, boxes, tiles = _glob_and_crops([(1.0, 0.0)] * 2)
    bad = dict(G)
    bad.pop("sem")
    with pytest.raises(ValueError):
        M.blend_reads(bad, tiles, canvas_hw=HW)               # 전역과 타일의 헤드 키가 다르다
    wrong = [(boxes[0], {k: v[..., :-1] for k, v in tiles[0][1].items()}), tiles[1]]
    with pytest.raises(ValueError):
        M.blend_reads(G, wrong, canvas_hw=HW)                 # 타일 출력 격자가 박스와 다르다
