"""RD-09 복원·재조명·종단: 알려진 알베도/법선/조명 장면에서 분해 정확도, 교대(R7<->R8) 수렴, 재조명 수식·정확도,
r_intr 정의, RD-06 inverse 이음매, RD-07 MRH 출력 어댑터."""
import math

import numpy as np
import pytest

from buc.read import intrinsics as X
import rd09_helpers as H

H_, W_ = 24, 32


def lt(d=H.LIGHT_DIR, k=H.KEY, a=H.AMB):
    return X.GeoLight(direction=tuple(d), intensity=k, ambient=a)


def noisy_head(sc, amp=0.25, seed=1):
    return sc["shading"] * np.exp(amp * H.smooth_field(H_, W_, seed))


# ------------------------------------------------------------------------------------------------ 복원
def test_recovers_chromatic_albedo_and_shading_with_known_light():
    sc = H.scene(H_, W_)
    r = X.decompose(sc["I"].astype(np.float32), normal=sc["normal"], light=lt())
    ds = np.abs(np.log(r.shading) - np.log(sc["shading"]))
    assert ds.max() <= 0.03 and ds.mean() <= 0.006
    rel = np.abs(r.albedo / sc["albedo"] - 1.0)
    assert rel.max() <= 0.03
    # 알베도는 조각별 상수: 각 조각 안 분산이 작고 두 조각의 색이 구분된다
    left, right = r.albedo[:, 2:14].reshape(-1, 3), r.albedo[:, 18:30].reshape(-1, 3)
    assert np.allclose(left.mean(0), (0.70, 0.40, 0.20), rtol=0.03) and np.allclose(right.mean(0), (0.20, 0.50, 0.70), rtol=0.03)
    assert left.std(0).max() < 0.02 and right.std(0).max() < 0.02
    assert r.r_intr < 0.01 and r.reliable and r.warnings == ()


def test_luminance_only_albedo_step_is_kept_when_geometry_disagrees_with_a_step():
    sc = H.scene(H_, W_, albedo=H.gray_patch_albedo(H_, W_, 0.30, 0.60))
    r = X.decompose(sc["I"].astype(np.float32), geo_shading=sc["shading"])
    a = r.albedo[:, :, 0].astype(np.float64)
    ratio = np.median(a[:, 22:30] / a[:, 2:10])         # 좌우 같은 위치 비(진짜 2.0)
    assert ratio == pytest.approx(2.0, rel=0.08)
    ds = np.abs(np.log(r.shading) - np.log(sc["shading"]))
    assert ds[:, :11].max() <= 0.02 and ds[:, 21:].max() <= 0.02          # 단차에서 먼 곳은 정확
    assert ds.max() <= 0.3                                                # 단차 바로 옆의 음영은 일부 단차를 흡수할 수 있다(한계)


def test_light_fit_path_improves_a_noisy_shading_head_and_finds_the_light():
    sc = H.scene(H_, W_)
    head = noisy_head(sc, 0.25, 1)
    I = sc["I"].astype(np.float32)
    r = X.decompose(I, normal=sc["normal"], shading_head=head)
    assert r.geo_source == "light_fit" and r.n_solves == 2
    assert H.angle_deg(r.light.direction, H.LIGHT_DIR) <= 4.0
    assert r.light.intensity == pytest.approx(H.KEY, rel=0.15) and r.light.ambient == pytest.approx(H.AMB, abs=0.06)
    e_head = np.abs(np.log(head) - np.log(sc["shading"])).mean()
    e_out = np.abs(np.log(r.shading) - np.log(sc["shading"])).mean()
    assert e_out <= 0.5 * e_head and e_out <= 0.03
    # 기하 사전이 영상(I)과 헤드가 모두 같은 말을 하는 쪽으로 끌어당긴다
    assert np.abs(r.albedo / sc["albedo"] - 1.0).mean() <= 0.04
    # 교대를 늘려도 나빠지지 않는다
    errs = []
    for k in (1, 2, 3):
        rk = X.decompose(I, normal=sc["normal"], shading_head=head, opts=X.IntrinsicsOpts(n_alt=k))
        errs.append(np.abs(np.log(rk.shading) - np.log(sc["shading"])).mean())
    assert errs[1] <= errs[0] + 1e-3 and errs[2] <= errs[1] + 1e-3


def test_albedo_head_can_seed_the_light_fit_when_no_shading_head():
    sc = H.scene(H_, W_)
    ahead = np.clip(sc["albedo"] * np.exp(0.1 * np.random.default_rng(2).standard_normal(sc["albedo"].shape)), 0.02, 1.5)
    r = X.decompose(sc["I"].astype(np.float32), normal=sc["normal"], albedo_head=ahead.astype(np.float32))
    assert r.geo_source == "light_fit" and r.light is not None
    assert H.angle_deg(r.light.direction, H.LIGHT_DIR) <= 8.0
    assert np.abs(np.log(r.shading) - np.log(sc["shading"])).mean() <= 0.06


def test_cast_shadow_is_not_absorbed_into_albedo_and_raises_r_intr():
    sc = H.scene(H_, W_)
    shadow = np.ones((H_, W_))
    shadow[8:16, 3:12] = 0.4                               # 기하가 모르는 드리운 그림자(음영이 실제로 어둡다)
    S_true = sc["shading"] * shadow
    I = (sc["albedo"] * S_true[..., None]).astype(np.float32)
    head = S_true * np.exp(0.05 * H.smooth_field(H_, W_, 3))      # 헤드는 그림자를 안다
    o = X.IntrinsicsOpts(gamma=0.3, eta=1.0, beta=0.05)
    r = X.decompose(I, normal=sc["normal"], light=lt(), shading_head=head, opts=o)
    clean = X.decompose(sc["I"].astype(np.float32), normal=sc["normal"], light=lt(), shading_head=sc["shading"], opts=o)
    inside = (slice(9, 15), slice(4, 11))
    assert np.log(r.shading[inside]).mean() - np.log(sc["shading"][inside]).mean() < -0.6     # 그림자를 음영으로 받아들임
    assert np.abs(r.albedo[inside] / sc["albedo"][inside] - 1.0).mean() < 0.3
    assert r.r_intr > 3.0 * clean.r_intr + 0.02


def test_r_intr_definition():
    sc = H.scene(H_, W_)
    ex = np.zeros((H_, W_), bool)
    ex[4:8, 4:12] = True
    head = noisy_head(sc, 0.3, 4)
    r = X.decompose(sc["I"].astype(np.float32), normal=sc["normal"], light=lt(), shading_head=head, exclude=ex)
    v = ~ex
    d = np.log(r.shading.astype(np.float64)) - np.log(np.maximum(r.geo_shading.astype(np.float64), 1e-4))
    assert r.r_intr == pytest.approx(math.sqrt(float(np.mean(d[v] ** 2))), rel=1e-4, abs=2e-6)
    assert r.meta()["r_intr"] == pytest.approx(r.r_intr)


# ------------------------------------------------------------------------------------------------ 재조명
def test_relight_same_light_returns_owned_L_d():
    sc = H.scene(H_, W_)
    head = noisy_head(sc, 0.2, 5)
    r = X.decompose(sc["I"].astype(np.float32), normal=sc["normal"], shading_head=head)
    out = X.relight(r, light=r.light, normal=sc["normal"])
    np.testing.assert_allclose(out, r.L_d, rtol=2e-6, atol=1e-9)
    out2 = X.relight(r, geo_shading=r.geo_shading.astype(np.float64))
    np.testing.assert_allclose(out2, r.L_d, rtol=2e-6, atol=1e-9)
    assert out.dtype == np.float32 and out.shape == r.L_d.shape


def test_relight_formula_ratio_with_floor_and_visibility():
    sc = H.scene(H_, W_)
    r = X.decompose(sc["I"].astype(np.float32), normal=sc["normal"], light=lt())
    new = X.GeoLight(direction=tuple(H.LIGHT2_DIR), intensity=0.9, ambient=0.0)      # 주변광 0 -> 어두운 곳이 바닥에 붙음
    vis = np.clip(0.5 + H.smooth_field(H_, W_, 6, 0.8), 0, 1)
    out = X.relight(r, light=new, normal=sc["normal"], visibility=vis)
    g_new = H.geo_shading(sc["normal"], H.LIGHT2_DIR, 0.0, 0.9)
    g_new = 0.0 + 0.9 * vis * np.maximum(sc["normal"] @ H.LIGHT2_DIR, 0.0)
    g_old = r.geo_shading.astype(np.float64)
    ratio = np.maximum(g_new, 1e-3) / np.maximum(g_old, 1e-3)
    expect = r.L_d.astype(np.float64) * ratio[..., None]
    np.testing.assert_allclose(out, expect, rtol=3e-6, atol=1e-9)
    assert (g_new <= 1e-3).any()                          # 바닥 분기를 실제로 지난다
    big = X.relight(r, light=new, normal=sc["normal"], visibility=vis, floor=0.2)
    ratio2 = np.maximum(g_new, 0.2) / np.maximum(g_old, 0.2)
    np.testing.assert_allclose(big, r.L_d.astype(np.float64) * ratio2[..., None], rtol=3e-6, atol=1e-9)
    assert np.isfinite(out).all() and (out >= 0).all()


def test_relight_to_a_different_light_matches_ground_truth():
    sc = H.scene(H_, W_)
    head = noisy_head(sc, 0.15, 7)
    r = X.decompose(sc["I"].astype(np.float32), normal=sc["normal"], shading_head=head)
    truth = sc["albedo"] * H.geo_shading(sc["normal"], H.LIGHT2_DIR)[..., None]
    out = X.relight(r, light=lt(H.LIGHT2_DIR), normal=sc["normal"])
    y = lambda a: H.luminance(a)
    err = np.mean(np.abs(y(out) - y(truth)) / y(truth))
    base = np.mean(np.abs(y(sc["I"]) - y(truth)) / y(truth))          # 재조명하지 않은 영상과의 거리
    assert err <= 0.05 and err <= 0.25 * base
    # 색도는 알베도가 지켜 준다(밝기와 무관하게)
    cs = lambda a: a / a.sum(-1, keepdims=True)
    assert np.abs(cs(out) - cs(truth)).max() <= 0.02


def test_relight_errors():
    sc = H.scene(H_, W_)
    I = sc["I"].astype(np.float32)
    r_nogeo = X.decompose(I, shading_head=sc["shading"])
    with pytest.raises(X.IntrinsicsError):
        X.relight(r_nogeo, light=lt(), normal=sc["normal"])
    r = X.decompose(I, normal=sc["normal"], light=lt())
    with pytest.raises(ValueError):
        X.relight(r)                                                          # 둘 다 없음
    with pytest.raises(ValueError):
        X.relight(r, light=lt(), geo_shading=sc["shading"], normal=sc["normal"])   # 둘 다 있음
    with pytest.raises(ValueError):
        X.relight(r, light=lt())                                              # 법선 없음
    with pytest.raises(ValueError):
        X.relight(r, geo_shading=sc["shading"][:-1])
    with pytest.raises(TypeError):
        X.relight(r, light=(0, 0, -1), normal=sc["normal"])
    with pytest.raises(TypeError):
        X.relight("nope", light=lt(), normal=sc["normal"])
    with pytest.raises(ValueError):
        X.relight(r, geo_shading=-sc["shading"])
    with pytest.raises(ValueError):
        X.relight(r, light=lt(), normal=sc["normal"], floor=0.0)


# ------------------------------------------------------------------------------------------------ RD-06 이음매
def test_rd06_inverse_output_feeds_decompose_with_status_exclusion():
    from buc.read import tone_color as T
    sc = H.scene(H_, W_)
    L = (sc["albedo"] * (1.2 * sc["shading"])[..., None]).astype(np.float32)           # 일부 화소가 클립될 만큼 밝게
    p = T.ToneParams(ev=0.4, wb_kelvin=5200.0, wb_tint=0.05, curve=T.FilmCurve(toe=0.3, shoulder=0.0))     # 하드 클립
    D = T.forward(L, p)
    inv = T.inverse(D, p)
    assert (inv.status != 0).any() and (inv.status == 0).mean() > 0.5
    r = X.decompose(inv.linear, normal=sc["normal"], geo_shading=1.2 * sc["shading"], status=inv.status,
                    opts=X.IntrinsicsOpts(gamma=2.0))
    assert np.array_equal(r.valid, inv.status == 0)
    assert np.array_equal(r.L_d, inv.linear)                                  # 소유한 L_d 는 inverse 결과 그대로
    np.testing.assert_allclose(r.albedo.astype(np.float64) * r.shading[..., None], r.L_d, rtol=5e-6, atol=1e-8)
    own = r.valid
    assert np.abs(np.log(r.shading[own]) - np.log(1.2 * sc["shading"][own])).max() <= 0.05
    # 클립된 화소의 음영은 기하 폴백(수치 오귀속 방지): 그 화소 알베도는 L_d/폴백
    np.testing.assert_allclose(r.shading[~own], 1.2 * sc["shading"][~own], rtol=2e-6)


# ------------------------------------------------------------------------------------------------ RD-07 MRH 출력 어댑터
def _mrh_outputs_from_scene(sc, torch):
    def n4(a):                                          # 4x4 상자 평균(H/4 격자의 셀 = 4x4 블록)
        return a.reshape(H_ // 4, 4, W_ // 4, 4, *a.shape[2:]).mean(axis=(1, 3))
    h4, w4 = math.ceil(H_ / 4), math.ceil(W_ / 4)
    alb = torch.from_numpy(np.ascontiguousarray(np.transpose(n4(sc["albedo"]), (2, 0, 1))[None])).float()
    sh = torch.from_numpy(np.ascontiguousarray(n4(sc["shading"])[None, None])).float()
    nr = torch.from_numpy(np.ascontiguousarray(np.transpose(n4(sc["normal"]), (2, 0, 1))[None])).float()
    assert alb.shape == (1, 3, h4, w4)
    return {"albedo": alb, "shading": sh, "normal": nr, "log_depth": torch.zeros(1, 1, h4, w4)}


def test_heads_from_readout_matches_torch_bilinear_and_renormalizes():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    Hh, Ww = 17, 30
    h4, w4 = math.ceil(Hh / 4), math.ceil(Ww / 4)
    g = torch.Generator().manual_seed(0)
    out = {"albedo": torch.rand(1, 3, h4, w4, generator=g), "shading": torch.rand(1, 1, h4, w4, generator=g) + 0.1,
           "normal": torch.randn(1, 3, h4, w4, generator=g), "log_depth": torch.randn(1, 1, h4, w4, generator=g)}
    out["normal"][:, :, 1, 2] = 0.0                        # 영벡터 화소
    hd = X.heads_from_readout(out, (Hh, Ww))
    assert set(hd) == {"albedo", "shading", "normal"}
    ref_a = F.interpolate(out["albedo"].double(), size=(Hh, Ww), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    ref_s = F.interpolate(out["shading"].double(), size=(Hh, Ww), mode="bilinear", align_corners=False)[0, 0].numpy()
    ref_n = F.interpolate(out["normal"].double(), size=(Hh, Ww), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    assert hd["albedo"].shape == (Hh, Ww, 3) and hd["shading"].shape == (Hh, Ww) and hd["normal"].shape == (Hh, Ww, 3)
    np.testing.assert_allclose(hd["albedo"], ref_a, atol=1e-6)
    np.testing.assert_allclose(hd["shading"], ref_s, atol=1e-6)
    nn = np.linalg.norm(ref_n, axis=-1, keepdims=True)
    expect_n = np.where(nn > 1e-6, ref_n / np.maximum(nn, 1e-6), 0.0)
    np.testing.assert_allclose(hd["normal"], expect_n, atol=1e-6)
    nrm = np.linalg.norm(hd["normal"], axis=-1)
    assert np.all((np.abs(nrm - 1) < 1e-6) | (nrm == 0))
    # numpy 입력도 같은 결과, 없는 헤드는 None, 입력은 그대로
    np_out = {k: v.numpy() for k, v in out.items() if k in ("albedo", "shading")}
    h2 = X.heads_from_readout(np_out, (Hh, Ww))
    np.testing.assert_allclose(h2["albedo"], hd["albedo"], atol=1e-6)
    assert h2["normal"] is None
    assert X.heads_from_readout({}, (Hh, Ww)) == {"albedo": None, "shading": None, "normal": None}
    for bad in ({"albedo": torch.zeros(3, h4, w4)}, {"shading": torch.zeros(2, 1, h4, w4)}):
        with pytest.raises(ValueError):
            X.heads_from_readout(bad, (Hh, Ww))


def test_end_to_end_from_mrh_like_outputs():
    torch = pytest.importorskip("torch")
    sc = H.scene(H_, W_)
    heads = X.heads_from_readout(_mrh_outputs_from_scene(sc, torch), (H_, W_))
    r = X.decompose(sc["I"].astype(np.float32), normal=heads["normal"], shading_head=heads["shading"],
                    albedo_head=heads["albedo"])
    assert r.geo_source == "light_fit"
    assert np.abs(np.log(r.shading) - np.log(sc["shading"])).mean() <= 0.05
    assert np.abs(r.albedo / sc["albedo"] - 1.0).mean() <= 0.06
    assert H.angle_deg(r.light.direction, H.LIGHT_DIR) <= 8.0


@pytest.mark.gpu
def test_gpu_tensors_accepted_by_adapter():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")
    sc = H.scene(H_, W_)
    cpu = _mrh_outputs_from_scene(sc, torch)
    gpu = {k: v.cuda() for k, v in cpu.items()}
    a, b = X.heads_from_readout(cpu, (H_, W_)), X.heads_from_readout(gpu, (H_, W_))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-6)
