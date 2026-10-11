"""RD-10 fit_lights 종단: NNLS+CMA-ES 복원, 치트 라이트(CL1/CL4/CL6), 가시성·플래그·사전분포, 폴백, RD-09 이음매."""
import json
import math

import numpy as np
import pytest

import rd10_helpers as H
from buc.read import fit_lights as F
from buc.read import intrinsics as X

BN, BV = H.ball()
BW = BV.astype(np.float64)
KEY_K, RIM_K = 0.6, 0.5


def ball_scene(rim=True, noise=0.0, seed=0):
    lights = [(H.KEY_DIR, KEY_K)] + ([(H.RIM_DIR, RIM_K)] if rim else [])
    s = H.shading_from(BN, lights, H.AMB)
    if noise:
        s = np.maximum(s + np.random.default_rng(seed).normal(0.0, noise, s.shape), 0.0)
    return s


# ------------------------------------------------------------------ 한 광원 복원 (동기 광원만)
def test_single_sun_recovered_no_cheat():
    n = H.dome(24, 32)
    s = H.shading_from(n, [(H.DOME_DIR, 0.75)], 0.25)
    r = F.fit_lights(s, n)
    assert r.status == "ok" and r.reliable
    assert len(r.lights) == 1 and r.lights[0].name == F.KEY_NAME and r.lights[0].role == "motivated"
    assert H.angle_deg(r.lights[0].direction, H.DOME_DIR) < 0.1
    assert r.lights[0].intensity == pytest.approx(0.75, abs=2e-3) and r.ambient == pytest.approx(0.25, abs=2e-3)
    assert r.r2 > 0.9999 and r.rms < 1e-3 and r.n_used == 24 * 32
    assert r.cheat_share == 0.0 and r.r_light == pytest.approx(0.0) and r.red is False       # CL1: 설명되면 치트는 없다
    assert r.geo_shading is not None and r.geo_shading.dtype == np.float32 and r.geo_shading.shape == s.shape
    assert np.allclose(r.geo_shading, s, atol=2e-3)
    assert r.lights[0].kind == "sun" and r.lights[0].angular_radius == 0.0
    assert r.n_evals > 0 and r.warnings == ()


def test_noise_does_not_invent_cheat_lights():
    n = H.dome(24, 32)
    s = np.maximum(H.shading_from(n, [(H.DOME_DIR, 0.75)], 0.25) + np.random.default_rng(3).normal(0, 0.02, (24, 32)), 0)
    r = F.fit_lights(s, n)
    assert r.status == "ok" and len(r.lights) == 1                       # 잔차가 잡음뿐이면 치트 라이트를 만들지 않는다
    assert H.angle_deg(r.lights[0].direction, H.DOME_DIR) < 2.0
    assert 0.98 < r.r2 < 1.0


def test_wrong_init_light_still_converges_and_accepts_geolight():
    n = H.dome(24, 32)
    s = H.shading_from(n, [(H.DOME_DIR, 0.75)], 0.25)
    far = X.GeoLight(direction=tuple(H.unit([0.5, 0.3, -0.8])), intensity=0.5, ambient=0.1)      # RD-09 GeoLight 도 init 가능
    assert H.angle_deg(far.direction, H.DOME_DIR) > 60
    r = F.fit_lights(s, n, init_light=far)
    assert H.angle_deg(r.lights[0].direction, H.DOME_DIR) < 0.5 and r.r2 > 0.9999
    near = F.Light(name="x", role="motivated", kind="sun", direction=tuple(H.DOME_DIR), intensity=0.1)
    r2 = F.fit_lights(s, n, init_light=near)
    assert H.angle_deg(r2.lights[0].direction, H.DOME_DIR) < 0.1


# ------------------------------------------------------------------ 치트 라이트: 출처 없는 림 (CL1 순서 · CL3 · CL4 · CL6)
def test_cheat_rim_light_found_after_motivated_key():
    s = ball_scene()
    r = F.fit_lights(s, BN, valid=BV)
    assert r.status == "ok" and len(r.lights) == 2
    key, cheat = r.lights                                               # CL1: 동기 광원이 앞, 치트가 뒤
    assert key.role == "motivated" and key.name == "key" and cheat.role == "cheat" and cheat.name == "cheat1"
    assert cheat.ghost is True and key.ghost is False                    # CL3
    assert H.angle_deg(key.direction, H.KEY_DIR) < 3.0
    assert H.angle_deg(cheat.direction, H.RIM_DIR) < 5.0
    assert key.intensity == pytest.approx(KEY_K, rel=0.03) and cheat.intensity == pytest.approx(RIM_K, rel=0.05)
    assert r.ambient == pytest.approx(H.AMB, abs=0.01)
    assert cheat.kind in ("area", "spot") and cheat.angular_radius == pytest.approx(F.CHEAT_ANGULAR_RADIUS)
    assert r.r2 > 0.999
    e, sh, cs = H.energy_oracle(BN, [(H.KEY_DIR, KEY_K), (H.RIM_DIR, RIM_K)], H.AMB, ["motivated", "cheat"], BW)
    assert r.cheat_share == pytest.approx(cs, abs=0.005) and r.energies[0] == pytest.approx(e[0], rel=0.05)
    assert r.r_light == pytest.approx(max(r.cheat_share / 0.35, 1 / 4)) and r.red is False and not r.recommend_safe
    # 같은 영상에서 치트를 막으면(max_cheat=0) 잔차가 남아 경고하고 R^2 가 낮다
    r0 = F.fit_lights(s, BN, valid=BV, opts=F.FitLightsOpts(max_cheat=0))
    assert len(r0.lights) == 1 and "cheat_limit" in r0.warnings and r0.r2 < r.r2 - 0.01
    assert r0.r_light == 0.0


def test_cheat_meta_cl6_and_json():
    r = F.fit_lights(ball_scene(), BN, valid=BV)
    m = r.meta()
    json.dumps(m, allow_nan=False)
    for k in ("status", "motivated", "cheat", "probe", "ambient", "r2", "rms", "cheat_share", "r_light", "red",
              "recommend_safe", "n_used", "n_evals", "warnings", "reliable"):
        assert k in m, k
    assert len(m["motivated"]) == 1 and len(m["cheat"]) == 1 and m["probe"] == []
    assert m["cheat"][0]["cheat"] == 1 and m["motivated"][0]["cheat"] == 0       # CL6: cheat=1, 에너지 몫 기록
    assert m["cheat"][0]["energy_share"] == pytest.approx(r.cheat_share, rel=1e-9)
    assert m["motivated"][0]["energy_share"] == pytest.approx(r.energies[0] / (r.ambient + sum(r.energies)), rel=1e-9)
    assert m["cheat"][0]["ghost"] is True and "position" not in m["cheat"][0]
    gl = r.geo_light()                                                      # 치트는 RD-09 GeoLight 에 섞이지 않는다
    assert gl.direction == r.lights[0].direction and gl.intensity == r.lights[0].intensity


def test_geo_shading_reproduces_render_and_invalid_pixels_ignored():
    s = ball_scene()
    r = F.fit_lights(s, BN, valid=BV)
    ref = F.render_shading(r.lights, BN, ambient=r.ambient)
    assert np.allclose(r.geo_shading, ref, atol=1e-5)                      # geo_shading = 같은 모형의 렌더
    assert np.allclose(r.geo_shading[BV], s[BV], atol=5e-3)
    g = s.copy(); g[~BV] = np.random.default_rng(1).uniform(2.0, 9.0, int((~BV).sum()))
    r2 = F.fit_lights(g, BN, valid=BV)                                    # 무효 화소의 값은 결과에 영향이 없다
    assert [l.direction for l in r2.lights] == [l.direction for l in r.lights]
    assert r2.ambient == r.ambient and r2.r2 == r.r2
    assert r.n_used == int(BV.sum())


def test_determinism_and_scale_equivariance():
    s = ball_scene()
    a = F.fit_lights(s, BN, valid=BV)
    b = F.fit_lights(s, BN, valid=BV)
    assert [l.direction for l in a.lights] == [l.direction for l in b.lights]
    assert [l.intensity for l in a.lights] == [l.intensity for l in b.lights] and a.r2 == b.r2 and a.n_evals == b.n_evals
    assert np.array_equal(a.geo_shading, b.geo_shading)
    c = F.fit_lights(3.0 * s, BN, valid=BV)                                # 영상 c배 -> 세기 c배, 방향·R^2 불변
    assert len(c.lights) == 2
    for la, lc in zip(a.lights, c.lights):
        assert H.angle_deg(la.direction, lc.direction) < 0.05
        assert lc.intensity == pytest.approx(3.0 * la.intensity, rel=2e-3)
    assert c.ambient == pytest.approx(3.0 * a.ambient, rel=2e-3) and c.r2 == pytest.approx(a.r2, abs=1e-6)
    assert c.cheat_share == pytest.approx(a.cheat_share, abs=1e-3)
    d = F.fit_lights(ball_scene(rim=False), BN, valid=BV, opts=F.FitLightsOpts(seed=7, max_cheat=0))
    assert d.status == "ok" and H.angle_deg(d.lights[0].direction, H.KEY_DIR) < 0.5      # 시드가 달라도 같은 장면을 푼다


def test_mirror_equivariance_single_light():
    n = H.dome(24, 32)
    s = H.shading_from(n, [(H.DOME_DIR, 0.75)], 0.25)
    nf = n[:, ::-1].copy(); nf[..., 0] *= -1.0
    r = F.fit_lights(s, n)
    rf = F.fit_lights(s[:, ::-1].copy(), nf)
    dx = np.array(r.lights[0].direction) * np.array([-1.0, 1.0, 1.0])
    assert H.angle_deg(rf.lights[0].direction, dx) < 0.5


# ------------------------------------------------------------------ 가시성(그림자) · 플래그
def test_visibility_callback_removes_shadow_bias():
    h, w = 24, 32
    n = H.dome(h, w)
    vis = H.make_shadow_vis(h, w)
    d = H.DOME_DIR
    s = H.shading_from(n, [(d, 0.75)], 0.25, vis=[vis(d)])
    assert (vis(d) < 0.5).mean() > 0.05
    calls = []

    def vis_logged(dd):
        dd = np.asarray(dd, np.float64)
        calls.append(dd.copy())
        return vis(dd)

    r1 = F.fit_lights(s, n, visibility=vis_logged)
    r0 = F.fit_lights(s, n)
    assert calls and all(c.shape == (3,) and abs(np.linalg.norm(c) - 1.0) < 1e-9 for c in calls)    # 단위 방향으로 호출
    assert H.angle_deg(r1.lights[0].direction, d) < 0.2 and r1.r2 > 0.9999 and len(r1.lights) == 1
    assert r1.lights[0].intensity == pytest.approx(0.75, abs=3e-3)
    assert H.angle_deg(r0.lights[0].direction, d) > 2.0 and r0.r2 < 0.99                  # 그림자를 모르면 어긋난다
    assert np.allclose(r1.geo_shading, s, atol=3e-3)                                      # 가시성이 geo_shading 에도 적용
    with pytest.raises(ValueError):
        F.fit_lights(s, n, visibility=lambda dd: np.ones((3, 3)), opts=F.FitLightsOpts(generations=2, n_coarse=32))


def test_flag_explains_darkened_region():
    n = H.dome(24, 32)
    d = H.DOME_DIR
    cov = np.zeros((24, 32)); cov[:, :10] = 0.7
    s = H.shading_from(n, [(d, 0.75)], 0.25, cover=[cov])
    fl = F.Flag(name="gobo", light=F.KEY_NAME, coverage=cov)                     # 차광판 = 키 광원만 가린다 (CL2: 물리 요소)
    r = F.fit_lights(s, n, flags=[fl])
    assert H.angle_deg(r.lights[0].direction, d) < 0.2 and r.r2 > 0.9999 and len(r.lights) == 1
    assert r.flags == (fl,) and r.warnings == ()
    assert np.allclose(r.geo_shading, s, atol=3e-3)
    rn = F.fit_lights(s, n, opts=F.FitLightsOpts(max_cheat=0))
    assert rn.r2 < 0.9999 and H.angle_deg(rn.lights[0].direction, d) > 1.0                   # 플래그를 모르면 설명이 어긋난다
    ghost_flag = F.Flag(name="g2", light="nobody", coverage=cov)
    r2 = F.fit_lights(s, n, flags=[fl, ghost_flag])
    assert "flag_unused" in r2.warnings and not r2.reliable


# ------------------------------------------------------------------ 방향 사전분포 (그림자 에지 · 반사 프로브)
def test_shadow_prior_resolves_underdetermined_azimuth():
    n = H.two_normal_scene()                                   # 값이 둘뿐이라 (ambient, k, 방향 2) 중 방향이 결정되지 않는다
    d = H.KEY_DIR
    s = H.shading_from(n, [(d, 0.7)], 0.2)
    rng = np.random.default_rng(0)
    edges = []
    for _ in range(6):
        q = rng.normal(size=3)
        edges.append(F.ShadowEdge(shadow_point=tuple(q), occluder_point=tuple(q + d * rng.uniform(0.5, 2.0))))
    prior = F.shadow_direction_prior(edges)
    r = F.fit_lights(s, n, direction_priors=[prior])
    assert H.angle_deg(r.lights[0].direction, d) < 1.0         # 그림자 에지 광선이 방위를 정한다
    assert r.r2 > 0.999 and len(r.lights) == 1
    assert r.priors == (prior,)
    near = F.DirectionPrior(direction=tuple(H.unit(d + np.array([0.05, 0.0, 0.0]))), sigma_rad=0.05, source="probe")
    r2 = F.fit_lights(s, n, direction_priors=[near])          # 자료가 못 정하는 방향은 사전을 따른다
    assert H.angle_deg(r2.lights[0].direction, near.direction) < 1.0 and r2.r2 > 0.999
    assert r2.meta()["probe"] and r2.meta()["probe"][0]["sigma_rad"] == pytest.approx(0.05)


def test_prior_is_soft_when_data_decides():
    s = H.shading_from(BN, [(H.KEY_DIR, 0.7)], H.AMB)
    wrong = F.DirectionPrior(direction=tuple(H.unit([0.3, -0.5, -0.7])), sigma_rad=0.5, source="probe")
    assert H.angle_deg(wrong.direction, H.KEY_DIR) > 40
    r = F.fit_lights(s, BN, valid=BV, direction_priors=[wrong])
    assert H.angle_deg(r.lights[0].direction, H.KEY_DIR) < 3.0           # 약한 틀린 사전은 뚜렷한 자료를 못 이긴다
    right = F.DirectionPrior(direction=tuple(H.KEY_DIR), sigma_rad=0.05)
    r2 = F.fit_lights(s, BN, valid=BV, direction_priors=[right])
    assert H.angle_deg(r2.lights[0].direction, H.KEY_DIR) < 0.2 and r2.r2 > 0.9999


# ------------------------------------------------------------------ 반그림자 -> 광원 종류
def test_penumbra_sets_key_kind_and_radius():
    s = H.shading_from(BN, [(H.KEY_DIR, 0.7)], H.AMB)
    sun = F.fit_lights(s, BN, valid=BV, penumbra=(0.05, 5.0), opts=F.FitLightsOpts(max_cheat=0))
    assert sun.lights[0].kind == "sun" and sun.lights[0].angular_radius == pytest.approx(0.005)
    area = F.fit_lights(s, BN, valid=BV, penumbra=(0.5, 2.0), opts=F.FitLightsOpts(max_cheat=0))
    assert area.lights[0].kind == "area" and area.lights[0].angular_radius == pytest.approx(0.125)
    with pytest.raises(ValueError):
        F.fit_lights(s, BN, valid=BV, penumbra=(0.0, 2.0))


# ------------------------------------------------------------------ 폴백 (기본 = 복사휘도 굽기)
def test_fallback_when_nothing_explains_the_shading():
    n = H.dome(24, 32)
    s = 0.5 + np.random.default_rng(0).normal(0.0, 0.05, (24, 32))
    r = F.fit_lights(s, n, opts=F.FitLightsOpts(max_cheat=1))
    assert r.status == "fallback_bake" and "fallback_bake" in r.warnings and "low_r2" in r.warnings
    assert r.lights == () and r.geo_shading is None and not r.reliable          # 하류는 L_d 를 그대로 소유한다
    assert r.geo_light() is None
    assert r.cheat_share == 0.0 and r.r_light == 0.0 and r.red is False
    json.dumps(r.meta(), allow_nan=False)                                       # nan 은 null
    assert r.meta()["status"] == "fallback_bake" and r.meta()["motivated"] == [] and r.meta()["cheat"] == []


def test_fallback_few_pixels_and_degenerate():
    n = H.dome(24, 32)
    s = H.shading_from(n, [(H.DOME_DIR, 0.75)], 0.25)
    v = np.zeros((24, 32), bool); v[:3, :5] = True               # 15 < MIN_PIXELS
    r = F.fit_lights(s, n, valid=v)
    assert r.status == "fallback_bake" and "few_pixels" in r.warnings and r.lights == () and r.n_used == 15
    v[:3, :6] = True                                             # 18 >= 16 : 폴백이 아니라 시도한다 (결과 품질은 불문)
    assert F.fit_lights(s, n, valid=v, opts=F.FitLightsOpts(generations=3, n_coarse=64)).n_used == 18
    shadeless = np.full((24, 32), 0.4)                           # 음영에 변화가 없으면 광원을 말할 수 없다
    r = F.fit_lights(shadeless, n)
    assert r.status == "fallback_bake" and "degenerate_shading" in r.warnings and r.lights == ()
    wn = n.copy(); wn[...] = 0.1                                 # 법선이 없다(길이 < NORMAL_MIN) -> 사용 화소 0
    r = F.fit_lights(s, wn)
    assert r.status == "fallback_bake" and "few_pixels" in r.warnings and r.n_used == 0


# ------------------------------------------------------------------ 음의 광원 없음 (CL2)
def test_fit_never_returns_negative_light():
    n = H.dome(24, 32)
    s = H.shading_from(n, [(H.DOME_DIR, 0.75)], 0.25)
    s[8:16, 12:24] *= 0.2                                        # 모형이 설명 못 하는 어두운 웅덩이 (음의 광원이 필요할 자료)
    r = F.fit_lights(s, n, opts=F.FitLightsOpts(max_cheat=2))
    assert all(l.intensity >= 0.0 for l in r.lights) and r.ambient >= 0.0
    assert len(r.lights) <= 1 + F.MAX_CHEAT_LIGHTS


# ------------------------------------------------------------------ RD-09 이음매 (R7 <-> R8 교대)
def test_rd09_alternation_seam():
    sc_n = H.dome(24, 32)
    s_true = H.shading_from(sc_n, [(H.DOME_DIR, 0.75)], 0.25)
    albedo = np.empty((24, 32, 3)); albedo[:, :16] = (0.7, 0.4, 0.2); albedo[:, 16:] = (0.2, 0.5, 0.7)
    I = (albedo * s_true[..., None]).astype(np.float32)
    head = s_true * np.exp(0.03 * np.sin(np.arange(32))[None, :])
    d1 = X.decompose(I, normal=sc_n, shading_head=head)                       # R7: S 와 초기 광원
    assert d1.geo_source == "light_fit"
    fit = F.fit_lights(d1.shading.astype(np.float64), sc_n, valid=d1.valid, init_light=d1.light)       # R8
    assert fit.status == "ok" and len(fit.lights) == 1
    assert H.angle_deg(fit.lights[0].direction, H.DOME_DIR) < 1.0
    gl = fit.geo_light()                                                      # RD-09 GeoLight 로 되돌림
    assert isinstance(gl, X.GeoLight) and gl.direction == fit.lights[0].direction
    assert gl.intensity == fit.lights[0].intensity and gl.ambient == fit.ambient
    assert np.allclose(fit.geo_shading, X.geometric_shading(sc_n, gl), atol=1e-5)
    d2 = X.decompose(I, normal=sc_n, geo_shading=fit.geo_shading)             # R7 재풀이 (geo_shading 경로)
    assert d2.geo_source == "geo_shading" and d2.r_intr < 0.02
    d3 = X.decompose(I, normal=sc_n, light=gl)                                # 고정 광원 경로도 같은 L_d
    assert np.array_equal(d3.L_d, d2.L_d) and d3.geo_source == "light"
    meta = {"read": {"lights": fit.meta()}}
    json.dumps(meta, allow_nan=False)
