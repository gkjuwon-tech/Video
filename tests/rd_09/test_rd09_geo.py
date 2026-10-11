"""RD-09 기하 음영 Ŝ_geo 와 조명 적합: 수식 오라클, 복원, 목적식 최적성, 가중·마스크 의미, 경계 사례."""
import math

import numpy as np
import pytest

from buc.read import intrinsics as X
import rd09_helpers as H


def light(d=H.LIGHT_DIR, k=H.KEY, a=H.AMB):
    return X.GeoLight(direction=tuple(d), intensity=k, ambient=a)


# ------------------------------------------------------------------------------------------------ geometric_shading
def test_geometric_shading_matches_formula():
    n = H.dome_normals(20, 28)
    g = X.geometric_shading(n, light())
    assert g.shape == (20, 28) and g.dtype == np.float64
    np.testing.assert_allclose(g, H.AMB + H.KEY * np.maximum(n @ H.LIGHT_DIR, 0.0), rtol=0, atol=1e-13)
    assert g.min() == pytest.approx(H.AMB, abs=1e-12)           # 어두운 쪽은 주변광만
    assert (n @ H.LIGHT_DIR > 0).any() and (n @ H.LIGHT_DIR < 0).any()      # 장면이 두 영역을 모두 가짐
    # float32 법선도 같은 값(1e-6)
    g32 = X.geometric_shading(n.astype(np.float32), light())
    np.testing.assert_allclose(g32, g, atol=2e-7)


def test_geometric_shading_visibility_scales_only_directional_term():
    n = H.dome_normals(20, 28)
    vis = np.clip(H.smooth_field(20, 28, 5, 0.5) + 0.5, 0, 1)
    g = X.geometric_shading(n, light(), visibility=vis)
    np.testing.assert_allclose(g, H.AMB + H.KEY * vis * np.maximum(n @ H.LIGHT_DIR, 0.0), atol=1e-13)
    np.testing.assert_allclose(X.geometric_shading(n, light(), visibility=np.ones((20, 28))),
                               X.geometric_shading(n, light()), atol=0)
    assert np.allclose(X.geometric_shading(n, light(), visibility=np.zeros((20, 28))), H.AMB, atol=1e-13)


def test_geometric_shading_zero_normal_is_ambient_and_limits():
    n = np.zeros((6, 6, 3))
    assert np.allclose(X.geometric_shading(n, light()), H.AMB, atol=0)
    n2 = H.dome_normals(8, 8)
    assert np.allclose(X.geometric_shading(n2, light(k=0.0, a=0.4)), 0.4, atol=0)
    # 광원 쪽을 정면으로 보는 법선 -> amb + k
    nf = np.broadcast_to(H.LIGHT_DIR, (4, 4, 3)).copy()
    assert np.allclose(X.geometric_shading(nf, light()), H.AMB + H.KEY, atol=1e-12)


def test_geometric_shading_validation():
    n = H.dome_normals(8, 8)
    with pytest.raises(TypeError):
        X.geometric_shading(n, (0, 0, -1))
    with pytest.raises(TypeError):
        X.geometric_shading(n.astype(np.float16), light())
    with pytest.raises(ValueError):
        X.geometric_shading(n[..., :2], light())
    with pytest.raises(ValueError):
        X.geometric_shading(np.where(n > 0.9, np.nan, n), light())
    for bad in (np.full((8, 8), 1.1), np.full((8, 8), -0.1), np.ones((8, 7))):
        with pytest.raises(ValueError):
            X.geometric_shading(n, light(), visibility=bad)


# ------------------------------------------------------------------------------------------------ fit_geo_light
def test_fit_recovers_noise_free_light():
    n = H.dome_normals(24, 32)
    for d, k, a in ((H.LIGHT_DIR, H.KEY, H.AMB), (H.LIGHT2_DIR, 0.5, 0.4), (H.unit([0.0, -0.6, -0.8]), 1.2, 0.1)):
        S = H.geo_shading(n, d, a, k)
        f = X.fit_geo_light(n, S)
        assert isinstance(f, X.LightFit) and isinstance(f.light, X.GeoLight)
        assert H.angle_deg(f.light.direction, d) <= 1.0
        assert f.light.intensity == pytest.approx(k, rel=0.02)
        assert f.light.ambient == pytest.approx(a, abs=0.01)
        assert f.rms <= 5e-3 and f.n_used == 24 * 32


def test_fit_objective_matches_global_brute_force_on_noisy_data():
    n = H.dome_normals(20, 28)
    rng = np.random.default_rng(7)
    S = H.geo_shading(n, H.LIGHT2_DIR, 0.3, 0.6) + 0.03 * rng.standard_normal(n.shape[:2])
    w = np.ones(S.shape)
    f = X.fit_geo_light(n, S)
    ob, db, ab, kb = H.brute_light(n.reshape(-1, 3), S.reshape(-1), w.reshape(-1), ndirs=2500)
    of = H.light_objective(n.reshape(-1, 3), S.reshape(-1), w.reshape(-1), f.light.direction, f.light.ambient,
                           f.light.intensity)
    base = float(np.sum((S - S.mean()) ** 2))
    assert of <= ob + 1e-3 * ob + 1e-5 * base
    assert H.angle_deg(f.light.direction, db) <= 6.0
    assert f.light.intensity >= 0 and f.light.ambient >= 0
    # 보고 rms = sqrt(Σ w r² / Σ w)
    r = S - f.light.ambient - f.light.intensity * np.maximum(n @ np.asarray(f.light.direction), 0.0)
    assert f.rms == pytest.approx(math.sqrt(float(np.mean(r * r))), rel=1e-6)


def test_fit_constraints_when_unconstrained_optimum_is_negative():
    # 광원 반대편이 밝은 음영: 어떤 ω 도 음의 세기를 쓸 수 없다 -> 세기 0 에 가깝고 주변광 ≈ 평균
    n = H.dome_normals(20, 28)
    S = 0.8 - 0.5 * np.maximum(n @ H.LIGHT_DIR, 0.0)
    f = X.fit_geo_light(n, S)
    assert f.light.intensity >= 0 and f.light.ambient >= 0
    of = H.light_objective(n.reshape(-1, 3), S.reshape(-1), np.ones(S.size), f.light.direction, f.light.ambient,
                           f.light.intensity)
    ob, *_ = H.brute_light(n.reshape(-1, 3), S.reshape(-1), np.ones(S.size), ndirs=1500)
    assert of <= ob * (1 + 1e-3) + 1e-6 * float(np.sum((S - S.mean()) ** 2))
    assert abs(np.linalg.norm(f.light.direction) - 1.0) < 1e-9


def test_fit_weight_and_mask_semantics():
    n = H.dome_normals(24, 32)
    S = H.geo_shading(n, H.LIGHT2_DIR, 0.3, 0.6)
    base = X.fit_geo_light(n, S)
    # 가중치 전체 상수배는 결과를 바꾸지 않는다
    f2 = X.fit_geo_light(n, S, weight=np.full(S.shape, 7.5))
    assert H.angle_deg(f2.light.direction, base.light.direction) < 1e-3
    assert f2.light.intensity == pytest.approx(base.light.intensity, rel=1e-4)
    assert f2.rms == pytest.approx(base.rms, abs=1e-6)
    # 가중 0 인 곳과 mask=False 인 곳은 값이 쓰레기여도 영향이 없다
    rng = np.random.default_rng(1)
    bad = rng.random(S.shape) < 0.3
    S_bad = np.where(bad, 5.0 * rng.random(S.shape), S)
    fw = X.fit_geo_light(n, S_bad, weight=(~bad).astype(np.float64))
    fm = X.fit_geo_light(n, S_bad, mask=~bad)
    for f in (fw, fm):
        assert H.angle_deg(f.light.direction, H.LIGHT2_DIR) <= 1.0
        assert f.light.intensity == pytest.approx(0.6, rel=0.02) and f.n_used == int((~bad).sum())
    # 쓰레기를 가중 1로 두면 눈에 띄게 틀어진다(위 두 경우가 우연히 통과한 것이 아님)
    ff = X.fit_geo_light(n, S_bad)
    assert abs(ff.light.intensity - 0.6) > 0.02 or H.angle_deg(ff.light.direction, H.LIGHT2_DIR) > 1.0


def test_fit_ignores_invalid_normals_and_counts_used():
    n = H.dome_normals(24, 32).copy()
    S = H.geo_shading(n, H.LIGHT_DIR)
    n[:6] = 0.0                         # 법선 없음(NORMAL_MIN 미만)
    S[:6] = 123.0
    n[6:8] *= 0.3                       # 노름 0.3 < 0.5 도 무효
    S[6:8] = -50.0
    f = X.fit_geo_light(n, S)
    assert f.n_used == (24 - 8) * 32
    assert H.angle_deg(f.light.direction, H.LIGHT_DIR) <= 1.0
    assert f.light.intensity == pytest.approx(H.KEY, rel=0.02)


def test_fit_normalizes_normals_used_for_cosine():
    n = H.dome_normals(24, 32)
    S = H.geo_shading(n, H.LIGHT_DIR)
    f1 = X.fit_geo_light(n, S)
    f2 = X.fit_geo_light(n * 2.5, S)             # 노름 2.5 (≥0.5)는 단위로 되돌려 쓴다
    assert H.angle_deg(f1.light.direction, f2.light.direction) < 1e-3
    assert f2.light.intensity == pytest.approx(f1.light.intensity, rel=1e-4)


def test_fit_errors_and_determinism():
    n = H.dome_normals(24, 32)
    S = H.geo_shading(n, H.LIGHT_DIR)
    few = np.zeros(S.shape, bool)
    few.reshape(-1)[:15] = True
    with pytest.raises(X.IntrinsicsError):
        X.fit_geo_light(n, S, mask=few)
    few.reshape(-1)[:16] = True
    X.fit_geo_light(n, S, mask=few)             # 정확히 MIN_LIGHT_PIXELS 개는 허용
    with pytest.raises(X.IntrinsicsError):
        X.fit_geo_light(n, S, weight=np.zeros(S.shape))
    with pytest.raises(ValueError):
        X.fit_geo_light(n, S[:-1])
    with pytest.raises(ValueError):
        X.fit_geo_light(n, S, weight=-np.ones(S.shape))
    with pytest.raises(TypeError):
        X.fit_geo_light(n, S, mask=np.ones(S.shape, np.uint8))
    with pytest.raises(ValueError):
        X.fit_geo_light(n, np.where(S > 0.5, np.nan, S))
    a, b = X.fit_geo_light(n, S), X.fit_geo_light(n, S)
    assert a.light == b.light and a.rms == b.rms and a.n_used == b.n_used
    n0, S0 = n.copy(), S.copy()
    X.fit_geo_light(n, S)
    assert np.array_equal(n, n0) and np.array_equal(S, S0)
