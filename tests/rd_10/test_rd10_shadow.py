"""RD-10 그림자 에지 제약 · 반그림자 각크기 · 빛 인과 힌트(RD-08 이음매)."""
import math

import numpy as np
import pytest

import rd10_helpers as H
from buc.read import fit_lights as F
from buc.read import metric_scale as MS

S_MIN = 1e-4


# ------------------------------------------------------------------ find_shadow_edges
def _oracle_edges(s, n, valid, grad_tau, normal_tau):
    """계약: 4-이웃 쌍 (x,y)가 둘 다 유효하고 |ln S_x - ln S_y| > grad_tau 이며 ||n_x - n_y|| < normal_tau 이면 둘 다 표시."""
    h, w = s.shape
    ls = np.log(np.maximum(s, S_MIN))
    out = np.zeros((h, w), bool)
    for y in range(h):
        for x in range(w):
            for dy, dx in ((0, 1), (1, 0)):
                y2, x2 = y + dy, x + dx
                if y2 >= h or x2 >= w or not (valid[y, x] and valid[y2, x2]):
                    continue
                if abs(ls[y, x] - ls[y2, x2]) > grad_tau and np.linalg.norm(n[y, x] - n[y2, x2]) < normal_tau:
                    out[y, x] = out[y2, x2] = True
    return out


def test_find_shadow_edges_planar_step():
    h, w = 20, 28
    n = np.zeros((h, w, 3)); n[...] = (0.0, 0.0, -1.0)         # 평면: 기하 에지 없음
    s = np.full((h, w), 0.9); s[:, 15:] = 0.3                  # 열 14|15 사이 음영 단차 = 드리운 그림자 에지
    m = F.find_shadow_edges(s, n)
    assert m.dtype == bool and m.shape == (h, w)
    want = np.zeros((h, w), bool); want[:, 14:16] = True
    assert np.array_equal(m, want)
    # 같은 단차가 법선 꺾임(기하 크리스)과 겹치면 그림자 에지가 아니다
    n2 = n.copy(); n2[:, 15:] = H.unit([0.0, -0.8, -0.6])
    assert not F.find_shadow_edges(s, n2).any()
    # 유효하지 않은 화소는 쌍에서 빠진다
    v = np.ones((h, w), bool); v[:, 15:] = False
    assert not F.find_shadow_edges(s, n, valid=v).any()
    # 임계 위·아래
    assert not F.find_shadow_edges(s, n, grad_tau=2.0).any()               # |ln(0.9/0.3)| = 1.0986
    assert F.find_shadow_edges(s, n, grad_tau=1.0).any()


@pytest.mark.parametrize("seed", [0, 1])
def test_find_shadow_edges_matches_oracle_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = H.dome(18, 24)
    s = H.shading_from(n, [(H.DOME_DIR, 0.7)], 0.2) * np.exp(0.6 * (rng.random((18, 24)) > 0.85))
    v = rng.random((18, 24)) > 0.1
    for gt, nt in ((0.15, 0.1), (0.05, 0.03), (0.3, 0.5)):
        got = F.find_shadow_edges(s, n, valid=v, grad_tau=gt, normal_tau=nt)
        assert np.array_equal(got, _oracle_edges(s, n, v, gt, nt)), (gt, nt)


def test_find_shadow_edges_errors():
    n = H.dome(8, 8); s = np.full((8, 8), 0.5)
    with pytest.raises(ValueError):
        F.find_shadow_edges(s, n, grad_tau=0.0)
    with pytest.raises(ValueError):
        F.find_shadow_edges(s[:-1], n)
    with pytest.raises(TypeError):
        F.find_shadow_edges(s.tolist(), n)


# ------------------------------------------------------------------ ShadowEdge / shadow_direction_prior
def test_shadow_edge_validation():
    F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=(1, 1, 1))
    with pytest.raises(ValueError):
        F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=(0, 0, 0))     # 길이 0
    with pytest.raises(ValueError):
        F.ShadowEdge(shadow_point=(0, 0, np.nan), occluder_point=(0, 0, 1))
    with pytest.raises(ValueError):
        F.ShadowEdge(shadow_point=(0, 0), occluder_point=(0, 0, 1))
    with pytest.raises(ValueError):
        F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=(0, 0, 1), weight=0.0)


def test_shadow_direction_prior_exact_and_spread():
    d = H.unit([-0.4, -0.5, -0.77])
    rng = np.random.default_rng(0)
    edges = []
    for _ in range(8):
        q = rng.normal(size=3) * 3.0
        edges.append(F.ShadowEdge(shadow_point=tuple(q), occluder_point=tuple(q + d * rng.uniform(0.5, 2.0))))
    p = F.shadow_direction_prior(edges)                        # 방향 = 그림자점 -> 가림체 에지점 (표면 -> 광원)
    assert p.source == "shadow" and H.angle_deg(p.direction, d) < 1e-6
    assert p.sigma_rad == pytest.approx(0.02, abs=1e-9)        # 일치하면 sigma_floor
    p2 = F.shadow_direction_prior(edges, sigma_floor=0.05)
    assert p2.sigma_rad == pytest.approx(0.05)
    # 두 방향(각각 +-10도)이 반반이면 평균 방향 + sigma = 각편차 rms = 10도
    e1 = H.unit([0.0, 0.0, -1.0])
    a = math.radians(10.0)
    u1 = np.array([math.sin(a), 0.0, -math.cos(a)]); u2 = np.array([-math.sin(a), 0.0, -math.cos(a)])
    ed = [F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=tuple(u1 * 2.0)),
          F.ShadowEdge(shadow_point=(1, 0, 0), occluder_point=tuple(np.array([1, 0, 0]) + u2 * 0.5))]
    p3 = F.shadow_direction_prior(ed)
    assert H.angle_deg(p3.direction, e1) < 1e-6 and p3.sigma_rad == pytest.approx(a, rel=1e-6)    # 가중은 방향 단위벡터에 대해서만
    # 가중
    ed_w = [F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=tuple(u1), weight=3.0),
            F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=tuple(u2), weight=1.0)]
    p4 = F.shadow_direction_prior(ed_w)
    want = H.unit(3.0 * u1 + u2)
    assert H.angle_deg(p4.direction, want) < 1e-6
    # 길이는 방향에 영향 없음
    ed_len = [F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=tuple(u1 * 100.0)),
              F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=tuple(u2 * 0.01))]
    assert H.angle_deg(F.shadow_direction_prior(ed_len).direction, e1) < 1e-6
    with pytest.raises(F.FitLightsError):
        F.shadow_direction_prior([])
    opp = [F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=(0, 0, 1)),
           F.ShadowEdge(shadow_point=(0, 0, 0), occluder_point=(0, 0, -1))]
    with pytest.raises(F.FitLightsError):
        F.shadow_direction_prior(opp)                          # 방향이 상쇄


# ------------------------------------------------------------------ 반그림자 -> 각크기 -> 태양/면광
def test_penumbra_and_classification():
    assert F.penumbra_angular_size(0.05, 5.0) == pytest.approx(0.01)       # w ~ d * theta_s
    assert F.penumbra_angular_size(0.5, 2.0) == pytest.approx(0.25)
    for bad in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, float("nan"))):
        with pytest.raises(ValueError):
            F.penumbra_angular_size(*bad)
    assert F.classify_source(0.01) == "sun" and F.classify_source(0.25) == "area"
    assert F.classify_source(F.SUN_ANGULAR_MAX) == "area" and F.classify_source(0.0) == "sun"
    with pytest.raises(ValueError):
        F.classify_source(-0.1)


# ------------------------------------------------------------------ 빛 인과 힌트 (RD-08 이음매)
def _ms():
    a = MS.anchor_subject_distance(2.0, 1.0)                   # s = 2 m / 상대단위, sigma_log = 0.10
    return MS.estimate_metric_scale([], anchors=[a])


def test_light_hint_area_with_metric_scale():
    d = H.unit([-0.5, -0.5, -0.7])
    area = F.Light(name="key", role="motivated", kind="area", direction=tuple(d), intensity=0.6, angular_radius=0.1)
    ms = _ms()
    h = F.light_hint(area, source_size_m=(0.5, 1.5), metric=ms)
    az, el = F.direction_to_az_el(tuple(d))
    assert (h.az_deg, h.el_deg) == (pytest.approx(az), pytest.approx(el))
    theta = 0.2                                                # 각지름 = 2 x angular_radius
    assert h.distance_m == pytest.approx((0.5 / theta, 1.5 / theta))       # D = 크기 / theta_s -> (2.5, 7.5) m
    lo, hi = ms.interval(2.0)                                  # s 의 2-sigma 구간
    assert h.distance_rel == pytest.approx((h.distance_m[0] / hi, h.distance_m[1] / lo))
    assert h.distance_rel[0] < h.distance_m[0] / ms.s < h.distance_m[1] / ms.s < h.distance_rel[1]
    assert h.metric_reliable is True
    h0 = F.light_hint(area)                                    # 스케일 없음
    assert h0.distance_m == h.distance_m and h0.distance_rel is None and h0.metric_reliable is None


def test_light_hint_sun_has_no_distance_and_validation():
    sun = F.Light(name="key", role="motivated", kind="sun", direction=tuple(H.unit([0, -1, 0.0001])), intensity=0.9)
    h = F.light_hint(sun, metric=_ms())
    assert h.distance_m is None and h.distance_rel is None
    assert h.el_deg == pytest.approx(90.0, abs=0.1)
    pt = F.Light(name="cheat1", role="cheat", kind="spot", direction=tuple(H.KEY_DIR), intensity=0.2)   # 각크기 0 -> 거리 미정
    assert F.light_hint(pt).distance_m is None
    area = F.Light(name="key", role="motivated", kind="area", direction=tuple(H.KEY_DIR), intensity=0.6, angular_radius=0.1)
    for sz in ((0.0, 1.0), (2.0, 1.0), (1.0, float("nan"))):
        with pytest.raises(ValueError):
            F.light_hint(area, source_size_m=sz)
