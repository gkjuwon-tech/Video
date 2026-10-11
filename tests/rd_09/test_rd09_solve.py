"""RD-09 분해 풀이: 에너지 최적성(독립 오라클), L_d 소유·재구성 불변식, 대칭·스케일 등변, 제외 마스크, 옵션 극한,
스트라이드(저해상 풀이 + 로그 쌍선형 업샘플), 경고."""
import math
import time

import numpy as np
import pytest

from buc.read import intrinsics as X
import rd09_helpers as H


def messy_case(seed=0, h=10, w=12):
    """모든 항이 켜진 임의 사례: 노이즈 낀 영상, 비일관 사전들, 임의 가중·제외."""
    rng = np.random.default_rng(seed)
    n = H.dome_normals(h, w)
    alb = np.clip(0.5 + 0.25 * rng.standard_normal((h, w, 3)), 0.05, 1.0)
    S = H.geo_shading(n) * np.exp(0.2 * H.smooth_field(h, w, seed))
    I = (alb * S[..., None]).astype(np.float32)
    geo = H.geo_shading(n) * np.exp(0.1 * rng.standard_normal((h, w)))
    head = S * np.exp(0.15 * rng.standard_normal((h, w)))
    ahead = np.clip(alb * np.exp(0.2 * rng.standard_normal((h, w, 3))), 0.01, 2.0)
    gw = rng.random((h, w))
    ex = rng.random((h, w)) < 0.1
    return dict(I=I, normal=n, geo_shading=geo, shading_head=head, albedo_head=ahead, geo_weight=gw, exclude=ex)


def E_of(case, opts):
    return lambda s: H.energy(s, case["I"], normal=case["normal"], geo=case["geo_shading"],
                              geo_weight=case["geo_weight"], shading_head=case["shading_head"],
                              albedo_head=case["albedo_head"], exclude=case["exclude"], opts=opts)


def solve(case, opts):
    kw = {k: v for k, v in case.items() if k != "I"}
    return X.decompose(case["I"], opts=opts, **kw)


@pytest.mark.parametrize("seed,opts", [
    (0, X.IntrinsicsOpts()),
    (1, X.IntrinsicsOpts(alpha=0.7, beta=0.2, gamma=1.3, eta=0.4, delta=0.15, tau_chroma=0.08, tau_normal=0.6)),
    (2, X.IntrinsicsOpts(alpha=2.0, beta=0.0, gamma=0.5, eta=0.0, delta=0.3)),
])
def test_solution_is_the_minimizer_of_the_contract_energy(seed, opts):
    case = messy_case(seed)
    r = solve(case, opts)
    s = np.log(r.shading.astype(np.float64))
    f = E_of(case, opts)
    valid = ~case["exclude"]
    g = H.grad_fd(f, s, 1e-3)
    g0 = H.grad_fd(f, np.zeros_like(s), 1e-3)
    assert np.abs(g[valid]).max() <= 2e-5 * (1.0 + np.abs(g0[valid]).max())
    e = f(s)
    rng = np.random.default_rng(seed)
    for _ in range(6):                      # 볼록: 어떤 섭동도 에너지를 올린다
        d = rng.standard_normal(s.shape) * valid
        for eps in (1e-2, 1e-1):
            assert f(s + eps * d) > e
    assert f(np.where(valid, np.log(np.maximum(case["shading_head"], 1e-4)), s)) > e
    assert f(np.where(valid, np.log(np.maximum(case["geo_shading"], 1e-4)), s)) > e


def test_result_invariants_ownership_and_reconstruction():
    case = messy_case(3)
    I = case["I"]
    for img in (I, I.astype(np.float64)):
        r = X.decompose(img, **{k: v for k, v in case.items() if k != "I"})
        assert np.array_equal(r.L_d, img.astype(np.float32))             # 첫 프레임 L_d 는 입력 그대로(비트)
        assert np.isfinite(r.shading).all() and (r.shading > 0).all()
        assert np.isfinite(r.albedo).all() and (r.albedo >= 0).all()
        np.testing.assert_allclose(r.albedo.astype(np.float64) * r.shading[..., None].astype(np.float64),
                                   r.L_d.astype(np.float64), rtol=5e-6, atol=1e-9)
        np.testing.assert_allclose(r.albedo, r.L_d.astype(np.float64) / r.shading[..., None].astype(np.float64),
                                   rtol=2e-6, atol=1e-9)
    assert np.array_equal(r.valid, ~case["exclude"])


def test_horizontal_flip_equivariance():
    case = messy_case(4, 10, 14)
    kw = {k: v for k, v in case.items() if k != "I"}
    a = X.decompose(case["I"], **kw)
    fl = {k: v[:, ::-1].copy() for k, v in case.items()}
    fl["normal"] = fl["normal"] * np.array([-1.0, 1.0, 1.0])
    b = X.decompose(fl["I"], **{k: v for k, v in fl.items() if k != "I"})
    np.testing.assert_allclose(b.shading, a.shading[:, ::-1], rtol=1e-5)
    np.testing.assert_allclose(b.albedo, a.albedo[:, ::-1], rtol=1e-5, atol=1e-8)
    assert np.array_equal(b.valid, a.valid[:, ::-1])
    # 위아래도 마찬가지
    fv = {k: v[::-1].copy() for k, v in case.items()}
    fv["normal"] = fv["normal"] * np.array([1.0, -1.0, 1.0])
    c = X.decompose(fv["I"], **{k: v for k, v in fv.items() if k != "I"})
    np.testing.assert_allclose(c.shading, a.shading[::-1], rtol=1e-5)


def test_scale_equivariance_with_albedo_prior_only():
    # 알베도 헤드만 절대 수준을 준다 -> 영상을 c배 하면 음영이 정확히 c배, 알베도는 그대로
    sc = H.scene(14, 18)
    I = sc["I"].astype(np.float64)
    ahead = np.clip(sc["albedo"] * np.exp(0.15 * np.random.default_rng(5).standard_normal(I.shape)), 0.01, 1.5)
    a = X.decompose(I, albedo_head=ahead)
    b = X.decompose(4.0 * I, albedo_head=ahead)
    np.testing.assert_allclose(b.shading, 4.0 * a.shading, rtol=1e-5)
    np.testing.assert_allclose(b.albedo, a.albedo, rtol=1e-5)
    assert a.geo_source == "none" and "no_geometry" in a.warnings


def test_excluded_pixels_fallback_and_isolation():
    case = messy_case(6, 12, 16)
    ex = np.zeros((12, 16), bool)
    ex[3:6, 4:9] = True
    case["exclude"] = ex
    kw = {k: v for k, v in case.items() if k != "I"}
    a = X.decompose(case["I"], **kw)
    # 폴백 순서: geo_shading > shading_head > albedo_head
    np.testing.assert_allclose(a.shading[ex], np.maximum(case["geo_shading"], 1e-4)[ex], rtol=2e-6)
    c = X.decompose(case["I"], **{k: v for k, v in kw.items() if k not in ("geo_shading", "normal")})
    np.testing.assert_allclose(c.shading[ex], np.maximum(case["shading_head"], 1e-4)[ex], rtol=2e-6)
    d = X.decompose(case["I"], **{k: v for k, v in kw.items() if k not in ("geo_shading", "normal", "shading_head")})
    I64 = case["I"].astype(np.float64)
    sa = (np.log(np.maximum(I64, 1e-6)) - np.log(np.maximum(case["albedo_head"], 1e-6))).mean(-1)
    np.testing.assert_allclose(d.shading[ex], np.exp(sa)[ex], rtol=2e-6)
    for r in (a, c, d):
        assert np.array_equal(r.valid, ~ex)
        assert np.array_equal(r.L_d[ex], case["I"][ex])
        np.testing.assert_allclose(r.albedo[ex], r.L_d[ex] / r.shading[ex][:, None], rtol=2e-6)
    # 제외 화소의 영상 값은 나머지 풀이에 영향이 없다
    I2 = case["I"].copy()
    I2[ex] = np.random.default_rng(0).random((int(ex.sum()), 3)).astype(np.float32) * 9.0
    a2 = X.decompose(I2, **kw)
    np.testing.assert_allclose(a2.shading[~ex], a.shading[~ex], rtol=1e-6)
    # 제외 화소의 사전 값도 풀이에 영향이 없다(제외는 그래프와 사전에서 모두 빠진다)
    kw3 = dict(kw)
    kw3["shading_head"] = np.where(ex, 50.0, kw["shading_head"])
    kw3["albedo_head"] = np.where(ex[..., None], 5.0, kw["albedo_head"])
    a3 = X.decompose(case["I"], **kw3)
    np.testing.assert_allclose(a3.shading[~ex], a.shading[~ex], rtol=1e-6)


def test_status_is_exclusion_and_dominant_exclusion_warns():
    case = messy_case(7, 12, 16)
    kw = {k: v for k, v in case.items() if k not in ("I", "exclude")}
    st = np.zeros((12, 16), np.uint8)
    st[2:5, 3:7] = 1
    st[8, 8] = 2
    st[9, 1] = 4
    a = X.decompose(case["I"], status=st, **kw)
    b = X.decompose(case["I"], exclude=st != 0, **kw)
    assert np.array_equal(a.shading, b.shading) and np.array_equal(a.albedo, b.albedo)
    assert np.array_equal(a.valid, st == 0)
    both = np.zeros((12, 16), bool)
    both[0, :] = True
    c = X.decompose(case["I"], status=st, exclude=both, **kw)
    assert np.array_equal(c.valid, (st == 0) & ~both)
    assert c.meta()["n_valid"] == int(c.valid.sum())
    assert "excluded_dominant" not in a.warnings
    big = np.zeros((12, 16), np.uint8)
    big[:, :9] = 2                                    # 75% 제외
    d = X.decompose(case["I"], status=big, **kw)
    assert "excluded_dominant" in d.warnings and not d.reliable
    edge = np.zeros((12, 16), bool)
    edge.reshape(-1)[: 12 * 16 // 2] = True          # 정확히 50% -> 경고 아님(초과일 때만)
    assert "excluded_dominant" not in X.decompose(case["I"], exclude=edge, **kw).warnings


def test_option_limits():
    sc = H.scene(16, 20)
    I = sc["I"].astype(np.float32)
    # alpha=0: 영상 내용과 무관(사전만으로 결정)
    o0 = X.IntrinsicsOpts(alpha=0.0)
    kw = dict(geo_shading=sc["shading"], shading_head=sc["shading"] * 1.1, normal=sc["normal"])
    a = X.decompose(I, opts=o0, **kw)
    b = X.decompose(np.random.default_rng(0).random(I.shape).astype(np.float32) + 0.01, opts=o0, **kw)
    np.testing.assert_allclose(a.shading, b.shading, rtol=1e-6)
    # gamma 가 압도하면 음영은 기하 음영
    og = X.IntrinsicsOpts(gamma=1e6, beta=0.1, alpha=1.0)
    g = X.decompose(I * np.float32(0.5), opts=og, geo_shading=sc["shading"])
    np.testing.assert_allclose(np.log(g.shading), np.log(sc["shading"]), atol=1e-4)
    # eta 만(그리고 알베도·기하 없음, 음영 매끄러움 약함) -> 헤드 근처
    oh = X.IntrinsicsOpts(alpha=0.0, beta=0.0, gamma=0.0, eta=1.0, delta=0.0)
    h = X.decompose(I, opts=oh, shading_head=sc["shading"])
    np.testing.assert_allclose(h.shading, sc["shading"], rtol=2e-6)
    # beta 가 매우 크면 음영은 (유효 영역에서) 거의 상수: 사전들의 가중 평균
    ob = X.IntrinsicsOpts(alpha=0.0, beta=1e6, gamma=1.0, eta=0.0, delta=0.0, tau_normal=1e6)
    s = X.decompose(I, opts=ob, geo_shading=sc["shading"], normal=sc["normal"])
    ls = np.log(s.shading)
    assert ls.max() - ls.min() < 1e-3
    assert ls.mean() == pytest.approx(np.log(np.maximum(sc["shading"], 1e-4)).mean(), abs=2e-3)


def test_chroma_weight_edge_awareness():
    # 같은 휘도 단차라도 색도가 바뀌면 알베도 변화를 허용(음영이 따라가지 않는다), 색도가 같으면 음영이 일부 흡수.
    h, w = 16, 24
    n = np.zeros((h, w, 3)); n[..., 2] = -1.0
    left, right = 0.25, 0.5
    chrom = np.empty((h, w, 3))
    chrom[:, :w // 2] = (np.array([0.8, 0.4, 0.2]) / 0.8 * left)
    chrom[:, w // 2:] = (np.array([0.2, 0.4, 0.8]) / 0.8 * right)
    gray = np.empty((h, w, 3)); gray[:, :w // 2] = left; gray[:, w // 2:] = right
    o = X.IntrinsicsOpts(alpha=1.0, beta=1.0, gamma=0.0, eta=0.2, delta=0.0)
    head = np.ones((h, w))
    rc = X.decompose(chrom.astype(np.float32), normal=n, shading_head=head, opts=o)
    rg = X.decompose(gray.astype(np.float32), normal=n, shading_head=head, opts=o)
    step = lambda r: abs(np.log(r.shading[:, w // 2 + 1]).mean() - np.log(r.shading[:, w // 2 - 2]).mean())
    assert step(rc) < 0.05                  # 색도 단차: 음영은 거의 안 움직임
    assert step(rg) > 2 * step(rc) + 0.02   # 순수 휘도 단차: 일부는 음영이 흡수(Retinex 모호성)


def test_normal_discontinuity_frees_shading_smoothness():
    h, w = 16, 24
    n = np.zeros((h, w, 3)); n[..., 2] = -1.0
    n2 = n.copy()
    n2[:, w // 2:] = H.unit([0.9, 0.0, -0.3])                  # 오른쪽 면이 완전히 다른 방향(주름)
    S = np.where(np.arange(w) < w // 2, 0.9, 0.3)[None, :].repeat(h, 0)
    I = (np.full((h, w, 3), 0.5) * S[..., None]).astype(np.float32)
    o = X.IntrinsicsOpts(alpha=0.0, beta=5.0, gamma=0.0, eta=1.0, delta=0.0, tau_normal=0.3)
    flat = X.decompose(I, normal=n, shading_head=S, opts=o)       # 법선 연속: 평활이 단차를 눌러 놓음
    crease = X.decompose(I, normal=n2, shading_head=S, opts=o)    # 주름: 평활 약화, 헤드의 단차 보존
    k = lambda r: np.log(r.shading[:, w // 2 - 2]).mean() - np.log(r.shading[:, w // 2 + 1]).mean()
    assert k(crease) > 1.05 and k(crease) > k(flat) + 0.3
    assert abs(k(crease) - math.log(3)) < 0.05


def test_stride_solves_low_res_and_upsamples_log_bilinear():
    h, w = 47, 64
    n = H.dome_normals(h, w)
    alb = H.two_patch_albedo(h, w)
    S = H.geo_shading(n)
    I = (alb * S[..., None]).astype(np.float32)
    head = S * np.exp(0.1 * H.smooth_field(h, w, 2))
    ex = np.zeros((h, w), bool)
    ex[10:13, 31:36] = True
    o = X.IntrinsicsOpts(max_side=32)
    r = X.decompose(I, normal=n, geo_shading=S, shading_head=head, exclude=ex, opts=o)
    f = r.solve_stride
    assert f == 2 and r.n_solves == 1 and r.meta()["solve_stride"] == 2
    lo = X.decompose(np.ascontiguousarray(I[::f, ::f]), normal=n[::f, ::f], geo_shading=S[::f, ::f],
                     shading_head=head[::f, ::f], exclude=ex[::f, ::f], opts=X.IntrinsicsOpts(max_side=64))
    assert lo.solve_stride == 1 and lo.shading.shape == (24, 32)
    ok = ~ex
    sel = np.zeros((h, w), bool); sel[::f, ::f] = True
    np.testing.assert_allclose(r.shading[sel & ok], lo.shading[ok[::f, ::f]], rtol=1e-6)
    # 나머지 화소: 저해상 ln 음영을 표본 좌표 (y/f, x/f)로 쌍선형 보간(끝은 마지막 표본으로 고정)
    from scipy import ndimage
    s = np.log(r.shading.astype(np.float64))
    sl = np.log(lo.shading.astype(np.float64))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    exp_s = ndimage.map_coordinates(sl, [yy / f, xx / f], order=1, mode="nearest")
    np.testing.assert_allclose(s[ok], exp_s[ok], atol=3e-6)
    assert s[1, 0] == pytest.approx(0.5 * (sl[0, 0] + sl[1, 0]), abs=3e-6)          # 구체적인 한 점
    assert s[46, 5] == pytest.approx(0.5 * (sl[23, 2] + sl[23, 3]), abs=3e-6)       # 마지막 행 46 은 표본 행 23, 47 은 고정
    assert s[10, 63] == pytest.approx(sl[5, 31], abs=3e-6)                          # 마지막 열 63 은 표본 열 31 로 고정
    # 전해상도 L_d 소유·재구성은 그대로, 제외 화소는 전해상도에서 폴백
    assert np.array_equal(r.L_d, I)
    np.testing.assert_allclose(r.albedo.astype(np.float64) * r.shading[..., None], r.L_d, rtol=5e-6)
    np.testing.assert_allclose(r.shading[ex], np.maximum(S, 1e-4)[ex], rtol=2e-6)
    assert np.array_equal(r.valid, ~ex)


def test_stride_one_when_small_and_exact_boundary():
    I = np.full((20, 20, 3), 0.3, np.float32)
    sh = np.full((20, 20), 0.5, np.float32)
    assert X.decompose(I, shading_head=sh, opts=X.IntrinsicsOpts(max_side=20)).solve_stride == 1
    assert X.decompose(I, shading_head=sh, opts=X.IntrinsicsOpts(max_side=19)).solve_stride == 2
    assert X.decompose(I, shading_head=sh, opts=X.IntrinsicsOpts(max_side=8)).solve_stride == 3


def test_no_geometry_and_light_fit_poor_warnings():
    sc = H.scene(16, 20)
    I = sc["I"].astype(np.float32)
    r = X.decompose(I, shading_head=sc["shading"])
    assert r.warnings == ("no_geometry",) and not r.reliable and r.light is None and r.geo_source == "none"
    assert np.isnan(r.r_intr)
    # 법선과 무관한 음영 헤드(백색 잡음): 광원 적합이 설명하지 못한다
    rng = np.random.default_rng(0)
    junk = 0.2 + 0.7 * rng.random((16, 20))
    p = X.decompose(I, normal=sc["normal"], shading_head=junk)
    assert "light_fit_poor" in p.warnings and p.geo_source == "light_fit"
    good = X.decompose(I, normal=sc["normal"], shading_head=sc["shading"])
    assert "light_fit_poor" not in good.warnings and "no_geometry" not in good.warnings


def test_alternation_count_and_fixed_light_paths():
    sc = H.scene(16, 20)
    I = sc["I"].astype(np.float32)
    head = sc["shading"] * np.exp(0.2 * H.smooth_field(16, 20, 9))
    for k in (1, 2, 3):
        r = X.decompose(I, normal=sc["normal"], shading_head=head, opts=X.IntrinsicsOpts(n_alt=k))
        assert r.n_solves == k
    lt = X.GeoLight(direction=tuple(H.LIGHT_DIR), intensity=H.KEY, ambient=H.AMB)
    a = X.decompose(I, normal=sc["normal"], light=lt, shading_head=head, opts=X.IntrinsicsOpts(n_alt=4))
    assert a.n_solves == 1 and a.geo_source == "light" and a.light == lt
    b = X.decompose(I, geo_shading=sc["shading"], shading_head=head, opts=X.IntrinsicsOpts(n_alt=4))
    assert b.n_solves == 1 and b.geo_source == "geo_shading" and b.light is None
    np.testing.assert_allclose(a.geo_shading, X.geometric_shading(sc["normal"], lt), rtol=1e-6)
    # 같은 음영을 light 로 주나 geo_shading 으로 주나 풀이가 같다
    c = X.decompose(I, normal=sc["normal"], geo_shading=X.geometric_shading(sc["normal"], lt), shading_head=head)
    np.testing.assert_allclose(c.shading, a.shading, rtol=1e-5)


def test_light_path_ignores_invalid_normals_for_geo_term():
    sc = H.scene(16, 20)
    I = sc["I"].astype(np.float32)
    lt = X.GeoLight(direction=tuple(H.LIGHT_DIR), intensity=H.KEY, ambient=H.AMB)
    n = sc["normal"].copy()
    n[:4] = 0.0                                          # 법선 없음 -> 기하 사전 없음(음영 헤드가 없으면 알베도 헤드가 대신)
    head = sc["shading"]
    a = X.decompose(I, normal=n, light=lt, shading_head=head)
    g = X.geometric_shading(n, lt)                       # 이 영역은 ambient 로만 계산됨 -> 그대로 쓰면 틀린 음영
    assert np.abs(np.log(a.shading[:4]) - np.log(sc["shading"][:4])).mean() < \
        np.abs(np.log(g[:4]) - np.log(sc["shading"][:4])).mean()


def test_speed_reasonable_at_working_resolution():
    h, w = 240, 320                          # max_side=160 -> 스트라이드 2, 풀이 120x160
    n = H.dome_normals(h, w)
    alb = H.two_patch_albedo(h, w)
    S = H.geo_shading(n)
    I = (alb * S[..., None]).astype(np.float32)
    t0 = time.time()
    r = X.decompose(I, normal=n, shading_head=S * np.exp(0.1 * H.smooth_field(h, w, 4)),
                    opts=X.IntrinsicsOpts(max_side=160))
    dt = time.time() - t0
    assert r.solve_stride == 2 and r.shading.shape == (h, w) and r.n_solves == 2
    assert dt < 20.0, dt
