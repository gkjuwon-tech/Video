"""RDR-08 가우시안 체적: τ = 수치 선적분 오라클, 조각 깊이 = t*의 z, SGGX 비등방 소광, 컬링,
합성 공식·순서 무관·순열 불변, backdrop 가림, 결정론·입력 불변."""
import math

import numpy as np
import torch

from buc.render.vol import VolOpts, render_vol
from rdr08_helpers import (K_of, at_pixel, fiber_S, make_strands, make_vol, pose, ray, rot_axis, tau_line_oracle,
                           tau_of, to_np)

W, H = 33, 33
FX = 200.0
K = K_of(FX, W, H)       # cx = cy = 16
I4 = np.eye(4)


def _r(vol=None, strands=None, c2w=I4, K=K, w=W, h=H, **kw):
    backdrop = kw.pop("backdrop", None)
    return render_vol(vol=vol, strands=strands, c2w=c2w, K=K, width=w, height=h, opts=VolOpts(**kw),
                      backdrop=backdrop)


def test_single_gaussian_tau_matches_line_integral():
    R = rot_axis([0.3, 1.0, -0.4], 35)
    s = np.array([0.06, 0.03, 0.02])
    mu = at_pixel(K, 16, 16, 3.0) + np.array([0.004, -0.003, 0.0])
    S = fiber_S([0.2, 1.0, 0.3], 0.3)
    rho = 4.0
    vol = make_vol(mu[None], R[None], s[None], [rho], [[0.8, 0.5, 0.2]], S[None])
    out = _r(vol, max_alpha=0.999999)
    a = to_np(out.alpha)
    dep = to_np(out.depth)
    Sigma = R @ np.diag(s ** 2) @ R.T
    for (u, v) in [(16, 16), (18, 15), (13, 17), (16, 20), (20, 19)]:
        C, wv = ray(K, I4, u, v)
        tau, tstar = tau_line_oracle(mu, Sigma, S, rho, C, wv)
        assert tau > 0.01
        got = tau_of(a[v, u])
        assert abs(got - tau) / tau < 1e-3, (u, v, got, tau)
        z_exp = (C + tstar * wv)[2]
        assert abs(dep[v, u] - z_exp) < 1e-5 * z_exp, (u, v, dep[v, u], z_exp)
    np.testing.assert_allclose(to_np(out.rgb)[16, 16], a[16, 16] * np.array([0.8, 0.5, 0.2]), atol=1e-9)


def test_center_ray_closed_form_and_depth():
    s = 0.05
    mu = at_pixel(K, 16, 16, 3.0)
    vol = make_vol(mu[None], np.eye(3)[None], [[s, s, s]], [2.0], [[1, 1, 1]])
    out = _r(vol)
    tau = 2.0 * 1.0 * math.sqrt(2 * math.pi) * s
    assert abs(tau_of(to_np(out.alpha)[16, 16]) - tau) < 1e-9
    assert abs(to_np(out.depth)[16, 16] - 3.0) < 1e-9


def test_fiber_anisotropy_ratio_equals_roughness():
    """같은 구형 가우시안, 섬유 SGGX(접선 = 세계 x). 접선을 따라 보면 τ가 roughness배."""
    mu = np.array([0.0, 0.0, 3.0])
    s = 0.05
    r = 0.1
    vol = make_vol(mu[None], np.eye(3)[None], [[s, s, s]], [3.0], [[1, 1, 1]], fiber_S([1, 0, 0], r)[None])
    a_front = to_np(_r(vol).alpha)[16, 16]
    Rb = np.stack([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]], axis=1)   # 카메라 z = 세계 +x
    c2w_b = pose(Rb, (-3.0, 0.0, 3.0))
    a_side = to_np(_r(vol, c2w=c2w_b).alpha)[16, 16]
    ratio = tau_of(a_side) / tau_of(a_front)
    assert abs(ratio - r) < 1e-3 * r + 1e-9, ratio


def test_culling_behind_and_far_and_cutoff():
    behind = make_vol([[0.0, 0.0, -2.0]], np.eye(3)[None], [[0.3, 0.3, 0.3]], [50.0], [[1, 0, 0]])
    out = _r(behind)
    assert np.isfinite(to_np(out.rgb)).all()
    assert to_np(out.alpha).max() == 0.0 and np.isinf(to_np(out.depth)).all()
    far = make_vol([[0.0, 0.0, 50.0]], np.eye(3)[None], [[3.0, 3.0, 3.0]], [5.0], [[1, 0, 0]])
    assert to_np(_r(far, far=20.0).alpha).max() == 0.0
    assert to_np(_r(far).alpha).max() > 0.5
    # cutoff: 4σ 밖 화소는 정확히 0, 안쪽은 > 0
    s = 0.03
    z = 3.0
    g = make_vol([at_pixel(K, 16, 16, z)], np.eye(3)[None], [[s, s, s]], [200.0], [[1, 1, 1]])
    a = to_np(_r(g).alpha)
    spx = FX * s / z            # 2 px
    for du in range(0, 16):
        m = du / spx            # 근사(작은 각): 광선-중심 마할라노비스 거리
        if m < 3.9:
            assert a[16, 16 + du] > 0, du
        elif m > 4.1:
            assert a[16, 16 + du] == 0.0, du


def _three(dtype=torch.float64):
    mus = np.array([at_pixel(K, 16, 16, 2.0), at_pixel(K, 16.5, 15.7, 3.0), at_pixel(K, 15.6, 16.3, 4.5)])
    s = np.array([[0.03, 0.025, 0.02], [0.05, 0.04, 0.03], [0.08, 0.07, 0.05]])
    rots = np.stack([rot_axis([1, 0, 0], 10), rot_axis([0, 1, 1], 25), rot_axis([1, 1, 0], 50)])
    cols = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
    return mus, rots, s, np.array([6.0, 5.0, 4.0]), cols


def test_compositing_formula_and_order_independent_alpha():
    mus, rots, s, rho, cols = _three()
    bg = (0.2, 0.3, 0.4)
    singles = [_r(make_vol(mus[i:i + 1], rots[i:i + 1], s[i:i + 1], rho[i:i + 1], cols[i:i + 1])) for i in range(3)]
    al = [to_np(x.alpha) for x in singles]
    dp = [to_np(x.depth) for x in singles]
    out = _r(make_vol(mus, rots, s, rho, cols), bg=bg)
    a_all, rgb_all, d_all = to_np(out.alpha), to_np(out.rgb), to_np(out.depth)
    np.testing.assert_allclose(a_all, 1 - (1 - al[0]) * (1 - al[1]) * (1 - al[2]), atol=1e-10)
    for (u, v) in [(16, 16), (17, 15), (15, 17)]:
        frags = sorted([(dp[i][v, u], al[i][v, u], cols[i]) for i in range(3) if al[i][v, u] > 0], key=lambda f: f[0])
        T, rgb, wz, wa = 1.0, np.zeros(3), 0.0, 0.0
        for z, a, c in frags:
            rgb += T * a * c
            wz += T * a * z
            wa += T * a
            T *= 1 - a
        rgb += T * np.array(bg)
        np.testing.assert_allclose(rgb_all[v, u], rgb, atol=1e-9)
        assert abs(d_all[v, u] - wz / wa) < 1e-9


def test_input_permutation_invariance_and_determinism():
    mus, rots, s, rho, cols = _three()
    pts = np.array([[[-0.1, -0.05, 2.5], [0.1, 0.05, 2.6]], [[0.05, -0.1, 3.5], [-0.05, 0.1, 3.4]]])
    st = make_strands(pts, [[0.004, 0.004], [0.003, 0.005]], [0.7, 0.6], [[1, 1, 0], [0, 1, 1]])
    perm = [2, 0, 1]
    st_p = make_strands(pts[::-1], [[0.003, 0.005], [0.004, 0.004]], [0.6, 0.7], [[0, 1, 1], [1, 1, 0]])
    o1 = _r(make_vol(mus, rots, s, rho, cols), st)
    o2 = _r(make_vol(mus[perm], rots[perm], s[perm], rho[perm], cols[perm]), st_p)
    for k in ("rgb", "alpha"):
        np.testing.assert_allclose(to_np(getattr(o1, k)), to_np(getattr(o2, k)), atol=1e-10)
    d1, d2 = to_np(o1.depth), to_np(o2.depth)
    assert np.array_equal(np.isinf(d1), np.isinf(d2))
    m = np.isfinite(d1)
    np.testing.assert_allclose(d1[m], d2[m], atol=1e-9)
    # 결정론(float32, 같은 바이트) + 입력·RNG 불변
    v32 = make_vol(mus, rots, s, rho, cols, dtype=torch.float32)
    before = [p.clone() for p in v32.params()] + [v32.sggx.clone()]
    rs = torch.get_rng_state()
    ns = np.random.get_state()[1].copy()
    a = _r(v32).numpy()
    b = _r(v32).numpy()
    for k in ("rgb", "alpha", "depth"):
        assert a[k].dtype == np.float32
        assert a[k].tobytes() == b[k].tobytes()
    for p, q in zip(before, list(v32.params()) + [v32.sggx]):
        assert torch.equal(p, q)
    assert torch.equal(rs, torch.get_rng_state())
    assert np.array_equal(ns, np.random.get_state()[1])


def test_backdrop_occlusion_and_merge():
    bd_rgb = np.zeros((H, W, 3))
    bd_a = np.zeros((H, W))
    bd_d = np.full((H, W), np.inf)
    bd_a[:, :] = 1.0
    bd_rgb[:, :] = [0.3, 0.6, 0.2]
    bd_d[:, :] = 5.0
    front = make_vol([at_pixel(K, 16, 16, 3.0)], np.eye(3)[None], [[0.05] * 3], [5.0], [[1.0, 0.0, 0.0]])
    behind = make_vol([at_pixel(K, 16, 16, 7.0)], np.eye(3)[None], [[0.1] * 3], [5.0], [[1.0, 0.0, 0.0]])
    bd = dict(rgb=bd_rgb, alpha=bd_a, depth=bd_d)
    a_f = to_np(_r(front).alpha)
    o = _r(front, backdrop=bd, bg=(0, 0, 1))
    exp = a_f[..., None] * np.array([1.0, 0, 0]) + (1 - a_f[..., None]) * bd_rgb
    np.testing.assert_allclose(to_np(o.rgb), exp, atol=1e-9)
    np.testing.assert_allclose(to_np(o.alpha), 1.0, atol=1e-12)
    ob = _r(behind, backdrop=bd, bg=(0, 0, 1))
    np.testing.assert_allclose(to_np(ob.rgb), bd_rgb, atol=1e-12)
    np.testing.assert_allclose(to_np(ob.depth), 5.0, atol=1e-9)
    # 반투명 backdrop: 뒤 체적이 비친다
    bd2 = dict(rgb=bd_rgb * 0.5, alpha=bd_a * 0.5, depth=bd_d)
    a_b = to_np(_r(behind).alpha)
    ob2 = _r(behind, backdrop=bd2, bg=(0, 0, 1))
    exp2 = 0.5 * bd_rgb + 0.5 * (a_b[..., None] * np.array([1.0, 0, 0]) + (1 - a_b[..., None]) * np.array([0, 0, 1.0]))
    np.testing.assert_allclose(to_np(ob2.rgb), exp2, atol=1e-9)
    # tensor backdrop 도 받는다
    bdt = {k: torch.tensor(v) for k, v in bd.items()}
    np.testing.assert_allclose(to_np(_r(front, backdrop=bdt, bg=(0, 0, 1)).rgb), exp, atol=1e-9)
