"""RDR-08 가닥(STRAND) 래스터: 프로파일 오라클, 얇은 가닥 커버리지 = o·2w, 두꺼운 가닥 피크 = o,
관절 이중 계산 없음, 깊이 보간, 교차 가닥의 정렬, 컬링·퇴화 선분."""
import numpy as np
import torch

from buc.render.vol import VolOpts, render_vol
from rdr08_helpers import K_of, at_pixel, make_strands, strand_alpha_oracle, to_np

W, H = 64, 32
FX = 500.0
K = K_of(FX, W, H)        # cx = 31.5, cy = 15.5
I4 = np.eye(4)


def _r(strands, c2w=I4, **kw):
    return render_vol(strands=strands, c2w=c2w, K=K, width=W, height=H, opts=VolOpts(**kw))


def _hline(v0, u0, u1, z, n_pts=8):
    us = np.linspace(u0, u1, n_pts)
    return np.stack([at_pixel(K, u, v0, z) for u in us])


def test_profile_matches_oracle_thick():
    z = 2.0
    w_px = 6.0
    r = w_px * z / FX
    pts = _hline(15.0, 4.0, 60.0, z)
    st = make_strands(pts[None], np.full((1, 8), r), [0.8], [[0.9, 0.7, 0.5]])
    a = to_np(_r(st).alpha)
    for v in range(H):
        d = abs(v - 15.0)
        exp = strand_alpha_oracle(0.8, d, w_px)
        exp = exp if exp >= 1 / 255 else 0.0
        assert abs(a[v, 30] - exp) < 1e-5, (v, a[v, 30], exp)
    assert abs(a[15, 30] - 0.8) < 1e-6
    assert abs(a[9, 30] - 0.4) < 1e-4            # d = w → o/2


def test_thin_strand_column_coverage():
    """서브픽셀 털: 열 합 = o·2w (커버리지 보존). 화소 중심/반 화소 두 위상."""
    z = 2.0
    for w_px in (0.25, 0.6, 1.0):
        sums = []
        for v0 in (15.0, 15.5):
            r = w_px * z / FX
            st = make_strands(_hline(v0, 4.0, 60.0, z)[None], np.full((1, 8), r), [0.5], [[1, 1, 1]])
            a = to_np(_r(st).alpha)
            cols = a[:, 15:49].sum(axis=0)
            # 오라클(표본화 + min_alpha 절단)과 일치, 이상값 o·2w 와는 4% 안
            d = np.abs(np.arange(H) - v0)
            orc = strand_alpha_oracle(0.5, d, w_px)
            orc = np.where(orc >= 1 / 255, orc, 0.0).sum()
            np.testing.assert_allclose(cols, orc, rtol=1e-4)
            np.testing.assert_allclose(cols, 0.5 * 2 * w_px, rtol=4e-2)
            sums.append(cols.mean())
        assert abs(np.mean(sums) - 0.5 * 2 * w_px) < 2e-2 * 0.5 * 2 * w_px


def test_radius_scales_with_depth_and_taper():
    """w = r·f/z: 같은 r을 두 배 멀리 두면 열 합이 절반. 반지름 선형 테이퍼."""
    r = 0.004
    st2 = make_strands(_hline(15.0, 4.0, 60.0, 2.0)[None], np.full((1, 8), r), [0.3], [[1, 1, 1]])
    st4 = make_strands(_hline(15.0, 4.0, 60.0, 4.0)[None], np.full((1, 8), r), [0.3], [[1, 1, 1]])
    c2 = to_np(_r(st2).alpha)[:, 20:44].sum(0)
    c4 = to_np(_r(st4).alpha)[:, 20:44].sum(0)
    np.testing.assert_allclose(c4 / c2, 0.5, rtol=3e-2)
    pts = np.stack([at_pixel(K, 4.0, 15.0, 2.0), at_pixel(K, 60.0, 15.0, 2.0)])
    ra, rb = 0.001, 0.005
    st = make_strands(pts[None], [[ra, rb]], [0.3], [[1, 1, 1]])
    cols = to_np(_r(st).alpha).sum(0)
    for u in (12, 32, 52):
        t = (u - 4.0) / 56.0
        w = ((1 - t) * ra + t * rb) * FX / 2.0
        assert abs(cols[u] - 0.3 * 2 * w) < 3e-2 * 0.3 * 2 * w, (u, cols[u])


def test_no_double_counting_at_joints():
    z = 2.0
    w_px = 3.0
    r = w_px * z / FX
    pts = np.stack([at_pixel(K, u, v, z) for u, v in [(6, 4), (20, 26), (34, 6), (48, 26), (60, 8)]])
    st = make_strands(pts[None], np.full((1, 5), r), [0.7], [[1, 1, 1]])
    a = to_np(_r(st).alpha)
    assert a.max() <= 0.7 + 1e-9
    assert abs(a[26, 20] - 0.7) < 1e-4 and abs(a[6, 34] - 0.7) < 1e-4


def test_depth_interpolation_along_strand():
    za, zb = 2.0, 4.0
    pa, pb = at_pixel(K, 8.0, 15.0, za), at_pixel(K, 56.0, 15.0, zb)
    st = make_strands(np.stack([pa, pb])[None], [[0.01, 0.01]], [0.9], [[1, 1, 1]])
    out = _r(st)
    d = to_np(out.depth)
    for u in (10, 20, 32, 50):
        t = (u - 8.0) / 48.0
        assert abs(d[15, u] - ((1 - t) * za + t * zb)) < 1e-6, (u, d[15, u])


def test_crossing_strands_sorted_by_depth():
    red, green, bg = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0.1, 0.1, 0.4])
    for z_h, z_v in ((2.0, 3.0), (3.0, 2.0)):
        h = np.stack([at_pixel(K, 4.0, 15.0, z_h), at_pixel(K, 60.0, 15.0, z_h)])
        v = np.stack([at_pixel(K, 32.0, 1.0, z_v), at_pixel(K, 32.0, 30.0, z_v)])
        rh, rv = 2.0 * z_h / FX, 2.0 * z_v / FX
        both = make_strands(np.stack([h, v]), [[rh, rh], [rv, rv]], [0.6, 0.8], [red, green])
        a_h = strand_alpha_oracle(0.6, 0.0, 2.0)
        a_v = strand_alpha_oracle(0.8, 0.0, 2.0)
        out = _r(both, bg=tuple(bg))
        if z_h < z_v:
            exp = a_h * red + (1 - a_h) * a_v * green + (1 - a_h) * (1 - a_v) * bg
            exp_d = (a_h * z_h + (1 - a_h) * a_v * z_v) / (a_h + (1 - a_h) * a_v)
        else:
            exp = a_v * green + (1 - a_v) * a_h * red + (1 - a_h) * (1 - a_v) * bg
            exp_d = (a_v * z_v + (1 - a_v) * a_h * z_h) / (a_v + (1 - a_v) * a_h)
        np.testing.assert_allclose(to_np(out.rgb)[15, 32], exp, atol=1e-6)
        assert abs(to_np(out.depth)[15, 32] - exp_d) < 1e-6
        assert abs(to_np(out.alpha)[15, 32] - (1 - (1 - a_h) * (1 - a_v))) < 1e-9


def test_culling_and_degenerate_segments():
    behind = np.stack([at_pixel(K, 10.0, 15.0, 2.0), np.array([0.0, 0.0, -1.0]), at_pixel(K, 50.0, 15.0, 2.0)])
    st = make_strands(behind[None], [[0.004] * 3], [0.9], [[1, 1, 1]])
    with np.errstate(all="raise"):
        out = _r(st)
    for k in ("rgb", "alpha"):
        assert np.isfinite(to_np(getattr(out, k))).all()
    assert to_np(out.alpha).max() == 0.0          # 두 선분 모두 한 끝이 near 앞
    p = at_pixel(K, 32.0, 15.0, 2.0)
    dot = make_strands(np.stack([p, p, p])[None], [[0.008] * 3], [0.5], [[1, 1, 1]])
    out = _r(dot)
    a = to_np(out.alpha)
    assert np.isfinite(a).all() and np.isfinite(to_np(out.rgb)).all()
    exp = strand_alpha_oracle(0.5, 0.0, 0.008 * FX / 2.0)
    assert abs(a[15, 32] - exp) < 1e-6
    np.testing.assert_allclose(a[15, 31:34], a[15, 33:30:-1], atol=1e-12)


def test_float32_output_dtype_and_min_alpha():
    st = make_strands(_hline(15.0, 4.0, 60.0, 2.0)[None], np.full((1, 8), 0.002), [0.5], [[1, 1, 1]],
                      dtype=torch.float32)
    out = _r(st)
    assert out.rgb.dtype == torch.float32 and out.alpha.dtype == torch.float32
    a = to_np(out.alpha)
    assert ((a == 0) | (a >= 1 / 255 - 1e-7)).all()
    out2 = _r(st, min_alpha=0.1)
    a2 = to_np(out2.alpha)
    assert ((a2 == 0) | (a2 >= 0.1 - 1e-6)).all() and (a2 > 0).sum() < (a > 0).sum()
