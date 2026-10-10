"""RDR-08 변환: matte_to_vol(α→밀도) 구성·원천 뷰 재현·램프·색 합성·시차·섬유 비, alpha_to_tau,
strands_to_vol 구성·커버리지 질량 동치."""
import math

import numpy as np
import pytest
import torch

from buc.core import geom
from buc.render.vol import (VolOpts, alpha_to_tau, matte_to_vol, render_vol, strands_to_vol, tau_to_alpha)
from rdr08_helpers import K_of, at_pixel, fiber_S, make_strands, pose, rot_axis, tau_of, to_np

W, H = 64, 48
FX = 120.0
K = K_of(FX, W, H)          # cx = 31.5, cy = 23.5
I4 = np.eye(4)
F64 = torch.float64


def _r(vol=None, strands=None, c2w=I4, K=K, w=W, h=H, **kw):
    return render_vol(vol=vol, strands=strands, c2w=c2w, K=K, width=w, height=h, opts=VolOpts(**kw))


def test_alpha_tau_roundtrip():
    a = np.array([0.0, 0.1, 0.5, 0.9, 0.999, 1.0, -0.2])
    t = alpha_to_tau(a)
    assert isinstance(t, np.ndarray)
    np.testing.assert_allclose(t[:5], -np.log(1 - a[:5]), rtol=1e-12)
    assert t[5] == pytest.approx(-math.log(1 - 0.999)) and t[6] == 0.0
    np.testing.assert_allclose(tau_to_alpha(t[:5]), a[:5], atol=1e-12)
    tt = alpha_to_tau(torch.tensor(a[:4]))
    assert isinstance(tt, torch.Tensor)
    np.testing.assert_allclose(tt.numpy(), t[:4], rtol=1e-12)


def _maps(alpha_val=0.6, z=2.0, rect=(12, 52, 10, 38), F=(0.8, 0.3, 0.1)):
    a = np.zeros((H, W))
    u0, u1, v0, v1 = rect
    a[v0:v1, u0:u1] = alpha_val
    rgb = np.zeros((H, W, 3))
    rgb[:] = F
    d = np.full((H, W), z)
    return a, rgb, d


def test_construction_values():
    c2w = pose(rot_axis([0.2, 1, 0.1], 20), (0.3, -0.1, 0.5))
    a = np.zeros((H, W))
    a[10, 20], a[10, 22], a[12, 20], a[11, 21], a[13, 30] = 0.5, 0.002, 0.9, 0.7, 0.3   # (11,21)·(13,30) stride 밖, 0.002 < 1/255
    rgb = np.random.default_rng(0).uniform(0, 1, (H, W, 3))
    d = np.full((H, W), 2.5)
    d[12, 20] = np.nan
    vol = matte_to_vol(a, rgb, d, K=K, c2w=c2w, stride=2, footprint=1.5, dtype=F64)
    assert vol.n == 1 and vol.means.dtype == F64
    Rw = c2w[:3, :3]
    mu = geom.unproject_world(K, Rw.T, -Rw.T @ c2w[:3, 3], np.array([20.0, 10.0]), np.array(2.5))
    np.testing.assert_allclose(to_np(vol.means)[0], mu, atol=1e-9)
    np.testing.assert_allclose(to_np(vol.colors)[0], rgb[10, 20], atol=1e-12)
    sx = 1.5 * 2 * 2.5 / FX
    np.testing.assert_allclose(to_np(vol.covariances())[0], Rw @ np.diag([sx, sx, 0.25 * sx]) ** 2 @ Rw.T, atol=1e-12)
    np.testing.assert_allclose(to_np(vol.sggx)[0], np.eye(3), atol=1e-12)
    # 밀도: 원천 광선 피크 τ = alpha_to_tau(α)/(2π f²)
    Sig = Rw @ np.diag([sx, sx, 0.25 * sx]) ** 2 @ Rw.T
    w = (mu - c2w[:3, 3]) / np.linalg.norm(mu - c2w[:3, 3])
    aa = w @ np.linalg.inv(Sig) @ w
    rho = (-math.log(0.5)) / (2 * math.pi * 1.5 ** 2) / math.sqrt(2 * math.pi / aa)
    assert abs(float(to_np(vol.densities())[0]) - rho) < 1e-9 * rho
    v2 = matte_to_vol(a, rgb, np.full((H, W), 2.5), K=K, c2w=c2w, stride=1, thickness=0.01, dtype=F64)
    assert v2.n == 4                                  # 0.002 픽셀만 빠짐
    np.testing.assert_allclose(np.sqrt(np.linalg.eigvalsh(to_np(v2.covariances())[0]))[0], 0.01, atol=1e-12)


def test_single_pixel_peak_and_fiber_ratio():
    Ks = K_of(200.0, 33, 33)
    a = np.zeros((33, 33))
    a[16, 16] = 0.8
    rgb = np.ones((33, 33, 3))
    d = np.full((33, 33), 2.0)
    iso = matte_to_vol(a, rgb, d, K=Ks, c2w=I4, dtype=F64)
    fib = matte_to_vol(a, rgb, d, K=Ks, c2w=I4, tangent=np.array([0.0, 1.0, 0.0]), roughness=0.2, dtype=F64)
    fib2 = matte_to_vol(a, rgb, d, K=Ks, c2w=I4, tangent=np.broadcast_to([0.0, 1.0, 0.0], (33, 33, 3)).copy(),
                        roughness=0.2, dtype=F64)
    np.testing.assert_allclose(to_np(fib.sggx), to_np(fib2.sggx), atol=1e-12)
    np.testing.assert_allclose(to_np(fib.sggx)[0], fiber_S([0, 1, 0], 0.2), atol=1e-12)
    peak = -math.log(0.2) / (2 * math.pi)
    for v in (iso, fib):
        t0 = tau_of(to_np(_r(v, K=Ks, w=33, h=33).alpha)[16, 16])
        assert abs(t0 - peak) < 1e-6 * peak
    Rs = np.stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], axis=1)     # 카메라 z = 세계 +y
    side = pose(Rs, (0.0, -3.0, 2.0))
    ti = tau_of(to_np(_r(iso, c2w=side, K=Ks, w=33, h=33).alpha)[16, 16])
    tf = tau_of(to_np(_r(fib, c2w=side, K=Ks, w=33, h=33).alpha)[16, 16])
    assert ti > 0 and abs(tf / ti - 0.2) < 2e-4


def test_uniform_matte_reproduction_and_mass():
    a, rgb, d = _maps()
    vol = matte_to_vol(a, rgb, d, K=K, c2w=I4, dtype=F64)
    assert vol.n == 40 * 28
    bg = np.array([0.1, 0.2, 0.9])
    out = _r(vol, bg=tuple(bg))
    A = to_np(out.alpha)
    RGB = to_np(out.rgb)
    inner = (slice(16, 32), slice(18, 46))
    assert np.abs(A[inner] - 0.6).max() < 2e-3
    exp = 0.6 * np.array([0.8, 0.3, 0.1]) + 0.4 * bg
    assert np.abs(RGB[inner] - exp).max() < 3e-3
    # 색이 하나면 모든 화소에서 정확히 αF + (1−α)B (합성 규칙)
    np.testing.assert_allclose(RGB, A[..., None] * np.array([0.8, 0.3, 0.1]) + (1 - A[..., None]) * bg, atol=1e-9)
    m_got = tau_of(A).sum()
    m_exp = alpha_to_tau(a).sum()
    assert abs(m_got - m_exp) < 2e-2 * m_exp
    np.testing.assert_allclose(to_np(out.depth)[inner], 2.0, rtol=1e-3)


def test_linear_ramp_preserved():
    a = np.zeros((H, W))
    u = np.arange(W)
    tau = 0.2 + 0.015 * (u - 12)
    a[10:38, 12:52] = (1 - np.exp(-tau))[None, 12:52]
    rgb = np.ones((H, W, 3))
    d = np.full((H, W), 2.0)
    out = _r(matte_to_vol(a, rgb, d, K=K, c2w=I4, dtype=F64))
    got = tau_of(to_np(out.alpha))[16:32, 18:46]
    exp = np.broadcast_to(tau[None, 18:46], got.shape)
    assert np.abs(got - exp).max() < 3e-3


def test_parallax_like_3d_layer():
    """카메라를 b=0.05 m 옮기면 z=2 m 층의 α 패턴이 f·b/z = 3 px 밀린다(고무막이 아니라 3D 층)."""
    a = np.zeros((H, W))
    rng = np.random.default_rng(3)
    a[12:36, 14:50] = rng.uniform(0.2, 0.8, (24, 36))
    a = np.round(a * 4) / 4                                 # 계단 무늬
    d = np.full((H, W), 2.0)
    vol = matte_to_vol(a, np.ones((H, W, 3)), d, K=K, c2w=I4, dtype=F64)
    src = to_np(_r(vol).alpha)
    moved = to_np(_r(vol, c2w=pose(None, (0.05, 0.0, 0.0))).alpha)
    np.testing.assert_allclose(moved[14:34, 13:44], src[14:34, 16:47], atol=3e-3)
    assert np.abs(moved[14:34, 16:47] - src[14:34, 16:47]).max() > 0.05


def test_matte_validation():
    a, rgb, d = _maps()
    with pytest.raises(ValueError):
        matte_to_vol(a[:10], rgb, d, K=K, c2w=I4)
    with pytest.raises(ValueError):
        matte_to_vol(a, rgb[..., :2], d, K=K, c2w=I4)
    with pytest.raises(ValueError):
        matte_to_vol(a, rgb, d, K=K, c2w=I4, stride=0)
    with pytest.raises(ValueError):
        matte_to_vol(a, rgb, d, K=K, c2w=I4, footprint=0.0)
    empty = matte_to_vol(np.zeros((H, W)), rgb, d, K=K, c2w=I4)
    assert empty.n == 0
    out = _r(empty, bg=(0.5, 0.5, 0.5))
    np.testing.assert_allclose(to_np(out.rgb), 0.5, atol=1e-7)
    assert np.isinf(to_np(out.depth)).all()


# ---------------------------------------------------------------------------------------------------------------
Ks = K_of(500.0, 64, 32)      # cx = 31.5, cy = 15.5


def test_strands_to_vol_structure():
    A, B, C = np.array([0.0, 0.0, 2.0]), np.array([0.1, 0.02, 2.1]), np.array([0.15, 0.1, 2.0])
    st = make_strands(np.stack([[A, B, B, C], [C, A, B, C]]), [[0.002, 0.004, 0.004, 0.006], [0.003] * 4],
                      [0.6, 0.9], [[1, 0, 0], [0, 1, 0]])
    vol = strands_to_vol(st, roughness=0.15)
    assert vol.n == 5                                  # 길이 0 선분 하나 제외
    segs = [(A, B, 0.002, 0.004, 0.6, 0), (B, C, 0.004, 0.006, 0.6, 0), (C, A, 0.003, 0.003, 0.9, 1),
            (A, B, 0.003, 0.003, 0.9, 1), (B, C, 0.003, 0.003, 0.9, 1)]
    cov = to_np(vol.covariances())
    for i, (p, q, ra, rb, o, sid) in enumerate(segs):
        L = np.linalg.norm(q - p)
        t = (q - p) / L
        rbar = 0.5 * (ra + rb)
        st_, sp = 0.5 * L, rbar
        np.testing.assert_allclose(to_np(vol.means)[i], 0.5 * (p + q), atol=1e-12)
        np.testing.assert_allclose(cov[i], st_ ** 2 * np.outer(t, t) + sp ** 2 * (np.eye(3) - np.outer(t, t)),
                                   atol=1e-12)
        np.testing.assert_allclose(to_np(vol.sggx)[i], fiber_S(t, 0.15), atol=1e-12)
        rho = 2 * o * rbar * L / ((2 * math.pi) ** 1.5 * st_ * sp ** 2)
        assert abs(float(to_np(vol.densities())[i]) - rho) < 1e-9 * rho
        np.testing.assert_allclose(to_np(vol.colors)[i], [1, 0, 0] if sid == 0 else [0, 1, 0])
        R = to_np(vol.rotations())[i]
        assert abs(np.linalg.det(R) - 1) < 1e-9
    v2 = strands_to_vol(st, min_std=0.05)
    ev = np.sqrt(np.linalg.eigvalsh(to_np(v2.covariances())))       # 오름차순
    np.testing.assert_allclose(ev[:, :2], 0.05, atol=1e-12)
    np.testing.assert_allclose(ev[:, 2], [max(0.5 * np.linalg.norm(q - p), 0.05) for p, q, *_ in segs], atol=1e-12)


def test_strands_to_vol_coverage_matches_strand():
    """멀리서 체적 LOD로 바꿔도 털의 투영 커버리지(열 합 = o·2w)가 같다."""
    z = 2.0
    r = 1.0 * z / 500.0            # w = 1 px
    pts = np.stack([at_pixel(Ks, u, 15.0, z) for u in np.linspace(4.0, 60.0, 17)])
    st = make_strands(pts[None], np.full((1, 17), r), [0.2], [[1, 1, 1]])
    a_s = to_np(render_vol(strands=st, c2w=I4, K=Ks, width=64, height=32).alpha)
    vol = strands_to_vol(st)
    a_v = to_np(render_vol(vol=vol, c2w=I4, K=Ks, width=64, height=32).alpha)
    cs = a_s[:, 16:48].sum(0)
    cv = tau_of(a_v)[:, 16:48].sum(0)
    np.testing.assert_allclose(cs, 0.4, rtol=3e-2)
    np.testing.assert_allclose(cv, 0.4, rtol=4e-2)
