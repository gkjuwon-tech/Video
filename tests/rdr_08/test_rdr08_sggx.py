"""RDR-08 SGGX 수학: 섬유·표면 생성자, 투영 면적, D 정규화 항등식, 위상 함수 정규화·상반성·스케일 불변,
numpy/torch 타입 보존, scatter_colors."""
import math

import numpy as np
import pytest
import torch

from buc.render.vol import (scatter_colors, sggx_D, sggx_fiber, sggx_phase_specular, sggx_sigma, sggx_surface)
from rdr08_helpers import fiber_S, fib_sphere, make_vol, rot_axis, surface_S, to_np

SPHERE = fib_sphere(400_000)
DOMEGA = 4 * math.pi / SPHERE.shape[0]


def _rand_dirs(n, seed):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_fiber_and_surface_constructors():
    t = np.array([1.0, 2.0, -0.5])
    S = to_np(sggx_fiber(t, 0.2))
    np.testing.assert_allclose(S, fiber_S(t, 0.2), atol=1e-12)
    n = np.array([0.0, -3.0, 4.0])
    S2 = to_np(sggx_surface(n, 0.3))
    np.testing.assert_allclose(S2, surface_S(n, 0.3), atol=1e-12)
    ev = np.linalg.eigvalsh(S)
    np.testing.assert_allclose(sorted(ev), [0.04, 1.0, 1.0], atol=1e-12)
    tn = t / np.linalg.norm(t)
    assert abs(float(to_np(sggx_sigma(S, t))) - 0.2) < 1e-12           # σ(t) = r
    perp = np.cross(tn, [0, 0, 1.0])
    assert abs(float(to_np(sggx_sigma(S, perp))) - 1.0) < 1e-12        # σ(⊥) = 1
    nn = n / np.linalg.norm(n)
    assert abs(float(to_np(sggx_sigma(S2, nn))) - 1.0) < 1e-12
    # 배치 브로드캐스트
    T = _rand_dirs(5, 1)
    Sb = to_np(sggx_fiber(T, 0.5))
    assert Sb.shape == (5, 3, 3)
    for i in range(5):
        np.testing.assert_allclose(Sb[i], fiber_S(T[i], 0.5), atol=1e-12)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
def test_roughness_validation(bad):
    with pytest.raises(ValueError):
        sggx_fiber(np.array([1.0, 0, 0]), bad)
    with pytest.raises(ValueError):
        sggx_surface(np.array([1.0, 0, 0]), bad)


def test_isotropic_values():
    S = np.eye(3)
    w = _rand_dirs(7, 2)
    np.testing.assert_allclose(to_np(sggx_sigma(S, w)), 1.0, atol=1e-12)
    np.testing.assert_allclose(to_np(sggx_D(S, w)), 1 / math.pi, atol=1e-12)
    wo = _rand_dirs(7, 3)
    np.testing.assert_allclose(to_np(sggx_phase_specular(S, w, wo)), 1 / (4 * math.pi), atol=1e-12)


def test_D_closed_form_and_unnormalized_dirs():
    S = rot_axis([1, 2, 3], 40) @ np.diag([1.0, 0.3, 0.05]) @ rot_axis([1, 2, 3], 40).T
    m = _rand_dirs(9, 4)
    Si = np.linalg.inv(S)
    exp = 1 / (math.pi * math.sqrt(np.linalg.det(S)) * np.einsum("ni,ij,nj->n", m, Si, m) ** 2)
    np.testing.assert_allclose(to_np(sggx_D(S, m * 3.7)), exp, rtol=1e-10)
    np.testing.assert_allclose(to_np(sggx_sigma(S, m * 0.2)), np.sqrt(np.einsum("ni,ij,nj->n", m, S, m)),
                               rtol=1e-10)


@pytest.mark.parametrize("kind,r", [("fiber", 0.5), ("surface", 0.4), ("fiber", 0.35), ("general", None)])
def test_projected_area_identity(kind, r):
    """∫ max(0, w·m) D(m) dm = σ(w) — SGGX의 정의 항등식(구면 구적)."""
    if kind == "fiber":
        S = fiber_S([0.3, 0.8, -0.5], r)
    elif kind == "surface":
        S = surface_S([0.2, -0.4, 0.9], r)
    else:
        R = rot_axis([0.4, -1, 0.2], 70)
        S = R @ np.diag([1.0, 0.5, 0.3]) @ R.T
    D = to_np(sggx_D(S, SPHERE))
    for w in _rand_dirs(4, 5):
        lhs = float(np.sum(np.maximum(0.0, SPHERE @ w) * D) * DOMEGA)
        sig = float(to_np(sggx_sigma(S, w)))
        assert abs(lhs - sig) / sig < 5e-3, (kind, lhs, sig)


@pytest.mark.parametrize("S", [fiber_S([1, 0.2, 0.1], 0.5), surface_S([0, 0.3, 1], 0.45), np.eye(3)])
def test_phase_normalization_and_reciprocity(S):
    for wi in _rand_dirs(3, 6):
        f = to_np(sggx_phase_specular(S, np.broadcast_to(wi, SPHERE.shape), SPHERE))
        assert np.isfinite(f).all() and (f >= 0).all()
        total = float(f.sum() * DOMEGA)
        assert abs(total - 1.0) < 1e-2, total
    wi, wo = _rand_dirs(6, 7), _rand_dirs(6, 8)
    lhs = to_np(sggx_sigma(S, wi)) * to_np(sggx_phase_specular(S, wi, wo))
    rhs = to_np(sggx_sigma(S, wo)) * to_np(sggx_phase_specular(S, wo, wi))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)
    np.testing.assert_allclose(to_np(sggx_phase_specular(S * 7.5, wi, wo)), to_np(sggx_phase_specular(S, wi, wo)),
                               rtol=1e-10)


def test_phase_degenerate_is_zero_not_nan():
    S = fiber_S([0, 1, 0], 0.3)
    wi = np.array([0.0, 0.0, 1.0])
    f = to_np(sggx_phase_specular(S, wi, -wi))
    assert np.isfinite(f) and float(f) == 0.0


def test_type_preservation_and_autograd():
    S_np = fiber_S([1, 1, 0], 0.3)
    w_np = np.array([0.2, 0.3, 0.9])
    out = sggx_sigma(S_np, w_np)
    assert isinstance(out, np.ndarray) or np.isscalar(out)
    S_t = torch.tensor(S_np, dtype=torch.float64, requires_grad=True)
    out_t = sggx_sigma(S_t, torch.tensor(w_np, dtype=torch.float64))
    assert isinstance(out_t, torch.Tensor) and out_t.dtype == torch.float64
    out_t.sum().backward()
    assert S_t.grad is not None and torch.isfinite(S_t.grad).all()
    St32 = torch.tensor(S_np, dtype=torch.float32)
    assert sggx_D(St32, torch.tensor(w_np, dtype=torch.float32)).dtype == torch.float32
    assert isinstance(sggx_fiber(torch.tensor([1.0, 0.0, 0.0]), 0.2), torch.Tensor)


def test_scatter_colors_formula():
    mu = np.array([[0.0, 0.0, 3.0], [0.5, -0.2, 4.0], [-0.3, 0.4, 2.5]])
    S = np.stack([fiber_S([1, 0, 0], 0.3), surface_S([0, 0, -1], 0.5), np.eye(3)])
    alb = np.array([[0.9, 0.8, 0.7], [0.2, 0.5, 0.9], [0.6, 0.6, 0.6]])
    vol = make_vol(mu, np.repeat(np.eye(3)[None], 3, 0), np.full((3, 3), 0.05), [1.0, 2.0, 3.0], alb, S)
    light_dir = np.array([0.3, -0.2, -1.0])
    light_rgb = np.array([2.0, 1.5, 1.0])
    cam = np.array([0.1, 0.0, 0.0])
    out = to_np(scatter_colors(vol, light_dir=light_dir, light_rgb=light_rgb, cam_center=cam))
    assert out.shape == (3, 3)
    wi = -light_dir / np.linalg.norm(light_dir)
    for i in range(3):
        wo = (cam - mu[i]) / np.linalg.norm(cam - mu[i])
        h = (wi + wo) / np.linalg.norm(wi + wo)
        Si = np.linalg.inv(S[i])
        D = 1 / (math.pi * math.sqrt(np.linalg.det(S[i])) * (h @ Si @ h) ** 2)
        f = D / (4 * math.sqrt(wi @ S[i] @ wi))
        np.testing.assert_allclose(out[i], alb[i] * light_rgb * f, rtol=1e-8)
    out2 = to_np(scatter_colors(vol, light_dir=light_dir, light_rgb=light_rgb, cam_center=cam,
                                albedo=torch.ones(3, 3, dtype=torch.float64)))
    np.testing.assert_allclose(out2 * alb, out, rtol=1e-8)
