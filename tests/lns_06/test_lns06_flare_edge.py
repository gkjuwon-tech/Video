"""LNS-06 수평 플레어(streak_layer)와 주변부 번짐(edge_softness): 계약서 수식 오라클·기하·검증."""
import numpy as np
import pytest

from buc.lens import prescriptions as P
from lns06_helpers import edge_ref, hdr_scene, hf_energy, impulse, noise_img, streak_ref

TINT = (0.55, 0.75, 1.0)


# ----------------------------------------------------------------------------------------------- streak_layer
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_streak_matches_contract_formula(dtype):
    x = hdr_scene(48, 64).astype(dtype)
    keep = x.copy()
    out = P.streak_layer(x, gain=0.7, length_frac=0.2, tint=(0.5, 0.8, 1.0), threshold=0.8)
    ref = streak_ref(x, 0.7, 0.2, (0.5, 0.8, 1.0), 0.8)
    assert out.shape == x.shape and out.dtype == dtype
    assert np.array_equal(x, keep)
    assert out.max() > 0.05
    assert np.abs(out - ref).max() <= 2e-5 * max(1.0, ref.max())
    assert out.min() >= 0.0
    # 기본 인자 = 계약서 기본값 (length_frac 0.35, tint DEFAULT_STREAK_TINT, threshold 0.8)
    d = P.streak_layer(x, gain=1.0)
    assert np.abs(d - streak_ref(x, 1.0, 0.35, TINT, 0.8)).max() <= 2e-5 * max(1.0, d.max())


def test_streak_point_source_geometry():
    h, w = 21, 201
    x0, y0 = 100, 10
    x = impulse(h, w, x0, y0, 10.8, base=0.0)           # E = 10 (모든 채널)
    out = P.streak_layer(x, gain=1.0, length_frac=0.05, tint=TINT, threshold=0.8).astype(np.float64)
    row = out[:, :, 1]
    # 세로 3탭 [.25,.5,.25] 밖은 0
    assert row[:y0 - 1].sum() == 0.0 and row[y0 + 2:].sum() == 0.0
    assert row[y0 - 1].sum() == pytest.approx(row[y0 + 1].sum(), rel=1e-5)
    assert row[y0].sum() == pytest.approx(2.0 * row[y0 + 1].sum(), rel=1e-5)
    # 채널 비 = tint
    for c in range(3):
        assert out[:, :, c].sum() == pytest.approx(out[:, :, 1].sum() * TINT[c] / TINT[1], rel=1e-5)
    # 가로 대칭, 중심에서 멀어질수록 감소
    r = row[y0]
    assert np.allclose(r, r[::-1], rtol=1e-5, atol=1e-9)
    assert (np.diff(r[x0:]) <= 1e-12).all()
    # 광속: 가운데 광원이면 지수 꼬리 e^{-100/10} 만 잃는다
    assert out[:, :, 1].sum() == pytest.approx(TINT[1] * 10.0, rel=0.01)
    assert out[:, :, 1].sum() <= TINT[1] * 10.0 * (1 + 1e-6)
    # 가로 표준편차 = sqrt(2)·L (L = 0.05·201)
    xs = np.arange(w) - x0
    sd = np.sqrt((r * xs ** 2).sum() / r.sum())
    assert sd == pytest.approx(np.sqrt(2.0) * 0.05 * w, rel=0.05)


def test_streak_scaling_and_zero_cases():
    x = hdr_scene(32, 48)
    base = P.streak_layer(x, gain=1.0, length_frac=0.3)
    assert np.allclose(P.streak_layer(x, gain=0.25, length_frac=0.3), 0.25 * base, rtol=1e-5, atol=1e-8)
    assert not P.streak_layer(x, gain=0.0).any()
    assert not P.streak_layer(np.full((16, 24, 3), 0.7, np.float32), gain=1.0).any()    # 임계 아래는 층이 없다
    assert not P.streak_layer(x, gain=1.0, threshold=1e6).any()
    # 임계가 높을수록 광속이 줄어든다
    f = [P.streak_layer(x, gain=1.0, threshold=t).sum() for t in (0.2, 0.8, 3.0)]
    assert f[0] > f[1] > f[2] > 0.0


def test_streak_validation():
    x = impulse(16, 24, 12, 8, 5.0, base=0.1)
    for kw in (dict(gain=-0.1), dict(gain=float("nan")), dict(gain=1.0, length_frac=0.0),
               dict(gain=1.0, length_frac=1.5), dict(gain=1.0, tint=(1, 1)), dict(gain=1.0, tint=(1, -1, 1)),
               dict(gain=1.0, threshold=-0.1), dict(gain=True)):
        with pytest.raises(ValueError):
            P.streak_layer(x, **kw)
    with pytest.raises(TypeError):
        P.streak_layer(x, 1.0)                                  # gain 은 키워드 전용
    with pytest.raises(TypeError):
        P.streak_layer((x * 255).astype(np.uint8), gain=1.0)
    with pytest.raises(ValueError):
        P.streak_layer(x[..., 0], gain=1.0)
    bad = x.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        P.streak_layer(bad, gain=1.0)


# ----------------------------------------------------------------------------------------------- edge_softness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("amount", [0.4, 1.0])
def test_edge_softness_matches_contract_formula(dtype, amount):
    x = noise_img(48, 80, seed=2, dtype=dtype)
    keep = x.copy()
    out = P.edge_softness(x, amount)
    assert out.shape == x.shape and out.dtype == dtype
    assert np.array_equal(x, keep)
    assert np.abs(out - edge_ref(x, amount)).max() < 3e-5
    assert np.abs(out - x).max() > 0.02                       # 실제로 번진다


def test_edge_softness_center_untouched_and_uniform_invariant():
    x = noise_img(57, 81, seed=1)
    out = P.edge_softness(x, 1.0)
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    R = np.hypot(w - 1, h - 1) / 2.0
    r = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0) / R
    inner = r <= P.EDGE_R_INNER
    assert inner.sum() > 200
    assert np.array_equal(out[inner], x[inner])                # r <= EDGE_R_INNER 는 비트 동일
    assert not np.array_equal(out[r > 0.8], x[r > 0.8])
    flat = np.full((40, 60, 3), 0.37, np.float32)
    assert np.abs(P.edge_softness(flat, 1.0) - flat).max() < 1e-6
    z = P.edge_softness(x, 0.0)
    assert np.array_equal(z, x) and z is not x                 # amount=0 은 복사


def test_edge_softness_radial_falloff_and_amount_monotone():
    x = noise_img(96, 160, seed=4)
    out = P.edge_softness(x, 1.0)
    e_in = hf_energy(x, 0.0, 0.35)
    ratios = [hf_energy(out, a, b) / hf_energy(x, a, b) for a, b in ((0.0, 0.35), (0.45, 0.6), (0.85, 1.01))]
    assert ratios[0] == pytest.approx(1.0, abs=1e-3)
    assert ratios[1] < 0.95 and ratios[2] < 0.5
    assert ratios[0] > ratios[1] > ratios[2]
    assert e_in > 0
    # 양이 클수록 가장자리 고주파가 줄어든다
    es = [hf_energy(P.edge_softness(x, a), 0.85, 1.01) for a in (0.0, 0.25, 0.5, 1.0)]
    assert es[0] > es[1] > es[2] > es[3]
    # 대략의 밝기 보존 (nearest 경계라 정확 보존은 아니다)
    assert float(out.mean()) == pytest.approx(float(x.mean()), rel=0.01)


def test_edge_softness_validation():
    x = noise_img(16, 24)
    for a in (-0.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            P.edge_softness(x, a)
    with pytest.raises(TypeError):
        P.edge_softness((x * 255).astype(np.uint8), 0.5)
    with pytest.raises(ValueError):
        P.edge_softness(x[..., :2], 0.5)
    with pytest.raises(ValueError):
        P.edge_softness(x[0], 0.5)
