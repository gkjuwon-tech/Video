"""LNS-04 화면 고정 효과: 비네팅, 할레이션, Newson 그레인."""
import numpy as np
import pytest

from buc.lens import sensor_film as sf
from lns04_helpers import (GRAIN_RADIUS_FRAC, GRAIN_SIGMA_MAX, HALATION_GAIN, HALATION_SIGMA_FRAC, HALATION_THRESHOLD,
                           HALATION_TINT, VIGNETTE_MAX_DROP, autocorr, corr, halation_oracle, scene, vignette_ref)


def test_constants_match_contract():
    assert sf.GRAIN_SIGMA_MAX == GRAIN_SIGMA_MAX and sf.GRAIN_RADIUS_FRAC == GRAIN_RADIUS_FRAC
    assert sf.HALATION_THRESHOLD == HALATION_THRESHOLD and sf.HALATION_SIGMA_FRAC == HALATION_SIGMA_FRAC
    assert tuple(sf.HALATION_TINT) == HALATION_TINT and sf.HALATION_GAIN == HALATION_GAIN
    assert sf.VIGNETTE_MAX_DROP == VIGNETTE_MAX_DROP
    assert sf.GRAIN_RADIUS_MIN_PX == 0.75


# ----------------------------------------------------------------------------------------------- 비네팅
@pytest.mark.parametrize("hw", [(65, 97), (64, 96), (33, 130)])
def test_vignette_gain_formula_symmetry_monotone(hw):
    h, w = hw
    for s in (0.0, 0.3, 1.0):
        g = sf.vignette_gain(h, w, s)
        assert g.shape == (h, w) and g.dtype == np.float32
        assert np.max(np.abs(g - vignette_ref(h, w, s))) < 2e-6
        assert g.max() <= 1.0 and g.min() > 0.0
        assert np.allclose(g, g[::-1], atol=1e-7) and np.allclose(g, g[:, ::-1], atol=1e-7)
        corner = 1.0 - VIGNETTE_MAX_DROP * s
        assert g[0, 0] == pytest.approx(corner, abs=2e-6) and g[-1, -1] == pytest.approx(corner, abs=2e-6)
    assert np.all(sf.vignette_gain(h, w, 0.0) == 1.0)
    g = sf.vignette_gain(65, 97, 0.7)
    assert g[32, 48] == 1.0  # 홀수 크기의 정중앙은 1
    # 중심에서 멀수록 어둡다(행·열 방향 단조)
    assert np.all(np.diff(g[32, 48:]) < 0) and np.all(np.diff(g[32:, 48]) < 0)
    # 세기 단조: 더 센 비네팅은 모든 화소에서 같거나 어둡다
    assert np.all(sf.vignette_gain(65, 97, 0.9) <= sf.vignette_gain(65, 97, 0.4) + 1e-7)


def test_vignette_aspect_independent_corner_and_errors():
    for hw in ((40, 40), (40, 200), (2, 300)):
        assert sf.vignette_gain(*hw, 1.0)[0, 0] == pytest.approx(1.0 - VIGNETTE_MAX_DROP, abs=2e-6)
    assert sf.vignette_gain(1, 1, 1.0).shape == (1, 1) and sf.vignette_gain(1, 1, 1.0)[0, 0] == 1.0
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            sf.vignette_gain(8, 8, bad)
    for bad in ((0, 8), (8, 0), (-1, 8)):
        with pytest.raises(ValueError):
            sf.vignette_gain(*bad, 0.5)


def test_apply_vignette_multiplies_and_preserves_hdr():
    x = scene().astype(np.float32)
    keep = x.copy()
    y = sf.apply_vignette(x, 0.6)
    assert np.array_equal(x, keep) and y.dtype == np.float32 and y.shape == x.shape
    assert np.max(np.abs(y - x * sf.vignette_gain(*x.shape[:2], 0.6)[..., None])) < 2e-6
    assert y.max() > 1.0  # 클램프 없음
    assert np.array_equal(sf.apply_vignette(x, 0.0), x)
    with pytest.raises(ValueError):
        sf.apply_vignette(x, 2.0)
    with pytest.raises(ValueError):
        sf.apply_vignette(x[..., :2], 0.5)


# ----------------------------------------------------------------------------------------------- 할레이션
def _hot(w=200, h=120, peak=20.0, bg=0.05):
    img = np.full((h, w, 3), bg, np.float32)
    img[h // 2 - 1:h // 2 + 2, w // 2 - 1:w // 2 + 2] = peak
    return img


def test_halation_matches_gaussian_oracle():
    x = scene(h=96, w=200)
    for amt in (0.3, 1.0):
        lay = sf.halation_layer(x, amt).astype(np.float64)
        ref = halation_oracle(x, amt)
        assert lay.shape == x.shape and lay.min() >= 0.0
        assert np.max(np.abs(lay - ref)) < 2e-3 * ref.max()
        out = sf.apply_halation(x, amt)
        assert out.dtype == np.float32
        assert np.max(np.abs(out.astype(np.float64) - (x + ref))) < 2e-3 * ref.max() + 1e-6


def test_halation_zero_and_below_threshold_are_identity():
    x = scene()
    assert np.array_equal(sf.apply_halation(x, 0.0), x)
    assert np.all(sf.halation_layer(x, 0.0) == 0.0)
    dim = np.clip(x, 0, HALATION_THRESHOLD - 0.01).astype(np.float32)  # 휘도가 문턱 아래
    assert np.all(sf.halation_layer(dim, 1.0) == 0.0)
    assert np.array_equal(sf.apply_halation(dim, 1.0), dim)


def test_halation_energy_tint_and_profile():
    w = 200
    x = _hot(w=w)
    amt = 0.8
    lay = sf.halation_layer(x, amt).astype(np.float64)
    hl = 9 * (20.0 - HALATION_THRESHOLD)  # 3x3 광원 (회색 20 -> 휘도 20)
    for c, t in enumerate(HALATION_TINT):
        assert lay[..., c].sum() == pytest.approx(amt * HALATION_GAIN * t * hl, rel=0.01)  # 에너지 보존(영상 안)
    tot = lay.sum(axis=(0, 1))
    assert tot[0] > tot[1] > tot[2] > 0  # 붉은 기운 > 초록 > 파랑
    assert tot[1] / tot[0] == pytest.approx(HALATION_TINT[1] / HALATION_TINT[0], rel=0.01)
    assert tot[2] / tot[0] == pytest.approx(HALATION_TINT[2] / HALATION_TINT[0], rel=0.01)
    # 등방·방사 단조 감소·반에너지 반경 = sigma*sqrt(2 ln 2)
    r = lay[..., 0]
    cy, cx = x.shape[0] // 2, w // 2
    assert np.allclose(r[cy, cx + 1:cx + 40], r[cy, cx - 1:cx - 40:-1], rtol=1e-3, atol=1e-9)
    assert np.allclose(r[cy + 1:cy + 40, cx], r[cy - 1:cy - 40:-1, cx], rtol=1e-3, atol=1e-9)
    assert np.all(np.diff(r[cy, cx:cx + 45]) <= 1e-12)
    yy, xx = np.mgrid[0:x.shape[0], 0:w]
    rad = np.hypot(yy - cy, xx - cx)
    order = np.argsort(rad.ravel())
    cum = np.cumsum(r.ravel()[order]) / r.sum()
    half_r = rad.ravel()[order][np.searchsorted(cum, 0.5)]
    sigma = HALATION_SIGMA_FRAC * w
    assert half_r == pytest.approx(sigma * np.sqrt(2 * np.log(2)), rel=0.2)
    assert r[cy, cx] > 20 * r[cy, cx + int(3.5 * sigma)]  # 먼 곳은 거의 0


def test_halation_scales_with_width_resolution_independent():
    a = sf.halation_layer(_hot(w=200, h=100), 1.0).astype(np.float64)
    b = sf.halation_layer(_hot(w=400, h=200), 1.0).astype(np.float64)
    # sigma 가 폭에 비례 -> 같은 상대 위치의 프로파일 모양이 같다(에너지 정규화 후)
    pa = a[50, 100:130, 0] / a[50, 100, 0]
    pb = b[100, 200:260:2, 0] / b[100, 200, 0]
    assert np.max(np.abs(pa - pb)) < 0.08


def test_halation_errors():
    x = scene()
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            sf.apply_halation(x, bad)
        with pytest.raises(ValueError):
            sf.halation_layer(x, bad)
    with pytest.raises(TypeError):
        sf.apply_halation(x.astype(np.float16), 0.5)
    with pytest.raises(ValueError):
        sf.apply_halation(x[..., :2], 0.5)


# ----------------------------------------------------------------------------------------------- 그레인
def _flat(u, h=256, w=256, c=3):
    return np.full((h, w, c), u, np.float32)


def test_grain_strength_zero_and_dtype_contract():
    u = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
    keep = u.copy()
    z = sf.grain_field(u, strength=0.0, radius_px=3.0, seed=1)
    assert np.array_equal(z, u) and z is not u
    g = sf.grain_field(u, strength=0.5, radius_px=3.0, seed=1)
    assert np.array_equal(u, keep) and g.dtype == np.float32 and g.shape == u.shape
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert sf.grain_field(u[..., 0], strength=0.5, radius_px=2.0, seed=1).shape == (20, 30)  # (H,W) 도 허용
    assert sf.grain_field(u.astype(np.float64), strength=0.5, radius_px=2.0, seed=1).dtype == np.float32


@pytest.mark.parametrize("u0", [0.15, 0.3, 0.5, 0.75])
@pytest.mark.parametrize("strength", [0.4, 1.0])
def test_grain_mean_preserved_and_signal_dependent_std(u0, strength):
    g = sf.grain_field(_flat(u0), strength=strength, radius_px=3.0, seed=2).astype(np.float64)
    expect = strength * GRAIN_SIGMA_MAX * 2.0 * np.sqrt(u0 * (1.0 - u0))
    assert abs(g.mean() - u0) < 0.006  # 상관 입자의 평균 표준오차 ~0.001 의 6 배
    assert g.std() == pytest.approx(expect, rel=0.2)


def test_grain_std_profile_peaks_midtone_and_vanishes_at_extremes():
    s = {u: sf.grain_field(_flat(u), strength=1.0, radius_px=3.0, seed=4).astype(np.float64).std()
         for u in (0.0, 0.05, 0.5, 0.95, 1.0)}
    assert s[0.0] == 0.0 and s[1.0] == 0.0  # 검정·흰색에는 입자가 없다
    assert s[0.5] > 2.0 * s[0.05] and s[0.5] > 2.0 * s[0.95]
    assert s[0.5] / s[0.05] == pytest.approx(0.5 / np.sqrt(0.05 * 0.95), rel=0.3)  # sqrt(u(1-u)) 모양


def test_grain_follows_local_signal_in_one_image():
    blocks = (0.15, 0.35, 0.55, 0.85)
    img = np.concatenate([_flat(u, 256, 64) for u in blocks], axis=1)
    g = sf.grain_field(img, strength=1.0, radius_px=2.5, seed=8).astype(np.float64)
    for i, u in enumerate(blocks):
        blk = g[:, i * 64 + 12:(i + 1) * 64 - 12]
        assert blk.std() == pytest.approx(GRAIN_SIGMA_MAX * 2 * np.sqrt(u * (1 - u)), rel=0.25)
        assert abs(blk.mean() - u) < 0.012  # 표준오차 ~0.0025


@pytest.mark.parametrize("r", [3.0, 6.0])
def test_grain_spatial_correlation_scales_with_radius(r):
    n = 384 if r > 4 else 256
    g = sf.grain_field(_flat(0.5, n, n, 1), strength=1.0, radius_px=r, seed=5)[..., 0].astype(np.float64)

    def rho(d):
        lag = max(1, int(round(d)))
        return 0.5 * (autocorr(g, lag, 1) + autocorr(g, lag, 0))

    assert 0.5 <= rho(r / 2) <= 0.92
    assert 0.15 <= rho(r) <= 0.6
    assert abs(rho(2.5 * r)) < 0.12
    assert abs(rho(4 * r)) < 0.12


def test_grain_larger_radius_is_smoother():
    f = lambda r: sf.grain_field(_flat(0.5, 256, 256, 1), strength=1.0, radius_px=r, seed=6)[..., 0].astype(np.float64)
    rs = [0.75, 2.0, 4.0, 8.0]
    c = [autocorr(f(r), 3, 1) for r in rs]
    assert c[0] < c[1] < c[2] < c[3]
    assert c[0] < 0.1 and c[3] > 0.75
    # 화소 단위 분산은 반경과 무관(진폭은 strength 와 u 로만 정해진다)
    stds = [f(r).std() for r in rs]
    assert max(stds) / min(stds) < 1.3


def test_grain_no_visible_tiling():
    g = sf.grain_field(_flat(0.5, 512, 512, 1), strength=1.0, radius_px=4.0, seed=3)[..., 0].astype(np.float64)
    for lag in (96, 128, 160, 256):
        assert abs(autocorr(g, lag, 1)) < 0.1 and abs(autocorr(g, lag, 0)) < 0.1


def test_grain_channels_independent():
    g = sf.grain_field(_flat(0.5, 256, 256, 3), strength=1.0, radius_px=3.0, seed=9).astype(np.float64)
    n = g - 0.5
    assert abs(corr(n[..., 0], n[..., 1])) < 0.1 and abs(corr(n[..., 1], n[..., 2])) < 0.1
    assert abs(corr(n[..., 0], n[..., 2])) < 0.1


def test_grain_determinism_and_frame_seed_independence():
    u = _flat(0.5, 192, 192, 3)
    a = sf.grain_field(u, strength=0.8, radius_px=3.0, seed=7, frame=4)
    assert a.tobytes() == sf.grain_field(u, strength=0.8, radius_px=3.0, seed=7, frame=4).tobytes()
    fields = [sf.grain_field(u, strength=0.8, radius_px=3.0, seed=7, frame=f).astype(np.float64) - 0.5
              for f in range(4)]
    other = sf.grain_field(u, strength=0.8, radius_px=3.0, seed=8, frame=0).astype(np.float64) - 0.5
    for i in range(4):
        for j in range(i + 1, 4):
            assert abs(corr(fields[i], fields[j])) < 0.1  # 프레임마다 새 입자
        assert abs(corr(fields[i], other)) < 0.1
    # 프레임 사이 통계는 같다
    assert max(f.std() for f in fields) / min(f.std() for f in fields) < 1.08


def test_grain_does_not_touch_global_rng():
    np.random.seed(99)
    before = np.random.get_state()[1].copy()
    sf.grain_field(_flat(0.5, 32, 32, 3), strength=1.0, radius_px=2.0, seed=1)
    assert np.array_equal(np.random.get_state()[1], before)


def test_grain_errors():
    u = _flat(0.5, 8, 8, 3)
    for kw in (dict(strength=-0.1, radius_px=2.0), dict(strength=1.1, radius_px=2.0),
               dict(strength=float("nan"), radius_px=2.0), dict(strength=0.5, radius_px=0.0),
               dict(strength=0.5, radius_px=-1.0), dict(strength=0.5, radius_px=float("inf")),
               dict(strength=0.5, radius_px=2.0, seed=-1), dict(strength=0.5, radius_px=2.0, frame=-1)):
        with pytest.raises(ValueError):
            sf.grain_field(u, **kw)
    with pytest.raises(TypeError):
        sf.grain_field(u.astype(np.float16), strength=0.5, radius_px=2.0)
    with pytest.raises(TypeError):
        sf.grain_field(u, strength=0.5, radius_px=2.0, seed=1.5)
    for bad in (u * 2.2, u - 1.0):
        with pytest.raises(ValueError):
            sf.grain_field(bad, strength=0.5, radius_px=2.0)  # u 는 [0,1] 안
    nan = u.copy()
    nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        sf.grain_field(nan, strength=0.5, radius_px=2.0)
    with pytest.raises(ValueError):
        sf.grain_field(np.zeros((4,), np.float32), strength=0.5, radius_px=2.0)
    with pytest.raises(ValueError):
        sf.grain_field(np.zeros((2, 4, 4, 3), np.float32), strength=0.5, radius_px=2.0)
    with pytest.raises(TypeError):
        sf.grain_field(u, 0.5, 2.0)  # 키워드 전용
