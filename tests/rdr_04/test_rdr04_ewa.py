"""RDR-04 EWA 비등방 필터: 공분산 수식, 항등/확대에서 순수 보간, 축소 대역 이득, 비등방 보존, 클램프 의미."""
import math

import numpy as np
import pytest

from buc.render.filters import EwaOpts, bspline_sample, ewa_covariance, ewa_sample, prefilter_texture
from rdr04_helpers import N, band_image, cos_x, cos_y, fit_amp, freq, rms_amp

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _rot(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


# --------------------------------------------------------------------------------------------- 공분산
def test_covariance_zero_for_identity_and_magnification():
    for J in (np.eye(2), 0.5 * np.eye(2), np.diag([0.9, 0.3]), _rot(33.0), 0.7 * _rot(-12.0), np.zeros((2, 2))):
        assert np.abs(ewa_covariance(J)).max() < 1e-12


def test_covariance_diagonal_formula():
    # J = diag(3, 2): 원본 px/목표 px. σ_i = κ sqrt(s_i² - 1) (원본 px), κ = 0.5. 큰 축(3)이 x.
    S = ewa_covariance(np.diag([3.0, 2.0]))
    assert S.shape == (2, 2)
    assert np.allclose(S, np.diag([0.25 * 8.0, 0.25 * 3.0]), atol=1e-12)
    S2 = ewa_covariance(np.diag([2.0, 3.0]))
    assert np.allclose(S2, np.diag([0.25 * 3.0, 0.25 * 8.0]), atol=1e-12)
    S3 = ewa_covariance(np.diag([3.0, 2.0]), opts=EwaOpts(aa_scale=1.0))
    assert np.allclose(S3, np.diag([8.0, 3.0]), atol=1e-12)
    S0 = ewa_covariance(np.diag([3.0, 2.0]), opts=EwaOpts(aa_scale=0.0))
    assert np.abs(S0).max() == 0.0


def test_covariance_grazing_axis_only():
    # 한 축만 축소(수평선 쪽 바닥면): 공분산은 그 축에만 생기고 다른 축은 0 -> 비등방의 정의
    S = ewa_covariance(np.diag([1.0, 8.0]))
    assert np.allclose(S, np.diag([0.0, 0.25 * 63.0]), atol=1e-12)


def test_covariance_symmetric_psd_and_batched():
    rng = np.random.default_rng(11)
    J = rng.normal(size=(5, 7, 2, 2)) * 2.0
    S = ewa_covariance(J)
    assert S.shape == (5, 7, 2, 2)
    assert np.allclose(S, np.swapaxes(S, -1, -2), atol=1e-12)
    assert np.linalg.eigvalsh(S).min() > -1e-10
    assert np.allclose(S[2, 3], ewa_covariance(J[2, 3]), atol=1e-12)         # 배치 = 개별


def test_covariance_rotation_covariance():
    J = np.array([[2.5, 0.4], [-0.3, 1.7]])
    S = ewa_covariance(J)
    Rs = _rot(37.0)
    assert np.allclose(ewa_covariance(Rs @ J), Rs @ S @ Rs.T, atol=1e-10)      # 원본 좌표 회전 -> 공분산 공변
    assert np.allclose(ewa_covariance(J @ Rs), S, atol=1e-10)                  # 목표 좌표 회전 -> 불변(등방 목표 화소)


def test_covariance_depends_only_on_singular_values_of_J():
    # 같은 특이값·좌특이벡터면 우특이벡터(목표 쪽 회전·반사)와 무관
    U = _rot(20.0)
    Sg = np.diag([4.0, 1.5])
    S = ewa_covariance(U @ Sg)
    expect = U @ np.diag([0.25 * 15.0, 0.25 * 1.25]) @ U.T
    assert np.allclose(S, expect, atol=1e-10)
    flip = np.diag([1.0, -1.0])
    assert np.allclose(ewa_covariance(U @ Sg @ flip), expect, atol=1e-10)


def test_covariance_aniso_clamp_widens_minor_axis():
    # 비율 한도 16: J = diag(1, 64) -> 작은 축 s2' = 64/16 = 4, σ_minor = 0.5 sqrt(15)
    o = EwaOpts(max_aniso=16.0, max_sigma_px=1e9)
    S = ewa_covariance(np.diag([1.0, 64.0]), opts=o)
    assert S[1, 1] == pytest.approx(0.25 * (64.0 ** 2 - 1.0), rel=1e-12)
    assert S[0, 0] == pytest.approx(0.25 * (16.0 - 1.0), rel=1e-12)
    # 한도 1 = 등방: 두 축 σ 가 큰 축의 것과 같다(s2' = s1)
    Si = ewa_covariance(np.diag([1.0, 8.0]), opts=EwaOpts(max_aniso=1.0))
    assert np.allclose(Si, 0.25 * 63.0 * np.eye(2), atol=1e-12)
    # 한도가 비율보다 크면 영향 없다
    S8 = ewa_covariance(np.diag([1.0, 8.0]), opts=EwaOpts(max_aniso=100.0))
    assert np.allclose(S8, np.diag([0.0, 0.25 * 63.0]), atol=1e-12)


def test_covariance_sigma_cap_and_recon_sigma():
    S = ewa_covariance(np.diag([1.0, 100.0]), opts=EwaOpts(max_sigma_px=3.0, max_aniso=1000.0))
    assert S[1, 1] == pytest.approx(9.0, rel=1e-12) and S[0, 0] == pytest.approx(0.0, abs=1e-12)
    S = ewa_covariance(np.eye(2), opts=EwaOpts(recon_sigma_px=0.6))
    assert np.allclose(S, 0.36 * np.eye(2), atol=1e-12)
    S = ewa_covariance(np.diag([3.0, 1.0]), opts=EwaOpts(recon_sigma_px=0.6))
    assert np.allclose(S, np.diag([2.0 + 0.36, 0.36]), atol=1e-12)


# --------------------------------------------------------------------------------------------- 표본
def test_identity_and_magnification_are_pure_cardinal_interpolation():
    img = band_image(64)
    t = prefilter_texture(img)
    rng = np.random.default_rng(21)
    uv = rng.uniform(0, 63, size=(400, 2))
    ref = bspline_sample(t, uv)
    for J in (np.eye(2), 0.4 * np.eye(2), np.diag([0.9, 0.5]), _rot(25.0) * 0.8):
        assert np.abs(ewa_sample(t, uv, J) - ref).max() < 1e-9
    # 정수 노드에서는 원 영상 그대로(첫 프레임 = 입력의 필터 쪽 조건)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
    out = ewa_sample(t, np.stack([xx, yy], -1), np.eye(2))
    assert out.shape == img.shape and np.abs(out - img).max() < 1e-9


def test_constant_image_any_jacobian():
    t = prefilter_texture(np.full((32, 40, 3), 0.42, np.float32))
    rng = np.random.default_rng(5)
    uv = rng.uniform(-5, 45, size=(300, 2))
    J = rng.normal(size=(300, 2, 2)) * 3.0
    out = ewa_sample(t, uv, J)
    assert np.abs(out - 0.42).max() < 1e-6                                   # 가중 정규화 -> 평균 보존


def test_linear_in_image_so_premultiplied_alpha_is_safe():
    a, b = band_image(48, seed=1), band_image(48, seed=2)
    rng = np.random.default_rng(8)
    uv = rng.uniform(4, 43, size=(120, 2))
    J = rng.uniform(-3, 3, size=(120, 2, 2))
    sa = ewa_sample(prefilter_texture(a), uv, J)
    sb = ewa_sample(prefilter_texture(b), uv, J)
    sab = ewa_sample(prefilter_texture(0.25 * a + 2.0 * b), uv, J)
    assert np.abs(sab - (0.25 * sa + 2.0 * sb)).max() < 1e-5
    # 투명 텍셀의 쓰레기 rgb 가 전치 알파 경로에서는 새지 않는다: 알파 0 영역에 거대한 rgb 가 있어도 rgb*alpha 가 0
    alpha = np.zeros((48, 48), np.float32)
    alpha[:, 24:] = 1.0
    garbage = np.full((48, 48), 1e3, np.float32)
    pre = garbage * alpha
    s = ewa_sample(prefilter_texture(pre), np.array([[8.0, 20.0], [10.0, 5.0]]), np.diag([2.0, 2.0]))
    assert np.abs(s).max() < 1e-3                                              # 왼쪽(알파 0) 안쪽은 0


def test_channels_independent_and_float32_matches_float64():
    img = band_image(40)
    rng = np.random.default_rng(13)
    uv = rng.uniform(0, 39, size=(80, 2))
    J = rng.uniform(-2.5, 2.5, size=(80, 2, 2))
    full = ewa_sample(prefilter_texture(img), uv, J)
    for c in range(3):
        one = ewa_sample(prefilter_texture(img[..., c]), uv, J)
        assert np.abs(one - full[:, c]).max() < 1e-9
    f64 = ewa_sample(prefilter_texture(img.astype(np.float64)), uv, J)
    assert np.array_equal(full, f64)


def test_mirror_symmetry_with_filtering():
    t = prefilter_texture(band_image(32))
    J = np.diag([2.0, 1.0])                                        # 대칭 J: 거울 좌표에서 같은 값
    a = ewa_sample(t, np.array([[-2.5, 9.0]]), J)
    b = ewa_sample(t, np.array([[2.5, 9.0]]), J)
    assert np.abs(a - b).max() < 1e-9
    a = ewa_sample(t, np.array([[31 + 2.5, 9.0]]), J)
    b = ewa_sample(t, np.array([[31 - 2.5, 9.0]]), J)
    assert np.abs(a - b).max() < 1e-9


# --------------------------------------------------------------------------------------------- 주파수 응답
def _row_gain(img_rows, m, J, opts=None, x0=10, x1=N - 10):
    """cos_x 영상 한 줄을 정수 노드에서 EWA 로 읽은 진폭."""
    t = prefilter_texture(img_rows)
    xs = np.arange(x0, x1, dtype=np.float64)
    uv = np.stack([xs, np.full_like(xs, 4.0)], axis=-1)
    out = ewa_sample(t, uv, J, opts=opts)
    return fit_amp(out, xs, freq(m))


@pytest.mark.parametrize("m,scale", [(13, 2.0), (26, 2.0), (26, 3.0), (52, 1.5)])
def test_isotropic_minification_gain_matches_gaussian(m, scale):
    """J = s I: 원본 px 단위 σ = 0.5 sqrt(s²-1) 인 가우시안. 노드(정수 uv)에서 이득 = exp(-2π² σ² f²) (보간 MTF ≈ 1)."""
    f = freq(m)
    sigma = 0.5 * math.sqrt(scale ** 2 - 1.0)
    expect = math.exp(-2.0 * math.pi ** 2 * sigma ** 2 * f ** 2)
    got = _row_gain(cos_x(m), m, scale * np.eye(2))
    assert got == pytest.approx(expect, abs=0.012)


def test_recon_sigma_adds_gaussian_blur_at_identity():
    m, s = 26, 0.6
    f = freq(m)
    expect = math.exp(-2.0 * math.pi ** 2 * s ** 2 * f ** 2)
    got = _row_gain(cos_x(m), m, np.eye(2), opts=EwaOpts(recon_sigma_px=s))
    assert got == pytest.approx(expect, abs=0.012)
    assert _row_gain(cos_x(m), m, np.eye(2)) == pytest.approx(1.0, abs=0.003)


def test_stopband_attenuated_more_with_larger_scale():
    g = [_row_gain(cos_x(90), 90, s * np.eye(2)) for s in (1.0, 1.5, 2.0, 3.0)]
    assert g[0] == pytest.approx(1.0, abs=0.003)
    assert all(g[i] > g[i + 1] for i in range(3))
    assert g[3] < 0.05


def test_anisotropic_footprint_keeps_unsqueezed_axis_sharp():
    """바닥면(수평선 쪽): y 로만 8배 축소. x 방향 0.25 cy/px 줄무늬는 살아야 하고(등방 필터는 지워 버린다),
    y 방향 고주파는 지워져야 한다."""
    J = np.diag([1.0, 8.0])
    gx = _row_gain(cos_x(64), 64, J)                                           # f=0.25 along x
    assert 0.97 <= gx <= 1.03
    gx_iso = _row_gain(cos_x(64), 64, J, opts=EwaOpts(max_aniso=1.0))          # 등방 대조군
    assert gx_iso < 0.05
    # y 방향: 노드 (x=64, y=4..) 열을 읽는다
    img = cos_y(115)
    t = prefilter_texture(img)
    ys = np.arange(20, 100, dtype=np.float64)
    out = ewa_sample(t, np.stack([np.full_like(ys, 64.0), ys], axis=-1), J)
    assert rms_amp(out) < 0.02


@pytest.mark.parametrize("aniso,expect", [(4.0, math.exp(-2 * math.pi ** 2 * 0.75 * 0.25 ** 2)),
                                          (2.0, math.exp(-2 * math.pi ** 2 * 3.75 * 0.25 ** 2))])
def test_aniso_clamp_semantics_on_minor_axis_gain(aniso, expect):
    """max_aniso=A: 작은 축 특이값이 s1/A 로 올라가 σ_minor = 0.5 sqrt((s1/A)² - 1). x 줄무늬 이득으로 검증."""
    got = _row_gain(cos_x(64), 64, np.diag([1.0, 8.0]), opts=EwaOpts(max_aniso=aniso))
    assert got == pytest.approx(expect, abs=0.03)


def test_rotated_anisotropy_keeps_the_right_direction():
    """J 가 45° 회전돼 있어도 보존되는 방향은 J 의 작은 특이 방향이다 (U 가 정확히 쓰이는지)."""
    R = _rot(45.0)
    J = R @ np.diag([1.0, 8.0])                                              # 큰 축 = R 의 둘째 열 방향 (대각선)
    # 대각선 큰 축에 직교하는 방향(작은 축)을 따라 줄무늬를 둔다: 작은 축 단위벡터 e = R[:,0] = (cos45, sin45)
    n = 129
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    fk = 0.2
    keep = np.cos(2 * np.pi * fk * (xx * R[0, 0] + yy * R[1, 0]))            # 작은 축 방향으로 변하는 무늬
    kill = np.cos(2 * np.pi * fk * (xx * R[0, 1] + yy * R[1, 1]))            # 큰 축 방향으로 변하는 무늬
    pts = np.stack([np.full(40, 64.0) + np.linspace(-6, 6, 40) * R[0, 0], np.full(40, 64.0) + np.linspace(-6, 6, 40) * R[1, 0]], -1)
    k_out = ewa_sample(prefilter_texture(keep.astype(np.float32)), pts, J)
    x_out = ewa_sample(prefilter_texture(kill.astype(np.float32)), pts, J)
    k_in = bspline_sample(prefilter_texture(keep.astype(np.float32)), pts)
    assert np.abs(k_out - k_in).max() < 0.03                                 # 작은 축 무늬는 그대로
    assert np.abs(x_out).max() < 0.05                                        # 큰 축 무늬는 지워진다 (<= 0.05)
