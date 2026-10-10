"""RDR-04 카디널 B-스플라인: 기저, 사전필터(노드 보간), 해석적 MTF 와 수치 커널의 일치, 거울 경계, 선형성."""
import numpy as np
import pytest

from buc.render.filters import bspline_basis, bspline_sample, interp_mtf, prefilter_texture
from rdr04_helpers import band_image

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _grid_uv(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([xx, yy], axis=-1)


# --------------------------------------------------------------------------------------------- 기저
@pytest.mark.parametrize("deg,center,at1,at2", [(1, 1.0, 0.0, 0.0), (3, 2 / 3, 1 / 6, 0.0), (5, 11 / 20, 26 / 120, 1 / 120)])
def test_basis_known_values(deg, center, at1, at2):
    assert bspline_basis(0.0, deg) == pytest.approx(center, abs=1e-12)
    assert bspline_basis(1.0, deg) == pytest.approx(at1, abs=1e-12)
    assert bspline_basis(-1.0, deg) == pytest.approx(at1, abs=1e-12)
    assert bspline_basis(2.0, deg) == pytest.approx(at2, abs=1e-12)
    assert bspline_basis(np.array([[0.0, 1.0]]), deg).shape == (1, 2)


@pytest.mark.parametrize("deg", [1, 3, 5])
def test_basis_support_symmetry_partition_of_unity_and_integral(deg):
    half = (deg + 1) / 2.0
    x = np.linspace(-half - 1, half + 1, 2001)
    b = bspline_basis(x, deg)
    assert np.all(b >= -1e-15)
    assert np.allclose(b, bspline_basis(-x, deg), atol=1e-14)
    assert np.all(bspline_basis(np.array([half, half + 0.5, -half - 3.0]), deg) == 0.0)
    for s in (0.0, 0.13, 0.5, 0.77):
        tot = sum(bspline_basis(s - k, deg) for k in range(-6, 7))
        assert tot == pytest.approx(1.0, abs=1e-12)
    fine = np.arange(-8, 8, 1 / 256)
    assert bspline_basis(fine, deg).sum() / 256 == pytest.approx(1.0, abs=1e-6)


def test_basis_cubic_is_c2_smooth():
    # 3차 B-스플라인은 C2: 매듭 x=1 에서 값·1차·2차 도함수가 연속이다(수치 차분).
    h = 1e-3
    for knot in (1.0, 2.0):
        xs = np.array([knot - h, knot, knot + h])
        v = bspline_basis(xs, 3)
        d1_l, d1_r = (v[1] - v[0]) / h, (v[2] - v[1]) / h
        assert d1_l == pytest.approx(d1_r, abs=2e-3)
        d2 = (v[0] - 2 * v[1] + v[2]) / h ** 2
        far = bspline_basis(np.array([knot - 2 * h, knot - h, knot]), 3)
        d2_l = (far[0] - 2 * far[1] + far[2]) / h ** 2
        assert d2 == pytest.approx(d2_l, abs=5e-2)


# --------------------------------------------------------------------------------------------- 사전필터
@pytest.mark.parametrize("deg", [1, 3, 5])
def test_cardinal_interpolation_reproduces_nodes(deg):
    """사전필터가 있으면 정수 좌표(노드)에서 원 영상을 되돌려 준다. 쌍선형이 아닌 '진짜 보간'의 정의."""
    img = np.random.default_rng(7).random((17, 23, 3)).astype(np.float32)
    t = prefilter_texture(img, degree=deg)
    out = bspline_sample(t, _grid_uv(17, 23))
    assert np.abs(out - img).max() < 1e-9
    assert t.coef.shape == img.shape
    if deg > 1:
        # 계수는 영상과 달라야 한다(사전필터가 실제로 일을 한다). 사전필터 없이 기저로만 보간하면 노드 값이 평활된다.
        assert np.abs(t.coef - img).max() > 0.05


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (2, 2), (3, 3), (2, 7), (9, 1)])
def test_tiny_images_interpolate_nodes(shape):
    img = np.random.default_rng(1).random(shape).astype(np.float32)
    out = bspline_sample(prefilter_texture(img), _grid_uv(*shape))
    assert out.shape == shape
    assert np.abs(out - img).max() < 1e-9


def test_degree1_prefilter_is_identity_and_equals_bilinear():
    img = np.random.default_rng(2).random((10, 11)).astype(np.float32)
    t = prefilter_texture(img, degree=1)
    assert np.abs(t.coef - img).max() < 1e-12
    uv = np.array([[2.25, 3.75], [0.0, 0.5], [9.9, 8.1]])
    out = bspline_sample(t, uv)
    ref = []
    for x, y in uv:
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        x1, y1 = min(x0 + 1, 10), min(y0 + 1, 9)
        ref.append((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x1]
                   + (1 - fx) * fy * img[y1, x0] + fx * fy * img[y1, x1])
    assert np.allclose(out, ref, atol=1e-6)


def test_polynomial_precision_interior():
    """3차 보간은 3차 이하 다항식을 (경계 영향이 사라지는) 안쪽에서 재현한다."""
    n = 48
    y, x = np.mgrid[0:n, 0:n].astype(np.float64) / (n - 1)
    img = (0.2 + 0.6 * x - 0.3 * y + 0.5 * x * y + 0.4 * x ** 2 - 0.2 * y ** 3 + 0.3 * x ** 2 * y).astype(np.float64)
    t = prefilter_texture(img, degree=3)
    rng = np.random.default_rng(5)
    uv = rng.uniform(14.0, n - 1 - 14.0, size=(300, 2))
    ref = (0.2 + 0.6 * uv[:, 0] / (n - 1) - 0.3 * uv[:, 1] / (n - 1) + 0.5 * uv[:, 0] * uv[:, 1] / (n - 1) ** 2
           + 0.4 * (uv[:, 0] / (n - 1)) ** 2 - 0.2 * (uv[:, 1] / (n - 1)) ** 3
           + 0.3 * (uv[:, 0] / (n - 1)) ** 2 * (uv[:, 1] / (n - 1)))
    assert np.abs(bspline_sample(t, uv) - ref).max() < 2e-5


@pytest.mark.parametrize("deg", [3, 5])
def test_constant_image_stays_constant_everywhere(deg):
    t = prefilter_texture(np.full((13, 9, 2), 0.37, np.float32), degree=deg)
    uv = np.random.default_rng(3).uniform(-30, 40, size=(500, 2))
    assert np.abs(bspline_sample(t, uv) - float(np.float32(0.37))).max() < 1e-9


def test_mirror_border_whole_sample_symmetric():
    img = band_image(32)
    t = prefilter_texture(img)
    for d in (0.3, 1.7, 4.2):
        a = bspline_sample(t, np.array([[-d, 7.3]]))
        b = bspline_sample(t, np.array([[d, 7.3]]))
        assert np.abs(a - b).max() < 1e-9                          # x=0 둘레 거울
        a = bspline_sample(t, np.array([[31 + d, 7.3]]))
        b = bspline_sample(t, np.array([[31 - d, 7.3]]))
        assert np.abs(a - b).max() < 1e-9                          # x=W-1 둘레 거울
        a = bspline_sample(t, np.array([[5.1, -d]]))
        b = bspline_sample(t, np.array([[5.1, d]]))
        assert np.abs(a - b).max() < 1e-9                          # y=0
        a = bspline_sample(t, np.array([[5.1, 31 + d]]))
        b = bspline_sample(t, np.array([[5.1, 31 - d]]))
        assert np.abs(a - b).max() < 1e-9                          # y=H-1
    far = bspline_sample(t, np.array([[1e5 + 0.3, -3e4 + 0.7]]))
    assert np.isfinite(far).all() and far.min() > -1.0 and far.max() < 2.0


def test_linearity_and_channel_independence():
    a, b = band_image(28, seed=1), band_image(28, seed=2)
    uv = np.random.default_rng(9).uniform(0, 27, size=(200, 2))
    sa = bspline_sample(prefilter_texture(a), uv)
    sb = bspline_sample(prefilter_texture(b), uv)
    sab = bspline_sample(prefilter_texture(0.3 * a + 1.7 * b), uv)
    assert np.abs(sab - (0.3 * sa + 1.7 * sb)).max() < 1e-5
    for c in range(3):
        sc = bspline_sample(prefilter_texture(a[..., c]), uv)
        assert np.abs(sc - sa[:, c]).max() < 1e-9


def test_float32_and_float64_inputs_agree():
    a = band_image(24)
    uv = np.random.default_rng(4).uniform(0, 23, size=(100, 2))
    s32 = bspline_sample(prefilter_texture(a), uv)
    s64 = bspline_sample(prefilter_texture(a.astype(np.float64)), uv)
    assert np.array_equal(s32, s64)


# --------------------------------------------------------------------------------------------- MTF
def test_interp_mtf_known_values():
    f = np.array([0.0, 0.1, 0.2, 0.35, 0.5])
    pre3 = interp_mtf(f, 3)
    raw3 = interp_mtf(f, 3, prefiltered=False)
    lin = interp_mtf(f, 1)
    pre5 = interp_mtf(f, 5)
    assert pre3.shape == f.shape and pre3.dtype == np.float64
    assert pre3[0] == pytest.approx(1.0, abs=1e-12) and raw3[0] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(pre3[1:], [0.999762, 0.995060, 0.915951, 0.492767], atol=2e-5)
    assert np.allclose(raw3[1:], [0.936117, 0.765870, 0.431173, 0.164256], atol=2e-5)
    assert np.allclose(lin[1:], [0.967531, 0.875140, 0.656638, 0.405285], atol=2e-5)
    assert pre5[3] == pytest.approx(0.975812, abs=2e-5)
    assert interp_mtf(0.35, 1, prefiltered=False) == pytest.approx(lin[3], abs=1e-12)    # 1차는 사전필터가 항등
    assert interp_mtf(-0.35, 3) == pytest.approx(pre3[3], abs=1e-12)                      # 우함수


def test_xff0_band_gate_prefiltered_cubic_vs_bilinear():
    """첫 프레임 게이트의 근거 수치: 0.35 cy/px 에서 카디널 3차 >= 0.9, 쌍선형 < 0.7, 비사전필터 3차 < 0.5."""
    assert interp_mtf(0.35, 3) >= 0.9
    assert interp_mtf(0.35, 5) >= interp_mtf(0.35, 3)
    assert interp_mtf(0.35, 1) < 0.7
    assert interp_mtf(0.35, 3, prefiltered=False) < 0.5
    f = np.linspace(0, 0.5, 51)
    m = interp_mtf(f, 3)
    assert np.all(np.diff(m) <= 1e-12)                                                     # 단조 감소
    assert np.all(interp_mtf(f, 3) >= interp_mtf(f, 3, prefiltered=False) - 1e-12)         # 사전필터가 항상 더 높다
    assert np.all(interp_mtf(f, 3) >= interp_mtf(f, 1) - 1e-12)


@pytest.mark.parametrize("deg", [1, 3, 5])
def test_interp_mtf_matches_numeric_kernel(deg):
    """해석식이 실제 보간 커널(델타 영상의 연속 보간)의 푸리에 변환과 같다 -> 식과 구현이 한 몸."""
    n = 64
    d = np.zeros((n, n), np.float64)
    d[32, 32] = 1.0
    t = prefilter_texture(d, degree=deg)
    xs = np.arange(-8.0, 8.0, 1 / 64) + 32.0
    k = bspline_sample(t, np.stack([xs, np.full_like(xs, 32.0)], axis=-1))
    for fr in (0.05, 0.15, 0.25, 0.35, 0.45):
        num = abs(np.sum(k * np.exp(-2j * np.pi * fr * (xs - 32.0))) / 64.0)
        assert num == pytest.approx(float(interp_mtf(fr, deg)), abs=2e-3)
    # 비사전필터 커널 = 기저 자체
    b = bspline_basis(xs - 32.0, deg)
    for fr in (0.05, 0.25, 0.45):
        num = abs(np.sum(b * np.exp(-2j * np.pi * fr * (xs - 32.0))) / 64.0)
        assert num == pytest.approx(float(interp_mtf(fr, deg, prefiltered=False)), abs=2e-3)
