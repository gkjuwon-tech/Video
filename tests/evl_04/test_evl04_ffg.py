"""EVL-04 FFG: 광류(DIS 대용 순수 numpy/scipy 구현)의 중앙값·P95, 단위 px@1080p."""
import numpy as np
import pytest

from evl04_support import ff, shift_img, texture


@pytest.fixture(scope="module")
def tex():
    return texture(192, 256, seed=3, sigma=2.5)


def _interior(flow, b=16):
    return flow[b:-b, b:-b]


def test_flow_shape_dtype_and_sign_convention(tex):
    m = ff()
    b = shift_img(tex, 0.0, 2.0)  # b(p) = a(p - (0,2))  =>  a(p) = b(p + (dx=2, dy=0))
    fl = m.estimate_flow(tex, b)
    assert fl.shape == (192, 256, 2) and fl.dtype == np.float32
    inner = _interior(fl)
    assert np.median(inner[..., 0]) == pytest.approx(2.0, abs=0.1)  # 채널 0 = dx
    assert np.median(inner[..., 1]) == pytest.approx(0.0, abs=0.1)  # 채널 1 = dy
    fl2 = m.estimate_flow(tex, shift_img(tex, -1.5, 0.0))
    assert np.median(_interior(fl2)[..., 1]) == pytest.approx(-1.5, abs=0.1)


@pytest.mark.parametrize("dy,dx", [(0.4, -0.3), (-1.2, 0.8), (2.0, 1.5), (-3.0, 3.0)])
def test_flow_accuracy_on_known_subpixel_translations(tex, dy, dx):
    m = ff()
    fl = _interior(m.estimate_flow(tex, shift_img(tex, dy, dx)))
    err = np.hypot(fl[..., 0] - dx, fl[..., 1] - dy)
    assert np.median(err) < 0.1, np.median(err)
    assert np.percentile(err, 95) < 0.5, np.percentile(err, 95)


def test_flow_identity_and_degenerate_inputs_are_finite_and_zero(tex):
    m = ff()
    fl = m.estimate_flow(tex, tex)
    assert np.isfinite(fl).all() and np.abs(fl).max() < 1e-2
    flat = np.full((64, 80, 3), 0.4, np.float32)
    f2 = m.estimate_flow(flat, flat)
    assert np.isfinite(f2).all() and np.abs(f2).max() < 1e-3  # 질감이 없어도 NaN 없이 0


def test_flow_is_deterministic_and_does_not_mutate(tex):
    m = ff()
    b = shift_img(tex, 0.5, 1.0)
    ka, kb = tex.copy(), b.copy()
    f1, f2 = m.estimate_flow(tex, b), m.estimate_flow(tex, b)
    assert np.array_equal(f1, f2) and np.array_equal(tex, ka) and np.array_equal(b, kb)


def test_ffg_median_scales_to_1080p_by_short_side():
    m = ff()
    img = texture(144, 192, seed=4)
    sh = shift_img(img, 0.0, 2.0)
    r = m.ffg(img, sh, ref_short=144)  # 영상 자체가 기준 해상도면 그대로 2 px
    assert r.median == pytest.approx(2.0, abs=0.15)
    r1080 = m.ffg(img, sh)  # 기본: 짧은 변 144 -> 1080, x7.5
    assert m.REF_SHORT == 1080
    assert r1080.median == pytest.approx(2.0 * 1080 / 144, rel=0.08)
    assert type(r1080.median) is float and type(r1080.p95) is float and isinstance(r1080.n, int)
    assert r1080.n == 144 * 192 and r1080.p95 >= r1080.median


def test_ffg_identity_is_near_zero(tex):
    m = ff()
    r = m.ffg(tex, tex, ref_short=192)
    assert r.median < 0.02 and r.p95 < 0.05


def test_ffg_mask_selects_region_stats():
    m = ff()
    base = texture(192, 256, seed=5, sigma=2.5)
    sh = shift_img(base, 3.0, 0.0)
    f0 = base.copy()
    f0[128:] = sh[128:]  # 아래 1/3 만 3 px 이동
    full = m.ffg(base, f0, ref_short=192)
    assert full.median < 0.3  # 위쪽 2/3 는 움직이지 않았으므로 중앙값은 0 근처
    region = np.zeros((192, 256), bool)
    region[144:184, 16:240] = True
    sal = m.ffg(base, f0, mask=region, ref_short=192)
    assert sal.n == int(region.sum())
    assert sal.median == pytest.approx(3.0, abs=0.3)
    assert 2.6 < sal.p95 < 3.6
    assert m.ffg_sal(base, f0, region, ref_short=192) == pytest.approx(sal.p95, abs=1e-9)


def test_ffg_validation(tex):
    m = ff()
    with pytest.raises(ValueError):
        m.ffg(tex, tex[:, :-1])
    with pytest.raises(ValueError):
        m.ffg(tex, tex, mask=np.ones((10, 10), bool))
    with pytest.raises(ValueError):
        m.ffg(tex, tex, mask=np.zeros(tex.shape[:2], bool))
    with pytest.raises(ValueError):
        m.ffg(tex, tex, ref_short=0)
    with pytest.raises(ValueError):
        m.estimate_flow(tex, tex[:-1])


def test_top_fraction_mask():
    m = ff()
    assert m.SAL_FRAC == 0.10
    s = np.arange(100, dtype=np.float64).reshape(10, 10)
    mk = m.top_fraction_mask(s)
    assert mk.dtype == bool and mk.shape == (10, 10) and int(mk.sum()) == 10
    assert mk[9].all() and not mk[:9].any()  # 값이 가장 큰 10 개
    flat = np.ones((4, 5))
    mk2 = m.top_fraction_mask(flat, 0.25)  # 동점: 래스터 순서가 앞선 5 개(ceil(0.25*20))
    assert int(mk2.sum()) == 5 and mk2.ravel()[:5].all()
    assert int(m.top_fraction_mask(np.arange(7.0).reshape(1, 7), 0.10).sum()) == 1  # ceil(0.7)
    with pytest.raises(ValueError):
        m.top_fraction_mask(s, 0.0)
    with pytest.raises(ValueError):
        m.top_fraction_mask(s, 1.5)
    bad = s.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.top_fraction_mask(bad)
