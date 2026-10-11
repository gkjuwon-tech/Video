"""LNS-06 apply_prescription_bokeh: 화소별 출사동 커널 산란 (cat-eye + 스월 + 애너모픽 스퀴즈)."""
import numpy as np
import pytest

from buc.lens import optics
from buc.lens import prescriptions as P
from lns06_helpers import L1, ang_diff180, cov2, hdr_scene, impulse, minor_axis_deg, patch_kernel, smooth_img

# 57x57 영상: R = 56·sqrt(2)/2 = 39.6, 주점 (28,28). 중심에서 20 px 떨어진 화소의 r_frac ≈ 0.505 (-> 양자화 0.5).
N = 57
CEN = 28
OFF = 20


def mk(cat=0.0, blades=0, rot=0.0, **kw):
    return P.Prescription(optics=optics.OpticsSpec(catseye=cat, blades=blades, blade_rot_deg=rot), **kw)


def one_source(x, y, d, n=N, val=(4.0, 2.0, 1.0)):
    img = impulse(n, n, x, y, np.array(val, np.float32), base=0.0)
    coc = np.zeros((n, n), np.float32)
    coc[y, x] = d
    return img, coc


# ----------------------------------------------------------------------------------------------- 점광원 = 커널
@pytest.mark.parametrize("x,y,ang", [(CEN + OFF, CEN, 0.0), (CEN, CEN + OFF, 90.0)])
@pytest.mark.parametrize("swirl,squeeze", [(0.0, 1.0), (0.8, 1.0), (0.6, 1.5)])
def test_point_source_output_is_exit_pupil_kernel(x, y, ang, swirl, squeeze):
    rx = mk(cat=0.8, blades=0, swirl=swirl, squeeze=squeeze)
    d = 12.0                       # 스퀴즈 1.5 의 세로 반지름까지 프레임 안 (half = 8)
    img, coc = one_source(x, y, d)
    out = P.apply_prescription_bokeh(img, coc, rx)
    ker = P.exit_pupil_kernel(d, 0.5, ang, rx).astype(np.float64)
    half = (ker.shape[0] - 1) // 2
    assert out.shape == img.shape and out.dtype == img.dtype
    for ch, v in enumerate((4.0, 2.0, 1.0)):
        assert out[..., ch].sum(dtype=np.float64) == pytest.approx(v, rel=1e-4)       # 광속 보존
        assert L1(patch_kernel(out, x, y, half, ch), ker) < 0.05                        # 양자화 오차 안에서 같은 모양
    # 패치 밖은 0
    mask = np.ones((N, N), bool)
    mask[y - half:y + half + 1, x - half:x + half + 1] = False
    assert out[mask].max() == 0.0


def test_swirl_changes_off_axis_bokeh_but_not_on_axis():
    d = 12.0
    img, coc = one_source(CEN + OFF, CEN, d)
    a = P.apply_prescription_bokeh(img, coc, mk(cat=1.0, swirl=0.0))
    b = P.apply_prescription_bokeh(img, coc, mk(cat=1.0, swirl=1.0))
    half = 7
    pa, pb = patch_kernel(a, CEN + OFF, CEN, half), patch_kernel(b, CEN + OFF, CEN, half)
    assert L1(pa, pb) > 0.1
    # 렌즈 모양의 작은 축 (반경 방향 = 0°) 이 스월로 비틀린다
    assert ang_diff180(minor_axis_deg(cov2(pb)), 0.0) > ang_diff180(minor_axis_deg(cov2(pa)), 0.0) + 3.0
    # 광축 위 (r=0) 는 스월과 무관
    img0, coc0 = one_source(CEN, CEN, d)
    assert np.array_equal(P.apply_prescription_bokeh(img0, coc0, mk(cat=1.0, swirl=0.0)),
                          P.apply_prescription_bokeh(img0, coc0, mk(cat=1.0, swirl=1.0)))


def test_anamorphic_on_axis_bokeh_is_tall_ellipse():
    d = 16.0
    img, coc = one_source(CEN, CEN, d)
    out = P.apply_prescription_bokeh(img, coc, mk(squeeze=2.0))
    c = cov2(out[..., 0], about=(CEN, CEN))
    assert c[1, 1] / c[0, 0] == pytest.approx(4.0, rel=0.06)
    assert out[..., 0].sum(dtype=np.float64) == pytest.approx(4.0, rel=1e-4)
    ref = P.apply_prescription_bokeh(img, coc, mk(squeeze=1.0))
    c1 = cov2(ref[..., 0], about=(CEN, CEN))
    assert c1[0, 0] == pytest.approx(c1[1, 1], rel=0.03)                   # 스퀴즈 1 은 원형
    assert c[0, 0] * c[1, 1] == pytest.approx(c1[0, 0] * c1[1, 1], rel=0.05)


# ----------------------------------------------------------------------------------------------- LNS-05 이음매
@pytest.mark.parametrize("cat,blades,rot", [(0.0, 0, 0.0), (0.6, 6, 0.0), (1.0, 5, 12.0)])
def test_swirl0_squeeze1_equals_apply_catseye(cat, blades, rot):
    rng = np.random.default_rng(7)
    h, w = 40, 56
    img = hdr_scene(h, w)
    coc = (10.0 * rng.random((h, w))).astype(np.float32)
    got = P.apply_prescription_bokeh(img, coc, mk(cat=cat, blades=blades, rot=rot))
    want = optics.apply_catseye(img, coc, strength=cat, blades=blades, blade_rot_deg=rot)
    assert got.dtype == img.dtype
    assert np.abs(got - want).max() < 1e-5 * max(1.0, float(want.max()))


# ----------------------------------------------------------------------------------------------- 일반 성질
def test_sharp_pixels_pass_through_and_energy_conserved():
    rx = mk(cat=0.5, blades=5, swirl=0.8, squeeze=1.5)
    img = smooth_img(48, 64)
    keep = img.copy()
    out = P.apply_prescription_bokeh(img, np.full((48, 64), 1.0, np.float32), rx)
    assert np.array_equal(out, img) and out is not img
    assert np.array_equal(P.apply_prescription_bokeh(img, np.zeros((48, 64)), rx), img)
    assert np.array_equal(img, keep)
    # 내부 광원들 (경계에서 충분히 떨어진) 은 총 광속이 보존된다
    n = 72
    src = np.zeros((n, n, 3), np.float32)
    coc = np.zeros((n, n), np.float32)
    rng = np.random.default_rng(3)
    for _ in range(12):
        x, y = rng.integers(26, n - 26, 2)
        src[y, x] += rng.random(3).astype(np.float32) * 5 + 0.5
        coc[y, x] = float(rng.integers(3, 13))
    o = P.apply_prescription_bokeh(src, coc, rx)
    assert np.isfinite(o).all() and o.min() >= 0.0
    assert o.sum(dtype=np.float64) == pytest.approx(src.sum(dtype=np.float64), rel=2e-3)


def test_deterministic_float64_and_input_immutable():
    rx = mk(cat=0.7, blades=7, swirl=0.5, squeeze=2.0)
    img = hdr_scene(36, 48).astype(np.float64)
    coc = np.tile(np.linspace(0.0, 9.0, 48), (36, 1))
    keep, ckeep = img.copy(), coc.copy()
    a = P.apply_prescription_bokeh(img, coc, rx)
    b = P.apply_prescription_bokeh(img, coc, rx)
    assert a.dtype == np.float64 and np.array_equal(a, b)
    assert np.array_equal(img, keep) and np.array_equal(coc, ckeep)
    assert np.isfinite(a).all()


def test_bokeh_validation():
    img = smooth_img(20, 30)
    coc = np.full((20, 30), 4.0, np.float32)
    rx = mk()
    with pytest.raises(ValueError):
        P.apply_prescription_bokeh(img, coc[:, :-1], rx)
    bad = coc.copy()
    bad[0, 0] = -1.0
    with pytest.raises(ValueError):
        P.apply_prescription_bokeh(img, bad, rx)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        P.apply_prescription_bokeh(img, bad, rx)
    with pytest.raises(TypeError):
        P.apply_prescription_bokeh(img, coc.astype(bool), rx)
    with pytest.raises(TypeError):
        P.apply_prescription_bokeh((img * 255).astype(np.uint8), coc, rx)
    with pytest.raises(ValueError):
        P.apply_prescription_bokeh(img[..., 0], coc, rx)
    for bad_rx in (None, optics.OpticsSpec(), {"squeeze": 2.0}):
        with pytest.raises(TypeError):
            P.apply_prescription_bokeh(img, coc, bad_rx)
