"""LNS-02 apply_dof: 항등, 볼록성, 상수 불변(테두리·깊이 불연속 포함), 에너지·스폿 크기, 원판 오라클,
Defocus Pyramid 일치, 칼선 방지(선명 전경/녹은 배경, 녹은 전경/선명 배경), 결정론, 검증, 속도."""
import time
import warnings

import numpy as np
import pytest

from buc.core import contracts as C
from buc.lens import dof
from lns02_helpers import (H, LENS, W, Z_FAR, coc_ref, conv_ref, focus_for_coc, grad_energy, second_moment, stripes,
                           tex)


def run(rgb, depth, focus, opts=None, **lens):
    kw = dict(LENS)
    kw.update(lens)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return dof.apply_dof(rgb, depth, focus_m=focus, opts=opts, **kw)


def plane(z=Z_FAR):
    return np.full((H, W), z, np.float32)


def test_result_types_and_coc_aov():
    rgb = tex()
    depth = np.where(np.arange(W)[None, :] < W // 2, 1.2, 6.0).astype(np.float32).repeat(H, 0)
    r = run(rgb, depth, 2.0)
    assert isinstance(r, dof.DofResult)
    assert r.rgb.shape == (H, W, 3) and r.rgb.dtype == np.float32
    assert r.coc_px.shape == (H, W) and r.coc_px.dtype == np.float32
    np.testing.assert_allclose(r.coc_px, np.minimum(coc_ref(depth, 2.0), 64.0), rtol=1e-4, atol=1e-4)
    assert np.isfinite(r.rgb).all()
    r2 = run(rgb, depth, 0.3)  # CoC 상한 클립
    assert float(r2.coc_px.max()) == pytest.approx(64.0, abs=1e-4)


def test_in_focus_is_identity():
    rgb = tex()
    r = run(rgb, plane(2.5), 2.5)
    np.testing.assert_array_equal(r.rgb, rgb)
    # 모든 CoC <= in_focus_px 이면 비트 동일
    zs = np.full((H, W), 2.5, np.float32)
    zs[:, :10] = 2.501
    r = run(rgb, zs, 2.5)
    assert float(r.coc_px.max()) <= 0.5
    np.testing.assert_array_equal(r.rgb, rgb)


def test_inputs_not_mutated_and_deterministic():
    rgb = tex()
    depth = (1.0 + 4.0 * np.random.default_rng(3).random((H, W))).astype(np.float32)
    a, d = rgb.copy(), depth.copy()
    r1 = run(rgb, depth, 1.7)
    r2 = run(rgb, depth, 1.7)
    np.testing.assert_array_equal(rgb, a)
    np.testing.assert_array_equal(depth, d)
    assert r1.rgb.tobytes() == r2.rgb.tobytes()
    assert r1.coc_px.tobytes() == r2.coc_px.tobytes()


def test_validation():
    rgb, depth = tex(), plane(3.0)
    bad = [
        (rgb[..., :2], depth), (rgb, depth[:, :-1]), (rgb[0], depth),
        (np.where(np.arange(W)[None, :, None] == 3, np.nan, rgb), depth),
        (rgb, np.where(np.arange(W)[None, :] == 3, np.nan, depth)),
        (rgb, np.where(np.arange(W)[None, :] == 3, 0.0, depth)),
        (rgb, np.where(np.arange(W)[None, :] == 3, -2.0, depth)),
    ]
    for a, d in bad:
        with pytest.raises(ValueError):
            run(a, d, 2.0)
    with pytest.raises(ValueError):
        run(rgb, depth, 0.05)  # focus <= f
    with pytest.raises(ValueError):
        run(rgb, depth, 2.0, f_stop=0.0)
    with pytest.raises(TypeError):
        dof.apply_dof(rgb, depth, 85.0, 1.4, 2.0, 36.0)


def test_infinite_depth_background():
    rgb = tex()
    depth = plane(np.inf)
    depth[:, :40] = 2.0
    r = run(rgb, depth, 2.0)
    assert np.isfinite(r.rgb).all() and np.isfinite(r.coc_px).all()
    assert float(r.coc_px[0, -1]) == pytest.approx(float(coc_ref(np.inf, 2.0)), rel=1e-4)


def test_convex_hull_and_constant_invariance():
    rng = np.random.default_rng(7)
    rgb = tex()
    # 거친 깊이: 블록 단위 계단(근·원 모두 흐림) + 무한대
    blocks = rng.choice([0.5, 0.9, 2.0, 5.0, 40.0, np.inf], size=(H // 8, W // 8))
    depth = np.kron(blocks, np.ones((8, 8))).astype(np.float32)
    r = run(rgb, depth, 2.0)
    lo, hi = rgb.reshape(-1, 3).min(0), rgb.reshape(-1, 3).max(0)
    assert (r.rgb >= lo - 1e-4).all() and (r.rgb <= hi + 1e-4).all()
    const = np.empty_like(rgb)
    const[...] = np.array([0.3, 0.55, 0.8], np.float32)
    rc = run(const, depth, 2.0)
    # 상수 색은 깊이 불연속·테두리 어디서도 그대로(어두운/밝은 테두리·헤일로 없음)
    assert np.abs(rc.rgb - const).max() < 1e-3


@pytest.mark.parametrize("c", [5.0, 12.0, 24.0])
def test_impulse_energy_and_spot_size(c):
    focus = focus_for_coc(c)
    img = np.zeros((H, W, 3), np.float32)
    img[H // 2, W // 2, :] = 1.0
    r = run(img, plane(), focus)
    assert float(r.coc_px[0, 0]) == pytest.approx(c, rel=1e-3)
    out = r.rgb[..., 0].astype(np.float64)
    assert out.sum() == pytest.approx(1.0, rel=2e-3)  # 에너지 보존
    m2, cx, cy = second_moment(out)
    assert abs(cx - W // 2) < 0.25 and abs(cy - H // 2) < 0.25
    assert m2 == pytest.approx(c * c / 8.0, rel=0.18)  # 원판 지름 c 의 E[r^2] = c^2/8
    # 지지 영역: 반경 c/2 + 0.1c + 1.5 px 밖으로 새는 에너지 < 3% (피라미드 보간 꼬리 허용)
    y, x = np.mgrid[0:H, 0:W]
    rr = np.hypot(x - W // 2, y - H // 2)
    assert out[rr > c / 2 + 0.1 * c + 1.5].sum() < 0.03 * out.sum()
    assert out.min() >= -1e-6


@pytest.mark.parametrize("c", [6.0, 14.0, 30.0])
def test_textured_plane_matches_disk_oracle(c):
    rgb = tex()
    focus = focus_for_coc(c)
    r = run(rgb, plane(), focus)
    ref = conv_ref(rgb.astype(np.float64), c)
    m = int(np.ceil(c / 2)) + 2
    err = np.abs(r.rgb[m:-m, m:-m] - ref[m:-m, m:-m]).mean()
    assert err < 0.02, err
    # 블러가 실제로 일어났다(원본과 충분히 다름)
    assert np.abs(r.rgb[m:-m, m:-m] - rgb[m:-m, m:-m]).mean() > 3 * err


@pytest.mark.parametrize("c", [20.0, 40.0])
def test_defocus_pyramid_matches_full_res(c):
    rgb = tex()
    focus = focus_for_coc(c)
    a = run(rgb, plane(), focus, opts=dof.DofOpts(pyramid=True)).rgb
    b = run(rgb, plane(), focus, opts=dof.DofOpts(pyramid=False)).rgb
    m = int(np.ceil(c / 2))
    assert np.abs(a[m:-m, m:-m] - b[m:-m, m:-m]).mean() < 0.02
    mm = m // 2
    assert np.abs(a[mm:-mm, mm:-mm] - b[mm:-mm, mm:-mm]).max() < 0.06


def _edge_scene(fg_depth, bg_depth, bg_img):
    rgb = bg_img.copy()
    rgb[:, :W // 2] = np.array([1.0, 0.0, 0.0], np.float32)
    depth = np.full((H, W), bg_depth, np.float32)
    depth[:, :W // 2] = fg_depth
    return rgb, depth


def test_sharp_foreground_over_melted_background_no_halo():
    """선명한 전경(초점면) 앞, 녹은 배경: 전경은 그대로, 배경으로 전경 색이 새지 않고 어두운 테두리도 없다."""
    focus = 1.5
    green = np.zeros((H, W, 3), np.float32)
    green[..., 1] = 1.0
    rgb, depth = _edge_scene(focus, 20.0, green)
    c_bg = float(coc_ref(20.0, focus))
    assert c_bg > 10
    r = run(rgb, depth, focus).rgb
    e = W // 2
    np.testing.assert_allclose(r[:, :e - 1], rgb[:, :e - 1], atol=0.02)  # 전경 내부 불변(가장자리 1 px 제외)
    bg = r[:, e + 1:]
    assert bg[..., 0].max() < 0.1   # 전경 색 번짐 없음
    assert bg[..., 1].min() > 0.9   # 배경 정규화(어두운 링 없음)


def test_melted_foreground_over_sharp_background_no_knife_edge():
    """녹은 전경(가까움) 뒤 선명 배경: 전경 실루엣이 칼선처럼 잘리지 않고 CoC 폭으로 부드럽게 넘어간다."""
    focus = 3.0
    bg = stripes()
    bg[..., 0] *= 0.0  # 배경 빨강 채널 0 -> 빨강은 전경 몫만
    rgb, depth = _edge_scene(0.8, focus, bg)
    c_fg = float(coc_ref(0.8, focus))
    assert 12 < c_fg < 30
    r = run(rgb, depth, focus).rgb
    e = W // 2  # 실루엣은 x=e-0.5
    prof = r[:, :, 0].mean(0).astype(np.float64)
    lo, hi = int(e - c_fg), int(e + c_fg)
    seg = prof[lo:hi + 1]
    assert np.all(np.diff(seg) <= 2e-3), "전경 알파 프로파일이 단조가 아니다"
    jumps = np.abs(np.diff(seg))
    assert jumps.max() < 2.5 / c_fg + 0.02, "실루엣에서 칼선(계단)이 있다"
    x = np.arange(len(prof), dtype=np.float64)
    i90 = x[prof >= 0.9].max()
    i10 = x[prof >= 0.1].max()
    width = i10 - i90
    assert 0.4 * c_fg <= width <= 1.3 * c_fg, (width, c_fg)
    # 전경 블러가 실루엣 밖 선명 배경 위로 번진다(부분 가림)
    assert prof[int(round(e + c_fg / 4))] > 0.08
    # 실루엣에서 CoC 이상 떨어진 배경은 선명 그대로
    far = int(np.ceil(e + c_fg / 2 + 3))
    np.testing.assert_allclose(r[:, far:], rgb[:, far:], atol=0.02)


def test_rack_across_pyramid_levels_has_no_popping():
    """CoC 를 4 -> 36 px 로 잘게 쓸어도 프레임 간 변화에 튐이 없다(8·16·32 px 단계 경계 포함)."""
    rgb = tex()
    cs = np.arange(4.0, 36.01, 0.5)
    outs, m2s = [], []
    imp = np.zeros((H, W, 3), np.float32)
    imp[H // 2, W // 2] = 1.0
    for c in cs:
        f = focus_for_coc(c)
        outs.append(run(rgb, plane(), f).rgb.astype(np.float64))
        m2s.append(second_moment(run(imp, plane(), f).rgb[..., 0])[0])
    m = 20
    d = np.array([np.abs(outs[i + 1][m:-m, m:-m] - outs[i][m:-m, m:-m]).mean() for i in range(len(outs) - 1)])
    for i in range(1, len(d) - 1):
        assert d[i] <= 2.0 * max(d[i - 1], d[i + 1]) + 3e-4, (cs[i], d[i - 1:i + 2])
    assert np.all(np.diff(m2s) > 0), "스폿 크기가 CoC 에 단조 증가하지 않는다"


def test_two_planes_rack_sharpness_ordering():
    rgb = tex()
    depth = np.full((H, W), 8.0, np.float32)
    depth[:, :W // 2] = 1.0
    sched = dof.rack_schedule(9, 24.0, 1.0, 8.0)
    left = np.zeros((H, W), bool)
    left[8:-8, 8:W // 2 - 16] = True
    right = np.zeros((H, W), bool)
    right[8:-8, W // 2 + 16:-8] = True
    el, er = [], []
    for f in sched:
        out = run(rgb, depth, float(f)).rgb
        el.append(grad_energy(out, left))
        er.append(grad_energy(out, right))
    assert np.all(np.diff(el) < 1e-6) and np.all(np.diff(er) > -1e-6)
    assert el[0] == pytest.approx(grad_energy(rgb, left), rel=1e-3)
    assert er[-1] == pytest.approx(grad_energy(rgb, right), rel=1e-3)


def test_apply_dof_theta_equals_apply_dof():
    th = C.Theta(focal_mm=85.0, f_stop=1.4, sensor_width_mm=36.0, focus_m=2.0, extra={"blades": 7})
    rgb = tex()
    depth = np.where(np.arange(W)[None, :] < 50, 0.9, 7.0).astype(np.float32).repeat(H, 0)
    a = dof.apply_dof_theta(rgb, depth, th)
    b = dof.apply_dof(rgb, depth, **LENS, focus_m=2.0, opts=dof.DofOpts(blades=7))
    assert a.rgb.tobytes() == b.rgb.tobytes()
    c = dof.apply_dof_theta(rgb, depth, th, focus_m=4.0, opts=dof.DofOpts())
    d = dof.apply_dof(rgb, depth, **LENS, focus_m=4.0, opts=dof.DofOpts())
    assert c.rgb.tobytes() == d.rgb.tobytes()
    e = dof.apply_dof_theta(rgb, depth, C.Theta(focal_mm=85.0, f_stop=1.4, sensor_width_mm=36.0, focus_m=2.0))
    f = dof.apply_dof(rgb, depth, **LENS, focus_m=2.0, opts=dof.DofOpts(blades=0))
    assert e.rgb.tobytes() == f.rgb.tobytes()


def test_speed_320x180():
    w, h = 320, 180
    rgb = tex(w, h)
    xx = np.arange(w)[None, :].repeat(h, 0)
    depth = np.where(xx < 100, 0.7, np.where(xx < 220, 2.0, 60.0)).astype(np.float32)
    lens = dict(focal_mm=85.0, f_stop=1.4, sensor_width_mm=36.0)
    c = dof.coc_px(depth, **lens, focus_m=2.0, width_px=w)
    assert c.max() > 35
    t = time.perf_counter()
    r = dof.apply_dof(rgb, depth, **lens, focus_m=2.0)
    dt = time.perf_counter() - t
    assert np.isfinite(r.rgb).all()
    assert dt < 6.0, dt
