"""WLD-03 시각 대리 지표(RDR-02 render_ldi): 고무 시트 없음 · 가림 해제 채움 · 관측 영역 일치 · halo · RD-18 합류.

장면: 96x64, fx=100, 전경 사각형 2 m(행 20..43, 열 40..63), 배경 6 m. 카메라를 +x 로 0.3 m 옮기면
전경은 -15 px, 배경은 -5 px 이동한다. 새 뷰에서 전경은 열 25..48, 가림 해제 띠는 열 49..58(행 20..43).
"""
import buc.world.depth_world as dw
import numpy as np
from wld03_helpers import (
    BLUE,
    RED,
    Y0,
    Y1,
    Z_BG,
    Z_FG,
    H,
    W,
    bg_gradient,
    box_scene,
    cheb_dist_to_rect_boundary,
    fg_mask,
    halo_scene,
    make_read,
    obs_of,
    shifted_c2w,
)

from buc.core import contracts as C
from buc.render import quads

FG_NEW = (Y0, Y1, 25, 49)     # 새 뷰 전경 [y0,y1)x[x0,x1)
STRIP = (Y0, Y1, 49, 59)      # 가림 해제 띠
MAXCOL = 89                    # 오른쪽 프레임 밖에서 온 화소 제외


def _render(world):
    obs = world.obs
    return quads.render_ldi(world, c2w=shifted_c2w(), K=obs.K, width=W, height=H)


def _gt_world():
    obs = obs_of()
    bg = bg_gradient()
    rgba_bg = np.concatenate([bg, np.ones((H, W, 1))], -1)
    fgm = fg_mask()
    rgba_fg = np.zeros((H, W, 4))
    rgba_fg[fgm, :3] = RED
    rgba_fg[fgm, 3] = 1.0
    layers = (C.LDILayer(rgba=rgba_fg, depth=np.full((H, W), Z_FG)),
              C.LDILayer(rgba=rgba_bg, depth=np.full((H, W), Z_BG)))
    return C.World(obs=obs, layers=layers, theta_in=C.Theta(aspect=W / H), level="L1", read_hash="g" * 64,
                   prov=C.Prov())


def _masks():
    v, u = np.mgrid[0:H, 0:W]
    dfg = cheb_dist_to_rect_boundary(*FG_NEW)
    band = dfg <= 2  # 새 뷰 전경 실루엣 ±2 px
    sy0, sy1, sx0, sx1 = STRIP
    strip_in = (v >= sy0 + 2) & (v < sy1 - 2) & (u >= sx0 + 2) & (u < sx1 - 1)
    strip_any = (v >= sy0 - 2) & (v < sy1 + 2) & (u >= sx0 - 2) & (u < sx1 + 2)
    keep = (u <= MAXCOL) & ~band
    return keep, strip_in, strip_any, dfg


def _build(ext_px=16.0, matte=None, scene=None):
    img, d = box_scene() if scene is None else scene
    return dw.build_depth_world(make_read(img, d), cfg=dw.DepthWorldConfig(ext_px=ext_px, tsq_stride=4),
                                matte=matte)


def test_no_rubber_sheet_depths():
    """실루엣 띠 밖에서 전경(2 m)·배경(6 m) 사이 중간 깊이 화소 ≤ 0.5 %."""
    res = _build()
    r = _render(res.world)
    keep, _, _, _ = _masks()
    vis = keep & (r.alpha > 0.5)
    dep = r.depth[vis].astype(np.float64)
    near_fg = np.abs(dep - Z_FG) <= 0.03 * Z_FG
    near_bg = np.abs(dep - Z_BG) <= 0.03 * Z_BG
    frac_mid = 1.0 - float(np.mean(near_fg | near_bg))
    assert vis.sum() > 0.7 * keep.sum()
    assert frac_mid <= 0.005, frac_mid


def test_disocclusion_filled_by_hidden_layer():
    res = _build(ext_px=16.0)
    r = _render(res.world)
    _, strip_in, _, _ = _masks()
    assert strip_in.sum() >= 100
    a = r.alpha[strip_in]
    dep = r.depth[strip_in].astype(np.float64)
    assert np.mean(a >= 0.99) >= 0.95
    assert np.mean(np.abs(dep - Z_BG) <= 0.03 * Z_BG) >= 0.95
    # 띠는 전경색으로 늘어나지 않는다(빨강 아님)
    red = (r.rgb[..., 0] > 0.6) & (r.rgb[..., 1] < 0.3)
    assert np.mean(red[strip_in]) <= 0.02


def test_without_hidden_layer_disocclusion_is_hole():
    res = _build(ext_px=0.0)
    r = _render(res.world)
    _, strip_in, _, _ = _masks()
    assert np.mean(r.alpha[strip_in] < 0.5) >= 0.80


def test_observed_regions_match_ground_truth():
    res = _build(ext_px=16.0)
    r = _render(res.world)
    g = _render(_gt_world())
    keep, _, strip_any, _ = _masks()
    m = keep & ~strip_any & (g.alpha > 0.99)
    assert m.sum() > 0.6 * H * W
    err = np.abs(r.rgb.astype(np.float64) - g.rgb.astype(np.float64)).max(-1)[m]
    assert float(err.mean()) <= 0.01, float(err.mean())
    assert float(np.quantile(err, 0.99)) <= 0.05
    assert np.mean(np.abs(r.depth[m] - g.depth[m]) <= 0.03 * g.depth[m]) >= 0.99


def _purple_far_from_fg(rgb, dfg):
    purple = (rgb[..., 0] > 0.25) & (rgb[..., 2] > 0.25)
    _v, u = np.mgrid[0:H, 0:W]
    region = (dfg >= 3) & (u <= MAXCOL)
    return int((purple & region).sum())


def test_halo_removed_by_matte():
    """혼합 링(보라)이 배경 깊이에 붙은 장면: matte 없으면 새 뷰에 보라 잔상, matte 있으면 없다."""
    img, d, m = halo_scene()
    _, _, _, dfg = _masks()
    plain = dw.build_depth_world(make_read(img, d), cfg=dw.DepthWorldConfig(ext_px=16.0, tsq_stride=4))
    matted = dw.build_depth_world(make_read(img, d), cfg=dw.DepthWorldConfig(ext_px=16.0, tsq_stride=4), matte=m)
    n_plain = _purple_far_from_fg(_render(plain.world).rgb, dfg)
    n_matte = _purple_far_from_fg(_render(matted.world).rgb, dfg)
    assert n_plain >= 15, n_plain
    assert n_matte <= 2, n_matte
    # matte 쪽 가림 해제 띠는 배경 파랑
    r = _render(matted.world)
    _, strip_in, _, _ = _masks()
    assert np.allclose(r.rgb[strip_in].mean(0), BLUE, atol=0.03)


def test_rd18_edge_unmix_joins():
    """실제 RD-18 UnmixResult 를 matte 로 넘기면 관측 시점 합성이 입력을 재구성한다."""
    from buc.read import edge_unmix as eu
    h, w = 40, 40
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (u - 19.5) ** 2 + (v - 19.5) ** 2
    alpha = np.clip(9.5 - np.sqrt(r2), 0.0, 1.0)  # 부드러운 원판 경계
    F = np.array([0.85, 0.3, 0.2])
    B = np.array([0.15, 0.35, 0.7])
    img = (alpha[..., None] * F + (1 - alpha[..., None]) * B).astype(np.float32)
    depth = np.where(alpha >= 0.5, 2.0, 5.0).astype(np.float32)
    un = eu.edge_unmix(img, depth)
    res = dw.build_depth_world(make_read(img, depth, obs=obs_of(w, h, f=40.0)),
                               cfg=dw.DepthWorldConfig(tsq_stride=2), matte=un)
    assert res.matted.sum() > 0
    assert not (res.matted & ~un.band).any()
    comp = np.zeros((h, w, 3))
    trans = np.ones((h, w, 1))
    for ly in res.world.layers:
        a = ly.rgba[..., 3:4].astype(np.float64)
        comp += ly.rgba[..., :3] * a * trans
        trans *= 1 - a
    tol = float(un.recon_err) + 2e-3
    assert np.abs(comp - img).max() <= tol
    l0, l1 = res.world.layers
    both = (l0.rgba[..., 3] > 1e-3) & (l1.rgba[..., 3] > 1e-3)
    assert np.all(l0.depth[both] < l1.depth[both])
    C.check_stage_io("S2", {"read": make_read(img, depth, obs=obs_of(w, h, f=40.0))}, {"world": res.world})
