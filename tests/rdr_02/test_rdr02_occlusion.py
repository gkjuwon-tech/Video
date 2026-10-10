"""RDR-02 가림·시차·구멍: 층 사이 시차, 층 안 깊이 불연속(고무 시트 금지), 카메라 뒤, 확대 시 핀홀 없음."""
import warnings

import numpy as np
import pytest

from buc.render.quads import RenderOpts, render_ldi
from rdr02_helpers import (FX, H, W, K_default, const_layer, edge_left, edge_right, layer, make_world, smooth_tex,
                           translate)

I4 = np.eye(4)


def _r(world, c2w=I4, **kw):
    return render_ldi(world, c2w=c2w, K=K_default(), width=W, height=H, **kw)


def test_two_layer_parallax_edges_and_disocclusion():
    # 앞(z=2) 빨강 카드 열 20..49, 뒤(z=6) 파랑 램프. 카메라 +x 0.1 m -> 앞은 fx*0.1/2, 뒤는 fx*0.1/6 만큼 왼쪽으로
    front = np.zeros((H, W, 4), np.float32)
    front[:, 20:50] = (1.0, 0.0, 0.0, 1.0)
    ramp = np.zeros((H, W, 4), np.float32)
    ramp[..., 2] = np.linspace(0.1, 0.9, W)[None, :]
    ramp[..., 3] = 1.0
    out = _r(make_world([layer(front, 2.0), layer(ramp, 6.0)]), translate(I4, (0.1, 0, 0)))
    s_f, s_b = FX() * 0.1 / 2.0, FX() * 0.1 / 6.0
    row = out.rgb[32, :, 0].astype(np.float64)                    # R 채널 = 앞 카드 알파
    assert edge_left(row, 35) == pytest.approx(20 - 0.5 - s_f, abs=0.3)
    assert edge_right(row, 35) == pytest.approx(49 + 0.5 - s_f, abs=0.3)
    # 드러난 배경은 뒤 램프를 s_b 만큼 민 값: B(u) = ramp(u + s_b) (선형 보간)
    cols = np.array([4, 8, 12, 60, 70, 80])                       # 앞 카드 밖, 뒤 램프 안쪽
    exp = 0.1 + 0.8 * (cols + s_b) / (W - 1)
    np.testing.assert_allclose(out.rgb[32, cols, 2], exp, atol=0.01)
    np.testing.assert_allclose(out.alpha[8:H - 8, 6:W - 10], 1.0, atol=1e-3)
    lid = out.layer_id[32]
    assert (lid[26:41] == 0).all() and (lid[cols] == 1).all()
    np.testing.assert_allclose(out.depth[32, 30:40], 2.0, rtol=1e-3)
    np.testing.assert_allclose(out.depth[32, cols], 6.0, rtol=1e-3)


def _split_depth_layer():
    rgba = np.zeros((H, W, 4), np.float32)
    rgba[:, :48] = (1.0, 0.0, 0.0, 1.0)                           # 왼쪽 반: 가까움 z=2
    rgba[:, 48:] = (0.0, 0.0, 1.0, 1.0)                           # 오른쪽 반: 멀음 z=5
    depth = np.where(np.arange(W)[None, :] < 48, 2.0, 5.0) * np.ones((H, 1))
    return make_world([layer(rgba, depth.astype(np.float32))])


def test_depth_discontinuity_inside_one_layer_tears_instead_of_stretching():
    # 카메라 +x 0.2 m: 가까운 쪽이 더 많이 왼쪽으로 -> 사이가 벌어진다(탈가림). 고무 시트라면 보라색이 채운다.
    out = _r(_split_depth_layer(), translate(I4, (0.2, 0, 0)))
    s_n, s_f = FX() * 0.2 / 2.0, FX() * 0.2 / 5.0
    gap_lo, gap_hi = 47.5 - s_n, 47.5 - s_f                       # 구멍 구간 [39.2, 44.2]
    cols = [c for c in range(W) if gap_lo + 1.0 <= c <= gap_hi - 1.0]
    assert len(cols) >= 2
    assert out.alpha[32, cols].max() < 0.1, out.alpha[32, cols]
    # 구멍 양쪽은 각자의 색(섞이지 않음)
    assert out.rgb[32, 10, 0] > 0.98 and out.rgb[32, 10, 2] < 0.02
    assert out.rgb[32, 70, 2] > 0.98 and out.rgb[32, 70, 0] < 0.02


def test_depth_discontinuity_inside_one_layer_nearer_side_occludes_when_overlapping():
    # 카메라 -x 0.2 m: 가까운 쪽이 오른쪽으로 더 밀려 먼 쪽을 덮는다. 겹침 구간은 빨강(가까움), 파랑 섞임 없음.
    out = _r(_split_depth_layer(), translate(I4, (-0.2, 0, 0)))
    s_n, s_f = FX() * 0.2 / 2.0, FX() * 0.2 / 5.0
    ov_lo, ov_hi = 47.5 + s_f, 47.5 + s_n                         # 겹침 [50.8, 55.8]
    cols = [c for c in range(W) if ov_lo + 1.0 <= c <= ov_hi - 1.0]
    assert len(cols) >= 2
    assert (out.rgb[32, cols, 0] > 0.97).all() and (out.rgb[32, cols, 2] < 0.03).all()
    assert (out.alpha[32, cols] > 0.97).all()
    np.testing.assert_allclose(out.depth[32, cols], 2.0, rtol=0.05)          # 가까운 쪽 깊이
    assert out.rgb[32, 62, 2] > 0.97 and out.rgb[32, 62, 0] < 0.03          # 겹침 밖의 먼 쪽은 파랑 그대로


def test_card_behind_camera_is_skipped_without_warnings_or_nans():
    world = make_world([const_layer((1, 0, 0), 1.0, 3.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _r(world, translate(I4, (0, 0, 4.0)), opts=RenderOpts(bg=(0.2, 0.3, 0.4)))     # 카드(z=3)를 지나쳤다
    np.testing.assert_allclose(out.rgb, np.broadcast_to([0.2, 0.3, 0.4], out.rgb.shape), atol=1e-6)
    assert out.alpha.max() < 1e-3 and np.isposinf(out.depth).all() and (out.layer_id == -1).all()


def test_card_crossing_the_near_plane_does_not_blow_up():
    # 카드가 카메라 평면 바로 앞(0.02 m): 거대한 풋프린트가 생기지만 max_footprint_px 로 제한되고 유한해야 한다
    world = make_world([const_layer((0.5, 0.5, 0.5), 1.0, 1.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _r(world, translate(I4, (0, 0, 0.98)), opts=RenderOpts(max_footprint_px=6.0))
    assert np.isfinite(out.rgb).all() and 0 <= out.alpha.min() and out.alpha.max() <= 1 + 1e-6
    assert out.rgb.max() <= 0.5 + 1e-5


def test_magnified_card_has_no_pinholes():
    world = make_world([layer(smooth_tex(), 3.0)])
    out = _r(world, translate(I4, (0.0, 0.0, 1.5)))               # 2배 확대
    inner = (slice(8, H - 8), slice(12, W - 12))
    assert out.alpha[inner].min() > 0.99
    assert np.isfinite(out.depth[inner]).all()
