"""RDR-02 합성: 직선 알파 over 연산, 배경, 깊이·층 AOV, 프린지(테두리 오염) 금지."""
import numpy as np
import pytest

from buc.render.quads import RenderOpts, render_ldi
from rdr02_helpers import FX, H, W, K_default, const_layer, layer, make_world, translate

I4 = np.eye(4)
INT = (slice(2, H - 2), slice(2, W - 2))


def _r(world, c2w=I4, **kw):
    return render_ldi(world, c2w=c2w, K=K_default(), width=W, height=H, **kw)


@pytest.mark.parametrize("a", [1.0, 0.5, 0.25])
def test_single_constant_layer_over_background_exact(a):
    bg = (0.2, 0.4, 0.6)
    c = np.array([0.3, 0.6, 0.9])
    out = _r(make_world([const_layer(c, a, 2.0)]), opts=RenderOpts(bg=bg))
    exp = a * c + (1 - a) * np.array(bg)
    np.testing.assert_allclose(out.rgb[INT], np.broadcast_to(exp, out.rgb[INT].shape), atol=1e-5)
    np.testing.assert_allclose(out.alpha[INT], a, atol=1e-5)


def test_empty_coverage_returns_background_inf_depth_and_no_layer():
    # 알파 0 층만 있는 월드: 배경, alpha 0, depth +inf, layer_id -1
    out = _r(make_world([const_layer((1, 0, 0), 0.0, 2.0)]), opts=RenderOpts(bg=(0.1, 0.2, 0.3)))
    np.testing.assert_allclose(out.rgb, np.broadcast_to([0.1, 0.2, 0.3], out.rgb.shape), atol=1e-6)
    assert out.alpha.max() < 1e-3
    assert np.isposinf(out.depth).all()
    assert (out.layer_id == -1).all()


def test_two_layers_over_operator_depth_and_layer_id():
    bg = np.array([0.1, 0.1, 0.1])
    front = const_layer((1.0, 0.0, 0.0), 0.4, 2.0)
    back = const_layer((0.0, 1.0, 0.0), 0.5, 4.0)
    out = _r(make_world([front, back]), opts=RenderOpts(bg=tuple(bg)))
    w1, w2 = 0.4, 0.6 * 0.5
    exp = w1 * np.array([1, 0, 0]) + w2 * np.array([0, 1, 0]) + (1 - w1 - w2) * bg
    np.testing.assert_allclose(out.rgb[INT], np.broadcast_to(exp, out.rgb[INT].shape), atol=1e-5)
    np.testing.assert_allclose(out.alpha[INT], w1 + w2, atol=1e-5)
    z_exp = (w1 * 2.0 + w2 * 4.0) / (w1 + w2)                     # 투과율 가중 기대 깊이
    np.testing.assert_allclose(out.depth[INT], z_exp, rtol=1e-4)
    assert (out.layer_id[INT] == 0).all()                         # w1 > w2 -> 앞 층


def test_layer_id_picks_the_layer_with_the_largest_transmittance_weight():
    out = _r(make_world([const_layer((1, 0, 0), 0.2, 2.0), const_layer((0, 1, 0), 1.0, 4.0)]))
    assert (out.layer_id[INT] == 1).all()                         # 0.2 < 0.8
    out = _r(make_world([const_layer((1, 0, 0), 0.6, 2.0), const_layer((0, 1, 0), 1.0, 4.0)]))
    assert (out.layer_id[INT] == 0).all()                         # 0.6 > 0.4


def test_no_dark_or_bright_fringe_on_soft_edges_with_garbage_rgb_in_transparent_area():
    # 흰 카드(직선 알파) 바깥의 rgb 는 쓰레기(마젠타). 필터가 투명 텍셀의 rgb 를 섞으면 G 가 줄어든다.
    rgba = np.zeros((H, W, 4), np.float32)
    rgba[...] = (1.0, 0.0, 1.0, 0.0)
    rgba[16:48, 30:66] = (1.0, 1.0, 1.0, 1.0)
    out = _r(make_world([layer(rgba, 3.0)]), translate(I4, (0.5 * 3.0 / FX(), 0.0, 0.0)))  # 0.5 px 이동
    a = out.alpha.astype(np.float64)
    assert ((a > 0.05) & (a < 0.95)).any(), "소프트 에지가 만들어지지 않았다"
    np.testing.assert_allclose(out.rgb[..., 1], a, atol=2e-4)     # 흰 카드: G == 알파 (오염 없음)
    np.testing.assert_allclose(out.rgb[..., 0], a, atol=2e-4)
    np.testing.assert_allclose(out.rgb[..., 2], a, atol=2e-4)


def test_edge_between_two_cards_is_convex_mix_no_halo():
    # 앞: 빨강 카드(이진 알파), 뒤: 초록 전면 층. 가장자리 화소 = A*빨강 + (1-A)*초록, 파랑 0 (어두운 테두리 금지)
    front = np.zeros((H, W, 4), np.float32)
    front[...] = (0.0, 0.0, 1.0, 0.0)
    front[:, 24:56] = (1.0, 0.0, 0.0, 1.0)
    back = np.zeros((H, W, 4), np.float32)
    back[...] = (0.0, 1.0, 0.0, 1.0)
    out = _r(make_world([layer(front, 2.0), layer(back, 5.0)]), translate(I4, (0.07, 0.0, 0.0)))
    sl = (slice(4, H - 4), slice(12, W - 12))
    r, g, b = (out.rgb[sl][..., i].astype(np.float64) for i in range(3))
    assert np.abs(b).max() < 2e-4, "파란 쓰레기 rgb 가 섞였다"
    np.testing.assert_allclose(r + g, 1.0, atol=3e-4)             # 볼록 조합: 어두운/밝은 테두리 없음
    assert ((r > 0.05) & (r < 0.95)).any()
    np.testing.assert_allclose(out.alpha[sl], 1.0, atol=1e-4)


def test_output_is_within_input_range_and_finite():
    from rdr02_helpers import smooth_tex, yaw
    rng = np.random.default_rng(5)
    a = smooth_tex(alpha=0.6)
    b = smooth_tex()
    b[..., :3] = b[..., :3][::-1]
    world = make_world([layer(a, 2.0), layer(b, 4.0 + rng.random((H, W)).astype(np.float32))])
    out = _r(world, yaw(translate(I4, (0.1, -0.05, 0.4)), 7.0), opts=RenderOpts(bg=(0.0, 0.0, 0.0)))
    assert out.rgb.dtype == np.float32 and out.alpha.dtype == np.float32 and out.depth.dtype == np.float32
    assert out.layer_id.dtype == np.int16
    assert out.rgb.shape == (H, W, 3) and out.alpha.shape == (H, W) and out.depth.shape == (H, W)
    for x in (out.rgb, out.alpha):
        assert np.isfinite(x).all()
    assert not np.isnan(out.depth).any()
    assert out.rgb.min() >= -1e-6 and out.rgb.max() <= max(a[..., :3].max(), b[..., :3].max()) + 1e-5
    assert out.alpha.min() >= 0 and out.alpha.max() <= 1 + 1e-6
    fin = np.isfinite(out.depth)
    assert (out.depth[fin] > 0).all()
    assert (np.isfinite(out.depth) == (out.alpha >= 1e-3)).mean() > 0.995   # 깊이는 알파 >= 1e-3 에서만 유한
    assert (out.layer_id >= -1).all() and (out.layer_id <= 1).all()
