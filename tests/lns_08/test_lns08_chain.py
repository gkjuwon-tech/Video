"""LNS-08 순방향 합성기: WB 수식, 체인 순서(LNS-04 공개 조각의 합성과 일치), Frames 어댑터."""
import numpy as np
import pytest

from buc.core import color
from buc.core import contracts as C
from buc.lens import chain
from buc.lens import sensor_film as sf
from buc.lens import theta as T
from lns08_helpers import WB_MIRED_STOPS, look_by_pieces, scene_hdr, scene_mid, theta, wb_gains_ref


# ------------------------------------------------------------------------------------------------ WB
@pytest.mark.parametrize("k,t", [(6500.0, 0.0), (3200.0, 0.0), (2000.0, 0.0), (12000.0, 0.0), (5600.0, 0.3),
                                 (4300.0, -0.5), (9000.0, 0.6), (1500.0, 1.0), (20000.0, -1.0)])
def test_wb_gains_formula_and_luma_preservation(k, t):
    g = chain.wb_gains(k, t)
    assert g.shape == (3,) and g.dtype == np.float64 and np.all(g > 0)
    assert np.max(np.abs(g - wb_gains_ref(k, t))) < 1e-12
    assert float(color.luminance(g)) == pytest.approx(1.0, abs=1e-12)  # 흰색의 휘도 보존


def test_wb_reference_is_identity_and_default_theta_is_neutral():
    g = chain.wb_gains(6500.0, 0.0)
    assert np.max(np.abs(g - 1.0)) < 1e-12
    x = scene_hdr()
    th = theta()  # wb_kelvin=6500, wb_tint=0
    assert np.array_equal(chain.apply_wb(x, th), x)  # 비트 동일(float32 반올림 후 1.0)
    assert np.array_equal(chain.remove_wb(x, th), x)
    assert chain.look(x, th, seed=2, frame=1).tobytes() == sf.apply_sensor_film(x, th, seed=2, frame=1).tobytes()


def test_wb_direction_and_monotonicity():
    # 카메라 WB 설정이 낮을수록(텅스텐) 일광 장면은 푸르게: b/r 단조 증가. 반대 방향은 붉게.
    ks = [2000.0, 3000.0, 4500.0, 6500.0, 9000.0, 12000.0]
    ratio = [chain.wb_gains(k, 0.0)[2] / chain.wb_gains(k, 0.0)[0] for k in ks]
    assert all(a > b for a, b in zip(ratio, ratio[1:]))
    assert ratio[0] > 1.0 > ratio[-1]
    # 가운데(g)는 틴트 0에서 k 에 따라 달라지지만, 틴트가 커지면 g 이득이 감소(마젠타)
    gs = [chain.wb_gains(6500.0, t)[1] for t in (-0.6, -0.3, 0.0, 0.3, 0.6)]
    assert all(a > b for a, b in zip(gs, gs[1:]))
    # 켈빈 변화는 g 를 기준으로 r 과 b 를 로그에서 대칭으로 움직인다(정규화는 비율을 바꾸지 않는다)
    g = chain.wb_gains(3200.0, 0.0)
    assert np.log2(g[2] / g[1]) == pytest.approx(-np.log2(g[0] / g[1]), abs=1e-9)
    assert np.log2(g[2] / g[1]) == pytest.approx(WB_MIRED_STOPS * (1e6 / 3200.0 - 1e6 / 6500.0), abs=1e-9)


def test_apply_and_remove_wb_are_float32_gain_and_inverse():
    th = theta(wb_kelvin=3400.0, wb_tint=-0.25)
    x = scene_hdr()
    g32 = wb_gains_ref(3400.0, -0.25).astype(np.float32)
    y = chain.apply_wb(x, th)
    assert y.dtype == np.float32 and np.max(np.abs(y - x * g32)) <= 1e-6 * float(np.max(np.abs(y)))
    assert float(y.max()) > 1.0  # HDR 를 자르지 않는다
    back = chain.remove_wb(y, th)
    assert back.dtype == np.float32 and np.max(np.abs(back - x)) <= 2e-6 * float(x.max())
    neg_free = chain.apply_wb(np.zeros((3, 3, 3), np.float32), th)
    assert np.all(neg_free == 0.0)


def test_wb_changes_color_not_gray_level():
    # 흰/회색의 휘도는 보존, 색조는 이동: 회색 패치가 푸르게 된다
    x = np.full((6, 6, 3), 0.25, np.float32)
    y = chain.apply_wb(x, theta(wb_kelvin=3000.0))
    assert y[0, 0, 2] > y[0, 0, 0]
    assert float(color.luminance(y[0, 0].astype(np.float64))) == pytest.approx(0.25, rel=1e-5)


# ------------------------------------------------------------------------------------------------ 체인 순서
CASES = [
    dict(),
    dict(lut="neutral_709", ev=0.7, wb_kelvin=4200.0),
    dict(lut="teal_orange", vignette=0.6, halation=0.5, wb_tint=0.3),
    dict(lut="kodak_warm", grain=0.5, halation=0.8, vignette=0.2, ev=-0.5, wb_kelvin=9000.0),
    dict(lut="mono_silver", grain=0.3, vignette=1.0),
    dict(lut="cross_process", grain=0.9, halation=1.0, ev=1.3, wb_kelvin=2500.0, wb_tint=-0.4, sensor_width_mm=8.8),
]


@pytest.mark.parametrize("kw", CASES)
def test_look_equals_wb_then_lns04_pieces(kw):
    th = theta(**kw)
    x = scene_hdr()
    for noise in (True, False):
        got = chain.look(x, th, seed=11, frame=3, noise=noise)
        ref = look_by_pieces(x, th, seed=11, frame=3, noise=noise)
        assert np.array_equal(got, ref) or np.max(np.abs(got - ref)) < 1e-6, (kw, noise)
        # 같은 정의: WB 를 먼저 곱한 뒤 apply_sensor_film
        alt = sf.apply_sensor_film(chain.apply_wb(x, th), th, seed=11, frame=3, noise=noise)
        assert got.tobytes() == alt.tobytes()


def test_wb_acts_on_scene_linear_before_sensor_clipping():
    # 밝은 회색에 낮은 켈빈: 파랑 이득 > 1 이라 센서에서 파랑이 먼저 포화한다(표시 뒤에 곱했다면 포화하지 않는다).
    x = np.full((8, 8, 3), 0.8, np.float32)
    th = theta(wb_kelvin=2500.0)
    g = wb_gains_ref(2500.0, 0.0)
    assert g[2] * 0.8 > 1.1 and g[0] < 1.0
    y = chain.look(x, th, noise=False)
    assert float(y[0, 0, 2]) == 1.0
    assert float(y[0, 0, 0]) == pytest.approx(float(color.linear_to_display(np.float32(0.8 * g[0]))), abs=3e-3)
    # WB 와 EV 는 센서 이전 채널 이득이라 교환 가능
    xs = scene_hdr()
    a = chain.look(xs, theta(wb_kelvin=3000.0, ev=0.5, vignette=0.4), noise=False)
    b = chain.look(chain.apply_wb(xs, theta(wb_kelvin=3000.0)), theta(ev=0.5, vignette=0.4), noise=False)
    assert np.max(np.abs(a - b)) < 1e-6
    # 비네팅 중심은 WB 후 노출을 그대로 받는다(가장자리만 어두워진다)
    v = chain.look(np.full((33, 65, 3), 0.3, np.float32), theta(vignette=1.0), noise=False)
    assert float(v[16, 32, 1]) > float(v[0, 0, 1]) + 0.1


def test_neutral_theta_matches_display_transform_up_to_adc():
    x = scene_mid()
    y = chain.look(x, theta(), noise=False)
    ref = color.linear_to_display(x).astype(np.float64)
    assert np.max(np.abs(y - ref)) < 3e-3 and y.dtype == np.float32


def test_ev_is_applied_once_not_twice():
    # 이중 적용 방지: ev=+1 은 선형 노출 2 배(중간톤 장면, 포화 없음)
    x = (scene_mid() * 0.5).astype(np.float32)
    y1 = chain.look(x, theta(ev=1.0, lut="neutral_709"), noise=False)
    y0 = chain.look((x * 2.0).astype(np.float32), theta(lut="neutral_709"), noise=False)
    assert np.max(np.abs(y1 - y0)) < 2e-3


# ------------------------------------------------------------------------------------------------ Frames
def make_frames(n=3, h=24, w=40, fps=24.0):
    rng = np.random.default_rng(5)
    rgb = np.clip(0.3 + 0.2 * rng.standard_normal((n, h, w, 3)), 0, 1).astype(np.float32)
    aov = {"depth": np.ones((n, h, w), np.float32)}
    return C.Frames(rgb=rgb, fps=fps, aov=aov, shot_hash="sh", world_hash="wh")


def test_look_frames_matches_per_frame_chain_and_preserves_metadata():
    th = theta(lut="teal_orange", grain=0.4, halation=0.3, vignette=0.3, wb_kelvin=4800.0, wb_tint=0.1)
    fr = make_frames()
    keep = fr.rgb.copy()
    out = chain.look_frames(fr, th, seed=9)
    assert isinstance(out, C.Frames) and np.array_equal(fr.rgb, keep)
    assert out.rgb.shape == fr.rgb.shape and out.fps == fr.fps
    assert out.shot_hash == "sh" and out.world_hash == "wh"
    assert set(out.aov) == {"depth"} and np.array_equal(out.aov["depth"], fr.aov["depth"])
    assert not np.shares_memory(out.aov["depth"], fr.aov["depth"])
    for i in range(3):
        ref = chain.look(color.srgb_to_linear(fr.rgb[i]), th, seed=9, frame=i)
        assert out.rgb[i].tobytes() == ref.tobytes()
    # 프레임마다 다른 그레인
    flat_frames = C.Frames(rgb=np.full((3, 24, 40, 3), 0.4, np.float32), fps=24.0)
    o2 = chain.look_frames(flat_frames, th, seed=9)
    assert not np.array_equal(o2.rgb[0], o2.rgb[1])
    o3 = chain.look_frames(flat_frames, theta(wb_kelvin=4800.0, lut="teal_orange"), seed=9, noise=False)
    assert np.array_equal(o3.rgb[0], o3.rgb[1])


def test_sampled_thetas_all_run_and_stay_in_range():
    x = scene_hdr(24, 40)
    for th in T.sample_thetas(25, seed=21):
        y = chain.look(x, th, seed=1)
        assert y.dtype == np.float32 and y.min() >= 0.0 and y.max() <= 1.0 and np.all(np.isfinite(y))
