"""LNS-04 이음매: 센서·필름 체인(EV -> 비네팅 -> 할레이션 -> 센서 -> 룩 -> 그레인), Frames 어댑터, 격리."""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from buc.core import color
from buc.core import contracts as C
from buc.lens import sensor_film as sf
from lns04_helpers import GRAIN_SIGMA_MAX, corr, display_ref, gray_img, scene

REPO = Path(__file__).resolve().parents[2]
EXACT = dict(noise=False)


def run(x, theta=None, **kw):
    return sf.apply_sensor_film(x, theta or C.Theta(), **kw)


def theta(**kw):
    return C.Theta(**kw)


# ----------------------------------------------------------------------------------------------- 기본 동작
def test_default_theta_noise_off_is_display_transform_up_to_adc():
    x = scene()
    keep = x.copy()
    y = run(x, **EXACT)
    assert np.array_equal(x, keep)  # 입력 불변
    assert y.dtype == np.float32 and y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    ref = display_ref(np.clip(x, 0, 1))
    assert np.max(np.abs(y - ref)) < 3e-3  # 12 비트 양자화뿐(톤 아래쪽 12.92 배 확대 포함)


def test_default_theta_noise_on_is_clean_for_big_sensor_and_deterministic():
    x = scene()
    a = run(x, seed=4, frame=1)
    assert a.tobytes() == run(x, seed=4, frame=1).tobytes()
    assert a.tobytes() != run(x, seed=4, frame=2).tobytes() and a.tobytes() != run(x, seed=5, frame=1).tobytes()
    ref = run(x, **EXACT)
    assert not np.array_equal(a, ref)
    assert np.mean(np.abs(a - ref)) < 4e-3  # 큰 센서·표준 θ 는 거의 깨끗하다(그레인 0)
    # 작은 센서·같은 폭 -> 같은 장면이 더 시끄럽다
    small = sf.sensor_spec_from_theta(theta(sensor_width_mm=5.79), 1920)
    noisy = run(x, seed=4, frame=1, spec=small)
    assert np.mean(np.abs(noisy - ref)) > 1.5 * np.mean(np.abs(a - ref))


def test_noise_flag_removes_sensor_noise_and_grain():
    x = scene()
    th = theta(grain=0.9)
    off1 = run(x, th, seed=1, **EXACT)
    off2 = run(x, th, seed=77, frame=9, **EXACT)
    assert np.array_equal(off1, off2)  # 확률 요소가 전혀 없다
    assert np.array_equal(off1, run(x, theta(grain=0.0), **EXACT))


def test_ev_is_exposure_gain_before_sensor():
    ramp = np.linspace(0.02, 0.45, 60, dtype=np.float32)
    x = np.broadcast_to(ramp[None, :, None], (8, 60, 3)).copy()
    for ev in (-1.0, 0.5, 1.0):
        y = run(x, theta(ev=ev), **EXACT)
        ref = display_ref(np.clip(x * 2.0 ** ev, 0, 1))
        assert np.max(np.abs(y - ref)) < 3e-3, ev
    over = run(gray_img(0.4), theta(ev=3.0), **EXACT)  # 센서 포화 -> 흰색
    assert np.all(over == 1.0)


def test_vignette_darkens_corners_by_the_gain():
    h, w = 65, 97
    x = gray_img(0.5, h, w)
    y = run(x, theta(vignette=1.0), **EXACT)
    g = sf.vignette_gain(h, w, 1.0)
    ref = display_ref(0.5 * g[..., None])
    assert np.max(np.abs(y - ref)) < 3e-3
    assert y[32, 48, 0] == pytest.approx(float(display_ref(0.5)), abs=3e-3)
    assert y[0, 0, 0] == pytest.approx(float(display_ref(0.5 * 0.2)), abs=3e-3)


def test_halation_glows_red_around_hdr_light():
    x = np.full((96, 160, 3), 0.1, np.float32)
    x[46:50, 78:82] = 6.0
    off = run(x, theta(halation=0.0), **EXACT)
    on = run(x, theta(halation=0.3), **EXACT)
    d = (on - off).astype(np.float64)
    assert d.min() >= -1e-6
    ring = d[48, 83:88]  # 광원 곁(오른쪽 1~5 px)
    assert np.all(ring[:, 0] > ring[:, 1]) and np.all(ring[:, 1] > ring[:, 2]) and ring[:, 0].min() > 0.01
    assert np.max(np.abs(d[:10, :10])) < 5e-3  # 먼 구석은 영향 없음


def test_chain_order_matches_composition_of_public_pieces():
    """EV -> 비네팅 -> 할레이션 -> 센서(양자화) -> 룩 순서. 순서가 바뀌면 비네팅·할레이션이 겹치는 영역에서 어긋난다."""
    x = scene(h=72, w=104)
    x[4:10, 90:100] = 8.0  # 모서리 광원: 비네팅과 할레이션의 순서가 결과를 가른다
    th = theta(ev=0.4, vignette=0.7, halation=0.9, lut="teal_orange")
    spec = sf.sensor_spec_from_theta(th, x.shape[1])
    v = sf.apply_vignette(x * np.float32(2.0 ** th.ev), th.vignette)
    h_ = sf.apply_halation(v, th.halation)
    q = sf.sensor_capture(h_, spec, seed=0, noise=False)
    ref = sf.apply_look(q, th.lut)
    got = run(x, th, **EXACT)
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    assert d.max() < 6e-3 and d.mean() < 5e-4
    # 순서가 다른 합성(할레이션 -> 비네팅)과는 눈에 띄게 달라야 한다
    alt = sf.apply_look(sf.sensor_capture(sf.apply_vignette(sf.apply_halation(x * np.float32(2.0 ** th.ev), th.halation),
                                                            th.vignette), spec, seed=0, noise=False), th.lut)
    assert np.abs(alt.astype(np.float64) - ref).max() > 8e-3


@pytest.mark.parametrize("name", ["none", "neutral_709", "mono_silver", "teal_orange"])
def test_lut_name_is_applied_in_chain(name):
    x = scene()
    y = run(x, theta(lut=name), **EXACT)
    ref = sf.apply_look(sf.sensor_capture(x, sf.sensor_spec_from_theta(theta(), x.shape[1]), seed=0, noise=False), name)
    assert np.max(np.abs(y.astype(np.float64) - ref)) < 1e-6


def test_grain_in_chain_is_isolated_signal_dependent_and_independent_of_sensor_noise():
    x = gray_img(color.srgb_to_linear(np.float32(0.5)).item(), 160, 224)  # 표시값 0.5 의 회색
    th0, th1 = theta(grain=0.0), theta(grain=0.8)
    base = run(x, th0, seed=3, frame=2)
    withg = run(x, th1, seed=3, frame=2)
    resid = (withg - base).astype(np.float64)  # 같은 센서 잡음 실현 -> 차이는 그레인뿐
    assert resid.std() == pytest.approx(0.8 * GRAIN_SIGMA_MAX, rel=0.25)
    assert abs(resid.mean()) < 0.004
    sensor_noise = (base - run(x, th0, seed=3, frame=2, noise=False)).astype(np.float64)
    assert sensor_noise.std() > 0
    assert abs(corr(resid, sensor_noise)) < 0.1  # 서로 다른 난수 스트림
    # 그레인은 프레임마다 새로 뿌려진다
    r2 = (run(x, th1, seed=3, frame=3) - run(x, th0, seed=3, frame=3)).astype(np.float64)
    assert abs(corr(resid, r2)) < 0.1


def test_grain_radius_follows_width_but_has_floor():
    # 폭이 넓을수록 입자(px)가 커진다: 같은 θ, 두 해상도에서 자기상관 길이 비교
    th = theta(grain=1.0)

    def ac(w):
        x = gray_img(float(color.srgb_to_linear(np.float32(0.5))), 128, w)
        r = (run(x, th, seed=1) - run(x, theta(grain=0.0), seed=1)).astype(np.float64)[..., 1]
        return float(np.corrcoef(r[:, :-3].ravel(), r[:, 3:].ravel())[0, 1])

    assert sf.GRAIN_RADIUS_FRAC > 0
    assert ac(2560) > ac(256) + 0.05  # 2560 px: 반경 3.07 px, 256 px: 바닥 0.75 px


# ----------------------------------------------------------------------------------------------- 오류
def test_apply_sensor_film_errors():
    x = scene()
    with pytest.raises(ValueError):
        run(x, C.Theta(lut="not_a_look"))
    with pytest.raises(TypeError):
        run(x.astype(np.float16))
    with pytest.raises(ValueError):
        run(x[..., :2])
    with pytest.raises(ValueError):
        run(x[0])
    bad = x.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        run(bad)
    with pytest.raises(ValueError):
        run(x, seed=-1)
    with pytest.raises(TypeError):
        sf.apply_sensor_film(x, C.Theta(), 1)  # seed 는 키워드 전용


# ----------------------------------------------------------------------------------------------- Frames 어댑터
def _frames(n=3, h=40, w=56, fill=None):
    rng = np.random.default_rng(2)
    rgb = rng.random((n, h, w, 3)).astype(np.float32) if fill is None else np.full((n, h, w, 3), fill, np.float32)
    aov = {"depth": rng.random((n, h, w)).astype(np.float32) + 1.0, "coc_px": rng.random((n, h, w)).astype(np.float32)}
    return C.Frames(rgb=rgb, fps=24.0, aov=aov, shot_hash="shot1234", world_hash="world567")


def test_frames_adapter_seam():
    fr = _frames()
    keep_rgb, keep_depth = fr.rgb.copy(), fr.aov["depth"].copy()
    out = sf.apply_sensor_film_frames(fr, C.Theta(), seed=3, noise=False)
    assert isinstance(out, C.Frames)
    out.validate()
    assert np.array_equal(fr.rgb, keep_rgb) and np.array_equal(fr.aov["depth"], keep_depth)  # 입력 불변
    assert out.rgb.shape == fr.rgb.shape and out.rgb.dtype == np.float32
    assert out.fps == fr.fps and out.shot_hash == fr.shot_hash and out.world_hash == fr.world_hash
    assert set(out.aov) == set(fr.aov)
    for k in fr.aov:
        assert np.array_equal(out.aov[k], fr.aov[k])
    # 표시값 -> 선형 -> 체인 -> 표시값: 기본 θ·잡음 끔이면 양자화 오차 안에서 항등
    assert np.max(np.abs(out.rgb - fr.rgb)) < 3e-3
    assert out.aov["depth"] is not fr.aov["depth"]


def test_frames_adapter_equals_per_frame_calls_with_frame_index():
    fr = _frames(n=3, fill=0.45)
    th = theta(grain=0.7, vignette=0.4, halation=0.3, lut="kodak_warm", sensor_width_mm=8.8)
    out = sf.apply_sensor_film_frames(fr, th, seed=11)
    for i in range(3):
        lin = color.srgb_to_linear(fr.rgb[i])
        ref = sf.apply_sensor_film(lin, th, seed=11, frame=i)
        assert np.array_equal(out.rgb[i], ref)  # i 번째 프레임은 frame=i
    assert not np.array_equal(out.rgb[0], out.rgb[1])  # 같은 입력이어도 프레임마다 다른 잡음·입자
    again = sf.apply_sensor_film_frames(fr, th, seed=11)
    assert again.rgb.tobytes() == out.rgb.tobytes() and again.content_hash() == out.content_hash()
    assert sf.apply_sensor_film_frames(fr, th, seed=12).content_hash() != out.content_hash()


def test_frames_adapter_spec_override_and_errors():
    fr = _frames(n=2)
    big = sf.SensorSpec(full_well_e=1e5, read_noise_e=1.0, bits=14)
    a = sf.apply_sensor_film_frames(fr, C.Theta(), seed=1, spec=big)
    b = sf.apply_sensor_film_frames(fr, C.Theta(), seed=1, spec=sf.SensorSpec(full_well_e=1500.0))
    assert np.mean(np.abs(b.rgb - fr.rgb)) > 3 * np.mean(np.abs(a.rgb - fr.rgb))
    with pytest.raises(ValueError):
        sf.apply_sensor_film_frames(fr, C.Theta(lut="bogus"), seed=1)
    with pytest.raises(TypeError):
        sf.apply_sensor_film_frames(fr.rgb, C.Theta(), seed=1)  # Frames 가 아니다


def test_theta_from_lns01_sampler_runs_clean():
    from buc.lens import theta as T
    x = scene(h=40, w=56)
    for th in T.sample_thetas(8, seed=2):
        y = sf.apply_sensor_film(x, th, seed=1)
        assert y.shape == x.shape and np.all(np.isfinite(y)) and y.min() >= 0 and y.max() <= 1
    for name in T.LUT_NAMES:
        assert sf.make_lut(name).shape == (65, 65, 65, 3)  # LNS-01 어휘 전부 구현


# ----------------------------------------------------------------------------------------------- 격리·성능
def test_import_isolation_and_global_rng():
    code = ("import sys, numpy as np\nnp.random.seed(5)\ns0 = np.random.get_state()[1].copy()\n"
            "import buc.lens.sensor_film as sf\nfrom buc.core.contracts import Theta\n"
            "assert 'torch' not in sys.modules, 'torch 금지'\n"
            "assert (np.random.get_state()[1] == s0).all(), 'import 가 전역 RNG 를 건드렸다'\n"
            "x = np.full((16, 16, 3), 0.3, np.float32)\n"
            "sf.apply_sensor_film(x, Theta(grain=0.5, vignette=0.5, halation=0.5, lut='teal_orange'), seed=1)\n"
            "assert (np.random.get_state()[1] == s0).all(), '체인이 전역 RNG 를 건드렸다'\n")
    r = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code], cwd=REPO, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]


def test_full_stack_speed_cpu():
    x = np.random.default_rng(0).random((360, 640, 3)).astype(np.float32) * 1.3
    th = theta(grain=0.6, vignette=0.5, halation=0.5, lut="teal_orange", ev=0.2)
    sf.apply_sensor_film(x[:32, :32], th, seed=0)  # 표 캐시 예열 허용
    t0 = time.perf_counter()
    y = sf.apply_sensor_film(x, th, seed=0)
    dt = time.perf_counter() - t0
    assert dt < 8.0, f"{dt:.1f}s"
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
