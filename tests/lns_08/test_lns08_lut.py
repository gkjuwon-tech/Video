"""LNS-08 룩(로그 부호화 + 65^3 LUT) 역변환 invert_look: 정확도, 소유 상태 비트, 랭크 결손·클립 룩."""
import numpy as np
import pytest

from buc.lens import chain
from buc.lens import sensor_film as sf
from lns08_helpers import (BLACK_CODE_EPS, CLEAN_LUTS, CLIPPY_LUTS, LOG_FLOOR_S, LUT_INV_TOL, LUTS, OWN_LUT_LOST,
                           OWN_SENSOR_BLACK)

# 자연색 표본에서 소유(status==0) 비율의 하한 (룩마다 클립 정도가 다르다)
OWNED_FLOOR = {"none": 0.99, "neutral_709": 0.97, "bleach_bypass": 0.97, "faded_matte": 0.97, "cross_process": 0.95,
               "teal_orange": 0.90, "kodak_warm": 0.90, "fuji_cool": 0.90, "neon_night": 0.35, "mono_silver": 0.0}


def natural_colors():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.003, 0.7, (1500, 3))
    b = (rng.uniform(0.003, 0.7, (500, 1)) * rng.uniform(0.6, 1.4, (500, 3))).clip(0.003, 0.9)
    return np.concatenate([a, b]).astype(np.float32)


@pytest.mark.parametrize("lut", LUTS)
def test_invert_look_recovers_sensor_value_where_owned(lut):
    s = natural_colors()
    y = sf.apply_look(s, lut)
    got, st = chain.invert_look(y, lut)
    assert got.dtype == np.float32 and got.shape == s.shape and st.dtype == np.uint8 and st.shape == s.shape[:-1]
    assert np.all(np.isfinite(got))
    ok = st == 0
    if lut == "mono_silver":
        assert not ok.any() and np.all((st & OWN_LUT_LOST) != 0)  # 3 -> 1 정보 손실: 전부 비소유
    else:
        assert ok.mean() >= OWNED_FLOOR[lut], (lut, ok.mean())
        err = np.abs(got - s)[ok]
        assert err.max() <= LUT_INV_TOL, (lut, float(err.max()))
    # 어느 룩이든 소유 여부와 무관하게 되돌린 값을 다시 찍으면 입력 표시값을 재현한다(look∘invert ≈ id, 치역 안)
    conv = (st & OWN_LUT_LOST) == 0 if lut != "mono_silver" else np.ones(len(s), bool)
    back = sf.apply_look(np.clip(got, 0.0, 1.0), lut)
    assert np.max(np.abs(back - y)[conv]) <= 1e-4, lut


def test_invert_look_none_is_srgb_decode_with_no_flags():
    s = np.concatenate([[0.0, 1e-4, 3e-4, 0.0031308, 0.5, 1.0], np.exp(np.linspace(np.log(1e-3), 0, 50))])
    s = np.stack([s, s[::-1], np.roll(s, 3)], axis=-1).astype(np.float32)
    y = sf.apply_look(s, "none")
    got, st = chain.invert_look(y, "none")
    assert np.all(st == 0)
    assert np.max(np.abs(got - s)) <= 2e-6
    assert got[0, 0] == 0.0 and got[5, 0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("lut", [n for n in LUTS if n != "none"])
def test_invert_look_accepts_any_leading_shape(lut):
    s = natural_colors()[:240]
    y = sf.apply_look(s, lut)
    a, sa = chain.invert_look(y, lut)
    b, sb = chain.invert_look(y.reshape(12, 20, 3), lut)
    assert b.shape == (12, 20, 3) and sb.shape == (12, 20)
    assert np.array_equal(a, b.reshape(-1, 3)) and np.array_equal(sa, sb.reshape(-1))


@pytest.mark.parametrize("lut", [n for n in LUTS if n != "none"])
def test_black_flag_marks_log_floor(lut):
    # 로그 부호화 바닥(0.18*2^-8 = 7.03e-4) 아래 노출은 코드값 0 으로 뭉개진다 -> 소유 불가.
    below = np.full((6, 6, 3), 0.18 * 2.0 ** -8 * 0.3, np.float32)  # 바닥의 30 %
    st = chain.invert_look(sf.apply_look(below, lut), lut)[1]
    assert np.all(st != 0)
    if lut in CLEAN_LUTS:
        assert np.all((st & OWN_SENSOR_BLACK) != 0)
    # 바닥보다 충분히 위(코드값 > 0.1)는 BLACK 이 아니다
    above = np.full((6, 6, 3), 0.01, np.float32)
    st2 = chain.invert_look(sf.apply_look(above, lut), lut)[1]
    assert np.all((st2 & OWN_SENSOR_BLACK) == 0)
    assert LOG_FLOOR_S == pytest.approx(7.03125e-4) and BLACK_CODE_EPS == 2e-3


def test_none_has_no_black_floor():
    dark = np.full((4, 4, 3), 2e-4, np.float32)
    got, st = chain.invert_look(sf.apply_look(dark, "none"), "none")
    assert np.all(st == 0) and np.max(np.abs(got - dark)) <= 2e-6


@pytest.mark.parametrize("lut", CLIPPY_LUTS)
def test_output_clipped_channels_are_never_owned(lut):
    rng = np.random.default_rng(3)
    s = np.concatenate([rng.uniform(0.003, 0.95, (3000, 3)),
                        np.exp(rng.uniform(np.log(8e-4), 0.0, (3000, 3)))]).astype(np.float32)
    y = sf.apply_look(s, lut)
    got, st = chain.invert_look(y, lut)
    clipped = ((y <= 1e-6) | (y >= 1.0 - 1e-6)).any(axis=-1)
    assert clipped.any()  # 이 표본에서 클립이 실제로 있다
    assert np.all((st[clipped] & OWN_LUT_LOST) != 0)
    # 소유로 판정된 화소는 모두 정확하다(거짓 소유 없음)
    ok = st == 0
    assert ok.any() and np.abs(got - s)[ok].max() <= LUT_INV_TOL


@pytest.mark.parametrize("lut", CLEAN_LUTS)
def test_clean_luts_are_fully_invertible_inside_the_unclipped_cube(lut):
    # 클립도 랭크 결손도 없는 룩: 바닥 위 ~ 포화 아래(0.75) 의 모든 색 조합에서 소유이고 정확하다.
    g = np.exp(np.linspace(np.log(4e-3), np.log(0.75), 13))
    s = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    got, st = chain.invert_look(sf.apply_look(s, lut), lut)
    assert np.all(st == 0), (lut, float((st != 0).mean()))
    assert np.abs(got - s).max() <= LUT_INV_TOL


def test_mono_silver_best_effort_is_gray_and_reproduces_display():
    s = natural_colors()[:300]
    y = sf.apply_look(s, "mono_silver")
    got, st = chain.invert_look(y, "mono_silver")
    assert np.all((st & OWN_LUT_LOST) != 0)
    assert np.max(np.abs(sf.apply_look(got, "mono_silver") - y)) <= 1e-4
