"""EVL-04 ΔE00: 32x32 패치 평균(선형 RGB) -> Lab -> CIEDE2000, 패치 평균 색차."""
import numpy as np
import pytest

from buc.core.color import delta_e_2000, rgb_lin_to_lab, srgb_to_linear
from evl04_support import ff, texture


def _flat(h, w, rgb):
    return np.broadcast_to(np.asarray(rgb, np.float32), (h, w, 3)).copy()


def test_white_vs_black_is_exactly_100_and_identity_is_zero():
    m = ff()
    assert m.PATCH == 32
    w, k = _flat(64, 96, (1, 1, 1)), _flat(64, 96, (0, 0, 0))
    assert m.delta_e00(w, k) == pytest.approx(100.0, abs=1e-2)
    assert m.delta_e00(w, w) == 0.0
    t = texture(64, 96, seed=1)
    assert m.delta_e00(t, t) == 0.0


def test_matches_independent_oracle_on_flat_color_offset():
    m = ff()
    a = _flat(64, 64, (0.50, 0.40, 0.30))
    b = _flat(64, 64, (0.55, 0.42, 0.28))
    lab = lambda e: rgb_lin_to_lab(srgb_to_linear(np.asarray(e, np.float64)))
    expect = float(delta_e_2000(lab((0.50, 0.40, 0.30)), lab((0.55, 0.42, 0.28))))
    assert 1.0 < expect < 10.0
    assert m.delta_e00(a, b) == pytest.approx(expect, abs=1e-4)
    assert m.delta_e00(b, a) == pytest.approx(expect, abs=1e-4)  # 대칭


def test_patch_mean_is_taken_in_linear_light_not_display_values():
    m = ff()
    # 한 패치 안에서 표시값 0 과 1 이 반반: 선형 평균 0.5 => 표시값 0.7354 의 회색과 같아야 한다.
    chk = np.zeros((32, 32, 3), np.float32)
    chk[:, ::2] = 1.0
    gray = _flat(32, 32, (0.5 ** (1 / 2.4) * 1.055 - 0.055,) * 3)  # linear 0.5 의 sRGB 부호화 값
    assert m.delta_e00(chk, gray) < 0.05
    naive_gray = _flat(32, 32, (0.5, 0.5, 0.5))  # 표시값 평균(0.5)과는 확실히 다르다
    assert m.delta_e00(chk, naive_gray) > 5.0


def test_is_low_pass_only_grain_scale_swaps_are_invisible():
    m = ff()
    chk = np.zeros((64, 96, 3), np.float32)
    chk[::2, ::2] = 1.0
    chk[1::2, 1::2] = 1.0
    inv = 1.0 - chk  # 패치 안 평균은 같고 픽셀별로는 정반대
    assert m.delta_e00(chk, inv) < 1e-3


def test_only_full_patches_count_and_result_is_patch_mean():
    m = ff()
    a = _flat(70, 100, (0.5, 0.5, 0.5))  # 패치 2x3 개(나머지 6 행·4 열은 버린다)
    b = a.copy()
    b[:32, :32] = (0.7, 0.5, 0.5)  # 첫 패치만 다르다
    b[64:, :] = 0.0  # 버려지는 줄(64..69)은 영향이 없다
    b[:, 96:] = 0.0
    one = m.delta_e00(a[:32, :32], b[:32, :32])
    assert one > 1.0
    assert m.delta_e00(a, b) == pytest.approx(one / 6.0, abs=1e-6)


def test_mask_restricts_the_patch_mean_and_min_cover_filters_patches():
    m = ff()
    a = _flat(64, 64, (0.5, 0.5, 0.5))
    b = a.copy()
    b[:, :32] = (0.6, 0.5, 0.5)  # 왼쪽 열 패치만 다르다
    full = m.delta_e00(a, b)
    right = np.zeros((64, 64), bool)
    right[:, 32:] = True
    assert m.delta_e00(a, b, mask=right) == pytest.approx(0.0, abs=1e-9)
    assert full > 0.1
    # 패치 안 마스크 점유율 정확히 50% 는 포함(>= min_cover), 0.6 을 요구하면 제외 => 포함 패치 없음 ValueError
    top = np.zeros((64, 64), bool)
    top[:, :] = False
    for r0 in (0, 32):
        top[r0:r0 + 16, :] = True
    c = a.copy()
    d = a.copy()
    for r0 in (0, 32):
        d[r0 + 16:r0 + 32, :] = (0.9, 0.5, 0.5)  # 마스크 밖 절반에만 차이
    assert m.delta_e00(c, d, mask=top) == pytest.approx(0.0, abs=1e-9)  # 점유 50% 패치가 포함됐고 차이는 0
    with pytest.raises(ValueError):
        m.delta_e00(c, d, mask=top, min_cover=0.6)
    # 마스크 안 픽셀만 평균에 들어간다: 차이를 마스크 안쪽에 두면 값이 생긴다
    e = a.copy()
    for r0 in (0, 32):
        e[r0:r0 + 16, :] = (0.9, 0.5, 0.5)
    assert m.delta_e00(a, e, mask=top) > 3.0


def test_validation():
    m = ff()
    a = _flat(64, 64, (0.5,) * 3)
    with pytest.raises(ValueError):
        m.delta_e00(a, a[:, :-1])
    with pytest.raises(ValueError):
        m.delta_e00(a[:20], a[:20])  # 패치(32)보다 작다
    with pytest.raises(ValueError):
        m.delta_e00(a, a, mask=np.ones((64, 63), bool))
    with pytest.raises(ValueError):
        m.delta_e00(a, a, mask=np.zeros((64, 64), bool))  # 포함 패치 없음
    with pytest.raises(ValueError):
        m.delta_e00(a, a, patch=0)


def test_inputs_unchanged_and_uint8_ok():
    m = ff()
    t = texture(64, 64, seed=3)
    u = (t * 255 + 0.5).astype(np.uint8)
    keep = u.copy()
    v = m.delta_e00(u, np.clip(u.astype(np.int16) + 6, 0, 255).astype(np.uint8))
    assert np.array_equal(u, keep) and type(v) is float and v > 0
