"""FND-06 color: sRGB 전달함수, 원색 행렬, 색공간 변환, Lab, ΔE2000. 표준 문서 상수와 독립 스칼라 오라클로 판정한다."""
import math

import numpy as np
import pytest

from buc.core import color

RNG = np.random.default_rng(7)
SPACES = ("srgb", "display-p3", "rec2020", "linear-srgb", "adobe-rgb")


# ---- 독립 오라클 (IEC 61966-2-1) ----
def srgb_dec(v):
    s = -1.0 if v < 0 else 1.0
    a = abs(v)
    return s * (a / 12.92 if a <= 0.04045 else ((a + 0.055) / 1.055) ** 2.4)


def srgb_enc(v):
    s = -1.0 if v < 0 else 1.0
    a = abs(v)
    return s * (12.92 * a if a <= 0.0031308 else 1.055 * a ** (1 / 2.4) - 0.055)


def xy_to_XYZ(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


def derive_matrix(prim, white_xy=(0.3127, 0.3290)):
    P = np.stack([xy_to_XYZ(*p) for p in prim], axis=1)
    S = np.linalg.solve(P, xy_to_XYZ(*white_xy))
    return P * S


PRIM = {
    "srgb": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
    "display-p3": ((0.68, 0.32), (0.265, 0.69), (0.15, 0.06)),
    "rec2020": ((0.708, 0.292), (0.17, 0.797), (0.131, 0.046)),
    "adobe-rgb": ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)),
}
D65 = xy_to_XYZ(0.3127, 0.3290)


# ---------------------------------------------------------------------------------------------------- sRGB 전달함수
def test_srgb_matches_scalar_oracle_float64():
    x = np.concatenate([np.linspace(-0.3, 1.5, 2001), [0.0, 0.04045, 0.0031308, 1.0, 0.5]])
    np.testing.assert_allclose(color.srgb_to_linear(x), [srgb_dec(v) for v in x], rtol=0, atol=1e-13)
    np.testing.assert_allclose(color.linear_to_srgb(x), [srgb_enc(v) for v in x], rtol=0, atol=1e-13)
    assert color.srgb_to_linear(np.float64(0.5)) == pytest.approx(0.21404114048223255, abs=1e-13)
    assert color.linear_to_srgb(np.float64(0.5)) == pytest.approx(0.7353569830524495, abs=1e-13)


def test_srgb_endpoints_symmetry_monotonic_roundtrip():
    assert color.srgb_to_linear(np.array([0.0, 1.0]))[0] == 0.0 and color.srgb_to_linear(np.array([0.0, 1.0]))[1] == 1.0
    e = color.linear_to_srgb(np.array([0.0, 1.0]))
    assert e[0] == 0.0 and e[1] == pytest.approx(1.0, abs=2e-16)
    x = np.linspace(0.0, 1.0, 5001)
    np.testing.assert_allclose(color.srgb_to_linear(-x), -color.srgb_to_linear(x), atol=0)     # 홀함수 확장
    np.testing.assert_allclose(color.linear_to_srgb(-x), -color.linear_to_srgb(x), atol=0)
    for f in (color.srgb_to_linear, color.linear_to_srgb):
        assert np.all(np.diff(f(np.linspace(-0.2, 3.0, 4001))) > 0)
    big = np.linspace(-1.0, 8.0, 3001)
    np.testing.assert_allclose(color.srgb_to_linear(color.linear_to_srgb(big)), big, rtol=1e-10, atol=1e-12)
    # 문턱에서 두 가지(선형부/거듭제곱부)가 거의 연속: IEC 상수(12.92, 0.04045)의 알려진 틈(~3e-8)까지만 허용
    for thr, f in [(0.04045, color.srgb_to_linear), (0.0031308, color.linear_to_srgb)]:
        assert abs(f(np.float64(thr + 1e-12)) - f(np.float64(thr - 1e-12))) < 1e-7


def test_srgb_dtype_policy_and_no_mutation():
    a32 = RNG.uniform(0, 1, (4, 5, 3)).astype(np.float32)
    keep = a32.copy()
    out = color.srgb_to_linear(a32)
    assert out.dtype == np.float32 and out.shape == a32.shape
    np.testing.assert_array_equal(a32, keep)
    np.testing.assert_allclose(out.ravel(), [srgb_dec(float(v)) for v in a32.ravel()], rtol=1e-6, atol=1e-7)
    assert color.linear_to_srgb(a32).dtype == np.float32
    assert color.srgb_to_linear(a32.astype(np.float64)).dtype == np.float64
    with pytest.raises(TypeError):
        color.srgb_to_linear(np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(TypeError):
        color.linear_to_srgb(np.zeros(3, np.int32))


def test_srgb_8bit_lut_and_quantizer():
    codes = np.arange(256, dtype=np.uint8)
    lin = color.srgb8_to_linear(codes)
    assert lin.dtype == np.float32 and lin.shape == (256,)
    np.testing.assert_allclose(lin, [srgb_dec(c / 255.0) for c in range(256)], rtol=1e-6, atol=0)
    assert lin[0] == 0.0 and lin[255] == 1.0
    back = color.linear_to_srgb8(lin)
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, codes)                                    # 256 코드 모두 왕복 동일
    np.testing.assert_array_equal(color.linear_to_srgb8(np.array([-1.0, 0.0, 2.0, np.float32(1.0)], np.float32)), [0, 0, 255, 255])
    img = RNG.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    assert color.srgb8_to_linear(img).shape == (5, 7, 3)
    with pytest.raises(TypeError):
        color.srgb8_to_linear(np.zeros(3, np.float32))
    # 반올림: 코드 100과 101의 중간 sRGB 값(선형값) 바로 아래/위는 각각 100/101
    mid = srgb_dec((100.5) / 255.0)
    assert color.linear_to_srgb8(np.array([mid * (1 - 1e-6), mid * (1 + 1e-6)], np.float32)).tolist() == [100, 101]


# ---------------------------------------------------------------------------------------------------- 다른 전달함수
def test_transfer_functions_per_space():
    x = np.linspace(0.0, 1.0, 401)
    for sp in SPACES:
        np.testing.assert_allclose(color.eotf(color.oetf(x, sp), sp), x, atol=1e-12)
        assert color.oetf(np.array([0.0]), sp)[0] == 0.0 and color.oetf(np.array([1.0]), sp)[0] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(color.eotf(x, "srgb"), color.srgb_to_linear(x), atol=0)
    np.testing.assert_allclose(color.eotf(x, "display-p3"), color.srgb_to_linear(x), atol=0)    # P3 는 sRGB 곡선
    np.testing.assert_allclose(color.oetf(x, "linear-srgb"), x, atol=0)
    np.testing.assert_allclose(color.eotf(x, "adobe-rgb"), x ** (563.0 / 256.0), atol=1e-13)
    a, b = 1.09929682680944, 0.018053968510807                                          # BT.2020 상수
    L = np.array([0.001, 0.0099, 0.18, 0.5, 0.9])
    want = np.where(L < b, 4.5 * L, a * L ** 0.45 - (a - 1))
    np.testing.assert_allclose(color.oetf(L, "rec2020"), want, atol=1e-12)
    with pytest.raises(ValueError):
        color.eotf(x, "unknown")
    with pytest.raises(ValueError):
        color.oetf(x, "aces")


# ---------------------------------------------------------------------------------------------------- 원색 행렬
def test_matrices_derived_from_primaries_and_d65():
    for sp in ("srgb", "display-p3", "rec2020", "adobe-rgb"):
        M = color.rgb_to_xyz_matrix(sp)
        assert M.shape == (3, 3) and M.dtype == np.float64
        np.testing.assert_allclose(M, derive_matrix(PRIM[sp]), atol=1e-9)
        np.testing.assert_allclose(M @ np.ones(3), D65, atol=1e-9)                      # 흰색 = D65
        np.testing.assert_allclose(color.xyz_to_rgb_matrix(sp) @ M, np.eye(3), atol=1e-10)
    np.testing.assert_array_equal(color.rgb_to_xyz_matrix("linear-srgb"), color.rgb_to_xyz_matrix("srgb"))
    np.testing.assert_allclose(color.WHITE_XYZ, D65, atol=1e-9)
    M = color.rgb_to_xyz_matrix("srgb")                                                 # IEC 행렬과 1e-3 이내
    np.testing.assert_allclose(M, [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], atol=1e-3)


def test_convert_linear_known_matrices_and_gamut_behaviour():
    srgb_to_2020 = [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]]   # ITU-R BT.2407
    srgb_to_p3 = [[0.8225, 0.1774, 0.0], [0.0332, 0.9669, 0.0], [0.0171, 0.0724, 0.9108]]
    eye = np.eye(3)
    np.testing.assert_allclose(color.convert_linear(eye, "srgb", "rec2020").T, srgb_to_2020, atol=1e-3)
    np.testing.assert_allclose(color.convert_linear(eye, "srgb", "display-p3").T, srgb_to_p3, atol=1e-3)
    for sp in SPACES:
        np.testing.assert_allclose(color.convert_linear(np.ones(3), sp, "srgb"), 1.0, atol=1e-9)    # 흰색 보존
        np.testing.assert_allclose(color.convert_linear(np.ones((2, 3)), "srgb", sp), 1.0, atol=1e-9)
        x = RNG.uniform(0, 1, (11, 3))
        np.testing.assert_array_equal(color.convert_linear(x, sp, sp), x)                       # 같은 공간은 항등(비트)
    x = RNG.uniform(0, 1, (50, 3))
    for sp in ("display-p3", "rec2020", "adobe-rgb"):
        np.testing.assert_allclose(color.convert_linear(color.convert_linear(x, "srgb", sp), sp, "srgb"), x, atol=1e-12)
    # sRGB 빨강은 P3 안에 들어오고, P3 빨강은 sRGB 밖(음수 성분, 잘라내지 않는다)
    p3_of_red = color.convert_linear(np.array([1.0, 0, 0]), "srgb", "display-p3")
    assert np.all(p3_of_red >= -1e-9)
    srgb_of_p3_red = color.convert_linear(np.array([1.0, 0, 0]), "display-p3", "srgb")
    assert srgb_of_p3_red[0] > 1.22 and srgb_of_p3_red[1] < -0.04 and srgb_of_p3_red[2] < -0.01
    with pytest.raises(ValueError):
        color.convert_linear(x, "srgb", "unknown")
    with pytest.raises(ValueError):
        color.convert_linear(np.ones((4, 2)), "srgb", "rec2020")


def test_decode_to_linear():
    u8 = RNG.integers(0, 256, size=(6, 7, 3), dtype=np.uint8)
    out = color.decode_to_linear(u8, "srgb")
    assert out.dtype == np.float32 and out.shape == (6, 7, 3)
    np.testing.assert_allclose(out, color.srgb8_to_linear(u8), rtol=1e-6, atol=0)
    assert color.decode_to_linear(np.full((1, 1, 3), 255, np.uint8), "srgb")[0, 0].tolist() == [1.0, 1.0, 1.0]
    u16 = RNG.integers(0, 65536, size=(4, 4, 3), dtype=np.uint16)
    ref16 = np.array([srgb_dec(v / 65535.0) for v in u16.ravel()]).reshape(u16.shape)
    np.testing.assert_allclose(color.decode_to_linear(u16, "srgb"), ref16, rtol=2e-6, atol=1e-8)
    f32 = RNG.uniform(0, 1, (5, 5, 3)).astype(np.float32)
    np.testing.assert_array_equal(color.decode_to_linear(f32, "linear-srgb"), f32)             # 항등(비트)
    np.testing.assert_allclose(color.decode_to_linear(f32, "srgb"), color.srgb_to_linear(f32), atol=0)
    np.testing.assert_array_equal(color.decode_to_linear(f32, "unknown"), color.decode_to_linear(f32, "srgb"))   # unknown = sRGB
    want = color.convert_linear(color.eotf(f32.astype(np.float64), "display-p3"), "display-p3", "srgb")
    np.testing.assert_allclose(color.decode_to_linear(f32, "display-p3"), want, atol=1e-6)
    sat = np.zeros((1, 1, 3), np.float32)
    sat[0, 0] = [0.0, 1.0, 0.0]                                                                # P3 순수 초록 -> sRGB 선형 음수 성분 유지
    assert color.decode_to_linear(sat, "display-p3")[0, 0, 0] < -0.05
    with pytest.raises(TypeError):
        color.decode_to_linear(np.zeros((2, 2, 3), np.int32), "srgb")
    with pytest.raises(ValueError):
        color.decode_to_linear(np.zeros((2, 2, 4), np.uint8), "srgb")
    with pytest.raises(ValueError):
        color.decode_to_linear(u8, "bt709-gamma")


def test_linear_to_display_for_frames():
    x = np.array([[-1.0, 0.0, 0.18], [0.5, 1.0, 2.0]], np.float32)
    y = color.linear_to_display(x)
    assert y.dtype == np.float32 and y.min() >= 0.0 and y.max() <= 1.0
    np.testing.assert_allclose(y[0], [0.0, 0.0, srgb_enc(0.18)], atol=1e-6)
    np.testing.assert_allclose(y[1], [srgb_enc(0.5), 1.0, 1.0], atol=1e-6)
    g = color.linear_to_display(np.linspace(0, 1, 100).astype(np.float32))
    assert np.all(np.diff(g) > 0)


def test_luminance_uses_rec709_row_of_matrix():
    M = color.rgb_to_xyz_matrix("srgb")
    x = RNG.uniform(0, 1, (9, 4, 3))
    np.testing.assert_allclose(color.luminance(x), x @ M[1], atol=1e-12)
    np.testing.assert_allclose(M[1], [0.2126, 0.7152, 0.0722], atol=1e-4)
    gray = np.full((3, 3, 3), 0.37)
    np.testing.assert_allclose(color.luminance(gray), 0.37, atol=1e-12)
    assert color.luminance(np.ones(3)) == pytest.approx(1.0, abs=1e-12)
    assert color.luminance(x).shape == (9, 4)
    assert color.luminance(x.astype(np.float32)).dtype == np.float32


# ---------------------------------------------------------------------------------------------------- Lab
def lab_oracle(xyz, white):
    eps, kap = 216.0 / 24389.0, 24389.0 / 27.0
    t = np.asarray(xyz, float) / white
    f = np.where(t > eps, np.cbrt(t), (kap * t + 16.0) / 116.0)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def test_lab_anchors_and_oracle():
    np.testing.assert_allclose(color.xyz_to_lab(color.WHITE_XYZ), [100, 0, 0], atol=1e-9)
    np.testing.assert_allclose(color.xyz_to_lab(np.zeros(3)), [0, 0, 0], atol=1e-12)
    L18 = color.xyz_to_lab(color.WHITE_XYZ * 0.18)
    assert L18[0] == pytest.approx(116 * 0.18 ** (1 / 3) - 16, abs=1e-9) and abs(L18[1]) < 1e-9 and abs(L18[2]) < 1e-9
    assert L18[0] == pytest.approx(49.5, abs=0.05)
    np.testing.assert_allclose(color.rgb_lin_to_lab(np.array([1.0, 0, 0])), [53.24, 80.09, 67.20], atol=0.06)   # sRGB 빨강
    np.testing.assert_allclose(color.rgb_lin_to_lab(np.ones(3)), [100, 0, 0], atol=1e-6)
    for _ in range(100):
        xyz = RNG.uniform(0, 1.1, 3) * color.WHITE_XYZ
        np.testing.assert_allclose(color.xyz_to_lab(xyz), lab_oracle(xyz, color.WHITE_XYZ), atol=1e-9)
    # 정의역 이음매(Y/Yn = 216/24389)에서 연속
    eps = 216.0 / 24389.0
    lo = color.xyz_to_lab(color.WHITE_XYZ * (eps - 1e-12))
    hi = color.xyz_to_lab(color.WHITE_XYZ * (eps + 1e-12))
    np.testing.assert_allclose(lo, hi, atol=1e-6)


def test_lab_roundtrip_batch_shapes_and_white_param():
    xyz = RNG.uniform(0.0, 1.0, (4, 6, 3))
    lab = color.xyz_to_lab(xyz)
    assert lab.shape == (4, 6, 3)
    np.testing.assert_allclose(color.lab_to_xyz(lab), xyz, atol=1e-10)
    w = np.array([0.9642, 1.0, 0.8249])                                            # D50
    np.testing.assert_allclose(color.xyz_to_lab(w, white=w), [100, 0, 0], atol=1e-9)
    np.testing.assert_allclose(color.lab_to_xyz(color.xyz_to_lab(xyz, white=w), white=w), xyz, atol=1e-10)
    rgb = RNG.uniform(0, 1, (30, 3))
    np.testing.assert_allclose(color.rgb_lin_to_lab(rgb), color.xyz_to_lab(rgb @ color.rgb_to_xyz_matrix("srgb").T), atol=1e-9)
    np.testing.assert_allclose(color.lab_to_rgb_lin(color.rgb_lin_to_lab(rgb)), rgb, atol=1e-9)


# ---------------------------------------------------------------------------------------------------- ΔE2000
SHARMA = [  # Sharma, Wu, Dalal (2005) 검증 쌍. (Lab1, Lab2, ΔE00)
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -0.9009, -85.5211), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((50.0, 2.5, 0.0), (58.0, 24.0, 15.0), 19.4535),
    ((50.0, 2.5, 0.0), (50.0, 3.1736, 0.5854), 1.0000),
]


@pytest.mark.parametrize("lab1,lab2,want", SHARMA)
def test_delta_e_2000_sharma_vectors(lab1, lab2, want):
    assert color.delta_e_2000(np.array(lab1), np.array(lab2)) == pytest.approx(want, abs=1e-4)


def test_delta_e_2000_properties():
    a = RNG.uniform([0, -80, -80], [100, 80, 80], size=(200, 3))
    b = RNG.uniform([0, -80, -80], [100, 80, 80], size=(200, 3))
    d = color.delta_e_2000(a, b)
    assert d.shape == (200,) and np.all(d >= 0) and np.all(np.isfinite(d))
    np.testing.assert_allclose(d, color.delta_e_2000(b, a), atol=1e-12)               # 대칭
    np.testing.assert_allclose(color.delta_e_2000(a, a), 0.0, atol=0)
    assert color.delta_e_2000(a[:, :], np.array([50.0, 0.0, 0.0])).shape == (200,)    # 방송
    # 색상각 360도 경계(h≈0 / h≈360)를 가로지르는 극소 차이는 극소 ΔE
    near = color.delta_e_2000(np.array([50.0, 10.0, 1e-4]), np.array([50.0, 10.0, -1e-4]))
    assert 0.0 < near < 1e-3
    # 회색 쌍(채도 0): 순수 명도 차이만 SL 가중으로 나온다. L̄=50 이면 SL=1, L̄=30 이면 SL>1.
    assert color.delta_e_2000(np.array([40.0, 0, 0]), np.array([60.0, 0, 0])) == pytest.approx(20.0, rel=1e-9)
    g = color.delta_e_2000(np.array([20.0, 0, 0]), np.array([40.0, 0, 0]))
    assert g == pytest.approx(20.0 / (1 + 0.015 * 20.0 ** 2 / math.sqrt(20 + 20.0 ** 2)), rel=1e-9)
    # kL 가중은 명도 항만 줄인다
    assert color.delta_e_2000(np.array([20.0, 0, 0]), np.array([40.0, 0, 0]), kL=2.0) == pytest.approx(g / 2, rel=1e-9)
