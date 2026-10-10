"""LNS-04 룩: 로그 부호화, 65^3 LUT 생성·삼선형 적용, 이름별 룩의 성격."""
import numpy as np
import pytest

from buc.core import color
from buc.lens import sensor_film as sf
from lns04_helpers import (LOG_MIDGRAY, LOG_STOPS, LUT_SIZE, LUTS, display_ref, lab_L, lab_chroma, log_decode_ref,
                           log_encode_ref)

_RNG = np.random.default_rng(11)
_LUTS = {}


def lut(name):
    if name not in _LUTS:
        _LUTS[name] = sf.make_lut(name)
    return _LUTS[name]


def look(x_lin, name):
    return sf.apply_look(np.asarray(x_lin, np.float32).reshape(-1, 1, 3), name).reshape(-1, 3).astype(np.float64)


def gray(v):
    return np.array([[v, v, v]], np.float32)


# 대표 패치(장면 선형)
GRAYS = (0.02, 0.18, 0.7)
CHROMA_PATCHES = np.array([(0.25, 0.12, 0.12), (0.12, 0.25, 0.12), (0.12, 0.12, 0.25), (0.25, 0.25, 0.12),
                           (0.12, 0.25, 0.25), (0.25, 0.12, 0.25)], np.float32)
BATTERY = np.concatenate([np.array([[g, g, g] for g in np.geomspace(0.004, 1.0, 12)], np.float32),
                          CHROMA_PATCHES, np.random.default_rng(3).random((40, 3)).astype(np.float32)])


def test_constants_match_contract():
    assert sf.LUT_SIZE == LUT_SIZE == 65
    assert sf.LOG_MIDGRAY == LOG_MIDGRAY and tuple(sf.LOG_STOPS) == LOG_STOPS


# ----------------------------------------------------------------------------------------------- 로그 부호화
def test_log_encode_known_values_and_roundtrip():
    lo, hi = LOG_STOPS
    assert float(sf.log_encode(np.float64(LOG_MIDGRAY))) == pytest.approx(-lo / (hi - lo), abs=1e-12)
    assert float(sf.log_encode(np.float64(LOG_MIDGRAY * 2.0))) == pytest.approx((1 - lo) / (hi - lo), abs=1e-12)
    assert float(sf.log_encode(np.float64(0.0))) == 0.0  # 0 도 유한
    assert float(sf.log_encode(np.float64(1e-12))) == 0.0 and float(sf.log_encode(np.float64(1e6))) == 1.0
    x = np.geomspace(LOG_MIDGRAY * 2.0 ** lo * 1.001, LOG_MIDGRAY * 2.0 ** hi * 0.999, 400)
    e = sf.log_encode(x)
    assert e.dtype == np.float64 and np.all(np.diff(e) > 0)
    assert np.max(np.abs(e - log_encode_ref(x))) < 1e-12
    assert np.max(np.abs(sf.log_decode(e) / x - 1.0)) < 1e-12
    e32 = sf.log_encode(x.astype(np.float32))
    assert e32.dtype == np.float32 and np.max(np.abs(e32 - e)) < 1e-6
    assert np.max(np.abs(sf.log_decode(np.linspace(0, 1, 50)) / log_decode_ref(np.linspace(0, 1, 50)) - 1)) < 1e-12


# ----------------------------------------------------------------------------------------------- LUT 적용
def test_identity_lut_layout():
    L = sf.identity_lut(5)
    assert L.shape == (5, 5, 5, 3) and L.dtype == np.float32
    assert np.allclose(L[2, 3, 4], [0.5, 0.75, 1.0])  # [r][g][b] 순서
    assert np.allclose(L[0, 0, 0], 0) and np.allclose(L[4, 4, 4], 1)
    assert sf.identity_lut().shape == (65, 65, 65, 3)


def test_apply_lut3d_identity_and_affine_exact():
    rng = np.random.default_rng(1)
    pts = rng.random((500, 3)).astype(np.float32)
    ident = sf.identity_lut(17)
    assert np.max(np.abs(sf.apply_lut3d(pts, ident) - pts)) < 2e-6
    # 격자 밖 임의 점에서 아핀 함수는 삼선형(및 사면체) 보간으로 정확히 복원된다
    A = np.array([[0.5, 0.2, -0.1], [0.1, 0.7, 0.05], [-0.2, 0.3, 0.9]])
    b = np.array([0.1, -0.05, 0.2])
    grid = sf.identity_lut(9).astype(np.float64)
    table = (grid @ A.T + b).astype(np.float32)
    got = sf.apply_lut3d(pts, table).astype(np.float64)
    assert np.max(np.abs(got - (pts.astype(np.float64) @ A.T + b))) < 5e-6
    # 모양 보존: (H,W,3) 와 (N,3) 모두
    img = rng.random((6, 7, 3)).astype(np.float32)
    assert sf.apply_lut3d(img, ident).shape == (6, 7, 3) and sf.apply_lut3d(img, ident).dtype == np.float32


def test_apply_lut3d_nodes_axes_clamp_and_continuity():
    n = 7
    R, G, B = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")
    table = np.stack([R * 100 + G * 10 + B, R, G], -1).astype(np.float32) / 1000.0  # 축 구분이 되는 표
    node = np.array([[2 / 6, 5 / 6, 1 / 6]], np.float32)
    assert np.allclose(sf.apply_lut3d(node, table), table[2, 5, 1], atol=1e-6)  # 정확한 노드 조회, 첫 축 = R
    assert np.allclose(sf.apply_lut3d(np.array([[1.0, 1.0, 1.0]], np.float32), table), table[6, 6, 6], atol=1e-6)
    assert np.allclose(sf.apply_lut3d(np.array([[-0.5, 3.0, 0.0]], np.float32), table), table[0, 6, 0], atol=1e-6)  # 클램프
    t = np.linspace(0.3, 0.7, 4001, dtype=np.float32)
    path = np.stack([t, 1 - t, 0.5 * np.ones_like(t)], -1)
    y = sf.apply_lut3d(path, table).astype(np.float64)
    assert np.max(np.abs(np.diff(y, axis=0))) < 3e-3  # 불연속 없음


def test_apply_lut3d_errors():
    ident = sf.identity_lut(5)
    px = np.zeros((2, 3), np.float32)
    with pytest.raises(ValueError):
        sf.apply_lut3d(px, np.zeros((5, 5, 4, 3), np.float32))  # 정육면체 아님
    with pytest.raises(ValueError):
        sf.apply_lut3d(px, np.zeros((1, 1, 1, 3), np.float32))  # 크기 < 2
    with pytest.raises(ValueError):
        sf.apply_lut3d(px, np.zeros((5, 5, 5), np.float32))
    with pytest.raises(ValueError):
        sf.apply_lut3d(np.zeros((2, 4), np.float32), ident)  # 채널 != 3
    with pytest.raises(TypeError):
        sf.apply_lut3d(np.zeros((2, 3), np.int32), ident)
    bad = px.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        sf.apply_lut3d(bad, ident)


# ----------------------------------------------------------------------------------------------- LUT 생성
@pytest.mark.parametrize("name", LUTS)
def test_every_lut_is_well_formed(name):
    L = lut(name)
    assert L.shape == (65, 65, 65, 3) and L.dtype == np.float32
    assert np.all(np.isfinite(L)) and L.min() >= 0.0 and L.max() <= 1.0
    diag = L[np.arange(65), np.arange(65), np.arange(65)].astype(np.float64)
    assert np.all(np.diff(diag, axis=0) >= -1e-6)  # 회색축 채널별 비감소(톤 역전 없음)
    for ax in range(3):
        assert np.max(np.abs(np.diff(L.astype(np.float64), axis=ax))) < 0.15  # 인접 노드 점프 없음(매끈함)
    # 결정론 + 반환 복사본 독립
    again = sf.make_lut(name)
    assert again.tobytes() == L.tobytes()
    again[...] = 0.123  # 반환 배열을 망가뜨려도 다음 호출은 영향받지 않는다
    assert sf.make_lut(name).tobytes() == L.tobytes()


def test_make_lut_size_and_errors():
    assert sf.make_lut("kodak_warm", 17).shape == (17, 17, 17, 3)
    assert sf.make_lut("neutral_709", 2).shape == (2, 2, 2, 3)
    for bad in ("bogus", "", "NONE"):
        with pytest.raises(ValueError):
            sf.make_lut(bad)
    for bad in (1, 0, -3):
        with pytest.raises(ValueError):
            sf.make_lut("none", bad)


def test_neutral_709_is_the_display_transform_on_nodes():
    L = lut("neutral_709").astype(np.float64)
    e = np.linspace(0, 1, 65)
    ref = display_ref(log_decode_ref(e))  # 노드 i 의 입력 장면값 -> 기준 표시값
    for ax, idx in enumerate(((slice(None), 20, 40), (10, slice(None), 33), (60, 5, slice(None)))):
        assert np.max(np.abs(L[idx][:, ax] - ref)) < 2e-6
    assert np.array_equal(lut("none"), lut("neutral_709"))  # 표 자체는 같다(none 은 적용만 우회)


def test_apply_look_none_is_exact_display_transform_and_neutral_is_close():
    x = np.concatenate([np.geomspace(1e-3, 8.0, 300), [0.0, 1e-5, -0.3, 1.0, 1.7]]).astype(np.float32)
    img = np.stack([x, x[::-1], np.roll(x, 31)], -1).reshape(-1, 1, 3)
    n = sf.apply_look(img, "none")
    assert n.dtype == np.float32 and n.shape == img.shape
    assert np.array_equal(n, color.linear_to_display(img))  # 비트 동일 우회
    nz = sf.apply_look(img, "neutral_709").astype(np.float64)
    d = np.abs(nz - n.astype(np.float64)).reshape(-1, 3)
    flat = img.reshape(-1, 3)
    lo = 0.18 * 2.0 ** LOG_STOPS[0] * 1.02
    inside = np.all(flat >= lo, axis=1)
    below_knee = inside & np.all(flat <= 0.9, axis=1)
    assert below_knee.sum() > 100 and d[below_knee].max() < 3e-3  # 로그 범위 안: 65 노드 삼선형 보간 오차뿐
    assert inside.sum() > below_knee.sum() and d[inside].max() < 1.2e-2  # 클리핑 무릎(x=1) 셀은 오차가 더 크다
    assert nz.min() >= 0.0 and nz.max() <= 1.0


def test_apply_look_accepts_table_and_name_equivalently():
    x = BATTERY.astype(np.float32)
    assert np.array_equal(sf.apply_look(x, "kodak_warm"), sf.apply_look(x, lut("kodak_warm")))
    with pytest.raises(ValueError):
        sf.apply_look(x, "bogus")
    with pytest.raises(ValueError):
        sf.apply_look(np.zeros((2, 4), np.float32), "none")


# ----------------------------------------------------------------------------------------------- 룩의 성격
def test_teal_orange_splits_shadows_and_highlights():
    sh = look(gray(0.02), "teal_orange")[0]
    hi = look(gray(0.7), "teal_orange")[0]
    assert sh[2] - sh[0] >= 0.01  # 그림자는 청록(파랑 > 빨강)
    assert hi[0] - hi[2] >= 0.01  # 하이라이트는 주황(빨강 > 파랑)
    base = look(gray(0.7), "neutral_709")[0]
    assert abs(base[0] - base[2]) < 1e-3  # 기준은 무채색


def test_warm_and_cool_tint_midtones_and_highlights():
    for g in (0.18, 0.7):
        k = look(gray(g), "kodak_warm")[0]
        f = look(gray(g), "fuji_cool")[0]
        assert k[0] - k[2] >= 0.01 and f[2] - f[0] >= 0.01


def test_mono_silver_is_neutral_and_monotone():
    out = look(BATTERY, "mono_silver")
    assert np.max(np.abs(out[:, 0] - out[:, 1])) < 1e-5 and np.max(np.abs(out[:, 1] - out[:, 2])) < 1e-5
    g = look(np.array([[v, v, v] for v in np.geomspace(0.005, 1, 40)], np.float32), "mono_silver")[:, 0]
    assert np.all(np.diff(g) > 0)
    # 휘도 순서를 따른다: 같은 휘도의 다른 색은 비슷한 회색
    a = look(np.array([[0.5, 0.5, 0.5]], np.float32), "mono_silver")[0, 0]
    assert 0.2 < a < 0.95


def test_faded_matte_lifts_blacks_and_drops_whites():
    z = look(np.zeros((1, 3), np.float32), "faded_matte")[0]
    w = look(np.ones((1, 3), np.float32), "faded_matte")[0]
    assert z.min() >= 0.03
    assert w.max() <= 0.98
    n = look(np.ones((1, 3), np.float32), "neutral_709")[0]
    assert n.min() > 0.99  # 기준 흰색은 1


def test_bleach_bypass_desaturates_and_adds_contrast():
    c = lab_chroma(look(CHROMA_PATCHES, "bleach_bypass"))
    c0 = lab_chroma(look(CHROMA_PATCHES, "neutral_709"))
    assert np.mean(c / c0) <= 0.85
    lo, hi = np.array([[0.02] * 3, [0.7] * 3], np.float32)
    span = lambda nm: lab_L(look(hi[None], nm))[0] - lab_L(look(lo[None], nm))[0]
    assert span("bleach_bypass") >= span("neutral_709") + 2.0


def test_neon_night_boosts_saturation():
    c = lab_chroma(look(CHROMA_PATCHES, "neon_night"))
    c0 = lab_chroma(look(CHROMA_PATCHES, "neutral_709"))
    assert np.mean(c / c0) >= 1.15


def test_cross_process_shifts_channels_differently():
    ramp = np.array([[v, v, v] for v in np.geomspace(0.01, 1, 20)], np.float32)
    o = look(ramp, "cross_process")
    assert np.max(np.abs(o[:, 0] - o[:, 2])) >= 0.04  # 회색이 더 이상 회색이 아니다
    assert np.mean(np.abs(o - look(ramp, "neutral_709"))) >= 0.03


def test_all_non_none_looks_are_mutually_distinct():
    names = [n for n in LUTS if n not in ("none",)]
    outs = {n: look(BATTERY, n) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.mean(np.abs(outs[a] - outs[b])) > 0.01, (a, b)
    assert np.array_equal(sf.apply_look(BATTERY.reshape(-1, 1, 3), "none"),
                          color.linear_to_display(BATTERY.reshape(-1, 1, 3)))
