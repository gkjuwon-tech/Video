"""RD-01 레터박스(검은 띠) 검출과 크롭."""
import numpy as np
import pytest

import rd01_helpers as H
from buc.read import ingest as ing

CONTENT = H.to_u8(H.rgb3(H.field(96, 128, seed=11)))          # (96,128,3) 밝기 ~0.5, 검은 행·열 없음


def _box(img, **kw):
    return ing.detect_letterbox(img, **kw)


def test_no_bars_returns_the_full_frame():
    assert _box(CONTENT) == (0, 0, 128, 96)


@pytest.mark.parametrize("top,bottom,left,right", [
    (16, 16, 0, 0),            # 시네마스코프형 위아래 띠
    (0, 0, 24, 24),            # 필러박스
    (8, 24, 0, 0),             # 비대칭
    (12, 12, 20, 20),          # 윈도박스(네 변)
    (0, 7, 0, 0),              # 한 변만
    (2, 0, 0, 3),              # 최소 두께(2 px)
])
def test_exact_box_for_lossless_images(top, bottom, left, right):
    img = H.add_bars(CONTENT, top, bottom, left, right)
    h, w = img.shape[:2]
    assert _box(img) == (left, top, w - right, h - bottom)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_dtype_scale_is_handled(dtype):
    base = {"uint8": CONTENT, "uint16": H.to_u16(CONTENT / 255.0), "float32": (CONTENT / 255.0).astype(np.float32)}[dtype]
    img = H.add_bars(base, 10, 6, 14, 0)
    assert _box(img) == (14, 10, 14 + 128, 10 + 96)


def test_one_pixel_border_is_not_a_letterbox_but_two_is():
    assert _box(H.add_bars(CONTENT, 1, 1, 1, 1)) == (0, 0, 130, 98)
    assert _box(H.add_bars(CONTENT, 2, 2, 2, 2)) == (2, 2, 130, 98)
    # 한 변이 1 px 이고 다른 변이 충분하면 얇은 변만 남는다
    img = H.add_bars(CONTENT, 1, 8, 0, 0)
    assert _box(img) == (0, 0, 128, 1 + 96)


def test_near_black_noise_of_a_dark_scene_is_not_a_bar():
    rng = np.random.default_rng(5)
    dark = np.clip(rng.normal(8, 3, (40, 128, 3)), 0, 255).astype(np.uint8)       # 평균 8/255 > 2/255
    img = np.concatenate([dark, CONTENT], axis=0)
    assert _box(img) == (0, 0, 128, 136)
    true_black_with_sensor_noise = np.clip(rng.normal(0.4, 0.7, (40, 128, 3)), 0, 255).round().astype(np.uint8)
    img2 = np.concatenate([true_black_with_sensor_noise, CONTENT], axis=0)         # 평균 <2/255, 표준편차 <2/255
    assert _box(img2) == (0, 40, 128, 136)


def test_black_inside_the_frame_is_not_a_bar():
    img = CONTENT.copy()
    img[40:60] = 0                                                                  # 가장자리와 닿지 않는 검은 줄
    assert _box(img) == (0, 0, 128, 96)
    img = H.add_bars(CONTENT, 10, 10, 0, 0)
    img[50:70] = 0
    assert _box(img) == (0, 10, 128, 106)


def test_pure_black_frame_and_mostly_black_frame_are_not_cropped():
    assert _box(np.zeros((60, 80, 3), np.uint8)) == (0, 0, 80, 60)
    logo = np.zeros((100, 100, 3), np.uint8)
    logo[70:90, 30:70] = 200
    assert _box(logo) == (0, 0, 100, 100)                 # 남는 높이 20% < min_keep_frac(0.5): 크롭하지 않는다


def test_min_keep_frac_threshold_and_options():
    img = H.add_bars(CONTENT[:40], 30, 30, 0, 0)          # 100 행 중 40 행만 내용 (40%)
    assert _box(img) == (0, 0, 128, 100)
    assert _box(img, opts=ing.IngestOpts(min_keep_frac=0.3)) == (0, 30, 128, 70)
    bars = H.add_bars(CONTENT, 5, 5, 0, 0)
    assert _box(bars, opts=ing.IngestOpts(min_bar_px=6)) == (0, 0, 128, 106)
    assert _box(bars, opts=ing.IngestOpts(min_bar_px=5)) == (0, 5, 128, 101)


def test_jpeg_compressed_bars_are_found_within_one_mcu():
    img = H.add_bars(CONTENT, 16, 16, 16, 0)
    r = ing.ingest(H.jpeg_bytes(img, quality=90))
    x0, y0, x1, y1 = r.crop_box
    assert abs(x0 - 16) <= 2 and abs(y0 - 16) <= 2 and abs(x1 - 144) <= 2 and abs(y1 - 112) <= 2


def test_detect_letterbox_validates_input_and_does_not_modify_it():
    img = H.add_bars(CONTENT, 8, 8, 0, 0)
    keep = img.copy()
    _box(img)
    np.testing.assert_array_equal(img, keep)
    with pytest.raises(ValueError):
        _box(img[..., 0])
    with pytest.raises(ValueError):
        _box(img[..., :2])
    with pytest.raises(TypeError):
        _box(img.astype(np.float64))


def test_opts_validation():
    for bad in (dict(min_bar_px=0), dict(min_keep_frac=0.0), dict(min_keep_frac=1.5), dict(bar_mean=-1.0),
                dict(bar_std=-0.1), dict(max_pixels=0)):
        with pytest.raises(ValueError):
            ing.IngestOpts(**bad)
    o = ing.IngestOpts()
    assert (o.letterbox, o.min_bar_px, o.min_keep_frac, o.apply_orientation) == (True, 2, 0.5, True)
    assert o.bar_mean == pytest.approx(2 / 255) and o.bar_std == pytest.approx(2 / 255)
    with pytest.raises(Exception):
        o.min_bar_px = 3                                  # frozen


# ---------------------------------------------------------------------------------------------------------------
# ingest 와의 결합: 크롭은 픽셀·선형 영상·해시·r_in 모두에 일관되게 적용된다
# ---------------------------------------------------------------------------------------------------------------
def test_ingest_crops_everything_consistently():
    inner = H.to_u8(H.rgb3(H.field(160, 200, seed=2)))
    full = H.add_bars(inner, 20, 20, 0, 0)
    r = ing.ingest(H.png_bytes(full))
    assert r.src_size == (200, 200) and r.crop_box == (0, 20, 200, 180) and r.bars == (20, 20, 0, 0)
    np.testing.assert_array_equal(r.image.pixels, inner)
    assert r.image_lin.shape == (160, 200, 3)
    assert r.r_in == 160
    assert "letterbox_cropped" in r.warnings
    plain = ing.ingest(H.png_bytes(inner))
    assert plain.crop_box == (0, 0, 200, 160) and "letterbox_cropped" not in plain.warnings
    assert r.image.h_I == plain.image.h_I                  # 해시는 크롭된 픽셀의 해시
    np.testing.assert_array_equal(r.image_lin, plain.image_lin)
    assert r.r_eff == plain.r_eff                          # r_eff 는 띠를 보지 않는다


def test_letterbox_can_be_disabled():
    full = H.add_bars(CONTENT, 16, 16, 0, 0)
    r = ing.ingest(H.png_bytes(full), opts=ing.IngestOpts(letterbox=False))
    assert r.crop_box == (0, 0, 128, 128) and r.image.pixels.shape == full.shape
    np.testing.assert_array_equal(r.image.pixels, full)
