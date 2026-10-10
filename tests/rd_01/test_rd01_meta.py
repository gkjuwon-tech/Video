"""RD-01 메타데이터: JPEG DQT·품질, EXIF 화이트리스트·방향, C2PA 존재 기록, 메타 불변성."""
import io
import json
import math

import numpy as np
import pytest
from PIL import Image

import rd01_helpers as H
from buc.read import ingest as ing

BASE = H.to_u8(H.rgb3(H.field(64, 96, seed=4)))          # (64,96,3)


# ---------------------------------------------------------------------------------------------------------------
# DQT / JPEG 품질
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1, 5, 10, 25, 40, 50, 60, 75, 85, 90, 95, 98, 100])
def test_jpeg_quality_is_recovered_exactly_for_ijg_tables(q):
    data = H.jpeg_bytes(BASE, quality=q)
    tabs = ing.jpeg_quant_tables(data)
    ref = Image.open(io.BytesIO(data)).quantization
    assert sorted(tabs) == sorted(ref) == [0, 1]
    for k in tabs:
        assert tabs[k].shape == (64,)
        np.testing.assert_array_equal(tabs[k], np.array(ref[k]))
    jq = ing.estimate_jpeg_quality(tabs)
    assert jq.quality == q and jq.exact is True and jq.fit_err == 0
    r = ing.ingest(data)
    assert r.fmt == "jpeg" and r.jpeg.quality == q and r.jpeg.exact is True


def test_dqt_parser_reads_zigzag_to_natural_order_from_raw_bytes():
    t0, t1 = H.ijg_table(H.STD_LUMA, 37), H.ijg_table(H.STD_CHROMA, 37)
    for merge in (False, True):
        tabs = ing.jpeg_quant_tables(H.raw_dqt_jpeg({0: t0, 1: t1}, merge=merge))
        assert sorted(tabs) == [0, 1]
        np.testing.assert_array_equal(tabs[0], t0)
        np.testing.assert_array_equal(tabs[1], t1)
    assert tabs[0][1] == t0[1] and tabs[0][8] == t0[8]
    # 16 비트 정밀도(Pq=1) 표
    big = np.arange(1, 65) * 7
    tabs = ing.jpeg_quant_tables(H.raw_dqt_jpeg({}, extra_16bit={2: big}))
    np.testing.assert_array_equal(tabs[2], big)


def test_grayscale_jpeg_has_one_table_and_quality_still_exact():
    buf = io.BytesIO()
    Image.fromarray(BASE[..., 0]).save(buf, "JPEG", quality=70)
    r = ing.ingest(buf.getvalue())
    assert sorted(ing.jpeg_quant_tables(buf.getvalue())) == [0]
    assert r.jpeg.quality == 70 and r.jpeg.exact is True and r.image.pixels.shape == (64, 96, 3)
    assert (r.image.pixels[..., 0] == r.image.pixels[..., 1]).all()


def test_custom_tables_are_not_exact_and_report_the_fit_error():
    t = H.ijg_table(H.STD_LUMA, 80).copy()
    t[10] += 7
    tabs = {0: t, 1: H.ijg_table(H.STD_CHROMA, 80)}
    jq = ing.estimate_jpeg_quality(tabs)
    errs = [max(np.abs(tabs[0] - H.ijg_table(H.STD_LUMA, q)).max(), np.abs(tabs[1] - H.ijg_table(H.STD_CHROMA, q)).max())
            for q in range(1, 101)]
    assert jq.exact is False and jq.fit_err > 0 and abs(jq.quality - 80) <= 5
    assert jq.fit_err == min(errs) and jq.quality == 1 + errs.index(min(errs))          # 최대절댓값 오차 최소, 동률이면 낮은 q
    buf = io.BytesIO()
    Image.fromarray(BASE).save(buf, "JPEG", qtables=[np.full(64, 3), np.full(64, 5)])
    r = ing.ingest(buf.getvalue())
    assert r.jpeg.exact is False and 1 <= r.jpeg.quality <= 100


def test_non_jpeg_has_no_jpeg_info_and_bad_jpeg_bytes_raise():
    assert ing.ingest(H.png_bytes(BASE)).jpeg is None
    assert ing.ingest(BASE, color_space="srgb").jpeg is None
    with pytest.raises(ing.IngestError):
        ing.jpeg_quant_tables(H.png_bytes(BASE))
    good = H.raw_dqt_jpeg({0: H.ijg_table(H.STD_LUMA, 50)})
    with pytest.raises(ing.IngestError):
        ing.jpeg_quant_tables(good[:30])                    # 세그먼트 중간에서 잘림
    with pytest.raises(ing.IngestError):
        ing.estimate_jpeg_quality({1: H.ijg_table(H.STD_CHROMA, 50)})        # 휘도표가 없다


# ---------------------------------------------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------------------------------------------
ALLOWED = {"make", "model", "orientation", "focal_mm", "focal_35mm", "f_number", "exposure_s", "iso"}


def test_exif_whitelist_values_and_types():
    ex = H.make_exif(orientation=1, make="Acme", model="X100", focal=23.0, f35=35, fnumber=2.0, exposure=0.008, iso=200)
    data = H.png_bytes(BASE, exif=ex)
    e = ing.read_exif(data)
    assert e["make"] == "Acme" and e["model"] == "X100" and e["orientation"] == 1 and e["iso"] == 200
    assert e["focal_mm"] == pytest.approx(23.0) and e["focal_35mm"] == pytest.approx(35.0)
    assert e["f_number"] == pytest.approx(2.0) and e["exposure_s"] == pytest.approx(0.008)
    assert isinstance(e["iso"], int) and isinstance(e["focal_mm"], float) and isinstance(e["make"], str)
    assert set(e) <= ALLOWED
    r = ing.ingest(data)
    assert r.exif == e and r.image.exif == e
    json.dumps(r.exif)


def test_exif_in_jpeg_and_webp_is_read_too():
    ex = H.make_exif(make="Acme", focal=50.0, f35=75, fnumber=2.8, iso=100)
    for data in (H.jpeg_bytes(BASE, exif=ex), H.webp_bytes(BASE, exif=ex)):
        e = ing.read_exif(data)
        assert e["make"] == "Acme" and e["focal_35mm"] == pytest.approx(75.0) and e["iso"] == 100


def test_no_private_exif_leaks_anywhere():
    ex = H.make_exif(make="Acme", focal=50.0, f35=75, private=True)
    data = H.png_bytes(BASE, exif=ex)
    r = ing.ingest(data)
    assert set(r.exif) <= ALLOWED
    dump = json.dumps(r.meta_input()) + json.dumps(r.image.exif)
    assert H.PRIVATE_MARK not in dump and H.SOFTWARE_MARK not in dump and "2031:01:02" not in dump and "gps" not in dump.lower()
    assert H.PRIVATE_MARK not in repr(r.exif)


def test_invalid_exif_values_are_dropped_and_garbage_never_raises():
    ex = H.make_exif(make="Acme", focal=0.0, f35=0, fnumber=0.0, iso=0)
    e = ing.read_exif(H.png_bytes(BASE, exif=ex))
    assert e == {"make": "Acme"}
    junk = H.png_insert_chunk(H.png_bytes(BASE), b"eXIf", b"garbage-not-tiff" * 4)
    assert ing.read_exif(junk) == {}
    assert ing.ingest(junk).exif == {}
    assert ing.read_exif(b"") == {} and ing.read_exif(b"not an image") == {}
    assert ing.read_exif(H.png_bytes(BASE)) == {}


def test_fov_prior_from_35mm_equivalent_focal():
    # 35mm 환산 초점거리는 대각선 43.2666 mm 센서 기준. 3:2 에서 가로 36 mm.
    assert ing.fov_x_prior_deg(50.0, 3000, 2000) == pytest.approx(39.5978, abs=1e-3)
    assert ing.fov_x_prior_deg(18.0, 6000, 4000) == pytest.approx(90.0, abs=1e-6)       # 36 mm 폭 / 18 mm -> tan(45°)
    diag = math.hypot(36, 24)
    f90 = diag / (2 * math.sqrt(2))                       # 정사각형이면 가로 = 대각선/√2
    assert ing.fov_x_prior_deg(f90, 1000, 1000) == pytest.approx(90.0, abs=1e-6)
    assert ing.fov_x_prior_deg(35.0, 4000, 3000) > ing.fov_x_prior_deg(50.0, 4000, 3000)
    for bad in ((0.0, 10, 10), (-5.0, 10, 10), (35.0, 0, 10), (35.0, 10, 0), (float("nan"), 10, 10)):
        with pytest.raises(ValueError):
            ing.fov_x_prior_deg(*bad)


def test_ingest_exposes_fov_prior_only_when_exif_has_it():
    ex = H.make_exif(f35=50)
    px = H.to_u8(H.rgb3(H.field(80, 120, seed=1)))
    r = ing.ingest(H.png_bytes(px, exif=ex))
    assert r.fov_x_prior_deg == pytest.approx(ing.fov_x_prior_deg(50.0, 120, 80), abs=1e-9)
    assert r.meta_input()["fov_x_prior_deg"] == pytest.approx(r.fov_x_prior_deg)
    assert ing.ingest(H.png_bytes(px)).fov_x_prior_deg is None
    # 방향이 적용된 크기를 쓴다 (세로 사진의 가로 FOV 는 더 좁다)
    rot = ing.ingest(H.png_bytes(px, exif=H.make_exif(f35=50, orientation=6)))
    assert rot.src_size == (120, 80) and rot.fov_x_prior_deg == pytest.approx(ing.fov_x_prior_deg(50.0, 80, 120), abs=1e-9)
    assert rot.fov_x_prior_deg < r.fov_x_prior_deg


# ---------------------------------------------------------------------------------------------------------------
# 방향
# ---------------------------------------------------------------------------------------------------------------
ASYM = H.to_u8(H.rgb3(H.field(6, 10, seed=9)))
ASYM[0, 0] = [255, 0, 0]
ASYM[0, -1] = [0, 255, 0]
ASYM[-1, 0] = [0, 0, 255]


@pytest.mark.parametrize("o", range(1, 9))
def test_orientation_is_applied_losslessly(o):
    data = H.png_bytes(ASYM, exif=H.make_exif(orientation=o))
    r = ing.ingest(data, opts=ing.IngestOpts(letterbox=False))
    want = np.ascontiguousarray(H.orient(ASYM, o))
    np.testing.assert_array_equal(r.image.pixels, want)
    assert r.src_size == want.shape[:2] and r.orientation == o and r.orientation_applied == (o != 1)
    assert r.exif["orientation"] == o and r.image.pixels.flags["C_CONTIGUOUS"]


def test_orientation_can_be_left_alone_and_invalid_values_mean_identity():
    data = H.png_bytes(ASYM, exif=H.make_exif(orientation=6))
    r = ing.ingest(data, opts=ing.IngestOpts(letterbox=False, apply_orientation=False))
    np.testing.assert_array_equal(r.image.pixels, ASYM)
    assert r.orientation == 6 and r.orientation_applied is False
    for bad in (0, 9, 77):
        r = ing.ingest(H.png_bytes(ASYM, exif=H.make_exif(orientation=bad)), opts=ing.IngestOpts(letterbox=False))
        np.testing.assert_array_equal(r.image.pixels, ASYM)
        assert r.orientation == 1 and r.orientation_applied is False and "orientation" not in r.exif


def test_orientation_in_webp_and_letterbox_is_found_in_display_orientation():
    r = ing.ingest(H.webp_bytes(ASYM, exif=H.make_exif(orientation=3)), opts=ing.IngestOpts(letterbox=False))
    np.testing.assert_array_equal(r.image.pixels, np.ascontiguousarray(H.orient(ASYM, 3)))
    content = H.to_u8(H.rgb3(H.field(80, 120, seed=3)))
    barred = H.add_bars(content, 0, 0, 20, 0)                                   # 저장 방향 기준 왼쪽 띠
    r = ing.ingest(H.png_bytes(barred, exif=H.make_exif(orientation=6)))        # 6: 시계방향 90° -> 왼쪽 띠가 위쪽 띠
    assert r.src_size == (140, 80) and r.bars == (20, 0, 0, 0) and r.crop_box == (0, 20, 80, 140)
    np.testing.assert_array_equal(r.image.pixels, np.ascontiguousarray(H.orient(content, 6)))


# ---------------------------------------------------------------------------------------------------------------
# C2PA
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("make,container", [(lambda d, **k: H.png_with_c2pa(d, **k), "png-cabx"),
                                            (lambda d, **k: H.jpeg_with_c2pa(d, **k), "jpeg-app11"),
                                            (lambda d, **k: H.webp_with_c2pa(d, **k), "webp-c2pa")])
@pytest.mark.parametrize("ai", [False, True])
def test_c2pa_presence_container_size_and_ai_claim(make, container, ai):
    plain = {"png-cabx": H.png_bytes(BASE), "jpeg-app11": H.jpeg_bytes(BASE), "webp-c2pa": H.webp_bytes(BASE)}[container]
    c = ing.detect_c2pa(make(plain, ai_claim=ai))
    assert (c.present, c.container, c.ai_claim) == (True, container, ai)
    assert c.manifest_bytes == len(H.c2pa_jumbf(ai))
    n = ing.detect_c2pa(plain)
    assert (n.present, n.container, n.manifest_bytes, n.ai_claim) == (False, None, 0, False)


def test_c2pa_false_positives_are_rejected():
    other = H.jumbf_box(b"jumb", H.jumbf_box(b"jumd", bytes(16) + b"\x03" + b"abcd\0") + H.jumbf_box(b"cbor", b"hello"))
    jpg = H.jpeg_insert_segment(H.jpeg_bytes(BASE), 0xEB, b"JP" + bytes(6) + other)            # JUMBF 이지만 c2pa 아님
    assert ing.detect_c2pa(jpg).present is False
    fake = H.jpeg_insert_segment(H.jpeg_bytes(BASE), 0xEB, b"XX" + bytes(6) + H.c2pa_jumbf())  # 'JP' 접두가 없다
    assert ing.detect_c2pa(fake).present is False
    comment = H.jpeg_insert_segment(H.jpeg_bytes(BASE), 0xFE, b"c2pa jumb in a comment")
    assert ing.detect_c2pa(comment).present is False
    assert ing.detect_c2pa(b"").present is False and ing.detect_c2pa(b"garbage").present is False
    assert ing.detect_c2pa(H.png_bytes(BASE)[:40]).present is False                             # 잘린 PNG


def test_c2pa_and_exif_are_recorded_but_never_change_pixels_or_numbers():
    ex = H.make_exif(make="Acme", focal=50.0, f35=75, iso=100)                   # 방향 없음: 픽셀에 영향 없는 메타
    plain = ing.ingest(H.png_bytes(BASE))
    for data in (H.png_with_c2pa(H.png_bytes(BASE), ai_claim=True), H.png_bytes(BASE, exif=ex),
                 H.png_with_c2pa(H.png_bytes(BASE, exif=ex), ai_claim=True)):
        r = ing.ingest(data)
        np.testing.assert_array_equal(r.image.pixels, plain.image.pixels)
        np.testing.assert_array_equal(r.image_lin, plain.image_lin)
        assert r.image.h_I == plain.image.h_I and r.crop_box == plain.crop_box
        assert (r.r_in, r.r_eff, r.color.space, r.warnings) == (plain.r_in, plain.r_eff, plain.color.space, plain.warnings)
        assert r.image.prov.taint == plain.image.prov.taint == frozenset({"USER"})       # AI 주장이 출처를 바꾸지 않는다
        assert r.image.prov.ext is False
    r = ing.ingest(H.png_with_c2pa(H.png_bytes(BASE), ai_claim=True))
    assert r.c2pa.present and r.c2pa.ai_claim and plain.c2pa.present is False
    assert r.meta_input()["c2pa"] == {"present": True, "container": "png-cabx", "manifest_bytes": len(H.c2pa_jumbf(True)), "ai_claim": True}
