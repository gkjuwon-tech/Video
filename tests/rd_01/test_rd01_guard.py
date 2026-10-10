"""도우미 자체 검사: 테스트 오라클(ICC 제작기, IJG 표, 방향 변환, 합성 장, 컨테이너 주입기)이 맞아야 나머지 테스트가 의미 있다.
이 파일은 RD-01 구현을 import 하지 않는다."""
import io
import struct

import numpy as np
import pytest
from PIL import Image, ImageCms

import rd01_helpers as H


# 알려진 D50 착색제 (ICC 공개 프로파일 값, 4 자리)
KNOWN_COLORANTS = {
    "srgb": [[0.4360, 0.2225, 0.0139], [0.3851, 0.7169, 0.0971], [0.1431, 0.0606, 0.7141]],
    "display-p3": [[0.5151, 0.2412, -0.0011], [0.2920, 0.6922, 0.0419], [0.1571, 0.0666, 0.7841]],
    "adobe-rgb": [[0.6097, 0.3111, 0.0195], [0.2053, 0.6257, 0.0609], [0.1492, 0.0632, 0.7446]],
    "rec2020": [[0.6734, 0.2790, -0.0019], [0.1656, 0.6753, 0.0300], [0.1251, 0.0456, 0.7969]],
}


@pytest.mark.parametrize("space", sorted(KNOWN_COLORANTS))
def test_colorants_match_published_values(space):
    col = H.colorants(H.PRIMS[space])
    np.testing.assert_allclose(col.T, np.array(KNOWN_COLORANTS[space]), atol=6e-4)
    # 세 착색제의 합 = D50 흰색
    np.testing.assert_allclose(col.sum(axis=1), H.D50_XYZ, atol=2e-4)


@pytest.mark.parametrize("space,form", [("srgb", "table"), ("srgb", "para"), ("display-p3", "table"), ("adobe-rgb", "table"),
                                        ("rec2020", "table"), ("srgb", "gamma22"), ("srgb", "linear")])
def test_made_icc_is_a_valid_profile_for_lcms(space, form):
    data = H.icc_for(space, form)
    assert data[36:40] == b"acsp" and struct.unpack(">I", data[:4])[0] == len(data)
    prof = ImageCms.ImageCmsProfile(io.BytesIO(data))
    assert ImageCms.getProfileDescription(prof).strip() == f"test {space} {form}"
    # lcms 로 sRGB 변환이 실제로 되는지 (태그 오프셋·곡선이 해석 가능)
    im = Image.new("RGB", (4, 4), (200, 100, 50))
    out = ImageCms.profileToProfile(im, prof, ImageCms.createProfile("sRGB"))
    assert out.size == (4, 4)


def test_srgb_profile_through_lcms_roundtrips_like_identity():
    prof = ImageCms.ImageCmsProfile(io.BytesIO(H.icc_for("srgb", "table")))
    im = Image.fromarray(np.arange(0, 256, 5, dtype=np.uint8).reshape(1, -1, 1).repeat(3, 2))
    out = np.asarray(ImageCms.profileToProfile(im, prof, ImageCms.createProfile("sRGB"))).astype(int)
    assert np.abs(out - np.asarray(im).astype(int)).max() <= 2


def test_ijg_tables_match_libjpeg_for_every_quality():
    arr = np.zeros((16, 16, 3), np.uint8)
    for q in range(1, 101):
        im = Image.open(io.BytesIO(H.jpeg_bytes(arr, quality=q)))
        np.testing.assert_array_equal(np.array(im.quantization[0]), H.ijg_table(H.STD_LUMA, q), err_msg=f"luma q={q}")
        np.testing.assert_array_equal(np.array(im.quantization[1]), H.ijg_table(H.STD_CHROMA, q), err_msg=f"chroma q={q}")


def test_zigzag_is_a_permutation_and_raw_dqt_roundtrips_through_pillow_header():
    assert sorted(H.ZIGZAG.tolist()) == list(range(64))
    assert H.ZIGZAG[:6].tolist() == [0, 1, 8, 16, 9, 2]
    t = H.ijg_table(H.STD_LUMA, 80)
    raw = H.raw_dqt_jpeg({0: t})
    assert raw[:4] == b"\xff\xd8\xff\xdb" and raw[-2:] == b"\xff\xd9"
    seg = raw[6:6 + 65]
    assert seg[0] == 0 and bytes(seg[1:]) == bytes(int(v) for v in t[H.ZIGZAG])


def test_orient_matches_pil_transpose_for_all_eight():
    a = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    m = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
         5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}
    np.testing.assert_array_equal(H.orient(a, 1), a)
    for o, t in m.items():
        np.testing.assert_array_equal(H.orient(a, o), np.asarray(Image.fromarray(a).transpose(t)), err_msg=f"orientation {o}")


def test_field_is_deterministic_in_range_and_has_the_requested_slope():
    a, b = H.field(256, seed=3), H.field(256, seed=3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, H.field(256, seed=4))
    assert a.min() >= 0 and a.max() <= 1 and abs(a.mean() - 0.5) < 0.02 and abs(a.std() - 0.15) < 0.02
    P = np.abs(np.fft.fft2(a - a.mean())) ** 2
    nu = np.hypot(np.fft.fftfreq(256)[None, :], np.fft.fftfreq(256)[:, None])
    lo = P[(nu > 0.02) & (nu < 0.04)].mean()
    hi = P[(nu > 0.16) & (nu < 0.32)].mean()
    slope = np.log(hi / lo) / np.log(0.24 / 0.03)
    assert -2.4 < slope < -1.6


def test_containers_with_injected_metadata_still_decode_in_pillow():
    base = H.to_u8(H.rgb3(H.field(32, seed=1)))
    for data in (H.png_with_c2pa(H.png_bytes(base)), H.png_with_srgb_chunk(H.png_bytes(base)),
                 H.jpeg_with_c2pa(H.jpeg_bytes(base)), H.webp_with_c2pa(H.webp_bytes(base))):
        assert Image.open(io.BytesIO(data)).convert("RGB").size == (32, 32)
    np.testing.assert_array_equal(H.pil_decode(H.png_with_c2pa(H.png_bytes(base), ai_claim=True)), base)
    assert Image.open(io.BytesIO(H.png_with_srgb_chunk(H.png_bytes(base)))).info.get("srgb") == 0


def test_make_exif_roundtrips_through_pillow():
    base = H.to_u8(H.rgb3(H.field(16, seed=1)))
    ex = H.make_exif(orientation=6, make="Acme", focal=50.0, f35=75, fnumber=2.8, exposure=0.004, iso=400, private=True)
    im = Image.open(io.BytesIO(H.png_bytes(base, exif=ex)))
    e = im.getexif()
    assert e[0x0112] == 6 and e[0x010F] == "Acme"
    sub = e.get_ifd(0x8769)
    assert sub[0x920A] == 50.0 and sub[0xA405] == 75 and sub[0x8827] == 400
    assert e[0x013B] == H.PRIVATE_MARK and e[0x8298] == H.PRIVATE_MARK
    assert len(e.get_ifd(0x8825)) == 4 and sub[0xA431] == H.PRIVATE_MARK
