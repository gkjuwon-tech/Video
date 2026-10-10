"""RD-01 색 관리: ICC 분류, 색공간 결정 우선순위, 광색역 -> Rec.709 선형 변환과 색역 처리."""
import numpy as np
import pytest
from PIL import ImageCms

import rd01_helpers as H
from buc.core import color
from buc.read import ingest as ing

RNG = np.random.default_rng(2024)


# ---------------------------------------------------------------------------------------------------------------
# classify_icc
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("space,form,want", [
    ("srgb", "table", "srgb"),
    ("srgb", "para", "srgb"),
    ("srgb", "gamma22", "srgb"),
    ("srgb", "linear", "linear-srgb"),
    ("display-p3", "table", "display-p3"),
    ("display-p3", "para", "display-p3"),
    ("adobe-rgb", "gamma-adobe", "adobe-rgb"),
    ("adobe-rgb", "table", "adobe-rgb"),
    ("rec2020", "table", "rec2020"),
])
def test_classify_known_profiles(space, form, want):
    c = ing.classify_icc(H.icc_for(space, form))
    assert c.valid is True and c.space == want
    assert c.description == f"test {space} {form}"


def test_classify_is_independent_of_tag_order():
    c = ing.classify_icc(H.icc_for("display-p3", "table", reverse_tags=True))
    assert (c.valid, c.space) == (True, "display-p3")


def test_classify_real_lcms_srgb_profile():
    data = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    c = ing.classify_icc(data)
    assert (c.valid, c.space) == (True, "srgb")


def test_classify_valid_but_unrecognized_profiles_are_unknown_not_invalid():
    for data in (H.icc_prophoto(),                                    # 원색이 목록 밖
                 H.icc_for("rec2020", "linear"),                      # 원색은 맞지만 곡선이 목록 밖
                 H.icc_for("adobe-rgb", "linear"),
                 H.make_icc(H.colorants(H.PRIMS["srgb"]), H.trc_gamma(1.0), data_space=b"GRAY"),    # RGB 가 아님
                 H.make_icc(H.colorants(H.PRIMS["srgb"]), H.trc_gamma(1.0), data_space=b"CMYK")):
        c = ing.classify_icc(data)
        assert c.valid is True and c.space == "unknown", c


def test_classify_garbage_is_invalid_and_never_raises():
    good = H.icc_for("srgb", "table")
    bad = [b"", b"junk", bytes(200), good[:100], good[:len(good) // 2], good[:36] + b"xxxx" + good[40:],
           bytes([good[0] ^ 0xFF]) + good[1:], bytes(RNG.integers(0, 256, 4096, dtype=np.uint8))]
    for data in bad:
        c = ing.classify_icc(data)
        assert c.valid is False and c.space == "unknown", data[:8]
    with pytest.raises(TypeError):
        ing.classify_icc("not bytes")


# ---------------------------------------------------------------------------------------------------------------
# to_linear_709: 색역 처리
# ---------------------------------------------------------------------------------------------------------------
def test_srgb_is_bit_identical_to_fnd06_decode():
    px = RNG.integers(0, 256, (24, 30, 3), dtype=np.uint8)
    lin, frac = ing.to_linear_709(px, "srgb")
    assert lin.dtype == np.float32 and lin.shape == px.shape and lin.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(lin, color.decode_to_linear(px, "srgb"))
    assert frac == 0.0


@pytest.mark.parametrize("space", ["display-p3", "adobe-rgb", "rec2020"])
def test_in_gamut_pixels_are_untouched_and_out_of_gamut_are_desaturated_with_luminance_kept(space):
    px = RNG.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    px[0, :3] = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]                 # 원색은 반드시 sRGB 밖
    raw = color.decode_to_linear(px, space).astype(np.float64)
    lin, frac = ing.to_linear_709(px, space)
    assert lin.min() >= 0.0 and np.isfinite(lin).all()
    neg = raw.min(axis=-1) < -1e-6
    assert neg.any() and not neg.all()
    assert frac == pytest.approx(neg.mean(), abs=1e-12)
    # 색역 안 화소: 값이 같다 (-1e-6 이내의 부동소수 잡음은 0 으로)
    keep = raw.min(axis=-1) >= 0
    np.testing.assert_array_equal(lin[keep], raw[keep].astype(np.float32))
    # 색역 밖 화소: 휘도 보존, 색상 방향 보존, 어떤 채널은 정확히 0 (최소 탈채도)
    o, r = lin[neg].astype(np.float64), raw[neg]
    np.testing.assert_allclose(color.luminance(o), color.luminance(r), rtol=1e-5, atol=1e-7)
    Y = color.luminance(r)[:, None]
    d_in, d_out = r - Y, o - Y
    cos = (d_in * d_out).sum(-1) / (np.linalg.norm(d_in, axis=-1) * np.linalg.norm(d_out, axis=-1) + 1e-30)
    assert cos.min() > 1 - 1e-5
    assert (np.abs(o.min(axis=-1)) < 2e-7).all()
    assert (np.linalg.norm(d_out, axis=-1) <= np.linalg.norm(d_in, axis=-1) + 1e-7).all()


def test_p3_gray_ramp_has_no_gamut_clip_and_matches_decode():
    g = np.arange(256, dtype=np.uint8)
    px = np.stack([g, g, g], axis=-1)[None]
    lin, frac = ing.to_linear_709(px, "display-p3")
    assert frac == 0.0 and lin.min() >= 0.0
    np.testing.assert_allclose(lin, color.decode_to_linear(px, "display-p3"), atol=2e-6)
    np.testing.assert_allclose(lin[..., 0], lin[..., 1], atol=2e-6)       # 무채색은 무채색으로


def test_float32_values_above_one_are_kept_and_unknown_is_treated_as_srgb():
    x = np.array([[[0.2, 0.5, 1.0], [1.5, 0.1, 0.0]]], np.float32)
    lin, _ = ing.to_linear_709(x, "srgb")
    np.testing.assert_array_equal(lin, color.decode_to_linear(x, "srgb"))
    assert lin.max() > 1.0
    np.testing.assert_array_equal(ing.to_linear_709(x, "unknown")[0], lin)
    with pytest.raises(ValueError):
        ing.to_linear_709(x, "bt2100")
    with pytest.raises(TypeError):
        ing.to_linear_709(x.astype(np.float64), "srgb")


# ---------------------------------------------------------------------------------------------------------------
# ingest: 색공간 결정 우선순위 (caller > ICC > PNG sRGB 청크 > 가정)
# ---------------------------------------------------------------------------------------------------------------
def _pix(h=40, w=48, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_untagged_png_is_assumed_srgb_with_warning():
    px = _pix()
    r = ing.ingest(H.png_bytes(px))
    assert (r.color.space, r.color.source, r.color.icc_sha256) == ("srgb", "assumed", None)
    assert r.image.color_space == "srgb" and "color_space_assumed" in r.warnings
    np.testing.assert_array_equal(r.image_lin, color.decode_to_linear(px, "srgb"))


def test_png_srgb_chunk_is_a_tag_not_an_assumption():
    r = ing.ingest(H.png_with_srgb_chunk(H.png_bytes(_pix())))
    assert (r.color.space, r.color.source) == ("srgb", "png-srgb")
    assert "color_space_assumed" not in r.warnings


@pytest.mark.parametrize("container", ["png", "jpeg", "webp"])
@pytest.mark.parametrize("space,form", [("display-p3", "para"), ("adobe-rgb", "gamma-adobe"), ("rec2020", "table")])
def test_icc_profile_decides_color_space_in_every_container(container, space, form):
    icc = H.icc_for(space, form)
    px = np.clip(_pix(seed=3) // 2 + 60, 0, 255).astype(np.uint8)       # 중간 톤: 색역 처리 변수를 줄인다
    enc = {"png": H.png_bytes, "jpeg": H.jpeg_bytes, "webp": H.webp_bytes}[container]
    data = enc(px, icc_profile=icc)
    r = ing.ingest(data)
    assert (r.color.space, r.color.source) == (space, "icc")
    assert r.image.color_space == space and r.color.icc_sha256 is not None and len(r.color.icc_sha256) == 64
    assert "color_space_assumed" not in r.warnings
    dec = H.pil_decode(data)                                              # 같은 디코더가 낸 부호화 값에서 출발
    want, _ = ing.to_linear_709(dec, space)
    np.testing.assert_array_equal(r.image.pixels, dec)                    # 픽셀은 변환하지 않고 보존한다
    np.testing.assert_array_equal(r.image_lin, want)


def test_p3_image_is_more_saturated_than_the_same_codes_read_as_srgb():
    px = np.zeros((40, 48, 3), np.uint8)
    px[..., 1] = 200                                                       # P3 초록: sRGB 로 보면 R 이 음수여야 한다
    px[..., 0] = 40
    p3 = ing.ingest(H.png_bytes(px, icc_profile=H.icc_for("display-p3", "para")))
    s = ing.ingest(H.png_bytes(px, icc_profile=H.icc_for("srgb", "para")))
    assert p3.gamut_clip_frac == 1.0 and "gamut_clipped" in p3.warnings
    assert s.gamut_clip_frac == 0.0 and "gamut_clipped" not in s.warnings
    assert p3.image_lin.min() >= 0
    # 같은 부호값이라도 P3 해석이 더 채도가 높다 (R/G 비가 더 작다)
    assert (p3.image_lin[0, 0, 0] / p3.image_lin[0, 0, 1]) < (s.image_lin[0, 0, 0] / s.image_lin[0, 0, 1])


def test_caller_color_space_overrides_icc_and_flags_conflict():
    px = _pix()
    data = H.png_bytes(px, icc_profile=H.icc_for("display-p3", "para"))
    r = ing.ingest(data, color_space="srgb")
    assert (r.color.space, r.color.source) == ("srgb", "caller")
    assert "color_space_conflict" in r.warnings
    np.testing.assert_array_equal(r.image_lin, color.decode_to_linear(px, "srgb"))
    same = ing.ingest(data, color_space="display-p3")
    assert same.color.source == "caller" and "color_space_conflict" not in same.warnings
    with pytest.raises(ValueError):
        ing.ingest(data, color_space="bt2100")


def test_unrecognized_icc_keeps_unknown_but_decodes_like_srgb():
    px = _pix(seed=5)
    r = ing.ingest(H.png_bytes(px, icc_profile=H.icc_prophoto()))
    assert (r.color.space, r.color.source) == ("unknown", "icc")
    assert r.image.color_space == "unknown" and "icc_unrecognized" in r.warnings
    np.testing.assert_array_equal(r.image_lin, color.decode_to_linear(px, "srgb"))


def test_corrupt_icc_is_ignored_with_warning_not_an_error():
    px = _pix(seed=6)
    r = ing.ingest(H.png_bytes(px, icc_profile=b"this is not an icc profile" * 8))
    assert (r.color.space, r.color.source) == ("srgb", "assumed")
    assert "icc_invalid" in r.warnings and "color_space_assumed" in r.warnings
    np.testing.assert_array_equal(r.image_lin, color.decode_to_linear(px, "srgb"))


def test_ndarray_paths_use_the_declared_space_and_keep_dtype():
    for dt, hi in ((np.uint8, 255), (np.uint16, 65535)):
        a = RNG.integers(0, hi + 1, (20, 24, 3)).astype(dt)
        r = ing.ingest(a, color_space="adobe-rgb")
        assert r.image.pixels.dtype == dt and r.image.color_space == "adobe-rgb" and r.color.source == "caller"
        want, _ = ing.to_linear_709(a, "adobe-rgb")
        np.testing.assert_array_equal(r.image_lin, want)
    f = RNG.uniform(0, 1, (20, 24, 3)).astype(np.float32)
    r = ing.ingest(f)
    assert r.image.pixels.dtype == np.float32 and r.color.source == "assumed" and "color_space_assumed" in r.warnings
    np.testing.assert_array_equal(r.image_lin, color.decode_to_linear(f, "srgb"))


def test_real_lcms_profile_in_a_png_is_classified_end_to_end():
    prof = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    r = ing.ingest(H.png_bytes(_pix(seed=8), icc_profile=prof))
    assert (r.color.space, r.color.source) == ("srgb", "icc")
