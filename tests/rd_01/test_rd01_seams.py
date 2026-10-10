"""RD-01 이음매·오류·결정론: FND-02(InputImage/ReadPacket/check_stage_io S-1)와 맞물리는지, meta.input 직렬화, 입력 검증."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import rd01_helpers as H
from buc.core import color, contracts
from buc.read import ingest as ing

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PX = H.to_u8(H.rgb3(H.field(96, 128, seed=21)))
REQUIRED_META = {"h_I", "raw_sha256", "format", "r_in", "r_eff", "color_space", "color_source", "icc_sha256", "crop_box", "src_size",
                 "orientation", "orientation_applied", "exif", "fov_x_prior_deg", "c2pa", "jpeg", "consent", "gamut_clip_frac",
                 "warnings", "prov", "ext", "comp", "hold"}


def _read_packet(r, **kw):
    """ingest 결과로 S-1 출력(ReadPacket)을 만든다: RD-21 이 할 일의 축소판 (깊이·카메라는 가짜)."""
    h, w = r.image.pixels.shape[:2]
    K = np.array([[1.2 * w, 0, (w - 1) / 2], [0, 1.2 * w, (h - 1) / 2], [0, 0, 1.0]])
    obs = contracts.Obs(K=K, R=np.eye(3), t=np.zeros(3), width=w, height=h)
    return contracts.ReadPacket(image_lin=r.image_lin, depth=np.full((h, w), 2.0, np.float32), obs=obs,
                                theta_in=contracts.Theta(), input_hash=r.image.h_I, prov=r.image.prov,
                                meta={"input": r.meta_input()}, **kw)


def test_ingest_result_feeds_stage_s_minus_1_contract_check():
    for data in (H.png_bytes(PX), H.png_bytes(H.add_bars(PX, 12, 12, 0, 0)), H.jpeg_bytes(PX)):
        r = ing.ingest(data)
        read = _read_packet(r)
        contracts.check_stage_io("S-1", {"image": r.image}, {"read": read})        # 크기·해시·taint 교차 검사
        assert read.image_lin.shape == r.image.pixels.shape
        back = contracts.loads(contracts.dumps(read))                               # meta.input 이 JSON 왕복 가능
        assert back.meta["input"]["h_I"] == r.image.h_I
        assert back.content_hash() == read.content_hash()


def test_letterboxed_input_gives_cropped_image_whose_hash_matches_read_packet():
    r = ing.ingest(H.png_bytes(H.add_bars(PX, 12, 12, 0, 0)))
    assert r.image.pixels.shape[:2] == (96, 128) and r.src_size == (120, 128)
    assert r.image.h_I == contracts.content_hash(r.image.pixels) == _read_packet(r).input_hash


def test_output_types_dtypes_and_domain():
    r = ing.ingest(H.png_bytes(PX))
    assert isinstance(r.image, contracts.InputImage) and r.image.pixels.dtype == np.uint8
    assert r.image_lin.dtype == np.float32 and r.image_lin.flags["C_CONTIGUOUS"] and r.image_lin.shape == (96, 128, 3)
    assert np.isfinite(r.image_lin).all() and r.image_lin.min() >= 0
    assert isinstance(r.crop_box, tuple) and all(type(v) is int for v in r.crop_box)
    assert isinstance(r.warnings, tuple) and list(r.warnings) == sorted(set(r.warnings))
    assert r.r_in == 96 and 0 < r.r_eff <= r.r_in and r.fmt == "png" and r.raw_sha256 == hashlib.sha256(H.png_bytes(PX)).hexdigest()
    assert r.crop_box == (0, 0, 128, 96) and r.src_size == (96, 128) and r.bars == (0, 0, 0, 0)
    assert "long_side_out_of_range" in r.warnings                      # 긴 변 128 < 512
    big = ing.ingest(np.zeros((8, 600, 3), np.uint8) + 100, color_space="srgb")
    assert "long_side_out_of_range" not in big.warnings


def test_meta_input_is_json_complete_and_consistent():
    ex = H.make_exif(make="Acme", f35=50)
    r = ing.ingest(H.jpeg_bytes(PX, quality=80, exif=ex, icc_profile=H.icc_for("display-p3", "para")))
    m = r.meta_input()
    assert REQUIRED_META <= set(m)
    assert json.loads(json.dumps(m)) == m
    assert m["h_I"] == r.image.h_I and m["format"] == "jpeg" and m["color_space"] == "display-p3" and m["color_source"] == "icc"
    assert m["r_in"] == r.r_in and m["r_eff"] == r.r_eff and m["crop_box"] == list(r.crop_box) and m["src_size"] == [96, 128]
    assert m["jpeg"]["quality"] == 80 and m["jpeg"]["exact"] is True and m["exif"]["make"] == "Acme"
    assert m["icc_sha256"] == r.color.icc_sha256 and m["warnings"] == list(r.warnings)
    assert m["consent"] is True and m["ext"] is False and m["comp"] is False and m["hold"] is False
    assert m["prov"] == {"taint": ["USER"], "mode": "engine", "roots": [], "ns": "user_vault"}
    assert ing.ingest(H.png_bytes(PX)).meta_input()["jpeg"] is None


def test_provenance_is_passed_through_untouched_and_default_is_user_vault():
    d = ing.ingest(H.png_bytes(PX))
    assert d.image.prov.taint == frozenset({"USER"}) and d.image.prov.ns == "user_vault" and d.image.consent is True
    p = contracts.Prov(taint=frozenset({"EXTGEN:soul", "HOLDOUT"}), mode="inspired", roots=("b", "a"), ns="ext_vault")
    r = ing.ingest(H.png_bytes(PX), prov=p, consent=False)
    assert r.image.prov.taint == p.taint and r.image.prov.mode == "inspired" and r.image.prov.roots == ("a", "b")
    assert r.image.prov.ns == "ext_vault" and r.image.consent is False
    m = r.meta_input()
    assert (m["ext"], m["hold"], m["comp"], m["consent"]) == (True, True, False, False)
    with pytest.raises(TypeError):
        ing.ingest(H.png_bytes(PX), prov={"taint": ["USER"]})


def test_deterministic_and_pure():
    data = H.png_bytes(H.add_bars(PX, 8, 8, 0, 0), exif=H.make_exif(make="Acme", f35=35), icc_profile=H.icc_for("adobe-rgb", "gamma-adobe"))
    s0 = np.random.get_state()[1].copy()
    a, b = ing.ingest(data), ing.ingest(data)
    assert (np.random.get_state()[1] == s0).all()
    assert a.image_lin.tobytes() == b.image_lin.tobytes() and a.image.pixels.tobytes() == b.image.pixels.tobytes()
    assert a.r_eff == b.r_eff and a.meta_input() == b.meta_input() and a.image.h_I == b.image.h_I
    assert json.dumps(a.meta_input(), sort_keys=True) == json.dumps(b.meta_input(), sort_keys=True)


def test_ndarray_input_is_copied_never_aliased_or_modified():
    a = np.ascontiguousarray(PX.copy())
    keep = a.copy()
    r = ing.ingest(a, color_space="srgb")
    np.testing.assert_array_equal(a, keep)
    assert not np.shares_memory(r.image.pixels, a) and not np.shares_memory(r.image_lin, a)
    a[...] = 0
    np.testing.assert_array_equal(r.image.pixels, keep)
    non_contig = np.asfortranarray(keep)
    r2 = ing.ingest(non_contig, color_space="srgb")
    np.testing.assert_array_equal(r2.image.pixels, keep)
    assert r2.image.pixels.flags["C_CONTIGUOUS"]
    assert r2.raw_sha256 == r.raw_sha256 == hashlib.sha256(keep.tobytes()).hexdigest()


def test_path_and_bytes_like_inputs_are_equivalent(tmp_path):
    data = H.png_bytes(PX)
    f = tmp_path / "in.png"
    f.write_bytes(data)
    ref = ing.ingest(data)
    for src in (str(f), f, bytearray(data), memoryview(data)):
        r = ing.ingest(src)
        assert r.image.h_I == ref.image.h_I and r.raw_sha256 == ref.raw_sha256 and r.r_eff == ref.r_eff
    with pytest.raises(FileNotFoundError):
        ing.ingest(str(tmp_path / "missing.png"))


# ---------------------------------------------------------------------------------------------------------------
# 형식과 오류
# ---------------------------------------------------------------------------------------------------------------
def test_sniff_format():
    assert ing.sniff_format(H.png_bytes(PX)) == "png"
    assert ing.sniff_format(H.jpeg_bytes(PX)) == "jpeg"
    assert ing.sniff_format(H.webp_bytes(PX)) == "webp"
    assert ing.sniff_format(b"\x00\x00\x00\x18ftypheic" + bytes(32)) == "heic"
    assert ing.sniff_format(b"\x76\x2f\x31\x01" + bytes(32)) == "exr"
    assert ing.sniff_format(b"GIF89a" + bytes(32)) == "unknown"
    assert ing.sniff_format(b"") == "unknown"


def test_unsupported_formats_raise_unsupported_format_error():
    assert issubclass(ing.UnsupportedFormatError, ing.IngestError) and issubclass(ing.IngestError, ValueError)
    for data in (b"\x00\x00\x00\x18ftypheic" + bytes(64), b"\x76\x2f\x31\x01" + bytes(64), b"GIF89a" + bytes(64), b"just text"):
        with pytest.raises(ing.UnsupportedFormatError):
            ing.ingest(data)


def test_cmyk_jpeg_is_refused_not_silently_converted():
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("CMYK", (32, 32), (10, 20, 30, 40)).save(buf, "JPEG")
    with pytest.raises(ing.IngestError):
        ing.ingest(buf.getvalue())


def test_alpha_channel_opaque_is_dropped_and_translucent_is_refused():
    import io
    from PIL import Image
    rgba = np.dstack([PX, np.full(PX.shape[:2], 255, np.uint8)])
    r = ing.ingest(H.png_bytes(rgba))
    np.testing.assert_array_equal(r.image.pixels, PX)
    rgba[5:9, 5:9, 3] = 128
    with pytest.raises(ing.IngestError):
        ing.ingest(H.png_bytes(rgba))
    buf = io.BytesIO()
    Image.fromarray(PX[..., 0]).save(buf, "PNG")                              # 회색조 PNG 는 3채널로 복제
    g = ing.ingest(buf.getvalue())
    assert g.image.pixels.shape == (96, 128, 3) and (g.image.pixels[..., 0] == PX[..., 0]).all()


def test_broken_inputs_raise_ingest_error_never_crash():
    good = H.png_bytes(PX)
    for data in (b"", good[:60], good[:len(good) // 2], H.jpeg_bytes(PX)[:200], H.webp_bytes(PX)[:30]):
        with pytest.raises(ing.IngestError):
            ing.ingest(data)


@pytest.mark.parametrize("bad,exc", [
    (np.zeros((8, 8), np.uint8), ing.IngestError),
    (np.zeros((8, 8, 4), np.uint8), ing.IngestError),
    (np.zeros((8, 8, 1), np.uint8), ing.IngestError),
    (np.zeros((0, 8, 3), np.uint8), ing.IngestError),
    (np.zeros((8, 8, 3), np.float64), TypeError),
    (np.zeros((8, 8, 3), np.int32), TypeError),
    (np.full((8, 8, 3), np.nan, np.float32), ing.IngestError),
    (np.full((8, 8, 3), np.inf, np.float32), ing.IngestError),
    (12345, TypeError),
    (None, TypeError),
])
def test_ndarray_and_type_validation(bad, exc):
    with pytest.raises(exc):
        ing.ingest(bad)


def test_pixel_budget_is_enforced_before_heavy_work():
    data = H.png_bytes(PX)
    with pytest.raises(ing.IngestError):
        ing.ingest(data, opts=ing.IngestOpts(max_pixels=96 * 128 - 1))
    assert ing.ingest(data, opts=ing.IngestOpts(max_pixels=96 * 128)).r_in == 96


def test_8bit_and_16bit_encodings_of_the_same_image_agree_in_linear_light():
    f = H.field(96, 128, seed=22).astype(np.float32)
    lin8 = ing.ingest(H.to_u8(H.rgb3(f)), color_space="srgb").image_lin
    lin16 = ing.ingest(H.to_u16(H.rgb3(f)), color_space="srgb").image_lin
    linf = ing.ingest(H.rgb3(f), color_space="srgb").image_lin
    np.testing.assert_allclose(lin16, linf, atol=2e-5, rtol=1e-4)
    np.testing.assert_allclose(lin8, linf, atol=2e-3, rtol=2e-2)
    np.testing.assert_array_equal(linf, color.decode_to_linear(H.rgb3(f), "srgb"))


def test_import_is_light_no_torch_no_rng_side_effect():
    code = ("import sys, numpy as np; s0 = np.random.get_state()[1].copy();"
            "import buc.read.ingest;"
            "assert 'torch' not in sys.modules, 'torch imported';"
            "assert (np.random.get_state()[1] == s0).all()")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=50)
    assert p.returncode == 0, p.stderr[-800:]


def test_runtime_budget_for_a_2k_image_is_cpu_friendly():
    import time
    img = H.rgb3(H.field(1080, 1920, seed=3).astype(np.float32))
    t0 = time.perf_counter()
    r = ing.ingest(img, color_space="srgb")
    dt = time.perf_counter() - t0
    assert r.r_in == 1080 and dt < 8.0                      # 설계 목표 5 s, 러너 변동 여유를 둔다
