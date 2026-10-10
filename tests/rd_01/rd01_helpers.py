"""RD-01 테스트 공용 도우미: 합성 영상(1/f 장), 레터박스, ICC 프로파일 제작기, PNG/JPEG/WebP 메타 주입기, IJG 표준 양자화표.

여기 있는 코드는 RD-01 구현(buc.read.ingest)을 참조하지 않는다. numpy, Pillow, FND-06(color/pyramid)만 쓴다.
ICC·EXIF·C2PA 바이트는 구현과 독립으로 ICC.1(v2) / ISO 19566-5(JUMBF) / TIFF 사양에서 직접 만든다."""
import io
import struct
import zlib

import numpy as np
from PIL import Image

from buc.core import color

# ---------------------------------------------------------------------------------------------------------------
# 합성 영상
# ---------------------------------------------------------------------------------------------------------------


def field(h, w=None, *, seed=0, amp_exp=1.0, std=0.15, mean=0.5):
    """진폭 스펙트럼 ν^-amp_exp (전력 ν^-2amp_exp) 의 무작위 위상 장. float64, [0,1] 로 클립. 결정론."""
    w = h if w is None else w
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    nu = np.hypot(fx, fy)
    nu[0, 0] = 1.0
    spec = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))) * nu ** (-amp_exp)
    spec[0, 0] = 0.0
    x = np.fft.ifft2(spec).real
    x = (x - x.mean()) / x.std() * std + mean
    return np.clip(x, 0.0, 1.0)


def upsampled_field(out, k, *, seed=0, method="bicubic"):
    """(out/k)² 크기에서 만든 장을 FND-06 resize 로 k 배 키운다 -> 실효 해상도가 1/k 인 float32 (out,out)."""
    base = field(out // k, seed=seed).astype(np.float32)
    if k == 1:
        return base
    from buc.core import pyramid
    return np.clip(pyramid.resize(base, (out, out), method), 0.0, 1.0).astype(np.float32)


def rgb3(gray):
    g = np.asarray(gray)
    return np.ascontiguousarray(np.stack([g, g, g], axis=-1))


def to_u8(x):
    return np.round(np.clip(np.asarray(x, np.float64), 0, 1) * 255).astype(np.uint8)


def to_u16(x):
    return np.round(np.clip(np.asarray(x, np.float64), 0, 1) * 65535).astype(np.uint16)


def add_bars(img, top=0, bottom=0, left=0, right=0, fill=0):
    h, w = img.shape[:2]
    out = np.full((h + top + bottom, w + left + right) + img.shape[2:], fill, dtype=img.dtype)
    out[top:top + h, left:left + w] = img
    return out


def orient(a, o):
    """EXIF Orientation o (1..8) 을 적용한 배열 (표시 방향). PIL Image.transpose 와 같은 의미를 numpy 로 쓴 것."""
    return {1: a, 2: a[:, ::-1], 3: a[::-1, ::-1], 4: a[::-1], 5: a.transpose(1, 0, 2), 6: np.rot90(a, -1),
            7: a.transpose(1, 0, 2)[::-1, ::-1], 8: np.rot90(a, 1)}[o]


# ---------------------------------------------------------------------------------------------------------------
# 파일 인코딩
# ---------------------------------------------------------------------------------------------------------------
def png_bytes(arr, **kw):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG", **kw)
    return b.getvalue()


def jpeg_bytes(arr, quality=90, **kw):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG", quality=quality, **kw)
    return b.getvalue()


def webp_bytes(arr, **kw):
    kw.setdefault("lossless", True)
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "WEBP", **kw)
    return b.getvalue()


def pil_decode(data):
    """Pillow 로 EXIF 회전·색관리 없이 그대로 푼 uint8 RGB."""
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# ---------------------------------------------------------------------------------------------------------------
# 컨테이너 주입기
# ---------------------------------------------------------------------------------------------------------------
def png_insert_chunk(data, ctype, payload):
    """첫 IDAT 앞에 청크를 끼운다 (CRC 포함)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        if data[pos + 4:pos + 8] == b"IDAT":
            break
        pos += 12 + n
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return data[:pos] + struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc) + data[pos:]


def jpeg_insert_segment(data, marker, payload):
    """SOI 바로 뒤에 마커 세그먼트(0xFFxx + 길이 + 페이로드)를 끼운다."""
    assert data[:2] == b"\xff\xd8"
    return data[:2] + bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload + data[2:]


def webp_append_chunk(data, fourcc, payload):
    ch = fourcc + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) % 2 else b"")
    d = data + ch
    return d[:4] + struct.pack("<I", len(d) - 8) + d[8:]


def jumbf_box(tbox, payload):
    return struct.pack(">I", 8 + len(payload)) + tbox + payload


def c2pa_jumbf(ai_claim=False, label=b"c2pa"):
    """최소 JUMBF 슈퍼박스(jumb ⊃ jumd(label) + CBOR 흉내 본문). ai_claim 이면 IPTC digitalSourceType 문자열을 싣는다."""
    jumd = jumbf_box(b"jumd", bytes.fromhex("63327061" "00110010" "800000AA00389B71") + b"\x03" + label + b"\0")
    body = b"actions" + (b" http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia" if ai_claim else b"")
    return jumbf_box(b"jumb", jumd + jumbf_box(b"cbor", body))


def jpeg_with_c2pa(data, ai_claim=False, label=b"c2pa"):
    box = c2pa_jumbf(ai_claim, label)
    return jpeg_insert_segment(data, 0xEB, b"JP" + struct.pack(">H", 1) + struct.pack(">I", 1) + box)


def png_with_c2pa(data, ai_claim=False):
    return png_insert_chunk(data, b"caBX", c2pa_jumbf(ai_claim))


def webp_with_c2pa(data, ai_claim=False):
    return webp_append_chunk(data, b"C2PA", c2pa_jumbf(ai_claim))


def png_with_srgb_chunk(data, intent=0):
    return png_insert_chunk(data, b"sRGB", bytes([intent]))


# ---------------------------------------------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------------------------------------------
def make_exif(*, orientation=None, make=None, model=None, focal=None, f35=None, fnumber=None, exposure=None,
              iso=None, private=False):
    """Pillow Exif 객체. private=True 이면 GPS·날짜·작성자·저작권·시리얼 같은 개인정보 태그를 심는다."""
    ex = Image.Exif()
    ex[0x0131] = SOFTWARE_MARK            # IFD0 가 비면 Pillow 가 하위 IFD 까지 버린다: 화이트리스트 밖 태그를 항상 하나 둔다
    if orientation is not None:
        ex[0x0112] = orientation
    if make is not None:
        ex[0x010F] = make
    if model is not None:
        ex[0x0110] = model
    sub = ex.get_ifd(0x8769)
    if focal is not None:
        sub[0x920A] = focal
    if f35 is not None:
        sub[0xA405] = f35
    if fnumber is not None:
        sub[0x829D] = fnumber
    if exposure is not None:
        sub[0x829A] = exposure
    if iso is not None:
        sub[0x8827] = iso
    if private:
        ex[0x0132] = "2031:01:02 03:04:05"
        ex[0x013B] = PRIVATE_MARK
        ex[0x8298] = PRIVATE_MARK
        ex[0x010E] = PRIVATE_MARK
        sub[0xA431] = PRIVATE_MARK
        gps = ex.get_ifd(0x8825)
        gps[1] = "N"
        gps[2] = (37.0, 25.0, 19.0)
        gps[3] = "E"
        gps[4] = (126.0, 58.0, 41.0)
    return ex


PRIVATE_MARK = "SECRET-7731-MARK"
SOFTWARE_MARK = "rd01-software-mark"

# ---------------------------------------------------------------------------------------------------------------
# JPEG 양자화표 (IJG)
# ---------------------------------------------------------------------------------------------------------------
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])
STD_CHROMA = np.array([
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32)
ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def ijg_table(std, q):
    """libjpeg jpeg_quality_scaling + jpeg_add_quant_table (force_baseline). 자연(행 우선) 순서 (64,) int."""
    q = max(1, min(100, int(q)))
    scale = 5000 // q if q < 50 else 200 - q * 2
    return np.clip((std * scale + 50) // 100, 1, 255).astype(np.int64)


def raw_dqt_jpeg(tables_natural, extra_16bit=None, merge=False):
    """DQT 만 있는 최소 바이트열: SOI + DQT 세그먼트(들) + EOI. 값은 지그재그 순서로 직렬화한다.
    tables_natural: {id: (64,) 자연순서}. extra_16bit: {id: (64,)} 이면 Pq=1(16비트) 로 쓴다."""
    out = b"\xff\xd8"
    segs = []
    for tid, t in sorted(tables_natural.items()):
        zz = np.asarray(t)[ZIGZAG]
        segs.append(bytes([tid]) + bytes(int(v) for v in zz))
    if merge:
        segs = [b"".join(segs)]
    for payload in segs:
        out += b"\xff\xdb" + struct.pack(">H", len(payload) + 2) + payload
    for tid, t in sorted((extra_16bit or {}).items()):
        zz = np.asarray(t)[ZIGZAG]
        payload = bytes([0x10 | tid]) + b"".join(struct.pack(">H", int(v)) for v in zz)
        out += b"\xff\xdb" + struct.pack(">H", len(payload) + 2) + payload
    return out + b"\xff\xd9"


# ---------------------------------------------------------------------------------------------------------------
# ICC 프로파일 제작기 (v2 matrix/TRC)
# ---------------------------------------------------------------------------------------------------------------
D65_XY = (0.3127, 0.3290)
D50_XYZ = np.array([0.9642, 1.0, 0.8249])
_BFD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
PRIMS = {
    "srgb": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "display-p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
    "adobe-rgb": ((0.640, 0.330), (0.210, 0.710), (0.150, 0.060)),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
}


def _xyz(xy):
    return np.array([xy[0] / xy[1], 1.0, (1 - xy[0] - xy[1]) / xy[1]])


def colorants(prim_xy, white_xy=D65_XY):
    """원색 xy 로부터 D50 PCS(브래드퍼드) 착색제 행렬 (열 = r,g,b 의 XYZ)."""
    P = np.stack([_xyz(p) for p in prim_xy], axis=1)
    w = _xyz(white_xy)
    M = P * np.linalg.solve(P, w)
    s, d = _BFD @ w, _BFD @ D50_XYZ
    return np.linalg.inv(_BFD) @ np.diag(d / s) @ _BFD @ M


def _s15(v):
    return struct.pack(">i", int(round(float(v) * 65536)))


def _xyz_tag(v):
    return b"XYZ \0\0\0\0" + b"".join(_s15(c) for c in v)


def trc_curv(fn, n=1024):
    x = np.linspace(0.0, 1.0, n)
    y = np.round(np.clip(fn(x), 0, 1) * 65535).astype(">u2")
    return b"curv\0\0\0\0" + struct.pack(">I", n) + y.tobytes()


def trc_gamma(g):
    return b"curv\0\0\0\0" + struct.pack(">I", 1) + struct.pack(">H", int(round(g * 256)))


def trc_linear():
    return b"curv\0\0\0\0" + struct.pack(">I", 0)


def trc_para_srgb():
    return b"para\0\0\0\0" + struct.pack(">HH", 3, 0) + b"".join(_s15(v) for v in (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045))


def srgb_eotf(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def rec2020_eotf(x):
    return color.eotf(np.asarray(x, np.float64), "rec2020")


def _desc(text):
    a = text.encode("ascii") + b"\0"
    return b"desc\0\0\0\0" + struct.pack(">I", len(a)) + a + struct.pack(">II", 0, 0) + struct.pack(">HB", 0, 0) + b"\0" * 67


def make_icc(col, trc, desc="test profile", reverse_tags=False, data_space=b"RGB ", cls=b"mntr"):
    """행렬/TRC 입력 프로파일 바이트. col: (3,3) 착색제(열=r,g,b). trc: TRC 태그 본문(3채널 공용)."""
    tags = [(b"desc", _desc(desc)), (b"wtpt", _xyz_tag(D50_XYZ)),
            (b"rXYZ", _xyz_tag(col[:, 0])), (b"gXYZ", _xyz_tag(col[:, 1])), (b"bXYZ", _xyz_tag(col[:, 2])),
            (b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc), (b"cprt", b"text\0\0\0\0test\0")]
    if reverse_tags:
        tags = tags[::-1]
    pad = lambda b: b + b"\0" * (-len(b) % 4)  # noqa: E731
    table_len = 4 + 12 * len(tags)
    off = 128 + table_len
    table, body, cache = b"", b"", {}
    for sig, data in tags:
        if data in cache and sig.endswith(b"TRC"):
            o, n = cache[data]
        else:
            o, n = off + len(body), len(data)
            body += pad(data)
            cache[data] = (o, n)
        table += sig + struct.pack(">II", o, n)
    total = 128 + table_len + len(body)
    hdr = bytearray(128)
    hdr[0:4] = struct.pack(">I", total)
    hdr[8:12] = struct.pack(">I", 0x02400000)
    hdr[12:16] = cls
    hdr[16:20] = data_space
    hdr[20:24] = b"XYZ "
    hdr[36:40] = b"acsp"
    hdr[68:80] = b"".join(_s15(c) for c in D50_XYZ)
    return bytes(hdr) + struct.pack(">I", len(tags)) + table + body


def icc_for(space, form="table", **kw):
    """표준 색공간 프로파일. form: table(1024점 곡선) | para(매개변수 곡선, sRGB 계열만) | gamma22 | linear."""
    col = colorants(PRIMS[space])
    if form == "para":
        trc = trc_para_srgb()
    elif form == "gamma22":
        trc = trc_gamma(2.2)
    elif form == "gamma-adobe":
        trc = trc_gamma(563 / 256)
    elif form == "linear":
        trc = trc_linear()
    else:
        fn = {"srgb": srgb_eotf, "display-p3": srgb_eotf, "rec2020": rec2020_eotf,
              "adobe-rgb": lambda x: np.asarray(x) ** (563 / 256)}[space]
        trc = trc_curv(fn)
    return make_icc(col, trc, desc=f"test {space} {form}", **kw)


PROPHOTO_XY = ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001))


def icc_prophoto():
    """COLOR_SPACES 에 없는 광색역 프로파일 (감마 1.8, 화이트 D50)."""
    return make_icc(colorants(PROPHOTO_XY, white_xy=(0.3457, 0.3585)), trc_gamma(1.8), desc="test prophoto")
