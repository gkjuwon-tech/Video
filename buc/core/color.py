"""FND-06 색 유틸: sRGB 전달함수, 색공간 EOTF/OETF, 원색 행렬, Lab, ΔE2000.

내부 계산은 float64. 행렬은 원색 xy 와 D65 에서 유도한다(반올림 표 금지). numpy 만 쓴다(torch 금지).
선형값은 장면 선형(Rec.709 원색)이 기준이며 색역 밖 음수는 보존한다(클램프는 ``linear_to_display`` 에서만).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "COLOR_SPACES", "WHITE_XYZ",
    "srgb_to_linear", "linear_to_srgb", "srgb8_to_linear", "linear_to_srgb8",
    "eotf", "oetf", "rgb_to_xyz_matrix", "xyz_to_rgb_matrix", "convert_linear", "decode_to_linear",
    "linear_to_display", "luminance", "rgb_lin_to_xyz", "xyz_to_rgb_lin", "xyz_to_lab", "lab_to_xyz",
    "rgb_lin_to_lab", "lab_to_rgb_lin", "delta_e_2000",
]

COLOR_SPACES = ("srgb", "display-p3", "rec2020", "linear-srgb", "adobe-rgb")

_D65_XY = (0.3127, 0.3290)
_PRIMARIES = {
    "srgb": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
    "display-p3": ((0.68, 0.32), (0.265, 0.69), (0.15, 0.06)),
    "rec2020": ((0.708, 0.292), (0.17, 0.797), (0.131, 0.046)),
    "adobe-rgb": ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)),
}
_PRIM_KEY = {"srgb": "srgb", "linear-srgb": "srgb", "display-p3": "display-p3", "rec2020": "rec2020",
             "adobe-rgb": "adobe-rgb"}

_REC2020_A = 1.09929682680944
_REC2020_B = 0.018053968510807
_ADOBE_GAMMA = 563.0 / 256.0


def _xy_to_XYZ(x, y):
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


WHITE_XYZ = _xy_to_XYZ(*_D65_XY)


def _derive(prim):
    P = np.stack([_xy_to_XYZ(*p) for p in prim], axis=1)
    S = np.linalg.solve(P, WHITE_XYZ)
    return P * S


_RGB2XYZ = {k: _derive(v) for k, v in _PRIMARIES.items()}
_XYZ2RGB = {k: np.linalg.inv(m) for k, m in _RGB2XYZ.items()}


def _space(name):
    if not isinstance(name, str) or name not in COLOR_SPACES:
        raise ValueError(f"색공간 {name!r}: {COLOR_SPACES} 중 하나여야 한다")
    return name


# ---------------------------------------------------------------------------------------------------------------
# 전달함수
# ---------------------------------------------------------------------------------------------------------------
def _float_in(x, name="x"):
    """float32 / float64 만 받는다. (float64 작업배열, 출력 dtype) 반환."""
    a = np.asarray(x)
    if a.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name}: float32/float64 여야 한다 (dtype {a.dtype})")
    return a.astype(np.float64), a.dtype


def _out(a64, dt):
    r = a64.astype(dt, copy=False) if a64.dtype != dt else a64
    return r[()] if r.ndim == 0 else r


def _srgb_dec64(v):
    a = np.abs(v)
    mag = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return np.copysign(mag, v)


def _srgb_enc64(v):
    a = np.abs(v)
    mag = np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / 2.4) - 0.055)
    return np.copysign(mag, v)


def srgb_to_linear(x):
    v, dt = _float_in(x)
    with np.errstate(invalid="ignore"):
        return _out(_srgb_dec64(v), dt)


def linear_to_srgb(x):
    v, dt = _float_in(x)
    with np.errstate(invalid="ignore"):
        return _out(_srgb_enc64(v), dt)


_LUT8 = _srgb_dec64(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)


def srgb8_to_linear(u8) -> np.ndarray:
    a = np.asarray(u8)
    if a.dtype != np.uint8:
        raise TypeError(f"u8: uint8 여야 한다 (dtype {a.dtype})")
    return _LUT8[a]


def linear_to_srgb8(x) -> np.ndarray:
    v, _ = _float_in(x)
    v = np.where(np.isnan(v), 0.0, np.clip(v, 0.0, 1.0))
    e = _srgb_enc64(v) * 255.0
    return np.floor(e + 0.5).clip(0, 255).astype(np.uint8)


def _rec2020_oetf(L):
    a = np.abs(L)
    mag = np.where(a < _REC2020_B, 4.5 * a, _REC2020_A * a ** 0.45 - (_REC2020_A - 1.0))
    return np.copysign(mag, L)


def _rec2020_eotf(V):
    a = np.abs(V)
    mag = np.where(a < 4.5 * _REC2020_B, a / 4.5, ((a + (_REC2020_A - 1.0)) / _REC2020_A) ** (1.0 / 0.45))
    return np.copysign(mag, V)


def eotf(x, space):
    """부호화값 -> 선형(해당 색공간의 원색 기준)."""
    _space(space)
    if space in ("srgb", "display-p3"):
        return srgb_to_linear(x)
    v, dt = _float_in(x)
    with np.errstate(invalid="ignore"):
        if space == "linear-srgb":
            r = v.copy()
        elif space == "rec2020":
            r = _rec2020_eotf(v)
        else:
            r = np.copysign(np.abs(v) ** _ADOBE_GAMMA, v)
    return _out(r, dt)


def oetf(x, space):
    """선형 -> 부호화값."""
    _space(space)
    if space in ("srgb", "display-p3"):
        return linear_to_srgb(x)
    v, dt = _float_in(x)
    with np.errstate(invalid="ignore"):
        if space == "linear-srgb":
            r = v.copy()
        elif space == "rec2020":
            r = _rec2020_oetf(v)
        else:
            r = np.copysign(np.abs(v) ** (1.0 / _ADOBE_GAMMA), v)
    return _out(r, dt)


# ---------------------------------------------------------------------------------------------------------------
# 행렬 / 변환
# ---------------------------------------------------------------------------------------------------------------
def rgb_to_xyz_matrix(space) -> np.ndarray:
    return _RGB2XYZ[_PRIM_KEY[_space(space)]].copy()


def xyz_to_rgb_matrix(space) -> np.ndarray:
    return _XYZ2RGB[_PRIM_KEY[_space(space)]].copy()


def _f64_last3(x, name):
    a = np.asarray(x)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name}: 수치 배열이어야 한다 (dtype {a.dtype})")
    a = np.array(a, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError(f"{name}: 마지막 축이 3 이어야 한다 (shape {a.shape})")
    return a


def convert_linear(rgb, src, dst) -> np.ndarray:
    """선형 RGB 를 src 원색에서 dst 원색으로. 같은 원색이면 항등(비트), 색역 밖 음수는 보존."""
    _space(src)
    _space(dst)
    a = _f64_last3(rgb, "rgb")
    if _PRIM_KEY[src] == _PRIM_KEY[dst]:
        return a
    M = _XYZ2RGB[_PRIM_KEY[dst]] @ _RGB2XYZ[_PRIM_KEY[src]]
    return a @ M.T


def decode_to_linear(pixels, color_space) -> np.ndarray:
    """uint8 / uint16 / float32 (...,3) 부호화 영상 -> Rec.709 선형 float32. 'unknown' 은 srgb 로 취급."""
    cs = "srgb" if color_space == "unknown" else color_space
    a = np.asarray(pixels)
    if a.dtype not in (np.uint8, np.uint16, np.float32):
        raise TypeError(f"pixels: uint8/uint16/float32 여야 한다 (dtype {a.dtype})")
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError(f"pixels: 마지막 축이 3 이어야 한다 (shape {a.shape})")
    _space(cs)
    if a.dtype == np.uint8:
        v = a.astype(np.float64) / 255.0
    elif a.dtype == np.uint16:
        v = a.astype(np.float64) / 65535.0
    else:
        v = a.astype(np.float64)
    lin = eotf(v, cs)
    lin = convert_linear(lin, cs, "srgb")
    return lin.astype(np.float32)


def linear_to_display(x) -> np.ndarray:
    """장면 선형 -> [0,1] 표시값(sRGB 부호화, 클램프). float32."""
    v, _ = _float_in(x)
    with np.errstate(invalid="ignore"):
        e = _srgb_enc64(np.clip(v, 0.0, 1.0))
    return np.clip(e, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# 휘도 / XYZ / Lab
# ---------------------------------------------------------------------------------------------------------------
def luminance(rgb_lin):
    a = np.asarray(rgb_lin)
    f32 = a.dtype == np.float32
    v = _f64_last3(a, "rgb_lin")
    y = v @ _RGB2XYZ["srgb"][1]
    if f32:
        y = y.astype(np.float32)
    return y


def rgb_lin_to_xyz(rgb) -> np.ndarray:
    return _f64_last3(rgb, "rgb") @ _RGB2XYZ["srgb"].T


def xyz_to_rgb_lin(xyz) -> np.ndarray:
    return _f64_last3(xyz, "xyz") @ _XYZ2RGB["srgb"].T


_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def _white(white):
    if white is None:
        return WHITE_XYZ
    w = np.array(white, dtype=np.float64)
    if w.shape != (3,):
        raise ValueError("white: shape (3,) 여야 한다")
    return w


def xyz_to_lab(xyz, white=None) -> np.ndarray:
    t = _f64_last3(xyz, "xyz") / _white(white)
    with np.errstate(invalid="ignore"):
        f = np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab, white=None) -> np.ndarray:
    lab = _f64_last3(lab, "lab")
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    fx3, fz3 = fx ** 3, fz ** 3
    x = np.where(fx3 > _EPS, fx3, (116.0 * fx - 16.0) / _KAPPA)
    y = np.where(L > _KAPPA * _EPS, fy ** 3, L / _KAPPA)
    z = np.where(fz3 > _EPS, fz3, (116.0 * fz - 16.0) / _KAPPA)
    return np.stack([x, y, z], axis=-1) * _white(white)


def rgb_lin_to_lab(rgb) -> np.ndarray:
    return xyz_to_lab(rgb_lin_to_xyz(rgb))


def lab_to_rgb_lin(lab) -> np.ndarray:
    return xyz_to_rgb_lin(lab_to_xyz(lab))


def delta_e_2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 (Sharma, Wu, Dalal 2005). 방송, 대칭, 색상각 경계 처리."""
    p1 = _f64_last3(lab1, "lab1")
    p2 = _f64_last3(lab2, "lab2")
    L1, a1, b1 = p1[..., 0], p1[..., 1], p1[..., 2]
    L2, a2, b2 = p2[..., 0], p2[..., 1], p2[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        C1 = np.hypot(a1, b1)
        C2 = np.hypot(a2, b2)
        Cb = (C1 + C2) / 2.0
        Cb7 = Cb ** 7
        G = 0.5 * (1.0 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
        a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
        C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
        h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

        dLp = L2 - L1
        dCp = C2p - C1p
        prod0 = (C1p * C2p) == 0.0
        dh = h2p - h1p
        dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
        dhp = np.where(prod0, 0.0, dh)
        dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

        Lbp = (L1 + L2) / 2.0
        Cbp = (C1p + C2p) / 2.0
        hsum = h1p + h2p
        hbp = np.where(np.abs(h1p - h2p) <= 180.0, hsum / 2.0,
                       np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0))
        hbp = np.where(prod0, hsum, hbp)

        T = (1.0 - 0.17 * np.cos(np.radians(hbp - 30.0)) + 0.24 * np.cos(np.radians(2.0 * hbp))
             + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0)) - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))
        dtheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
        Cbp7 = Cbp ** 7
        RC = 2.0 * np.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
        SL = 1.0 + 0.015 * (Lbp - 50.0) ** 2 / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
        SC = 1.0 + 0.045 * Cbp
        SH = 1.0 + 0.015 * Cbp * T
        RT = -np.sin(np.radians(2.0 * dtheta)) * RC

        x = dLp / (kL * SL)
        y = dCp / (kC * SC)
        z = dHp / (kH * SH)
        return np.sqrt(np.maximum(x * x + y * y + z * z + RT * y * z, 0.0))
