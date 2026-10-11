"""CAM-06 테스트 공용 도우미. 구현 내부가 아니라 계약서(CAM-06.md)의 공식만 쓴다.

합성 신호는 해석적(시간의 닫힌 식)이라 표본 추출·보간·정렬 오차를 구현과 독립으로 잴 수 있다.
GPMF 쓰기 도우미는 GoPro GPMF 바이너리 구조(KLV: 키 4B, 형식 1B, 구조 크기 1B, 반복 2B 빅엔디언, 4바이트 패딩)를
테스트가 직접 만든다.
"""
import importlib
import math
import struct

import numpy as np

RATE = 200.0
G0 = np.array([0.3, -9.7, 1.1])      # 중력 벡터(고정): accel = G0 + 동적 성분


def imu():
    return importlib.import_module("buc.camera.imu")


# ------------------------------------------------------------------ 해석적 신호
def make_signal(seed, *, k=5, fmax=6.0, fmin=0.4, amp=1.0):
    """t(배열) -> (len(t), 3). 축마다 k개의 정현파 합. 결정론."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(fmin, fmax, (3, k))
    ph = rng.uniform(0, 2 * np.pi, (3, k))
    a = amp * rng.uniform(0.3, 1.0, (3, k)) / np.sqrt(k)

    def sig(t):
        t = np.asarray(t, float)[:, None, None]
        return np.sum(a[None] * np.sin(2 * np.pi * f[None] * t + ph[None]), axis=2)
    return sig


def tone(t, freq, amp=1.0, phase=0.3, axis=0):
    out = np.zeros((len(t), 3))
    out[:, axis] = amp * np.sin(2 * np.pi * freq * np.asarray(t) + phase)
    return out


def make_clip(n=2000, *, t0=0.0, rate=RATE, gyro=None, accel=None, meta=None, device="dev", source="test", seed=1):
    m = imu()
    t = t0 + np.arange(n) / rate
    if gyro is None:
        gyro = make_signal(seed, amp=1.5)(t)
    if accel is None:
        accel = G0 + make_signal(seed + 100, amp=2.0, fmin=1.0)(t)
    return m.ImuClip(t0, rate, gyro, accel, source, device, meta or {})


def make_raw(t_g, g, t_a=None, a=None, source="test", device="dev"):
    m = imu()
    gs = m.Series(t_g, g)
    acs = None if a is None else m.Series(t_g if t_a is None else t_a, a)
    return m.RawImu(gs, acs, source, device, {})


def lsq_amp(x, t, freq):
    """x(t) ~ a sin(wt) + b cos(wt) + c 의 진폭 sqrt(a²+b²). x (n,) 또는 (n,3) 이면 열별."""
    w = 2 * np.pi * freq
    X = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    return np.hypot(coef[0], coef[1])


# ------------------------------------------------------------------ GPMF 쓰기
def klv(key, typ, size, repeat, payload):
    assert len(key) == 4 and len(payload) == size * repeat
    tb = b"\x00" if typ == 0 else typ.encode()
    return key + tb + bytes([size]) + struct.pack(">H", repeat) + payload + b"\x00" * ((-len(payload)) % 4)


def container(key, children):
    assert len(children) < 65536
    return klv(key, 0, 1, len(children), children)


def text(key, s):
    b = s.encode()
    return klv(key, "c", len(b), 1, b)


def stmp(us):
    return klv(b"STMP", "J", 8, 1, struct.pack(">Q", int(us)))


def scal(values, typ="s"):
    vals = list(values) if np.ndim(values) else [values]
    code = {"s": ">i2", "l": ">i4"}[typ]
    isz = {"s": 2, "l": 4}[typ]
    return klv(b"SCAL", typ, isz, len(vals), np.asarray(vals).astype(code).tobytes())


def sensor(key, ints, typ="s"):
    code = {"s": ">i2", "l": ">i4"}[typ]
    isz = {"s": 2, "l": 4}[typ]
    return klv(key, typ, 3 * isz, ints.shape[0], np.asarray(ints).astype(code).tobytes())


def gps_stream():
    pay = struct.pack(">5l", 1, 2, 3, 4, 5) * 3
    return container(b"STRM", text(b"STNM", "GPS (Lat., Long., Alt., 2D, 3D speed)") + klv(b"GPS5", "l", 20, 3, pay))


def stream(key, vals, scale, st_us, *, typ="s", name="x", with_stmp=True):
    sc = np.asarray(scale, float)
    ints = np.round(vals * sc).astype(np.int64)
    ch = text(b"STNM", name)
    if with_stmp:
        ch += stmp(st_us)
    ch += scal(scale if np.ndim(scale) else [scale], "s" if typ == "s" else "l") + sensor(key, ints, typ)
    deq = ints / sc
    return container(b"STRM", ch), deq


def make_gpmf(gyro, accel, *, rate_g=RATE, rate_a=RATE, block_s=1.0, t_start_s=12.345678, scal_g=64.0, scal_a=418.0,
              jitter_us=None, gps=True, typ="s", with_stmp=True):
    """gyro (N,3), accel (N',3) 물리 단위. 블록(DEVC)마다 block_s 초. 반환: bytes, 양자화 복원값, 블록 STMP 목록(µs)."""
    out = b""
    n_blk = int(math.ceil(len(gyro) / (rate_g * block_s)))
    bg, ba = int(round(rate_g * block_s)), int(round(rate_a * block_s))
    deq_g, deq_a, st_g, st_a = [], [], [], []
    for k in range(n_blk):
        jit = 0 if jitter_us is None else int(jitter_us[k])
        s_us = int(round(t_start_s * 1e6 + k * block_s * 1e6)) + jit
        ch = text(b"DVNM", "Hero")
        sg, dg = stream(b"GYRO", gyro[k * bg:(k + 1) * bg], scal_g, s_us, typ=typ, name="Gyroscope", with_stmp=with_stmp)
        ch += sg
        deq_g.append(dg)
        st_g.append(s_us)
        if accel is not None:
            sa, da = stream(b"ACCL", accel[k * ba:(k + 1) * ba], scal_a, s_us, typ=typ, name="Accelerometer", with_stmp=with_stmp)
            ch += sa
            deq_a.append(da)
            st_a.append(s_us)
        if gps:
            ch += gps_stream()
        out += container(b"DEVC", klv(b"DVID", "L", 4, 1, struct.pack(">I", 1)) + ch)
    return out, np.concatenate(deq_g), (np.concatenate(deq_a) if accel is not None else None), st_g


# ------------------------------------------------------------------ CSV
TUM_HEADER = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
              "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")


def tum_text(t_ns, g, a, header=TUM_HEADER):
    rows = [header]
    for i in range(len(t_ns)):
        rows.append(",".join([str(int(t_ns[i]))] + [repr(float(x)) for x in list(g[i]) + list(a[i])]))
    return "\n".join(rows) + "\n"


def generic_text(t, g, a=None, tname="t", unit_scale=1.0):
    head = [tname, "gx", "gy", "gz"] + ([] if a is None else ["ax", "ay", "az"])
    rows = [",".join(head)]
    for i in range(len(t)):
        vals = [t[i] * unit_scale] + list(g[i]) + ([] if a is None else list(a[i]))
        rows.append(",".join(repr(float(x)) for x in vals))
    return "\n".join(rows) + "\n"
