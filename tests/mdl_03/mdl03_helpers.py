"""MDL-03 테스트 공용 도구. 결정론: 지역 Generator만 쓴다(전역 RNG 불변).

- 가짜 PCA 기저(D=16, k=8)와 그 기저로 만든 TapSet.
- DAT-02 정본 생산자 `RealifyResult.shard_fields()`로 샤드 필드를 만든다(필드 이름을 손으로 흉내내지 않는다).
- 학습 가능한 합성 과제: 라벨이 층-8 탭 특징의 함수(탭 격자 셀마다 상수).
"""
from __future__ import annotations

import numpy as np
import torch

from buc.data import realify as R
from buc.data import shards as S
from buc.gov import prov as P
from buc.teacher import taps as T

LAYERS = (8, 15, 23)
D = 16
K = 8
GH = GW = 4
H = W = 16 * GH          # 64
ENGINE = P.as_record(["ENGINE"])
N_OCT = R.max_octaves(H, W)  # 6


def gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def make_basis(seed: int = 0, k: int = K, scale_mul: float = 1.0) -> T.PCABasis:
    g = gen(seed)
    mean = torch.randn(len(LAYERS), D, generator=g)
    comps = torch.randn(len(LAYERS), D, k, generator=g)
    scale = (0.01 + 0.02 * torch.rand(len(LAYERS), k, generator=g)) * scale_mul
    return T.PCABasis(LAYERS, mean, comps, scale)


def tiny_cfg(**kw):
    from buc.models.mrh.neck import MRHConfig
    base = dict(tap_layers=LAYERS, tap_dim=K, neck_ch=32, rcu_per_fusion=1, head_ch=16, sem_dim=8,
                n_sigma_bands=4, t_embed_dim=16)
    base.update(kw)
    return MRHConfig(**base)


def rand_taps(seed: int, b: int = 2, k: int = K, gh: int = GH, gw: int = GW, mul: float = 1.0) -> dict:
    g = gen(seed)
    return {l: torch.randn(b, k, gh, gw, generator=g) * mul for l in LAYERS}


def _up(a: np.ndarray, f: int) -> np.ndarray:
    return np.repeat(np.repeat(a, f, axis=-2), f, axis=-1)


def task_labels(feat8: np.ndarray):
    """feat8: float [k,gh,gw] (층 8 특징). -> gbuf[12,H,W] f32, depth[H,W] f32, sem[H,W] i16."""
    f = feat8 / (np.abs(feat8).max() + 1e-9)
    logd = 1.0 + 0.5 * f[0] + 0.3 * f[1]
    nrm = np.stack([0.4 * f[2], 0.4 * f[3], -np.ones_like(f[2])])
    nrm = nrm / np.linalg.norm(nrm, axis=0, keepdims=True)
    sem = np.where(f[4] > 0, 9, 8).astype(np.int16)
    gb = np.zeros((12,) + logd.shape, np.float32)
    gb[0] = logd
    gb[1:4] = nrm
    gb[4:7] = 0.5
    gb[9] = 1.0
    gb[11] = 1.0
    gbuf = _up(gb, 16).astype(np.float32)
    depth = np.exp(gbuf[0]).astype(np.float32)
    return gbuf, depth, _up(sem, 16).astype(np.int16)


def make_result(i: int, basis: T.PCABasis, *, keep: bool = True, band_valid=None, tap_sigma: float = 0.5,
                key: str | None = None, gbuf=None, sem=None) -> R.RealifyResult:
    g = gen(1000 + i)
    q = {l: torch.randint(-100, 101, (1, basis.k, GH, GW), generator=g, dtype=torch.int64).to(torch.int8)
         for l in LAYERS}
    key = key or f"s{i}"
    ts = T.TapSet(layers=q, grid=(GH, GW), basis_fp=basis.fingerprint, prov=ENGINE, ephemeral=False, key=key)
    feat8 = ts.features(basis)[8][0].numpy()
    gb, depth, sm = task_labels(feat8)
    if gbuf is not None:
        gb = gbuf
    if sem is not None:
        sm = sem
    labels = {"gbuf": gb, "depth": depth, "sem": sm, "inst": np.zeros(sm.shape, np.int32)}
    img = np.full((gb.shape[-2], gb.shape[-1], 3), 0.5, np.float32)
    bv = tuple(band_valid) if band_valid is not None else (True,) * N_OCT
    return R.RealifyResult(key=key, image=img, taps=ts, band_valid=bv, lowband_err=0.0, keep=keep,
                           reject_reason=None if keep else "lowband", sigma=0.5, tap_sigma=float(tap_sigma),
                           steps=2, seed=7, labels=labels, prov=ENGINE, curve_key="c" * 64, table_key="t" * 64)


def make_sample(i: int, basis: T.PCABasis, **kw) -> S.Sample:
    r = make_result(i, basis, **kw)
    return S.Sample(key=r.key, fields=r.shard_fields(), prov=ENGINE)


def make_samples(n: int, basis: T.PCABasis, **kw) -> list:
    return [make_sample(i, basis, tap_sigma=0.3 + 0.1 * (i % 4), **kw) for i in range(n)]


def smooth_field(seed: int, b: int, c: int, h: int, w: int, kmax: int = 2) -> torch.Tensor:
    """저주파(정수 주파수 ≤ kmax) 무작위 실수장 [b,c,h,w] float32."""
    g = gen(seed)
    yy = torch.arange(h, dtype=torch.float64).view(h, 1) / h
    xx = torch.arange(w, dtype=torch.float64).view(1, w) / w
    out = torch.zeros(b, c, h, w, dtype=torch.float64)
    for ky in range(kmax + 1):
        for kx in range(kmax + 1):
            a = torch.randn(b, c, 1, 1, generator=g, dtype=torch.float64)
            ph = torch.rand(b, c, 1, 1, generator=g, dtype=torch.float64) * 6.283
            out = out + a * torch.cos(6.283185307179586 * (ky * yy + kx * xx) + ph)
    return out.float()
