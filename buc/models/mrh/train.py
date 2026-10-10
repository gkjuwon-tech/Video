"""MDL-03 M3 MRH 라벨·손실·데이터·학습 배선.

- 라벨은 H/4로 내려서(블록 평균) 쓰고, 연속 항은 대역 사영(DAT-02 `band_project`) 안에서 잰다.
  전해상 옥타브 o <-> 1/stride 해상 옥타브 o-s 이므로 band_valid를 `valid[s:]`로 다시 매긴다.
- v0 학습은 깊이·법선·sem(알베도 가중치 0). 가중치 0 항은 계산은 하되 그래프에 기여하지 않는다.
- CUDA 불필요, import 부작용·전역 RNG 변경 없음.
"""
from __future__ import annotations

import json
import numbers
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from buc.data import realify as _realify
from buc.data.shards import iter_shards, npy_decode
from buc.models.mrh.heads import MRH
from buc.models.mrh.neck import MRHConfig, MRHSpecError
from buc.teacher.taps import TapBasisError
from buc.train.amp import AmpConfig
from buc.train.loop import LoopConfig, Trainer

__all__ = ["MRHDataError", "STRIDE", "MIN_VALID_PX", "SEM_TEMP", "N_SEM_IDS", "band_valid_at_stride",
           "downsample_labels", "default_sem_table", "LossWeights", "mrh_loss", "depth_absrel",
           "normal_angle_deg", "decode_sample", "collate", "MRHDataset", "MRHTrainConfig", "make_trainer"]

STRIDE = 4
MIN_VALID_PX = 16
SEM_TEMP = 0.07
N_SEM_IDS = 64
_RIDGE = 1e-8


class MRHDataError(ValueError):
    """라벨·표본·배치 데이터 위반."""


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


# ----------------------------------------------------------------------------------------- 라벨

def _log2_stride(stride) -> int:
    if not _is_int(stride) or int(stride) < 1 or (int(stride) & (int(stride) - 1)) != 0:
        raise MRHSpecError(f"stride: 2의 거듭제곱 정수여야 한다: {stride!r}")
    return int(stride).bit_length() - 1


def band_valid_at_stride(valid, stride: int = STRIDE) -> tuple:
    s = _log2_stride(stride)
    v = tuple(bool(x) for x in valid)
    if not v:
        raise MRHSpecError("band_valid가 비었다")
    return v[s:] if len(v) > s else (v[-1],)


def downsample_labels(gbuf: torch.Tensor, sem: torch.Tensor, stride: int = STRIDE) -> dict:
    s = _log2_stride(stride)
    del s
    if gbuf.ndim != 4 or gbuf.shape[1] != 12:
        raise MRHDataError(f"gbuf: [B,12,H,W]여야 한다: {tuple(gbuf.shape)}")
    b, _, h, w = gbuf.shape
    if h % stride or w % stride:
        raise MRHDataError(f"gbuf 크기 {h}x{w}가 stride {stride}의 배수가 아니다")
    if sem.ndim != 3 or tuple(sem.shape) != (b, h, w):
        raise MRHDataError(f"sem: [B,H,W]={(b, h, w)}여야 한다: {tuple(sem.shape)}")
    g = gbuf.float()
    pooled = F.avg_pool2d(g, stride)
    mask = (pooled[:, 11:12] >= 1.0 - 1e-6).float()
    nrm = pooled[:, 1:4]
    norm = nrm.norm(dim=1, keepdim=True)
    nrm = torch.where(norm > 1e-6, nrm / norm.clamp_min(1e-12), torch.zeros_like(nrm))
    return {"mask": mask, "log_depth": pooled[:, 0:1] * mask, "normal": nrm * mask,
            "albedo": pooled[:, 4:7] * mask, "sem": sem[:, ::stride, ::stride].long()}


def default_sem_table(n_ids: int = N_SEM_IDS, dim: int = 768, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(int(seed))
    t = torch.randn(int(n_ids), int(dim), generator=g, dtype=torch.float32)
    return t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)


# ----------------------------------------------------------------------------------------- 손실

@dataclass(frozen=True)
class LossWeights:
    depth: float = 1.0
    normal: float = 1.0
    sem: float = 0.5
    albedo: float = 0.0


def _depth_term(pred, gt, mask, valid4) -> torch.Tensor:
    """한 표본의 대역 안 affine-inv 깊이 S_b. pred/gt/mask: [h,w]."""
    m = mask.double()
    stack = torch.stack([m * pred.double(), m, m * gt.double()])
    bp, bm, bg = _realify.band_project(stack, valid4).reshape(3, -1)
    a11, a12, a22 = bp @ bp, bp @ bm, bm @ bm
    eye = torch.eye(2, dtype=torch.float64, device=bp.device)
    A = torch.stack([torch.stack([a11, a12]), torch.stack([a12, a22])]) + _RIDGE * eye
    rhs = torch.stack([bp @ bg, bm @ bg])
    sol = torch.linalg.solve(A, rhs)
    r = sol[0] * bp + sol[1] * bm - bg
    return (r @ r) / torch.clamp(m.sum(), min=1.0)


def _vector_term(pred, gt, mask, valid4) -> torch.Tensor:
    """법선/알베도 한 표본의 S_b. pred/gt: [c,h,w], mask: [h,w]."""
    m = mask.double()
    d = m[None] * (pred.double() - gt.double())
    r = _realify.band_project(d, valid4)
    return (r * r).sum() / torch.clamp(m.sum(), min=1.0)


def _mean_or_zero(terms: list, ref: torch.Tensor) -> torch.Tensor:
    if terms:
        return torch.stack(terms).mean()
    return ref.sum().double() * 0.0


def mrh_loss(out, batch, *, sem_table, weights=None, ignore_sem_ids=(0,)):
    weights = LossWeights() if weights is None else weights
    if not isinstance(weights, LossWeights):
        raise MRHSpecError("weights는 LossWeights여야 한다")
    for k in ("log_depth", "normal", "albedo", "sem"):
        if k not in out:
            raise MRHSpecError(f"out에 '{k}' 헤드가 없다")
    gt = downsample_labels(batch["gbuf"], batch["sem"], STRIDE)
    mask = gt["mask"]
    bsz, _, h, w = mask.shape
    bv = batch["band_valid"]
    if len(bv) != bsz:
        raise MRHDataError(f"band_valid 길이 {len(bv)} != 배치 {bsz}")
    for k in ("log_depth", "normal", "albedo"):
        if tuple(out[k].shape[0::2]) != (bsz, h) or out[k].shape[3] != w:
            raise MRHSpecError(f"out['{k}'] 모양 {tuple(out[k].shape)}이 라벨 {(bsz, h, w)}와 다르다")
    dev = out["log_depth"].device
    mask = mask.to(dev)
    gt = {k: v.to(dev) for k, v in gt.items()}
    n_px = mask.flatten(1).sum(1)

    depth_t, normal_t, albedo_t = [], [], []
    p_depth, p_norm, p_alb = out["log_depth"].float(), out["normal"].float(), out["albedo"].float()
    for i in range(bsz):
        if float(n_px[i]) < MIN_VALID_PX:
            continue
        v4 = band_valid_at_stride(bv[i], STRIDE)
        depth_t.append(_depth_term(p_depth[i, 0], gt["log_depth"][i, 0], mask[i, 0], v4))
        normal_t.append(_vector_term(p_norm[i], gt["normal"][i], mask[i, 0], v4))
        albedo_t.append(_vector_term(p_alb[i], gt["albedo"][i], mask[i, 0], v4))
    l_depth = _mean_or_zero(depth_t, p_depth)
    l_normal = _mean_or_zero(normal_t, p_norm)
    l_albedo = _mean_or_zero(albedo_t, p_alb)

    # sem: 대역 사영 없음
    sem_pred = out["sem"].float()
    table = sem_table.to(device=dev, dtype=torch.float32)
    if table.ndim != 2 or table.shape[1] != sem_pred.shape[1]:
        raise MRHSpecError(f"sem_table {tuple(table.shape)}이 sem 채널 {sem_pred.shape[1]}과 맞지 않는다")
    n_ids = table.shape[0]
    lab = gt["sem"]
    ok = (lab >= 0) & (lab < n_ids)
    for ig in ignore_sem_ids:
        ok = ok & (lab != int(ig))
    if bool(ok.any()):
        feats = sem_pred.permute(0, 2, 3, 1)[ok]
        tab = table / table.norm(dim=1, keepdim=True).clamp_min(1e-12)
        logits = (feats @ tab.t()) / SEM_TEMP
        tgt = lab[ok]
        l_sem = F.cross_entropy(logits, tgt)
        sem_acc = float((logits.argmax(1) == tgt).float().mean())
    else:
        l_sem = sem_pred.sum() * 0.0
        sem_acc = 0.0

    total = (weights.depth * l_depth.float() + weights.normal * l_normal.float()
             + weights.sem * l_sem.float() + weights.albedo * l_albedo.float())
    metrics = {"loss/depth": float(l_depth.detach()), "loss/normal": float(l_normal.detach()),
               "loss/sem": float(l_sem.detach()), "loss/albedo": float(l_albedo.detach()), "sem_acc": sem_acc}
    return total, metrics


# ----------------------------------------------------------------------------------------- 지표

def depth_absrel(pred_log: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """표본별 AbsRel. 마스크 화소에서 log 공간 최소제곱 affine 정렬 후 mean(|exp(a*p+b)-d|/d). 빈 마스크는 nan."""
    bsz = pred_log.shape[0]
    res = torch.full((bsz,), float("nan"), dtype=torch.float64, device=pred_log.device)
    for i in range(bsz):
        m = mask[i, 0] > 0.5
        if int(m.sum()) < 1:
            continue
        x = pred_log[i, 0][m].double()
        d = gt_depth[i, 0][m].double()
        y = torch.log(d)
        xc, yc = x - x.mean(), y - y.mean()
        var = (xc * xc).mean()
        a = (xc * yc).mean() / var if float(var) > 1e-12 else torch.zeros((), dtype=torch.float64, device=x.device)
        b = y.mean() - a * x.mean()
        res[i] = ((torch.exp(a * x + b) - d).abs() / d).mean()
    return res


def normal_angle_deg(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """표본별 마스크 평균 각도 오차(도). 빈 마스크는 nan."""
    p = pred.double()
    p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-6)
    dot = (p * gt.double()).sum(1).clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.acos(dot))
    m = (mask[:, 0] > 0.5).double()
    n = m.flatten(1).sum(1)
    out = (ang * m).flatten(1).sum(1) / n.clamp_min(1.0)
    return torch.where(n > 0, out, torch.full_like(out, float("nan")))


# ----------------------------------------------------------------------------------------- 데이터

def _field(sample, name: str) -> bytes:
    f = sample.fields
    if name not in f:
        raise MRHDataError(f"표본 {sample.key!r}: 필드 '{name}' 없음")
    return f[name]


def _decode_npy(sample, name: str) -> np.ndarray:
    try:
        return npy_decode(_field(sample, name))
    except (ValueError, TypeError) as e:
        if isinstance(e, MRHDataError):
            raise
        raise MRHDataError(f"표본 {sample.key!r}: '{name}' 디코드 실패: {e}") from None


def decode_sample(sample, basis, cfg: MRHConfig) -> dict:
    try:
        meta = json.loads(bytes(_field(sample, "meta.json")).decode("utf-8"))
    except (ValueError, UnicodeError) as e:
        raise MRHDataError(f"표본 {sample.key!r}: meta.json 파싱 실패: {e}") from None
    if not isinstance(meta, dict):
        raise MRHDataError(f"표본 {sample.key!r}: meta.json이 객체가 아니다")
    if meta.get("v") != _realify.REALIFY_TAG:
        raise MRHDataError(f"표본 {sample.key!r}: meta.v={meta.get('v')!r} != {_realify.REALIFY_TAG!r}")
    if meta.get("basis_fp") != basis.fingerprint.hex():
        raise TapBasisError(f"표본 {sample.key!r}: PCA 기저 지문 불일치")
    layers = tuple(cfg.tap_layers)
    if meta.get("tap_layers") is None or tuple(meta["tap_layers"]) != layers:
        raise MRHDataError(f"표본 {sample.key!r}: tap_layers {meta.get('tap_layers')!r} != {list(layers)}")
    if not meta.get("keep"):
        raise MRHDataError(f"표본 {sample.key!r}: keep가 거짓(거부된 표본)")
    if any(l not in basis.layers for l in layers):
        raise MRHDataError(f"기저 층 {list(basis.layers)}에 tap_layers {list(layers)}가 없다")
    if basis.k != cfg.tap_dim:
        raise MRHDataError(f"기저 k={basis.k} != cfg.tap_dim {cfg.tap_dim}")

    taps, grid = {}, None
    for l in layers:
        q = _decode_npy(sample, f"tap{l}.npy")
        if q.dtype != np.int8 or q.ndim != 3 or q.shape[0] != basis.k:
            raise MRHDataError(f"표본 {sample.key!r}: tap{l}은 int8 [k={basis.k},gh,gw]여야 한다: {q.dtype} {q.shape}")
        if grid is None:
            grid = tuple(q.shape[1:])
        elif tuple(q.shape[1:]) != grid:
            raise MRHDataError(f"표본 {sample.key!r}: 층 간 격자 불일치")
        scale = basis.scale[basis.layers.index(l)]
        taps[l] = torch.from_numpy(np.ascontiguousarray(q)).float() * scale.view(-1, 1, 1)
    gbuf = _decode_npy(sample, "gbuf.npy")
    sem = _decode_npy(sample, "sem.npy")
    gh, gw = grid
    if gbuf.ndim != 3 or gbuf.shape != (12, 16 * gh, 16 * gw):
        raise MRHDataError(f"표본 {sample.key!r}: gbuf {gbuf.shape} != (12,{16 * gh},{16 * gw})")
    if sem.shape != (16 * gh, 16 * gw):
        raise MRHDataError(f"표본 {sample.key!r}: sem {sem.shape} != ({16 * gh},{16 * gw})")
    try:
        t = float(meta["tap_sigma"])
        bv = tuple(bool(v) for v in meta["band_valid"])
    except (KeyError, TypeError, ValueError):
        raise MRHDataError(f"표본 {sample.key!r}: tap_sigma/band_valid 없음") from None
    if not (0.0 < t <= 1.0) or not bv:
        raise MRHDataError(f"표본 {sample.key!r}: tap_sigma {t!r} 또는 band_valid 위반")
    return {"taps": taps, "t": torch.tensor(t, dtype=torch.float32), "band_valid": bv,
            "gbuf": torch.from_numpy(np.ascontiguousarray(gbuf.astype(np.float32))),
            "sem": torch.from_numpy(np.ascontiguousarray(sem.astype(np.int64))), "key": str(sample.key)}


def collate(examples) -> dict:
    examples = list(examples)
    if not examples:
        raise MRHDataError("빈 배치")
    e0 = examples[0]
    for e in examples[1:]:
        if set(e["taps"]) != set(e0["taps"]):
            raise MRHDataError("표본 간 탭 층 집합이 다르다")
        for l in e0["taps"]:
            if e["taps"][l].shape != e0["taps"][l].shape:
                raise MRHDataError(f"표본 간 탭 모양 불일치(층 {l})")
        if e["gbuf"].shape != e0["gbuf"].shape or e["sem"].shape != e0["sem"].shape:
            raise MRHDataError("표본 간 gbuf/sem 모양 불일치")
    return {"taps": {l: torch.stack([e["taps"][l] for e in examples]) for l in e0["taps"]},
            "t": torch.stack([e["t"] for e in examples]),
            "band_valid": tuple(tuple(e["band_valid"]) for e in examples),
            "gbuf": torch.stack([e["gbuf"] for e in examples]),
            "sem": torch.stack([e["sem"] for e in examples]),
            "keys": tuple(e["key"] for e in examples)}


class MRHDataset:
    def __init__(self, samples, basis, cfg: MRHConfig):
        self.basis = basis
        self.cfg = cfg
        self._ex = [decode_sample(s, basis, cfg) for s in samples]

    def __len__(self) -> int:
        return len(self._ex)

    @property
    def basis_fp(self) -> bytes:
        return self.basis.fingerprint

    def batch(self, indices) -> dict:
        idx = np.asarray(indices).reshape(-1)
        n = len(self._ex)
        if idx.size == 0 or idx.dtype.kind not in "iu" or int(idx.min()) < 0 or int(idx.max()) >= n:
            raise MRHDataError(f"indices: [0,{n}) 정수여야 한다")
        return collate([self._ex[int(i)] for i in idx])

    @classmethod
    def from_shards(cls, paths, basis, cfg: MRHConfig) -> "MRHDataset":
        return cls(iter_shards(paths), basis, cfg)


# ----------------------------------------------------------------------------------------- 학습

@dataclass(frozen=True)
class MRHTrainConfig:
    model: MRHConfig = field(default_factory=MRHConfig)
    weights: LossWeights = field(default_factory=LossWeights)
    lr: float = 3e-4
    weight_decay: float = 0.01
    sem_table_seed: int = 0
    n_sem_ids: int = N_SEM_IDS
    ignore_sem_ids: tuple = (0,)

    def __post_init__(self):
        if not isinstance(self.model, MRHConfig) or not isinstance(self.weights, LossWeights):
            raise MRHSpecError("model은 MRHConfig, weights는 LossWeights여야 한다")
        if isinstance(self.lr, bool) or not isinstance(self.lr, numbers.Real) or not self.lr > 0:
            raise MRHSpecError(f"lr: 양수여야 한다: {self.lr!r}")
        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, numbers.Real) \
                or self.weight_decay < 0:
            raise MRHSpecError(f"weight_decay: 0 이상이어야 한다: {self.weight_decay!r}")
        if not _is_int(self.sem_table_seed) or not _is_int(self.n_sem_ids) or int(self.n_sem_ids) < 1:
            raise MRHSpecError("sem_table_seed·n_sem_ids: 정수(n_sem_ids >= 1)여야 한다")
        ig = tuple(self.ignore_sem_ids)
        if any(not _is_int(i) for i in ig):
            raise MRHSpecError("ignore_sem_ids: 정수 시퀀스여야 한다")
        object.__setattr__(self, "ignore_sem_ids", tuple(int(i) for i in ig))

    def to_dict(self) -> dict:
        return {"model": self.model.to_dict(),
                "weights": {"depth": self.weights.depth, "normal": self.weights.normal,
                            "sem": self.weights.sem, "albedo": self.weights.albedo},
                "lr": float(self.lr), "weight_decay": float(self.weight_decay),
                "sem_table_seed": int(self.sem_table_seed), "n_sem_ids": int(self.n_sem_ids),
                "ignore_sem_ids": list(self.ignore_sem_ids)}


def make_trainer(tcfg: MRHTrainConfig, dataset: MRHDataset, *, run_id, ckpt_dir, max_steps, batch_size,
                 seed=0, amp=None, hub=None, log=None, **loop_kw) -> Trainer:
    if tuple(dataset.cfg.tap_layers) != tuple(tcfg.model.tap_layers) or dataset.cfg.tap_dim != tcfg.model.tap_dim:
        raise MRHSpecError("데이터셋의 탭 설정(tap_layers, tap_dim)이 모델 설정과 다르다")
    model = MRH(tcfg.model, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    table = default_sem_table(tcfg.n_sem_ids, tcfg.model.sem_dim, tcfg.sem_table_seed)
    weights, ignore = tcfg.weights, tcfg.ignore_sem_ids

    def compute_loss(m, batch):
        out = m(batch["taps"], batch["t"])
        return mrh_loss(out, batch, sem_table=table, weights=weights, ignore_sem_ids=ignore)

    lc = LoopConfig(run_id=run_id, ckpt_dir=ckpt_dir, num_samples=len(dataset), batch_size=batch_size,
                    max_steps=max_steps, seed=seed, amp=amp or AmpConfig("fp32", "cpu"),
                    hparams={"mrh": tcfg.to_dict(), "basis_fp": dataset.basis_fp.hex()}, **loop_kw)
    return Trainer(model, opt, compute_loss, dataset.batch, lc, hub=hub, log=log)
