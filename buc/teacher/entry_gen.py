"""TCH-11 입구 생성 Entry-Gen: 교사 T2I를 8 NFE로 돌려 AI풍 dev·학습 입력 이미지를 만든다.

계약서: ops/harness/contracts/TCH-11.md. 공개 API는 계약서와 1:1이다.
- 프롬프트 -> 합성 캡션 -> TeacherSampler.generate(순수 노이즈, 표 전체) -> decode_latent -> uint8 RGB.
- 선택(hero search)은 없다. 모든 검증은 교사 forward 이전에 끝난다.
- import는 표준 라이브러리, numpy, torch, PIL, buc.core.det, buc.gov.prov 뿐이며 부작용이 없다.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from buc.core import det
from buc.gov import prov as P

__all__ = [
    "ENTRY_TAG", "MODES", "MODE_SUFFIX", "DEFAULT_TAINT", "EntryGenError", "BannedPromptError",
    "compose_prompt", "check_prompts", "entry_seed", "entry_key", "to_uint8_rgb", "EntryImage",
    "generate_entries", "write_entries",
]

ENTRY_TAG = "buc.tch11.entry.v1"
MODES = ("survey", "cinematic", "plain")
MODE_SUFFIX: Mapping[str, str] = {
    "survey": "diffuse light, deep focus, frontal view",
    "cinematic": "cinematic still, anamorphic, shallow depth of field, film grain",
    "plain": "",
}
DEFAULT_TAINT = ("ENGINE",)
_SEED_LIMIT = 1 << 64


class EntryGenError(ValueError):
    """입구 생성 입력·출력이 잘못됐다."""


class BannedPromptError(EntryGenError):
    """금칙 패턴에 걸린 프롬프트가 있다."""


# --------------------------------------------------------------------------- 프롬프트

def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def compose_prompt(prompt, mode: str = "survey") -> str:
    """원문을 strip하고 모드 접미사를 붙인다. 접미사가 비면 원문 그대로."""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt는 str이어야 한다: {type(prompt).__name__}")
    if not isinstance(mode, str) or mode not in MODES:
        raise EntryGenError(f"모르는 mode: {mode!r} (허용: {MODES})")
    p = prompt.strip()
    if not p:
        raise EntryGenError("빈 프롬프트")
    suffix = MODE_SUFFIX[mode]
    return p if suffix == "" else f"{p}, {suffix}"


def _compile_banned(banned) -> tuple:
    if isinstance(banned, (str, bytes)) or not isinstance(banned, Sequence):
        raise TypeError("banned는 패턴의 시퀀스여야 한다")
    pats = []
    for b in banned:
        if isinstance(b, re.Pattern):
            pats.append(b)
        elif isinstance(b, str):
            try:
                pats.append(re.compile(b, re.IGNORECASE))
            except re.error as e:
                raise EntryGenError(f"금칙 정규식 오류: {b!r}") from e
        else:
            raise TypeError(f"금칙 패턴은 str 또는 컴파일된 패턴이어야 한다: {type(b).__name__}")
    return tuple(pats)


def check_prompts(prompts, banned=()) -> tuple:
    """프롬프트 열을 검증하고 strip한 튜플을 돌려준다. 금칙에 걸리면 BannedPromptError."""
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("prompts는 str의 시퀀스여야 한다")
    pats = _compile_banned(banned)
    out = []
    for p in prompts:
        if not isinstance(p, str):
            raise TypeError(f"프롬프트 원소는 str이어야 한다: {type(p).__name__}")
        s = p.strip()
        if not s:
            raise EntryGenError("빈 프롬프트")
        for pat in pats:
            if pat.search(s):
                raise BannedPromptError(f"금칙 패턴에 걸린 프롬프트: {s!r}")
        out.append(s)
    return tuple(out)


# --------------------------------------------------------------------------- 시드·키

def entry_seed(seed: int, index: int) -> int:
    """장 시드 = det.derive_seed(seed, ENTRY_TAG, index). 범위 밖이면 ValueError."""
    if not _is_int(seed) or not (0 <= int(seed) < _SEED_LIMIT):
        raise ValueError(f"seed는 [0, 2^64) 의 정수여야 한다: {seed!r}")
    if not _is_int(index) or int(index) < 0:
        raise ValueError(f"index는 0 이상의 정수여야 한다: {index!r}")
    return det.derive_seed(int(seed), ENTRY_TAG, int(index))


def entry_key(prompt, mode, seed, height, width, table_key, level=0, k=0) -> str:
    """입구 키(32 hex). index와 무관하다."""
    return det.hash_parts(ENTRY_TAG, "key", prompt, mode, seed, height, width, table_key, level, k).hex()[:32]


# --------------------------------------------------------------------------- 픽셀

def to_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    """[B,3,H,W] in [-1,1] -> uint8 [B,H,W,3]. clamp 후 floor((x+1)*127.5+0.5)."""
    if not isinstance(img, torch.Tensor):
        raise EntryGenError(f"텐서여야 한다: {type(img).__name__}")
    if img.ndim != 4:
        raise EntryGenError(f"[B,3,H,W] 4차원이어야 한다: {tuple(img.shape)}")
    if img.shape[1] != 3:
        raise EntryGenError(f"채널은 3이어야 한다: {img.shape[1]}")
    x = img.detach().to(torch.float64)
    if not bool(torch.isfinite(x).all()):
        raise EntryGenError("이미지에 NaN/Inf가 있다")
    x = x.clamp(-1.0, 1.0)
    u = torch.floor((x + 1.0) * 127.5 + 0.5).clamp(0, 255).to(torch.uint8)
    return np.ascontiguousarray(u.permute(0, 2, 3, 1).cpu().numpy())


# --------------------------------------------------------------------------- 결과

@dataclasses.dataclass(frozen=True, eq=False)
class EntryImage:
    key: str
    index: int
    prompt: str
    caption: str
    mode: str
    seed: int
    level: int
    k: int
    height: int
    width: int
    image: np.ndarray
    prov: Any
    table_key: str
    nfe: int

    def png_bytes(self) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.image, mode="RGB").save(buf, format="PNG")
        return buf.getvalue()

    def image_f32(self) -> np.ndarray:
        return self.image.astype(np.float32) / np.float32(255.0)

    def meta(self) -> dict:
        return {
            "v": ENTRY_TAG, "key": self.key, "index": int(self.index), "prompt": self.prompt,
            "caption": self.caption, "mode": self.mode, "seed": int(self.seed), "level": int(self.level),
            "k": int(self.k), "height": int(self.height), "width": int(self.width), "nfe": int(self.nfe),
            "table_key": self.table_key, "prov": self.prov.to_dict(),
        }

    def shard_fields(self) -> dict:
        meta = json.dumps(self.meta(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {"rgb.png": self.png_bytes(), "meta.json": meta.encode("utf-8")}


# --------------------------------------------------------------------------- 생성 루프

def _resolve_seeds(seed, n: int) -> list:
    if _is_int(seed):
        return [entry_seed(seed, i) for i in range(n)]
    if isinstance(seed, (str, bytes)) or not isinstance(seed, Sequence):
        raise TypeError("seed는 정수 또는 정수 시퀀스여야 한다")
    out = list(seed)
    if len(out) != n:
        raise ValueError(f"seed 길이 {len(out)} != 프롬프트 수 {n}")
    for s in out:
        if not _is_int(s) or not (0 <= int(s) < _SEED_LIMIT):
            raise ValueError(f"seed는 [0, 2^64) 의 정수여야 한다: {s!r}")
    return [int(s) for s in out]


def _check_nonneg_int(v, name: str) -> int:
    if not _is_int(v) or int(v) < 0:
        raise ValueError(f"{name}는 0 이상의 정수여야 한다: {v!r}")
    return int(v)


def generate_entries(sampler, prompts, *, seed=0, height=512, width=512, mode="survey", batch_size=4, level=0,
                     k=0, banned=(), taint=DEFAULT_TAINT) -> list:
    """프롬프트마다 입구 이미지 1장. 모든 검증을 교사 forward 전에 끝낸다."""
    ps = check_prompts(prompts, banned=banned)
    n = len(ps)
    if not isinstance(mode, str) or mode not in MODES:
        raise EntryGenError(f"모르는 mode: {mode!r} (허용: {MODES})")
    teacher = sampler.teacher
    teacher.spec.check_image_size(height, width)
    if not _is_int(batch_size) or int(batch_size) < 1:
        raise ValueError(f"batch_size는 1 이상의 정수여야 한다: {batch_size!r}")
    batch_size = int(batch_size)
    level = _check_nonneg_int(level, "level")
    k = _check_nonneg_int(k, "k")
    seeds = _resolve_seeds(seed, n)
    try:
        prov = P.ProvRecord(taint=frozenset(taint), mode="none")
    except (P.TagError, P.ProvError) as e:
        raise EntryGenError(f"출처 태그가 잘못됐다: {e}") from e
    if prov.ext or prov.comp or prov.hold:
        raise EntryGenError("입구 생성 출력은 외부·COMP·보류 출처를 가질 수 없다")

    table = sampler.table
    out: list = []
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        chunk = ps[lo:hi]
        captions = [compose_prompt(p, mode) for p in chunk]
        res = sampler.generate(captions, seeds=seeds[lo:hi], height=height, width=width, level=level, k=k)
        with torch.no_grad():
            img = teacher.decode_latent(res.z)
        arr = to_uint8_rgb(img)
        for j, (p, cap) in enumerate(zip(chunk, captions)):
            i = lo + j
            s_i = seeds[i]
            pix = np.array(arr[j], dtype=np.uint8, copy=True, order="C")
            pix.setflags(write=False)
            out.append(EntryImage(
                key=entry_key(p, mode, s_i, height, width, table.key, level, k),
                index=i, prompt=p, caption=cap, mode=mode, seed=s_i, level=level, k=k,
                height=int(height), width=int(width), image=pix, prov=prov, table_key=table.key,
                nfe=len(table),
            ))
    return out


def write_entries(entries, out_dir) -> list:
    """장마다 {key}.png 와 {key}.json 을 tmp+replace로 쓴다. PNG 경로를 입력 순서대로 돌려준다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for e in entries:
        fields = e.shard_fields()
        for name, data in (("rgb.png", fields["rgb.png"]), ("meta.json", fields["meta.json"])):
            ext = "png" if name == "rgb.png" else "json"
            dst = out / f"{e.key}.{ext}"
            tmp = out / f".{e.key}.{ext}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dst)
        paths.append(out / f"{e.key}.png")
    return paths
