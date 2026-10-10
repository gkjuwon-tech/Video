"""TCH-08 참조 교사 슬롯: klein 4B 로더, Reference Ring / Ring-R, 교체형 슬롯.

계약서: ops/harness/contracts/TCH-08.md. 공개 API는 계약서와 1:1이다.
- L2b(매체 질감) 외관 단계의 참조 조건 사다리 `Ring-R -> klein 참조 -> Ref-Adapter -> 2.5D 카드`의 부품.
- 라이선스 플래그(사람의 확인)와 스냅샷의 라이선스 원문을 둘 다 요구한다. 9B 등 비상업 변형은 거부한다.
- import는 CUDA·diffusers·transformers·huggingface_hub 없이 되고 부작용이 없다(무거운 import는 전부 지연).
- klein 저장소 이름, VAE 채널, TE 차원, 4스텝 sigma 표, 변환기 호출 규약은 [2차] 잠정값이다(GPU 테스트가 확인).
"""
from __future__ import annotations

import contextlib
import gc
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from buc.core import det
from buc.teacher.sample import SigmaError, SigmaTable, TeacherSampler, add_noise
from buc.teacher.zimage import (
    TeacherDependencyError,
    TeacherError,
    TeacherSpecError,
    TextEmbedCache,
    VaeCodec,
    enable_determinism_mode,
    make_noise,
    module_bytes,
    snapshot_dir,
)

__all__ = [
    "RefSlotError", "RefLicenseError", "RefBudgetError", "KLEIN_REPO", "KLEIN_OFFICIAL_REPOS",
    "FORBIDDEN_REPO_MARKERS", "KLEIN_MAX_REFS", "REF_TOKEN_BUDGET", "RING_STRIP_PX", "RING_CANVAS_HW",
    "RINGR_K_BASE", "KLEIN_TABLE", "KleinSpec", "check_license", "ref_tokens", "check_ref_budget",
    "KleinLoaders", "default_klein_loaders", "load_klein", "KleinRefTeacher", "compose_strip", "ring_canvas",
    "split_canvas", "strip_hook", "renoise_up", "RingRConfig", "RingResult", "ring_sample", "choose_l2b_path",
    "RefSlot",
]


# --------------------------------------------------------------------------- 예외

class RefSlotError(TeacherError):
    """참조 교사 슬롯의 모든 오류의 뿌리."""


class RefLicenseError(RefSlotError, PermissionError):
    """라이선스 확인 플래그가 없거나 스냅샷의 라이선스 원문이 Apache-2.0 이라는 근거를 주지 않는다."""


class RefBudgetError(RefSlotError, ValueError):
    """참조 이미지·잠재의 개수 또는 토큰 예산 초과."""


# --------------------------------------------------------------------------- 상수

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"          # 잠정([2차])
KLEIN_OFFICIAL_REPOS = frozenset({KLEIN_REPO})
FORBIDDEN_REPO_MARKERS = ("9b",)                           # 9B 변형은 비상업 라이선스
KLEIN_MAX_REFS = 4
REF_TOKEN_BUDGET = 2304                                    # = 768^2 / 256
RING_STRIP_PX = 128
RING_CANVAS_HW = (1152, 1024)
RINGR_K_BASE = 1 << 20
KLEIN_TABLE = SigmaTable.from_flow_shift(4, 3.0, name="klein-4b-n4-provisional")

_REV_RE = re.compile(r"[0-9A-Za-z._-]+")
_REPO_RE = re.compile(r"[0-9A-Za-z._-]+/[0-9A-Za-z._-]+")
_INT_FIELDS = ("latent_channels", "vae_scale", "patch_size", "text_dim", "max_text_len")
_LEVELS = ("L1", "L1.5", "L2a", "L2b")


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


# --------------------------------------------------------------------------- 스펙

@dataclass(frozen=True)
class KleinSpec:
    repo: str = KLEIN_REPO
    revision: Optional[str] = None
    latent_channels: int = 32
    vae_scale: int = 8
    patch_size: int = 2
    text_dim: int = 2560
    max_text_len: int = 512
    license_verified: bool = False
    unofficial_ok: bool = False

    def validate(self) -> "KleinSpec":
        if not isinstance(self.repo, str):
            raise TeacherSpecError(f"repo는 str이어야 한다: {self.repo!r}")
        low = self.repo.lower()
        for m in FORBIDDEN_REPO_MARKERS:
            if m.lower() in low:
                raise TeacherSpecError(f"금지된 변형(비상업 라이선스)이다: {self.repo!r} ({m!r})")
        if _REPO_RE.fullmatch(self.repo) is None:
            raise TeacherSpecError(f"repo 형식이 잘못됐다: {self.repo!r}")
        if self.repo not in KLEIN_OFFICIAL_REPOS and not self.unofficial_ok:
            raise TeacherSpecError(f"공식 klein 저장소가 아니다(제3자 파생물 금지): {self.repo!r}")
        for f in _INT_FIELDS:
            v = getattr(self, f)
            if not (isinstance(v, int) and not isinstance(v, bool)) or v <= 0:
                raise TeacherSpecError(f"{f}는 양의 정수여야 한다: {v!r}")
        rev = self.revision
        if rev is not None and not (isinstance(rev, str) and _REV_RE.fullmatch(rev) is not None
                                    and rev not in (".", "..")):
            raise TeacherSpecError(f"revision 형식이 잘못됐다: {rev!r}")
        if self.license_verified is not True:
            raise RefLicenseError("klein 라이선스 원문을 사람이 확인하기 전에는 쓸 수 없다(license_verified=True 필요)")
        return self

    def check_image_size(self, height, width) -> None:
        unit = self.vae_scale * self.patch_size
        for name, v in (("height", height), ("width", width)):
            if not (isinstance(v, int) and not isinstance(v, bool)) or v <= 0 or v % unit != 0:
                raise ValueError(f"{name}={v!r}는 {unit}의 양의 배수여야 한다")

    def latent_shape(self, height, width, batch=1) -> tuple:
        self.check_image_size(height, width)
        if not (isinstance(batch, int) and not isinstance(batch, bool)) or batch <= 0:
            raise ValueError(f"batch는 양의 정수여야 한다: {batch!r}")
        return (batch, self.latent_channels, height // self.vae_scale, width // self.vae_scale)


# --------------------------------------------------------------------------- 라이선스 원문

_LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt")
_LICENSE_KEY_RE = re.compile(r"^license\s*:\s*(.*?)\s*$", re.IGNORECASE)


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise RefLicenseError(f"라이선스 파일을 읽을 수 없다: {p}") from e


def _front_matter(text: str) -> list:
    """README.md YAML 머리말의 줄 목록(없으면 빈 목록)."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    out = []
    for ln in lines[1:]:
        if ln.strip() in ("---", "..."):
            return out
        out.append(ln)
    return []


def check_license(snapshot) -> str:
    """스냅샷의 라이선스 원문이 Apache-2.0 이라는 긍정 근거를 요구한다. 어긋나거나 근거 0이면 RefLicenseError."""
    snap = Path(snapshot)
    evidence = 0
    for name in _LICENSE_FILES:
        p = snap / name
        if p.is_file():
            text = _read_text(p)
            if "Apache License" not in text or "Version 2.0" not in text:
                raise RefLicenseError(f"{name}가 Apache License 2.0 원문이 아니다: {p}")
            evidence += 1
    readme = snap / "README.md"
    if readme.is_file():
        for ln in _front_matter(_read_text(readme)):
            m = _LICENSE_KEY_RE.match(ln)
            if m is None:
                continue
            val = m.group(1).strip().strip("'\"").strip().lower()
            if val != "apache-2.0":
                raise RefLicenseError(f"README 머리말의 license가 apache-2.0이 아니다: {val!r}")
            evidence += 1
    if evidence == 0:
        raise RefLicenseError(f"Apache-2.0 이라는 긍정 근거(LICENSE 원문 또는 README license)가 없다: {snap}")
    return "Apache-2.0"


# --------------------------------------------------------------------------- 참조 예산

def ref_tokens(h: int, w: int) -> int:
    return (h // 16) * (w // 16)


def check_ref_budget(sizes) -> int:
    sizes = [tuple(s) for s in sizes]
    if len(sizes) > KLEIN_MAX_REFS:
        raise RefBudgetError(f"참조는 최대 {KLEIN_MAX_REFS}장이다: {len(sizes)}")
    total = 0
    for s in sizes:
        if len(s) != 2 or not all(_is_int(v) and v > 0 and v % 16 == 0 for v in s):
            raise RefBudgetError(f"참조 이미지 크기는 16의 양의 배수 (h, w)여야 한다: {s!r}")
        total += ref_tokens(int(s[0]), int(s[1]))
    if total > REF_TOKEN_BUDGET:
        raise RefBudgetError(f"참조 토큰 {total} > 예산 {REF_TOKEN_BUDGET}")
    return int(total)


# --------------------------------------------------------------------------- 로더 묶음

@dataclass(frozen=True)
class KleinLoaders:
    tokenizer: Callable[[Path], Any]
    text_encoder: Callable[[Path, Any], torch.nn.Module]
    transformer: Callable[[Path, Any], torch.nn.Module]
    vae: Callable[[Path, Any], torch.nn.Module]
    encode_prompts: Callable[[Any, torch.nn.Module, list, int], list]
    forward: Callable[..., list]       # forward(raw_transformer, x_list, t, cap_list, refs) -> -v 원소의 list


def default_klein_loaders() -> KleinLoaders:
    """실제 diffusers/transformers 로더(오프라인 전용).

    미검증([2차]): diffusers의 FLUX.2 클래스 이름과 forward 호출 규약(패킹, 위치 id, timestep 스케일)은
    GPU에서 확인·보정해야 한다. 의존성이 없으면 TeacherDependencyError.
    """
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise TeacherDependencyError("diffusers와 transformers가 필요하다(gpu extra): " + str(e)) from e
    from buc.teacher.zimage import default_loaders
    base = default_loaders()          # 토크나이저·텍스트 인코더·encode_prompts는 TCH-01과 같은 방식

    def transformer(path, dtype):
        import diffusers as dfs
        cls = getattr(dfs, "Flux2Transformer2DModel", None)
        if cls is None:
            raise TeacherDependencyError("설치된 diffusers에 Flux2Transformer2DModel이 없다(버전을 올려라)")
        return cls.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)

    def vae(path, dtype):
        import diffusers as dfs
        cls = getattr(dfs, "AutoencoderKLFlux2", None) or getattr(dfs, "AutoencoderKL")
        return cls.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)

    def forward(raw, x_list, t, cap_list, refs):
        # 미검증 규약: 잠재를 2x2 패치 토큰으로 펴고, 참조 토큰을 뒤에 붙여 한 시퀀스로 넣는다.
        # 변환기는 sigma(노이즈 수준)를 받고 v = eps - x0 를 낸다. TCH-02 규약은 t = 1 - sigma, 반환 -v.
        def pack(lat):                                   # [C,h,w] -> [h/2*w/2, 4C]
            c, h, w = lat.shape
            p = lat.reshape(c, h // 2, 2, w // 2, 2).permute(1, 3, 0, 2, 4)
            return p.reshape((h // 2) * (w // 2), c * 4)

        def ids(h, w, t_off):                            # 위치 id (t, h, w, l)
            gh, gw = torch.meshgrid(torch.arange(h // 2), torch.arange(w // 2), indexing="ij")
            return torch.stack([torch.full_like(gh, t_off), gh, gw, torch.zeros_like(gh)], dim=-1).reshape(-1, 4)

        outs = []
        for xi, ti, ci in zip(x_list, t, cap_list):
            lat = xi[:, 0]                               # [C,h,w]
            c, h, w = lat.shape
            toks = [pack(lat)]
            pos = [ids(h, w, 0)]
            for r_i, r in enumerate(refs):
                toks.append(pack(r.to(device=lat.device, dtype=lat.dtype)))
                pos.append(ids(r.shape[-2], r.shape[-1], 10 * (r_i + 1)))
            hs = torch.cat(toks, dim=0)[None]
            img_ids = torch.cat(pos, dim=0).to(lat.device)
            txt_ids = torch.zeros(ci.shape[0], 4, dtype=torch.long, device=lat.device)
            txt_ids[:, 3] = torch.arange(ci.shape[0], device=lat.device)
            sigma = (1.0 - ti.float()).reshape(1).to(device=lat.device, dtype=lat.dtype)
            out = raw(hidden_states=hs, encoder_hidden_states=ci[None].to(lat.dtype), timestep=sigma,
                      img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]
            v = out[0, : (h // 2) * (w // 2)]            # 참조 토큰 자리는 버린다
            v = v.reshape(h // 2, w // 2, c, 2, 2).permute(2, 0, 3, 1, 4).reshape(c, h, w)
            outs.append((-v).unsqueeze(1))
        return outs

    return KleinLoaders(tokenizer=base.tokenizer, text_encoder=base.text_encoder, transformer=transformer,
                        vae=vae, encode_prompts=base.encode_prompts, forward=forward)


# --------------------------------------------------------------------------- 교사

def _freeze(m: torch.nn.Module) -> torch.nn.Module:
    m.eval()
    m.requires_grad_(False)
    return m


def _check_param_dtype(m: torch.nn.Module, dtype, what: str) -> None:
    for name, p in m.named_parameters():
        if p.is_floating_point() and p.dtype != dtype:
            raise TeacherSpecError(f"{what}의 파라미터 {name} dtype {p.dtype} != 요청 {dtype}(조용히 캐스트하지 않는다)")


class _LazyTE:
    """텍스트 인코더·토크나이저의 지연 적재와 해제(TCH-01 _LazyTE와 같은 규칙)."""

    def __init__(self, loaders: KleinLoaders, spec: KleinSpec, snapshot: Path, dtype, te_device):
        self.loaders, self.spec, self.snapshot, self.dtype, self.te_device = loaders, spec, snapshot, dtype, te_device
        self.tok = None
        self.te = None

    def _load(self) -> None:
        tok = self.loaders.tokenizer(self.snapshot / "tokenizer")
        te = self.loaders.text_encoder(self.snapshot / "text_encoder", self.dtype)
        hidden = getattr(getattr(te, "config", None), "hidden_size", None)
        if hidden != self.spec.text_dim:
            raise TeacherSpecError(f"텍스트 인코더 hidden_size={hidden!r} != text_dim={self.spec.text_dim}")
        _freeze(te)
        te.to(self.te_device)
        self.tok, self.te = tok, te

    def encode(self, prompts: list) -> list:
        if self.te is None:
            self._load()
        return self.loaders.encode_prompts(self.tok, self.te, list(prompts), self.spec.max_text_len)

    def release(self) -> None:
        self.tok = None
        self.te = None
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


class _BoundTransformer:
    """TCH-02 변환기 규약(x_list, t, cap_list, return_dict) 래퍼. 현재 참조 잠재를 loaders.forward 에 넘긴다."""

    def __init__(self, owner: "KleinRefTeacher"):
        self._owner = owner

    def __call__(self, x_list, t, cap_list, return_dict: bool = False, **kw):
        o = self._owner
        outs = o._loaders.forward(o.raw_transformer, x_list, t, cap_list, list(o._refs))
        if not isinstance(outs, (list, tuple)):
            outs = list(outs)
        outs = list(outs)
        if return_dict:
            from types import SimpleNamespace
            return SimpleNamespace(sample=outs)
        return (outs,)


class KleinRefTeacher:
    def __init__(self, *, spec, device, dtype, snapshot, license, raw_transformer, vae, text, loaders):
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.snapshot = snapshot
        self.license = license
        self.raw_transformer = raw_transformer
        self.vae = vae
        self.text = text
        self._loaders = loaders
        self._refs: tuple = ()
        self.transformer = _BoundTransformer(self)

    @property
    def refs(self) -> tuple:
        return self._refs

    def set_refs(self, latents) -> None:
        if latents is None:
            self._refs = ()
            return
        if isinstance(latents, torch.Tensor) or not isinstance(latents, Sequence):
            raise ValueError("latents는 텐서[C,h,w]의 시퀀스 또는 None이어야 한다")
        items = list(latents)
        if len(items) > KLEIN_MAX_REFS:
            raise RefBudgetError(f"참조 잠재는 최대 {KLEIN_MAX_REFS}개다: {len(items)}")
        c = self.spec.latent_channels
        out = []
        for i, t in enumerate(items):
            if not isinstance(t, torch.Tensor) or t.ndim != 3:
                raise ValueError(f"참조 잠재 {i}는 텐서[C,h,w]여야 한다")
            if t.shape[0] != c:
                raise ValueError(f"참조 잠재 {i}의 채널 {t.shape[0]} != latent_channels {c}")
            if t.shape[1] < 1 or t.shape[2] < 1 or not t.is_floating_point():
                raise ValueError(f"참조 잠재 {i}는 비지 않은 부동소수 텐서여야 한다")
            if not bool(torch.isfinite(t.detach()).all()):
                raise ValueError(f"참조 잠재 {i}에 NaN/Inf가 있다")
            out.append(t.detach().to(device=self.device, dtype=self.dtype, copy=True).requires_grad_(False))
        self._refs = tuple(out)

    def encode_refs(self, images) -> list:
        if isinstance(images, torch.Tensor) or not isinstance(images, Sequence):
            raise ValueError("images는 텐서[3,h,w]의 시퀀스여야 한다")
        imgs = list(images)
        sizes = []
        for i, im in enumerate(imgs):
            if not isinstance(im, torch.Tensor) or im.ndim != 3 or im.shape[0] != 3:
                raise ValueError(f"참조 이미지 {i}는 텐서[3,h,w]여야 한다")
            sizes.append((int(im.shape[1]), int(im.shape[2])))
        check_ref_budget(sizes)                                   # VAE 호출 전에
        return [self.vae.encode(im.unsqueeze(0))[0] for im in imgs]

    def encode_text(self, prompts) -> list:
        return self.text.encode(prompts, device=self.device, dtype=self.dtype)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x)

    def decode_latent(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        return self.vae.decode(z, clamp=clamp)

    def make_noise(self, seed, batch, height, width, *, level=0, k=0) -> torch.Tensor:
        shape = self.spec.latent_shape(height, width, batch)
        return make_noise(seed, shape, level=level, k=k, dtype=torch.float32, device=self.device)

    def release_text_encoder(self) -> None:
        self.text.unload()

    def to(self, device) -> "KleinRefTeacher":
        dev = torch.device(device)
        self.raw_transformer.to(dev)
        self.vae.vae.to(dev)
        self._refs = tuple(r.to(dev) for r in self._refs)
        self.device = dev
        return self

    def info(self) -> dict:
        return {
            "repo": self.spec.repo,
            "revision": self.spec.revision if self.spec.revision is not None else self.snapshot.name,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "snapshot": str(self.snapshot),
            "license": self.license,
            "weights_bytes": module_bytes(self.raw_transformer) + module_bytes(self.vae.vae),
            "n_refs": len(self._refs),
            "te_unloaded": bool(self.text.unloaded),
        }


def load_klein(spec: Optional[KleinSpec] = None, *, device="cuda", dtype=torch.bfloat16, te_device="cpu",
               seed=None, deterministic: bool = True, hub_cache=None, te_cache_dir=None,
               loaders: Optional[KleinLoaders] = None, allow_fp16: bool = False,
               env: Optional[Mapping] = None) -> KleinRefTeacher:
    spec = KleinSpec() if spec is None else spec
    spec.validate()                                               # 파일시스템·로더보다 먼저
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TeacherSpecError(f"dtype은 부동소수 torch.dtype이어야 한다: {dtype!r}")
    if dtype == torch.float16 and not allow_fp16:
        raise TeacherSpecError("fp16은 allow_fp16=True가 필요하다; 기본은 bfloat16")
    snapshot = snapshot_dir(spec, hub_cache=hub_cache, env=env)
    license_name = check_license(snapshot)
    device = torch.device(device)
    te_device = torch.device(te_device)
    loaders = default_klein_loaders() if loaders is None else loaders
    if deterministic:
        enable_determinism_mode(seed)

    transformer = loaders.transformer(snapshot / "transformer", dtype)
    _check_param_dtype(transformer, dtype, "transformer")
    vae_mod = loaders.vae(snapshot / "vae", dtype)
    cfg = getattr(vae_mod, "config", None)
    if getattr(cfg, "latent_channels", None) != spec.latent_channels:
        raise TeacherSpecError(f"VAE latent_channels={getattr(cfg, 'latent_channels', None)!r} "
                               f"!= {spec.latent_channels}")
    _check_param_dtype(vae_mod, dtype, "vae")
    scaling = getattr(cfg, "scaling_factor", None)
    if scaling is None:
        raise TeacherSpecError("VAE config에 scaling_factor가 없다")
    shift = getattr(cfg, "shift_factor", None)

    _freeze(transformer).to(device)
    _freeze(vae_mod).to(device)
    codec = VaeCodec(vae_mod, scaling_factor=scaling, shift_factor=0.0 if shift is None else shift,
                     vae_scale=spec.vae_scale, latent_channels=spec.latent_channels)

    lazy = _LazyTE(loaders, spec, snapshot, dtype, te_device)
    namespace = f"klein|{spec.repo}@{snapshot.name}|{dtype}|te:{te_device.type}|d{spec.text_dim}"
    text = TextEmbedCache(lazy.encode, namespace=namespace, max_len=spec.max_text_len, text_dim=spec.text_dim,
                          cache_dir=te_cache_dir, release_fn=lazy.release)
    return KleinRefTeacher(spec=spec, device=device, dtype=dtype, snapshot=snapshot, license=license_name,
                           raw_transformer=transformer, vae=codec, text=text, loaders=loaders)


# --------------------------------------------------------------------------- Reference Ring: 띠·캔버스·훅

def _check_mult16(v, name: str) -> int:
    if not _is_int(v) or v <= 0 or v % 16 != 0:
        raise ValueError(f"{name}는 16의 양의 배수여야 한다: {v!r}")
    return int(v)


def compose_strip(swatches, *, strip_px: int = RING_STRIP_PX, width: int = 1024) -> torch.Tensor:
    """견본 1-4장을 가로로 이어 붙인 띠 float32[3,strip_px,width]. 슬롯 j = 열 [j*width//n, (j+1)*width//n)."""
    strip_px = _check_mult16(strip_px, "strip_px")
    width = _check_mult16(width, "width")
    if isinstance(swatches, torch.Tensor) or not isinstance(swatches, Sequence):
        raise ValueError("swatches는 텐서[3,h,w]의 시퀀스여야 한다")
    sw = list(swatches)
    n = len(sw)
    if not 1 <= n <= KLEIN_MAX_REFS:
        raise ValueError(f"견본은 1-{KLEIN_MAX_REFS}장이어야 한다: {n}")
    for i, s in enumerate(sw):
        if not isinstance(s, torch.Tensor) or s.ndim != 3 or s.shape[0] != 3 or s.shape[1] < 1 or s.shape[2] < 1:
            raise ValueError(f"견본 {i}는 텐서[3,h,w]여야 한다")
        if not s.is_floating_point() or not bool(torch.isfinite(s.detach()).all()):
            raise ValueError(f"견본 {i}는 유한한 부동소수여야 한다")
    out = torch.zeros(3, strip_px, width, dtype=torch.float32)
    for j, s in enumerate(sw):
        lo, hi = j * width // n, (j + 1) * width // n
        x = s.detach().to(device="cpu", dtype=torch.float32)
        if tuple(x.shape[-2:]) != (strip_px, hi - lo):
            x = F.interpolate(x[None], size=(strip_px, hi - lo), mode="bilinear", align_corners=False,
                              antialias=False)[0]
        out[:, :, lo:hi] = x
    return out


def ring_canvas(view: torch.Tensor, strip: torch.Tensor) -> torch.Tensor:
    """view[B,3,H,W] 또는 [3,H,W] 위에 띠를 붙인다 -> [B,3,s+H,W](띠가 위)."""
    if not isinstance(view, torch.Tensor) or view.ndim not in (3, 4):
        raise ValueError("view는 [B,3,H,W] 또는 [3,H,W] 텐서여야 한다")
    v = view.unsqueeze(0) if view.ndim == 3 else view
    if not isinstance(strip, torch.Tensor) or strip.ndim not in (3, 4):
        raise ValueError("strip은 [3,s,W] 또는 [B,3,s,W] 텐서여야 한다")
    b = v.shape[0]
    st = strip.unsqueeze(0) if strip.ndim == 3 else strip
    if st.shape[0] not in (1, b) or st.shape[1] != v.shape[1] or st.shape[-1] != v.shape[-1]:
        raise ValueError(f"띠 {tuple(strip.shape)}와 뷰 {tuple(view.shape)}의 배치·채널·폭이 맞지 않는다")
    st = st.to(device=v.device, dtype=v.dtype).expand(b, -1, -1, -1)
    return torch.cat([st, v], dim=-2)


def split_canvas(x: torch.Tensor, strip_rows: int) -> tuple:
    """축 -2 로 (띠, 본문)으로 나눈다(뷰)."""
    if not isinstance(x, torch.Tensor) or x.ndim < 2:
        raise ValueError("x는 2차원 이상의 텐서여야 한다")
    if not _is_int(strip_rows) or not 0 < strip_rows < x.shape[-2]:
        raise ValueError(f"strip_rows={strip_rows!r}는 (0, {x.shape[-2]}) 안의 정수여야 한다")
    return x[..., :strip_rows, :], x[..., strip_rows:, :]


def strip_hook(ref_latent: torch.Tensor, strip_rows: int) -> Callable:
    """x0 예측의 띠 영역을 참조 잠재로 덮는 x0_hook(x0, info). 입력은 바꾸지 않는다."""
    if not _is_int(strip_rows) or strip_rows < 1:
        raise ValueError(f"strip_rows는 양의 정수여야 한다: {strip_rows!r}")
    if not isinstance(ref_latent, torch.Tensor) or ref_latent.ndim not in (3, 4):
        raise ValueError("ref_latent는 [C,s,w] 또는 [B,C,s,w] 텐서여야 한다")
    if ref_latent.shape[-2] != strip_rows:
        raise ValueError(f"ref_latent의 띠 행 {ref_latent.shape[-2]} != strip_rows {strip_rows}")
    s = int(strip_rows)

    def hook(x0: torch.Tensor, info=None) -> torch.Tensor:
        if x0.ndim != 4 or not s < x0.shape[-2]:
            raise ValueError(f"x0는 [B,C,H,W]이고 H > {s} 여야 한다: {tuple(x0.shape)}")
        if ref_latent.shape[-3] != x0.shape[1] or ref_latent.shape[-1] != x0.shape[-1]:
            raise ValueError(f"ref_latent {tuple(ref_latent.shape)}가 x0 {tuple(x0.shape)}와 채널·폭이 다르다")
        if ref_latent.ndim == 4 and ref_latent.shape[0] not in (1, x0.shape[0]):
            raise ValueError(f"ref_latent 배치 {ref_latent.shape[0]} != x0 배치 {x0.shape[0]}")
        y = x0.clone()
        y[..., :s, :] = ref_latent.to(device=y.device, dtype=y.dtype)
        return y

    return hook


# --------------------------------------------------------------------------- Ring-R

def renoise_up(z: torch.Tensor, sigma_from, sigma_to, noise: torch.Tensor) -> torch.Tensor:
    """flow 규약 재노이즈: a*z + b*noise, a=(1-st)/(1-sf), b=sqrt(st^2-(a*sf)^2). 0 < sf < st <= 1."""
    vals = []
    for v, name in ((sigma_from, "sigma_from"), (sigma_to, "sigma_to")):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise SigmaError(f"{name}는 실수여야 한다: {v!r}")
        vals.append(float(v))
    sf, st = vals
    if not (math.isfinite(sf) and math.isfinite(st)) or not (0.0 < sf < st <= 1.0):
        raise SigmaError(f"0 < sigma_from < sigma_to <= 1 이어야 한다: {sigma_from!r}, {sigma_to!r}")
    if tuple(z.shape) != tuple(noise.shape):
        raise ValueError(f"z {tuple(z.shape)}와 noise {tuple(noise.shape)}의 모양이 다르다")
    a = (1.0 - st) / (1.0 - sf)
    b = math.sqrt(max(st * st - (a * sf) ** 2, 0.0))
    return a * z.to(torch.float32) + b * noise.to(torch.float32)


@dataclass(frozen=True)
class RingRConfig:
    resample_at: tuple = (3,)
    r: int = 2

    def __post_init__(self):
        try:
            idx = tuple(self.resample_at)
        except TypeError as e:
            raise ValueError(f"resample_at은 정수 열이어야 한다: {self.resample_at!r}") from e
        if any(not _is_int(i) or i < 1 for i in idx):
            raise ValueError(f"resample_at의 인덱스는 1 이상의 정수여야 한다: {idx!r}")
        idx = tuple(int(i) for i in idx)
        if len(set(idx)) != len(idx):
            raise ValueError(f"resample_at의 인덱스는 고유해야 한다: {idx!r}")
        if not _is_int(self.r) or not 0 <= self.r <= 64:
            raise ValueError(f"r은 [0, 64]의 정수여야 한다: {self.r!r}")
        object.__setattr__(self, "resample_at", idx)
        object.__setattr__(self, "r", int(self.r))

    def extra_nfe(self, start_index: int) -> int:
        return self.r * sum(1 for s in self.resample_at if s > start_index)


@dataclass(frozen=True)
class RingResult:
    z: torch.Tensor
    strip: torch.Tensor
    body: torch.Tensor
    calls: int
    nfe: int
    sigmas: tuple
    start_index: int
    table_key: str


def _ring_seeds(seeds, b: int) -> list:
    def chk(s) -> int:
        if not _is_int(s):
            raise TypeError(f"시드는 정수여야 한다: {s!r}")
        s = int(s)
        if not 0 <= s < (1 << 64):
            raise ValueError(f"시드가 [0, 2^64) 밖이다: {s}")
        return s

    if _is_int(seeds):
        base = chk(seeds)
        return [det.derive_seed(base, i) for i in range(b)]
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, (Sequence, np.ndarray)):
        raise TypeError("seeds는 정수 또는 정수 시퀀스여야 한다")
    out = [chk(s) for s in seeds]
    if len(out) != b:
        raise ValueError(f"seeds 길이 {len(out)} != 배치 {b}")
    return out


def ring_sample(sampler: TeacherSampler, x0: torch.Tensor, cond, *, ref_latent: torch.Tensor, strip_rows: int,
                sigma, seeds, ring_r: Optional[RingRConfig] = None, steps: Optional[int] = None, level: int = 0,
                k: int = 0, observer: Optional[Callable] = None) -> RingResult:
    """Reference Ring(ring_r=None) 또는 Ring-R. 정의는 계약서 TCH-08 ring_sample 항."""
    teacher, table = sampler.teacher, sampler.table
    # ---- 검증(변환기 호출 전)
    if not isinstance(x0, torch.Tensor) or x0.ndim != 4 or not x0.is_floating_point():
        raise ValueError("x0는 부동소수 텐서[B,C,h,w]여야 한다")
    b, c, h, w = x0.shape
    if b < 1 or h < 1 or w < 1:
        raise ValueError(f"x0의 차원은 양수여야 한다: {tuple(x0.shape)}")
    if c != teacher.spec.latent_channels:
        raise ValueError(f"x0의 채널 {c} != latent_channels {teacher.spec.latent_channels}")
    if not bool(torch.isfinite(x0.detach()).all()):
        raise ValueError("x0에 NaN/Inf가 있다")
    if not _is_int(strip_rows) or strip_rows <= 0 or strip_rows % 2 != 0 or strip_rows >= h:
        raise ValueError(f"strip_rows는 h={h}보다 작은 양의 짝수여야 한다: {strip_rows!r}")
    strip_rows = int(strip_rows)
    if not isinstance(ref_latent, torch.Tensor) or ref_latent.ndim not in (3, 4) or not ref_latent.is_floating_point():
        raise ValueError("ref_latent는 부동소수 텐서 [C,strip_rows,w] 또는 [B,C,strip_rows,w]여야 한다")
    if tuple(ref_latent.shape[-3:]) != (c, strip_rows, w):
        raise ValueError(f"ref_latent {tuple(ref_latent.shape)}가 [C={c},strip_rows={strip_rows},w={w}]가 아니다")
    if ref_latent.ndim == 4 and ref_latent.shape[0] not in (1, b):
        raise ValueError(f"ref_latent 배치 {ref_latent.shape[0]} != 뷰 수 {b}")
    if not bool(torch.isfinite(ref_latent.detach()).all()):
        raise ValueError("ref_latent에 NaN/Inf가 있다")
    if ring_r is not None and not isinstance(ring_r, RingRConfig):
        raise TypeError(f"ring_r는 RingRConfig 또는 None이어야 한다: {type(ring_r).__name__}")
    if observer is not None and not callable(observer):
        raise TypeError("observer는 호출 가능해야 한다")
    if not _is_int(level) or not _is_int(k):
        raise TypeError("level, k는 정수여야 한다")
    snap = table.snap(sigma)
    s0 = snap.index
    n_steps = len(table) - s0 if steps is None else steps
    table.schedule(s0, n_steps)                                    # 범위 검증(SigmaError: ValueError)
    n_steps = int(n_steps)
    seed_list = _ring_seeds(seeds, b)

    # ---- 시작 상태: 띠를 참조로 덮은 x0 -> add_noise
    vs = teacher.spec.vae_scale
    x = x0.detach().to(device=teacher.device, dtype=torch.float32, copy=True)
    x[:, :, :strip_rows] = ref_latent.detach().to(device=x.device, dtype=torch.float32)
    eps = torch.stack([teacher.make_noise(sd, 1, vs * h, vs * w, level=level, k=k)[0] for sd in seed_list])
    z = add_noise(x, snap.sigma, eps.to(device=x.device, dtype=torch.float32))

    hook = strip_hook(ref_latent, strip_rows)
    pts = table.points
    calls = 0
    sigmas: list = []

    def one_step(z_in: torch.Tensor, idx: int) -> torch.Tensor:
        nonlocal calls
        r = sampler.denoise(z_in, cond, start_index=idx, steps=1, x0_hook=hook, observer=observer)
        calls += r.calls
        sigmas.extend(r.sigmas)
        return r.z

    for s in range(s0, s0 + n_steps):
        if ring_r is not None and s in ring_r.resample_at and s > s0:
            for j in range(ring_r.r):
                kk = k + RINGR_K_BASE + 64 * s + j
                eta = torch.stack([teacher.make_noise(sd, 1, vs * h, vs * w, level=level, k=kk)[0]
                                   for sd in seed_list]).to(device=z.device, dtype=torch.float32)
                z = renoise_up(z, pts[s], pts[s - 1], eta)
                z = one_step(z, s - 1)
        z = one_step(z, s)

    strip, body = split_canvas(z, strip_rows)
    return RingResult(z=z, strip=strip.clone(), body=body.clone(), calls=int(calls), nfe=len(sigmas),
                      sigmas=tuple(sigmas), start_index=int(s0), table_key=table.key)


# --------------------------------------------------------------------------- L2b 사다리

def choose_l2b_path(level, *, ring_r_green: bool, klein_ok: bool, adapter_ok: bool) -> str:
    if not isinstance(level, str) or level not in _LEVELS:
        raise ValueError(f"level은 {_LEVELS} 중 하나여야 한다: {level!r}")
    if level != "L2b":
        return "none"
    if ring_r_green:
        return "ring_r"
    if klein_ok:
        return "klein_ref"
    if adapter_ok:
        return "ref_adapter"
    return "cards_2_5d"


# --------------------------------------------------------------------------- 교체형 슬롯

def _default_mover(obj, device):
    return obj.to(device)


class RefSlot:
    """카드당 상주 교사 하나(CT-M2). 참조 교사를 쓸 때만 상주를 park 로 내리고 ref 를 device 로 올린다."""

    def __init__(self, resident, ref_factory: Callable[[], Any], *, device, park_device="cpu",
                 mover: Optional[Callable[[Any, Any], Any]] = None):
        if not callable(ref_factory):
            raise TypeError("ref_factory는 호출 가능해야 한다")
        self._resident = resident
        self._factory = ref_factory
        self.device = device
        self.park_device = park_device
        self._mover = _default_mover if mover is None else mover
        self._ref = None
        self._state = "resident"
        self.stats = {"loads": 0, "swaps_in": 0, "swaps_out": 0}

    @property
    def state(self) -> str:
        return self._state

    @property
    def ref(self):
        return self._ref

    @property
    def resident(self):
        return self._resident

    def _move(self, obj, device):
        moved = self._mover(obj, device)
        return obj if moved is None else moved

    def _best_effort(self, fn) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001 - 복구 중의 2차 오류는 원래 오류를 가리지 않는다
            pass

    def swap_in(self):
        if self._state == "ref":
            raise RefSlotError("이미 ref 상태다")
        try:
            self._resident = self._move(self._resident, self.park_device)      # 상주를 먼저 내린다
            fresh = False
            if self._ref is None:
                self._ref = self._factory()
                fresh = True
            try:
                self._ref = self._move(self._ref, self.device)
            except BaseException:
                if fresh:
                    self._ref = None                                            # 올리다 실패한 새 교사는 버린다
                else:
                    self._best_effort(lambda: self._move(self._ref, self.park_device))
                raise
            if fresh:
                self.stats["loads"] += 1
        except BaseException:
            self._resident = self._move_back_resident()
            raise
        self._state = "ref"
        self.stats["swaps_in"] += 1
        return self._ref

    def _move_back_resident(self):
        try:
            return self._move(self._resident, self.device)
        except Exception:  # noqa: BLE001
            return self._resident

    def swap_out(self) -> None:
        if self._state != "ref":
            raise RefSlotError("ref 상태가 아니다")
        self._ref = self._move(self._ref, self.park_device)                     # 역순: ref 를 먼저 내린다
        self._resident = self._move(self._resident, self.device)
        self._state = "resident"
        self.stats["swaps_out"] += 1

    @contextlib.contextmanager
    def use(self):
        ref = self.swap_in()
        try:
            yield ref
        finally:
            if self._state == "ref":
                self.swap_out()

    def close(self) -> None:
        if self._state == "ref":
            self.swap_out()
        self._ref = None
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
