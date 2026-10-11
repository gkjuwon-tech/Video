"""TCH-01 교사 로더: Z-Image-Turbo BF16 적재, 텍스트 인코더 encode-only 캐시, VAE 코덱, 결정론 모드.

계약서: ops/harness/contracts/TCH-01.md. 공개 API는 계약서와 1:1이다.
- 가중치는 오프라인 HF 캐시(RUN-05 prefetch)에서만 읽는다. 네트워크 접근 0.
- import는 CUDA·diffusers·transformers·huggingface_hub 없이 되고 부작용이 없다(무거운 import는 전부 지연).
"""
from __future__ import annotations

import gc
import math
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from buc.core import det

__all__ = [
    "TeacherError", "TeacherDependencyError", "TeacherSpecError", "TeacherWeightsMissing", "TEUnloadedError",
    "TeacherNumericError", "ZIMAGE_REPO", "OFFICIAL_REPOS", "REQUIRED_SUBDIRS", "TeacherSpec", "hub_cache_dir",
    "snapshot_dir", "enable_determinism_mode", "make_noise", "module_bytes", "module_fingerprint",
    "TextEmbedCache", "VaeCodec", "Loaders", "default_loaders", "load_teacher", "ZImageTeacher",
]


# --------------------------------------------------------------------------- 예외

class TeacherError(RuntimeError):
    """교사 로더의 모든 오류의 뿌리."""


class TeacherDependencyError(TeacherError, ImportError):
    """diffusers/transformers 등 선택 의존성이 없다."""


class TeacherSpecError(TeacherError, ValueError):
    """스펙·구조·dtype 정책 위반."""


class TeacherWeightsMissing(TeacherError, FileNotFoundError):
    """오프라인 HF 캐시에 가중치(스냅샷)가 없다."""


class TEUnloadedError(TeacherError):
    """텍스트 인코더가 해제된 뒤 캐시 미스가 발생했다."""


class TeacherNumericError(TeacherError, FloatingPointError):
    """NaN/Inf 출력."""


# --------------------------------------------------------------------------- 스펙·경로

ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
OFFICIAL_REPOS = frozenset({ZIMAGE_REPO})
REQUIRED_SUBDIRS = ("transformer", "vae", "text_encoder", "tokenizer")

_REV_RE = re.compile(r"[0-9A-Za-z._-]+")
_REPO_RE = re.compile(r"[0-9A-Za-z._-]+/[0-9A-Za-z._-]+")
_INT_FIELDS = ("num_layers", "hidden_dim", "latent_channels", "vae_scale", "patch_size", "text_dim", "max_text_len")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _valid_revision(rev) -> bool:
    return isinstance(rev, str) and _REV_RE.fullmatch(rev) is not None and rev not in (".", "..")


@dataclass(frozen=True)
class TeacherSpec:
    repo: str = ZIMAGE_REPO
    revision: Optional[str] = None
    num_layers: int = 30
    hidden_dim: int = 3840
    latent_channels: int = 16
    vae_scale: int = 8
    patch_size: int = 2
    text_dim: int = 2560
    max_text_len: int = 512
    unofficial_ok: bool = False

    def validate(self) -> "TeacherSpec":
        if not isinstance(self.repo, str) or _REPO_RE.fullmatch(self.repo) is None:
            raise TeacherSpecError(f"repo 형식이 잘못됐다: {self.repo!r}")
        if self.repo not in OFFICIAL_REPOS and not self.unofficial_ok:
            raise TeacherSpecError(f"공식 저장소가 아니다(제3자 파생물 금지): {self.repo!r}")
        for f in _INT_FIELDS:
            v = getattr(self, f)
            if not _is_int(v) or v <= 0:
                raise TeacherSpecError(f"{f}는 양의 정수여야 한다: {v!r}")
        if self.revision is not None and not _valid_revision(self.revision):
            raise TeacherSpecError(f"revision 형식이 잘못됐다: {self.revision!r}")
        return self

    def check_image_size(self, height, width) -> None:
        unit = self.vae_scale * self.patch_size
        for name, v in (("height", height), ("width", width)):
            if not _is_int(v) or v <= 0 or v % unit != 0:
                raise ValueError(f"{name}={v!r}는 {unit}의 양의 배수여야 한다")

    def latent_shape(self, height, width, batch=1) -> tuple:
        self.check_image_size(height, width)
        if not _is_int(batch) or batch <= 0:
            raise ValueError(f"batch는 양의 정수여야 한다: {batch!r}")
        return (batch, self.latent_channels, height // self.vae_scale, width // self.vae_scale)


def hub_cache_dir(env: Optional[Mapping] = None) -> Path:
    env = os.environ if env is None else env
    v = env.get("HF_HUB_CACHE")
    if v:
        return Path(v)
    v = env.get("HF_HOME")
    if v:
        return Path(v) / "hub"
    v = env.get("BUC_ROOT")
    if v:
        return Path(v) / "hf" / "hub"
    return Path("/workspace/buc/hf/hub")


def snapshot_dir(spec: TeacherSpec, hub_cache=None, env: Optional[Mapping] = None) -> Path:
    spec.validate()
    hub = Path(hub_cache) if hub_cache is not None else hub_cache_dir(env)
    root = hub / ("models--" + spec.repo.replace("/", "--"))
    rev = spec.revision
    if rev is None:
        ref = root / "refs" / "main"
        try:
            rev = ref.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise TeacherWeightsMissing(f"HF 캐시에 refs/main이 없다: {ref}") from e
        if not _valid_revision(rev):
            raise TeacherWeightsMissing(f"refs/main 내용이 올바른 리비전이 아니다: {ref}")
    snap = root / "snapshots" / rev
    if not snap.is_dir():
        raise TeacherWeightsMissing(f"스냅샷 폴더가 없다: {snap}")
    missing = [str(snap / s) for s in REQUIRED_SUBDIRS if not (snap / s).is_dir()]
    if missing:
        raise TeacherWeightsMissing("스냅샷에 없는 하위 폴더: " + ", ".join(missing))
    return snap


# --------------------------------------------------------------------------- 결정론

def _disable_uninitialized_fill() -> None:
    """``torch.use_deterministic_algorithms(True)`` also turns on ``torch.utils.deterministic.fill_uninitialized_memory``:
    every ``torch.empty*`` (ours, diffusers', and ATen-internal ones inside fused kernels) then comes back NaN-filled.
    That is a debugging poison, not part of the reproducibility contract (same input -> same bits), and it turns any
    read of an allocated-but-unwritten buffer (e.g. a padded tile that a kernel masks out later: 0 * NaN = NaN) into a
    NaN transformer output that plain diffusers, which never enables it, does not see. We keep every determinism
    guarantee and drop only the poison."""
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
    except Exception:  # noqa: BLE001  (torch < 2.1 has no such switch: nothing to disable)
        pass


def enable_determinism_mode(seed=None) -> dict:
    det.enable_determinism(seed)
    _disable_uninitialized_fill()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    return {
        "seed": None if seed is None else int(seed),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "fp16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def make_noise(seed, shape, *, level=0, k=0, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Philox WNF 표준정규 노이즈. 장치·전역 RNG·배치 크기와 무관(배치 앞부분 접두 불변)."""
    try:
        shape = tuple(int(s) for s in shape)
    except TypeError as e:
        raise ValueError(f"shape는 정수 열이어야 한다: {shape!r}") from e
    if len(shape) == 0 or any(s <= 0 for s in shape):
        raise ValueError(f"shape의 모든 차원은 양수여야 한다: {shape}")
    n = math.prod(shape)
    z = det.wnf(seed, level, np.arange(n, dtype=np.int64), k, channels=1, dtype=np.float32)[..., 0]
    t = torch.from_numpy(np.ascontiguousarray(z)).reshape(shape)
    return t.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- 바이트·지문

def module_bytes(module: torch.nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in module.parameters()))


def _hashable_array(v: torch.Tensor) -> np.ndarray:
    t = v.detach().cpu().contiguous()
    try:
        return t.numpy()
    except (TypeError, RuntimeError):
        # bf16, float8 등 numpy가 모르는 dtype: 같은 폭의 정수로 비트 뷰
        width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
        return t.view(width).numpy()


def module_fingerprint(module: torch.nn.Module) -> bytes:
    entries = {}
    for name, v in module.state_dict().items():
        if v.device.type == "meta":
            raise TeacherError(f"meta 텐서는 지문을 낼 수 없다: {name}")
        entries[f"{name}:{v.dtype}"] = det.hash_array(_hashable_array(v))
    return det.merkle_root_keyed(entries)


# --------------------------------------------------------------------------- 텍스트 임베딩 캐시

class TextEmbedCache:
    """프롬프트 -> 임베딩[L,D] 캐시. 미스만 중복 제거해 인코더를 한 번 호출하고, 디스크에 원자적으로 남긴다."""

    def __init__(self, encode_fn: Callable[[list], list], *, namespace: str, max_len: int,
                 text_dim: Optional[int] = None, cache_dir=None, release_fn: Optional[Callable[[], None]] = None):
        self._encode_fn = encode_fn
        self.namespace = namespace
        self.max_len = max_len
        self.text_dim = text_dim
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._release_fn = release_fn
        self._mem: dict = {}
        self._unloaded = False
        self.stats = {"hits": 0, "disk_hits": 0, "misses": 0, "encode_calls": 0}

    @property
    def unloaded(self) -> bool:
        return self._unloaded

    def key(self, prompt: str) -> str:
        return det.hash_parts("buc.tch01.te.v1", self.namespace, self.max_len, prompt).hex()

    # -- 검증
    def _check_embedding(self, t, *, from_encoder: bool) -> torch.Tensor:
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            raise TeacherError("텍스트 임베딩은 2D 텐서[L,D]여야 한다")
        if self.text_dim is not None and t.shape[-1] != self.text_dim:
            raise TeacherSpecError(f"임베딩 차원 {t.shape[-1]} != text_dim {self.text_dim}")
        if not bool(torch.isfinite(t.detach().float()).all()):
            raise TeacherNumericError("텍스트 임베딩에 NaN/Inf가 있다")
        return t

    @staticmethod
    def _store_copy(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to("cpu", copy=True).contiguous()

    # -- 디스크
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def _disk_load(self, key: str):
        if self.cache_dir is None:
            return None
        p = self._path(key)
        if not p.is_file():
            return None
        try:
            t = torch.load(p, map_location="cpu", weights_only=True)
            self._check_embedding(t, from_encoder=False)
            return t.detach().contiguous()
        except Exception:  # 깨진 파일은 미스
            return None

    def _disk_store(self, key: str, t: torch.Tensor) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_dir / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            torch.save(t, tmp)
            os.replace(tmp, self._path(key))
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # -- 공개
    def encode(self, prompts, *, device=None, dtype=None) -> list:
        prompts = list(prompts)
        for p in prompts:
            if not isinstance(p, str):
                raise TypeError(f"프롬프트는 str이어야 한다: {type(p).__name__}")
        if not prompts:
            return []
        unique = list(dict.fromkeys(prompts))
        hits = disk_hits = 0
        miss_list = []
        for p in unique:
            if p in self._mem:
                hits += 1
                continue
            t = self._disk_load(self.key(p))
            if t is not None:
                self._mem[p] = t
                disk_hits += 1
            else:
                miss_list.append(p)
        if miss_list:
            if self._unloaded:
                raise TEUnloadedError("텍스트 인코더가 해제됐고 캐시에 없는 프롬프트가 있다: "
                                      + ", ".join(repr(p[:40]) for p in miss_list[:3]))
            self.stats["encode_calls"] += 1
            out = self._encode_fn(list(miss_list))
            try:
                out = list(out)
            except TypeError as e:
                raise TeacherError("인코더 출력은 텐서 리스트여야 한다") from e
            if len(out) != len(miss_list):
                raise TeacherError(f"인코더 출력 개수 {len(out)} != 요청 {len(miss_list)}")
            fresh = []
            for t in out:
                self._check_embedding(t, from_encoder=True)
                fresh.append(self._store_copy(t))
            for p, t in zip(miss_list, fresh):  # 전부 검증된 뒤에만 캐시에 올린다
                self._mem[p] = t
                self._disk_store(self.key(p), t)
            self.stats["misses"] += len(miss_list)
        self.stats["hits"] += hits
        self.stats["disk_hits"] += disk_hits
        kw: dict = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return [self._mem[p].to(copy=True, **kw).requires_grad_(False) for p in prompts]

    def unload(self) -> None:
        if self._unloaded:
            return
        self._unloaded = True
        if self._release_fn is not None:
            self._release_fn()


# --------------------------------------------------------------------------- VAE 코덱

class VaeCodec:
    """x[-1,1] <-> 잠재. encode는 결정론(.mode()), 스케일/시프트는 diffusers 관례: (z - shift) * scale."""

    def __init__(self, vae, *, scaling_factor: float, shift_factor: float = 0.0, vae_scale: int = 8,
                 latent_channels: int = 16):
        scaling_factor = float(scaling_factor)
        shift_factor = 0.0 if shift_factor is None else float(shift_factor)
        if not math.isfinite(scaling_factor) or scaling_factor == 0.0 or not math.isfinite(shift_factor):
            raise TeacherSpecError("scaling_factor는 0이 아닌 유한수, shift_factor는 유한수여야 한다")
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.vae_scale = vae_scale
        self.latent_channels = latent_channels

    def _param(self) -> torch.Tensor:
        return next(self.vae.parameters())

    def encode(self, x: torch.Tensor, *, out_dtype=torch.float32) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x는 [B,3,H,W] 텐서여야 한다")
        b, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"채널은 3이어야 한다: {c}")
        if b <= 0 or h <= 0 or w <= 0 or h % self.vae_scale or w % self.vae_scale:
            raise ValueError(f"H,W는 {self.vae_scale}의 양의 배수여야 한다: {h}x{w}")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("입력에 NaN/Inf가 있다")
        p = self._param()
        with torch.no_grad():
            xv = x.to(device=p.device, dtype=p.dtype)
            z = self.vae.encode(xv).latent_dist.mode()
            z = (z.to(torch.float32) - self.shift_factor) * self.scaling_factor
        if z.ndim != 4 or z.shape[1] != self.latent_channels:
            raise TeacherSpecError(f"VAE 잠재 채널 {tuple(z.shape)}이 latent_channels={self.latent_channels}와 다르다")
        if not bool(torch.isfinite(z).all()):
            raise TeacherNumericError("VAE encode 출력에 NaN/Inf가 있다")
        return z.to(out_dtype).detach()

    def decode(self, z: torch.Tensor, *, clamp: bool = True, out_dtype=torch.float32) -> torch.Tensor:
        if not isinstance(z, torch.Tensor) or z.ndim != 4:
            raise ValueError("z는 [B,C,h,w] 텐서여야 한다")
        if z.shape[1] != self.latent_channels:
            raise ValueError(f"잠재 채널 {z.shape[1]} != {self.latent_channels}")
        if z.shape[0] <= 0 or z.shape[2] <= 0 or z.shape[3] <= 0:
            raise ValueError("잠재의 차원은 양수여야 한다")
        if not bool(torch.isfinite(z).all()):
            raise ValueError("잠재에 NaN/Inf가 있다")
        p = self._param()
        with torch.no_grad():
            zz = (z.to(device=p.device, dtype=torch.float32) / self.scaling_factor + self.shift_factor).to(p.dtype)
            y = self.vae.decode(zz).sample
            y = y.to(torch.float32)
        if not bool(torch.isfinite(y).all()):
            raise TeacherNumericError("VAE decode 출력에 NaN/Inf가 있다(검은 이미지 방지)")
        if clamp:
            y = y.clamp(-1.0, 1.0)
        return y.to(out_dtype).detach()


# --------------------------------------------------------------------------- 로더 묶음

@dataclass(frozen=True)
class Loaders:
    tokenizer: Callable[[Path], Any]
    text_encoder: Callable[[Path, Any], torch.nn.Module]
    transformer: Callable[[Path, Any], torch.nn.Module]
    vae: Callable[[Path, Any], torch.nn.Module]
    encode_prompts: Callable[[Any, torch.nn.Module, list, int], list]


def default_loaders() -> Loaders:
    """실제 diffusers/transformers 로더(오프라인 전용). 의존성이 없으면 TeacherDependencyError."""
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise TeacherDependencyError("diffusers와 transformers가 필요하다(gpu extra): " + str(e)) from e

    def tokenizer(path):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True)

    def text_encoder(path, dtype):
        from transformers import AutoModel
        return AutoModel.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)

    def transformer(path, dtype):
        import diffusers as dfs
        cls = getattr(dfs, "ZImageTransformer2DModel", None)
        if cls is None:
            raise TeacherDependencyError("설치된 diffusers에 ZImageTransformer2DModel이 없다(버전을 올려라)")
        return cls.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)

    def vae(path, dtype):
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)

    def encode_prompts(tok, te, prompts, max_len):
        # 프롬프트마다 따로 인코딩해 배치 구성과 무관하게 같은 임베딩을 얻는다(캐시 결정론).
        out = []
        dev = next(te.parameters()).device
        with torch.no_grad():
            for p in prompts:
                msgs = [{"role": "user", "content": p}]
                try:
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=True)
                except TypeError:
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = tok([text], padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
                ids = enc.input_ids.to(dev)
                mask = enc.attention_mask.to(dev).bool()
                hs = te(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]
                out.append(hs[0][mask[0]].detach())
        return out

    return Loaders(tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae,
                   encode_prompts=encode_prompts)


# --------------------------------------------------------------------------- 교사

class _LazyTE:
    """텍스트 인코더·토크나이저의 지연 적재와 해제를 한 곳에 둔다(강한 참조는 여기에만)."""

    def __init__(self, loaders: Loaders, spec: TeacherSpec, snapshot: Path, dtype, te_device):
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


def _freeze(m: torch.nn.Module) -> torch.nn.Module:
    m.eval()
    m.requires_grad_(False)
    return m


def _check_param_dtype(m: torch.nn.Module, dtype, what: str) -> None:
    for name, p in m.named_parameters():
        if p.is_floating_point() and p.dtype != dtype:
            raise TeacherSpecError(f"{what}의 파라미터 {name} dtype {p.dtype} != 요청 {dtype}(조용히 캐스트하지 않는다)")


class ZImageTeacher:
    def __init__(self, *, spec, device, dtype, snapshot, transformer, layers, vae, text, determinism, fingerprints):
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.snapshot = snapshot
        self.transformer = transformer
        self.layers = layers
        self.vae = vae
        self.text = text
        self.determinism = determinism
        self.fingerprints = fingerprints

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

    def info(self) -> dict:
        fps = None if self.fingerprints is None else {k: v.hex() for k, v in self.fingerprints.items()}
        return {
            "repo": self.spec.repo,
            "revision": self.spec.revision if self.spec.revision is not None else self.snapshot.name,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "snapshot": str(self.snapshot),
            "weights_bytes": module_bytes(self.transformer) + module_bytes(self.vae.vae),
            "te_unloaded": self.text.unloaded,
            "determinism": None if self.determinism is None else dict(self.determinism),
            "fingerprints": fps,
        }


def load_teacher(spec: Optional[TeacherSpec] = None, *, device="cuda", dtype=torch.bfloat16, te_device="cpu",
                 seed=None, deterministic: bool = True, hub_cache=None, te_cache_dir=None,
                 loaders: Optional[Loaders] = None, fingerprint: bool = False, allow_fp16: bool = False,
                 env: Optional[Mapping] = None) -> ZImageTeacher:
    spec = TeacherSpec() if spec is None else spec
    spec.validate()
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TeacherSpecError(f"dtype은 부동소수 torch.dtype이어야 한다: {dtype!r}")
    if dtype == torch.float16 and not allow_fp16:
        raise TeacherSpecError("fp16은 allow_fp16=True가 필요하다(수술은 TCH-09 몫); 기본은 bfloat16")
    snapshot = snapshot_dir(spec, hub_cache=hub_cache, env=env)
    device = torch.device(device)
    te_device = torch.device(te_device)
    loaders = default_loaders() if loaders is None else loaders
    determinism = enable_determinism_mode(seed) if deterministic else None

    transformer = loaders.transformer(snapshot / "transformer", dtype)
    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise TeacherSpecError("transformer에 layers 속성이 없다")
    if len(layers) != spec.num_layers:
        raise TeacherSpecError(f"transformer.layers 길이 {len(layers)} != num_layers {spec.num_layers}")
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
    namespace = f"{spec.repo}@{snapshot.name}|{dtype}|te:{te_device.type}|d{spec.text_dim}"
    text = TextEmbedCache(lazy.encode, namespace=namespace, max_len=spec.max_text_len, text_dim=spec.text_dim,
                          cache_dir=te_cache_dir, release_fn=lazy.release)

    fps = None
    if fingerprint:
        fps = {"transformer": module_fingerprint(transformer), "vae": module_fingerprint(vae_mod)}
    return ZImageTeacher(spec=spec, device=device, dtype=dtype, snapshot=snapshot, transformer=transformer,
                         layers=transformer.layers, vae=codec, text=text, determinism=determinism,
                         fingerprints=fps)
