"""RUN-06 체크포인트: 원자 쓰기, RNG 전수 저장, 설정 해시, 상태 수집·복원, 관리자(정리·주기·허브 동기·재개).

체크포인트는 ``torch.load(weights_only=True)`` 로 열 수 있는 값(텐서·파이썬 기본형)만 담는다.
키·``"v"`` 를 바꾸면 호환이 깨지므로 버전과 계약서를 함께 올린다. CUDA 없이 import 가능하며
import 부작용이 없다. 로거 이름은 ``"buc.train"``.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import numpy as np
import torch

from buc.core.det import hash_bytes
from buc.rt.hf_sync import SyncResult, checkpoint_name, parse_checkpoint_name, validate_run_id

__all__ = [
    "CheckpointError", "CheckpointMismatchError", "LoadedCheckpoint", "CheckpointManager",
    "atomic_write", "atomic_torch_save", "capture_rng", "restore_rng", "config_hash",
    "make_state", "apply_state",
]

_LOG = logging.getLogger("buc.train")
STATE_VERSION = 1


class CheckpointError(RuntimeError):
    pass


class CheckpointMismatchError(CheckpointError):
    """저장된 설정 정체성(config_hash)이 지금 설정과 다르다."""


class LoadedCheckpoint(NamedTuple):
    step: int
    state: dict
    path: Path


# --------------------------------------------------------------------------- 원자 쓰기
def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return  # 디렉터리 fsync를 못 여는 파일시스템
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path, write_fn: Callable) -> Path:
    """``<이름>.tmp`` 에 쓰고 fsync 한 뒤 rename. 실패(BaseException 포함)하면 tmp만 지우고 기존 파일을 보존한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(path.parent)
    return path


def atomic_torch_save(obj, path) -> Path:
    return atomic_write(path, lambda f: torch.save(obj, f))


# --------------------------------------------------------------------------- RNG
def capture_rng() -> dict:
    """python·numpy(가우시안 캐시 포함)·torch CPU·CUDA 난수 상태의 스냅숏(기본형·텐서뿐)."""
    py_version, py_internal, py_gauss = random.getstate()
    alg, keys, pos, has_gauss, cached = np.random.get_state()
    cuda: list = []
    if torch.cuda.is_available():
        cuda = [t.clone() for t in torch.cuda.get_rng_state_all()]
    return {
        "python": {"version": int(py_version), "internal": [int(x) for x in py_internal],
                   "gauss_next": py_gauss},
        "numpy": {"alg": str(alg), "keys": torch.from_numpy(np.asarray(keys).astype(np.int64)),
                  "pos": int(pos), "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda,
    }


def restore_rng(s: Mapping) -> None:
    py = s["python"]
    random.setstate((int(py["version"]), tuple(int(x) for x in py["internal"]), py["gauss_next"]))
    npy = s["numpy"]
    np.random.set_state((str(npy["alg"]), npy["keys"].numpy().astype(np.uint32), int(npy["pos"]),
                         int(npy["has_gauss"]), float(npy["cached_gaussian"])))
    torch.set_rng_state(s["torch_cpu"].clone())
    cuda = s.get("torch_cuda") or []
    if cuda and torch.cuda.is_available():
        if len(cuda) != torch.cuda.device_count():
            _LOG.warning("CUDA device count differs from checkpoint (%d vs %d); restoring the common prefix",
                         len(cuda), torch.cuda.device_count())
        for i, st in enumerate(cuda[:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(st.clone(), i)


# --------------------------------------------------------------------------- 설정 해시
def _json_default(o):
    if isinstance(o, Mapping):
        return dict(o)
    raise TypeError(f"config_hash: not JSON serializable: {type(o).__name__}")


def config_hash(cfg: Mapping) -> str:
    try:
        raw = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)
    except ValueError as e:  # nan/inf
        raise TypeError(f"config_hash: {e}") from None
    return hash_bytes(raw.encode("utf-8")).hex()


# --------------------------------------------------------------------------- 상태
def _snapshot(obj):
    """텐서는 CPU 복사본으로, 컨테이너는 새로 만들어 이후 학습이 스냅숏을 바꾸지 못하게 한다."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, Mapping):
        return {k: _snapshot(v) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(_snapshot(v) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(_snapshot(v) for v in obj)
    return obj


def make_state(step: int, *, model, optimizer=None, scheduler=None, ema=None, amp=None,
               config_hash: str, git_commit: Optional[str] = None, extra: Optional[Mapping] = None) -> dict:
    for name, p in model.named_parameters():
        if p.is_floating_point() and p.dtype != torch.float32:
            raise ValueError(f"master parameters must be fp32; {name} is {p.dtype}")
    return {
        "v": STATE_VERSION,
        "step": int(step),
        "model": _snapshot(model.state_dict()),
        "ema": _snapshot(ema.state_dict()) if ema is not None else None,
        "optimizer": _snapshot(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler": _snapshot(scheduler.state_dict()) if scheduler is not None else None,
        "amp": amp.state_dict() if amp is not None else None,
        "rng": capture_rng(),
        "config_hash": config_hash,
        "git_commit": git_commit,
        "extra": _snapshot(dict(extra)) if extra else {},
    }


def apply_state(state: Mapping, *, model, optimizer=None, scheduler=None, ema=None, amp=None,
                restore_rng_state: bool = True) -> None:
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(_snapshot(state["optimizer"]))  # 복사본: 옵티마이저가 상태 텐서를 제자리 갱신한다
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(_snapshot(state["scheduler"]))
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    if amp is not None and state.get("amp") is not None:
        amp.load_state_dict(state["amp"])
    if restore_rng_state and state.get("rng") is not None:
        restore_rng(state["rng"])


# --------------------------------------------------------------------------- 관리자
class CheckpointManager:
    def __init__(self, ckpt_dir, run_id: str, *, keep_last: int = 3, every_steps: int = 1000,
                 every_seconds: float = 900.0, upload_every_seconds: float = 1800.0, hub=None,
                 background_upload: bool = False, clock: Callable[[], float] = time.monotonic):
        self.run_id = validate_run_id(run_id)
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
            raise ValueError("keep_last must be an int >= 1")
        if isinstance(every_steps, bool) or not isinstance(every_steps, int) or every_steps < 1:
            raise ValueError("every_steps must be an int >= 1")
        if not (every_seconds > 0):
            raise ValueError("every_seconds must be > 0")
        if not (upload_every_seconds > 0):
            raise ValueError("upload_every_seconds must be > 0")
        self.ckpt_dir = Path(ckpt_dir)
        self.keep_last = keep_last
        self.every_steps = every_steps
        self.every_seconds = float(every_seconds)
        self.upload_every_seconds = float(upload_every_seconds)
        self.hub = hub
        self.background_upload = bool(background_upload)
        self._clock = clock
        self.last_upload: Optional[SyncResult] = None
        self._thread: Optional[threading.Thread] = None
        self._uploading_step: Optional[int] = None
        self._lock = threading.Lock()
        self._remove_stale_tmp()
        local = self.list_local()
        self._last_saved_step = local[-1][0] if local else 0
        now = self._clock()
        self._last_save_time = now
        self._last_upload_time = now

    # ---- 파일 ----------------------------------------------------------------------------------
    def _remove_stale_tmp(self) -> None:
        if not self.ckpt_dir.is_dir():
            return
        for p in self.ckpt_dir.glob("*.tmp"):
            try:
                p.unlink()
            except OSError:
                pass

    def path_for(self, step: int) -> Path:
        return self.ckpt_dir / checkpoint_name(step)

    def list_local(self) -> list:
        """완성본 ``step_%09d.pt`` 만, 스텝 오름차순 ``[(step, Path)]``."""
        out = []
        if not self.ckpt_dir.is_dir():
            return out
        for p in self.ckpt_dir.iterdir():
            parsed = parse_checkpoint_name(p.name)
            if parsed is None or parsed[1] != "pt" or p.name != checkpoint_name(parsed[0]) or not p.is_file():
                continue
            out.append((parsed[0], p))
        out.sort(key=lambda t: t[0])
        return out

    def should_save(self, step: int) -> bool:
        if step <= self._last_saved_step:
            return False
        return (step - self._last_saved_step >= self.every_steps
                or self._clock() - self._last_save_time >= self.every_seconds)

    def _prune(self, keep_path: Path) -> None:
        local = self.list_local()
        for step, p in local[:max(0, len(local) - self.keep_last)]:
            if p == keep_path or step == self._uploading_step:
                continue
            try:
                p.unlink()
            except OSError as e:
                _LOG.warning("could not prune checkpoint %s: %s", p.name, e)

    def save(self, step: int, state: Mapping, *, final: bool = False) -> Path:
        path = self.path_for(step)
        atomic_torch_save(state, path)
        self._last_saved_step = max(self._last_saved_step, int(step))
        self._last_save_time = self._clock()
        self._prune(path)
        self.maybe_upload(step, force=final)
        return path

    # ---- 업로드 --------------------------------------------------------------------------------
    def _upload_now(self, step: int, path: Path) -> SyncResult:
        try:
            res = self.hub.try_upload_checkpoint(path, self.run_id, step)
        except Exception as e:  # 허브 장애가 학습을 막으면 안 된다
            msg = f"{type(e).__name__}: {e}"
            _LOG.warning("checkpoint upload failed (best effort): %s", msg)
            res = SyncResult(ok=False, error=msg)
        self.last_upload = res
        return res

    def _run_background(self, step: int, path: Path) -> None:
        try:
            self._upload_now(step, path)
        finally:
            self._uploading_step = None

    def maybe_upload(self, step: int, *, force: bool = False) -> Optional[SyncResult]:
        hub = self.hub
        if hub is None or not getattr(hub, "enabled", False):
            return None
        now = self._clock()
        if not force and now - self._last_upload_time < self.upload_every_seconds:
            return None
        path = self.path_for(step)
        if not path.is_file():
            return None
        if force:
            self.flush()
        elif self.background_upload:
            with self._lock:
                if self._thread is not None and self._thread.is_alive():
                    return None  # 쌓지 않고 건너뛴다
                self._last_upload_time = now
                self._uploading_step = int(step)
                self._thread = threading.Thread(target=self._run_background, args=(int(step), path),
                                                name="buc-ckpt-upload", daemon=True)
                self._thread.start()
            return None
        self._last_upload_time = now
        return self._upload_now(step, path)

    def flush(self, timeout: Optional[float] = None) -> bool:
        t = self._thread
        if t is not None:
            t.join(timeout)
            return not t.is_alive()
        return True

    def close(self) -> None:
        self.flush()

    # ---- 재개 ----------------------------------------------------------------------------------
    def _fetch_from_hub(self) -> None:
        hub = self.hub
        if hub is None or not getattr(hub, "enabled", False):
            return
        try:
            refs = [r for r in hub.list_checkpoints(self.run_id) if r.remote_path.endswith(".pt")]
            if not refs:
                return
            local = self.list_local()
            latest_local = local[-1][0] if local else -1
            ref = refs[-1]
            if ref.step > latest_local:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                hub.download_file(ref.remote_path, self.path_for(ref.step))
        except Exception as e:  # 허브 오류는 경고 후 로컬로
            _LOG.warning("hub checkpoint fetch failed, falling back to local: %s: %s", type(e).__name__, e)

    def resume(self, *, expect_config_hash: Optional[str] = None) -> Optional[LoadedCheckpoint]:
        self._fetch_from_hub()
        for step, path in reversed(self.list_local()):
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as e:
                _LOG.warning("checkpoint %s unreadable (%s: %s); trying an older one", path.name, type(e).__name__, e)
                continue
            if not isinstance(state, dict) or state.get("v") != STATE_VERSION:
                _LOG.warning("checkpoint %s has an unsupported format; trying an older one", path.name)
                continue
            if state.get("step") != step:
                _LOG.warning("checkpoint %s holds step %r, not %d; trying an older one",
                             path.name, state.get("step"), step)
                continue
            if expect_config_hash is not None and state.get("config_hash") != expect_config_hash:
                raise CheckpointMismatchError(
                    f"checkpoint {path.name} was written for a different configuration "
                    f"(config_hash {state.get('config_hash')!r} != expected {expect_config_hash!r})")
            self._last_saved_step = max(self._last_saved_step, step)
            self._last_save_time = self._clock()
            return LoadedCheckpoint(step, state, path)
        return None

