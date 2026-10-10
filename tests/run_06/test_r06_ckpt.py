"""원자 쓰기(tmp -> fsync -> rename), RNG 전수 저장, 설정 해시, 상태 수집·복원."""
import json
import os
import random

import numpy as np
import pytest
import torch

from buc.train import ckpt as C
from buc.train.amp import Amp, AmpConfig
from buc.train.loop import ModelEMA


# ---- atomic_write ------------------------------------------------------------------------------
def test_atomic_write_orders_fsync_before_rename(tmp_path, monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (events.append(("replace", os.path.basename(a), os.path.basename(b))),
                                                      real_replace(a, b))[1])
    dst = tmp_path / "step_000000001.pt"
    out = C.atomic_write(dst, lambda f: f.write(b"hello"))
    assert out == dst and dst.read_bytes() == b"hello"
    assert events[0] == "fsync"  # 데이터 fsync가 rename보다 먼저
    i = next(k for k, e in enumerate(events) if isinstance(e, tuple))
    assert events[i] == ("replace", "step_000000001.pt.tmp", "step_000000001.pt")
    assert "fsync" in events[:i]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_000000001.pt"]  # .tmp 잔재 없음


def test_atomic_write_final_path_untouched_until_commit(tmp_path):
    dst = tmp_path / "x.pt"
    dst.write_bytes(b"OLD")
    seen = {}

    def writer(f):
        f.write(b"NEW-PARTIAL")
        f.flush()
        seen["during"] = dst.read_bytes()  # 쓰는 도중 독자는 옛 완성본만 본다

    C.atomic_write(dst, writer)
    assert seen["during"] == b"OLD" and dst.read_bytes() == b"NEW-PARTIAL"


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_atomic_write_failure_preserves_old_file_and_cleans_tmp(tmp_path, exc):
    dst = tmp_path / "x.pt"
    dst.write_bytes(b"OLD")

    def writer(f):
        f.write(b"half")
        raise exc

    with pytest.raises(type(exc)):
        C.atomic_write(dst, writer)
    assert dst.read_bytes() == b"OLD"
    assert [p.name for p in tmp_path.iterdir()] == ["x.pt"]
    fresh = tmp_path / "y.pt"
    with pytest.raises(type(exc)):
        C.atomic_write(fresh, writer)
    assert not fresh.exists() and [p.name for p in tmp_path.iterdir()] == ["x.pt"]


def test_atomic_write_creates_parent_dirs_and_torch_save_is_weights_only_loadable(tmp_path):
    p = C.atomic_torch_save({"a": torch.arange(3), "b": [1, 2.5, None, "s"]}, tmp_path / "d1" / "d2" / "s.pt")
    got = torch.load(p, weights_only=True)
    assert torch.equal(got["a"], torch.arange(3)) and got["b"] == [1, 2.5, None, "s"]


# ---- RNG ---------------------------------------------------------------------------------------
def draw():
    return (random.random(), float(np.random.rand()), torch.rand(3).tolist(), torch.randn(2).tolist(),
            torch.nn.functional.dropout(torch.ones(8), 0.5, True).tolist())


def test_rng_capture_restore_roundtrip_through_disk(tmp_path):
    random.seed(1), np.random.seed(2), torch.manual_seed(3)
    draw()
    state = C.capture_rng()
    p = C.atomic_torch_save(state, tmp_path / "rng.pt")
    expect = draw()
    random.seed(99), np.random.seed(98), torch.manual_seed(97)  # 흩뜨린다
    assert draw() != expect
    C.restore_rng(torch.load(p, weights_only=True))
    assert draw() == expect


def test_rng_capture_is_a_snapshot_not_a_view():
    torch.manual_seed(0)
    st = C.capture_rng()
    expect = draw()
    C.restore_rng(st)
    a = draw()
    C.restore_rng(st)  # 같은 스냅숏을 두 번 복원해도 같다
    assert draw() == a == expect


def test_numpy_gauss_cache_is_captured():
    np.random.seed(5)
    np.random.randn(1)  # 레거시 가우시안은 캐시를 남긴다(has_gauss=1)
    st = C.capture_rng()
    a = np.random.randn(3).tolist()
    np.random.seed(6)
    C.restore_rng(st)
    assert np.random.randn(3).tolist() == a


# ---- config_hash -------------------------------------------------------------------------------
def test_config_hash():
    a = C.config_hash({"lr": 1e-3, "layers": [1, 2], "nest": {"b": 1, "a": 2}})
    assert len(a) == 64 and int(a, 16) >= 0
    assert a == C.config_hash({"nest": {"a": 2, "b": 1}, "layers": [1, 2], "lr": 1e-3})  # 키 순서 무관
    assert a != C.config_hash({"lr": 1e-3, "layers": [1, 3], "nest": {"b": 1, "a": 2}})
    assert a != C.config_hash({"lr": 2e-3, "layers": [1, 2], "nest": {"b": 1, "a": 2}})
    import hashlib
    raw = json.dumps({"x": 1}, sort_keys=True, separators=(",", ":")).encode()
    assert C.config_hash({"x": 1}) != hashlib.sha256(raw).hexdigest()  # sha256이 아니라 blake3(FND-03)
    from buc.core.det import hash_bytes
    assert C.config_hash({"x": 1}) == hash_bytes(raw).hex()
    for bad in ({"f": object()}, {"x": float("nan")}):
        with pytest.raises(TypeError):
            C.config_hash(bad)


# ---- make_state / apply_state ------------------------------------------------------------------
def tiny():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    ema = ModelEMA(m, 0.9)
    amp = Amp(AmpConfig(dtype="fp16", device_type="cpu", init_scale=32.0))
    return m, opt, sch, ema, amp


def train_a_bit(m, opt, sch, ema, amp, n=3):
    for _ in range(n):
        opt.zero_grad()
        with amp.autocast():
            loss = m(torch.randn(5, 3)).float().pow(2).mean()
        amp.scale(loss).backward()
        if amp.step(opt).stepped:
            sch.step()
            ema.update(m)


def test_make_state_schema_and_snapshot_semantics():
    m, opt, sch, ema, amp = tiny()
    train_a_bit(m, opt, sch, ema, amp)
    st = C.make_state(3, model=m, optimizer=opt, scheduler=sch, ema=ema, amp=amp, config_hash="c" * 64,
                      git_commit="abc123", extra={"note": "x"})
    assert st["v"] == 1 and st["step"] == 3 and st["config_hash"] == "c" * 64 and st["git_commit"] == "abc123"
    assert st["extra"] == {"note": "x"}
    assert set(st) >= {"v", "step", "model", "ema", "optimizer", "scheduler", "amp", "rng", "config_hash",
                       "git_commit", "extra"}
    assert set(st["rng"]) >= {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in st["model"].values())
    snap = {k: v.clone() for k, v in st["model"].items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)  # 이후 학습이 진행돼도 스냅숏은 그대로
    for k in snap:
        assert torch.equal(st["model"][k], snap[k])
    bare = C.make_state(0, model=m, config_hash="d" * 64)
    assert bare["optimizer"] is None and bare["ema"] is None and bare["scheduler"] is None and bare["amp"] is None
    assert bare["git_commit"] is None and bare["extra"] == {}


def test_state_roundtrip_through_disk_with_weights_only(tmp_path):
    m, opt, sch, ema, amp = tiny()
    train_a_bit(m, opt, sch, ema, amp)
    st = C.make_state(3, model=m, optimizer=opt, scheduler=sch, ema=ema, amp=amp, config_hash="e" * 64)
    p = C.atomic_torch_save(st, tmp_path / "s.pt")
    loaded = torch.load(p, map_location="cpu", weights_only=True)  # 허브에서 받은 파일을 안전하게 연다
    m2, opt2, sch2, ema2, amp2 = tiny()
    with torch.no_grad():
        for q in m2.parameters():
            q.zero_()
    random.seed(5), np.random.seed(5), torch.manual_seed(5)
    C.apply_state(loaded, model=m2, optimizer=opt2, scheduler=sch2, ema=ema2, amp=amp2)
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
    for k, v in ema.state_dict().items():
        assert torch.equal(v, ema2.state_dict()[k])
    assert amp2.scale_value == amp.scale_value and amp2.skipped_steps == amp.skipped_steps
    assert sch2.state_dict() == sch.state_dict()
    assert opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    for pa, pb in zip(opt.state_dict()["state"].values(), opt2.state_dict()["state"].values()):
        assert torch.equal(pa["exp_avg"], pb["exp_avg"]) and torch.equal(pa["step"], pb["step"])


def test_apply_state_rng_flag():
    m, *_ = tiny()
    torch.manual_seed(11)
    st = C.make_state(1, model=m, config_hash="f" * 64)
    expect = torch.rand(2).tolist()
    torch.manual_seed(12)
    C.apply_state(st, model=m, restore_rng_state=False)
    assert torch.rand(2).tolist() != expect
    C.apply_state(st, model=m)
    assert torch.rand(2).tolist() == expect


def test_fp32_master_is_enforced_when_saving():
    m = torch.nn.Linear(2, 2).half()
    with pytest.raises(ValueError, match="fp32"):
        C.make_state(0, model=m, config_hash="0" * 64)
    m = torch.nn.Linear(2, 2).bfloat16()
    with pytest.raises(ValueError):
        C.make_state(0, model=m, config_hash="0" * 64)
