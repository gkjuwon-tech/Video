"""CheckpointManager: 이름·정리·주기, 재개(손상 폴백·설정 불일치), RUN-03 허브 이음매(업로드 주기·최선 노력·재개)."""
import threading

import pytest
import torch
from run06_helpers import FakeHub, make_sync

from buc.rt import hf_sync as H
from buc.train import ckpt as C

HASH = "a" * 64


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def state(step, h=HASH):
    return {"v": 1, "step": step, "model": {"w": torch.full((2,), float(step))}, "ema": None, "optimizer": None,
            "scheduler": None, "amp": None, "rng": C.capture_rng(), "config_hash": h, "git_commit": None, "extra": {}}


def mgr(tmp_path, clock=None, **kw):
    base = dict(keep_last=3, every_steps=4, every_seconds=100.0, upload_every_seconds=50.0)
    base.update(kw)
    return C.CheckpointManager(tmp_path / "ck", "r06", clock=clock or Clock(), **base)


def names(tmp_path):
    return sorted(p.name for p in (tmp_path / "ck").iterdir())


def test_save_uses_run03_names_and_prunes_to_keep_last(tmp_path):
    m = mgr(tmp_path, keep_last=2)
    for s in (4, 8, 12, 16):
        p = m.save(s, state(s))
        assert p == tmp_path / "ck" / H.checkpoint_name(s)  # step_000000004.pt
        assert p.read_bytes()
    assert names(tmp_path) == ["step_000000012.pt", "step_000000016.pt"]
    assert [s for s, _ in m.list_local()] == [12, 16]


def test_list_local_ignores_temp_sidecar_and_malformed_names(tmp_path):
    m = mgr(tmp_path)
    m.save(8, state(8))
    d = tmp_path / "ck"
    for junk in ("step_000000009.pt.tmp", "step_000000009.pt.sha256.json", "step_9.pt", "latest.pt", "step_000000010.bin"):
        (d / junk).write_bytes(b"x")
    assert [s for s, _ in m.list_local()] == [8]
    m.save(12, state(12), final=False)  # 정리가 남의 파일을 지우지 않는다
    assert (d / "step_9.pt").exists() and (d / "latest.pt").exists()


def test_stale_tmp_files_are_removed_on_start(tmp_path):
    d = tmp_path / "ck"
    d.mkdir()
    (d / "step_000000004.pt.tmp").write_bytes(b"half")
    mgr(tmp_path)
    assert names(tmp_path) == []


def test_should_save_by_steps_or_seconds(tmp_path):
    clk = Clock()
    m = mgr(tmp_path, clk)
    assert not m.should_save(3) and m.should_save(4) and m.should_save(5)
    assert not m.should_save(0)
    clk.t += 99.9
    assert not m.should_save(1)
    clk.t += 0.2  # 100초 경과
    assert m.should_save(1)
    m.save(1, state(1))
    assert not m.should_save(1) and not m.should_save(4) and m.should_save(5)
    clk.t += 100.0
    assert m.should_save(2) and not m.should_save(1)  # 이미 저장한 스텝 이하는 안 한다


def test_manager_after_restart_counts_from_latest_local(tmp_path):
    m = mgr(tmp_path)
    m.save(8, state(8))
    m2 = mgr(tmp_path)
    assert not m2.should_save(11) and m2.should_save(12)


def test_resume_empty_and_newest(tmp_path):
    m = mgr(tmp_path)
    assert m.resume() is None
    for s in (4, 8, 12):
        m.save(s, state(s))
    got = mgr(tmp_path).resume(expect_config_hash=HASH)
    assert got.step == 12 and got.path == tmp_path / "ck" / "step_000000012.pt"
    assert torch.equal(got.state["model"]["w"], torch.full((2,), 12.0))


def test_resume_skips_corrupt_and_mismatched_step_files(tmp_path):
    m = mgr(tmp_path)
    m.save(4, state(4))
    m.save(8, state(8))
    d = tmp_path / "ck"
    (d / "step_000000012.pt").write_bytes(b"not a checkpoint at all")  # 손상
    C.atomic_torch_save(state(7), d / "step_000000010.pt")  # 이름의 스텝과 내용의 스텝이 다름
    got = mgr(tmp_path).resume()
    assert got.step == 8
    for f in ("step_000000012.pt", "step_000000010.pt"):
        assert (d / f).exists()  # 손상본은 지우지 않는다(조사용)


def test_resume_config_mismatch_raises_and_does_not_fall_back(tmp_path):
    m = mgr(tmp_path)
    m.save(4, state(4))
    m.save(8, state(8, h="b" * 64))
    with pytest.raises(C.CheckpointMismatchError):
        mgr(tmp_path).resume(expect_config_hash=HASH)
    assert issubclass(C.CheckpointMismatchError, C.CheckpointError)
    assert mgr(tmp_path).resume().step == 8  # 기대 해시를 안 주면 검사하지 않는다


def test_invalid_arguments(tmp_path):
    for kw in (dict(keep_last=0), dict(every_steps=0), dict(every_seconds=0), dict(upload_every_seconds=0)):
        with pytest.raises(ValueError):
            mgr(tmp_path, **kw)
    with pytest.raises(ValueError):
        C.CheckpointManager(tmp_path, "../evil")


# ---- 허브 이음매 -------------------------------------------------------------------------------
def test_upload_cadence_and_final_force(tmp_path):
    clk, hub = Clock(), FakeHub()
    m = mgr(tmp_path, clk, hub=make_sync(hub), upload_every_seconds=50.0)
    m.save(4, state(4))
    assert hub.n("upload") == 0 and m.last_upload is None  # 주기 전
    clk.t += 49.0
    m.save(8, state(8))
    assert hub.n("upload") == 0
    clk.t += 2.0
    m.save(12, state(12))
    assert m.last_upload.ok and m.last_upload.remote_path == "ckpt/r06/step_000000012.pt"
    assert hub.files["ckpt/r06/step_000000012.pt"] == (tmp_path / "ck" / "step_000000012.pt").read_bytes()
    assert "ckpt/r06/step_000000012.pt.sha256.json" in hub.files  # 사이드카까지 올라가야 완성본
    n = hub.n("upload")
    clk.t += 10.0
    m.save(16, state(16))
    assert hub.n("upload") == n  # 직전 업로드 후 10초
    m.save(20, state(20), final=True)
    assert hub.n("upload") > n and "ckpt/r06/step_000000020.pt" in hub.files  # final은 주기 무시


def test_maybe_upload_force_without_new_save(tmp_path):
    hub = FakeHub()
    m = mgr(tmp_path, hub=make_sync(hub), upload_every_seconds=1e9)
    m.save(4, state(4))
    assert hub.n("upload") == 0
    r = m.maybe_upload(4, force=True)
    assert r.ok and "ckpt/r06/step_000000004.pt" in hub.files
    assert m.maybe_upload(99, force=True) is None  # 없는 스텝 파일은 조용히 무시


def test_upload_failure_never_breaks_saving(tmp_path):
    class Boom(FakeHub):
        def upload_file(self, local, remote):
            raise H.TransientHubError("net down")

    hub = Boom()
    m = mgr(tmp_path, hub=make_sync(hub))
    p = m.save(4, state(4), final=True)
    assert p.exists() and m.last_upload.ok is False and "net down" in m.last_upload.error
    assert m.resume().step == 4


def test_disabled_hub_and_no_hub_are_silent(tmp_path):
    for hub in (None, H.HubSync(None)):
        m = mgr(tmp_path / ("a" if hub is None else "b"), hub=hub)
        m.save(4, state(4), final=True)
        assert m.last_upload is None
        assert m.resume().step == 4


def test_resume_fetches_from_hub_when_local_is_empty_or_older(tmp_path):
    hub = FakeHub()
    src = mgr(tmp_path / "src", hub=make_sync(hub))
    src.save(4, state(4), final=True)
    src.save(8, state(8), final=True)
    fresh = mgr(tmp_path / "new", hub=make_sync(hub))
    got = fresh.resume(expect_config_hash=HASH)
    assert got.step == 8 and (tmp_path / "new" / "ck" / "step_000000008.pt").exists()
    # 로컬이 허브보다 오래됐으면 허브 것을 받는다
    old = mgr(tmp_path / "old", hub=make_sync(hub))
    old.save(4, state(4))
    assert old.resume().step == 8
    # 로컬이 같거나 새로우면 데이터는 받지 않는다
    before = hub.n("download")
    same = mgr(tmp_path / "new", hub=make_sync(hub))
    assert same.resume().step == 8
    newer = mgr(tmp_path / "new", hub=make_sync(hub))
    newer.save(9, state(9))
    assert newer.resume().step == 9
    assert hub.n("download") == before


def test_resume_survives_hub_failures(tmp_path):
    class Down(FakeHub):
        def list_files(self, prefix):
            raise H.TransientHubError("hub down")

    m = mgr(tmp_path, hub=make_sync(Down()))
    m.save(4, state(4))
    assert m.resume().step == 4  # 허브 장애는 로컬 재개를 막지 않는다
    m2 = mgr(tmp_path / "e", hub=make_sync(Down()))
    assert m2.resume() is None


def test_background_upload_does_not_block_and_flush_waits(tmp_path):
    gate, started = threading.Event(), threading.Event()

    class Slow(FakeHub):
        def upload_file(self, local, remote):
            started.set()
            assert gate.wait(10)
            super().upload_file(local, remote)

    hub, clk = Slow(), Clock()
    m = mgr(tmp_path, clk, hub=make_sync(hub), upload_every_seconds=10.0, background_upload=True)
    clk.t += 11.0
    m.save(4, state(4))  # 업로드가 막혀 있어도 save는 즉시 돌아온다
    assert started.wait(10) and hub.n("upload") == 0
    clk.t += 11.0
    m.save(8, state(8))  # 이전 업로드가 진행 중이면 쌓지 않고 건너뛴다
    gate.set()
    m.flush(timeout=10)
    assert "ckpt/r06/step_000000004.pt" in hub.files and "ckpt/r06/step_000000008.pt" not in hub.files
    assert m.last_upload.ok
    m.save(12, state(12), final=True)  # final은 끝날 때까지 기다린다
    assert "ckpt/r06/step_000000012.pt" in hub.files
