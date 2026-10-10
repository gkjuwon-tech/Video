"""업로드: 사이드카 바이트 배치, 데이터→사이드카 순서, 완성본 판정, 멱등·충돌."""
import hashlib

import pytest
from run03_fakes import SIDE, blob, put_complete, sidecar_bytes

from buc.rt import hf_sync as H

R = "ckpt/run-a/step_000000042.pt"


def uploads(hub):
    return [c[1] for c in hub.calls if c[0] == "upload"]


def test_layout_sidecar_bytes_and_result(hub, sync, ckpt):
    data = blob(2 * 1024 * 1024 + 512 * 1024 + 17, seed=1)  # 1 MiB 청크 경계를 넘긴다
    r = sync.upload_checkpoint(ckpt("m.pt", data), "run-a", 42)
    assert r.ok and r.uploaded and r.skipped is None and r.error is None
    assert r.remote_path == R
    assert r.sha256 == hashlib.sha256(data).hexdigest()
    assert r.size == len(data)
    assert set(hub.files) == {R, R + SIDE}
    assert hub.files[R] == data
    assert hub.files[R + SIDE] == sidecar_bytes(data)


def test_ext_follows_local_suffix(hub, sync, ckpt):
    r = sync.upload_checkpoint(ckpt("w.safetensors"), "run-a", 3)
    assert r.remote_path == "ckpt/run-a/step_000000003.safetensors"


def test_upload_checks_private_first_then_data_then_sidecar(hub, sync, ckpt):
    sync.upload_checkpoint(ckpt(), "run-a", 42)
    ops = [c[0] for c in hub.calls if c[0] in ("is_private", "upload")]
    assert ops[0] == "is_private"
    assert uploads(hub) == [R, R + SIDE]  # 데이터 먼저, 사이드카 나중


def test_data_failure_leaves_no_sidecar_and_no_checkpoint(hub, sync, ckpt):
    hub.add_fault("upload", "step_000000042.pt", H.HubAuthError("denied"), times=None)
    with pytest.raises(H.HubAuthError):
        sync.upload_checkpoint(ckpt(), "run-a", 42)
    assert not any(k.endswith(SIDE) for k in hub.files)
    assert sync.list_checkpoints("run-a") == []


def test_sidecar_failure_hides_data(hub, sync, ckpt):
    hub.add_fault("upload", SIDE, H.TransientHubError("flaky"), times=None)
    with pytest.raises(H.TransientHubError):
        sync.upload_checkpoint(ckpt(), "run-a", 42)
    assert R in hub.files and R + SIDE not in hub.files  # 데이터만 있는 상태
    assert sync.list_checkpoints("run-a") == []
    assert sync.latest_checkpoint("run-a") is None


def test_latest_is_numeric_not_lexicographic(hub, sync, ckpt):
    steps = [2, 9, 10, 999_999_999, 1_000_000_000]
    for s in (9, 1_000_000_000, 2, 999_999_999, 10):  # 일부러 섞인 순서
        sync.upload_checkpoint(ckpt(data=blob(64, s % 97)), "run-a", s)
    refs = sync.list_checkpoints("run-a")
    assert [r.step for r in refs] == steps
    assert all(isinstance(r, H.CheckpointRef) and r.run_id == "run-a" for r in refs)
    last = sync.latest_checkpoint("run-a")
    assert last.step == 1_000_000_000
    assert last.remote_path == "ckpt/run-a/step_1000000000.pt"
    assert last == refs[-1]


def test_listing_ignores_orphans_other_runs_and_foreign_files(hub, sync):
    put_complete(hub, "ckpt/run-a/step_000000005.pt", b"five")
    hub.files["ckpt/run-a/step_000000050.pt"] = b"orphan data, no sidecar"
    hub.files["ckpt/run-a/step_000000060.pt" + SIDE] = sidecar_bytes(b"x")  # 데이터 없는 사이드카
    hub.files["ckpt/run-a/step_000000070.pt.tmp"] = b"partial"
    hub.files["ckpt/run-a/notes.txt"] = b"hello"
    put_complete(hub, "ckpt/run-ab/step_000000099.pt", b"other run (prefix lookalike)")
    put_complete(hub, "ckpt/other/step_000000098.pt", b"other run")
    put_complete(hub, "elsewhere/run-a/step_000000097.pt", b"other prefix")
    assert [(r.step, r.remote_path) for r in sync.list_checkpoints("run-a")] == [(5, "ckpt/run-a/step_000000005.pt")]
    assert sync.latest_checkpoint("run-a").step == 5
    assert [r.step for r in sync.list_checkpoints("run-ab")] == [99]
    assert sync.list_checkpoints("run-zzz") == []
    assert sync.latest_checkpoint("run-zzz") is None


def test_listing_never_downloads(hub, sync):
    put_complete(hub, "ckpt/run-a/step_000000005.pt", b"five")
    sync.list_checkpoints("run-a")
    sync.latest_checkpoint("run-a")
    assert hub.count("download") == 0


def test_custom_prefix(hub, ckpt, sleeps):
    s = H.HubSync(hub, prefix="vault", sleep=sleeps.append)
    r = s.upload_checkpoint(ckpt(), "run-a", 1)
    assert r.remote_path == "vault/run-a/step_000000001.pt"
    assert [x.step for x in s.list_checkpoints("run-a")] == [1]


def test_reupload_is_idempotent(hub, sync, ckpt):
    p = ckpt()
    sync.upload_checkpoint(p, "run-a", 42)
    n = hub.count("upload")
    r = sync.upload_checkpoint(p, "run-a", 42)
    assert r.ok and not r.uploaded and r.skipped == "up_to_date"
    assert r.remote_path == R and r.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert hub.count("upload") == n  # 클라이언트 업로드 0회 추가


def test_different_content_conflicts_unless_overwrite(hub, sync, ckpt):
    sync.upload_checkpoint(ckpt(data=b"v1" * 100), "run-a", 42)
    new = ckpt("m2.pt", b"v2" * 100)
    with pytest.raises(H.ConflictError):
        sync.upload_checkpoint(new, "run-a", 42)
    assert hub.files[R] == b"v1" * 100 and hub.files[R + SIDE] == sidecar_bytes(b"v1" * 100)
    r = sync.upload_checkpoint(new, "run-a", 42, overwrite=True)
    assert r.ok and r.uploaded
    assert hub.files[R] == b"v2" * 100 and hub.files[R + SIDE] == sidecar_bytes(b"v2" * 100)


def test_reupload_completes_half_finished_remote(hub, sync, ckpt):
    p = ckpt()
    hub.files[R] = p.read_bytes()  # 이전 실행이 데이터만 올리고 죽었다
    r = sync.upload_checkpoint(p, "run-a", 42)
    assert r.ok and r.uploaded
    assert hub.files[R + SIDE] == sidecar_bytes(p.read_bytes())
    assert [x.step for x in sync.list_checkpoints("run-a")] == [42]


@pytest.mark.parametrize("name", ["c.pt.tmp", "c.pt.part", "c.tmp"])
def test_temp_files_refused(hub, sync, ckpt, name):
    with pytest.raises(ValueError):
        sync.upload_checkpoint(ckpt(name), "run-a", 1)
    assert hub.count("upload") == 0 and not hub.files


def test_missing_file_and_no_extension(hub, sync, ckpt, tmp_path):
    with pytest.raises(FileNotFoundError):
        sync.upload_checkpoint(tmp_path / "nope.pt", "run-a", 1)
    with pytest.raises(ValueError):
        sync.upload_checkpoint(ckpt("noext"), "run-a", 1)
    assert hub.count("upload") == 0


@pytest.mark.parametrize("run_id,step,exc", [("a/b", 1, ValueError), ("", 1, ValueError), ("ok", -1, ValueError), ("ok", True, TypeError)])
def test_bad_args_touch_nothing(hub, sync, ckpt, run_id, step, exc):
    with pytest.raises(exc):
        sync.upload_checkpoint(ckpt(), run_id, step)
    assert hub.count("upload") == 0 and not hub.files


def test_generic_upload_file_writes_data_then_sidecar(hub, sync, ckpt):
    p = ckpt("blob.bin", blob(5000, 3))
    r = sync.upload_file(p, "vault/x/y.bin")
    assert r.ok and r.uploaded and r.remote_path == "vault/x/y.bin"
    assert uploads(hub) == ["vault/x/y.bin", "vault/x/y.bin" + SIDE]
    assert hub.files["vault/x/y.bin" + SIDE] == sidecar_bytes(p.read_bytes())
    with pytest.raises(ValueError):
        sync.upload_file(p, "../escape.bin")
    with pytest.raises(ValueError):
        sync.upload_file(p, "/abs.bin")
