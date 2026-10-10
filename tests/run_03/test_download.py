"""다운로드: 무결성 검증, 원자성(.part → replace), 재개, 일시 손상 재시도."""
import hashlib
import json
from pathlib import Path

import pytest
from run03_fakes import SIDE, blob, put_complete, sidecar_bytes

from buc.rt import hf_sync as H

R = "ckpt/run-a/step_000000042.pt"


def data_downloads(hub):
    return [c for c in hub.calls if c[0] == "download" and not c[1].endswith(SIDE)]


def flip(remote, data, nth):
    return bytes([data[0] ^ 1]) + data[1:]


def test_roundtrip_creates_dirs_and_bytes_equal(hub, sync, ckpt, tmp_path):
    data = blob(1024 * 1024 + 123, 5)
    sync.upload_checkpoint(ckpt(data=data), "run-a", 42)
    dest_dir = tmp_path / "deep" / "er" / "dir"
    out = sync.download_checkpoint("run-a", dest_dir)
    assert out == dest_dir / "step_000000042.pt"
    assert out.read_bytes() == data
    assert sorted(p.name for p in dest_dir.iterdir()) == ["step_000000042.pt"]  # .part·임시 파일 없음


def test_step_selection_and_missing(hub, sync, ckpt, tmp_path):
    for s in (1, 5, 10):
        sync.upload_checkpoint(ckpt(data=blob(50, s)), "run-a", s)
    assert sync.download_checkpoint("run-a", tmp_path / "d1").name == "step_000000010.pt"  # None = 최신
    assert sync.download_checkpoint("run-a", tmp_path / "d2", step=5).read_bytes() == blob(50, 5)
    with pytest.raises(H.HubNotFoundError):
        sync.download_checkpoint("run-a", tmp_path / "d3", step=6)
    with pytest.raises(H.HubNotFoundError):
        sync.download_checkpoint("run-none", tmp_path / "d4")
    assert not (tmp_path / "d3").exists() or not any((tmp_path / "d3").iterdir())


def test_fetch_latest(hub, sync, ckpt, tmp_path):
    assert sync.fetch_latest("run-a", tmp_path / "d") is None  # 새로 시작
    sync.upload_checkpoint(ckpt(data=b"one" * 30), "run-a", 1)
    sync.upload_checkpoint(ckpt(data=b"two" * 30), "run-a", 2)
    p = sync.fetch_latest("run-a", tmp_path / "d")
    assert p == tmp_path / "d" / "step_000000002.pt" and p.read_bytes() == b"two" * 30


def test_data_is_written_to_part_file_next_to_dest(hub, sync, ckpt, tmp_path):
    sync.upload_checkpoint(ckpt(), "run-a", 42)
    dest = tmp_path / "out" / "x.pt"
    sync.download_file(R, dest)
    (call,) = data_downloads(hub)
    assert Path(call[2]) == dest.with_name("x.pt.part")


def test_persistent_corruption_raises_leaves_nothing(hub, sync, ckpt, tmp_path, sleeps):
    sync.upload_checkpoint(ckpt(), "run-a", 42)
    hub.mutator = flip
    hub.calls.clear()
    dest_dir = tmp_path / "out"
    with pytest.raises(H.IntegrityError):
        sync.download_checkpoint("run-a", dest_dir)
    assert len(data_downloads(hub)) == 4  # 기본 max_attempts
    assert sleeps == [1.0, 2.0, 4.0]
    assert not dest_dir.exists() or list(dest_dir.iterdir()) == []


def test_failed_download_keeps_existing_dest(hub, sync, ckpt, tmp_path):
    sync.upload_checkpoint(ckpt(), "run-a", 42)
    hub.mutator = flip
    dest = tmp_path / "x.pt"
    dest.write_bytes(b"old")
    with pytest.raises(H.IntegrityError):
        sync.download_file(R, dest, force=True)
    assert dest.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src", "x.pt"]  # .part 잔재 없음


def test_transient_corruption_is_retried(hub, sync, ckpt, tmp_path, sleeps):
    data = blob(4000, 8)
    sync.upload_checkpoint(ckpt(data=data), "run-a", 42)
    hub.mutator = lambda remote, d, nth: flip(remote, d, nth) if nth == 1 else d
    hub.calls.clear()
    out = sync.download_checkpoint("run-a", tmp_path / "o")
    assert out.read_bytes() == data
    assert len(data_downloads(hub)) == 2
    assert sleeps == [1.0]


def test_truncation_detected(hub, sync, ckpt, tmp_path):
    sync.upload_checkpoint(ckpt(data=blob(500, 1)), "run-a", 42)
    hub.mutator = lambda remote, d, nth: d[:-1]
    with pytest.raises(H.IntegrityError):
        sync.download_file(R, tmp_path / "t.pt")
    assert not (tmp_path / "t.pt").exists()


def test_size_in_sidecar_is_enforced_even_if_sha_matches(hub, sync, tmp_path):
    data = b"payload" * 20
    hub.files[R] = data
    hub.files[R + SIDE] = json.dumps({"v": 1, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data) + 1},
                                     sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(H.IntegrityError):
        sync.download_file(R, tmp_path / "s.pt")
    assert not (tmp_path / "s.pt").exists()


@pytest.mark.parametrize("bad", [
    b"not json",
    b"[]",
    b'{"v":2,"sha256":"' + b"0" * 64 + b'","size":3}',  # 모르는 버전
    b'{"v":1,"sha256":"xyz","size":3}',
    b'{"v":1,"sha256":"' + b"0" * 64 + b'","size":-1}',
    b'{"v":1,"sha256":"' + b"0" * 64 + b'"}',
])
def test_malformed_sidecar_is_integrity_error(hub, sync, tmp_path, bad):
    hub.files[R] = b"abc"
    hub.files[R + SIDE] = bad
    with pytest.raises(H.IntegrityError):
        sync.download_file(R, tmp_path / "m.pt")
    assert not (tmp_path / "m.pt").exists()


def test_missing_sidecar_is_not_found_and_data_untouched(hub, sync, tmp_path):
    hub.files[R] = b"abc"  # 사이드카 없음 = 미완성
    with pytest.raises(H.HubNotFoundError):
        sync.download_file(R, tmp_path / "n.pt")
    assert not (tmp_path / "n.pt").exists()
    assert data_downloads(hub) == []


def test_skip_when_local_matches_and_force_redownloads(hub, sync, ckpt, tmp_path):
    data = blob(3000, 4)
    sync.upload_checkpoint(ckpt(data=data), "run-a", 42)
    dest = tmp_path / "k.pt"
    sync.download_file(R, dest)
    assert len(data_downloads(hub)) == 1
    assert sync.download_file(R, dest) == dest
    assert len(data_downloads(hub)) == 1  # 일치하면 데이터 다운로드 0회
    sync.download_file(R, dest, force=True)
    assert len(data_downloads(hub)) == 2


def test_local_mismatch_is_replaced(hub, sync, ckpt, tmp_path):
    data = blob(3000, 4)
    sync.upload_checkpoint(ckpt(data=data), "run-a", 42)
    dest = tmp_path / "k.pt"
    dest.write_bytes(b"stale junk")
    sync.download_file(R, dest)
    assert dest.read_bytes() == data


def test_download_file_generic_and_path_validation(hub, sync, tmp_path):
    put_complete(hub, "vault/x/y.bin", b"zzz")
    out = sync.download_file("vault/x/y.bin", tmp_path / "a" / "b" / "y.bin")
    assert out.read_bytes() == b"zzz"
    with pytest.raises(ValueError):
        sync.download_file("../x", tmp_path / "q")
    with pytest.raises(H.HubNotFoundError):
        sync.download_file("vault/none.bin", tmp_path / "q")


def test_sidecar_oracle_matches_what_upload_wrote(hub, sync, ckpt):
    p = ckpt(data=blob(777, 2))
    sync.upload_checkpoint(p, "run-a", 42)
    assert hub.files[R + SIDE] == sidecar_bytes(p.read_bytes())
