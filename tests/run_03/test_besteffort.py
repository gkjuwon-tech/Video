"""최선 노력 업로드: 학습을 죽이지 않는다. 인자 오류는 숨기지 않는다."""
import logging

import pytest

from buc.rt import hf_sync as H


def test_success_matches_strict(hub, sync, ckpt):
    r = sync.try_upload_checkpoint(ckpt(), "run-a", 1)
    assert r.ok and r.uploaded and r.error is None and r.skipped is None
    assert r.remote_path == "ckpt/run-a/step_000000001.pt"
    assert [x.step for x in sync.list_checkpoints("run-a")] == [1]


@pytest.mark.parametrize("make", [
    lambda h: h.add_fault("upload", ".pt", H.HubAuthError("denied"), times=None),
    lambda h: h.add_fault("upload", ".pt", H.TransientHubError("down"), times=None),
    lambda h: h.add_fault("upload", ".pt", OSError("disk"), times=None),
    lambda h: setattr(h, "private", False),
    lambda h: h.add_fault("is_private", "", H.TransientHubError("down"), times=None),
])
def test_failures_never_raise(hub, sync, ckpt, caplog, make):
    make(hub)
    caplog.set_level(logging.DEBUG)
    r = sync.try_upload_checkpoint(ckpt(), "run-a", 1)
    assert r.ok is False and r.uploaded is False and r.skipped is None
    assert isinstance(r.error, str) and r.error
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert not any(k.endswith(".sha256.json") for k in hub.files)


def test_custom_logger_receives_warning(hub, sleeps, ckpt, caplog):
    hub.add_fault("upload", ".pt", H.HubAuthError("denied"), times=None)
    s = H.HubSync(hub, sleep=sleeps.append, log=logging.getLogger("my.custom"))
    with caplog.at_level(logging.WARNING):
        s.try_upload_checkpoint(ckpt(), "run-a", 1)
    assert any(rec.name == "my.custom" and rec.levelno >= logging.WARNING for rec in caplog.records)


def test_no_sticky_state_after_failure(hub, sync, ckpt):
    hub.add_fault("upload", ".pt", H.HubAuthError("denied"), times=1)
    assert not sync.try_upload_checkpoint(ckpt(), "run-a", 1).ok
    assert sync.try_upload_checkpoint(ckpt(), "run-a", 1).ok


def test_disabled_client_skips_and_logs_marker(ckpt, caplog):
    s = H.HubSync(None)
    with caplog.at_level(logging.INFO):
        r = s.try_upload_checkpoint(ckpt(), "run-a", 1)
    assert r.ok is False and r.uploaded is False and r.skipped == "no_token" and r.error is None
    assert "hf_token=no" in caplog.text


def test_disabled_client_strict_methods_raise_auth(ckpt, tmp_path):
    s = H.HubSync(None)
    with pytest.raises(H.HubAuthError):
        s.upload_file(ckpt(), "vault/a.pt")
    with pytest.raises(H.HubAuthError):
        s.latest_checkpoint("run-a")
    with pytest.raises(H.HubAuthError):
        s.download_checkpoint("run-a", tmp_path / "d")


def test_missing_local_file_is_soft_in_best_effort_hard_in_strict(hub, sync, tmp_path):
    gone = tmp_path / "gone.pt"  # 트레이너가 파일을 지운 경우
    r = sync.try_upload_checkpoint(gone, "run-a", 1)
    assert not r.ok and r.error
    with pytest.raises(FileNotFoundError):
        sync.upload_checkpoint(gone, "run-a", 1)


@pytest.mark.parametrize("name,run_id,step,exc", [
    ("c.pt", "bad/id", 1, ValueError),
    ("c.pt", "run-a", -1, ValueError),
    ("c.pt", "run-a", True, TypeError),
    ("c.pt.tmp", "run-a", 1, ValueError),
])
def test_argument_errors_still_raise(hub, sync, ckpt, name, run_id, step, exc):
    with pytest.raises(exc):
        sync.try_upload_checkpoint(ckpt(name), run_id, step)
    assert hub.count("upload") == 0
