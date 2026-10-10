"""재시도 정책: 백오프 대기열은 결정론적이고(지터 없음), 대상 예외만 재시도한다."""
import pytest
from run03_fakes import SIDE, put_complete

from buc.rt import hf_sync as H


def test_policy_defaults():
    p = H.RetryPolicy()
    assert (p.max_attempts, p.base_delay, p.factor, p.max_delay) == (4, 1.0, 2.0, 30.0)


@pytest.mark.parametrize("kw", [
    dict(max_attempts=0), dict(base_delay=-0.1), dict(factor=0.9), dict(base_delay=5.0, max_delay=4.0),
])
def test_policy_validation(kw):
    with pytest.raises(ValueError):
        H.RetryPolicy(**kw)


def test_exception_hierarchy():
    for e in (H.HubAuthError, H.HubNotFoundError, H.TransientHubError, H.IntegrityError, H.ConflictError,
              H.RepoNotPrivateError):
        assert issubclass(e, H.HubSyncError)
    assert issubclass(H.HubSyncError, RuntimeError)
    assert issubclass(H.HubNotFoundError, FileNotFoundError)


def test_transient_then_success(hub, ckpt, sleeps):
    s = H.HubSync(hub, sleep=sleeps.append)
    hub.add_fault("upload", "step_000000001.pt", H.TransientHubError("flaky"), times=2)
    r = s.upload_checkpoint(ckpt(), "run-a", 1)
    assert r.ok and r.uploaded
    assert sleeps == [1.0, 2.0]
    assert hub.count("upload", "step_000000001.pt", data_only=True) == 3
    assert hub.count("upload", SIDE) == 1


def test_exhausted_raises_last_error_without_sleeping_after_last(hub, ckpt, sleeps):
    s = H.HubSync(hub, sleep=sleeps.append)
    hub.add_fault("upload", "step_000000001.pt", H.TransientHubError("down"), times=None)
    with pytest.raises(H.TransientHubError):
        s.upload_checkpoint(ckpt(), "run-a", 1)
    assert hub.count("upload", data_only=True) == 4
    assert sleeps == [1.0, 2.0, 4.0]
    assert hub.count("upload", SIDE) == 0  # 데이터가 실패했으니 사이드카는 시도조차 안 한다


def test_delay_cap(hub, ckpt, sleeps):
    pol = H.RetryPolicy(max_attempts=6, base_delay=1.0, factor=3.0, max_delay=5.0)
    s = H.HubSync(hub, retry=pol, sleep=sleeps.append)
    hub.add_fault("upload", ".pt", H.TransientHubError("down"), times=None)
    with pytest.raises(H.TransientHubError):
        s.upload_checkpoint(ckpt(), "run-a", 1)
    assert sleeps == pytest.approx([1.0, 3.0, 5.0, 5.0, 5.0])


def test_single_attempt_policy(hub, ckpt, sleeps):
    s = H.HubSync(hub, retry=H.RetryPolicy(max_attempts=1), sleep=sleeps.append)
    hub.add_fault("upload", ".pt", H.TransientHubError("x"), times=None)
    with pytest.raises(H.TransientHubError):
        s.upload_checkpoint(ckpt(), "run-a", 1)
    assert hub.count("upload") == 1 and sleeps == []


@pytest.mark.parametrize("exc", [H.HubAuthError("no"), H.HubNotFoundError("gone"), KeyError("bug")])
def test_non_retryable_errors_fail_fast(hub, ckpt, sleeps, exc):
    s = H.HubSync(hub, sleep=sleeps.append)
    hub.add_fault("upload", ".pt", exc, times=None)
    with pytest.raises(type(exc)):
        s.upload_checkpoint(ckpt(), "run-a", 1)
    assert hub.count("upload") == 1 and sleeps == []


def test_download_not_found_fails_fast(hub, sync, tmp_path, sleeps):
    with pytest.raises(H.HubNotFoundError):
        sync.download_file("vault/none.bin", tmp_path / "q")
    assert hub.count("download") == 1 and sleeps == []


def test_listing_is_retried(hub, sync, sleeps):
    put_complete(hub, "ckpt/run-a/step_000000005.pt", b"five")
    hub.add_fault("list", "", H.TransientHubError("blip"), times=1)
    assert [r.step for r in sync.list_checkpoints("run-a")] == [5]
    assert sleeps == [1.0]


def test_download_transient_is_retried(hub, sync, tmp_path, sleeps):
    put_complete(hub, "vault/a.bin", b"data")
    hub.add_fault("download", "vault/a.bin", H.TransientHubError("blip"), times=1)
    assert sync.download_file("vault/a.bin", tmp_path / "a.bin").read_bytes() == b"data"
    assert sleeps == [1.0]
