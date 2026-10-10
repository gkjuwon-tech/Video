"""비공개 저장소 검사, 토큰 해석, 비밀 가림, from_env 배선."""
import logging
import os

import pytest
from run03_fakes import TOKEN, FakeHub

from buc.rt import hf_sync as H


# ---------- 비공개 검사
def test_verify_private_public_repo(hub, sync):
    hub.private = False
    with pytest.raises(H.RepoNotPrivateError):
        sync.verify_private()
    hub.private = True
    sync.verify_private()


def test_public_repo_gets_nothing(hub, sync, ckpt):
    hub.private = False
    with pytest.raises(H.RepoNotPrivateError):
        sync.upload_checkpoint(ckpt(), "run-a", 1)
    with pytest.raises(H.RepoNotPrivateError):
        sync.upload_file(ckpt("b.bin"), "vault/b.bin")
    assert hub.count("upload") == 0 and not hub.files


def test_private_true_is_cached_false_is_not(hub, sync, ckpt):
    sync.upload_checkpoint(ckpt(), "run-a", 1)
    sync.upload_checkpoint(ckpt(), "run-a", 2)
    assert hub.count("is_private") == 1
    hub2 = FakeHub(private=False)
    s2 = H.HubSync(hub2, sleep=lambda _: None)
    with pytest.raises(H.RepoNotPrivateError):
        s2.upload_checkpoint(ckpt(), "run-a", 1)
    hub2.private = True  # 공개 판정은 캐시되지 않으므로 바뀐 뒤 다시 확인한다
    assert s2.upload_checkpoint(ckpt(), "run-a", 1).ok
    assert hub2.count("is_private") == 2


# ---------- 토큰 해석
def write_token(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_resolve_token_precedence(tmp_path):
    f = write_token(tmp_path / "tok", "file-token\n")
    assert H.resolve_token(env={"HF_TOKEN": "env-token"}, token_path=f) == "env-token"
    assert H.resolve_token(env={}, token_path=f) == "file-token"
    assert H.resolve_token(env={"HF_TOKEN": "  "}, token_path=f) == "file-token"  # 공백뿐인 환경변수는 없는 것
    assert H.resolve_token(env={"HF_TOKEN": " \tenv-token\n"}, token_path=f) == "env-token"


def test_resolve_token_none(tmp_path):
    assert H.resolve_token(env={}, token_path=tmp_path / "missing") is None
    assert H.resolve_token(env={}, token_path=write_token(tmp_path / "empty", " \n")) is None


def test_resolve_token_defaults_use_process_env_and_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert H.resolve_token() is None
    write_token(tmp_path / ".cache" / "huggingface" / "token", "home-token\n")
    assert H.resolve_token() == "home-token"
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert H.resolve_token() == "env-token"


# ---------- 가림
def test_redact():
    assert H.redact(f"Bearer {TOKEN} failed; again {TOKEN}", [TOKEN]) == "Bearer *** failed; again ***"
    assert H.redact("abc", [""]) == "abc"  # 빈 비밀로 모든 글자를 바꾸면 안 된다
    assert H.redact("", [TOKEN]) == ""
    assert H.redact("a-SECRET1-b-SECRET2", ["SECRET1", "SECRET2"]) == "a-***-b-***"
    assert H.redact("nothing here", []) == "nothing here"


def test_hubconfig_never_shows_token():
    c = H.HubConfig(token=TOKEN)
    assert (c.repo_id, c.repo_type) == (H.DEFAULT_REPO_ID, H.DEFAULT_REPO_TYPE)
    for text in (repr(c), str(c), f"{c}", f"{c!r}"):
        assert TOKEN not in text
    with pytest.raises(Exception):
        c.repo_id = "x"  # frozen


def leaky_hub(msg_kind="transient"):
    h = FakeHub()
    exc = {"transient": H.TransientHubError, "auth": H.HubAuthError}[msg_kind](f"401 for Bearer {TOKEN} at /api")
    h.add_fault("upload", ".pt", exc, times=None)
    return h


@pytest.mark.parametrize("kind", ["transient", "auth"])
def test_strict_errors_do_not_carry_token(ckpt, kind, caplog):
    caplog.set_level(logging.DEBUG)
    s = H.HubSync(leaky_hub(kind), sleep=lambda _: None, secrets=(TOKEN,))
    with pytest.raises(H.HubSyncError) as ei:
        s.upload_checkpoint(ckpt(), "run-a", 1)
    assert TOKEN not in str(ei.value) and TOKEN not in repr(ei.value)
    assert "Bearer" in str(ei.value)  # 나머지 진단 정보는 보존
    assert TOKEN not in caplog.text


def test_best_effort_result_and_logs_do_not_carry_token(ckpt, caplog):
    caplog.set_level(logging.DEBUG)
    s = H.HubSync(leaky_hub(), sleep=lambda _: None, secrets=(TOKEN,))
    r = s.try_upload_checkpoint(ckpt(), "run-a", 1)
    assert not r.ok and r.error and TOKEN not in r.error and "***" in r.error
    assert TOKEN not in caplog.text and TOKEN not in repr(r)


# ---------- from_env
def test_from_env_without_token_is_disabled_and_never_builds_client(ckpt, tmp_path):
    def factory(**kw):
        raise AssertionError("토큰이 없으면 클라이언트를 만들면 안 된다")
    s = H.HubSync.from_env(env={}, token_path=tmp_path / "none", client_factory=factory)
    with pytest.raises(H.HubAuthError):
        s.upload_checkpoint(ckpt(), "run-a", 1)
    with pytest.raises(H.HubAuthError):
        s.list_checkpoints("run-a")
    with pytest.raises(H.HubAuthError):
        s.fetch_latest("run-a", tmp_path / "d")
    with pytest.raises(H.HubAuthError):
        s.download_file("vault/a.bin", tmp_path / "a.bin")
    assert s.try_upload_checkpoint(ckpt(), "run-a", 1).skipped == "no_token"


def test_from_env_builds_client_with_defaults(ckpt, tmp_path, sleeps):
    seen = {}
    hub = FakeHub()

    def factory(**kw):
        seen.update(kw)
        return hub

    s = H.HubSync.from_env(env={"HF_TOKEN": TOKEN}, token_path=tmp_path / "none", client_factory=factory,
                           sleep=sleeps.append)
    assert seen == {"repo_id": "sdsdsdsdsdsdsd123/buc-artifacts", "repo_type": "model", "token": TOKEN}
    assert s.upload_checkpoint(ckpt(), "run-a", 1).ok


def test_from_env_custom_repo_and_secret_wiring(ckpt, tmp_path):
    seen = {}
    hub = leaky_hub()

    def factory(**kw):
        seen.update(kw)
        return hub

    s = H.HubSync.from_env(env={"HF_TOKEN": TOKEN}, repo_id="me/other", repo_type="dataset", client_factory=factory,
                           sleep=lambda _: None)
    assert seen["repo_id"] == "me/other" and seen["repo_type"] == "dataset"
    r = s.try_upload_checkpoint(ckpt(), "run-a", 1)
    assert not r.ok and TOKEN not in r.error  # from_env가 secrets=(token,)을 배선했다


def test_from_env_reads_process_env_by_default(ckpt, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    seen = {}
    H.HubSync.from_env(client_factory=lambda **kw: seen.update(kw) or FakeHub())
    assert seen["token"] == TOKEN
    assert os.environ["HF_TOKEN"] == TOKEN
