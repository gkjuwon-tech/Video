"""이름·경로 순수 함수: 디스크에 남는 배치이므로 표로 고정한다."""
import pytest

from buc.rt import hf_sync as H


def test_constants():
    assert H.DEFAULT_REPO_ID == "sdsdsdsdsdsdsd123/buc-artifacts"
    assert H.DEFAULT_REPO_TYPE == "model"
    assert H.TOKEN_ENV == "HF_TOKEN"
    assert H.DEFAULT_PREFIX == "ckpt"
    assert H.SIDECAR_SUFFIX == ".sha256.json"


@pytest.mark.parametrize("step,ext,want", [
    (0, "pt", "step_000000000.pt"),
    (123, "pt", "step_000000123.pt"),
    (999_999_999, "safetensors", "step_999999999.safetensors"),
    (1_000_000_000, "pt", "step_1000000000.pt"),
])
def test_checkpoint_name_table(step, ext, want):
    assert H.checkpoint_name(step, ext) == want


def test_checkpoint_name_default_ext():
    assert H.checkpoint_name(7) == "step_000000007.pt"


@pytest.mark.parametrize("step,exc", [(-1, ValueError), (True, TypeError), (1.0, TypeError), ("1", TypeError)])
def test_checkpoint_name_bad_step(step, exc):
    with pytest.raises(exc):
        H.checkpoint_name(step)


@pytest.mark.parametrize("ext", ["", "p.t", "p/t", "pt ", "pü"])
def test_checkpoint_name_bad_ext(ext):
    with pytest.raises(ValueError):
        H.checkpoint_name(1, ext)


@pytest.mark.parametrize("name,want", [
    ("step_000000123.pt", (123, "pt")),
    ("step_000000000.pth", (0, "pth")),
    ("step_1000000000.safetensors", (1_000_000_000, "safetensors")),
])
def test_parse_checkpoint_name(name, want):
    assert H.parse_checkpoint_name(name) == want


@pytest.mark.parametrize("name", [
    "step_000000123.pt.sha256.json",  # 사이드카는 체크포인트가 아니다
    "step_123.pt",  # 9자리 미만
    "step_000000123",  # 확장자 없음
    "step_000000123.pt.tmp",
    "step_000000123.",
    "foo.pt",
    "",
    "xstep_000000123.pt",
])
def test_parse_checkpoint_name_rejects(name):
    assert H.parse_checkpoint_name(name) is None


@pytest.mark.parametrize("step", [0, 1, 999_999_999, 10**9, 10**12])
@pytest.mark.parametrize("ext", ["pt", "safetensors"])
def test_name_roundtrip(step, ext):
    assert H.parse_checkpoint_name(H.checkpoint_name(step, ext)) == (step, ext)


def test_checkpoint_path():
    assert H.checkpoint_path("run-a", 5) == "ckpt/run-a/step_000000005.pt"
    assert H.checkpoint_path("run_1.v2", 7, "safetensors", prefix="vault") == "vault/run_1.v2/step_000000007.safetensors"


@pytest.mark.parametrize("rid", ["a", "run-1", "R_2.x", "0abc", "a" * 128])
def test_run_id_ok(rid):
    H.validate_run_id(rid)


@pytest.mark.parametrize("rid", ["", "-a", ".a", "_a", "a/b", "a b", "..", "a" * 129, "ü", "a\n", "a\x00"])
def test_run_id_bad(rid):
    with pytest.raises(ValueError):
        H.validate_run_id(rid)


@pytest.mark.parametrize("p", ["a", "ckpt/r/step_000000001.pt", "a/b.c-d_e", "x" * 512])
def test_remote_path_ok(p):
    H.validate_remote_path(p)


@pytest.mark.parametrize("p", ["", "/a", "a//b", "a/../b", "..", ".", "a/./b", "a\\b", "a\x00b", "a/", "x" * 513])
def test_remote_path_bad(p):
    with pytest.raises(ValueError):
        H.validate_remote_path(p)
