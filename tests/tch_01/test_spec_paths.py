"""TCH-01 스펙 검증, HF 캐시 경로 해석, import 위생."""
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from buc.teacher import zimage as Z
from tch01_fakes import REV, make_hub

ROOT = Path(__file__).resolve().parents[2]


def test_public_api_exported():
    for n in ("TeacherError", "TeacherDependencyError", "TeacherSpecError", "TeacherWeightsMissing",
              "TEUnloadedError", "TeacherNumericError", "ZIMAGE_REPO", "OFFICIAL_REPOS", "REQUIRED_SUBDIRS",
              "TeacherSpec", "hub_cache_dir", "snapshot_dir", "enable_determinism_mode", "make_noise",
              "module_bytes", "module_fingerprint", "TextEmbedCache", "VaeCodec", "Loaders", "default_loaders",
              "load_teacher", "ZImageTeacher"):
        assert hasattr(Z, n), n


def test_exception_hierarchy():
    assert issubclass(Z.TeacherDependencyError, (Z.TeacherError, ImportError))
    assert issubclass(Z.TeacherSpecError, (Z.TeacherError, ValueError))
    assert issubclass(Z.TeacherWeightsMissing, (Z.TeacherError, FileNotFoundError))
    assert issubclass(Z.TeacherNumericError, (Z.TeacherError, FloatingPointError))
    assert issubclass(Z.TEUnloadedError, Z.TeacherError)
    assert issubclass(Z.TeacherError, RuntimeError)


def test_spec_defaults_match_design_numbers():
    s = Z.TeacherSpec()
    assert (s.repo, s.revision) == ("Tongyi-MAI/Z-Image-Turbo", None)
    assert (s.num_layers, s.hidden_dim) == (30, 3840)           # PARADIGM: 6.15B S3-DiT, 30층, d=3840
    assert (s.latent_channels, s.vae_scale, s.patch_size) == (16, 8, 2)
    assert s.text_dim == 2560                                    # Qwen3-4B
    assert Z.ZIMAGE_REPO in Z.OFFICIAL_REPOS and len(Z.OFFICIAL_REPOS) == 1
    assert Z.REQUIRED_SUBDIRS == ("transformer", "vae", "text_encoder", "tokenizer")
    s.validate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.repo = "x/y"


@pytest.mark.parametrize("repo", ["madebyollin/sdxl-vae-fp16-fix", "someone/Z-Image-Turbo-fp8", "Tongyi-MAI/Z-Image"])
def test_spec_rejects_third_party_repos(repo):
    with pytest.raises(Z.TeacherSpecError):
        Z.TeacherSpec(repo=repo).validate()
    Z.TeacherSpec(repo=repo, unofficial_ok=True).validate()


@pytest.mark.parametrize("field", ["num_layers", "hidden_dim", "latent_channels", "vae_scale", "patch_size",
                                   "text_dim", "max_text_len"])
def test_spec_rejects_nonpositive_ints(field):
    for bad in (0, -1):
        with pytest.raises(Z.TeacherSpecError):
            dataclasses.replace(Z.TeacherSpec(), **{field: bad}).validate()


@pytest.mark.parametrize("rev", ["../etc", "a/b", "", "a b", "x\n"])
def test_spec_rejects_bad_revision(rev):
    with pytest.raises(Z.TeacherSpecError):
        Z.TeacherSpec(revision=rev).validate()
    Z.TeacherSpec(revision="abc123.DEF_-9").validate()


def test_image_size_and_latent_shape():
    s = Z.TeacherSpec()
    s.check_image_size(1024, 1024)
    s.check_image_size(512, 768)
    for h, w in [(1000, 1024), (1024, 1020), (8, 16), (0, 16), (16, 0), (-16, 16), (1024, 1032)]:
        with pytest.raises(ValueError):
            s.check_image_size(h, w)
    assert s.latent_shape(1024, 1024) == (1, 16, 128, 128)
    assert s.latent_shape(512, 768, batch=3) == (3, 16, 64, 96)
    with pytest.raises(ValueError):
        s.latent_shape(1000, 1024)


def test_hub_cache_dir_precedence():
    assert Z.hub_cache_dir({"HF_HUB_CACHE": "/a/hub", "HF_HOME": "/b", "BUC_ROOT": "/c"}) == Path("/a/hub")
    assert Z.hub_cache_dir({"HF_HOME": "/b", "BUC_ROOT": "/c"}) == Path("/b/hub")
    assert Z.hub_cache_dir({"BUC_ROOT": "/c"}) == Path("/c/hf/hub")
    assert Z.hub_cache_dir({}) == Path("/workspace/buc/hf/hub")


def test_hub_cache_dir_reads_os_environ_by_default(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", "/tmp/xx_hf_home")
    assert Z.hub_cache_dir() == Path("/tmp/xx_hf_home/hub")


def test_snapshot_dir_via_refs_main(tmp_path):
    hub = make_hub(tmp_path)
    p = Z.snapshot_dir(Z.TeacherSpec(), hub_cache=hub)
    assert p == hub / "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots" / REV
    for s in Z.REQUIRED_SUBDIRS:
        assert (p / s).is_dir()


def test_snapshot_dir_explicit_revision_beats_ref(tmp_path):
    hub = make_hub(tmp_path, rev="b" * 40, via_ref=False)
    make_hub(tmp_path)  # refs/main -> REV (같은 hub에 두 번째 스냅샷)
    p = Z.snapshot_dir(Z.TeacherSpec(revision="b" * 40), hub_cache=hub)
    assert p.name == "b" * 40
    assert Z.snapshot_dir(Z.TeacherSpec(), hub_cache=hub).name == REV


def test_snapshot_dir_uses_env_when_no_hub_given(tmp_path):
    hub = make_hub(tmp_path)
    p = Z.snapshot_dir(Z.TeacherSpec(), env={"HF_HUB_CACHE": str(hub)})
    assert p.name == REV


def test_snapshot_dir_missing_pieces(tmp_path):
    with pytest.raises(Z.TeacherWeightsMissing):
        Z.snapshot_dir(Z.TeacherSpec(), hub_cache=tmp_path / "nope")
    hub = make_hub(tmp_path / "a", via_ref=False)  # refs/main 없음, revision 미지정
    with pytest.raises(Z.TeacherWeightsMissing):
        Z.snapshot_dir(Z.TeacherSpec(), hub_cache=hub)
    hub = make_hub(tmp_path / "b", subdirs=("transformer", "text_encoder", "tokenizer"))  # vae 없음
    with pytest.raises(Z.TeacherWeightsMissing) as ei:
        Z.snapshot_dir(Z.TeacherSpec(), hub_cache=hub)
    assert "vae" in str(ei.value)
    with pytest.raises(Z.TeacherWeightsMissing):  # 지정한 revision이 스냅샷에 없음
        Z.snapshot_dir(Z.TeacherSpec(revision="c" * 40), hub_cache=make_hub(tmp_path / "c"))


def test_snapshot_dir_validates_spec_first(tmp_path):
    with pytest.raises(Z.TeacherSpecError):
        Z.snapshot_dir(Z.TeacherSpec(repo="evil/repo"), hub_cache=make_hub(tmp_path))


def _run(code):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=50)


def test_import_has_no_heavy_deps_and_no_side_effects():
    code = (
        "import sys, os, torch\n"
        "before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,\n"
        "          torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)\n"
        "import buc.teacher.zimage as z\n"
        "for m in ('diffusers', 'transformers', 'huggingface_hub', 'accelerate', 'safetensors'):\n"
        "    assert m not in sys.modules, m\n"
        "assert not torch.cuda.is_initialized()\n"
        "after = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,\n"
        "         torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)\n"
        "assert before == after, (before, after)\n"
        "assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ\n"
        "print('OK')\n"
    )
    p = _run(code)
    assert p.returncode == 0 and "OK" in p.stdout, p.stdout + p.stderr


def test_default_loaders_raise_dependency_error_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", None)      # None => import 시 ImportError
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(Z.TeacherDependencyError):
        Z.default_loaders()
    with pytest.raises(ImportError):
        Z.default_loaders()
