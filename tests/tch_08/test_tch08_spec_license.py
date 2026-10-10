"""KleinSpec 검증 순서, 라이선스 원문 검사, 참조 예산, 상수, import 부작용."""
import subprocess
import sys
from dataclasses import replace

import pytest
from tch08_fakes import APACHE, make_snapshot

from buc.teacher import ref_slot as R
from buc.teacher.zimage import TeacherError, TeacherSpecError


def test_exception_hierarchy():
    assert issubclass(R.RefSlotError, TeacherError)
    assert issubclass(R.RefLicenseError, R.RefSlotError) and issubclass(R.RefLicenseError, PermissionError)
    assert issubclass(R.RefBudgetError, R.RefSlotError) and issubclass(R.RefBudgetError, ValueError)


def test_constants():
    assert R.KLEIN_REPO in R.KLEIN_OFFICIAL_REPOS
    assert R.KLEIN_MAX_REFS == 4
    assert R.REF_TOKEN_BUDGET == 2304
    assert R.RING_STRIP_PX == 128 and tuple(R.RING_CANVAS_HW) == (1152, 1024)
    assert R.RINGR_K_BASE == 1 << 20
    assert "9b" in tuple(m.lower() for m in R.FORBIDDEN_REPO_MARKERS)
    assert R.KLEIN_TABLE.points == pytest.approx((1.0, 0.9, 0.75, 0.5), abs=1e-12)


def test_import_has_no_heavy_side_effects():
    code = ("import sys; import buc.teacher.ref_slot as r; "
            "bad=[m for m in ('diffusers','transformers','huggingface_hub') if m in sys.modules]; "
            "print('BAD' if bad else 'OK', bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("OK"), out.stdout


def test_spec_defaults_and_geometry():
    s = R.KleinSpec()
    assert s.repo == R.KLEIN_REPO and s.license_verified is False and s.unofficial_ok is False
    assert s.latent_channels == 32 and s.vae_scale == 8 and s.patch_size == 2
    v = replace(s, license_verified=True)
    assert v.validate() is v or v.validate() is None
    assert v.latent_shape(512, 768, 2) == (2, 32, 64, 96)
    with pytest.raises(ValueError):
        v.check_image_size(520, 512)
    with pytest.raises(ValueError):
        v.latent_shape(512, 512, 0)


def test_license_flag_required():
    with pytest.raises(R.RefLicenseError):
        R.KleinSpec().validate()


@pytest.mark.parametrize("repo", ["black-forest-labs/FLUX.2-klein-9B", "someone/flux.2-KLEIN-9b-int8"])
def test_forbidden_9b_even_with_unofficial_ok(repo):
    s = R.KleinSpec(repo=repo, license_verified=True, unofficial_ok=True)
    with pytest.raises(TeacherSpecError):
        s.validate()


def test_forbidden_marker_checked_before_license():
    with pytest.raises(TeacherSpecError) as ei:
        R.KleinSpec(repo="black-forest-labs/FLUX.2-klein-9B").validate()
    assert not isinstance(ei.value, R.RefLicenseError)


def test_unofficial_repo_rules():
    with pytest.raises(TeacherSpecError):
        R.KleinSpec(repo="random/klein-4b-quant", license_verified=True).validate()
    R.KleinSpec(repo="random/klein-4b-quant", license_verified=True, unofficial_ok=True).validate()


@pytest.mark.parametrize("field,val", [("latent_channels", 0), ("vae_scale", -8), ("text_dim", 0),
                                       ("revision", "../x"), ("revision", "a b")])
def test_bad_fields(field, val):
    s = replace(R.KleinSpec(license_verified=True), **{field: val})
    with pytest.raises(TeacherSpecError):
        s.validate()


# ----------------------------------------------------------------------------- 라이선스 원문

def test_license_file_apache(tmp_path):
    _, snap = make_snapshot(tmp_path, R.KLEIN_REPO)
    assert R.check_license(snap) == "Apache-2.0"


def test_license_readme_front_matter_only(tmp_path):
    _, snap = make_snapshot(tmp_path, R.KLEIN_REPO, license_text=None,
                            readme="---\nlicense: Apache-2.0\nlibrary_name: diffusers\n---\n# klein\n")
    assert R.check_license(snap) == "Apache-2.0"


def test_license_missing(tmp_path):
    _, snap = make_snapshot(tmp_path, R.KLEIN_REPO, license_text=None)
    with pytest.raises(R.RefLicenseError):
        R.check_license(snap)


def test_license_non_apache_file(tmp_path):
    _, snap = make_snapshot(tmp_path, R.KLEIN_REPO, license_text="FLUX Non-Commercial License v2.0\n")
    with pytest.raises(R.RefLicenseError):
        R.check_license(snap)


def test_license_conflicting_readme(tmp_path):
    _, snap = make_snapshot(tmp_path, R.KLEIN_REPO, license_text=APACHE,
                            readme="---\nlicense: other\nlicense_name: flux-non-commercial-license\n---\n")
    with pytest.raises(R.RefLicenseError):
        R.check_license(snap)


def test_license_md_variant(tmp_path):
    _, snap = make_snapshot(tmp_path, R.KLEIN_REPO, license_text=None)
    (snap / "LICENSE.md").write_text(APACHE)
    assert R.check_license(snap) == "Apache-2.0"


# ----------------------------------------------------------------------------- 참조 예산

def test_ref_tokens():
    assert R.ref_tokens(512, 512) == 1024
    assert R.ref_tokens(768, 768) == 2304
    assert R.ref_tokens(1024, 1152) == 64 * 72


@pytest.mark.parametrize("sizes,total", [([], 0), ([(512, 512)], 1024), ([(512, 512), (512, 512)], 2048),
                                         ([(768, 768)], 2304), ([(256, 256)] * 4, 1024),
                                         ([(512, 256), (256, 512), (256, 256)], 1280)])
def test_budget_ok(sizes, total):
    assert R.check_ref_budget(sizes) == total


@pytest.mark.parametrize("sizes", [[(512, 512)] * 3, [(768, 768), (16, 16)], [(16, 16)] * 5,
                                   [(500, 512)], [(512, 0)], [(-16, 16)], [(1024, 1024)]])
def test_budget_rejects(sizes):
    with pytest.raises(R.RefBudgetError):
        R.check_ref_budget(sizes)
