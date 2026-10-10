"""load_klein: 적재 순서·라이선스·dtype·동결·지연 TE·참조 래퍼·예산·샘플러 호환."""
import json

import pytest
import torch
from tch08_fakes import KC, KleinRecorder, make_snapshot, make_z_teacher, target_of

from buc.teacher import ref_slot as R
from buc.teacher.sample import TeacherSampler
from buc.teacher.zimage import ZIMAGE_REPO, TeacherSpecError, TeacherWeightsMissing

KSPEC = R.KleinSpec(latent_channels=KC, text_dim=8, max_text_len=16, license_verified=True)


def _loaders(rec):
    return R.KleinLoaders(tokenizer=rec.tokenizer, text_encoder=rec.text_encoder, transformer=rec.transformer,
                          vae=rec.vae, encode_prompts=rec.encode_prompts, forward=rec.forward)


def _load(tmp_path, rec=None, spec=KSPEC, dtype=torch.float32, **kw):
    rec = rec or KleinRecorder()
    hub = kw.pop("hub", None)
    if hub is None:
        hub, _ = make_snapshot(tmp_path, spec.repo)
    t = R.load_klein(spec, device="cpu", dtype=dtype, hub_cache=hub, loaders=_loaders(rec),
                     te_cache_dir=tmp_path / "tec", deterministic=False, **kw)
    return t, rec


def test_license_flag_checked_before_filesystem_and_loaders(tmp_path):
    rec = KleinRecorder()
    with pytest.raises(R.RefLicenseError):
        R.load_klein(R.KleinSpec(latent_channels=KC, text_dim=8), device="cpu", dtype=torch.float32,
                     hub_cache=tmp_path / "nothing", loaders=_loaders(rec), deterministic=False)
    assert rec.log == []


def test_license_text_checked_before_loaders(tmp_path):
    rec = KleinRecorder()
    hub, _ = make_snapshot(tmp_path, KSPEC.repo, license_text="All rights reserved.\n")
    with pytest.raises(R.RefLicenseError):
        R.load_klein(KSPEC, device="cpu", dtype=torch.float32, hub_cache=hub, loaders=_loaders(rec),
                     deterministic=False)
    assert rec.log == []


def test_missing_snapshot_subdir(tmp_path):
    rec = KleinRecorder()
    hub, _ = make_snapshot(tmp_path, KSPEC.repo, subdirs=("transformer", "vae", "tokenizer"))
    with pytest.raises(TeacherWeightsMissing):
        R.load_klein(KSPEC, device="cpu", dtype=torch.float32, hub_cache=hub, loaders=_loaders(rec),
                     deterministic=False)
    assert rec.log == []


def test_fp16_policy(tmp_path):
    rec = KleinRecorder()
    hub, _ = make_snapshot(tmp_path, KSPEC.repo)
    with pytest.raises(TeacherSpecError):
        R.load_klein(KSPEC, device="cpu", dtype=torch.float16, hub_cache=hub, loaders=_loaders(rec),
                     deterministic=False)
    assert rec.log == []


def test_dtype_mismatch_not_silently_cast(tmp_path):
    rec = KleinRecorder(transformer_dtype=torch.float32)
    with pytest.raises(TeacherSpecError):
        _load(tmp_path, rec=rec, dtype=torch.bfloat16)


def test_vae_channel_mismatch(tmp_path):
    with pytest.raises(TeacherSpecError):
        _load(tmp_path, rec=KleinRecorder(vae_channels=KC + 1))


def test_load_frozen_lazy_te_and_info(tmp_path):
    t, rec = _load(tmp_path)
    assert rec.log == ["transformer", "vae"]                      # TE·토크나이저는 지연
    assert t.license == "Apache-2.0"
    for m in (t.raw_transformer, t.vae.vae):
        assert not m.training
        assert all(not p.requires_grad for p in m.parameters())
    assert t.refs == ()
    e1 = t.encode_text(["oil paint", "ink"])
    assert "text_encoder" in rec.log and "tokenizer" in rec.log
    n = rec.log.count("encode_prompts")
    e2 = t.encode_text(["oil paint"])
    assert rec.log.count("encode_prompts") == n                  # 캐시 히트
    assert torch.equal(e1[0], e2[0])
    info = t.info()
    json.dumps(info)
    assert info["license"] == "Apache-2.0" and info["repo"] == KSPEC.repo and info["n_refs"] == 0
    assert info["weights_bytes"] > 0 and info["te_unloaded"] is False
    t.release_text_encoder()
    assert t.info()["te_unloaded"] is True


def test_te_cache_namespace_differs_from_engine(tmp_path):
    t, _ = _load(tmp_path)
    z = make_z_teacher(tmp_path)
    assert t.text.key("same prompt") != z.text.key("same prompt")


def _img(h, w, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, h, w, generator=g) * 2 - 1


def test_encode_refs_budget_before_vae(tmp_path):
    t, rec = _load(tmp_path)
    calls = rec.vae_obj.encode_calls
    with pytest.raises(R.RefBudgetError):
        t.encode_refs([_img(16, 16, i) for i in range(5)])
    with pytest.raises(R.RefBudgetError):
        t.encode_refs([_img(24, 16, 0)])
    assert rec.vae_obj.encode_calls == calls
    lat = t.encode_refs([_img(32, 48, 1), _img(16, 32, 2)])
    assert [tuple(x.shape) for x in lat] == [(KC, 4, 6), (KC, 2, 4)]
    assert all(x.dtype == torch.float32 for x in lat)
    again = t.encode_refs([_img(32, 48, 1)])
    assert torch.equal(again[0], lat[0])                          # 결정론(.mode())


def test_set_refs_validation(tmp_path):
    t, _ = _load(tmp_path)
    ok = torch.randn(KC, 4, 4)
    with pytest.raises(R.RefBudgetError):
        t.set_refs([ok] * 5)
    with pytest.raises(ValueError):
        t.set_refs([torch.randn(KC + 1, 4, 4)])
    with pytest.raises(ValueError):
        t.set_refs([torch.full((KC, 4, 4), float("nan"))])
    with pytest.raises(ValueError):
        t.set_refs([torch.randn(1, KC, 4, 4)])
    t.set_refs([ok, ok * 2])
    assert len(t.refs) == 2 and all(r.dtype == t.dtype and not r.requires_grad for r in t.refs)
    ok.add_(100.0)                                                # 외부 변경이 슬롯을 바꾸지 않는다
    assert float(t.refs[0].abs().max()) < 50
    t.set_refs(None)
    assert t.refs == ()
    t.set_refs([ok])
    t.set_refs([])
    assert t.refs == ()


def test_bound_transformer_drives_tch02_sampler(tmp_path):
    t, rec = _load(tmp_path)
    s = TeacherSampler(t, table=R.KLEIN_TABLE)
    caps = [torch.randn(3, 8, generator=torch.Generator().manual_seed(9))]
    base = s.generate(caps, seeds=[7], height=32, width=48)
    assert base.calls == 4 and base.sigmas == pytest.approx((1.0, 0.9, 0.75, 0.5))
    assert all(r == [] for r in rec.forward_refs)
    want = target_of(caps[0], KC, 4, 6)
    assert torch.allclose(base.z[0], want, atol=1e-4)

    r1 = torch.full((KC, 2, 2), 0.25)
    r2 = torch.full((KC, 4, 4), -0.75)
    t.set_refs([r1, r2])
    rec.forward_refs.clear()
    out = s.generate(caps, seeds=[7], height=32, width=48)
    assert len(rec.forward_refs) == 4
    for refs in rec.forward_refs:
        assert len(refs) == 2 and torch.equal(refs[0].float(), r1) and torch.equal(refs[1].float(), r2)
    assert torch.allclose(out.z[0], want + (0.25 - 0.75), atol=1e-4)
    out2 = s.generate(caps, seeds=[7], height=32, width=48)
    assert torch.equal(out.z, out2.z)


def test_direct_transformer_call_convention(tmp_path):
    t, rec = _load(tmp_path)
    x = [torch.zeros(KC, 1, 2, 2)]
    out = t.transformer(x, torch.tensor([0.5]), [torch.zeros(3, 8)], return_dict=False)
    assert isinstance(out, tuple) and isinstance(out[0], list) and tuple(out[0][0].shape) == (KC, 1, 2, 2)


def test_make_noise_matches_tch01_definition(tmp_path):
    from buc.teacher.zimage import make_noise
    t, _ = _load(tmp_path)
    n = t.make_noise(11, 2, 32, 48, level=1, k=3)
    assert tuple(n.shape) == (2, KC, 4, 6)
    assert torch.equal(n, make_noise(11, (2, KC, 4, 6), level=1, k=3))


def test_to_device_returns_self(tmp_path):
    t, _ = _load(tmp_path)
    t.set_refs([torch.ones(KC, 2, 2)])
    assert t.to("cpu") is t
    assert str(t.device) == "cpu" and t.refs[0].device.type == "cpu"


def test_engine_repo_rejected_as_klein(tmp_path):
    with pytest.raises(TeacherSpecError):
        R.KleinSpec(repo=ZIMAGE_REPO, license_verified=True).validate()
