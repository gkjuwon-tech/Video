"""TCH-02 변환기 호출 규약·입력 검증·dtype·오류 우선 검증 (진짜 load_teacher + 가짜 변환기)."""
import inspect

import pytest
import torch

from buc.teacher import sample as S
from buc.teacher.zimage import TeacherError, TeacherNumericError, TeacherSpecError
from tch02_fakes import C, H, IMG_H, IMG_W, SPEC, W, make_caps, make_teacher, rand_latent, target_of, targets


def _sampler(tmp_path, dtype=torch.float32, **kw):
    fk = {k: kw.pop(k) for k in list(kw) if k in ("mode", "a", "nan", "drop_last", "bad_shape")}
    t = make_teacher(tmp_path, dtype=dtype, **fk)
    return S.TeacherSampler(t, **kw), t.transformer


def _kinds(fn):
    return {n: (p.kind.name, p.default) for n, p in inspect.signature(fn).parameters.items() if n != "self"}


def test_signatures_are_pinned():
    P = inspect.Parameter
    ini = _kinds(S.TeacherSampler.__init__)
    assert list(ini)[:1] == ["teacher"] and ini["table"] == ("KEYWORD_ONLY", None)
    assert ini["max_views_per_call"] == ("KEYWORD_ONLY", None)
    sd = _kinds(S.TeacherSampler.sdedit)
    assert list(sd)[:2] == ["x0", "cond"]
    for n in ("sigma", "seeds"):
        assert sd[n] == ("KEYWORD_ONLY", P.empty)
    assert sd["steps"] == ("KEYWORD_ONLY", 2) and sd["level"] == ("KEYWORD_ONLY", 0) and sd["k"] == ("KEYWORD_ONLY", 0)
    assert sd["x0_hook"] == ("KEYWORD_ONLY", None) and sd["observer"] == ("KEYWORD_ONLY", None)
    assert sd["grad"] == ("KEYWORD_ONLY", False)
    dn = _kinds(S.TeacherSampler.denoise)
    assert list(dn)[:2] == ["z", "cond"] and dn["start_index"] == ("KEYWORD_ONLY", P.empty)
    assert dn["steps"] == ("KEYWORD_ONLY", P.empty) and dn["grad"] == ("KEYWORD_ONLY", False)
    px = _kinds(S.TeacherSampler.predict_x0)
    assert list(px)[:3] == ["z", "sigma", "cond"] and px["grad"] == ("KEYWORD_ONLY", False)
    ge = _kinds(S.TeacherSampler.generate)
    assert list(ge)[:1] == ["cond"]
    for n in ("seeds", "height", "width"):
        assert ge[n] == ("KEYWORD_ONLY", P.empty)
    assert ge["steps"] == ("KEYWORD_ONLY", None)
    assert [f.name for f in S.StepInfo.__dataclass_fields__.values()] == \
        ["step", "sigma", "sigma_next", "view_lo", "view_hi"]
    assert {f for f in S.SampleResult.__dataclass_fields__} >= {
        "x0_hat", "z", "z_start", "sigmas", "sigma_end", "start_index", "requested_sigma", "steps", "calls",
        "table_key"}


def test_default_table_and_attrs(tmp_path):
    smp, _ = _sampler(tmp_path)
    assert smp.table is S.DEFAULT_TABLE or smp.table == S.DEFAULT_TABLE
    assert smp.teacher.spec is SPEC or smp.teacher.spec == SPEC


def test_transformer_call_protocol(tmp_path):
    smp, tr = _sampler(tmp_path)
    caps = make_caps(3)
    z = rand_latent(3, seed=1)
    sig = smp.table.points[5]
    x0 = smp.predict_x0(z, sig, caps)
    assert len(tr.calls) == 1
    c = tr.calls[0]
    assert c.n == 3 and c.x_shapes == [(C, 1, H, W)] * 3                 # [C,1,h,w], 뷰마다 하나
    assert c.x_dtype == torch.float32 and c.t_dtype == torch.float32
    torch.testing.assert_close(c.t, torch.full((3,), 1.0 - sig), atol=1e-7, rtol=0)   # t = 1 - σ
    assert c.return_dict is False                                          # return_dict=False 로 부른다
    assert c.cap_lens == [cp.shape[0] for cp in caps]                      # 가변 길이 그대로(패딩 없음)
    assert not c.grad_enabled                                              # 기본은 no_grad
    assert x0.dtype == torch.float32 and x0.shape == z.shape and not x0.requires_grad


def test_perfect_denoiser_returns_target_at_every_table_sigma(tmp_path):
    smp, tr = _sampler(tmp_path)
    caps = make_caps(2, seed=4)
    z = rand_latent(2, seed=2)
    for s in smp.table.points:
        x0 = smp.predict_x0(z, s, caps)
        torch.testing.assert_close(x0, targets(caps), atol=2e-5, rtol=0)
    assert len(tr.calls) == 8


def test_velocity_sign_and_sigma_use_with_affine_denoiser(tmp_path):
    # x̂0 = a z + x*  이어야 한다. 부호나 σ를 잘못 쓰면 여기서 갈린다.
    smp, _ = _sampler(tmp_path, mode="affine", a=0.5)
    caps = make_caps(2, seed=5)
    z = rand_latent(2, seed=3)
    for s in (smp.table.points[0], smp.table.points[4], smp.table.points[7]):
        x0 = smp.predict_x0(z, s, caps)
        torch.testing.assert_close(x0, 0.5 * z + targets(caps), atol=3e-5, rtol=0)


def test_predict_x0_is_strict_about_table_points(tmp_path):
    smp, tr = _sampler(tmp_path)
    z = rand_latent(1)
    for bad in (0.25, 0.31, 0.0, 1.5, float("nan")):
        with pytest.raises(S.SigmaError):
            smp.predict_x0(z, bad, make_caps(1))
    assert tr.calls == []


def test_string_prompts_use_teacher_text_path(tmp_path):
    smp, tr = _sampler(tmp_path)
    prompts = ["a mossy stone wall", "photo of a door", "x"]
    z = rand_latent(3, seed=6)
    sig = smp.table.points[6]
    a = smp.predict_x0(z, sig, prompts)
    caps = smp.teacher.encode_text(prompts)
    b = smp.predict_x0(z, sig, caps)
    assert torch.equal(a, b)
    torch.testing.assert_close(a, targets(caps), atol=2e-5, rtol=0)
    assert tr.calls[0].cap_lens == [min(len(p.encode()), SPEC.max_text_len) for p in prompts]


def test_single_cond_broadcasts_and_length_mismatch_fails_early(tmp_path):
    smp, tr = _sampler(tmp_path)
    cap = make_caps(1, seed=7)
    z = rand_latent(4, seed=7)
    sig = smp.table.points[3]
    one = smp.predict_x0(z, sig, cap)
    four = smp.predict_x0(z, sig, cap * 4)
    assert torch.equal(one, four)
    assert tr.calls[0].cap_lens == [cap[0].shape[0]] * 4
    n = len(tr.calls)
    with pytest.raises(ValueError):
        smp.predict_x0(z, sig, make_caps(3))
    with pytest.raises(ValueError):
        smp.predict_x0(z, sig, [])
    with pytest.raises(TypeError):
        smp.predict_x0(z, sig, ["abc", make_caps(1)[0], "d", "e"])
    assert len(tr.calls) == n


@pytest.mark.parametrize("make", [
    lambda: torch.zeros(0, C, H, W),                       # B=0
    lambda: torch.zeros(1, C + 1, H, W),                   # 채널 불일치
    lambda: torch.zeros(1, C, H + 1, W),                   # 홀수 잠재 -> 이미지가 16의 배수 아님
    lambda: torch.zeros(1, C, H, W + 1),
    lambda: torch.zeros(C, H, W),                          # 3D
    lambda: torch.full((1, C, H, W), float("nan")),
    lambda: torch.full((1, C, H, W), float("inf")),
    lambda: torch.zeros(1, C, H, W, dtype=torch.long),     # 부동소수 아님
])
def test_bad_latents_fail_before_any_transformer_call(tmp_path, make):
    smp, tr = _sampler(tmp_path)
    bad = make()
    cap = make_caps(1)
    with pytest.raises((ValueError, TypeError)):
        smp.predict_x0(bad, smp.table.points[5], cap)
    with pytest.raises((ValueError, TypeError)):
        smp.sdedit(bad, cap, sigma=0.5, seeds=1)
    with pytest.raises((ValueError, TypeError)):
        smp.denoise(bad, cap, start_index=5, steps=1)
    assert tr.calls == []


def test_non_finite_transformer_output_raises_numeric_error(tmp_path):
    smp, tr = _sampler(tmp_path, nan=True)
    with pytest.raises(TeacherNumericError):
        smp.predict_x0(rand_latent(1), smp.table.points[2], make_caps(1))
    with pytest.raises(TeacherNumericError):
        smp.sdedit(rand_latent(1), make_caps(1), sigma=0.5, seeds=1)


def test_wrong_output_count_or_shape_raises_teacher_error(tmp_path):
    for kw in ({"drop_last": True}, {"bad_shape": True}):
        smp, _ = _sampler(tmp_path / ("a" if "drop_last" in kw else "b"), **kw)
        with pytest.raises(TeacherError):
            smp.predict_x0(rand_latent(2), smp.table.points[2], make_caps(2))


def test_bf16_teacher_gets_bf16_inputs_and_returns_float32(tmp_path):
    smp, tr = _sampler(tmp_path, dtype=torch.bfloat16)
    caps = make_caps(2, seed=8)
    z = rand_latent(2, seed=8)
    x0 = smp.predict_x0(z, smp.table.points[6], caps)
    assert tr.calls[0].x_dtype == torch.bfloat16
    assert x0.dtype == torch.float32
    torch.testing.assert_close(x0, targets([c.float() for c in caps]), atol=6e-2, rtol=0)
    r = smp.sdedit(rand_latent(2, seed=9), caps, sigma=0.6, seeds=[1, 2], steps=2)
    assert r.z.dtype == r.x0_hat.dtype == r.z_start.dtype == torch.float32
    torch.testing.assert_close(r.x0_hat, targets([c.float() for c in caps]), atol=8e-2, rtol=0)


def test_max_views_per_call_validation(tmp_path):
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            _sampler(tmp_path / str(bad), max_views_per_call=bad)


def test_custom_table_is_used(tmp_path):
    tab = S.SigmaTable((1.0, 0.5, 0.25), name="t3")
    smp, tr = _sampler(tmp_path, table=tab)
    assert smp.table.key == tab.key
    r = smp.sdedit(rand_latent(1), make_caps(1), sigma=0.4, seeds=1, steps=2)     # 0.4 -> 0.5(idx1)
    assert r.start_index == 1 and r.sigmas == (0.5, 0.25) and r.sigma_end == 0.0 and r.table_key == tab.key
    with pytest.raises(S.SigmaError):
        smp.predict_x0(rand_latent(1), 0.3, make_caps(1))             # 기본 표의 점이어도 이 표엔 없다
