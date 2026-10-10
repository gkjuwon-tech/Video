"""TCH-06 Teacher-DCA · DR (진짜 load_teacher + 해석적 오라클을 가진 가짜 변환기, CPU)."""
import dataclasses
import inspect
import sys

import pytest
import torch

from buc.core import det
from buc.teacher import dca as D
from buc.teacher import sample as S
from buc.teacher.zimage import TeacherError, TeacherNumericError
from tch06_fakes import H, W, make_caps, make_teacher, rand_latent, target_of, targets

B = 3
SEEDS = [11, 22, 33]
SIG = (0.75, 0.5)


def _mk(tmp_path, dtype=torch.float32, table=None, max_views=None, **fake):
    t = make_teacher(tmp_path, dtype=dtype, **fake)
    return S.TeacherSampler(t, table=table, max_views_per_call=max_views), t


def _eps(teacher, seeds, k, level=0):
    return torch.stack([teacher.make_noise(s, 1, 8 * H, 8 * W, level=level, k=k)[0] for s in seeds])


def _expected_err(mode, x0, caps, seeds, teacher, sigmas, a=0.5, level=0):
    """v-공간 오차 [B,S] 해석값: mean((x0_hat - x0)^2)/σ^2, x0_hat = x*(c) (+ a z)."""
    cols = []
    for j, s in enumerate(sigmas):
        z = (1 - s) * x0.double() + s * _eps(teacher, seeds, j, level).double()
        tgt = targets(caps).double()
        x0h = tgt if mode == "perfect" else a * z + tgt
        cols.append(((x0h - x0.double()) ** 2).mean(dim=(1, 2, 3)) / s ** 2)
    return torch.stack(cols, 1)


def _close(got, want, rtol=1e-4, atol=1e-6):
    torch.testing.assert_close(got.double(), want.double(), rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------- 형식·서명

def test_signatures_are_pinned():
    P = inspect.Parameter
    for fn, pos in ((D.teacher_dca, ["sampler", "x0", "cond_a", "cond_b"]), (D.teacher_dr, ["sampler", "x0", "cond"]),
                    (D.delta_photo, ["sampler", "frame", "ref", "c_photo", "c_style"])):
        ps = inspect.signature(fn).parameters
        assert list(ps)[:len(pos)] == pos
        assert ps["seeds"].kind is P.KEYWORD_ONLY and ps["seeds"].default is P.empty
        assert ps["level"].kind is P.KEYWORD_ONLY and ps["level"].default == 0
    assert inspect.signature(D.teacher_dca).parameters["sigmas"].default == D.DCA_SIGMAS
    assert inspect.signature(D.teacher_dr).parameters["sigmas"].default == D.DR_SIGMAS
    assert inspect.signature(D.dr_verdict).parameters["max_rel"].default == D.DR_REL_MAX
    assert inspect.signature(D.dr_rel).parameters["floor"].default == D.DR_FLOOR


def test_import_has_no_heavy_side_effects():
    assert "diffusers" not in sys.modules and "transformers" not in sys.modules


def test_result_types(tmp_path):
    smp, _ = _mk(tmp_path)
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3, 0.7)
    r = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
    assert dataclasses.is_dataclass(r)
    assert r.dca.dtype == torch.float32 and tuple(r.dca.shape) == (B,)
    assert r.err_a.dtype == r.err_b.dtype == torch.float32 and tuple(r.err_a.shape) == (B, 2) == tuple(r.err_b.shape)
    assert r.sigmas == SIG and isinstance(r.sigmas, tuple) and r.calls == 4 and r.table_key == smp.table.key
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.calls = 5
    d = D.teacher_dr(smp, x0, ca, seeds=SEEDS)
    assert dataclasses.is_dataclass(d) and d.dr.dtype == torch.float32 and tuple(d.dr.shape) == (B,)
    assert tuple(d.per_sigma.shape) == (B, 1) and d.sigmas == (0.5,) and d.calls == 1 and d.table_key == smp.table.key


# ----------------------------------------------------------------------------- 해석적 오라클

def test_perfect_denoiser_matches_closed_form(tmp_path):
    smp, t = _mk(tmp_path, mode="perfect")
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3, 0.7)
    r = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
    ea = _expected_err("perfect", x0, ca, SEEDS, t, SIG)
    eb = _expected_err("perfect", x0, cb, SEEDS, t, SIG)
    _close(r.err_a, ea)
    _close(r.err_b, eb)
    _close(r.dca, (eb - ea).mean(1))
    # 완전 디노이저의 잔차는 노이즈에 의존하지 않으므로 시드를 바꿔도 같다
    r2 = D.teacher_dca(smp, x0, ca, cb, seeds=[5, 6, 7])
    _close(r2.dca, r.dca, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("a", [0.5, -0.25])
def test_affine_denoiser_matches_closed_form_and_depends_on_seed(tmp_path, a):
    smp, t = _mk(tmp_path, mode="affine", a=a)
    x0, ca, cb = rand_latent(B, 4), make_caps(B, 5), make_caps(B, 6, -0.4)
    r = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
    ea = _expected_err("affine", x0, ca, SEEDS, t, SIG, a)
    eb = _expected_err("affine", x0, cb, SEEDS, t, SIG, a)
    _close(r.err_a, ea)
    _close(r.err_b, eb)
    _close(r.dca, (eb - ea).mean(1))
    r2 = D.teacher_dca(smp, x0, ca, cb, seeds=[5, 6, 7])
    assert not torch.allclose(r2.err_a, r.err_a, rtol=1e-3)            # 시드가 바뀌면 z가 바뀐다


def test_sign_convention_positive_means_closer_to_a(tmp_path):
    smp, _ = _mk(tmp_path, mode="perfect")
    ca, cb = make_caps(B, 2), make_caps(B, 3, 0.7)
    x_a = targets(ca)                                   # x 가 정확히 조건 a 의 데이터
    r = D.teacher_dca(smp, x_a, ca, cb, seeds=SEEDS)
    assert float(r.err_a.abs().max()) < 1e-6 and bool((r.err_b > 1e-3).all())
    assert bool((r.dca > 0).all())
    r_sw = D.teacher_dca(smp, x_a, cb, ca, seeds=SEEDS)
    assert bool((r_sw.dca < 0).all())


def test_antisymmetry_and_zero(tmp_path):
    smp, _ = _mk(tmp_path, mode="affine", a=0.5)
    x0, ca, cb = rand_latent(B, 7), make_caps(B, 8), make_caps(B, 9, 0.5)
    ab = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
    ba = D.teacher_dca(smp, x0, cb, ca, seeds=SEEDS)
    _close(ab.dca, -ba.dca, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(ab.err_a, ba.err_b, rtol=0, atol=0)      # 공통 z: 같은 조건이면 같은 오차
    aa = D.teacher_dca(smp, x0, ca, ca, seeds=SEEDS)
    assert torch.equal(aa.dca, torch.zeros(B)) and torch.equal(aa.err_a, aa.err_b)


def test_dr_matches_closed_form_and_dca_consistency(tmp_path):
    smp, t = _mk(tmp_path, mode="affine", a=0.5)
    x0, ca, cb = rand_latent(B, 4), make_caps(B, 5), make_caps(B, 6, 0.3)
    d = D.teacher_dr(smp, x0, ca, seeds=SEEDS, sigmas=SIG)
    sig = torch.tensor(SIG, dtype=torch.float64)
    want = _expected_err("affine", x0, ca, SEEDS, t, SIG) * sig ** 2        # 잔차 r = e·σ²
    _close(d.per_sigma, want)
    _close(d.dr, want.mean(1))
    assert d.sigmas == SIG and d.calls == 2
    r = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS, sigmas=SIG)
    _close(r.err_a * sig.float() ** 2, d.per_sigma)                        # DCA 의 오차 = DR/σ² (같은 시드·σ)
    # 기본 σ (0.5,) 의 DR: 잔차가 σ 위치 k=0 노이즈로 계산된다
    d1 = D.teacher_dr(smp, x0, ca, seeds=SEEDS)
    _close(d1.per_sigma[:, 0], _expected_err("affine", x0, ca, SEEDS, t, (0.5,))[:, 0] * 0.25)


def test_dr_of_a_good_image_is_small_and_separates_real_from_fake(tmp_path):
    """합성 E0 (e): 교사 데이터에 가까운 이미지 vs 먼 이미지. DR 로 완전 분리되고 합격선이 동작한다."""
    smp, _ = _mk(tmp_path, mode="perfect")
    cap = make_caps(1, 0)
    base = targets(cap)                                              # [1,C,h,w]
    def make(n, scale, seed):
        return base + scale * rand_latent(n, seed)
    real, cand, fake = make(8, 0.05, 1), make(8, 0.05, 2), make(8, 0.5, 3)
    dr = lambda x, s: D.teacher_dr(smp, x, cap, seeds=list(range(s, s + 8))).dr          # noqa: E731
    d_real, d_cand, d_fake = dr(real, 100), dr(cand, 200), dr(fake, 300)
    assert D.dr_auc(d_real, d_fake) == 1.0
    ref = d_real.mean()
    assert D.dr_verdict(d_cand, ref).accept.all(), D.dr_rel(d_cand, ref)          # 같은 분포 -> 합격
    assert not D.dr_verdict(d_fake, ref).accept.any()                              # 먼 이미지 -> 불합격


# ----------------------------------------------------------------------------- 변환기 호출 규약

def test_call_order_common_noise_and_t_convention(tmp_path):
    smp, t = _mk(tmp_path, mode="affine", a=0.5)
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3, 0.7)
    sigmas = (0.5, 0.75)                                           # 일부러 내림차순이 아닌 순서
    r = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS, sigmas=sigmas)
    calls = t.transformer.calls
    assert len(calls) == 4 == r.calls and r.sigmas == sigmas
    assert [round(float(c.t[0]), 6) for c in calls] == [0.5, 0.5, 0.25, 0.25]      # σ 우선, t = 1-σ
    assert all(c.n == B and bool((c.t == c.t[0]).all()) for c in calls)
    for c, cond in zip(calls, (ca, cb, ca, cb)):                                    # 같은 σ 안에서 a -> b
        for i in range(B):
            assert torch.equal(c.caps[i], cond[i])
    assert torch.equal(calls[0].z, calls[1].z) and torch.equal(calls[2].z, calls[3].z)    # 공통 난수
    assert not torch.allclose(calls[0].z, calls[2].z)
    for j, s in enumerate(sigmas):                                                 # z = (1-σ)x0 + σ ε_j
        want = ((1 - s) * x0 + s * _eps(t, SEEDS, j)).unsqueeze(2)
        torch.testing.assert_close(calls[2 * j].z, want, rtol=1e-6, atol=1e-6)
    # 열 순서는 주어진 σ 순서 그대로, 열 j 는 ε_j (σ 의 위치가 노이즈를 정한다)
    _close(r.err_a, _expected_err("affine", x0, ca, SEEDS, t, sigmas))


def test_dr_call_pattern(tmp_path):
    smp, t = _mk(tmp_path, mode="affine")
    x0, ca = rand_latent(B, 1), make_caps(B, 2)
    D.teacher_dr(smp, x0, ca, seeds=SEEDS, sigmas=(0.75, 0.5, 0.3))
    calls = t.transformer.calls
    assert [round(float(c.t[0]), 6) for c in calls] == [0.25, 0.5, 0.7] and all(c.n == B for c in calls)


def test_str_and_tensor_conditions_and_broadcast(tmp_path):
    smp, _ = _mk(tmp_path, mode="affine")
    x0 = rand_latent(B, 1)
    r1 = D.teacher_dca(smp, x0, ["one"], ["two", "three", "four"], seeds=SEEDS)
    r2 = D.teacher_dca(smp, x0, ["one"] * B, ["two", "three", "four"], seeds=SEEDS)
    assert torch.equal(r1.dca, r2.dca) and torch.isfinite(r1.dca).all()
    r3 = D.teacher_dca(smp, x0, ["one"], make_caps(B, 2), seeds=SEEDS)          # a 는 문자열, b 는 텐서
    assert torch.isfinite(r3.dca).all()
    r4 = D.teacher_dca(smp, x0, make_caps(1, 2), make_caps(1, 3), seeds=SEEDS)  # 둘 다 길이 1 -> 브로드캐스트
    assert tuple(r4.dca.shape) == (B,)


def test_int_seed_expands_like_tch02(tmp_path):
    smp, _ = _mk(tmp_path, mode="affine")
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3, 0.7)
    r1 = D.teacher_dca(smp, x0, ca, cb, seeds=1234)
    r2 = D.teacher_dca(smp, x0, ca, cb, seeds=[det.derive_seed(1234, i) for i in range(B)])
    assert torch.equal(r1.dca, r2.dca)


# ----------------------------------------------------------------------------- 뷰 독립·청크·접두·결정성

def test_view_independence_order_and_chunks(tmp_path):
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3, 0.7)
    smp, _ = _mk(tmp_path, mode="affine", a=0.5)
    full = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
    for i in range(B):                                                         # 한 장씩
        one = D.teacher_dca(smp, x0[i:i + 1], ca[i:i + 1], cb[i:i + 1], seeds=SEEDS[i:i + 1])
        torch.testing.assert_close(one.dca[0], full.dca[i], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(one.err_a[0], full.err_a[i], rtol=1e-6, atol=1e-7)
    perm = [2, 0, 1]                                                            # 순서 섞기 -> 결과도 같은 순서로 섞인다
    pr = D.teacher_dca(smp, x0[perm], [ca[i] for i in perm], [cb[i] for i in perm], seeds=[SEEDS[i] for i in perm])
    torch.testing.assert_close(pr.dca, full.dca[perm], rtol=1e-6, atol=1e-7)
    for mv, n_calls in ((1, 12), (2, 8), (5, 4)):                               # 청크: 결과 동일, 호출 수 = 4 × 청크 수
        smp_c, t_c = _mk(tmp_path / f"c{mv}", mode="affine", a=0.5, max_views=mv)
        rc = D.teacher_dca(smp_c, x0, ca, cb, seeds=SEEDS)
        torch.testing.assert_close(rc.dca, full.dca, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(rc.err_b, full.err_b, rtol=1e-6, atol=1e-7)
        assert rc.calls == n_calls == len(t_c.transformer.calls)
    smp_c, t_c = _mk(tmp_path / "dr", mode="affine", max_views=2)
    assert D.teacher_dr(smp_c, x0, ca, seeds=SEEDS, sigmas=SIG).calls == 4 == len(t_c.transformer.calls)


def test_sigma_prefix_determinism_seed_and_level(tmp_path):
    smp, _ = _mk(tmp_path, mode="affine", a=0.5)
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3, 0.7)
    both = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
    first = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS, sigmas=(0.75,))
    torch.testing.assert_close(first.err_a, both.err_a[:, :1], rtol=1e-6, atol=1e-8)       # 접두 불변
    torch.testing.assert_close(first.err_b, both.err_b[:, :1], rtol=1e-6, atol=1e-8)
    again = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
    assert torch.equal(again.dca, both.dca) and torch.equal(again.err_a, both.err_a)        # 비트 동일
    other = D.teacher_dca(smp, x0, ca, cb, seeds=[SEEDS[0] + 1] + SEEDS[1:])
    assert not torch.equal(other.dca[0], both.dca[0]) and torch.equal(other.dca[1:], both.dca[1:])
    lv = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS, level=2)
    assert not torch.equal(lv.err_a, both.err_a)
    assert torch.equal(D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS, level=0).dca, both.dca)
    d0 = D.teacher_dr(smp, x0, ca, seeds=SEEDS)
    assert not torch.equal(D.teacher_dr(smp, x0, ca, seeds=SEEDS, level=2).dr, d0.dr)
    assert torch.equal(D.teacher_dr(smp, x0, ca, seeds=SEEDS).dr, d0.dr)


# ----------------------------------------------------------------------------- 표와 σ

def test_custom_table_is_respected(tmp_path):
    tab = S.SigmaTable((1.0, 0.8, 0.6), name="t3")
    smp, t = _mk(tmp_path, table=tab, mode="perfect")
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3, 0.7)
    r = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS, sigmas=(0.8, 0.6))
    assert r.table_key == tab.key != S.DEFAULT_TABLE.key and r.sigmas == (0.8, 0.6)
    _close(r.err_a, _expected_err("perfect", x0, ca, SEEDS, t, (0.8, 0.6)))
    assert [round(float(c.t[0]), 6) for c in t.transformer.calls] == [0.2, 0.2, 0.4, 0.4]
    n = len(t.transformer.calls)
    with pytest.raises(S.SigmaError):                                           # 기본 σ (0.75) 는 이 표에 없다
        D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
    assert len(t.transformer.calls) == n


# ----------------------------------------------------------------------------- Δ_photo

def test_delta_photo(tmp_path):
    smp, t = _mk(tmp_path, mode="affine", a=0.5)
    ca, cb = make_caps(B, 2), make_caps(B, 3, 0.9)                              # photo, style
    frame, ref = rand_latent(B, 1), rand_latent(B, 2)
    d = D.delta_photo(smp, frame, ref, ca, cb, seeds=SEEDS)
    assert len(t.transformer.calls) == 8 and d.dtype == torch.float32 and tuple(d.shape) == (B,)
    want = (D.teacher_dca(smp, frame, ca, cb, seeds=SEEDS).dca - D.teacher_dca(smp, ref, ca, cb, seeds=SEEDS).dca)
    torch.testing.assert_close(d, want, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(D.delta_photo(smp, ref, ref, ca, cb, seeds=SEEDS), torch.zeros(B), rtol=0, atol=1e-6)


def test_delta_photo_sign_and_validation(tmp_path):
    smp, t = _mk(tmp_path, mode="perfect")
    ph, st = make_caps(B, 2), make_caps(B, 3, 0.9)
    d = D.delta_photo(smp, targets(ph), targets(st), ph, st, seeds=SEEDS)          # 사진풍으로 끌린 프레임
    assert bool((d > 0).all())
    assert bool((D.delta_photo(smp, targets(st), targets(ph), ph, st, seeds=SEEDS) < 0).all())
    n = len(t.transformer.calls)
    with pytest.raises(ValueError):
        D.delta_photo(smp, rand_latent(B, 1), rand_latent(2, 1), ph, st, seeds=SEEDS)
    assert len(t.transformer.calls) == n


# ----------------------------------------------------------------------------- 오류 우선 검증

def _bad_inputs():
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3)
    nan = x0.clone()
    nan[0, 0, 0, 0] = float("nan")
    return [
        (S.SigmaError, dict(x0=x0, ca=ca, cb=cb, kw=dict(sigmas=(0.6,)))),
        (S.SigmaError, dict(x0=x0, ca=ca, cb=cb, kw=dict(sigmas=(0.75, 0.6)))),          # 두 번째만 표 밖
        (D.DCAError, dict(x0=x0, ca=ca, cb=cb, kw=dict(sigmas=()))),
        (D.DCAError, dict(x0=x0, ca=ca, cb=cb, kw=dict(sigmas=(0.5, 0.5)))),
        (D.DCAError, dict(x0=x0, ca=ca, cb=cb, kw=dict(sigmas=(0.5, 0.5 + 1e-12)))),     # 같은 표 점
        (ValueError, dict(x0=x0, ca=ca, cb=make_caps(2, 3), kw={})),                   # b 만 길이가 틀림(a 는 정상)
        (ValueError, dict(x0=x0, ca=make_caps(2, 3), cb=cb, kw={})),
        (TypeError, dict(x0=x0, ca=ca, cb=["x", torch.zeros(3, 8), "y"], kw={})),        # b 에 str/텐서 혼합
        (ValueError, dict(x0=x0, ca=ca, cb=cb, kw=dict(seeds=[1, 2]))),
        (TypeError, dict(x0=x0, ca=ca, cb=cb, kw=dict(seeds=[1.5, 2, 3]))),
        (ValueError, dict(x0=x0[:, :8], ca=ca, cb=cb, kw={})),
        (ValueError, dict(x0=x0[0], ca=ca, cb=cb, kw={})),
        (ValueError, dict(x0=nan, ca=ca, cb=cb, kw={})),
        (ValueError, dict(x0=rand_latent(B, 1, h=3, w=5), ca=ca, cb=cb, kw={})),         # 8h 가 16 의 배수 아님
    ]


@pytest.mark.parametrize("idx", range(14))
def test_dca_validation_fails_before_any_forward(tmp_path, idx):
    exc, case = _bad_inputs()[idx]
    smp, t = _mk(tmp_path)
    kw = dict(seeds=SEEDS)
    kw.update(case["kw"])
    with pytest.raises(exc):
        D.teacher_dca(smp, case["x0"], case["ca"], case["cb"], **kw)
    assert t.transformer.calls == [], "검증은 변환기 호출 전에 실패해야 한다"


@pytest.mark.parametrize("sigmas,exc", [((0.6,), S.SigmaError), ((), D.DCAError), ((0.5, 0.5), D.DCAError),
                                        ((0.5, 0.31), S.SigmaError)])
def test_dr_validation_fails_before_any_forward(tmp_path, sigmas, exc):
    smp, t = _mk(tmp_path)
    with pytest.raises(exc):
        D.teacher_dr(smp, rand_latent(B, 1), make_caps(B, 2), seeds=SEEDS, sigmas=sigmas)
    with pytest.raises(ValueError):
        D.teacher_dr(smp, rand_latent(B, 1), make_caps(2, 2), seeds=SEEDS)
    with pytest.raises(ValueError):
        D.teacher_dr(smp, rand_latent(B, 1), make_caps(B, 2), seeds=[1])
    assert t.transformer.calls == []


@pytest.mark.parametrize("fake,exc", [(dict(nan=True), TeacherNumericError), (dict(drop_last=True), TeacherError),
                                      (dict(bad_shape=True), TeacherError)])
def test_transformer_failures_propagate(tmp_path, fake, exc):
    smp, _ = _mk(tmp_path, **fake)
    with pytest.raises(exc):
        D.teacher_dca(smp, rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3), seeds=SEEDS)
    with pytest.raises(exc):
        D.teacher_dr(smp, rand_latent(B, 1), make_caps(B, 2), seeds=SEEDS)


# ----------------------------------------------------------------------------- grad · dtype

def test_no_grad_even_when_inputs_require_grad(tmp_path):
    smp, t = _mk(tmp_path, mode="affine")
    x0 = rand_latent(B, 1).requires_grad_(True)
    ca = [c.clone().requires_grad_(True) for c in make_caps(B, 2)]
    cb = make_caps(B, 3)
    with torch.enable_grad():
        r = D.teacher_dca(smp, x0, ca, cb, seeds=SEEDS)
        d = D.teacher_dr(smp, x0, ca, seeds=SEEDS)
        p = D.delta_photo(smp, x0, x0, ca, cb, seeds=SEEDS)
    for tensor in (r.dca, r.err_a, r.err_b, d.dr, d.per_sigma, p):
        assert not tensor.requires_grad
    assert all((not c.grad_enabled) and (not c.x_requires_grad) for c in t.transformer.calls)


@pytest.mark.parametrize("mode", ["perfect", "affine"])
def test_bf16_teacher_is_close_to_fp32(tmp_path, mode):
    x0, ca, cb = rand_latent(B, 1), make_caps(B, 2), make_caps(B, 3, 0.7)
    s32, _ = _mk(tmp_path / "f32", mode=mode)
    s16, t16 = _mk(tmp_path / "b16", dtype=torch.bfloat16, mode=mode)
    r32 = D.teacher_dca(s32, x0, ca, cb, seeds=SEEDS)
    r16 = D.teacher_dca(s16, x0, ca, cb, seeds=SEEDS)
    assert r16.dca.dtype == torch.float32 and torch.isfinite(r16.dca).all()
    assert t16.transformer.calls[0].z.dtype == torch.float32
    scale = float(r32.err_b.abs().max())
    # bf16 은 상대 2^-8: 오차 차는 (오차 크기)의 몇 % 안에서 일치하고, 부호(어느 조건이 더 가까운지)는 유지된다
    assert float((r16.dca - r32.dca).abs().max()) < 0.02 * scale
    assert torch.equal(torch.sign(r16.dca), torch.sign(r32.dca))
