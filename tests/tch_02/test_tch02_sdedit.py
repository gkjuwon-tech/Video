"""TCH-02 SDEdit·denoise·generate: 시작 상태, 경로(독립 오라클), 시드 의미, 결정론, 오류."""
import pytest
import torch

from buc.core import det
from buc.teacher import sample as S
from buc.teacher import zimage
from tch02_fakes import C, H, W, make_caps, make_teacher, rand_latent, targets

P = S.DEFAULT_TABLE.points


def _sampler(tmp_path, **kw):
    fk = {k: kw.pop(k) for k in list(kw) if k in ("mode", "a")}
    t = make_teacher(tmp_path, **fk)
    return S.TeacherSampler(t, **kw), t.transformer


def eps_of(seed, level=0, k=0):
    return zimage.make_noise(seed, (1, C, H, W), level=level, k=k)[0]


def oracle_path(x0, eps, caps, a, pairs):
    """float64 독립 오라클(아핀 디노이저 x̂0 = a z + x*)."""
    s0 = pairs[0][0]
    z = (1 - s0) * x0.double() + s0 * eps.double()
    z_start = z.clone()
    tg = targets(caps).double()
    x0hat = None
    for s, sn in pairs:
        x0hat = a * z + tg
        z = (sn / s) * z + (1 - sn / s) * x0hat
    return z_start, x0hat, z


def test_sdedit_snaps_requests_and_builds_noisy_start(tmp_path):
    smp, tr = _sampler(tmp_path)
    x0, caps = rand_latent(3, seed=1), make_caps(3)
    seeds = [11, 22, 33]
    r = smp.sdedit(x0, caps, sigma=0.25, seeds=seeds, steps=1)
    assert r.requested_sigma == 0.25 and r.start_index == 7 and r.sigmas == pytest.approx((0.3,), abs=1e-12)
    assert r.steps == 1 and r.calls == 1 and r.table_key == S.DEFAULT_TABLE.key
    assert r.sigma_end == 0.0
    want = torch.stack([0.7 * x0[i] + 0.3 * eps_of(s) for i, s in enumerate(seeds)])
    torch.testing.assert_close(r.z_start, want, atol=2e-6, rtol=1e-6)
    assert torch.equal(r.z, r.x0_hat)                          # 끝(σ=0)까지 가면 z == x̂0 비트 동일
    torch.testing.assert_close(r.x0_hat, targets(caps), atol=3e-5, rtol=0)


def test_default_steps_is_two_and_path_matches_oracle(tmp_path):
    smp, tr = _sampler(tmp_path, mode="affine", a=0.5)
    x0, caps = rand_latent(2, seed=2), make_caps(2, seed=2)
    seeds = [5, 6]
    r = smp.sdedit(x0, caps, sigma=0.64, seeds=seeds)           # 0.64 -> idx 5 (9/14)
    assert r.steps == 2 and r.calls == 2 and len(tr.calls) == 2
    assert r.start_index == 5
    assert r.sigmas == pytest.approx((P[5], P[6]), abs=1e-12) and r.sigma_end == pytest.approx(P[7], abs=1e-12)
    eps = torch.stack([eps_of(s) for s in seeds])
    zs, xh, zf = oracle_path(x0, eps, caps, 0.5, [(P[5], P[6]), (P[6], P[7])])
    torch.testing.assert_close(r.z_start.double(), zs, atol=2e-6, rtol=0)
    torch.testing.assert_close(r.x0_hat.double(), xh, atol=3e-5, rtol=0)
    torch.testing.assert_close(r.z.double(), zf, atol=3e-5, rtol=0)
    # 호출 σ는 t=1-σ 로 정확히 이 순서
    assert [float(c.t[0]) for c in tr.calls] == pytest.approx([1 - P[5], 1 - P[6]], abs=1e-6)


def test_affine_path_reaching_zero(tmp_path):
    smp, _ = _sampler(tmp_path, mode="affine", a=0.3)
    x0, caps, seeds = rand_latent(2, seed=3), make_caps(2, seed=3), [9, 10]
    r = smp.sdedit(x0, caps, sigma=0.5, seeds=seeds, steps=2)   # idx6 -> idx7 -> 0
    eps = torch.stack([eps_of(s) for s in seeds])
    _, xh, zf = oracle_path(x0, eps, caps, 0.3, [(P[6], P[7]), (P[7], 0.0)])
    assert r.sigma_end == 0.0
    torch.testing.assert_close(r.z.double(), zf, atol=3e-5, rtol=0)
    assert torch.equal(r.z, r.x0_hat)
    torch.testing.assert_close(r.x0_hat.double(), xh, atol=3e-5, rtol=0)


@pytest.mark.parametrize("sigma,steps", [(0.2, 1), (0.5, 2), (0.64, 3), (0.9, 2), (1.0, 4)])
def test_perfect_denoiser_recovers_target_for_any_start_and_steps(tmp_path, sigma, steps):
    smp, _ = _sampler(tmp_path)
    x0, caps = rand_latent(2, seed=4), make_caps(2, seed=4)
    r = smp.sdedit(x0, caps, sigma=sigma, seeds=7, steps=steps)
    torch.testing.assert_close(r.x0_hat, targets(caps), atol=5e-5, rtol=0)
    assert r.calls == steps


def test_generate_full_schedule_from_pure_noise(tmp_path):
    smp, tr = _sampler(tmp_path)
    caps = make_caps(2, seed=5)
    seeds = [101, 102]
    r = smp.generate(caps, seeds=seeds, height=8 * H, width=8 * W)
    assert r.steps == 8 and r.calls == 8 and len(tr.calls) == 8 and r.start_index == 0
    assert r.sigmas == pytest.approx(P, abs=1e-12) and r.sigma_end == 0.0
    assert r.z.shape == (2, C, H, W) and r.z.dtype == torch.float32
    assert torch.equal(r.z_start, torch.stack([eps_of(s) for s in seeds]))     # σ=1 이면 z_start == ε 비트 동일
    assert torch.equal(r.z, r.x0_hat)
    torch.testing.assert_close(r.x0_hat, targets(caps), atol=5e-5, rtol=0)
    part = smp.generate(caps, seeds=seeds, height=8 * H, width=8 * W, steps=3)
    assert part.steps == 3 and part.sigma_end == pytest.approx(P[3], abs=1e-12)


def test_sdedit_at_sigma_one_equals_generate_bitwise(tmp_path):
    smp, _ = _sampler(tmp_path, mode="affine", a=0.4)
    caps = make_caps(2, seed=6)
    g = smp.generate(caps, seeds=[1, 2], height=8 * H, width=8 * W, steps=5)
    d = smp.sdedit(rand_latent(2, seed=77) * 100, caps, sigma=1.0, seeds=[1, 2], steps=5)   # x0 는 무관해야 한다
    assert torch.equal(g.z_start, d.z_start) and torch.equal(g.z, d.z) and torch.equal(g.x0_hat, d.x0_hat)


def test_denoise_continues_from_given_state_bitwise(tmp_path):
    smp, _ = _sampler(tmp_path, mode="affine", a=0.5)
    x0, caps = rand_latent(2, seed=8), make_caps(2, seed=8)
    r = smp.sdedit(x0, caps, sigma=0.64, seeds=[3, 4], steps=2)
    d = smp.denoise(r.z_start, caps, start_index=r.start_index, steps=2)
    assert torch.equal(d.z, r.z) and torch.equal(d.x0_hat, r.x0_hat) and torch.equal(d.z_start, r.z_start)
    assert d.requested_sigma == pytest.approx(P[5], abs=1e-12)
    # 쪼개서 이어 가도 같다: 1 step 후 상태에서 1 step 더
    a = smp.denoise(r.z_start, caps, start_index=5, steps=1)
    b = smp.denoise(a.z, caps, start_index=6, steps=1)
    torch.testing.assert_close(b.z, r.z, atol=1e-6, rtol=0)
    assert a.sigma_end == pytest.approx(P[6], abs=1e-12)


def test_seed_semantics(tmp_path):
    smp, _ = _sampler(tmp_path, mode="affine", a=0.5)
    x0, caps = rand_latent(3, seed=9), make_caps(3, seed=9)
    kw = dict(sigma=0.6, steps=1)
    # int 시드는 뷰마다 derive_seed(seed, i)
    a = smp.sdedit(x0, caps, seeds=1234, **kw)
    b = smp.sdedit(x0, caps, seeds=[det.derive_seed(1234, i) for i in range(3)], **kw)
    assert torch.equal(a.z_start, b.z_start) and torch.equal(a.z, b.z)
    # 같은 시드 -> 비트 동일, 다른 시드 -> 다름
    c = smp.sdedit(x0, caps, seeds=1234, **kw)
    assert torch.equal(a.z, c.z) and torch.equal(a.x0_hat, c.x0_hat)
    d = smp.sdedit(x0, caps, seeds=1235, **kw)
    assert not torch.equal(a.z_start, d.z_start)
    # 시드 목록 길이 불일치, 범위 밖
    for bad in ([1, 2], [1, 2, 3, 4], -1, 2 ** 64):
        with pytest.raises((ValueError, OverflowError)):
            smp.sdedit(x0, caps, seeds=bad, **kw)
    # level, k 는 노이즈 선택자
    e = smp.sdedit(x0, caps, seeds=[7, 8, 9], level=3, k=5, **kw)
    want = torch.stack([(1 - P[5]) * x0[i] + P[5] * eps_of(s, level=3, k=5) for i, s in enumerate([7, 8, 9])])  # 0.6 -> P[5]
    torch.testing.assert_close(e.z_start, want, atol=2e-6, rtol=1e-6)
    f = smp.sdedit(x0, caps, seeds=[7, 8, 9], level=0, k=0, **kw)
    assert not torch.equal(e.z_start, f.z_start)


def test_validation_errors_come_before_transformer_calls(tmp_path):
    smp, tr = _sampler(tmp_path)
    x0, caps = rand_latent(2, seed=10), make_caps(2)
    bad_calls = [
        lambda: smp.sdedit(x0, caps, sigma=0.0, seeds=1),
        lambda: smp.sdedit(x0, caps, sigma=1.5, seeds=1),
        lambda: smp.sdedit(x0, caps, sigma=float("nan"), seeds=1),
        lambda: smp.sdedit(x0, caps, sigma=0.3, seeds=1, steps=0),
        lambda: smp.sdedit(x0, caps, sigma=0.3, seeds=1, steps=2),            # idx7 에서 2 step 은 표를 넘는다
        lambda: smp.sdedit(x0, caps, sigma=0.9, seeds=1, steps=7),            # idx2 + 7 > 8
        lambda: smp.sdedit(x0, make_caps(3), sigma=0.5, seeds=1),
        lambda: smp.denoise(x0, caps, start_index=8, steps=1),
        lambda: smp.denoise(x0, caps, start_index=-1, steps=1),
        lambda: smp.denoise(x0, caps, start_index=2, steps=0),
        lambda: smp.generate(caps, seeds=1, height=8 * H + 1, width=8 * W),
        lambda: smp.generate(caps, seeds=1, height=8 * H, width=0),
        lambda: smp.generate(caps, seeds=1, height=8 * H, width=8 * W, steps=9),
        lambda: smp.generate(caps, seeds=[1], height=8 * H, width=8 * W),     # cond 는 2, seeds 는 1
    ]
    for f in bad_calls:
        with pytest.raises((ValueError, TypeError, OverflowError)):
            f()
    assert tr.calls == []


def test_results_are_detached_and_do_not_alias_inputs(tmp_path):
    smp, _ = _sampler(tmp_path)
    x0 = rand_latent(1, seed=11).requires_grad_(True)
    keep = x0.detach().clone()
    r = smp.sdedit(x0, make_caps(1), sigma=0.5, seeds=1, steps=1)
    for t in (r.x0_hat, r.z, r.z_start):
        assert not t.requires_grad and t.dtype == torch.float32
    assert torch.equal(x0.detach(), keep)
    r.z.zero_()                                                  # 결과를 바꿔도 입력 불변
    assert torch.equal(x0.detach(), keep)


def test_result_is_frozen_dataclass(tmp_path):
    smp, _ = _sampler(tmp_path)
    r = smp.sdedit(rand_latent(1), make_caps(1), sigma=0.5, seeds=1, steps=1)
    with pytest.raises(Exception):
        r.steps = 5
    assert isinstance(r.sigmas, tuple) and all(isinstance(s, float) for s in r.sigmas)
    assert isinstance(r.sigma_end, float) and isinstance(r.start_index, int) and isinstance(r.table_key, str)
