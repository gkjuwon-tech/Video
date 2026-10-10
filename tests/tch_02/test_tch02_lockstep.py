"""TCH-02 뷰 배치 lockstep: 뷰 독립·순서 불변·청크·observer·x0_hook·역전파."""
import pytest
import torch

from buc.teacher import sample as S
from tch02_fakes import make_caps, make_teacher, rand_latent, targets

P = S.DEFAULT_TABLE.points
B = 5


def _sampler(tmp_path, **kw):
    fk = {k: kw.pop(k) for k in list(kw) if k in ("mode", "a")}
    t = make_teacher(tmp_path, **fk)
    return S.TeacherSampler(t, **kw), t.transformer


def _run(smp, x0, caps, seeds, **kw):
    return smp.sdedit(x0, caps, sigma=0.6, seeds=seeds, steps=2, **kw)


def test_one_forward_per_step_with_all_views(tmp_path):
    smp, tr = _sampler(tmp_path)
    r = _run(smp, rand_latent(B, seed=1), make_caps(B), list(range(B)))
    assert r.calls == len(tr.calls) == 2
    assert [c.n for c in tr.calls] == [B, B]
    assert all(c.cap_lens == [3 + i for i in range(B)] for c in tr.calls)


def test_view_independence_batch_vs_single(tmp_path):
    smp, _ = _sampler(tmp_path, mode="affine", a=0.5)
    x0, caps, seeds = rand_latent(B, seed=2), make_caps(B, seed=2), [10, 20, 30, 40, 50]
    full = _run(smp, x0, caps, seeds)
    for i in range(B):
        one = _run(smp, x0[i:i + 1], caps[i:i + 1], [seeds[i]])
        assert torch.equal(one.z[0], full.z[i]), i
        assert torch.equal(one.x0_hat[0], full.x0_hat[i]), i
        assert torch.equal(one.z_start[0], full.z_start[i]), i


def test_int_seed_prefix_invariance(tmp_path):
    smp, _ = _sampler(tmp_path)
    x0, caps = rand_latent(B, seed=3), make_caps(B, seed=3)
    big = smp.sdedit(x0, caps, sigma=0.6, seeds=99, steps=1)
    small = smp.sdedit(x0[:2], caps[:2], sigma=0.6, seeds=99, steps=1)
    assert torch.equal(big.z_start[:2], small.z_start)


def test_view_order_purity(tmp_path):
    smp, _ = _sampler(tmp_path, mode="affine", a=0.5)
    x0, caps, seeds = rand_latent(B, seed=4), make_caps(B, seed=4), [1, 2, 3, 4, 5]
    base = _run(smp, x0, caps, seeds)
    perm = [3, 0, 4, 1, 2]
    p = _run(smp, x0[perm], [caps[i] for i in perm], [seeds[i] for i in perm])
    for j, i in enumerate(perm):
        assert torch.equal(p.z[j], base.z[i]) and torch.equal(p.x0_hat[j], base.x0_hat[i])


@pytest.mark.parametrize("chunk,sizes", [(1, [1] * 5), (2, [2, 2, 1]), (3, [3, 2]), (5, [5]), (64, [5])])
def test_chunking_changes_calls_not_results(tmp_path, chunk, sizes):
    ref_s, _ = _sampler(tmp_path / "ref", mode="affine", a=0.5)
    smp, tr = _sampler(tmp_path / "chk", mode="affine", a=0.5, max_views_per_call=chunk)
    x0, caps, seeds = rand_latent(B, seed=5), make_caps(B, seed=5), [6, 7, 8, 9, 10]
    ref = _run(ref_s, x0, caps, seeds)
    out = _run(smp, x0, caps, seeds)
    assert torch.equal(out.z, ref.z) and torch.equal(out.x0_hat, ref.x0_hat)
    assert [c.n for c in tr.calls] == sizes * 2
    assert out.calls == len(sizes) * 2 and out.steps == 2


def test_observer_sees_every_transformer_call_before_it_happens(tmp_path):
    smp, tr = _sampler(tmp_path, max_views_per_call=2)
    seen = []

    def obs(info):
        seen.append((info, len(tr.calls)))                      # 호출 직전의 변환기 호출 수

    r = _run(smp, rand_latent(B, seed=6), make_caps(B), [1, 2, 3, 4, 5], observer=obs)
    assert r.calls == len(seen) == 6
    key = [(i.step, i.view_lo, i.view_hi) for i, _ in seen]
    assert key == [(0, 0, 2), (0, 2, 4), (0, 4, 5), (1, 0, 2), (1, 2, 4), (1, 4, 5)]
    assert [n for _, n in seen] == list(range(6))               # k번째 관찰 시점에 변환기는 k번 불렸다
    for info, _ in seen:
        assert isinstance(info, S.StepInfo)
        assert info.sigma == pytest.approx(P[5 + info.step], abs=1e-12)
        assert info.sigma_next == pytest.approx(P[6 + info.step], abs=1e-12)
    # observer 가 있어도 결과는 같다
    plain = _run(_sampler(tmp_path / "p")[0], rand_latent(B, seed=6), make_caps(B), [1, 2, 3, 4, 5])
    assert torch.equal(plain.z, r.z)


def test_observer_exception_propagates_without_forward(tmp_path):
    smp, tr = _sampler(tmp_path)

    def boom(info):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        _run(smp, rand_latent(2), make_caps(2), [1, 2], observer=boom)
    assert tr.calls == []


def test_x0_hook_identity_is_a_noop(tmp_path):
    smp, _ = _sampler(tmp_path, mode="affine", a=0.5)
    x0, caps = rand_latent(2, seed=7), make_caps(2, seed=7)
    a = _run(smp, x0, caps, [1, 2])
    b = _run(smp, x0, caps, [1, 2], x0_hook=lambda x, info: x)
    assert torch.equal(a.z, b.z) and torch.equal(a.x0_hat, b.x0_hat)


def test_x0_hook_blend_follows_ddim_style_step(tmp_path):
    smp, _ = _sampler(tmp_path)                                  # 완전 디노이저: x̂0 == x*
    x0, caps = rand_latent(2, seed=8), make_caps(2, seed=8)
    ref = rand_latent(2, seed=88)
    seen = []

    def hook(x, info):
        seen.append((x.dtype, tuple(x.shape), info))
        return 0.5 * x + 0.5 * ref

    r = smp.sdedit(x0, caps, sigma=0.5, seeds=[3, 4], steps=2, x0_hook=hook)      # idx6 -> idx7 -> 0
    assert len(seen) == 2
    assert all(d == torch.float32 and s == (2, 16, 4, 6) for d, s, _ in seen)
    assert [(i.step, i.view_lo, i.view_hi) for _, _, i in seen] == [(0, 0, 2), (1, 0, 2)]
    assert seen[0][2].sigma == pytest.approx(P[6]) and seen[1][2].sigma_next == 0.0
    tg = targets(caps)
    x0p = 0.5 * tg + 0.5 * ref
    torch.testing.assert_close(r.x0_hat, tg, atol=5e-5, rtol=0)                  # x0_hat 은 hook 전 원 예측
    torch.testing.assert_close(r.z, x0p, atol=5e-5, rtol=0)                      # 끝(σ=0)이면 z == hook 후 x̂0
    torch.testing.assert_close(r.z, 0.5 * r.x0_hat + 0.5 * ref, atol=1e-6, rtol=0)
    one = smp.sdedit(x0, caps, sigma=0.5, seeds=[3, 4], steps=1, x0_hook=hook)    # σ=0.5 -> 0.3 한 step
    want = (P[7] / P[6]) * one.z_start + (1 - P[7] / P[6]) * x0p
    torch.testing.assert_close(one.z, want, atol=5e-5, rtol=0)
    assert one.sigma_end == pytest.approx(P[7], abs=1e-12)


def test_x0_hook_with_chunks_is_called_once_per_step_on_full_batch(tmp_path):
    smp, tr = _sampler(tmp_path, max_views_per_call=2)
    shapes = []
    _run(smp, rand_latent(B, seed=9), make_caps(B), [1, 2, 3, 4, 5],
         x0_hook=lambda x, info: (shapes.append((tuple(x.shape), info.view_lo, info.view_hi)) or x))
    assert shapes == [((B, 16, 4, 6), 0, B)] * 2


def test_x0_hook_bad_return_fails(tmp_path):
    smp, _ = _sampler(tmp_path)
    for bad in (lambda x, i: x[:1], lambda x, i: x.unsqueeze(0), lambda x, i: None,
                lambda x, i: torch.full_like(x, float("nan"))):
        with pytest.raises((ValueError, TypeError)):
            _run(smp, rand_latent(2), make_caps(2), [1, 2], x0_hook=bad)


def test_grad_flows_to_x0_when_requested(tmp_path):
    smp, tr = _sampler(tmp_path, mode="affine", a=0.5)
    x0 = rand_latent(2, seed=10).requires_grad_(True)
    r = smp.sdedit(x0, make_caps(2), sigma=0.5, seeds=[1, 2], steps=1, grad=True)   # z_start=.5 x0+.5ε, x̂0=.5 z+x*
    assert tr.calls[0].grad_enabled
    assert r.x0_hat.requires_grad and r.z.requires_grad
    r.x0_hat.sum().backward()
    torch.testing.assert_close(x0.grad, torch.full_like(x0, 0.5 * (1 - P[6])), atol=2e-6, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in tr_params(smp))        # 교사는 동결 그대로


def tr_params(smp):
    return list(smp.teacher.transformer.parameters())


def test_grad_flows_through_multiple_steps_and_denoise_and_predict(tmp_path):
    smp, _ = _sampler(tmp_path, mode="affine", a=0.5)
    # denoise 두 step(σ6->σ7->0): z1 = (σ7/σ6) z + (1-σ7/σ6) x̂0,  x̂0 = .5 z + x*  -> dz1/dz = σ7/σ6 + .5 (1-σ7/σ6)
    z = rand_latent(1, seed=11).requires_grad_(True)
    r = smp.denoise(z, make_caps(1), start_index=6, steps=2, grad=True)
    r.z.sum().backward()
    k1 = P[7] / P[6] + 0.5 * (1 - P[7] / P[6])
    k2 = 0.5                                         # 마지막 step 은 z == x̂0 = .5 z1 + x*
    torch.testing.assert_close(z.grad, torch.full_like(z, k1 * k2), atol=2e-5, rtol=0)
    z2 = rand_latent(1, seed=12).requires_grad_(True)
    smp.predict_x0(z2, P[4], make_caps(1), grad=True).sum().backward()
    torch.testing.assert_close(z2.grad, torch.full_like(z2, 0.5), atol=2e-6, rtol=0)
    z3 = rand_latent(1, seed=13).requires_grad_(True)
    out = smp.predict_x0(z3, P[4], make_caps(1))               # grad=False -> 끊김
    assert not out.requires_grad


def test_no_grad_default_even_when_global_grad_enabled(tmp_path):
    smp, tr = _sampler(tmp_path)
    assert torch.is_grad_enabled()
    smp.sdedit(rand_latent(1).requires_grad_(True), make_caps(1), sigma=0.5, seeds=1, steps=1)
    assert not tr.calls[0].grad_enabled
    assert torch.is_grad_enabled()                              # 전역 상태를 되돌려 놓는다
