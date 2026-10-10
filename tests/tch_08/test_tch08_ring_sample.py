"""ring_sample: 해석 오라클 대조(Ring / Ring-R), sdedit 동치, σ 순서·NFE, 띠 보존, lockstep, 검증 우선."""
import pytest
import torch
from tch08_fakes import STRIP, C, H, W, make_caps, make_z_teacher, rand_latent, target_of

from buc.core import det
from buc.teacher import ref_slot as R
from buc.teacher.sample import TeacherSampler, add_noise, euler_step
from buc.teacher.zimage import make_noise

A = 0.5


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    t = make_z_teacher(tmp_path_factory.mktemp("z"), a=A)
    return t, TeacherSampler(t)


def _ref_latent(seed=4):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(C, STRIP, W, generator=g)


def _oracle(x0, caps, ref, seeds, table, sigma, ring_r, level=0, k=0, steps=None):
    """독립 참조 구현: 가짜 디노이저 x0_hat = A*z + x*(cap) 를 직접 계산한다."""
    s0 = table.snap(sigma).index
    steps = len(table) - s0 if steps is None else steps
    pts = list(table.points) + [0.0]
    x = x0.clone().float()
    x[:, :, :STRIP] = ref
    b = x.shape[0]
    eps = torch.stack([make_noise(seeds[i], (1, C, H, W), level=level, k=k)[0] for i in range(b)])
    z = add_noise(x, pts[s0], eps)
    tgt = torch.stack([target_of(c) for c in caps])
    sig = []

    def step(z, idx):
        sig.append(pts[idx])
        x0h = A * z + tgt
        x0h = x0h.clone()
        x0h[:, :, :STRIP] = ref
        return euler_step(z, x0h, pts[idx], pts[idx + 1])

    for s in range(s0, s0 + steps):
        if ring_r is not None and s in ring_r.resample_at and s > s0:
            for j in range(ring_r.r):
                kk = k + R.RINGR_K_BASE + 64 * s + j
                eta = torch.stack([make_noise(seeds[i], (1, C, H, W), level=level, k=kk)[0] for i in range(b)])
                z = R.renoise_up(z, pts[s], pts[s - 1], eta)
                z = step(z, s - 1)
        z = step(z, s)
    return z, sig


@pytest.mark.parametrize("sigma,ring_r", [(1.0, None), (0.75, None), (1.0, R.RingRConfig()),
                                          (0.9, R.RingRConfig(resample_at=(3, 4, 5), r=2)),
                                          (0.6, R.RingRConfig(resample_at=(2, 3, 7), r=1))])
def test_matches_oracle(setup, sigma, ring_r):
    t, s = setup
    x0 = rand_latent(2, seed=1)
    caps = make_caps(2, seed=2)
    ref = _ref_latent()
    seeds = [101, 202]
    res = R.ring_sample(s, x0, caps, ref_latent=ref, strip_rows=STRIP, sigma=sigma, seeds=seeds, ring_r=ring_r)
    want, sig = _oracle(x0, caps, ref, seeds, s.table, sigma, ring_r)
    assert torch.allclose(res.z, want, atol=1e-4, rtol=1e-4)
    assert res.sigmas == pytest.approx(tuple(sig), abs=1e-12)
    s0 = s.table.snap(sigma).index
    extra = 0 if ring_r is None else ring_r.extra_nfe(s0)
    assert res.nfe == len(s.table) - s0 + extra == len(sig)
    assert res.calls == res.nfe
    assert res.start_index == s0 and res.table_key == s.table.key
    assert torch.equal(res.strip, res.z[:, :, :STRIP]) and torch.equal(res.body, res.z[:, :, STRIP:])


def test_default_ring_r_is_ten_nfe_with_sigma_order(setup):
    t, s = setup
    p = s.table.points
    res = R.ring_sample(s, rand_latent(1), make_caps(1), ref_latent=_ref_latent(), strip_rows=STRIP, sigma=1.0,
                        seeds=[5], ring_r=R.RingRConfig())
    assert res.nfe == 10 and res.calls == 10
    assert res.sigmas == pytest.approx((p[0], p[1], p[2], p[2], p[2], p[3], p[4], p[5], p[6], p[7]))


def test_strip_equals_reference_at_end(setup):
    t, s = setup
    ref = _ref_latent(9)
    for rr in (None, R.RingRConfig()):
        res = R.ring_sample(s, rand_latent(2, 3), make_caps(2, 4), ref_latent=ref, strip_rows=STRIP, sigma=0.9,
                            seeds=[1, 2], ring_r=rr)
        assert torch.equal(res.strip, ref.expand(2, C, STRIP, W))


def test_ring_without_r_is_sdedit_with_hook_bitwise(setup):
    t, s = setup
    x0 = rand_latent(2, seed=6)
    caps = make_caps(2, seed=7)
    ref = _ref_latent(8)
    res = R.ring_sample(s, x0, caps, ref_latent=ref, strip_rows=STRIP, sigma=0.75, seeds=11, level=2, k=3)
    xs = x0.clone()
    xs[:, :, :STRIP] = ref
    sd = s.sdedit(xs, caps, sigma=0.75, seeds=11, steps=len(s.table) - s.table.snap(0.75).index, level=2, k=3,
                  x0_hook=R.strip_hook(ref, STRIP))
    assert torch.equal(res.z, sd.z)


def test_int_seed_derivation(setup):
    t, s = setup
    x0, caps, ref = rand_latent(2, 1), make_caps(2), _ref_latent()
    a = R.ring_sample(s, x0, caps, ref_latent=ref, strip_rows=STRIP, sigma=0.9, seeds=77, ring_r=R.RingRConfig())
    b = R.ring_sample(s, x0, caps, ref_latent=ref, strip_rows=STRIP, sigma=0.9,
                      seeds=[det.derive_seed(77, 0), det.derive_seed(77, 1)], ring_r=R.RingRConfig())
    assert torch.equal(a.z, b.z)


def test_deterministic_and_seed_sensitive(setup):
    t, s = setup
    x0, caps, ref = rand_latent(1, 1), make_caps(1), _ref_latent()
    kw = dict(ref_latent=ref, strip_rows=STRIP, sigma=1.0, ring_r=R.RingRConfig(), steps=5)
    a = R.ring_sample(s, x0, caps, seeds=[3], **kw)
    b = R.ring_sample(s, x0, caps, seeds=[3], **kw)
    c = R.ring_sample(s, x0, caps, seeds=[4], **kw)
    assert torch.equal(a.z, b.z)
    assert not torch.allclose(a.body, c.body, atol=1e-3)          # 5 step 에서 끝나 노이즈가 남는다
    assert a.nfe == 5 + 2


def test_resample_noise_matters(setup):
    """Ring-R 은 Ring 과 다른 결과를 낸다(steps 를 줄여 노이즈가 남게)."""
    t, s = setup
    x0, caps, ref = rand_latent(1, 1), make_caps(1), _ref_latent()
    kw = dict(ref_latent=ref, strip_rows=STRIP, sigma=1.0, seeds=[3], steps=6)
    a = R.ring_sample(s, x0, caps, **kw)
    b = R.ring_sample(s, x0, caps, ring_r=R.RingRConfig(), **kw)
    assert (a.body - b.body).abs().max() > 1e-3


def test_lockstep_view_independence(setup):
    t, s = setup
    x0 = rand_latent(3, seed=12)
    caps = make_caps(3, seed=13)
    ref = torch.stack([_ref_latent(20 + i) for i in range(3)])
    seeds = [31, 32, 33]
    kw = dict(strip_rows=STRIP, sigma=0.9, ring_r=R.RingRConfig(resample_at=(3, 4), r=2))
    full = R.ring_sample(s, x0, caps, ref_latent=ref, seeds=seeds, **kw)
    for i in range(3):
        one = R.ring_sample(s, x0[i:i + 1], caps[i:i + 1], ref_latent=ref[i], seeds=[seeds[i]], **kw)
        assert torch.allclose(full.z[i], one.z[0], atol=1e-6)
    perm = [2, 0, 1]
    pz = R.ring_sample(s, x0[perm], [caps[i] for i in perm], ref_latent=ref[perm], seeds=[seeds[i] for i in perm],
                       **kw)
    assert torch.allclose(pz.z, full.z[perm], atol=1e-6)


def test_perfect_denoiser_body_is_target(tmp_path):
    t = make_z_teacher(tmp_path, a=0.0)
    s = TeacherSampler(t)
    caps = make_caps(2, seed=5)
    res = R.ring_sample(s, rand_latent(2, 9), caps, ref_latent=_ref_latent(), strip_rows=STRIP, sigma=1.0,
                        seeds=[1, 2], ring_r=R.RingRConfig(resample_at=(2, 3, 4), r=2))
    for i in range(2):
        assert torch.allclose(res.body[i], target_of(caps[i])[:, STRIP:], atol=1e-4)


def test_observer_sees_every_forward(setup):
    t, s = setup
    seen = []
    res = R.ring_sample(s, rand_latent(1), make_caps(1), ref_latent=_ref_latent(), strip_rows=STRIP, sigma=1.0,
                        seeds=[1], ring_r=R.RingRConfig(), observer=lambda info: seen.append(info.sigma))
    assert seen == pytest.approx(list(res.sigmas))


@pytest.mark.parametrize("kw", [dict(strip_rows=1), dict(strip_rows=0), dict(strip_rows=H),
                                dict(ref_latent=torch.zeros(C, STRIP, W - 1)),
                                dict(ref_latent=torch.zeros(C + 1, STRIP, W)),
                                dict(ref_latent=torch.zeros(3, C, STRIP, W))])
def test_validation_before_transformer(setup, kw):
    t, s = setup
    n0 = len(t.transformer.calls)
    base = dict(ref_latent=_ref_latent(), strip_rows=STRIP, sigma=0.9, seeds=[1, 2])
    base.update(kw)
    with pytest.raises(ValueError):
        R.ring_sample(s, rand_latent(2), make_caps(2), **base)
    assert len(t.transformer.calls) == n0
