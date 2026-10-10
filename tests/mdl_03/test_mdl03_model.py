"""MRH 본체 구조: 설정 검증, 모양, 출력 범위, FiLM 영 초기화, 뷰 독립, seed 결정론, 예산, 입력 검증."""
import json

import pytest
import torch

import mdl03_helpers as Hp
from buc.models.mrh import heads as Hd
from buc.models.mrh import neck as N


# ----------------------------------------------------------------------------------------- 설정

def test_config_defaults_and_to_dict():
    c = N.MRHConfig()
    assert c.tap_layers == (8, 15, 23)
    assert (c.tap_dim, c.neck_ch, c.rcu_per_fusion, c.head_ch, c.sem_dim, c.n_sigma_bands) == (128, 256, 4, 128, 768, 4)
    d = c.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["tap_layers"] == [8, 15, 23]
    assert d["neck_ch"] == 256 and d["sem_dim"] == 768


@pytest.mark.parametrize("kw", [
    dict(tap_layers=(8, 8, 23)), dict(tap_layers=(8, 15)), dict(tap_layers=(23, 15, 8)),
    dict(tap_layers=(-1, 15, 23)), dict(neck_ch=0), dict(tap_dim=-3), dict(sem_dim=True),
    dict(rcu_per_fusion=0), dict(n_sigma_bands=0), dict(head_ch=2.5),
])
def test_config_rejects(kw):
    with pytest.raises(N.MRHSpecError):
        N.MRHConfig(**kw)


def test_head_channels_order():
    cfg = Hp.tiny_cfg(sem_dim=12, n_sigma_bands=3)
    hc = Hd.head_channels(cfg)
    assert list(hc) == ["log_depth", "normal", "albedo", "rough_metal", "shading", "defocus", "sem", "sigma_hat"]
    assert list(hc.values()) == [1, 3, 3, 2, 1, 1, 12, 3]
    assert tuple(Hd.HEAD_NAMES) == tuple(hc)


# ----------------------------------------------------------------------------------------- 모양·범위

@pytest.mark.parametrize("gh,gw", [(4, 4), (3, 5)])
def test_shapes_quarter_res(gh, gw):
    cfg = Hp.tiny_cfg()
    m = Hd.MRH(cfg, seed=0).eval()
    taps = Hp.rand_taps(1, b=2, gh=gh, gw=gw)
    with torch.no_grad():
        out = m(taps, torch.tensor([0.3, 0.6]))
        nk = m.features(taps, 0.5)
    for name, c in Hd.head_channels(cfg).items():
        assert out[name].shape == (2, c, 4 * gh, 4 * gw), name
        assert out[name].dtype == torch.float32
        assert bool(torch.isfinite(out[name]).all())
    assert nk.feat4.shape == (2, cfg.neck_ch, 4 * gh, 4 * gw)
    assert nk.feat16.shape == (2, cfg.neck_ch, gh, gw)
    assert all(p.dtype == torch.float32 for p in m.parameters())


def _perturb(m, seed, s=0.3):
    g = Hp.gen(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * s)


def test_output_ranges_even_when_perturbed():
    cfg = Hp.tiny_cfg()
    m = Hd.MRH(cfg, seed=1).eval()
    _perturb(m, 5)
    taps = Hp.rand_taps(2, b=3, mul=10.0)
    with torch.no_grad():
        out = m(taps, 0.4)
    for name in ("normal", "sem"):
        n = out[name].norm(dim=1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-4), name
    for name in ("albedo", "rough_metal"):
        assert float(out[name].min()) >= 0.0 and float(out[name].max()) <= 1.0
    for name in ("shading", "defocus"):
        assert float(out[name].min()) >= 0.0
    assert float(out["sigma_hat"].min()) >= Hd.SIGMA_MIN * (1 - 1e-6)
    assert Hd.SIGMA_MIN == pytest.approx(1e-3)


def test_film_zero_init_then_t_dependent():
    cfg = Hp.tiny_cfg()
    m = Hd.MRH(cfg, seed=2).eval()
    taps = Hp.rand_taps(3, b=2)
    with torch.no_grad():
        a = m(taps, 0.25)
        b = m(taps, torch.tensor([0.7, 1.0]))
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), f"초기 출력이 t에 의존한다: {k}"
    _perturb(m, 9, s=0.05)
    with torch.no_grad():
        a = m(taps, 0.25)
        b = m(taps, 0.7)
    diff = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert diff > 1e-4, "FiLM이 t를 전혀 쓰지 않는다"


def test_view_independence_and_no_batchnorm():
    cfg = Hp.tiny_cfg()
    m = Hd.MRH(cfg, seed=3)
    assert not any(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in m.modules())
    _perturb(m, 11, s=0.05)
    m.eval()
    taps = Hp.rand_taps(4, b=3)
    t = torch.tensor([0.2, 0.5, 0.9])
    with torch.no_grad():
        full = m(taps, t)
        for i in (2, 0):
            one = m({l: v[i:i + 1] for l, v in taps.items()}, t[i:i + 1])
            for k in full:
                assert torch.allclose(full[k][i:i + 1], one[k], atol=1e-5, rtol=1e-4), k


def test_seed_determinism_and_global_rng_untouched():
    cfg = Hp.tiny_cfg()
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    a = Hd.MRH(cfg, seed=7)
    b = Hd.MRH(cfg, seed=7)
    c = Hd.MRH(cfg, seed=8)
    assert torch.equal(torch.get_rng_state(), before)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa if sa[k].numel() > 1 and float(sa[k].abs().sum()) > 0)


# ----------------------------------------------------------------------------------------- 기본 설정 예산

def test_default_budget_and_rcu_count():
    m = Hd.MRH(N.MRHConfig(), seed=0)
    assert isinstance(m.neck, N.MRHNeck) and isinstance(m.heads, Hd.MRHHeads)
    n_neck = sum(p.numel() for p in m.neck.parameters())
    n_head = sum(p.numel() for p in m.heads.parameters())
    assert 9e6 <= n_neck <= 18e6, n_neck
    assert 0.5e6 <= n_head <= 3.5e6, n_head
    assert n_neck + n_head <= 20e6
    rcus = [x for x in m.neck.modules() if isinstance(x, N.ResidualConvUnit)]
    assert len(rcus) == 12
    for r in rcus:
        convs = [c for c in r.modules() if isinstance(c, torch.nn.Conv2d)]
        assert len(convs) == 2
        for c in convs:
            assert c.kernel_size == (3, 3) and c.in_channels == 256 and c.out_channels == 256


def test_default_forward_runs():
    m = Hd.MRH(N.MRHConfig(), seed=0).eval()
    g = Hp.gen(0)
    taps = {l: torch.randn(1, 128, 2, 3, generator=g) for l in (8, 15, 23)}
    with torch.no_grad():
        out = m(taps, 0.5)
    assert out["sem"].shape == (1, 768, 8, 12)
    assert out["sigma_hat"].shape == (1, 4, 8, 12)


# ----------------------------------------------------------------------------------------- 입력 검증

def _bad_inputs():
    good = Hp.rand_taps(5, b=2)
    missing = {l: v for l, v in good.items() if l != 15}
    extra = dict(good)
    extra[30] = good[8]
    wrongk = dict(good)
    wrongk[23] = torch.zeros(2, Hp.K + 1, Hp.GH, Hp.GW)
    grid = dict(good)
    grid[15] = torch.zeros(2, Hp.K, Hp.GH + 1, Hp.GW)
    dims3 = dict(good)
    dims3[8] = torch.zeros(Hp.K, Hp.GH, Hp.GW)
    return [(missing, 0.5), (extra, 0.5), (wrongk, 0.5), (grid, 0.5), (dims3, 0.5),
            (good, 0.0), (good, 1.5), (good, float("nan")), (good, torch.tensor([0.5, 0.5, 0.5])),
            (good, torch.tensor([0.5, -0.1]))]


@pytest.mark.parametrize("idx", range(10))
def test_input_validation(idx):
    m = Hd.MRH(Hp.tiny_cfg(), seed=0)
    taps, t = _bad_inputs()[idx]
    with pytest.raises(N.MRHSpecError):
        m(taps, t)


def test_int8_taps_rejected():
    m = Hd.MRH(Hp.tiny_cfg(), seed=0)
    taps = {l: torch.zeros(1, Hp.K, Hp.GH, Hp.GW, dtype=torch.int8) for l in Hp.LAYERS}
    with pytest.raises(TypeError):
        m(taps, 0.5)


def test_import_has_no_cuda_requirement():
    import importlib
    for mod in ("buc.models.mrh.neck", "buc.models.mrh.heads", "buc.models.mrh.train"):
        importlib.import_module(mod)
