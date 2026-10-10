"""손실·지표: 대역 안 affine-inv 깊이(오라클), 무효 대역 무시, 법선·알베도 오라클, sem CE, 가중 합, 무기울기, 지표."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

import mdl03_helpers as Hp
from buc.data import realify as R
from buc.models.mrh import heads as Hd
from buc.models.mrh import train as TR

B, Hf, h = 2, 64, 16
SEMD = 8
N_IDS = 16


def _batch(seed=0, band_valid=None, lm=None):
    g = Hp.gen(seed)
    gbuf = torch.zeros(B, 12, Hf, Hf)
    ld = Hp.smooth_field(seed, B, 1, h, h, kmax=2) * 0.5 + 1.0
    gbuf[:, 0:1] = ld.repeat_interleave(4, -2).repeat_interleave(4, -1)
    n = torch.randn(B, 3, h, h, generator=g)
    n[:, 2] = -3.0
    n = n / n.norm(dim=1, keepdim=True)
    gbuf[:, 1:4] = n.repeat_interleave(4, -2).repeat_interleave(4, -1)
    alb = torch.rand(B, 3, h, h, generator=g)
    gbuf[:, 4:7] = alb.repeat_interleave(4, -2).repeat_interleave(4, -1)
    gbuf[:, 11] = 1.0 if lm is None else lm
    sem = torch.randint(0, N_IDS, (B, h, h), generator=g).repeat_interleave(4, -2).repeat_interleave(4, -1)
    bv = band_valid or (True,) * Hp.N_OCT
    return {"gbuf": gbuf, "sem": sem.long(), "band_valid": (tuple(bv),) * B, "t": torch.full((B,), 0.5),
            "keys": tuple(f"k{i}" for i in range(B))}


def _out(seed, ld):
    g = Hp.gen(100 + seed)
    def unit(c):
        x = torch.randn(B, c, h, h, generator=g)
        return x / x.norm(dim=1, keepdim=True)
    return {"log_depth": ld, "normal": unit(3), "albedo": torch.rand(B, 3, h, h, generator=g),
            "rough_metal": torch.rand(B, 2, h, h, generator=g), "shading": torch.rand(B, 1, h, h, generator=g),
            "defocus": torch.rand(B, 1, h, h, generator=g), "sem": unit(SEMD),
            "sigma_hat": torch.rand(B, 4, h, h, generator=g) + 0.01}



def _table():
    g = Hp.gen(55)
    t = torch.randn(N_IDS, SEMD, generator=g)
    return t


def _loss(out, batch, **kw):
    kw.setdefault("sem_table", _table())
    return TR.mrh_loss(out, batch, **kw)


def _gt4(batch):
    return TR.downsample_labels(batch["gbuf"], batch["sem"], 4)


# ----------------------------------------------------------------------------------------- 깊이

def test_depth_affine_invariant():
    bt = _batch(1)
    g4 = _gt4(bt)["log_depth"]
    tot, met = _loss(_out(1, 2.5 * g4 - 0.7), bt)
    assert met["loss/depth"] <= 1e-7
    tot, met = _loss(_out(1, g4 + 0.1 * Hp.smooth_field(9, B, 1, h, h)), bt)
    assert met["loss/depth"] > 1e-4


def _checker(n):
    i = torch.arange(n)
    return (1.0 - 2.0 * ((i.view(n, 1) + i.view(1, n)) % 2)).float()


def test_depth_ignores_invalid_band():
    valid = (True, True, False, True, True, True)  # 1/4 해상 옥타브 0(체커)만 무효
    bt = _batch(2, band_valid=valid)
    g4 = _gt4(bt)["log_depth"]
    pred = 1.7 * g4 + 0.3 + 0.4 * _checker(h).view(1, 1, h, h)
    _, met = _loss(_out(2, pred), bt)
    assert met["loss/depth"] <= 1e-7
    _, met_all = _loss(_out(2, pred), _batch(2))
    assert met_all["loss/depth"] > 1e-3


def _np_depth(pred, gt, mask, valid4):
    vals = []
    for b in range(pred.shape[0]):
        m = mask[b, 0].double()
        n = float(m.sum())
        if n < TR.MIN_VALID_PX:
            continue
        bp = R.band_project((m * pred[b, 0].double()), valid4).numpy().ravel()
        bm = R.band_project(m, valid4).numpy().ravel()
        bg = R.band_project((m * gt[b, 0].double()), valid4).numpy().ravel()
        A = np.array([[bp @ bp, bp @ bm], [bp @ bm, bm @ bm]]) + 1e-8 * np.eye(2)
        a, c = np.linalg.solve(A, np.array([bp @ bg, bm @ bg]))
        r = a * bp + c * bm - bg
        vals.append(float(r @ r) / max(n, 1.0))
    return float(np.mean(vals)) if vals else 0.0


@pytest.mark.parametrize("valid", [(True,) * 6, (False, True, True, False, True, True), (True, True, True, True, True, False)])
def test_depth_matches_oracle(valid):
    lm = torch.ones(B, Hf, Hf)
    lm[0, 0:16, 0:20] = 0.0
    lm[1, 40:64, 8:12] = 0.0
    bt = _batch(3, band_valid=valid, lm=lm)
    lab = _gt4(bt)
    pred = 0.8 * lab["log_depth"] + 0.2 * Hp.smooth_field(4, B, 1, h, h, kmax=3) + 0.05 * torch.randn(
        B, 1, h, h, generator=Hp.gen(4))
    _, met = _loss(_out(3, pred), bt)
    exp = _np_depth(pred, lab["log_depth"], lab["mask"], TR.band_valid_at_stride(valid, 4))
    assert met["loss/depth"] == pytest.approx(exp, rel=1e-4, abs=1e-9)


def test_depth_excludes_samples_below_min_valid():
    lm = torch.ones(B, Hf, Hf)
    lm[1] = 0.0
    lm[1, :8, :8] = 1.0   # 1/4 해상 유효 4 px < 16
    bt = _batch(5, lm=lm)
    lab = _gt4(bt)
    pred = lab["log_depth"] + 0.1 * Hp.smooth_field(6, B, 1, h, h)
    _, met = _loss(_out(5, pred), bt)
    exp = _np_depth(pred[:1], lab["log_depth"][:1], lab["mask"][:1], (True,) * 4)
    assert met["loss/depth"] == pytest.approx(exp, rel=1e-4)


def test_depth_gradient_descends():
    bt = _batch(7)
    lab = _gt4(bt)
    pred = (lab["log_depth"] + 0.2 * Hp.smooth_field(8, B, 1, h, h)).requires_grad_(True)
    out = _out(7, pred)
    tot, met = _loss(out, bt, weights=TR.LossWeights(depth=1.0, normal=0.0, sem=0.0, albedo=0.0))
    (gr,) = torch.autograd.grad(tot, pred)
    assert bool(torch.isfinite(gr).all()) and float(gr.abs().max()) > 0
    _, met2 = _loss(_out(7, (pred - 0.05 * gr / gr.norm() * pred.norm()).detach()), bt)
    assert met2["loss/depth"] < met["loss/depth"]


# ----------------------------------------------------------------------------------------- 법선·알베도·sem·합

def _np_vec(pred, gt, mask, valid4):
    vals = []
    for b in range(pred.shape[0]):
        m = mask[b, 0].double()
        n = float(m.sum())
        if n < TR.MIN_VALID_PX:
            continue
        s = 0.0
        for c in range(pred.shape[1]):
            r = R.band_project(m * (pred[b, c].double() - gt[b, c].double()), valid4)
            s += float((r * r).sum())
        vals.append(s / max(n, 1.0))
    return float(np.mean(vals))


def test_normal_and_albedo_match_oracle():
    valid = (True, False, True, True, False, True)
    lm = torch.ones(B, Hf, Hf)
    lm[0, 20:40, 20:40] = 0.0
    bt = _batch(9, band_valid=valid, lm=lm)
    lab = _gt4(bt)
    out = _out(9, lab["log_depth"])
    _, met = _loss(out, bt, weights=TR.LossWeights(albedo=1.0))
    v4 = TR.band_valid_at_stride(valid, 4)
    assert met["loss/normal"] == pytest.approx(_np_vec(out["normal"], lab["normal"], lab["mask"], v4), rel=1e-4)
    assert met["loss/albedo"] == pytest.approx(_np_vec(out["albedo"], lab["albedo"], lab["mask"], v4), rel=1e-4)


def test_sem_ce_oracle_and_ignore():
    bt = _batch(11)
    s = bt["sem"].clone()
    s[0, :16, :] = 0          # 무시 id
    s[1, :, :16] = 99         # 범위 밖
    bt["sem"] = s
    out = _out(11, _gt4(bt)["log_depth"])
    _, met = _loss(out, bt)
    tab = _table()
    tab = tab / tab.norm(dim=1, keepdim=True)
    logits = torch.einsum("bchw,nc->bnhw", out["sem"].double(), tab.double()) / TR.SEM_TEMP
    lab = s[:, ::4, ::4]
    ok = (lab != 0) & (lab >= 0) & (lab < N_IDS)
    ce = Fn.cross_entropy(logits, lab.clamp(0, N_IDS - 1), reduction="none")[ok].mean()
    acc = (logits.argmax(1) == lab)[ok].double().mean()
    assert met["loss/sem"] == pytest.approx(float(ce), rel=1e-4)
    assert met["sem_acc"] == pytest.approx(float(acc), abs=1e-6)


def test_total_is_weighted_sum():
    bt = _batch(12)
    out = _out(12, _gt4(bt)["log_depth"] * 0.3 + Hp.smooth_field(1, B, 1, h, h))
    w = TR.LossWeights(depth=0.7, normal=1.3, sem=0.2, albedo=0.4)
    tot, met = _loss(out, bt, weights=w)
    exp = 0.7 * met["loss/depth"] + 1.3 * met["loss/normal"] + 0.2 * met["loss/sem"] + 0.4 * met["loss/albedo"]
    assert float(tot) == pytest.approx(exp, rel=1e-4)
    d = TR.LossWeights()
    assert (d.depth, d.normal, d.sem, d.albedo) == (1.0, 1.0, 0.5, 0.0)


def test_zero_weight_heads_get_no_gradient():
    cfg = Hp.tiny_cfg(sem_dim=SEMD)
    m = Hd.MRH(cfg, seed=0)
    out = m(Hp.rand_taps(13, b=B), 0.5)
    bt = _batch(13)
    tot, _ = _loss(out, bt, weights=TR.LossWeights(depth=1.0, normal=1.0, sem=0.5, albedo=0.0))
    names = list(out)
    grads = torch.autograd.grad(tot, [out[k] for k in names], allow_unused=True, retain_graph=True)
    gd = dict(zip(names, grads))
    for k in ("albedo", "rough_metal", "shading", "defocus", "sigma_hat"):
        assert gd[k] is None or float(gd[k].abs().max()) == 0.0, k
    for k in ("log_depth", "normal", "sem"):
        assert gd[k] is not None and float(gd[k].abs().max()) > 0, k
    tot.backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in m.neck.parameters())


# ----------------------------------------------------------------------------------------- 지표

def test_depth_absrel_metric():
    g = Hp.gen(20)
    d = torch.exp(torch.rand(2, 1, 8, 8, generator=g) * 2.0) + 0.5
    mask = torch.ones(2, 1, 8, 8)
    r = TR.depth_absrel(2.0 * torch.log(d) + 1.0, d, mask)
    assert r.shape == (2,) and float(r.max()) <= 1e-5
    noise = 0.1 * torch.randn(2, 1, 8, 8, generator=g)
    mask[0, 0, :3] = 0.0
    p = torch.log(d) + noise
    r = TR.depth_absrel(p, d, mask)
    for b in range(2):
        mm = mask[b, 0].bool().numpy()
        x = p[b, 0].double().numpy()[mm]
        y = np.log(d[b, 0].double().numpy()[mm])
        a, c = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
        dd = d[b, 0].double().numpy()[mm]
        exp = np.mean(np.abs(np.exp(a * x + c) - dd) / dd)
        assert float(r[b]) == pytest.approx(exp, rel=1e-3)


def test_normal_angle_metric():
    g = Hp.gen(21)
    n = torch.randn(1, 3, 6, 6, generator=g)
    n = n / n.norm(dim=1, keepdim=True)
    th = math.radians(10.0)
    rot = torch.tensor([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    p = torch.einsum("ij,bjhw->bihw", rot, n)
    mask = torch.ones(1, 1, 6, 6)
    ang = np.degrees(np.arccos(np.clip((p * n).sum(1)[0].double().numpy(), -1, 1)))
    assert float(TR.normal_angle_deg(p, n, mask)[0]) == pytest.approx(float(ang.mean()), abs=1e-2)
    # 회전축(x)에 수직인 법선만이면 정확히 10°
    q = torch.zeros(1, 3, 6, 6)
    q[:, 1] = 0.6
    q[:, 2] = -0.8
    assert float(TR.normal_angle_deg(torch.einsum("ij,bjhw->bihw", rot, q), q, mask)[0]) == pytest.approx(10.0, abs=1e-2)
    mask[0, 0, :, :3] = 0.0
    p2 = p.clone()
    p2[:, :, :, :3] = -n[:, :, :, :3]  # 무효 영역의 180° 오차는 무시
    assert float(TR.normal_angle_deg(p2, n, mask)[0]) == pytest.approx(float(ang[:, 3:].mean()), abs=1e-2)
