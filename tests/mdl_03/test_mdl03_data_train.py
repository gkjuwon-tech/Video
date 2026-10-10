"""데이터·학습 이음매: DAT-02 샤드 필드 → DAT-05 → 디코드, 거부 규칙, RUN-06 Trainer 배선(과적합·재개·기저 지문)."""
import numpy as np
import pytest
import torch

import mdl03_helpers as Hp
from buc.data import realify as R
from buc.data import shards as S
from buc.models.mrh import heads as Hd
from buc.models.mrh import train as TR
from buc.teacher import taps as T
from buc.train.ckpt import CheckpointMismatchError


def test_shard_roundtrip_decode_matches_tapset(tmp_path):
    basis = Hp.make_basis(0)
    cfg = Hp.tiny_cfg()
    results = [Hp.make_result(i, basis, tap_sigma=0.3 + 0.1 * i,
                              band_valid=(True, False, True, True, True, i % 2 == 0)) for i in range(3)]
    with S.ShardWriter(tmp_path / "sh", max_samples=10) as w:
        assert R.write_results(w, results) == 3
    paths = sorted(str(p) for p in (tmp_path / "sh").glob("*.tar"))
    assert len(paths) == 1
    ds = TR.MRHDataset.from_shards(paths, basis, cfg)
    assert len(ds) == 3 and ds.basis_fp == basis.fingerprint
    bt = ds.batch(np.array([2, 0], dtype=np.int64))
    assert bt["keys"] == ("s2", "s0")
    feats = {i: results[i].taps.features(basis) for i in (2, 0)}
    for l in Hp.LAYERS:
        assert bt["taps"][l].dtype == torch.float32
        assert bt["taps"][l].shape == (2, Hp.K, Hp.GH, Hp.GW)
        assert torch.equal(bt["taps"][l][0], feats[2][l][0])
        assert torch.equal(bt["taps"][l][1], feats[0][l][0])
    assert torch.allclose(bt["t"], torch.tensor([0.5, 0.3]), atol=1e-7)
    assert bt["band_valid"] == (results[2].band_valid, results[0].band_valid)
    assert bt["gbuf"].shape == (2, 12, Hp.H, Hp.W) and bt["gbuf"].dtype == torch.float32
    assert np.array_equal(bt["gbuf"][1].numpy(), results[0].labels["gbuf"])
    assert bt["sem"].dtype == torch.int64 and np.array_equal(bt["sem"][0].numpy(), results[2].labels["sem"])


def test_decode_rejects_other_basis():
    b0, b1 = Hp.make_basis(0), Hp.make_basis(1)
    s = Hp.make_sample(0, b0)
    with pytest.raises(T.TapBasisError):
        TR.decode_sample(s, b1, Hp.tiny_cfg())
    with pytest.raises(T.TapBasisError):
        TR.MRHDataset([s], b1, Hp.tiny_cfg())


def test_decode_rejects_bad_samples():
    b = Hp.make_basis(0)
    cfg = Hp.tiny_cfg()
    with pytest.raises(TR.MRHDataError):
        TR.decode_sample(Hp.make_sample(0, b, keep=False), b, cfg)
    with pytest.raises(TR.MRHDataError):
        TR.decode_sample(Hp.make_sample(0, b), b, Hp.tiny_cfg(tap_layers=(7, 15, 23)))
    bad = Hp.make_result(0, b, gbuf=np.zeros((12, Hp.H - 16, Hp.W), np.float32),
                         sem=np.zeros((Hp.H - 16, Hp.W), np.int16))
    with pytest.raises(TR.MRHDataError):
        TR.decode_sample(S.Sample(key=bad.key, fields=bad.shard_fields(), prov=Hp.ENGINE), b, cfg)
    s = Hp.make_sample(1, b)
    f = dict(s.fields)
    del f["gbuf.npy"]
    with pytest.raises(TR.MRHDataError):
        TR.decode_sample(S.Sample(key=s.key, fields=f, prov=Hp.ENGINE), b, cfg)


def test_collate_rejects_mismatched_shapes():
    b = Hp.make_basis(0)
    cfg = Hp.tiny_cfg()
    e0 = TR.decode_sample(Hp.make_sample(0, b), b, cfg)
    e1 = dict(TR.decode_sample(Hp.make_sample(1, b), b, cfg))
    e1["gbuf"] = e1["gbuf"][:, :32]
    with pytest.raises(TR.MRHDataError):
        TR.collate([e0, e1])


def _tcfg(lr=3e-3):
    return TR.MRHTrainConfig(model=Hp.tiny_cfg(), lr=lr, weight_decay=0.0, n_sem_ids=16)


def test_train_config_to_dict_and_hparams(tmp_path):
    basis = Hp.make_basis(0)
    ds = TR.MRHDataset(Hp.make_samples(4, basis), basis, Hp.tiny_cfg())
    tc = _tcfg()
    d = tc.to_dict()
    import json
    assert json.loads(json.dumps(d)) == d
    tr = TR.make_trainer(tc, ds, run_id="mdl03-hp", ckpt_dir=tmp_path / "ck", max_steps=0, batch_size=2)
    assert tr.cfg.hparams["basis_fp"] == basis.fingerprint.hex()
    assert tr.cfg.hparams["mrh"] == d
    assert tr.cfg.num_samples == 4 and tr.cfg.amp.dtype == "fp32"
    assert isinstance(tr.model, Hd.MRH)


def test_overfit_tiny_task(tmp_path):
    basis = Hp.make_basis(0)
    ds = TR.MRHDataset(Hp.make_samples(4, basis), basis, Hp.tiny_cfg())
    logs = []
    tr = TR.make_trainer(_tcfg(), ds, run_id="mdl03-fit", ckpt_dir=tmp_path / "ck", max_steps=80, batch_size=4,
                         seed=3, log=logs.append, log_every=1, max_abs_every=1000)
    res = tr.fit()
    assert res.step == 80 and res.stop_reason == "max_steps"
    tl = [e for e in logs if e.get("event") == "train"]
    assert len(tl) == 80
    first = np.mean([e["loss"] for e in tl[:3]])
    last = np.mean([e["loss"] for e in tl[-3:]])
    assert np.isfinite(last) and last <= 0.25 * first, (first, last)
    assert {"loss/depth", "loss/normal", "loss/sem", "sem_acc"} <= set(tl[-1]["metrics"])
    assert tl[-1]["metrics"]["loss/depth"] < tl[0]["metrics"]["loss/depth"]


def _params(tr):
    return {k: v.detach().clone() for k, v in tr.model.state_dict().items()}


def test_resume_bit_identical(tmp_path):
    basis = Hp.make_basis(0)
    ds = TR.MRHDataset(Hp.make_samples(6, basis), basis, Hp.tiny_cfg())
    kw = dict(batch_size=3, seed=5, log=lambda e: None, max_abs_every=1000)
    a = TR.make_trainer(_tcfg(), ds, run_id="mdl03-a", ckpt_dir=tmp_path / "a", max_steps=10, **kw)
    a.fit()
    b1 = TR.make_trainer(_tcfg(), ds, run_id="mdl03-a", ckpt_dir=tmp_path / "b", max_steps=4, **kw)
    b1.fit()
    b2 = TR.make_trainer(_tcfg(), ds, run_id="mdl03-a", ckpt_dir=tmp_path / "b", max_steps=10, **kw)
    r = b2.fit()
    assert r.resumed_from == 4 and r.step == 10
    pa, pb = _params(a), _params(b2)
    assert list(pa) == list(pb)
    assert all(torch.equal(pa[k], pb[k]) for k in pa)


def test_resume_refused_for_other_basis(tmp_path):
    b0, b1 = Hp.make_basis(0), Hp.make_basis(0, scale_mul=2.0)
    assert b0.fingerprint != b1.fingerprint
    d0 = TR.MRHDataset(Hp.make_samples(4, b0), b0, Hp.tiny_cfg())
    d1 = TR.MRHDataset(Hp.make_samples(4, b1), b1, Hp.tiny_cfg())
    kw = dict(batch_size=2, seed=0, log=lambda e: None, max_abs_every=1000)
    TR.make_trainer(_tcfg(), d0, run_id="mdl03-c", ckpt_dir=tmp_path / "c", max_steps=2, **kw).fit()
    with pytest.raises(CheckpointMismatchError):
        TR.make_trainer(_tcfg(), d1, run_id="mdl03-c", ckpt_dir=tmp_path / "c", max_steps=4, **kw).fit()
