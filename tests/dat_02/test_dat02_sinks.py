"""하류 이음매: 링 push, DAT-05 샤드 왕복, DAT-01 렌더 연동."""
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

import dat02_fakes as F
from buc.data import realify as R
from buc.data import shards as S
from buc.gov import prov as P
from buc.teacher import taps as T


def _items(n=3):
    return [R.RealifyItem(key=f"r{i}", image=F.smooth_image(20 + i), prov=F.ENGINE, caption=f"room {i}",
                          labels={"depth": np.linspace(1, 2, F.IMG * F.IMG, dtype=np.float32).reshape(F.IMG, F.IMG),
                                  "sem": np.full((F.IMG, F.IMG), i, np.int16)})
            for i in range(n)]


def _run(rig, its, **kw):
    t, s, bus = rig
    kw.setdefault("curve", R.provisional_curve(s.table, 6))
    return R.realify(s, bus, its, sigma=0.5, seed=5, token_offset=F.OFF, **kw)


def test_ring_gets_only_kept_per_item_tapsets():
    rig = F.make_rig()
    ring = T.TapRing(F.basis(), max_entries=16)
    res = _run(rig, _items(), ring=ring, tau=10.0)
    assert all(r.keep for r in res)
    assert len(ring) == 3
    assert rig[2].stats["pushed"] == 0                 # 버스의 배치 링 경로를 쓰지 않는다(항목별 push)
    seqs = [r.ring_seq for r in res]
    assert seqs == sorted(seqs) and len(set(seqs)) == 3
    for r in res:
        got = ring.get(r.ring_seq)
        assert got.key == r.key
        assert P.as_record(got.prov) == r.prov
        for l in (8, 15, 23):
            assert torch.equal(got.layers[l], r.taps.layers[l])
    res2 = _run(rig, _items(), ring=ring, tau=0.0)                  # 잡음 교사: err>0 → 전부 거부
    assert all((not r.keep) and r.ring_seq is None for r in res2)
    assert len(ring) == 3


def test_shard_roundtrip(tmp_path):
    rig = F.make_rig()
    its = _items()
    res = _run(rig, its, tau=10.0)
    rej = _run(rig, [R.RealifyItem(key="rej", image=F.smooth_image(1), prov=F.ENGINE, caption="x")], tau=0.0)
    with S.ShardWriter(tmp_path / "out") as w:
        n = R.write_results(w, list(res) + list(rej))
    assert n == 3
    infos = w.infos
    assert sum(i.n for i in infos) == 3
    got = {smp.key: smp for i in infos for smp in S.iter_shard(i.path)}
    assert set(got) == {"r0", "r1", "r2"}
    for r in res:
        smp = got[r.key]
        assert set(smp.fields) == {"rgb.png", "tap8.npy", "tap15.npy", "tap23.npy", "depth.npy", "sem.npy",
                                   "meta.json"}
        assert smp.prov == r.prov
        png = np.asarray(Image.open(io.BytesIO(smp.fields["rgb.png"])).convert("RGB"))
        assert png.dtype == np.uint8 and png.shape == (F.IMG, F.IMG, 3)
        np.testing.assert_array_equal(png, np.round(r.image * 255).astype(np.uint8))
        for l in (8, 15, 23):
            a = S.npy_decode(smp.fields[f"tap{l}.npy"])
            assert a.dtype == np.int8 and a.shape == (F.K, *F.GRID)
            np.testing.assert_array_equal(a, r.taps.layers[l][0].numpy())
        np.testing.assert_array_equal(S.npy_decode(smp.fields["sem.npy"]), r.labels["sem"])
        raw = smp.fields["meta.json"]
        meta = json.loads(raw)
        assert raw == json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
        assert meta["v"] == R.REALIFY_TAG and meta["key"] == r.key
        assert meta["sigma"] == r.sigma and meta["tap_sigma"] == r.tap_sigma and meta["steps"] == r.steps
        assert meta["seed"] == r.seed and meta["band_valid"] == list(r.band_valid) and meta["n_oct"] == 6
        assert meta["lowband_err"] == pytest.approx(r.lowband_err) and meta["tau"] == 10.0
        assert meta["d_thresh"] == R.D_THRESH
        assert meta["curve_key"] == r.curve_key and meta["table_key"] == r.table_key
        assert meta["basis_fp"] == F.basis().fingerprint.hex()
        assert meta["grid"] == list(F.GRID) and meta["tap_layers"] == [8, 15, 23] and meta["keep"] is True


def test_shard_fields_deterministic():
    a = _run(F.make_rig(), _items(1), tau=10.0)[0].shard_fields()
    b = _run(F.make_rig(), _items(1), tau=10.0)[0].shard_fields()
    assert a == b and all(isinstance(v, bytes) for v in a.values())


def test_item_from_synth_end_to_end():
    from buc.data.synth_render import render_room
    smp = render_room(3, width=64, height=48)
    rgb_before = smp.rgb.copy()
    it = R.item_from_synth(smp, key="room3", caption="a photo of a room")
    assert it.key == "room3" and it.caption == "a photo of a room"
    assert it.image.dtype == np.float32 and it.image.shape == (48, 64, 3)
    np.testing.assert_array_equal(it.image, np.clip(smp.rgb_display(), 0, 1).astype(np.float32))
    assert set(it.labels) == {"gbuf", "depth", "sem", "inst"}
    for k in it.labels:
        np.testing.assert_array_equal(it.labels[k], getattr(smp, k))
        assert not np.shares_memory(it.labels[k], getattr(smp, k))
    assert P.as_record(it.prov) == P.as_record(smp.prov)
    t, s, bus = F.make_rig()
    r = R.realify(s, bus, [it], curve=R.provisional_curve(s.table, 5), sigma=0.5, seed=0, token_offset=F.OFF,
                  tau=10.0)[0]
    assert r.image.shape == (48, 64, 3)
    assert r.taps.grid == (3, 4) and tuple(r.taps.layers[8].shape) == (1, F.K, 3, 4)
    assert np.array_equal(smp.rgb, rgb_before)
    assert P.train_ok(r.prov)
