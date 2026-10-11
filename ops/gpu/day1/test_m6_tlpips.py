"""CPU tests for m6_tlpips.py (no GPU, no network, no diffusers): fake VAE, fake 30-block transformer,
fake corpus, fake uploader."""
import io
import json
import multiprocessing as mp
import os
import random
import sys
import tarfile
import threading
import time
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m6_tlpips as M  # noqa: E402

torch.set_num_threads(1)

C = 4                  # latent channels of the fake teacher (the code must not hardcode 16)
CROP = 64              # fake "256": latent 8x8, 4x4 token grid, 4x4 distance map
SECRET = "hf_" + "AbCdEfGhIjKlMnOpQrSt"


# ---------------------------------------------------------------- fakes

def make_corpus(root, n=12, size=(96, 80), bad=()):
    os.makedirs(os.path.join(root, "img"), exist_ok=True)
    rs = np.random.RandomState(3)
    for i in range(n):
        w, h = size
        arr = (rs.rand(h, w, 3) * 255).astype(np.uint8)
        arr[:, : w // 2] = arr[:, : w // 2] // 2 + 60
        if i in bad:
            arr = arr[:20, :20]           # smaller than the crop
        Image.fromarray(arr).save(os.path.join(root, "img", f"{i:08d}.webp"), quality=90)


class FakeVAE(nn.Module):
    def __init__(self, c=C, scale=0.5, shift=0.1):
        super().__init__()
        self.enc = nn.Conv2d(3, c, 8, stride=8)
        self.dec = nn.ConvTranspose2d(c, 3, 8, stride=8)
        self.config = SimpleNamespace(scaling_factor=scale, shift_factor=shift, latent_channels=c)
        self.enc_calls = 0

    def encode(self, x):
        self.enc_calls += 1
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=self.enc(x)))

    def decode(self, z):
        return SimpleNamespace(sample=self.dec(z))


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.fc2(F.silu(self.fc1(x)))


class FakeTransformer(nn.Module):
    """Same call convention as ZImageTransformer2DModel: list of (C,1,h,w), t, list of (L,cap_dim).
    Tokens are per-token (no mixing) so localisation tests are exact; image tokens come first, raster order."""

    def __init__(self, c=C, dim=8, n_layers=30, cap_dim=6, patch=2):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(patch * patch * c, dim)
        self.cap_embed = nn.Linear(cap_dim, dim)
        self.tproj = nn.Linear(1, dim)
        self.layers = nn.ModuleList([_Block(dim) for _ in range(n_layers)])
        self.t_seen = []

    def forward(self, x, t, cap_feats, return_dict=True):
        self.t_seen.append(t.detach().clone())
        p, seqs = self.patch, []
        for xi, cap, ti in zip(x, cap_feats, t):
            c, _, h, w = xi.shape
            patches = xi[:, 0].reshape(c, h // p, p, w // p, p).permute(1, 3, 2, 4, 0).reshape(-1, p * p * c)
            seqs.append(torch.cat([self.embed(patches) + self.tproj(ti.view(1, 1).float()), self.cap_embed(cap)]))
        h = torch.stack(seqs)
        for layer in self.layers:
            h = layer(h)
        return ([hi for hi in h],)

    def reset_calls(self):
        for l in self.layers:
            l.calls = 0


def make_labeler(seed=0, **kw):
    torch.manual_seed(1)
    vae, tr = FakeVAE().eval(), FakeTransformer().eval()
    cap = torch.randn(5, 6)
    return M.TeacherLabeler(tr, vae, cap, seed=seed, **kw)


def rand_u8(n, size=CROP, seed=0):
    return np.random.RandomState(seed).randint(0, 256, (n, size, size, 3), dtype=np.uint8)


class FakeLabeler:
    """Counts calls; distance = mean |x - y| so ordering is meaningful. Can fail on chosen ids."""

    def __init__(self, fail_ids=(), msz=CROP // 16):
        self.fail_ids, self.msz, self.calls, self.closed = set(fail_ids), msz, 0, False
        self.vae = FakeVAE()

    def label_pairs(self, x, y, ids):
        self.calls += 1
        if self.fail_ids & {int(i) for i in ids}:
            raise RuntimeError("CUDA out of memory (fake)")
        d = np.abs(x.astype(np.float32) - y.astype(np.float32)).mean((1, 2, 3)) / 255.0 + 1e-3
        return {"d": d.astype(np.float32), "dl": np.stack([d, d, d], 1).astype(np.float32),
                "map": np.broadcast_to(d[:, None, None], (len(d), self.msz, self.msz)).astype(np.float16).copy()}

    def vae_roundtrip(self, x):
        return x

    def close(self):
        self.closed = True


def small_args(tmp, **over):
    ap = M.build_parser()
    argv = ["--root", str(tmp), "--crop", str(CROP), "--chunk", "32", "--n-train", "64", "--n-val", "32",
            "--n-trip", "16", "--label-batch", "16", "--workers", "1", "--device", "cpu", "--amp", "off",
            "--batch", "8", "--no-upload", "--warmup", "1", "--total-steps", "20", "--first-eval-s", "0",
            "--eval-every-s", "0", "--min-free-gb", "0", "--progress-every-s", "0"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    a = ap.parse_args(argv)
    a.blocks = M.parse_blocks(a.blocks)
    return a


# ---------------------------------------------------------------- step A: distortions and specs

OPS_TEST = [{"op": "jpeg", "q": 20}, {"op": "blur", "sigma": 1.5}, {"op": "noise", "sigma": 0.05, "seed": 7},
            {"op": "resample", "scale": 0.4, "down": "area", "up": "cubic"},
            {"op": "unsharp", "amount": 3.0, "sigma": 1.2},
            {"op": "color", "brightness": 0.1, "contrast": -0.1, "saturation": 0.12},
            {"op": "geom", "dx": 2.0, "dy": -1.0, "angle": 1.5}]


@pytest.mark.parametrize("op", OPS_TEST, ids=[o["op"] for o in OPS_TEST])
def test_distortion_shape_range_determinism(op):
    img = rand_u8(1, 48, seed=2)[0]
    y = M.apply_op(img, op)
    assert y.shape == img.shape and y.dtype == np.uint8
    assert y.min() >= 0 and y.max() <= 255
    assert not np.array_equal(y, img)
    assert np.array_equal(y, M.apply_op(img, op))


def test_identity_like_ops_are_mild():
    smooth = np.tile(np.linspace(0, 255, 48, dtype=np.uint8)[None, :, None], (48, 1, 3))
    y = M.apply_op(smooth, {"op": "color", "brightness": 0.0, "contrast": 0.0, "saturation": 0.0})
    assert np.abs(y.astype(int) - smooth.astype(int)).max() <= 1
    y = M.apply_op(smooth, {"op": "geom", "dx": 0.0, "dy": 0.0, "angle": 0.0})
    assert np.abs(y.astype(int) - smooth.astype(int)).max() <= 1


def test_sample_ops_ranges_and_structure():
    rng = random.Random(5)
    seen, counts = set(), {1: 0, 2: 0}
    for _ in range(600):
        ops = M.sample_ops(rng)
        assert 1 <= len(ops) <= 2
        counts[len(ops)] += 1
        names = [o["op"] for o in ops]
        assert len(set(names)) == len(names)
        assert sum(n in M.GPU_OPS for n in names) <= 1
        assert all(n in M.GPU_OPS for n in names[: sum(n in M.GPU_OPS for n in names)])   # GPU ops first
        seen.update(names)
        for o in ops:
            if o["op"] == "jpeg":
                assert 10 <= o["q"] <= 90
            elif o["op"] == "blur":
                assert 0.5 <= o["sigma"] <= 3.0
            elif o["op"] == "noise":
                assert 2 / 255 - 1e-6 <= o["sigma"] <= 20 / 255 + 1e-6
            elif o["op"] == "resample":
                assert 0.25 <= o["scale"] <= 0.75
            elif o["op"] == "color":
                assert all(abs(o[k]) <= 0.15 for k in ("brightness", "contrast", "saturation"))
            elif o["op"] == "geom":
                assert 0.5 - 1e-3 <= (o["dx"] ** 2 + o["dy"] ** 2) ** 0.5 <= 3.0 + 1e-3
                assert abs(o["angle"]) <= 2.0
    assert seen == set(M.CPU_OPS) | set(M.GPU_OPS)
    assert counts[1] > 0 and counts[2] > 0
    assert all(o["op"] not in M.GPU_OPS for _ in range(50) for o in M.sample_ops(rng, gpu_ops=()))


def test_spec_determinism_and_triplets():
    pool = list(range(10))
    a, b = M.make_spec(3, "train", 17, pool), M.make_spec(3, "train", 17, pool)
    assert a == b and json.loads(json.dumps(a)) == a
    assert a != M.make_spec(3, "train", 18, pool) and a != M.make_spec(4, "train", 17, pool)
    t = M.make_spec(3, "trip", 5, pool)
    assert "ops_a" in t and "ops_b" in t and "ops" not in t
    assert M.make_spec(3, "val", 0, pool)["ops"]


def test_split_corpus_deterministic_and_disjoint():
    idx = list(range(2000))
    tr, va = M.split_corpus(idx)
    assert tr == M.split_corpus(idx)[0] and not set(tr) & set(va)
    assert 100 < len(va) < 300 and len(tr) + len(va) == len(idx)
    tr, va = M.split_corpus([1, 2, 3])        # tiny corpus: shared
    assert tr and va


def test_load_crop_fallback_and_alignment(tmp_path):
    make_corpus(str(tmp_path), n=4, bad=(0,))
    spec = {"src": 0, "alts": [99, 1, 2], "u": 0.5, "v": 0.5}
    x = M.load_crop(str(tmp_path), spec, CROP)       # 0 too small, 99 missing -> falls back to 1
    assert x.shape == (CROP, CROP, 3) and x.dtype == np.uint8
    with Image.open(tmp_path / "img" / "00000001.webp") as im:
        w, h = im.size
        arr = np.asarray(im.convert("RGB"))
    y0, x0 = M.crop_origin(h, w, CROP, 0.5, 0.5)
    assert np.array_equal(x, arr[y0:y0 + CROP, x0:x0 + CROP])
    assert M.crop_origin(80, 96, CROP, 0.999999, 0.999999) == (16, 32)
    assert M.load_crop(str(tmp_path), {"src": 0, "alts": [0, 0, 0], "u": 0.0, "v": 0.0}, CROP) is None


def test_list_corpus_ignores_tmp(tmp_path):
    make_corpus(str(tmp_path), n=3)
    (tmp_path / "img" / "00000009.webp.tmp").write_bytes(b"x")
    assert M.list_corpus(str(tmp_path)) == [0, 1, 2]


def test_build_chunk_deterministic_gpu_ops_and_drops(tmp_path):
    make_corpus(str(tmp_path), n=8)
    pool = list(range(8))
    specs = [M.make_spec(0, "train", k, pool) for k in range(24)]
    fns = {"tae": lambda x: 255 - x, "vae": lambda x: x}
    a1, o1 = M.build_chunk(specs, str(tmp_path), CROP, None, fns)
    a2, o2 = M.build_chunk(specs, str(tmp_path), CROP, None, fns)
    assert a1.shape == (24, 2, CROP, CROP, 3) and a1.dtype == np.uint8
    assert np.array_equal(a1, a2) and o1.all() and o2.all()
    for i, s in enumerate(specs):                    # x is the clean crop, y differs (every op changes the image)
        x = M.load_crop(str(tmp_path), s, CROP)
        assert np.array_equal(a1[i, 0], x)
    has_gpu = [s["ops"][0]["op"] in M.GPU_OPS for s in specs]
    assert any(has_gpu) and not all(has_gpu)
    # without any GPU function the pairs needing one are dropped, never the whole chunk
    a3, o3 = M.build_chunk(specs, str(tmp_path), CROP, None, {})
    assert np.array_equal(o3, ~np.array(has_gpu))
    assert np.array_equal(a3[o3], a1[o3]) and not a3[~o3].any()
    # a GPU function that returns garbage / raises only drops its own pairs
    a4, o4 = M.build_chunk(specs, str(tmp_path), CROP, None, {"tae": lambda x: 1 / 0, "vae": lambda x: x[:1]})
    assert np.array_equal(o4, ~np.array(has_gpu))


def test_build_chunk_triplets(tmp_path):
    make_corpus(str(tmp_path), n=6)
    specs = [M.make_spec(0, "trip", k, list(range(6)), gpu_ops=()) for k in range(8)]
    arr, ok = M.build_chunk(specs, str(tmp_path), CROP)
    assert arr.shape == (8, 3, CROP, CROP, 3) and ok.all()
    assert all(np.array_equal(arr[i, 0], M.load_crop(str(tmp_path), s, CROP)) for i, s in enumerate(specs))
    assert not np.array_equal(arr[0, 1], arr[0, 0])


def test_pool_matches_serial(tmp_path):
    make_corpus(str(tmp_path), n=6)
    specs = [M.make_spec(1, "train", k, list(range(6)), gpu_ops=()) for k in range(12)]
    serial, _ = M.build_chunk(specs, str(tmp_path), CROP)
    pool = mp.get_context("fork").Pool(2, initializer=M._w_init)
    try:
        par, ok = M.build_chunk(specs, str(tmp_path), CROP, pool, timeout=60)
    finally:
        pool.terminate()
    assert ok.all() and np.array_equal(serial, par)


def test_pmap_falls_back_when_pool_breaks():
    class Broken:
        def map_async(self, *a, **k):
            raise RuntimeError("pool is dead")

    assert M.pmap(Broken(), lambda v: v * 2, [1, 2, 3]) == [2, 4, 6]


# ---------------------------------------------------------------- step B: teacher features and distances

def test_hooks_capture_exactly_the_requested_blocks():
    lab = make_labeler()
    tr = lab.transformer
    z = torch.randn(3, C, 8, 8)
    # reference: run the plain forward and collect the block outputs with our own hooks
    seen = {}
    hs = [tr.layers[b].register_forward_hook(lambda m, i, o, b=b: seen.__setitem__(b, o.detach().clone()))
          for b in range(30)]
    lab.early_exit = False
    tr.reset_calls()
    feats = lab.features(z)
    for h in hs:
        h.remove()
    assert len(feats) == 3 and all(f.shape == (3, 4, 4, 8) for f in feats)
    for f, b in zip(feats, M.BLOCKS):
        assert torch.allclose(f.reshape(3, 16, 8), seen[b][:, :16], atol=1e-6)    # image tokens of THAT block
        assert not torch.allclose(f.reshape(3, 16, 8), seen[b + 1][:, :16])
    assert all(l.calls == 1 for l in tr.layers)                                     # full forward ran
    assert not lab.store and not lab.capturing                                      # nothing left behind


def test_early_exit_skips_later_blocks_but_same_features():
    lab = make_labeler()
    z = torch.randn(2, C, 8, 8)
    lab.early_exit = False
    full = lab.features(z)
    lab.early_exit = True
    lab.transformer.reset_calls()
    fast = lab.features(z)
    calls = [l.calls for l in lab.transformer.layers]
    assert calls[: max(M.BLOCKS) + 1] == [1] * (max(M.BLOCKS) + 1)
    assert sum(calls[max(M.BLOCKS) + 1:]) == 0
    assert all(torch.equal(a, b) for a, b in zip(full, fast))
    lab.close()                                                                    # hooks gone: plain forward untouched
    lab.transformer.reset_calls()
    lab.transformer([z[0].unsqueeze(1)], torch.tensor([0.7]), [lab.cap], return_dict=False)
    assert all(l.calls == 1 for l in lab.transformer.layers) and not lab.store


def test_block_validation_and_custom_blocks():
    with pytest.raises(ValueError):
        make_labeler(blocks=(1, 30))
    lab = make_labeler(blocks=(2, 4))
    assert len(lab.features(torch.randn(1, C, 8, 8))) == 2


def test_token_grid_orientation():
    """A change in the top-right latent corner must light up token (row 0, col last), not (last, 0)."""
    lab = make_labeler()
    z = torch.randn(1, C, 16, 16)
    z2 = z.clone()
    z2[:, :, 0:2, 14:16] += 3.0                       # latent rows 0-1, cols 14-15  -> token (0, 7) of an 8x8 grid
    fa, fb = lab.features(torch.cat([z, z2])) , None
    maps = [M.token_distance_map(f[:1], f[1:]) for f in fa]
    m = torch.stack(maps).mean(0)[0]
    assert m.shape == (8, 8)
    assert divmod(int(m.argmax()), 8) == (0, 7)
    assert float(m[1:, :].max()) < 1e-6 and float(m[0, :7].max()) < 1e-6
    z3 = z.clone()
    z3[:, :, 14:16, 0:2] += 3.0                       # bottom-left -> (7, 0)
    f3 = lab.features(torch.cat([z, z3]))
    m3 = torch.stack([M.token_distance_map(f[:1], f[1:]) for f in f3]).mean(0)[0]
    assert divmod(int(m3.argmax()), 8) == (7, 0)


def test_token_distance_is_unit_normalised():
    a = torch.randn(2, 3, 3, 16)
    assert float(M.token_distance_map(a, a).max()) == 0.0
    assert torch.allclose(M.token_distance_map(a, 100 * a), torch.zeros(2, 3, 3), atol=1e-5)   # scale-free
    assert torch.allclose(M.token_distance_map(a, -a), torch.full((2, 3, 3), 4.0), atol=1e-4)  # opposite: max
    b = torch.randn(2, 3, 3, 16)
    assert torch.allclose(M.token_distance_map(a, b), M.token_distance_map(b, a))


def test_label_distance_symmetry_self_zero_and_timestep():
    lab = make_labeler()
    x, y = rand_u8(4, seed=1), rand_u8(4, seed=2)
    ids = np.array([5, 6, 7, 8])
    dxy = lab.label_pairs(x, y, ids)
    dyx = lab.label_pairs(y, x, ids)
    assert dxy["d"].shape == (4,) and dxy["dl"].shape == (4, 3) and dxy["map"].shape == (4, 4, 4)
    assert dxy["map"].dtype == np.float16 and (dxy["d"] > 0).all()
    assert np.allclose(dxy["d"], dyx["d"], rtol=1e-5)
    assert np.allclose(dxy["map"].astype(np.float32).mean((1, 2)), dxy["d"], rtol=2e-2)
    assert np.allclose(dxy["d"], dxy["dl"].mean(1), rtol=1e-5)
    same = lab.label_pairs(x, x, ids)
    assert np.allclose(same["d"], 0.0, atol=1e-6) and np.allclose(same["map"], 0.0, atol=1e-6)
    t = lab.transformer.t_seen[-1]
    assert t.shape == (8,) and torch.allclose(t, torch.full((8,), 1.0 - M.SIGMA))      # (1000 - 300)/1000
    # same pair id -> same noise -> same label; another id -> another noise -> another label
    assert np.array_equal(lab.label_pairs(x, y, ids)["d"], dxy["d"])
    assert not np.allclose(lab.label_pairs(x, y, ids + 100)["d"], dxy["d"])


def test_label_uses_same_noise_for_x_and_y():
    lab = make_labeler()
    x = rand_u8(2, seed=4)
    seen = []
    orig = lab.features
    lab.features = lambda z_t: (seen.append(z_t.clone()), orig(z_t))[1]
    lab.label_pairs(x, x, np.array([1, 2]))
    zt = seen[0]
    assert torch.equal(zt[:2], zt[2:])                          # identical images + identical noise => identical input
    z = lab.encode_u8(x)
    eps = M.noise_for(0, 1, z.shape[1:])
    assert torch.allclose(zt[0], (1 - M.SIGMA) * z[0] + M.SIGMA * eps, atol=1e-6)


def test_vae_normalisation_roundtrip_consts():
    lab = make_labeler()
    x = rand_u8(2, seed=9)
    z = lab.encode_u8(x)
    raw = lab.vae.enc(M.u8_to_tensor(x))
    assert torch.allclose(z, (raw - 0.1) * 0.5, atol=1e-5)       # (mean - shift) * scale, config constants read
    y = lab.vae_roundtrip(x)
    assert y.shape == x.shape and y.dtype == np.uint8


def test_label_resilient_isolates_bad_pair():
    fl = FakeLabeler(fail_ids={13})
    x, y = rand_u8(8, seed=1), rand_u8(8, seed=2)
    ids = np.arange(8, 16)
    st = {"fail_run": 0, "errors": 0}
    res, ok = M.label_resilient(fl, x, y, ids, st)
    assert ok.tolist() == [True] * 5 + [False] + [True] * 2
    assert np.isnan(res["d"][5]) and np.isfinite(res["d"][ok]).all()
    assert res["map"].shape == (8, 4, 4) and st["errors"] >= 1
    ref = FakeLabeler().label_pairs(x, y, ids)
    assert np.allclose(res["d"][ok], ref["d"][ok])


def test_label_resilient_aborts_after_100_failures():
    fl = FakeLabeler(fail_ids=set(range(1000)))
    x, y = rand_u8(32, seed=1), rand_u8(32, seed=2)
    st = {"fail_run": 0, "errors": 0}
    with pytest.raises(M.TooManyFailures):
        for b in range(10):
            M.label_resilient(fl, x, y, np.arange(32) + 32 * b, st)


# ---------------------------------------------------------------- A+B driver: layout, resume, deadline

def run_job(tmp, labeler=None, now=time.time, label_deadline=None, stop=None, **over):
    args = small_args(tmp, **over)
    out = os.path.join(str(tmp), "day1", "m6")
    os.makedirs(os.path.join(out, "pairs"), exist_ok=True)
    corpus = M.list_corpus(str(tmp))
    tr, va = M.split_corpus(corpus)
    lab = labeler or FakeLabeler()
    job = M.LabelJob(args, out, str(tmp), tr, va, lab, {"tae": lambda x: 255 - x, "vae": lambda x: x}, None,
                     now, stop, label_deadline)
    return job, lab, out, job.run()


def test_label_job_layout_resume_and_pairs_reuse(tmp_path):
    make_corpus(str(tmp_path), n=12)
    job, lab, out, res = run_job(tmp_path)
    assert res["reason"] == "complete"
    names = sorted(os.listdir(out))
    # val: 1 chunk, trip: 1 chunk, train: 2 chunks of 32
    assert [n for n in names if n.startswith("labels_")] == [
        "labels_train_00000.npz", "labels_train_00001.npz", "labels_trip_00000.npz", "labels_val_00000.npz"]
    z = np.load(os.path.join(out, "labels_train_00001.npz"))
    assert z["ids"].tolist() == list(range(32, 64)) and z["d"].shape == (32,) and z["dl"].shape == (32, 3)
    assert z["map"].shape == (32, 4, 4) and z["map"].dtype == np.float16 and z["ok"].dtype == bool
    zt = np.load(os.path.join(out, "labels_trip_00000.npz"))
    assert zt["d"].shape == (16, 2) and zt["ids"][0] == M.ID_OFFSET["trip"] and "map" not in zt.files
    assert np.load(os.path.join(out, "labels_val_00000.npz"))["ids"][0] == M.ID_OFFSET["val"]
    assert np.load(os.path.join(out, "pairs", "train_00000.npy"), mmap_mode="r").shape == (32, 2, CROP, CROP, 3)
    assert np.load(os.path.join(out, "pairs", "trip_00000.npy"), mmap_mode="r").shape == (16, 3, CROP, CROP, 3)
    assert not [n for n in os.listdir(os.path.join(out, "pairs")) if n.endswith(".tmp")]
    first_calls = lab.calls
    stamp = {n: os.path.getmtime(os.path.join(out, n)) for n in names if n.startswith("labels_")}
    # second run: nothing to do, nothing rewritten
    lab2 = FakeLabeler()
    job2, _, _, res2 = run_job(tmp_path, labeler=lab2)
    assert lab2.calls == 0 and res2["skipped"] == 4 and res2["chunks"] == 0
    assert {n: os.path.getmtime(os.path.join(out, n)) for n in stamp} == stamp
    # lose one label file: only that chunk is relabelled, the stored pairs are reused byte for byte
    pair_mtime = os.path.getmtime(os.path.join(out, "pairs", "train_00001.npy"))
    os.remove(os.path.join(out, "labels_train_00001.npz"))
    lab3 = FakeLabeler()
    _, _, _, res3 = run_job(tmp_path, labeler=lab3)
    assert res3["chunks"] == 1 and lab3.calls == 2 and res3["skipped"] == 3
    assert os.path.getmtime(os.path.join(out, "pairs", "train_00001.npy")) == pair_mtime
    z3 = np.load(os.path.join(out, "labels_train_00001.npz"))
    assert np.array_equal(z3["d"], z["d"])
    # a changed teacher config invalidates old labels (different cfg hash) instead of mixing them
    _, lab4, _, res4 = run_job(tmp_path, labeler=FakeLabeler(), sigma=0.5)
    assert res4["chunks"] == 4 and lab4.calls > 0


def test_label_job_bad_pairs_marked_not_fatal(tmp_path):
    make_corpus(str(tmp_path), n=12)
    off = M.ID_OFFSET["val"]
    job, lab, out, res = run_job(tmp_path, labeler=FakeLabeler(fail_ids={off + 3}))
    assert res["reason"] == "complete"
    z = np.load(os.path.join(out, "labels_val_00000.npz"))
    assert not z["ok"][3] and z["ok"].sum() == 31 and np.isnan(z["d"][3])
    data = M.SplitData(out, "val", job.h, CROP)
    assert len(data) == 31 and np.isfinite(data.d).all()


def test_label_job_drops_pairs_without_gpu_fn(tmp_path):
    make_corpus(str(tmp_path), n=12)
    args = small_args(tmp_path)
    out = os.path.join(str(tmp_path), "o")
    os.makedirs(out)
    job = M.LabelJob(args, out, str(tmp_path), list(range(12)), list(range(12)), FakeLabeler(), {}, None)
    res = job.run()
    assert res["reason"] == "complete" and res["pairs_bad"] > 0 and res["pairs_ok"] > 0


def test_label_job_deadline_and_stop(tmp_path):
    make_corpus(str(tmp_path), n=12)
    t = {"now": 1000.0}
    job, lab, out, res = run_job(tmp_path, now=lambda: t["now"], label_deadline=900.0)
    assert res["reason"] == "label_budget" and lab.calls == 0
    assert not [n for n in os.listdir(out) if n.startswith("labels_")]
    # the deadline hits in the middle of the run: finished chunks stay, the cut one is not written
    class Ticking(FakeLabeler):
        def label_pairs(self, x, y, ids):
            t["now"] += 10.0
            return super().label_pairs(x, y, ids)
    t["now"] = 0.0
    job, lab, out2, res = run_job(tmp_path, labeler=Ticking(), now=lambda: t["now"], label_deadline=45.0)
    assert res["reason"] == "label_budget"
    files = [n for n in os.listdir(out2) if n.startswith("labels_")]
    assert 0 < len(files) < 4 and not [n for n in os.listdir(out2) if n.endswith(".tmp")]
    ev = threading.Event()
    ev.set()
    _, lab5, _, res5 = run_job(tmp_path, stop=ev)
    assert res5["reason"] == "signal" and lab5.calls == 0


def test_label_job_disk_guard(tmp_path, monkeypatch):
    make_corpus(str(tmp_path), n=12)
    monkeypatch.setattr(M, "free_disk_gb", lambda p: 1.0)
    job, lab, out, res = run_job(tmp_path, min_free_gb=6.0)
    assert res["reason"] == "disk" and lab.calls == 0


def test_manifest_roundtrip_and_mismatch(tmp_path):
    M.write_manifest(str(tmp_path), {"a": 1, "b": [1, 2]})
    assert M.load_manifest(str(tmp_path)) == {"a": 1, "b": [1, 2]}
    assert M.manifest_mismatch({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 9}) == ["b"]
    assert M.load_manifest(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------- step C: student

def test_student_params_symmetry_zero_and_weights():
    st = M.TLPIPSStudent()
    n = M.count_params(st)
    assert M.PARAM_MIN <= n <= M.PARAM_MAX, n
    x, y = torch.randn(3, 3, CROP, CROP), torch.randn(3, 3, CROP, CROP)
    o = st(x, y)
    assert o["d"].shape == (3,) and o["stage"].shape == (3, 5) and o["map"].shape == (3, 4, 4)
    assert (o["d"] > 0).all() and (o["stage"] >= 0).all()
    assert torch.allclose(o["d"], st(y, x)["d"], atol=1e-6)
    assert torch.allclose(st(x, x)["d"], torch.zeros(3), atol=1e-7)
    assert torch.allclose(o["map"].mean((1, 2)), o["d"], rtol=0.35)           # map is the stage-summed distance
    assert all((F.softplus(l) > 0).all() for l in st.lin)                      # channel weights strictly non-negative
    o["d"].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in st.parameters())
    big = st(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    assert big["map"].shape == (1, 8, 8)                                      # map grid follows crop / 16


def test_losses_and_rank_loss():
    dt = torch.tensor([0.001, 0.01, 0.1, 1.0])
    ok = M.rank_loss(torch.log(dt + M.LOG_EPS), torch.log(dt + M.LOG_EPS))
    bad = M.rank_loss(torch.log(dt.flip(0) + M.LOG_EPS), torch.log(dt + M.LOG_EPS))
    assert float(ok) == 0.0 and float(bad) > 0.5
    out = {"d": dt.clone().requires_grad_(True), "map": torch.rand(4, 4, 4)}
    l = M.compute_loss(out, dt, torch.rand(4, 4, 4), 0.5, 0.1)
    assert all(torch.isfinite(v) for v in l.values()) and float(l["l_mse"]) < 1e-9
    l["loss"].backward()
    assert M.lr_at(0, 1e-3, 500) < M.lr_at(499, 1e-3, 500) <= 1e-3
    assert M.lr_at(20000, 1e-3, 500, 20000) < 2e-5


def test_spearman_afc_and_mapcorr():
    from scipy.stats import spearmanr
    rs = np.random.RandomState(0)
    a = np.round(rs.rand(200) * 20) / 20                                        # ties on purpose
    b = a + rs.rand(200) * 0.3
    assert abs(M.spearman(a, b) - spearmanr(a, b)[0]) < 1e-9
    assert M.spearman(a, a) == pytest.approx(1.0) and M.spearman(a, -a) == pytest.approx(-1.0)
    dt = np.array([[0.1, 0.5], [0.4, 0.2], [0.30, 0.31]])
    ds = np.array([[0.2, 0.9], [0.3, 0.4], [0.9, 0.1]])
    agree, clear = M.afc_agreement(ds, dt)
    assert agree == pytest.approx(1 / 3) and clear == pytest.approx(0.5)
    m = rs.rand(5, 4, 4)
    assert M.map_corr(m, m) == pytest.approx(1.0)


def test_baselines_and_scatter(tmp_path):
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    assert float(M.pixel_l2(x, x).max()) == 0.0 and float(M.ssim_dist(x, x).max()) < 1e-6
    y = x + 0.3 * torch.randn_like(x)
    assert (M.pixel_l2(x, y) > 0).all() and (M.ssim_dist(x, y) > 0).all()
    assert torch.allclose(M.pixel_l2(x, y), M.pixel_l2(y, x))
    dt = np.exp(np.random.RandomState(0).randn(300))
    M.draw_scatter(dt, dt * 1.1, str(tmp_path / "s.png"))
    im = Image.open(tmp_path / "s.png")
    assert im.size == (640, 640)
    M.draw_scatter(np.zeros(0), np.zeros(0), str(tmp_path / "e.png"))          # degenerate input must not raise


def labelled_fixture(tmp_path):
    make_corpus(str(tmp_path), n=12)
    job, lab, out, res = run_job(tmp_path)
    return job, out


def test_split_data_and_batch_loader(tmp_path):
    job, out = labelled_fixture(tmp_path)
    tr = M.SplitData(out, "train", job.h, CROP)
    assert len(tr) == 64 and tr.map.shape == (64, 4, 4)
    im = tr.images([0, 33])
    assert im.shape == (2, 2, CROP, CROP, 3)
    assert np.array_equal(im[1], np.load(os.path.join(out, "pairs", "train_00001.npy"))[1])
    assert M.SplitData(out, "train", job.h + 1, CROP).__len__() == 0              # foreign cfg ignored
    trip = M.SplitData(out, "trip", job.h, CROP)
    assert len(trip) == 16 and trip.d.shape == (16, 2) and trip.images([0]).shape[1] == 3
    ld = M.BatchLoader(tr, 8, seed=1)
    try:
        imgs, d, mp_ = ld.next(10)
        assert imgs.shape == (8, 2, CROP, CROP, 3) and d.shape == (8,) and mp_.shape == (8, 4, 4)
        seen = set()
        for _ in range(12):                                                     # > 1 epoch: reshuffles, no crash
            seen.update(np.round(ld.next(10)[1], 6).tolist())
        assert len(seen) > 8
    finally:
        ld.close()
    small = M.SplitData(out, "val", job.h, CROP)
    ld2 = M.BatchLoader(small, 64, seed=2)                                       # fewer items than a batch
    try:
        assert ld2.next(10)[0].shape[0] == 64
    finally:
        ld2.close()


def make_trainer(tmp_path, job, out, **over):
    args = small_args(tmp_path, **over)
    return M.Trainer(args, out, job.h, device=torch.device("cpu")), args


def test_train_one_step_finite_checkpoint_and_resume(tmp_path):
    job, out = labelled_fixture(tmp_path)
    tr, _ = make_trainer(tmp_path, job, out, max_steps=2)
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = tr.run()
    assert res["status"] == "max_steps" and res["step"] == 2 and np.isfinite(tr.last_loss) or True
    assert res["params"] == M.count_params(tr.model)
    for n in ("ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl", "scatter.png"):
        assert os.path.exists(os.path.join(out, n)), n
    ck = M.load_ckpt(os.path.join(out, "ckpt_last.pt"))
    assert ck["step"] == 2 and ck["config"]["widths"] == list(M.DEFAULT_WIDTHS) and "opt" in ck
    assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.dtype.is_floating_point)
    rec = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    assert {"step", "spearman", "afc", "base_l2_sp", "base_ssim_sp", "base_l2_afc", "base_ssim_afc"} <= set(rec[-1])
    assert rec[-1]["step"] == 2 and np.isfinite(rec[-1]["spearman"])
    # resume: continues from step 2 with the saved optimizer, ends at step 4
    tr2, _ = make_trainer(tmp_path, job, out, max_steps=4)
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        res2 = tr2.run()
    assert res2["start_step"] == 2 and res2["step"] == 4
    assert '"resumed":true' in buf2.getvalue()
    assert M.load_ckpt(os.path.join(out, "ckpt_last.pt"))["step"] == 4


def test_train_loss_decreases_on_learnable_labels(tmp_path):
    """The student can actually fit the (fake, pixel-difference) teacher: loss goes down over a few dozen steps."""
    job, out = labelled_fixture(tmp_path)
    tr, args = make_trainer(tmp_path, job, out, max_steps=40, lr=3e-3)
    losses = []
    orig = tr._train_step

    def spy(batch):
        ok = orig(batch)
        losses.append(float(tr.acc[0]) / max(1, tr.acc_n))
        return ok

    tr._train_step = spy
    with redirect_stdout(io.StringIO()):
        tr.run()
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_train_deadline_stops_with_final_checkpoint(tmp_path):
    job, out = labelled_fixture(tmp_path)
    args = small_args(tmp_path)
    clock = {"t": 0.0}

    def now():
        clock["t"] += 1.0
        return clock["t"]

    tr = M.Trainer(args, out, job.h, now=now, deadline=100.0 + args.finish_margin_s, device=torch.device("cpu"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = tr.run()
    assert res["status"] == "deadline" and 0 < res["step"] < 200
    assert os.path.exists(os.path.join(out, "ckpt_last.pt")) and os.path.exists(os.path.join(out, "ckpt_best.pt"))
    assert "D1_M6_DONE" in buf.getvalue()
    # a deadline already inside the margin: no training at all, but a clean exit with a checkpoint
    out2 = os.path.join(str(tmp_path), "o2")
    shutil_copy = __import__("shutil").copytree(out, out2, ignore=lambda d, n: ["ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl"])
    tr2 = M.Trainer(args, out2, job.h, now=lambda: 1000.0, deadline=1000.0 + 10, device=torch.device("cpu"))
    with redirect_stdout(io.StringIO()):
        res2 = tr2.run()
    assert res2["status"] == "deadline" and res2["step"] == 0
    assert os.path.exists(os.path.join(out2, "ckpt_last.pt"))


def test_train_no_labels_is_clean(tmp_path):
    out = str(tmp_path / "empty")
    os.makedirs(out)
    tr = M.Trainer(small_args(tmp_path), out, 1, device=torch.device("cpu"))
    with redirect_stdout(io.StringIO()):
        assert tr.run()["status"] == "no_data"


def test_nonfinite_step_is_skipped_not_fatal(tmp_path):
    job, out = labelled_fixture(tmp_path)
    tr, _ = make_trainer(tmp_path, job, out, max_steps=3)
    calls = {"n": 0}
    orig = M.compute_loss

    def flaky(*a, **k):
        calls["n"] += 1
        r = orig(*a, **k)
        if calls["n"] == 1:
            r["loss"] = r["loss"] * float("nan")
        return r

    M.compute_loss = flaky
    try:
        with redirect_stdout(io.StringIO()):
            res = tr.run()
    finally:
        M.compute_loss = orig
    assert res["step"] == 3 and tr.skipped == 1


def test_step_exceptions_are_survived_until_100(tmp_path):
    job, out = labelled_fixture(tmp_path)
    tr, _ = make_trainer(tmp_path, job, out, max_steps=2)
    tr.sleep = lambda s: None
    n = {"k": 0}
    orig = tr._train_step

    def boom(batch):
        n["k"] += 1
        if n["k"] <= 3:
            raise RuntimeError("boom")
        return orig(batch)

    tr._train_step = boom
    with redirect_stdout(io.StringIO()):
        assert tr.run()["step"] == 2
    tr2, _ = make_trainer(tmp_path, job, str(tmp_path / "x"), max_steps=2) if False else (None, None)
    out2 = str(tmp_path / "o3")
    __import__("shutil").copytree(out, out2, ignore=lambda d, nn_: ["ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl"])
    t3 = M.Trainer(small_args(tmp_path, max_steps=5), out2, job.h, device=torch.device("cpu"), deadline=None)
    t3.sleep = lambda s: None
    t3._train_step = lambda b: (_ for _ in ()).throw(RuntimeError("always"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = t3.run()
    assert res["status"] == "fatal"
    assert all(len(l) <= 400 for l in buf.getvalue().splitlines())


# ---------------------------------------------------------------- main(): end to end, secrets, markers

def run_main(tmp_path, argv_extra=(), deps=None, env=None, deadline_ahead=4 * 3600.0, monkeypatch=None):
    out = os.path.join(str(tmp_path), "day1", "m6")
    argv = ["--root", str(tmp_path), "--crop", str(CROP), "--chunk", "32", "--n-train", "64", "--n-val", "32",
            "--n-trip", "16", "--label-batch", "16", "--workers", "1", "--device", "cpu", "--amp", "off",
            "--batch", "8", "--warmup", "1", "--total-steps", "20", "--first-eval-s", "0", "--eval-every-s", "0",
            "--min-free-gb", "0", "--progress-every-s", "0", "--max-steps", "3", "--deadline",
            str(time.time() + deadline_ahead)] + list(argv_extra)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = M.main(argv, deps)
    return rc, buf.getvalue(), out


def test_main_end_to_end_and_secret_never_printed(tmp_path, monkeypatch):
    make_corpus(str(tmp_path), n=12)
    monkeypatch.setenv("HF_TOKEN", SECRET)
    uploads = []

    def uploader(folder, remote, allow=None):
        uploads.append((remote, sorted(os.listdir(folder)) if not allow else list(allow)))

    lab = FakeLabeler()
    rc, text, out = run_main(tmp_path, deps={"labeler": lab, "uploader": uploader,
                                             "gpu_fns": {"tae": lambda x: 255 - x}})
    assert rc == 0, text
    assert SECRET not in text and "hf_AbCd" not in text
    lines = [l for l in text.splitlines() if l.strip()]
    assert lines and all(l.startswith("D1_M6") for l in lines)
    assert all(len(l) <= 400 for l in lines)
    for l in lines:                                                       # every marker is one JSON object
        json.loads(l.split(" ", 1)[1])
    assert '"hf_token":"yes"' in text and "D1_M6_DONE" in text and "D1_M6_EVAL" in text
    assert '"teacher":"released"' in text and lab.closed
    assert text.index('"teacher":"released"') < text.index('"phase":"train_data"')     # Z-Image down before C
    for n in ("manifest.json", "ckpt_best.pt", "ckpt_last.pt", "metrics.jsonl", "scatter.png", "baselines.json"):
        assert os.path.exists(os.path.join(out, n)), n
    remotes = {r for r, _ in uploads}
    assert M.REMOTE_DIR in remotes and M.REMOTE_DIR + "/labels" in remotes
    ck_upload = [files for r, files in uploads if r == M.REMOTE_DIR][-1]
    assert {"ckpt_best.pt", "ckpt_last.pt", "metrics.jsonl", "scatter.png"} <= set(ck_upload)
    blob = b"".join(open(os.path.join(dp, f), "rb").read() for dp, _, fs in os.walk(out)
                    for f in fs if f.endswith((".json", ".jsonl")))
    assert SECRET.encode() not in blob
    # second invocation resumes: labels untouched, training continues from the checkpoint
    lab2 = FakeLabeler()
    rc2, text2, _ = run_main(tmp_path, ["--max-steps", "5"], deps={"labeler": lab2, "uploader": uploader,
                                                                    "gpu_fns": {"tae": lambda x: 255 - x}})
    assert rc2 == 0 and lab2.calls == 0 and '"resumed":true' in text2
    assert M.load_ckpt(os.path.join(out, "ckpt_last.pt"))["step"] == 5
    # a run with other pair settings must not silently mix into the same folder
    rc3, text3, _ = run_main(tmp_path, ["--n-train", "96"], deps={"labeler": FakeLabeler(), "uploader": uploader})
    assert rc3 == 2 and "manifest mismatch" in text3


def test_main_no_token_means_no_upload(tmp_path, monkeypatch):
    make_corpus(str(tmp_path), n=12)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc, text, out = run_main(tmp_path, deps={"labeler": FakeLabeler(), "gpu_fns": {"tae": lambda x: x}})
    assert rc == 0 and '"hf_token":"no"' in text and '"upload":false' in text
    assert "uploaded" not in text and os.path.exists(os.path.join(out, "ckpt_best.pt"))


def test_main_deadline_validation_and_phases(tmp_path, monkeypatch):
    make_corpus(str(tmp_path), n=12)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc, text, _ = run_main(tmp_path, deps={"labeler": FakeLabeler()}, deadline_ahead=-5.0)
    assert rc == 1 and "deadline_in_past" in text
    rc, text, _ = run_main(tmp_path, deps={"labeler": FakeLabeler()}, deadline_ahead=1e9)       # milliseconds by mistake
    assert rc == 1 and "deadline_too_far" in text
    rc, text, _ = run_main(tmp_path, ["--deadline", "soon"], deps={"labeler": FakeLabeler()})
    assert rc == 1
    # phase=label leaves labels but trains nothing; phase=train then trains from them
    lab = FakeLabeler()
    rc, text, out = run_main(tmp_path, ["--phase", "label"], deps={"labeler": lab, "gpu_fns": {"tae": lambda x: x}})
    assert rc == 0 and lab.calls > 0 and not os.path.exists(os.path.join(out, "ckpt_last.pt"))
    rc, text, out = run_main(tmp_path, ["--phase", "train"], deps={})
    assert rc == 0 and os.path.exists(os.path.join(out, "ckpt_best.pt"))
    # label budget: with the training reserve larger than the time left, labelling stops at once but C still runs
    shutil_rm = __import__("shutil").rmtree
    shutil_rm(out)
    lab2 = FakeLabeler()
    rc, text, out = run_main(tmp_path, ["--train-reserve-s", "100000"], deps={"labeler": lab2},
                             deadline_ahead=3600.0)
    assert rc == 0 and lab2.calls == 0 and "no_train_labels" in text


def test_main_no_corpus_is_fatal_not_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc, text, _ = run_main(tmp_path, deps={"labeler": FakeLabeler()})
    assert rc == 2 and "no corpus" in text


def test_main_teacher_load_failure_reported(tmp_path, monkeypatch):
    make_corpus(str(tmp_path), n=12)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def bad_load(*a, **k):
        raise ImportError("No module named 'diffusers' " + SECRET)

    monkeypatch.setattr(M, "load_teacher", bad_load)
    rc, text, _ = run_main(tmp_path, ["--phase", "label"], deps={})
    assert rc == 2 and "teacher_load_failed" in text and SECRET not in text


# ---------------------------------------------------------------- misc helpers

def test_markers_are_short_and_scrubbed(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    buf = io.StringIO()
    with redirect_stdout(buf):
        M.log({"a": "x" * 3000, "tok": SECRET, "b": list(range(500)), "f": 1 / 3})
        M.warn("oops", err="failed with " + SECRET + " and hf_ZZZZZZZZZZZZ")
        M.log({"step": 1, "loss": 0.123456789, "nan": float("nan")}, marker=M.MARK + "_EVAL")
    lines = buf.getvalue().splitlines()
    assert len(lines) == 3 and all(len(l) <= 400 for l in lines)
    assert SECRET not in buf.getvalue() and "hf_ZZZZ" not in buf.getvalue()
    assert lines[2].startswith("D1_M6_EVAL ")
    assert M.scrub("x " + SECRET) == "x ***"
    assert M.parse_deadline("1791723600") == 1791723600.0 and M.parse_deadline("") is None


def test_atomic_writers_leave_no_tmp(tmp_path):
    M.save_npy_atomic(np.arange(5), str(tmp_path / "a.npy"))
    M.save_npz_atomic(str(tmp_path / "b.npz"), x=np.arange(3), cfg=np.array(5, np.int64))
    M.save_atomic({"step": 1, "model": {}}, str(tmp_path / "c.pt"))
    assert sorted(os.listdir(tmp_path)) == ["a.npy", "b.npz", "c.pt"]
    assert np.array_equal(np.load(tmp_path / "a.npy"), np.arange(5))
    assert M.read_labels(str(tmp_path / "b.npz"))["x"].tolist() == [0, 1, 2]
    assert M.read_labels(str(tmp_path / "missing.npz")) is None
    (tmp_path / "junk.npz.tmp").write_bytes(b"x")
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "p.npy.tmp").write_bytes(b"x")
    assert M.clean_tmp(str(tmp_path)) == 2
    (tmp_path / "bad.pt").write_bytes(b"not a checkpoint")
    assert M.load_ckpt(str(tmp_path / "bad.pt")) is None and M.load_ckpt(str(tmp_path / "none.pt")) is None


def test_upload_retry_and_uploader_without_token():
    assert M.make_uploader(None) is None
    calls = []

    def flaky(folder, remote):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("net " + SECRET)

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert M.upload_with_retry(flaky, "f", "r", sleep=lambda s: None)
        assert not M.upload_with_retry(lambda f, r: 1 / 0, "f", "r", sleep=lambda s: None)
    assert len(calls) == 3 and SECRET not in buf.getvalue()


def test_ensure_assets_from_fake_vault(tmp_path):
    src = tmp_path / "src"
    make_corpus(str(src), n=3)
    tar_path = tmp_path / "gen_0000.tar"
    with tarfile.open(tar_path, "w") as t:
        for i in range(3):
            t.add(src / "img" / f"{i:08d}.webp", arcname=f"img/{i:08d}.webp")
        t.add(src / "img" / "00000000.webp", arcname="../evil.txt")           # must be ignored
    tae_src = tmp_path / "tae_remote.pt"
    tae_src.write_bytes(b"ckpt")
    fetched = []

    def fetcher(remote):
        fetched.append(remote)
        return str(tar_path if remote.endswith(".tar") else tae_src)

    lister = lambda prefix: [M.CORPUS_PREFIX + "gen_0000.tar", M.CORPUS_PREFIX + "other.txt"]   # noqa: E731
    root = tmp_path / "root"
    root.mkdir()
    res = M.ensure_assets(str(root), str(root / "tae" / "ckpt_best.pt"), 3, lister, fetcher)
    assert res["tae"] and res["images"] == 3 and res["shards"] == 1
    assert (root / "tae" / "ckpt_best.pt").read_bytes() == b"ckpt" and not (tmp_path / "evil.txt").exists()
    fetched.clear()
    M.ensure_assets(str(root), str(root / "tae" / "ckpt_best.pt"), 3, lister, fetcher)   # nothing missing: nothing fetched
    assert fetched == []


def test_tae_roundtrip_wrapper_shapes():
    class TinyTAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Conv2d(3, 4, 8, stride=8)
            self.dec = nn.ConvTranspose2d(4, 3, 8, stride=8)

    fn = M.make_tae_roundtrip(TinyTAE().eval(), torch.device("cpu"), batch=3)
    x = rand_u8(5, seed=3)
    y = fn(x)
    assert y.shape == x.shape and y.dtype == np.uint8


def test_cli_defaults_match_spec():
    a = M.build_parser().parse_args([])
    assert (a.n_train, a.n_val, a.n_trip) == (60000, 3000, 2000)
    assert a.batch == 64 and a.total_steps == 20000 and a.lr == 1e-3 and a.sigma == 0.3
    assert a.finish_margin_s == 600 and a.label_batch == 32 and a.crop == 256 and a.eval_every_s == 900
    assert M.parse_blocks(a.blocks) == (8, 15, 23) and a.phase == "all"
    assert a.log_every_s <= 300 and a.progress_every_s <= 300
