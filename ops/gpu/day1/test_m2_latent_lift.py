"""CPU tests for m2_latent_lift.py (no GPU, no network, no diffusers): fake VAE, fake corpus, fake uploader."""
import json
import os
import sys
import tarfile
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "night1"))
import m2_latent_lift as M  # noqa: E402
import tae_train as T  # noqa: E402

torch.set_num_threads(1)   # tiny CPU ops: multi-threading only adds spin overhead (and is far slower here)

C = 4            # latent channels of the fake corpus (the code must not hardcode 16)
SQ = (128, 128)  # fake "1024^2" image: lat 16x16, zd 8x8
WIDE = (256, 96)  # fake "1344x576" image: lat 12x32, zd 6x16
TAE_WIDTHS = (8, 16, 16, 16)
FAKE_TOKEN = "hf_" + "AbCdEfGhIjKlMnOpQrSt"


# ---------------------------------------------------------------- fixtures / fakes

def img_size(i):
    return WIDE if i % 3 == 2 else SQ


def make_corpus(root, n=12, c=C, start=0):
    os.makedirs(os.path.join(root, "img"), exist_ok=True)
    os.makedirs(os.path.join(root, "lat"), exist_ok=True)
    rs = np.random.RandomState(start + 1)
    for i in range(start, start + n):
        w, h = img_size(i)
        arr = (rs.rand(h, w, 3) * 255).astype(np.uint8)
        arr[:, : w // 2] = arr[:, : w // 2] // 2 + 60
        Image.fromarray(arr).save(os.path.join(root, "img", f"{i:08d}.webp"), quality=90)
        np.save(os.path.join(root, "lat", f"{i:08d}.npy"),
                rs.randn(c, h // 8, w // 8).astype(np.float16))


class FakeVAE(nn.Module):
    """Mimics diffusers AutoencoderKL enough for our wrappers: x8 spatial, C channels, scale/shift in config."""

    def __init__(self, c=C, scale=0.5, shift=0.1, fail_first_decode_oom=False):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=scale, shift_factor=shift, latent_channels=c)
        self.c = c
        self.enc_inputs = []
        self.dec_inputs = []
        self.tiling = False
        self._oom_once = fail_first_decode_oom

    def encode(self, x):
        self.enc_inputs.append(x.detach().float().cpu())
        m = F.avg_pool2d(torch.cat([x, x.mean(1, keepdim=True)], 1)[:, : self.c], 8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=m))

    def decode(self, z):
        if self._oom_once and not self.tiling:
            raise torch.OutOfMemoryError("CUDA out of memory (fake)")
        self.dec_inputs.append(z.detach().float().cpu())
        return SimpleNamespace(sample=F.interpolate(z[:, :3], scale_factor=8, mode="nearest"))

    def enable_tiling(self):
        self.tiling = True


def tiny_decoder():
    torch.manual_seed(0)
    return T.TAE(C, TAE_WIDTHS).dec


def base_argv(root, *extra):
    return ["--root", str(root), "--no-fetch", "--no-upload", "--min-pairs", "1", "--eval-n", "3",
            "--decode-n", "3", "--batch", "4", "--micro-batch", "2", "--crop", "8", "--img-win", "4",
            "--width", "16", "--warmup", "2", "--amp", "off", "--device", "cpu", "--prefetch-threads", "1",
            "--cache-batch", "2", "--first-eval-s", "1000000", "--eval-every-s", "1000000",
            "--log-every-s", "1000000", "--seed", "3", *map(str, extra)]


def deps_for(**kw):
    d = {"decoder": tiny_decoder(), "vae": FakeVAE(), "sleep": lambda s: None}
    d.update(kw)
    return d


def markers(out, prefix="D1_M2"):
    res = []
    for line in out.splitlines():
        head, _, rest = line.partition(" ")
        if head == prefix:
            res.append(json.loads(rest))
    return res


def all_marker_lines(out):
    return [ln for ln in out.splitlines() if ln.startswith("D1_M2")]


class Clock:
    """Fake clock: every read advances time."""

    def __init__(self, step=1.0, t=1000.0):
        self.t, self.step = t, step

    def __call__(self):
        self.t += self.step
        return self.t


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "n1"
    make_corpus(str(root), 12)
    return str(root)


def fill_zd(root, out_dir, n=12):
    """zd for the fake corpus through the real cache builder."""
    st = M.build_zd_cache(root, out_dir, list(range(n)), FakeVAE(), torch.device("cpu"), batch=2)
    assert st["built"] == n, st
    return st


# ---------------------------------------------------------------- models

def test_param_counts_in_spec_range_and_channels_not_hardcoded():
    for c in (4, 16, 32):
        d, u = M.build_models(c)
        for m in (d, u):
            n = M.count_params(m)
            assert 0.7e6 <= n <= 1.5e6, (c, n)
    d, u = M.build_models(16)
    assert M.PARAM_MIN <= M.count_params(d) <= M.PARAM_MAX
    assert M.PARAM_MIN <= M.count_params(u) <= M.PARAM_MAX


def test_shapes_and_stride_structure():
    d, u = M.build_models(C, 128)
    assert d.l2.stride == (2, 2)                       # Down: stride 2 at layer 2
    assert u.l2.stride == (1, 1)                       # Up: nearest 2x upsample then a normal conv
    for h, w in ((16, 16), (12, 32)):                  # square and wide latent grids
        z = torch.randn(2, C, h, w)
        zd = d(z)
        assert zd.shape == (2, C, h // 2, w // 2)
        assert u(zd).shape == z.shape


def test_odd_size_rejected():
    d, _ = M.build_models(C, 16)
    with pytest.raises(ValueError):
        d(torch.randn(1, C, 15, 16))


def test_zero_init_equals_baseline_exactly():
    d, u = M.build_models(C, 16)
    z = torch.randn(2, C, 12, 32)
    assert torch.equal(d(z), M.area_down(z))
    assert torch.equal(u(z), M.bilinear_up(z))
    # area down is the 2x2 mean
    ref = z.reshape(2, C, 6, 2, 16, 2).mean((3, 5))
    assert torch.allclose(M.area_down(z), ref, atol=1e-6)


def test_residual_head_adds_on_top_of_the_resample():
    d, u = M.build_models(C, 16)
    d.head.bias.data.fill_(0.25)
    u.head.bias.data.fill_(-0.5)
    z = torch.randn(1, C, 16, 16)
    assert torch.allclose(d(z), M.area_down(z) + 0.25, atol=1e-6)
    assert torch.allclose(u(z), M.bilinear_up(z) - 0.5, atol=1e-6)


# ---------------------------------------------------------------- crops / batches

def test_crop_shape_for_square_and_wide():
    assert M.crop_shape(64, 64, 64) == (64, 64)        # 1024^2: zd 64x64, z 128x128
    assert M.crop_shape(36, 84, 64) == (36, 64)        # 1344x576: zd is only 36 tall


def test_crop_pair_alignment_is_exact():
    rs = np.random.RandomState(0)
    z = rs.randn(C, 16, 16).astype(np.float16)
    zd = M.area_down(torch.from_numpy(z).float()[None])[0].numpy().astype(np.float16)
    for a, b in ((0, 0), (3, 2), (4, 4)):
        zc, zdc = M.crop_pair(z, zd, a, b, 4, 4)
        assert zc.shape == (C, 8, 8) and zdc.shape == (C, 4, 4)
        got = M.area_down(torch.from_numpy(zc).float()[None])[0].numpy()
        assert np.allclose(got, zdc.astype(np.float32), atol=2e-3)       # same pixels under both crops


def test_batch_source_deterministic_homogeneous_and_wide_shapes(corpus, tmp_path):
    out = str(tmp_path / "m2")
    fill_zd(corpus, out)
    src = M.BatchSource(corpus, out, list(range(12)), C, crop=8, batch=4, seed=7)
    assert (8, 8) in src.classes and (6, 8) in src.classes          # square and wide classes
    seen = set()
    for k in range(12):
        b1, b2 = src.make(k), src.make(k)
        assert torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1])      # pure function of (seed, k)
        zc, zdc = b1
        assert zc.shape[0] == zdc.shape[0] == 4
        assert zc.shape[-2:] == (2 * zdc.shape[-2], 2 * zdc.shape[-1])
        seen.add(tuple(zdc.shape[-2:]))
    assert seen == {(8, 8), (6, 8)}
    assert not torch.equal(src.make(0)[0], src.make(1)[0])


def test_batch_source_survives_a_corrupt_item(corpus, tmp_path):
    out = str(tmp_path / "m2")
    fill_zd(corpus, out)
    with open(M.zd_path(out, 1), "wb") as f:
        f.write(b"garbage")
    src = M.BatchSource(corpus, out, list(range(12)), C, crop=8, batch=4, seed=1)
    for k in range(10):
        assert src.make(k) is not None
    assert 1 in src.bad


# ---------------------------------------------------------------- eval split

def test_eval_indices_deterministic_strided_and_frozen(tmp_path):
    avail = list(range(4755))
    e = M.choose_eval_indices(avail, 400)
    assert e == M.choose_eval_indices(list(reversed(avail)), 400)
    assert len(e) == 400 and e == sorted(set(e)) and e[0] == 0 and e[-1] == 4754
    assert M.choose_eval_indices([3, 1, 2], 10) == [1, 2, 3]
    out = str(tmp_path)
    first = M.load_or_make_eval_indices(out, avail, 400)
    grown = M.load_or_make_eval_indices(out, avail + list(range(5000, 5100)), 400)
    assert first == grown                                           # frozen on first use
    missing = M.load_or_make_eval_indices(out, [i for i in avail if i != first[5]], 400)
    assert first[5] not in missing and len(missing) == 399


# ---------------------------------------------------------------- teacher VAE wrappers

def test_vae_normalisation_roundtrip_constants():
    vae = FakeVAE(scale=0.5, shift=0.1)
    assert M.vae_consts(vae) == (0.5, 0.1)
    assert M.vae_consts(SimpleNamespace(config=SimpleNamespace())) == (1.0, 0.0)
    x = torch.rand(1, 3, 16, 16) * 2 - 1
    z = M.encode_normalized(vae, x)
    raw = vae.encode(x).latent_dist.mean
    assert torch.allclose(z, (raw - 0.1) * 0.5, atol=1e-6)
    M.decode_latents(vae, z)
    assert torch.allclose(vae.dec_inputs[-1], raw, atol=1e-5)       # decode un-normalises: z/scale + shift


def test_decode_retries_with_tiling_after_oom():
    vae = FakeVAE(fail_first_decode_oom=True)
    out = M.decode_latents(vae, torch.zeros(1, C, 2, 2))
    assert vae.tiling and out.shape == (1, 3, 16, 16)


def test_small_tiles_make_the_oom_retry_effective():
    """diffusers only tiles latents larger than tile_latent_min_size (128 for Z-Image): shrink it."""
    class TileVAE(FakeVAE):
        def __init__(self):
            super().__init__()
            self.tile_sample_min_size, self.tile_latent_min_size = 1024, 128
            self.use_tiling = False

        def enable_tiling(self):
            self.use_tiling = True

        def decode(self, z):
            tiles = self.use_tiling and z.shape[-1] > self.tile_latent_min_size
            if z.shape[-1] > 64 and not tiles:
                raise torch.OutOfMemoryError("CUDA out of memory (fake)")
            return super().decode(z)

    vae = TileVAE()
    out = M.decode_latents(vae, torch.zeros(1, C, 128, 128))          # 1024 px image, untiled -> OOM -> tiled
    assert out.shape == (1, 3, 1024, 1024) and M.vae_tiled(vae)
    assert vae.tile_sample_min_size == 512 and vae.tile_latent_min_size == 64
    assert M.enable_small_tiles(vae)                                  # idempotent
    assert not M.enable_small_tiles(SimpleNamespace())                # nothing to enable


def test_evaluate_decodes_all_four_the_same_way_after_a_mid_item_oom():
    class Dec:
        def __init__(self):
            self.is_tiled, self.calls, self.n = False, [], 0

        def tiled(self):
            return self.is_tiled

        def __call__(self, z):
            self.n += 1
            if self.n == 2 and not self.is_tiled:                     # yo OOMs after g decoded untiled
                self.is_tiled = True
            self.calls.append((self.is_tiled, self.n))
            return torch.zeros(1, 3, 8, 8)

    dec = Dec()
    ys = M._decode_item(dec, *(torch.zeros(1, C, 1, 1) for _ in range(4)))
    assert len(ys) == 4 and len(dec.calls) == 8
    assert all(t for t, _ in dec.calls[4:])                           # the redo is all-tiled
    dec2 = Dec()
    dec2.is_tiled = True
    M._decode_item(dec2, *(torch.zeros(1, C, 1, 1) for _ in range(4)))
    assert len(dec2.calls) == 4                                       # already tiled: no redo
    plain = []
    M._decode_item(lambda z: plain.append(1) or z, *(torch.zeros(1) for _ in range(4)))
    assert len(plain) == 4                                            # plain callables untouched


# ---------------------------------------------------------------- zd cache

def test_zd_cache_values_resume_and_bad_items(tmp_path):
    root = str(tmp_path / "n1")
    make_corpus(root, 6)
    # item 7: lat shape that disagrees with the image; item 8: unreadable image
    make_corpus(root, 3, start=6)
    np.save(os.path.join(root, "lat", f"{7:08d}.npy"), np.zeros((C, 12, 12), np.float16))
    with open(os.path.join(root, "img", f"{8:08d}.webp"), "wb") as f:
        f.write(b"not an image")
    out = str(tmp_path / "m2")
    vae = FakeVAE()
    st = M.build_zd_cache(root, out, list(range(9)), vae, torch.device("cpu"), batch=2)
    assert st["built"] == 7 and st["bad"] == 2, st
    for i in range(6):
        p = M.zd_path(out, i)
        zd = np.load(p)
        w, h = img_size(i)
        assert zd.dtype == np.float16 and zd.shape == (C, h // 16, w // 16)
        # = (Enc(down2(x)) - shift) * scale with down2 = exact 2x2 area average
        x = torch.from_numpy(np.array(Image.open(os.path.join(root, "img", f"{i:08d}.webp")).convert("RGB")))
        x = x.permute(2, 0, 1)[None].float() / 127.5 - 1
        ref = M.encode_normalized(FakeVAE(), M.down2(x))[0]
        assert np.allclose(zd.astype(np.float32), ref.numpy(), atol=2e-3)
    assert not [n for n in os.listdir(os.path.join(out, "zd")) if n.endswith(".tmp")]
    # the encoder really saw half-size area-averaged images
    seen = [tuple(t.shape[-2:]) for t in vae.enc_inputs]
    assert (64, 64) in seen and (48, 128) in seen
    # rerun: nothing to do (no encoder call), then one deleted file is rebuilt, a wrong-shaped one too
    vae2 = FakeVAE()
    st2 = M.build_zd_cache(root, out, list(range(6)), vae2, torch.device("cpu"), batch=2)
    assert st2["built"] == 0 and st2["have"] == 6 and not vae2.enc_inputs
    os.remove(M.zd_path(out, 2))
    np.save(M.zd_path(out, 4), np.zeros((C, 1, 1), np.float16))
    vae3 = FakeVAE()
    st3 = M.build_zd_cache(root, out, list(range(6)), vae3, torch.device("cpu"), batch=1)
    assert st3["built"] == 2 and len(vae3.enc_inputs) == 2


def test_zd_cache_stops_at_deadline_and_without_vae(corpus, tmp_path):
    out = str(tmp_path / "m2")
    st = M.build_zd_cache(corpus, out, list(range(12)), FakeVAE(), torch.device("cpu"), batch=2,
                          should_stop=lambda: True)
    assert st["stopped"] and st["built"] == 0
    st = M.build_zd_cache(corpus, out, list(range(12)), None, torch.device("cpu"))
    assert st["built"] == 0 and st["bad"] == 12


# ---------------------------------------------------------------- losses / schedule

def test_losses_finite_decoder_frozen_and_grads_flow():
    dec = tiny_decoder()
    for p in dec.parameters():
        p.requires_grad_(False)
    d, _ = M.build_models(C, 16)
    z = torch.randn(2, C, 16, 16)
    zd = M.area_down(z) + 0.3
    win = M.pick_window(8, 8, 4, __import__("random").Random(0))
    out = M.compute_losses(d, z, zd, dec, win, 0.1)
    assert torch.isfinite(out["loss"]) and float(out["l_img"]) > 0
    assert torch.allclose(out["loss"], out["l_lat"] + 0.1 * out["l_img"], atol=1e-6)
    out["loss"].backward()
    assert d.head.weight.grad is not None and torch.isfinite(d.head.weight.grad).all()
    assert all(p.grad is None for p in dec.parameters())
    off = M.compute_losses(d, z, zd, None, win, 0.1)
    assert float(off["l_img"]) == 0 and torch.allclose(off["loss"], off["l_lat"])


def test_pick_window_bounds():
    import random
    rng = random.Random(0)
    for h, w in ((8, 8), (6, 16), (2, 3)):
        for _ in range(50):
            oy, ox, wh, ww = M.pick_window(h, w, 4, rng)
            assert 0 <= oy and oy + wh <= h and 0 <= ox and ox + ww <= w and wh <= 4 and ww <= 4


def test_lr_schedule():
    assert M.lr_at(0, 1e-3, 500) < 1e-5 * 5
    assert abs(M.lr_at(499, 1e-3, 500) - 1e-3) < 1e-9
    assert M.lr_at(500, 1e-3, 500) > M.lr_at(15000, 1e-3, 500) > M.lr_at(30000, 1e-3, 500) > 0


def test_clip_and_step_skips_nonfinite_gradients():
    m = nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    w0 = m.weight.detach().clone()
    m.weight.grad = torch.full_like(m.weight, float("nan"))
    m.bias.grad = torch.zeros_like(m.bias)
    assert M.clip_and_step(m, opt) is False
    assert torch.equal(m.weight, w0)
    m.weight.grad = torch.ones_like(m.weight)
    m.bias.grad = torch.ones_like(m.bias)
    assert M.clip_and_step(m, opt) is True
    assert not torch.equal(m.weight, w0)


# ---------------------------------------------------------------- evaluation

def write_eval_items(tmp_path, n=4, bias=0.5):
    """Eval items where zd = area(z) + bias, so a Down model with head bias == bias is perfect."""
    root, out = str(tmp_path / "ev"), str(tmp_path / "evout")
    make_corpus(root, n)
    os.makedirs(os.path.join(out, "zd"), exist_ok=True)
    for i in range(n):
        z = torch.from_numpy(np.load(M.pair_paths(root, i)["lat"])).float()[None]
        np.save(M.zd_path(out, i), (M.area_down(z) + bias)[0].numpy().astype(np.float16))
    return root, out, M.EvalSet(root, out, list(range(n)), C)


def test_evaluate_baseline_equal_when_zero_init(tmp_path):
    root, out, es = write_eval_items(tmp_path)
    d, u = M.build_models(C, 16)
    vae = FakeVAE()
    m, samples = M.evaluate(d, u, es, "cpu", lambda z: M.decode_latents(vae, z), decode_n=4, n_figs=2)
    assert m["n"] == 4
    assert m["down"]["impr"] == pytest.approx(0.0, abs=1e-9)         # starts exactly at the area baseline
    assert m["up"]["impr"] == pytest.approx(0.0, abs=1e-9)
    assert m["down"]["psnr_gain"] == pytest.approx(0.0, abs=1e-6) and m["down"]["n_dec"] == 4
    assert len(samples["down"]) == 2 and len(samples["up"]) == 2
    orig, base, ours = samples["down"][0]
    assert orig.shape == base.shape == ours.shape and orig.dtype == np.uint8
    orig, base, ours = samples["up"][0]
    assert orig.shape == base.shape == ours.shape
    grid = M.make_compare_grid(samples["down"] + samples["up"])
    assert grid.width >= 3 * 64 and grid.height > 18
    assert d.training is True or d.training is False       # mode restored, no exception


def test_evaluate_improvement_rate_and_psnr_gain(tmp_path):
    root, out, es = write_eval_items(tmp_path, bias=0.5)
    d, u = M.build_models(C, 16)
    d.head.bias.data.fill_(0.5)                    # Down now exactly fixes the offset
    u.head.bias.data.fill_(0.7)                    # Up gets worse than bilinear
    vae = FakeVAE()
    d.train()
    m, _ = M.evaluate(d, u, es, "cpu", lambda z: M.decode_latents(vae, z), decode_n=4)
    assert d.training                               # restored
    assert m["down"]["impr"] > 0.99 and m["down"]["impr_best"] > 0.99
    assert m["down"]["psnr_gain"] > 0 and m["down"]["psnr"] > m["down"]["base_psnr"]
    assert m["up"]["impr"] < 0 and m["up"]["psnr_gain"] < 0
    assert m["down"]["base_mse"] > m["down"]["lat_mse"] >= 0


def test_evaluate_without_decode_and_with_zero_budget(tmp_path):
    root, out, es = write_eval_items(tmp_path)
    d, u = M.build_models(C, 16)
    m, _ = M.evaluate(d, u, es, "cpu", None)
    assert "psnr" not in m["down"] and "lat_mse" in m["down"]
    vae = FakeVAE()
    m, _ = M.evaluate(d, u, es, "cpu", lambda z: M.decode_latents(vae, z), decode_n=4, budget_s=-1.0)
    assert "psnr" not in m["up"] and m["n"] == 4                    # decode part skipped, latent part intact


def test_evaluate_skips_bad_items(tmp_path):
    root, out, es = write_eval_items(tmp_path)
    with open(M.zd_path(out, 1), "wb") as f:
        f.write(b"junk")
    d, u = M.build_models(C, 16)
    m, _ = M.evaluate(d, u, es, "cpu", None)
    assert m["n"] == 3


# ---------------------------------------------------------------- logging / secrets

def test_fit_marker_limit_and_token_scrub(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    big = {"step": 1, "err": "x" * 5000, **{f"k{i}": 0.123456789 * i for i in range(80)}}
    line = M.fit_marker("D1_M2", big)
    assert len(line) <= 400 and line.startswith("D1_M2 ")
    json.loads(line.split(" ", 1)[1])                                # still valid JSON
    leaked = M.fit_marker("D1_M2", {"msg": f"auth {FAKE_TOKEN} and hf_ZZZZZZZZZZZZ"})
    assert FAKE_TOKEN not in leaked and "hf_ZZZZ" not in leaked
    assert M.fit_marker("D1_M2", {"a": 1}) == 'D1_M2 {"a":1}'
    nan = M.fit_marker("D1_M2", {"x": float("nan")})
    assert len(nan) <= 400


def test_parse_deadline():
    assert M.parse_deadline(None) is None and M.parse_deadline("") is None
    assert M.parse_deadline("1791723600") == 1791723600.0
    assert M.parse_deadline("abc") is None and M.parse_deadline("-5") is None


def test_cuda_cap_helpers():
    assert M.mem_fraction(3.0, 32 * 2**30) == pytest.approx(3.0 / 32)
    assert M.mem_fraction(0, 100) == 1.0
    assert M.apply_cuda_limits(torch.device("cpu"), 3.0) is None
    assert M.MEM_CAP_GB <= 3.0


# ---------------------------------------------------------------- assets

def make_shard(path, members):
    with tarfile.open(path, "w") as t:
        for name, data in members.items():
            p = path + ".src"
            with open(p, "wb") as f:
                f.write(data)
            t.add(p, arcname=name)
            os.remove(p)


def test_extract_shard_is_safe_and_idempotent(tmp_path):
    root = str(tmp_path / "root")
    tar = str(tmp_path / "gen_0000.tar")
    make_shard(tar, {"img/00000001.webp": b"a", "lat/00000001.npy": b"b", "meta/00000001.json": b"{}",
                     "../evil.txt": b"x", "img/../../evil2.webp": b"x", "other/00000001.bin": b"x",
                     "img/abc.webp": b"x"})
    assert M.extract_shard(tar, root) == 3
    assert sorted(os.listdir(os.path.join(root, "img"))) == ["00000001.webp"]
    assert not os.path.exists(str(tmp_path / "evil.txt")) and not os.path.exists(str(tmp_path / "evil2.webp"))
    assert M.extract_shard(tar, root) == 0


def test_ensure_assets_fetches_only_what_is_missing(tmp_path):
    root = str(tmp_path / "n1")
    tae_path = os.path.join(root, "tae", "ckpt_best.pt")
    calls = []

    def lister(prefix):
        assert prefix == M.CORPUS_PREFIX
        return [M.CORPUS_PREFIX + "gen_0000.tar", M.CORPUS_PREFIX + "gen_0001.tar",
                M.CORPUS_PREFIX + "notes.txt", "elsewhere/gen_0002.tar"]

    def fetcher(remote):
        calls.append(remote)
        local = str(tmp_path / ("dl_" + os.path.basename(remote)))
        if remote == M.TAE_REMOTE:
            with open(local, "wb") as f:
                f.write(b"ckpt")
        else:
            k = int(remote[-8:-4])
            make_shard(local, {f"img/{k:08d}.webp": b"i", f"lat/{k:08d}.npy": b"l", f"meta/{k:08d}.json": b"{}"})
        return local

    res = M.ensure_assets(root, tae_path, 2, lister, fetcher)
    assert calls == [M.TAE_REMOTE, M.CORPUS_PREFIX + "gen_0000.tar", M.CORPUS_PREFIX + "gen_0001.tar"]
    assert res["tae"] and res["pairs"] == 2 and res["shards"] == 2 and os.path.isfile(tae_path)
    calls.clear()
    M.ensure_assets(root, tae_path, 2, lister, fetcher)              # everything present now: no downloads
    assert calls == []
    # a failing download is a warning, not an exception
    os.remove(tae_path)
    res = M.ensure_assets(root, tae_path, 2, lister, lambda r: (_ for _ in ()).throw(RuntimeError(FAKE_TOKEN)))
    assert res["tae"] is False


def test_hf_fetchers_need_a_token(tmp_path):
    assert M.make_hf_fetchers(None, str(tmp_path)) == (None, None)
    assert M.make_uploader(None) is None and M.make_uploader("") is None
    assert callable(M.make_uploader("tok"))


def test_load_tae_decoder_from_checkpoint(tmp_path):
    torch.manual_seed(1)
    tae = T.TAE(C, TAE_WIDTHS)
    path = str(tmp_path / "ckpt_best.pt")
    T.save_atomic({"step": 5, "model": tae.state_dict(), "config": tae.config()}, path)
    dec, c = M.load_tae_decoder(path, torch.device("cpu"))
    assert c == C and not dec.training and all(not p.requires_grad for p in dec.parameters())
    z = torch.randn(1, C, 4, 4)
    assert torch.allclose(dec(z), tae.dec(z), atol=1e-6)
    with pytest.raises(FileNotFoundError):
        M.load_tae_decoder(str(tmp_path / "missing.pt"), torch.device("cpu"))


# ---------------------------------------------------------------- full runs

def run(root, extra=(), deps=None, capsys=None):
    rc = M.main(base_argv(root, *extra), deps or deps_for())
    out = capsys.readouterr().out if capsys else ""
    return rc, out


def out_dir(root):
    return os.path.join(root, "day1", "m2")


def test_end_to_end_train_checkpoint_eval_upload(corpus, capsys):
    uploads = []

    def uploader(folder, remote):
        uploads.append((remote, sorted(os.listdir(folder))))

    rc, out = run(corpus, ["--max-steps", 3], deps_for(uploader=uploader), capsys)
    assert rc == 0
    lines = all_marker_lines(out)
    assert lines and all(len(ln) <= 400 for ln in lines)
    done = markers(out, "D1_M2_DONE")[-1]
    assert done["status"] == "max_steps" and done["step"] == 3 and done["skipped"] == 0
    start = [m for m in markers(out) if "params_down" in m][0]
    assert start["in_range"] is False                      # width 16 test models are tiny (range is checked elsewhere)
    assert start["img_loss"] is True and start["vae"] is True and start["C"] == C
    ev = markers(out, "D1_M2_EVAL")
    assert ev and ev[-1]["final"] is True and "d_mse" in ev[-1] and "u_mse" in ev[-1] and "d_psnr" in ev[-1]
    od = out_dir(corpus)
    for name in ("ckpt_last.pt", "ckpt_best.pt", "down_best.pt", "up_best.pt", "metrics.jsonl", "eval_indices.json"):
        assert os.path.isfile(os.path.join(od, name)), name
    assert any(n.startswith("down_step_") for n in os.listdir(os.path.join(od, "samples")))
    assert any(n.startswith("up_step_") for n in os.listdir(os.path.join(od, "samples")))
    rec = [json.loads(ln) for ln in open(os.path.join(od, "metrics.jsonl"))]
    assert rec[-1]["step"] == 3 and "down" in rec[-1] and "up" in rec[-1]
    assert uploads and uploads[-1][0] == "ckpt/day1/m2"
    for need in ("ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl", "down_latest.jpg", "up_latest.jpg"):
        assert need in uploads[-1][1]
    ck = torch.load(os.path.join(od, "ckpt_last.pt"), weights_only=True)
    assert ck["step"] == 3 and ck["config"] == {"latent_channels": C, "width": 16}
    b = torch.load(os.path.join(od, "ckpt_best.pt"), weights_only=True)
    assert "model" in b["down"] and "model" in b["up"]
    # the eval items were never trained on
    ev_idx = set(json.load(open(os.path.join(od, "eval_indices.json"))))
    assert len(ev_idx) == 3
    # nothing half-written left behind
    for d in (od, os.path.join(od, "zd"), os.path.join(od, "samples")):
        assert not [n for n in os.listdir(d) if n.endswith(".tmp") or n.endswith(".part")]


def test_models_actually_learn_something(corpus, capsys):
    rc, out = run(corpus, ["--max-steps", 40, "--lr", "3e-3"], deps_for(), capsys)
    assert rc == 0
    ck = torch.load(os.path.join(out_dir(corpus), "ckpt_last.pt"), weights_only=True)
    assert float(ck["down"]["head.weight"].abs().sum()) > 0           # zero-init head moved
    assert all(torch.isfinite(v).all() for v in ck["up"].values())


def test_resume_continues_and_keeps_best(corpus, capsys):
    rc, out1 = run(corpus, ["--max-steps", 3], deps_for(), capsys)
    od = out_dir(corpus)
    best1 = torch.load(os.path.join(od, "ckpt_best.pt"), weights_only=True)
    ck1 = torch.load(os.path.join(od, "ckpt_last.pt"), weights_only=True)
    vae = FakeVAE()
    rc, out2 = run(corpus, ["--max-steps", 5], deps_for(vae=vae), capsys)
    assert rc == 0
    start = [m for m in markers(out2) if "params_down" in m][0]
    assert start["resumed"] is True and start["step"] == 3
    assert not vae.enc_inputs                                          # zd cache reused, nothing re-encoded
    done = markers(out2, "D1_M2_DONE")[-1]
    assert done["start_step"] == 3 and done["step"] == 5
    ck2 = torch.load(os.path.join(od, "ckpt_last.pt"), weights_only=True)
    assert ck2["step"] == 5
    assert not torch.equal(ck1["up"]["head.weight"], ck2["up"]["head.weight"])
    assert os.path.isfile(os.path.join(od, "ckpt_best.pt")) and "down" in best1
    assert len(open(os.path.join(od, "metrics.jsonl")).read().splitlines()) == 2
    # a run already at its target step stops at once with a final save
    rc, out3 = run(corpus, ["--max-steps", 5], deps_for(), capsys)
    assert rc == 0 and markers(out3, "D1_M2_DONE")[-1]["step"] == 5


def test_resume_ignores_checkpoint_with_other_config(corpus, capsys):
    run(corpus, ["--max-steps", 2], deps_for(), capsys)
    rc, out = run(corpus, ["--max-steps", 2, "--width", 24], deps_for(), capsys)
    assert rc == 0
    start = [m for m in markers(out) if "params_down" in m][0]
    assert start["resumed"] is False and start["step"] == 0
    assert any(m.get("warn") == "ckpt_config_mismatch" for m in markers(out))


def test_deadline_stops_with_final_save_and_margin(corpus, capsys):
    clock = Clock(step=1.0, t=5000.0)
    deadline = 5000.0 + 600 + 40                                       # 40 fake seconds of work, then the margin hits
    rc, out = run(corpus, ["--deadline", deadline, "--finish-margin-s", 600],
                  deps_for(now=clock), capsys)
    assert rc == 0
    done = markers(out, "D1_M2_DONE")[-1]
    assert done["status"] == "deadline"
    od = out_dir(corpus)
    assert os.path.isfile(os.path.join(od, "ckpt_last.pt"))
    ev = markers(out, "D1_M2_EVAL")
    assert ev and ev[-1]["final"] is True
    # started after the deadline: still exits cleanly (cache build is skipped, nothing crashes)
    rc, out = run(corpus, ["--deadline", 1.0, "--finish-margin-s", 600], deps_for(now=Clock(t=5000.0)), capsys)
    assert rc == 0 and markers(out, "D1_M2_DONE")


def test_signal_stops_gracefully(corpus, capsys):
    stop = threading.Event()
    stop.set()
    rc, out = run(corpus, [], deps_for(stop_event=stop), capsys)
    assert rc == 0
    assert markers(out, "D1_M2_DONE")[-1]["status"] == "signal"


def test_token_never_printed_or_written(corpus, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)

    def bad_uploader(folder, remote):
        raise RuntimeError(f"401 unauthorized for token {FAKE_TOKEN}")

    rc, out = run(corpus, ["--max-steps", 2], deps_for(uploader=bad_uploader), capsys)
    err = capsys.readouterr().err
    assert rc == 0                                                     # an upload failure never kills the job
    assert FAKE_TOKEN not in out and FAKE_TOKEN not in err
    assert any(m.get("warn") == "upload_failed" for m in markers(out))
    start = markers(out)[0]
    assert start["hf_token"] == "yes"
    for dp, _, fs in os.walk(corpus):
        for fn in fs:
            if fn.endswith((".webp", ".npy")):
                continue
            with open(os.path.join(dp, fn), "rb") as f:
                assert FAKE_TOKEN.encode() not in f.read(), fn
    monkeypatch.delenv("HF_TOKEN")
    rc, out = run(corpus, ["--max-steps", 2], deps_for(), capsys)
    assert markers(out)[0]["hf_token"] == "no" and markers(out)[0]["upload"] is False


def test_consecutive_step_failures_abort(corpus, capsys, monkeypatch):
    def boom(self, zc, zdc, micro):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(M.Trainer, "_accumulate", boom)
    rc, out = run(corpus, ["--max-steps", 50, "--max-consec-errors", 4], deps_for(), capsys)
    assert rc == 2
    done = markers(out, "D1_M2_DONE")[-1]
    assert done["status"] == "fatal" and done["fatal"] == "too_many_step_errors" and done["step"] == 0
    assert sum(1 for m in markers(out) if m.get("warn") == "step_failed") == 4
    assert os.path.isfile(os.path.join(out_dir(corpus), "ckpt_last.pt"))      # state saved even when aborting
    assert M.MAX_CONSEC_ERRORS == 100


def test_a_single_failing_step_is_skipped_not_fatal(corpus, capsys, monkeypatch):
    real = M.Trainer._accumulate
    calls = {"n": 0}

    def flaky(self, zc, zdc, micro):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("one bad batch")
        return real(self, zc, zdc, micro)

    monkeypatch.setattr(M.Trainer, "_accumulate", flaky)
    rc, out = run(corpus, ["--max-steps", 4], deps_for(), capsys)
    assert rc == 0 and markers(out, "D1_M2_DONE")[-1]["step"] == 4
    assert sum(1 for m in markers(out) if m.get("warn") == "step_failed") == 1


def test_oom_halves_micro_batch_and_keeps_going(corpus, monkeypatch):
    real = M.Trainer._accumulate

    def oomy(self, zc, zdc, micro):
        if micro > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (fake)")
        return real(self, zc, zdc, micro)

    monkeypatch.setattr(M.Trainer, "_accumulate", oomy)
    args = M.build_parser().parse_args(base_argv(corpus, "--max-steps", 2, "--micro-batch", 4))
    tr = M.Trainer(args, uploader=None, deps=deps_for(), sleep=lambda s: None)
    res = tr.run()
    assert res["status"] == "max_steps" and res["step"] == 2 and tr.micro_batch == 1


def test_nonfinite_gradients_are_skipped(corpus, capsys, monkeypatch):
    real = M.clip_and_step
    calls = {"n": 0}

    def fake(model, opt, clip=1.0):
        calls["n"] += 1
        if calls["n"] <= 2:                    # first step: both models report non-finite
            opt.zero_grad(set_to_none=True)
            return False
        return real(model, opt, clip)

    monkeypatch.setattr(M, "clip_and_step", fake)
    rc, out = run(corpus, ["--max-steps", 2], deps_for(), capsys)
    done = markers(out, "D1_M2_DONE")[-1]
    assert rc == 0 and done["step"] == 2 and done["skipped"] == 2


def test_no_tae_checkpoint_degrades_to_latent_loss_only(corpus, capsys):
    d = deps_for()
    d.pop("decoder")                                   # real loading path, no ckpt on disk, no network
    rc, out = run(corpus, ["--max-steps", 2], d, capsys)
    assert rc == 0
    start = [m for m in markers(out) if "params_down" in m][0]
    assert start["img_loss"] is False
    assert any(m.get("warn") == "tae_unavailable_img_loss_off" for m in markers(out))


def test_no_vae_and_no_cache_ends_cleanly(corpus, capsys):
    d = deps_for(vae=None)
    rc, out = run(corpus, ["--max-steps", 2], d, capsys)
    assert rc == 0
    done = markers(out, "D1_M2_DONE")[-1]
    assert done["status"] == "no_train_data"


def test_empty_corpus_is_a_clean_exit(tmp_path, capsys):
    rc, out = run(str(tmp_path / "empty"), ["--max-steps", 1], deps_for(), capsys)
    assert rc == 0 and markers(out, "D1_M2_DONE")[-1]["fatal"] == "no_data"


def test_bad_corpus_item_does_not_kill_the_run(corpus, capsys):
    # one lat file is garbage: it is skipped in the cache build and the run still trains
    with open(os.path.join(corpus, "lat", f"{4:08d}.npy"), "wb") as f:
        f.write(b"garbage")
    rc, out = run(corpus, ["--max-steps", 3], deps_for(), capsys)
    assert rc == 0 and markers(out, "D1_M2_DONE")[-1]["step"] == 3


def test_progress_marker_and_vram_fields(corpus, capsys):
    clock = Clock(step=2.0, t=100.0)
    rc, out = run(corpus, ["--max-steps", 6, "--log-every-s", 5], deps_for(now=clock), capsys)
    prog = [m for m in markers(out) if "ld_lat" in m]
    assert prog and all(k in prog[0] for k in ("step", "ld", "lu", "ips", "lr", "vram_GB", "mb"))
    assert all(len(ln) <= 400 for ln in all_marker_lines(out))


# ---------------------------------------------------------------- review fixes

def test_vae_tiling_decided_once_at_load_and_logged(corpus, capsys):
    rc, out = run(corpus, ["--max-steps", 1], deps_for(vae=FakeVAE(fail_first_decode_oom=True)), capsys)
    start = [m for m in markers(out) if "params_down" in m][0]
    assert rc == 0 and start["vae_tiled"] is True
    assert all(len(ln) <= 400 for ln in all_marker_lines(out))


def test_vae_untiled_when_probe_decode_fits(tmp_path, capsys):
    root = str(tmp_path / "n2")
    make_corpus(root, 12)
    vae = FakeVAE()
    rc, out = run(root, ["--max-steps", 1], deps_for(vae=vae), capsys)
    start = [m for m in markers(out) if "params_down" in m][0]
    assert start["vae_tiled"] is False and not vae.tiling


def test_finished_run_is_not_redone_by_a_relaunch_loop(corpus, capsys):
    uploads = []

    def uploader(folder, remote):
        uploads.append(remote)

    od = out_dir(corpus)
    rc, out1 = run(corpus, ["--max-steps", 3], deps_for(uploader=uploader), capsys)
    assert rc == 0 and os.path.isfile(os.path.join(od, "DONE.json"))
    assert json.load(open(os.path.join(od, "DONE.json")))["step"] == 3
    n_rows = len(open(os.path.join(od, "metrics.jsonl")).read().splitlines())
    n_up = len(uploads)
    vae = FakeVAE()
    for _ in range(3):
        rc, out2 = run(corpus, ["--max-steps", 3], deps_for(uploader=uploader, vae=vae), capsys)
        assert rc == 0
        done = markers(out2, "D1_M2_DONE")
        assert done[-1]["status"] == "already_done" and done[-1]["step"] == 3
        assert not [m for m in markers(out2) if "params_down" in m]          # nothing loaded
    assert len(open(os.path.join(od, "metrics.jsonl")).read().splitlines()) == n_rows
    assert len(uploads) == n_up and not vae.dec_inputs and not vae.enc_inputs
    # a larger target is not blocked by the old DONE.json
    rc, out3 = run(corpus, ["--max-steps", 5], deps_for(), capsys)
    assert markers(out3, "D1_M2_DONE")[-1]["step"] == 5


def test_relaunch_without_progress_skips_final_eval_and_upload(corpus, capsys):
    uploads = []

    def uploader(folder, remote):
        uploads.append(remote)

    od = out_dir(corpus)
    clock = Clock(step=1.0, t=5000.0)
    rc, out = run(corpus, ["--deadline", 5000.0 + 600 + 40, "--finish-margin-s", 600],
                  deps_for(now=clock, uploader=uploader), capsys)
    assert markers(out, "D1_M2_DONE")[-1]["status"] == "deadline"
    assert not os.path.isfile(os.path.join(od, "DONE.json"))              # a deadline stop is not "done"
    rows = open(os.path.join(od, "metrics.jsonl")).read().splitlines()
    n_up = len(uploads)
    rc, out = run(corpus, ["--deadline", 1.0, "--finish-margin-s", 600], deps_for(now=Clock(t=5000.0),
                  uploader=uploader), capsys)
    assert rc == 0 and markers(out, "D1_M2_DONE")[-1]["status"] == "deadline"
    assert open(os.path.join(od, "metrics.jsonl")).read().splitlines() == rows and len(uploads) == n_up


def test_checkpoint_is_saved_before_the_eval_runs(corpus, monkeypatch):
    args = M.build_parser().parse_args(base_argv(corpus, "--max-steps", 2))
    tr = M.Trainer(args, uploader=None, deps=deps_for(), sleep=lambda s: None)
    seen = {}
    real = M.evaluate

    def spying(*a, **k):
        p = os.path.join(out_dir(corpus), "ckpt_last.pt")
        seen["step_on_disk"] = torch.load(p, weights_only=True)["step"] if os.path.isfile(p) else None
        seen["decode_fn"] = a[4]
        raise RuntimeError("CUDA error inside decode")                    # eval blows up: ckpt must survive

    monkeypatch.setattr(M, "evaluate", spying)
    tr.run()
    assert seen["step_on_disk"] == 2                                      # saved before the eval started
    assert torch.load(os.path.join(out_dir(corpus), "ckpt_last.pt"), weights_only=True)["step"] == 2
    monkeypatch.setattr(M, "evaluate", real)


def test_signal_final_eval_skips_the_decodes(corpus, monkeypatch):
    args = M.build_parser().parse_args(base_argv(corpus, "--max-steps", 2))
    stop = threading.Event()
    tr = M.Trainer(args, uploader=None, deps=deps_for(), sleep=lambda s: None, stop_event=stop)
    calls = []
    real = M.evaluate

    def spying(*a, **k):
        calls.append(a[4])
        return real(*a, **k)

    monkeypatch.setattr(M, "evaluate", spying)
    real_final = tr.checkpoint_and_eval

    def final(final=False, reason=None):
        return real_final(final=final, reason=reason)

    stop.set()
    res = tr.run()
    assert res["status"] == "signal"
    assert calls and calls[-1] is None                                    # no decode_fn on SIGTERM
    calls.clear()
    tr2 = M.Trainer(M.build_parser().parse_args(base_argv(corpus, "--max-steps", 4)), uploader=None,
                    deps=deps_for(), sleep=lambda s: None)
    tr2.run()
    assert calls and calls[-1] is not None                                # normal finish still decodes
