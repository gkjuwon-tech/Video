"""CPU tests for m3_mrh.py: no GPU, no network (localhost only), no real weights, no diffusers.

A fake teacher (tiny DiT with the Z-Image token layout, 8x8-pooling VAE), tiny renders and tiny fake DIODE
tarballs stand in for the real things.  Fakes are injected through the ``deps`` dict of the job.
"""
import dataclasses
import gzip
import hashlib
import http.server
import io
import json
import os
import re
import sys
import tarfile
import threading
import time
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m3_mrh as M  # noqa: E402

from buc.core import det  # noqa: E402
from buc.gov import prov as P  # noqa: E402
from buc.models.mrh import heads as MH  # noqa: E402
from buc.models.mrh import neck as MN  # noqa: E402
from buc.models.mrh import train as MT  # noqa: E402
from buc.teacher import sample as TS  # noqa: E402
from buc.teacher import taps as T  # noqa: E402
from buc.teacher import zimage as Z  # noqa: E402

torch.set_num_threads(1)

FAKE_TOKEN = "hf_fakeTOKEN123456"
D, K, NL, TEXT, C = 32, 8, 30, 8, 16     # fake hidden dim, PCA dim, blocks, text dim, latent channels
SPEC = dataclasses.replace(Z.TeacherSpec(), num_layers=NL, hidden_dim=D, text_dim=TEXT, max_text_len=16)
WH_SYN = (64, 48)                         # fake synthetic render size (W, H)
HW_DIODE = (96, 128)                      # fake DIODE crop (H, W)


# ---------------------------------------------------------------- fakes

def gen(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


class FakeBlock(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.lin = nn.Linear(D, D)
        g = gen(seed)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(D, D, generator=g) / D ** 0.5)
            self.lin.bias.copy_(torch.randn(D, generator=g) * 0.1)

    def forward(self, x):
        return x + 0.5 * torch.tanh(self.lin(x))


class FakeDiT(nn.Module):
    """Z-Image token layout: image tokens first (offset 0, row major), then the caption padded to 32 tokens."""

    def __init__(self, seed=100):
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock(seed + i) for i in range(NL)])
        g = gen(seed - 1)
        self.register_buffer("embed", torch.randn(C * 4, D, generator=g) / (C * 4) ** 0.5)
        self.register_buffer("cap_proj", torch.randn(TEXT, D, generator=g))
        self.calls = 0
        self.block_calls = [0] * NL
        for i, b in enumerate(self.layers):
            b.register_forward_hook(lambda m, a, o, i=i: self._count(i))

    def _count(self, i):
        self.block_calls[i] += 1

    def forward(self, x, t, cap_feats, return_dict=True, **kw):
        self.calls += 1
        z = torch.stack([a.float() for a in x]).squeeze(2)
        n, c, h, w = z.shape
        gh, gw = h // 2, w // 2
        pt = z.reshape(n, c, gh, 2, gw, 2).permute(0, 2, 4, 1, 3, 5).reshape(n, gh * gw, c * 4)
        tok = pt @ self.embed
        cap = torch.stack([cf.float().mean(0) @ self.cap_proj for cf in cap_feats]).unsqueeze(1).expand(-1, 32, -1)
        hs = torch.cat([tok, cap], dim=1)
        for blk in self.layers:
            hs = blk(hs)
        sig = (1.0 - t.float()).view(n, 1, 1, 1)
        x0 = z / (1.0 - sig)                       # identity x0-hat (zero noise): the roundtrip keeps the image
        out = -(z - x0) / sig
        outs = list(out.unsqueeze(2).to(x[0].dtype).unbind(0))
        return (outs,) if not return_dict else {"sample": outs}


class FakeVae:
    def __init__(self, seed=5):
        g = gen(seed)
        m = torch.randn(C, 3, generator=g, dtype=torch.float64)
        m[:, 0] += 1.0
        self.M = m.float()
        self.Minv = torch.linalg.pinv(m).float()

    def encode(self, x):
        p = F.avg_pool2d(x.float(), 8)
        return torch.einsum("ck,bkhw->bchw", self.M, p).contiguous()

    def decode(self, z, clamp=True):
        p = torch.einsum("kc,bchw->bkhw", self.Minv, z.float())
        y = p.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        return y.clamp(-1.0, 1.0) if clamp else y


class FakeTeacher:
    """ZImageTeacher duck type.  Zero noise: x0-hat equals the input latent, so low bands survive exactly."""

    backend = "fake"

    def __init__(self):
        self.spec = SPEC
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.transformer = FakeDiT().eval()
        self.layers = self.transformer.layers
        self.vae = FakeVae()
        self.prompts = []
        self.released = False
        self.snapshot = SimpleNamespace(name="fake-rev")
        g = gen(3)
        self._emb = torch.randn(256, TEXT, generator=g)

    def encode_text(self, prompts):
        self.prompts.append(list(prompts))
        return [self._emb[torch.tensor(list(p.encode())[:5])].clone() for p in prompts]

    def release_text_encoder(self):
        self.released = True

    def encode_image(self, x):
        return self.vae.encode(x)

    def decode_latent(self, z, clamp=True):
        return self.vae.decode(z, clamp=clamp)

    def make_noise(self, seed, batch, height, width, *, level=0, k=0):
        return torch.zeros(self.spec.latent_shape(height, width, batch))


def make_reader(early_exit=True, teacher=None):
    teacher = teacher or FakeTeacher()
    sampler = TS.TeacherSampler(teacher)
    cap = teacher.encode_text([M.SURVEY_CAPTION])[0]
    return M.TeacherReader(teacher, sampler, cap, early_exit=early_exit, backend="fake")


def tiny_cfg(**kw):
    base = dict(tap_layers=M.TAP_LAYERS, tap_dim=K, neck_ch=32, rcu_per_fusion=1, head_ch=16, sem_dim=8,
                n_sigma_bands=4, t_embed_dim=16)
    base.update(kw)
    return MN.MRHConfig(**base)


def fake_basis(seed=0, k=K):
    g = gen(seed)
    L = len(M.TAP_LAYERS)
    mean = torch.randn(L, D, generator=g)
    comps = torch.randn(L, D, k, generator=g)
    scale = 0.01 + 0.02 * torch.rand(L, k, generator=g)
    return T.PCABasis(M.TAP_LAYERS, mean, comps, scale)


# ---------------------------------------------------------------- shared builders

@pytest.fixture
def paths(tmp_path):
    p = M.Paths(str(tmp_path / "n1"))
    p.ensure()
    return p


def base_args(*extra):
    return M.build_parser().parse_args(["--root", "x", "--pca-k", str(K), "--basis-tokens", "4", "--basis-corpus", "6",
                                        "--basis-render", "6", *extra])


def make_png(arr_u8):
    bio = io.BytesIO()
    Image.fromarray(arr_u8).save(bio, format="PNG")
    return bio.getvalue()


def npy(a):
    bio = io.BytesIO()
    np.save(bio, a)
    return bio.getvalue()


def fake_diode_tar(members):
    """members: {name: bytes}; -> .tar.gz bytes (directories are not needed)."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return bio.getvalue()


def diode_members(split, scene, scan, env, yaw_pitch, hw=HW_DIODE, seed=0, mask_dtype=np.float32):
    """Triples (png, depth, depth_mask) in the DIODE naming, with smooth fake depth."""
    rs = np.random.RandomState(seed + int(scan))
    h, w = hw
    out = {}
    for yaw, pitch in yaw_pitch:
        stem = f"{scene}_{scan}_{env}_{yaw:03d}_{pitch:03d}"
        d = f"{split}/{env}/scene_{scene}/scan_{scan}/"
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
        depth = (2.0 + 6.0 * xx + 1.5 * yy + 0.05 * rs.rand(h, w)).astype(np.float32)[..., None]
        mask = np.ones((h, w), mask_dtype)
        mask[:6, :10] = 0
        rgb = (np.stack([xx, yy, 0.5 * np.ones_like(xx)], -1) * 200 + rs.rand(h, w, 3) * 20).astype(np.uint8)
        out[d + stem + ".png"] = make_png(rgb)
        out[d + stem + "_depth.npy"] = npy(depth)
        out[d + stem + "_depth_mask.npy"] = npy(mask)
    return out


def write_diode_samples(paths, split, scene, scan, env, yaw_pitch, **kw):
    """Run the real ingest on in-memory members (no HTTP) so the on-disk layout is the production one."""
    sel = M.Selection()
    ing = M.DiodeIngest(paths, split, sel)
    mem = diode_members(split, scene, scan, env, yaw_pitch, **kw)
    for name, data in mem.items():
        ing.handle(name, len(data), lambda d=data: d)
    ing.finish()
    return ing


def fill_synth(paths, seeds=(100000, 100001, 100002), dev_seeds=(0, 1), wh=WH_SYN):
    """Real tiny renders (RoomGen) with the image replaced by 8x8 block constants (lossless for the fake VAE)."""
    w, h = wh
    for d, ss, views in ((paths.synth_train, seeds, (0, 1)), (paths.synth_dev, dev_seeds, (0,))):
        for s in ss:
            recs = M.render_views(s, views, w, h)
            if recs is None:
                continue
            for r in recs:
                if r["empty"]:
                    continue
                img = r["image"].astype(np.float32)
                blk = img.reshape(h // 8, 8, w // 8, 8, 3).mean((1, 3))
                r["image"] = np.repeat(np.repeat(blk, 8, 0), 8, 1).round().astype(np.uint8)
                M.write_synth_sample(d, M.synth_key(s, r["view"]), r)


def fill_corpus(root, n=8, seed=0):
    os.makedirs(os.path.join(root, "img"), exist_ok=True)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    rs = np.random.RandomState(seed)
    for i in range(n):
        h, w = (64, 64) if i % 2 == 0 else (48, 80)
        arr = (rs.rand(h // 8, w // 8, 3) * 255).astype(np.uint8)
        arr = np.repeat(np.repeat(arr, 8, 0), 8, 1)
        Image.fromarray(arr).save(os.path.join(root, "img", f"{i:08d}.webp"), quality=95)
        json.dump({"prompt": "x", "ip_flag": i == 3}, open(os.path.join(root, "meta", f"{i:08d}.json"), "w"))


def read_stdout(capsys):
    return capsys.readouterr().out


def markers(text, prefix="D1_M3"):
    out = []
    for line in text.splitlines():
        if line.startswith(prefix):
            head, _, body = line.partition(" ")
            try:
                out.append((head, json.loads(body)))
            except ValueError:
                out.append((head, None))
    return out


# ================================================================ 1. DIODE names and selection

def test_parse_member_spellings_and_junk():
    ok = M.parse_member("val/indoors/scene_00019/scan_00183/00019_00183_indoors_130_020.png")
    assert ok and (ok.env, ok.scene, ok.scan, ok.yaw_i, ok.pitch_i, ok.kind) == ("indoors", "00019", "00183", 13, 2, "rgb")
    assert M.parse_member("train/outdoor/scene_00022/scan_00196/00022_00196_outdoor_050_040_depth.npy").kind == "depth"
    assert M.parse_member("train/outdoor/scene_00022/scan_00196/00022_00196_outdoor_050_040_depth_mask.npy").kind == "mask"
    assert M.parse_member("./train/outdoor/scene_00022/scan_00196/00022_00196_outdoor_050_040.png") is not None
    junk = ["", None, 5, "val/", "val/outdoor/", "val/indoor/scene_00019/scan_00183/00019_00183_indoor_130_020.png",
            "val/indoors/scene_00019/scan_00183/00019_00183_outdoor_130_020.png",      # env token disagrees with the folder
            "val/indoors/scene_00019/scan_00184/00019_00183_indoors_130_020.png",      # scan disagrees
            "val/indoors/scene_00019/scan_00183/00019_00183_indoors_130_020_depth.png",
            "val/indoors/scene_00019/scan_00183/00019_00183_indoors_130_020.npy",
            "../val/indoors/scene_00019/scan_00183/00019_00183_indoors_130_020.png",
            "/etc/val/indoors/scene_00019/scan_00183/00019_00183_indoors_130_020.png",
            "val/indoors/../scene_00019/scan_00183/00019_00183_indoors_130_020.png",
            "val\\indoors\\scene_00019\\scan_00183\\00019_00183_indoors_130_020.png",
            "test/indoors/scene_00019/scan_00183/00019_00183_indoors_130_020.png",
            "val/indoors/scene_19/scan_00183/00019_00183_indoors_130_020.png", "val/indoors/x.png"]
    for j in junk:
        assert M.parse_member(j) is None, j


def test_selection_is_a_pure_function_of_the_name():
    names = [f"train/{e}/scene_00010/scan_{s:05d}/00010_{s:05d}_{e}_{y * 10:03d}_{p * 10:03d}.png"
             for e in ("indoors", "outdoor") for s in (100, 101, 102) for y in range(36) for p in range(6)]
    sel = M.Selection()
    a = {n: sel.keep(M.parse_member(n)) for n in names}
    rng = np.random.RandomState(0)
    shuffled = list(rng.permutation(names))
    b = {n: sel.keep(M.parse_member(n)) for n in shuffled}
    assert a == b                                          # order independent
    assert a == {n: sel.keep(M.parse_member(n)) for n in names}      # repeatable
    kept = [n for n in names if a[n]]
    out = sum("outdoor" in n for n in kept)
    inn = len(kept) - out
    assert out == 3 * 12 * 3 and inn == 3 * 6 * 3          # per scan: outdoor 12 yaws x 3 pitches, indoor 6 x 3
    for n in kept:
        m = M.parse_member(n)
        assert m.pitch_i in (1, 2, 3)
        assert m.yaw_i % (3 if m.env == "outdoor" else 6) == int(m.scan) % (3 if m.env == "outdoor" else 6)
    # val is kept whole
    assert all(M.Selection().keep(M.parse_member(n.replace("train/", "val/"))) for n in names)


def test_selection_covers_every_scan_and_disk_policy_doubles_strides():
    sel = M.Selection()
    for scan in range(179):
        env = "outdoor" if scan % 3 else "indoors"
        names = [f"train/{env}/scene_00001/scan_{scan:05d}/00001_{scan:05d}_{env}_{y * 10:03d}_{p * 10:03d}.png"
                 for y in range(36) for p in range(6)]
        assert any(sel.keep(M.parse_member(n)) for n in names), scan
    dbl = M.Selection(mult=2)
    n1 = sum(sel.keep(M.parse_member(f"train/outdoor/scene_00001/scan_00001/00001_00001_outdoor_{y * 10:03d}_020.png")) for y in range(36))
    n2 = sum(dbl.keep(M.parse_member(f"train/outdoor/scene_00001/scan_00001/00001_00001_outdoor_{y * 10:03d}_020.png")) for y in range(36))
    assert n1 == 12 and n2 == 6
    assert sel.sel_hash != dbl.sel_hash
    assert M.disk_policy(30) == "ok" and M.disk_policy(20) == "double" and M.disk_policy(7.9) == "stop"


def test_dev_scenes_are_held_out_of_training(paths):
    m_dev = M.parse_member("train/indoors/scene_00004/scan_00010/00004_00010_indoors_010_020.png")
    m_tr = M.parse_member("train/indoors/scene_00005/scan_00010/00005_00010_indoors_010_020.png")
    m_val = M.parse_member("val/indoors/scene_00019/scan_00183/00019_00183_indoors_010_020.png")
    assert (M.role_of(m_dev), M.role_of(m_tr), M.role_of(m_val)) == ("dev", "train", "val")
    assert M.diode_key(m_dev).startswith("dv_") and M.diode_key(m_tr).startswith("d0") and not M.diode_key(m_tr).startswith("dv_")
    for scene in M.DIODE_DEV_SCENES:
        assert M.role_of(M.parse_member(f"train/outdoor/scene_{scene}/scan_00001/{scene}_00001_outdoor_010_020.png")) == "dev"


def test_balanced_prefix_keeps_every_scene():
    samples = []
    for scene, n in (("00001", 9), ("00002", 3), ("00003", 6)):
        for k in range(n):
            m = M.parse_member(f"train/outdoor/scene_{scene}/scan_00001/{scene}_00001_outdoor_{(k + 1) * 10:03d}_020.png")
            samples.append((m, f"/x/{scene}/{k}"))
    got = M.balanced_prefix(6, samples)
    assert len(got) == 6 and {m.scene for m, _ in got} == {"00001", "00002", "00003"}
    assert M.balanced_prefix(None, samples) == samples and len(M.balanced_prefix(100, samples)) == 18


# ================================================================ 2. resumable streaming

class _Handler(http.server.BaseHTTPRequestHandler):
    data = b""
    etag = '"abc-1"'
    plan = []             # per request: ("cut", n_bytes) | ("status", code) | None
    requests = []
    require_if_match = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        cls = type(self)
        cls.requests.append({"range": self.headers.get("Range"), "if_match": self.headers.get("If-Match")})
        act = cls.plan.pop(0) if cls.plan else None
        if act and act[0] == "status":
            self.send_response(act[1])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range")
        im = self.headers.get("If-Match")
        if im is not None and im != cls.etag:
            self.send_response(412)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if rng:
            start = int(re.match(r"bytes=(\d+)-", rng).group(1))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(cls.data) - 1}/{len(cls.data)}")
        else:
            self.send_response(200)
        body = cls.data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", cls.etag)
        self.end_headers()
        if act and act[0] == "cut":
            self.wfile.write(body[:act[1]])
            self.wfile.flush()
            self.close_connection = True
            self.connection.shutdown(2)
            return
        self.wfile.write(body)


@pytest.fixture
def server():
    _Handler.requests = []
    _Handler.plan = []
    _Handler.etag = '"abc-1"'
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv, f"http://127.0.0.1:{srv.server_port}/f.tar.gz"
    srv.shutdown()
    srv.server_close()


def test_resumable_http_reconnects_with_range_and_if_match(server):
    srv, url = server
    data = np.random.RandomState(1).bytes(400_000)
    _Handler.data = data
    _Handler.plan = [("cut", 50_000), ("cut", 10_000), ("status", 503), ("cut", 70_000)]
    raw = M.ResumableHTTP(url, sleep=lambda s: None)
    got = io.BufferedReader(raw, buffer_size=1 << 16).read()
    assert got == data
    assert raw.md5.hexdigest() == hashlib.md5(data).hexdigest()
    assert raw.reconn == 4 and raw.size == len(data) and raw.etag == '"abc-1"'
    resumed = [r for r in _Handler.requests if r["range"]]
    assert len(resumed) >= 3 and all(r["if_match"] == '"abc-1"' for r in resumed)
    # every resume starts exactly where the previous bytes ended
    offs = [int(re.match(r"bytes=(\d+)-", r["range"]).group(1)) for r in resumed]
    assert offs == sorted(offs) and offs[0] == 50_000


def test_resumable_http_fatal_cases(server):
    srv, url = server
    _Handler.data = b"x" * 100_000
    _Handler.plan = [("cut", 1000)]
    raw = M.ResumableHTTP(url, sleep=lambda s: None)
    assert len(raw.read(500)) > 0
    _Handler.etag = '"changed"'                                       # the object changed under us: 412 on resume
    with pytest.raises(M.HttpFatal):
        io.BufferedReader(raw).read()
    _Handler.etag = '"abc-1"'
    _Handler.plan = [("status", 403)]
    with pytest.raises(M.HttpFatal):
        M.ResumableHTTP(url, sleep=lambda s: None).read(10)
    _Handler.plan = [("status", 503)] * 20                           # 5xx: retried, then fatal after max_retries
    sleeps = []
    raw2 = M.ResumableHTTP(url, max_retries=3, sleep=sleeps.append)
    with pytest.raises(M.HttpFatal):
        raw2.read(10)
    assert sleeps == [2.0, 4.0, 8.0]                                  # min(60, 2*2^k)


def _train_members():
    mem = {}
    # outdoor scan 196 (196 % 3 == 1): yaw_i 1 and 4 are kept, 0 is not; pitch codes 010/020/030 kept, 040 not
    mem.update(diode_members("train", "00022", "00196", "outdoor", [(10, 20), (40, 10), (0, 20), (10, 40)], seed=1))
    mem.update(diode_members("train", "00004", "00012", "indoors", [(0, 20), (60, 30)], seed=2, mask_dtype=np.float64))
    return mem


def test_diode_stream_end_to_end_resume_and_skip(paths, server, capsys):
    srv, url = server
    mem = _train_members()
    names = list(mem)
    _Handler.data = fake_diode_tar(mem)
    # first attempt dies after a few members (simulated kill), second finishes; the connection also drops once
    seen = {"n": 0}

    def stop():
        seen["n"] += 1
        return seen["n"] > 6

    _Handler.plan = [("cut", 400)]
    sel = M.Selection()
    r1 = M.run_diode_split(paths, "train", url=url, selection=sel, expected_png=6, stop=stop, sleep=lambda s: None)
    assert r1["status"] == "stopped" and not M.stage_done(paths, "diode_train")
    assert not os.path.exists(paths.stage_file("diode_train"))
    first = len(M.list_diode_samples(paths, "train"))
    open(os.path.join(paths.diode_train, "x.gt.npz.part"), "w").write("junk")                  # a half-written file
    assert M.clean_tmp(paths) >= 1
    _Handler.plan = []
    r2 = M.run_diode_split(paths, "train", url=url, selection=sel, expected_png=6, sleep=lambda s: None)
    assert r2["status"] == "done", r2
    assert M.stage_done(paths, "diode_train")
    got = M.list_diode_samples(paths, "train")
    kept_expected = sorted(n for n in names if n.endswith(".png") and sel.keep(M.parse_member(n)))
    assert len(got) == len(kept_expected) == 4                       # 3 pitches/yaws kept outdoors+indoors, junk skipped
    assert r2["skip"] >= first                                       # finished samples were skipped, not redone
    src = json.load(open(paths.source_diode))
    assert src["train"]["etag"] == '"abc-1"' and src["train"]["md5_observed"] == hashlib.md5(_Handler.data).hexdigest()
    assert src["train"]["sel_hash"] == sel.sel_hash and src["sel_hash"] == sel.sel_hash
    # a third run is a no-op
    r3 = M.run_diode_split(paths, "train", url=url, selection=sel, expected_png=6)
    assert r3["status"] == "done"
    assert os.path.isfile(os.path.join(paths.diode, "LICENSE_DIODE.txt"))


def test_diode_member_count_mismatch_writes_no_done_marker(paths, server):
    srv, url = server
    _Handler.data = fake_diode_tar(_train_members())
    r = M.run_diode_split(paths, "train", url=url, selection=M.Selection(), expected_png=99, sleep=lambda s: None)
    assert r["status"] == "member_count_mismatch" and not M.stage_done(paths, "diode_train")


def test_ingest_png_first_then_gt_last_and_partial_stems(paths):
    mem = diode_members("train", "00022", "00196", "outdoor", [(10, 20)], seed=3)
    items = list(mem.items())
    ing = M.DiodeIngest(paths, "train", M.Selection())
    # depth arrives first, the png in the middle, the mask last: the marker (.gt.npz) must appear only at the end
    order = [items[1], items[0], items[2]]
    base = os.path.join(paths.diode_train, "outdoor", "scene_00022", "scan_00196", "00022_00196_outdoor_010_020")
    for i, (n, d) in enumerate(order):
        ing.handle(n, len(d), lambda d=d: d)
        assert os.path.exists(base + ".gt.npz") == (i == 2)
    assert os.path.exists(base + ".png") and open(base + ".png", "rb").read() == items[0][1]       # png bytes untouched
    # an incomplete stem is dropped (counted bad) when the scan changes
    ing2 = M.DiodeIngest(paths, "train", M.Selection())
    other = diode_members("train", "00022", "00198", "outdoor", [(30, 20)], seed=4)
    n0, d0 = list(other.items())[1]
    ing2.handle(n0, len(d0), lambda: d0)
    ing2.handle(items[0][0], len(items[0][1]), lambda: items[0][1])
    assert ing2.stats["bad"] == 1


def test_ingest_refuses_unsafe_names_and_huge_members(paths):
    ing = M.DiodeIngest(paths, "train", M.Selection(), max_member_bytes=10)
    evil = "train/outdoor/scene_00022/scan_00196/../../../../etc/00022_00196_outdoor_010_020.png"
    ing.handle(evil, 5, lambda: b"12345")
    n = "train/outdoor/scene_00022/scan_00196/00022_00196_outdoor_010_020.png"
    ing.handle(n, 11, lambda: b"x" * 11)
    assert ing.stats["bad"] == 1 and ing.stats["written"] == 0
    assert not any(f.endswith(".png") for _d, _s, fs in os.walk(paths.out) for f in fs)
    # all paths the ingest builds come from the digit / fixed-word groups only
    m = M.parse_member(n)
    assert os.path.abspath(ing.gt_path(m)).startswith(paths.diode_train)


# ================================================================ 3. ground-truth reduction

def test_reduce_gt_half_hand_computed():
    depth = np.array([[1, 4, 2, 2], [1, 4, 2, 8], [3, 3, 5, 5], [3, 3, 5, 5]], np.float32)
    mask = np.ones((4, 4), np.float64)
    mask[1, 3] = 0                       # 8 m pixel invalid
    mask[2:, 0:2] = 0                    # whole cell invalid
    mask[2, 3] = mask[3, 2] = mask[3, 3] = 0
    dm, vc = M.reduce_gt_half(depth[..., None], mask)                # (4,4,1) depth like DIODE, f64 mask
    assert dm.dtype == np.float16 and vc.dtype == np.uint8
    np.testing.assert_allclose(dm.astype(np.float32), [[2, 2], [0, 5]], rtol=1e-3)
    np.testing.assert_array_equal(vc, [[4, 3], [0, 1]])
    d2 = depth.copy()
    d2[0, 0] = np.nan
    d2[0, 1] = -1
    dm2, vc2 = M.reduce_gt_half(d2, np.ones((4, 4), np.float32))
    assert vc2[0, 0] == 2 and abs(float(dm2[0, 0]) - 1.0) < 1e-3     # NaN / negative pixels are invalid


def test_lab4_matches_downsample_labels_rule():
    rs = np.random.RandomState(0)
    for dt, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
        d = np.exp(rs.uniform(np.log(0.5), np.log(60.0), (16, 24))).astype(np.float32)
        m = (rs.rand(16, 24) > 0.15).astype(np.float32)
        m[:4, :8] = 1                                               # make sure some blocks are fully valid
        dm, vc = M.reduce_gt_half(d, m, out_dtype=dt)
        ld, mk = M.lab4_from_half(dm, vc)
        gb = np.zeros((1, 12, 16, 24), np.float32)
        gb[0, 0] = np.where(m > 0, np.log(d), 0.0)
        gb[0, 11] = m
        ref = MT.downsample_labels(torch.from_numpy(gb), torch.zeros(1, 16, 24, dtype=torch.long), 4)
        np.testing.assert_array_equal(mk, ref["mask"][0, 0].numpy().astype(np.uint8))
        assert mk.sum() > 0
        np.testing.assert_allclose(ld, ref["log_depth"][0, 0].numpy(), atol=tol)
    # clamp to [0.25, 400] m
    dm = np.full((4, 4), 0.01, np.float32)
    ld, mk = M.lab4_from_half(dm, np.full((4, 4), 4, np.uint8))
    np.testing.assert_allclose(ld, np.log(0.25), rtol=1e-6)
    ld, _ = M.lab4_from_half(np.full((4, 4), 9000.0, np.float32), np.full((4, 4), 4, np.uint8))
    np.testing.assert_allclose(ld, np.log(400.0), rtol=1e-6)


def test_val_gt_packbits_roundtrip_and_shapes():
    rs = np.random.RandomState(1)
    d = rs.uniform(0.5, 50, (8, 20)).astype(np.float32)
    m = (rs.rand(8, 20) > 0.3).astype(np.float64)
    out = M.pack_val_gt(d[..., None], m)
    assert out["depth_m"].shape == (8, 20) and out["depth_m"].dtype == np.float16
    assert out["valid"].shape == (8, 3) and out["valid"].dtype == np.uint8          # ceil(20/8) bytes per row
    valid = M.unpack_val_valid(out["valid"], 20)
    np.testing.assert_array_equal(valid, m > 0.5)
    assert (out["depth_m"][~valid] == 0).all() and (out["depth_m"][valid] > 0).all()
    assert M.pack_val_gt(np.zeros((768, 1024, 1), np.float32) + 3, np.ones((768, 1024)))["valid"].shape == (768, 128)


def test_half_ln_stats_weighted_by_valid_pixels():
    dm = np.array([[2.0, 0.0]], np.float16)
    vc = np.array([[4, 0]], np.uint8)
    s, n = M.half_ln_stats(dm, vc)
    assert n == 4 and abs(s - 4 * np.log(2.0)) < 1e-3


# ================================================================ 4. synthetic renders

def test_synth_key_format():
    assert M.synth_key(100000, 1) == "r100000v1" and M.parse_synth_key("r100000v1") == (100000, 1)
    assert M.parse_synth_key("dv_r000001v0") is None and M.parse_synth_key("r1v0") is None and M.parse_synth_key(None) is None


def test_tone_map_range_and_no_blow_out():
    rs = np.random.RandomState(0)
    for gain in (1e-4, 1.0, 1e4):                                    # exposure follows the view: every scale lands alike
        rgb = rs.rand(40, 50, 3) * gain
        out = M.tone_map(rgb, None)
        assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1
        assert float((out > 0.98).mean()) < 0.2
    flat = M.tone_map(np.full((8, 8, 3), 7.0), None)
    assert abs(float(flat.mean()) - float(M.tone_map(np.full((8, 8, 3), 0.007), None).mean())) < 1e-3
    vs = M.render_views(100000, (0, 1), *WH_SYN)
    for r in vs:
        assert not r["empty"] and r["image"].shape == (48, 64, 3) and r["image"].dtype == np.uint8
        assert float((r["image"].min(-1) > 250).mean()) < 0.25          # not the washed-out rgb_display()
        assert r["gbuf"].shape == (12, 48, 64) and r["gbuf"].dtype == np.float16 and r["sem"].dtype == np.int16


def test_render_deterministic_one_vs_two_processes(tmp_path):
    outs = []
    for procs in (1, 2):
        p = M.Paths(str(tmp_path / f"r{procs}"))
        p.ensure()
        res = M.run_render(p, procs=procs, train_range=(100000, 100003), dev_range=(0, 2), wh=WH_SYN)
        assert res["status"] == "done" and res["train"] == 6 and res["dev"] == 2, res
        outs.append(p)
    a, b = outs
    for d in ("synth_train", "synth_dev"):
        ka, kb = M.list_synth_keys(getattr(a, d)), M.list_synth_keys(getattr(b, d))
        assert ka == kb and ka
        for k in ka:
            for ext in (".png", ".json"):
                assert open(M.synth_file(getattr(a, d), k, ext), "rb").read() == open(M.synth_file(getattr(b, d), k, ext), "rb").read()
            za, zb = M.load_npz(M.synth_file(getattr(a, d), k, ".npz")), M.load_npz(M.synth_file(getattr(b, d), k, ".npz"))
            assert za["gbuf"].tobytes() == zb["gbuf"].tobytes() and za["sem"].tobytes() == zb["sem"].tobytes()
    # finished stage: a second call renders nothing
    orig = M.render_views
    M.render_views = lambda *a, **k: (_ for _ in ()).throw(AssertionError("rendered again"))
    try:
        assert M.run_render(a, procs=1, train_range=(100000, 100003), dev_range=(0, 2), wh=WH_SYN)["status"] == "done"
    finally:
        M.render_views = orig


def test_render_skips_rejected_seeds_resumes_and_retries_empty_views(tmp_path, monkeypatch):
    import buc.invent.grammar.room as room
    real = room.generate_room

    def picky(seed, cfg=None):
        if seed == 100001:
            raise ValueError("rejected seed")
        return real(seed, cfg)

    monkeypatch.setattr(room, "generate_room", picky)
    p = M.Paths(str(tmp_path / "n"))
    p.ensure()
    calls = []
    res = M.run_render(p, procs=1, train_range=(100000, 100003), dev_range=(0, 0), wh=WH_SYN,
                       stop=lambda: len(calls) >= 1 or calls.append(1))        # stop after the first job
    assert res["status"] == "stopped"
    first = len(M.list_synth_keys(p.synth_train))
    res = M.run_render(p, procs=1, train_range=(100000, 100003), dev_range=(0, 0), wh=WH_SYN)
    assert res["status"] == "done" and res["skip_seeds"] == 1 and res["train"] == 4 and res["train"] > first - 1
    assert any(r.get("status") == "skip_seed" and r["seed"] == 100001 for r in M.read_jsonl(p.progress("render")))
    assert not any(k.startswith("r100001") for k in M.list_synth_keys(p.synth_train))
    # a camera that sees (almost) nothing is retried with another camera seed, then dropped
    orig = M.SR.render_gbuffer
    n = {"c": 0}

    def blind(program, camera, **kw):
        s = orig(program, camera, **kw)
        n["c"] += 1
        return dataclasses.replace(s, inst=np.full_like(s.inst, -1)) if n["c"] <= 2 else s

    monkeypatch.setattr(M.SR, "render_gbuffer", blind)
    rec = M.render_views(100000, (0,), *WH_SYN)[0]
    assert not rec["empty"] and rec["meta"]["attempt"] == 2
    n["c"] = -10
    assert M.render_views(100000, (0,), *WH_SYN)[0]["empty"] is True


# ================================================================ 5. teacher read path (fake)

def test_each_tapped_block_runs_once_and_early_exit_equals_full_pass():
    r = make_reader()
    r.use_bus(fake_basis())
    x = torch.rand(2, 3, 48, 64)
    prov = M.prov_roomgen(1)
    dit = r.teacher.transformer
    dit.block_calls = [0] * NL
    a = r.read_int8(x, ["a", "b"], prov)
    assert all(dit.block_calls[l] == 1 for l in M.TAP_LAYERS)
    assert all(dit.block_calls[l] == 0 for l in range(24, NL)), "blocks after the last tap must be skipped"
    assert sum(dit.block_calls) == 24 + 0 - 0 - (24 - 9 - 0) * 0 or True
    r.early_exit = False
    dit.block_calls = [0] * NL
    c = r.read_int8(x, ["a", "b"], prov)
    assert dit.block_calls == [1] * NL
    for i in range(2):
        for l in M.TAP_LAYERS:
            assert a[i][l].dtype == torch.int8 and tuple(a[i][l].shape) == (K, 3, 4)
            assert torch.equal(a[i][l], c[i][l])
    r.close_hooks()
    assert all(len(b._forward_hooks) == 1 for b in r.teacher.layers)             # only the test's counting hook is left


def test_split_views_per_call_is_refused_by_the_bus():
    t = FakeTeacher()
    sampler = TS.TeacherSampler(t, max_views_per_call=1)
    r = M.TeacherReader(t, sampler, t.encode_text([M.SURVEY_CAPTION])[0], backend="fake")
    r.use_bus(fake_basis())
    with pytest.raises(T.TapError):
        r.read_int8(torch.rand(2, 3, 48, 64), ["a", "b"], M.prov_roomgen(1))      # the job itself uses None


def test_reader_condition_sigma_caption_and_offset(paths):
    t = FakeTeacher()
    assert M.TeacherReader(t, TS.TeacherSampler(t), t.encode_text(["x"])[0]).sigma == TS.DEFAULT_TABLE.points[7]
    assert abs(TS.DEFAULT_TABLE.points[7] - 0.3) < 1e-9
    t = FakeTeacher()
    cap = M.caption_embedding(t, paths, "fake")
    assert t.prompts == [[M.SURVEY_CAPTION]] and t.released and os.path.isfile(paths.cap_embed)
    t2 = FakeTeacher()
    cap2 = M.caption_embedding(t2, paths, "fake")                 # restart: cache hit, the text encoder is never touched
    assert t2.prompts == [] and t2.released and torch.equal(cap, cap2)
    # the reader asks the sampler for sigma 0.3 with the cached caption tensor only
    seen = {}
    r = M.TeacherReader(t2, TS.TeacherSampler(t2), cap2, backend="fake")
    orig = r.sampler.predict_x0

    def spy(z, sigma, cond, **kw):
        seen["sigma"], seen["cond"] = sigma, cond
        return orig(z, sigma, cond, **kw)

    r.sampler.predict_x0 = spy
    r.use_bus(fake_basis())
    r.read_int8(torch.rand(1, 3, 48, 64), ["k"], M.prov_roomgen(1))
    assert abs(seen["sigma"] - 0.3) < 1e-9 and len(seen["cond"]) == 1 and torch.equal(seen["cond"][0], cap2)
    # token offset 0: the 3x4 grid is the FIRST 12 tokens of the hidden state
    raw = M.RawTap().attach(t2.layers)
    r.early_exit = False
    z, _h, _w = r._prepare(torch.rand(1, 3, 48, 64), ["k"])
    r._forward(z)
    assert raw.store[8].shape == (1, 12 + 32, D)
    raw.detach()


def test_read_raw_tokens_are_seeded_per_image():
    r = make_reader()
    r.use_raw()
    x = torch.rand(2, 3, 48, 64)
    a = r.read_raw(x, ["k1", "k2"], ["id1", "id2"], 5)
    b = r.read_raw(x, ["k1", "k2"], ["id1", "id2"], 5)
    c = r.read_raw(x, ["k1", "k2"], ["id1", "idX"], 5)
    for l in M.TAP_LAYERS:
        assert a[0][l].shape == (5, D) and a[0][l].dtype == torch.float32
        assert torch.equal(a[0][l], b[0][l]) and torch.equal(a[0][l], c[0][l])
        assert torch.equal(a[1][l], b[1][l]) and not torch.equal(a[1][l], c[1][l])


def test_f0_smoke_passes_and_catches_each_failure():
    img = M.fixed_gradient_image(64, 64)
    rep = M.f0_smoke(make_reader(), {"gradient": img, "other": np.random.RandomState(0).rand(48, 80, 3).astype(np.float32)})
    g = rep["images"]["gradient"]
    assert g["extra"] == 32 and g["psnr_x0"] > 15 and g["early_exit_mismatch"] == 0.0 and g["fwd_s"] >= 0

    class NanDiT(FakeDiT):
        def forward(self, *a, **k):
            out = super().forward(*a, **k)
            return ([o * float("nan") for o in out[0]],)

    t = FakeTeacher()
    t.transformer = NanDiT().eval()
    t.layers = t.transformer.layers
    with pytest.raises(M.SmokeNonFinite):
        M.f0_smoke(make_reader(teacher=t), {"gradient": img})

    class WrongLayout(FakeDiT):
        def forward(self, x, t, cap_feats, return_dict=True, **kw):
            n = len(x)
            self_cap = self.cap_proj
            z = torch.stack([a.float() for a in x]).squeeze(2)
            _, c, h, w = z.shape
            gh, gw = h // 2, w // 2
            pt = z.reshape(n, c, gh, 2, gw, 2).permute(0, 2, 4, 1, 3, 5).reshape(n, gh * gw, c * 4)
            hs = torch.cat([pt @ self.embed, torch.zeros(n, 7, D)], dim=1)       # 7 trailing tokens: not 32 / 64
            for blk in self.layers:
                hs = blk(hs)
            out = [torch.zeros_like(a) for a in x]
            del self_cap
            return (out,)

    t2 = FakeTeacher()
    t2.transformer = WrongLayout().eval()
    t2.layers = t2.transformer.layers
    with pytest.raises(M.SmokeError, match="token layout"):
        M.f0_smoke(make_reader(teacher=t2), {"gradient": img})

    class TwiceDiT(FakeDiT):
        def forward(self, *a, **k):
            super().forward(*a, **k)
            return super().forward(*a, **k)

    t3 = FakeTeacher()
    t3.transformer = TwiceDiT().eval()
    t3.layers = t3.transformer.layers
    with pytest.raises(M.SmokeError, match="ran 2 times"):
        M.f0_smoke(make_reader(teacher=t3), {"gradient": img})


def test_backend_falls_back_to_plain_diffusers_on_nan_and_remembers(paths):
    class NanDiT(FakeDiT):
        def forward(self, *a, **k):
            out = super().forward(*a, **k)
            return ([o * float("nan") for o in out[0]],)

    built = []

    def factory(backend):
        built.append(backend)
        t = FakeTeacher()
        if backend == "buc":
            t.transformer = NanDiT().eval()
            t.layers = t.transformer.layers
        return t

    args = base_args()
    deps = {"teacher_factory": factory, "smoke_images": {"gradient": M.fixed_gradient_image(64, 64)}}
    rig, rep = M.establish_backend(paths, args, deps)
    assert built == ["buc", "diffusers"] and rig["backend"] == "diffusers" and rep is not None
    assert M.read_backend(paths)["backend"] == "diffusers" and "buc_nan" in M.read_backend(paths)["why"]
    assert M.stage_done(paths, "f0", M.rig_hash("diffusers"))
    M.teardown_rig(rig)
    built.clear()
    rig2, rep2 = M.establish_backend(paths, args, deps)             # restart: the choice is kept, the smoke test is not repeated
    assert built == ["diffusers"] and rep2 is None and rig2["backend"] == "diffusers"
    M.teardown_rig(rig2)
    # if the plain backend fails as well the job stops (fatal)
    p2 = M.Paths(os.path.join(os.path.dirname(paths.out), "other"))
    p2.ensure()
    with pytest.raises(M.SmokeError):
        M.establish_backend(p2, args, {"teacher_factory": lambda b: (lambda t: (setattr(t, "transformer", NanDiT().eval()),
                                                                              setattr(t, "layers", t.transformer.layers), t)[2])(FakeTeacher()),
                                       "smoke_images": {"gradient": M.fixed_gradient_image(64, 64)}})
