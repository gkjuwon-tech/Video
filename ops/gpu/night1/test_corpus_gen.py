"""CPU tests for corpus_gen.py (no GPU, network, diffusers or transformers)."""
import json
import os
import re
import sys
import tarfile
import types
from collections import Counter

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_gen as cg  # noqa: E402


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def fake_gen_factory(clock=None, tick=0.0, fail_on=()):
    calls = []

    def gen(prompt, h, w, seed):
        calls.append(seed)
        if clock is not None:
            clock[0] += tick
        if seed in fail_on:
            raise RuntimeError("boom")
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 255, (h // 16, w // 16, 3), dtype=np.uint8)  # tiny stand-in
        return Image.fromarray(arr)

    gen.calls = calls
    return gen


def fake_encode(imgs):
    return np.stack([
        np.full((4, im.height // 8, im.width // 8), 0.5, dtype=np.float32) for im in imgs])


def make_job(root, **kw):
    kw.setdefault("uploader", None)
    kw.setdefault("shard_size", 5)
    kw.setdefault("batch", 2)
    kw.setdefault("upload_threads", False)
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("free_fn", lambda p: 10**12)
    gen = kw.pop("gen_fn", None) or fake_gen_factory()
    enc = kw.pop("encode_fn", fake_encode)
    return cg.CorpusJob(str(root), gen, enc, **kw), gen


# ---------------------------------------------------------------------------
# prompt grammar
# ---------------------------------------------------------------------------

def test_prompt_deterministic_and_seed_is_index():
    a = [cg.build_prompt(i) for i in range(300)]
    b = [cg.build_prompt(i) for i in range(300)]
    assert a == b
    assert all(p["seed"] == p["index"] == i for i, p in enumerate(a))
    assert len({p["prompt"] for p in a}) > 250  # varied


def test_category_and_size_ratios():
    cats = Counter(cg.category_for(i) for i in range(1000))
    assert cats == {"indoor": 300, "landscape": 200, "hair": 200, "material": 150, "surreal": 150}
    sizes = Counter(cg.size_for(i) for i in range(1000))
    assert sizes[(1024, 1024)] == 700 and sizes[(576, 1344)] == 300
    for (h, w) in sizes:
        assert h % 8 == 0 and w % 8 == 0
    # not the same pattern in every block
    assert [cg.category_for(i) for i in range(100)] != [cg.category_for(i) for i in range(100, 200)]


def test_prompts_have_no_forbidden_instructions():
    banned = ["text", "logo", "brand", "sign", "letter", "caption", "poster", "watermark",
              "celebrity", "actor", "president"]
    for i in range(1500):
        p = cg.build_prompt(i)["prompt"].lower()
        words = set(re.findall(r"[a-z0-9]+", p))
        for b in banned:
            assert b not in words and b + "s" not in words, (i, b, p)
    for i in range(300):
        p = cg.build_prompt(i)
        if p["category"] == "surreal":
            assert "small" in p["prompt"] or "tiny" in p["prompt"] or "minuscule" in p["prompt"]
        if p["w"] > p["h"]:
            assert "cinemascope" in p["prompt"]


def test_print_prompts_cli(capsys):
    assert cg.main(["--print-prompts", "4"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["index"] == 0


# ---------------------------------------------------------------------------
# VAE encoding
# ---------------------------------------------------------------------------

class FakeVAE:
    def __init__(self, scale=0.5, shift=0.25, channels=16):
        self.config = types.SimpleNamespace(scaling_factor=scale, shift_factor=shift,
                                            latent_channels=channels)
        self.batches = []
        self.last_x = None

    def encode(self, x):
        import torch
        self.batches.append(x.shape[0])
        self.last_x = x
        b, _, h, w = x.shape
        mean = torch.ones(b, self.config.latent_channels, h // 8, w // 8)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mean=mean))


def test_encode_images_normalization_batches_and_range():
    vae = FakeVAE(scale=0.5, shift=0.25, channels=16)
    imgs = [Image.new("RGB", (64, 32), (255, 0, 128)) for _ in range(19)]
    lat = cg.encode_images(vae, imgs, max_batch=8)
    assert lat.shape == (19, 16, 4, 8) and lat.dtype == np.float16
    assert np.allclose(lat, (1.0 - 0.25) * 0.5)
    assert vae.batches == [8, 8, 3]
    x = vae.last_x
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert abs(float(x[0, 0, 0, 0]) - 1.0) < 1e-6 and abs(float(x[0, 1, 0, 0]) + 1.0) < 1e-6


def test_encode_images_config_defaults():
    vae = FakeVAE(channels=3)
    vae.config = types.SimpleNamespace(latent_channels=3)  # no scale / shift
    lat = cg.encode_images(vae, [Image.new("RGB", (16, 16))])
    assert np.allclose(lat, 1.0) and lat.shape == (1, 3, 2, 2)


def test_check_latent():
    im = Image.new("RGB", (64, 32))
    ok = np.zeros((4, 4, 8), np.float16)
    assert cg.check_latent(ok, im, 4) is None
    assert cg.check_latent(ok, im, 16)
    bad = ok.copy()
    bad[0, 0, 0] = np.nan
    assert cg.check_latent(bad, im) == "non_finite"
    assert cg.check_latent(np.zeros((4, 3, 8), np.float16), im)


# ---------------------------------------------------------------------------
# layout / resume / shards
# ---------------------------------------------------------------------------

def test_layout_and_files(tmp_path):
    job, _ = make_job(tmp_path)
    job.end = 7
    res = job.run()
    assert res["reason"] == "complete" and res["n"] == 7
    for i in range(7):
        p = cg.item_paths(str(tmp_path), i)
        assert os.path.basename(p["img"]) == f"{i:08d}.webp"
        assert all(os.path.exists(v) for v in p.values())
        lat = np.load(p["lat"])
        assert lat.dtype == np.float16 and lat.shape[0] == 4
        meta = json.load(open(p["meta"], encoding="utf-8"))
        spec = cg.build_prompt(i)
        for k in ("prompt", "seed", "h", "w", "category"):
            assert meta[k] == spec[k]
        assert meta["t_gen"] >= 0
        with Image.open(p["img"]) as im:
            assert im.format == "WEBP"
    assert not [n for d in ("img", "lat", "meta", "shards") for n in os.listdir(tmp_path / d)
                if n.endswith(".tmp")]


def test_resume_no_duplicates_no_regeneration(tmp_path):
    job, gen = make_job(tmp_path, max_images=12)
    job.run()
    assert len(gen.calls) == 12
    before = {f: os.path.getmtime(tmp_path / "meta" / f) for f in os.listdir(tmp_path / "meta")}
    job2, gen2 = make_job(tmp_path, max_images=12)
    res = job2.run()
    assert gen2.calls == [] and res["n"] == 12 and res["new"] == 0
    assert len(os.listdir(tmp_path / "meta")) == 12
    assert before == {f: os.path.getmtime(tmp_path / "meta" / f) for f in os.listdir(tmp_path / "meta")}
    # partially deleted item is regenerated alone
    os.remove(tmp_path / "meta" / f"{3:08d}.json")
    job3, gen3 = make_job(tmp_path, max_images=12)
    job3.run()
    assert gen3.calls == [3]


def test_shard_tar_contents_and_upload(tmp_path):
    uploads = []
    job, _ = make_job(tmp_path, max_images=10, uploader=lambda l, r: uploads.append((l, r)))
    res = job.run()
    assert res["last_shard"] == "gen_0001.tar"
    assert [r for _, r in uploads] == ["corpus/night1/gen_0000.tar", "corpus/night1/gen_0001.tar"]
    with tarfile.open(tmp_path / "shards" / "gen_0000.tar") as t:
        names = sorted(t.getnames())
    exp = sorted([f"img/{i:08d}.webp" for i in range(5)] + [f"lat/{i:08d}.npy" for i in range(5)]
                 + [f"meta/{i:08d}.json" for i in range(5)])
    assert names == exp
    # originals stay on disk
    assert (tmp_path / "img" / f"{0:08d}.webp").exists()
    # second run: nothing re-uploaded, tar untouched
    job2, _ = make_job(tmp_path, max_images=10, uploader=lambda l, r: uploads.append((l, r)))
    job2.run()
    assert len(uploads) == 2


def test_threaded_upload_path(tmp_path):
    uploads = []
    job, _ = make_job(tmp_path, max_images=10, upload_threads=True,
                      uploader=lambda l, r: uploads.append(r))
    job.run()
    assert sorted(uploads) == ["corpus/night1/gen_0000.tar", "corpus/night1/gen_0001.tar"]
    assert (tmp_path / "shards" / "gen_0001.tar.uploaded").exists()


def test_upload_failure_is_survived_and_retried_next_run(tmp_path, capsys):
    state = {"fail": True, "ok": []}

    def up(local, remote):
        if state["fail"]:
            raise RuntimeError("network down hf_abcdefghijkl123")
        state["ok"].append(remote)

    job, _ = make_job(tmp_path, max_images=10, uploader=up)
    res = job.run()
    out = capsys.readouterr().out
    assert res["n"] == 10 and res["reason"] == "complete"
    assert "upload_failed" in out and "hf_abcdefghijkl123" not in out
    assert not list((tmp_path / "shards").glob("*.uploaded"))
    state["fail"] = False
    job2, gen2 = make_job(tmp_path, max_images=10, uploader=up)
    job2.run()
    assert gen2.calls == []
    assert sorted(state["ok"]) == ["corpus/night1/gen_0000.tar", "corpus/night1/gen_0001.tar"]


def test_item_failures_are_skipped(tmp_path, capsys):
    job, gen = make_job(tmp_path, max_images=10, gen_fn=fake_gen_factory(fail_on={2, 7}))
    res = job.run()
    assert res["n"] == 8 and res["failed"] == 2 and res["reason"] == "complete"
    assert "gen_failed" in capsys.readouterr().out
    # shards still packed with what exists
    with tarfile.open(tmp_path / "shards" / "gen_0000.tar") as t:
        assert len(t.getnames()) == 4 * 3
    # next run retries just the holes
    job2, gen2 = make_job(tmp_path, max_images=10)
    job2.run()
    assert sorted(gen2.calls) == [2, 7]
    with tarfile.open(tmp_path / "shards" / "gen_0000.tar") as t:
        assert len(t.getnames()) == 5 * 3  # rebuilt after hole filled


def test_encode_failure_isolated_and_bad_latent_skipped(tmp_path):
    def enc(imgs):
        if len(imgs) > 1:
            raise RuntimeError("batch oom")
        return fake_encode(imgs)

    job, _ = make_job(tmp_path, max_images=6, encode_fn=enc)
    res = job.run()
    assert res["n"] == 6  # batch failure falls back to single encodes

    def nan_enc(imgs):
        a = fake_encode(imgs)
        a[0] = np.nan
        return a

    d2 = tmp_path / "b"
    job2, _ = make_job(d2, max_images=4, encode_fn=nan_enc, batch=1)
    assert job2.run()["n"] == 0


def test_deadline_stops_and_packs_partial(tmp_path):
    clock = [1000.0]
    job, gen = make_job(tmp_path, gen_fn=fake_gen_factory(clock, tick=10.0), now=lambda: clock[0],
                        deadline=1000.0 + 45.0, margin=5.0, shard_size=100, max_images=1000,
                        uploader=lambda l, r: None)
    res = job.run()
    assert res["reason"] == "deadline"
    assert 3 <= res["n"] <= 5 and len(gen.calls) == res["n"]
    assert (tmp_path / "shards" / "gen_0000_partial.tar").exists()
    assert (tmp_path / "shards" / "gen_0000_partial.tar.uploaded").exists()
    # resume later continues without redoing
    job2, gen2 = make_job(tmp_path, max_images=res["n"], shard_size=100)
    job2.run()
    assert gen2.calls == []


def test_deadline_already_passed_does_nothing(tmp_path):
    job, gen = make_job(tmp_path, now=lambda: 100.0, deadline=50.0, max_images=10)
    res = job.run()
    assert res["reason"] == "deadline" and gen.calls == []


def test_disk_guard(tmp_path, capsys):
    free = {"v": 10**12}

    def free_fn(_):
        return free["v"]

    gen = fake_gen_factory()

    def gen2(*a):
        r = gen(*a)
        if len(gen.calls) >= 4:
            free["v"] = 7 * 2**30
        return r

    job, _ = make_job(tmp_path, gen_fn=gen2, free_fn=free_fn, max_images=100)
    res = job.run()
    out = capsys.readouterr().out
    assert res["reason"] == "disk" and res["n"] < 100
    assert 'N1_GEN {"stop": "disk"}' in out


def test_progress_markers(tmp_path, capsys):
    clock = [0.0]
    job, _ = make_job(tmp_path, gen_fn=fake_gen_factory(clock, tick=100.0), now=lambda: clock[0],
                      progress_every=250.0, max_images=10)
    job.run()
    lines = [json.loads(l[len("N1_GEN "):]) for l in capsys.readouterr().out.splitlines()
             if l.startswith("N1_GEN ")]
    prog = [d for d in lines if "rate_img_per_min" in d]
    assert len(prog) >= 2
    for k in ("n", "rate_img_per_min", "vram_GB", "last_shard"):
        assert k in prog[0]


def test_consecutive_failures_abort(tmp_path):
    def always_fail(*a):
        raise RuntimeError("dead")

    job, _ = make_job(tmp_path, gen_fn=always_fail, max_images=1000, max_consecutive_fail=10)
    res = job.run()
    assert res["reason"] == "errors" and res["failed"] == 10


def test_stop_request(tmp_path):
    job, gen = make_job(tmp_path, max_images=100)
    job.request_stop()
    assert job.run()["reason"] == "signal" and gen.calls == []


# ---------------------------------------------------------------------------
# CLI / secrets
# ---------------------------------------------------------------------------

def test_main_without_token_logs_hf_token_no(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("N1_DEADLINE", raising=False)
    rc = cg.main(["--root", str(tmp_path), "--max-images", "3", "--shard-size", "2"],
                 deps={"gen_fn": fake_gen_factory(), "encode_fn": fake_encode,
                       "free_fn": lambda p: 10**12, "upload_threads": False})
    out = capsys.readouterr().out
    assert rc == 0 and '"hf_token": "no"' in out
    assert len(os.listdir(tmp_path / "meta")) == 3


def test_main_never_prints_token(tmp_path, monkeypatch, capsys):
    secret = "hf_SUPERSECRETVALUE1234567890"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("N1_DEADLINE", str(10**10))

    def bad_up(local, remote):
        raise RuntimeError(f"401 for token {secret}")

    rc = cg.main(["--root", str(tmp_path), "--max-images", "4", "--shard-size", "2"],
                 deps={"gen_fn": fake_gen_factory(), "encode_fn": fake_encode, "uploader": bad_up,
                       "sleep": lambda s: None, "free_fn": lambda p: 10**12,
                       "upload_threads": False, "upload_backoff": 0.0})
    cap = capsys.readouterr()
    assert rc == 0
    assert secret not in cap.out and secret not in cap.err
    assert '"hf_token": "yes"' in cap.out
    for f in tmp_path.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".uploaded"):
            assert secret not in f.read_text(errors="ignore")


def test_deadline_parse():
    assert cg.parse_deadline("123.5") == 123.5
    assert cg.parse_deadline(None) is None and cg.parse_deadline("x") is None


def test_scrub():
    assert "hf_abcdefghijk" not in cg.scrub("err hf_abcdefghijk end")
