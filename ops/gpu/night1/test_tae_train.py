"""CPU tests for tae_train.py (no GPU, no network, no diffusers/transformers)."""
import json
import math
import os
import random
import sys
import threading

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tae_train as T  # noqa: E402

C = 4          # latent channels used by the fake corpus (the code must not hardcode 16)
SZ = 64        # fake image size


def make_corpus(root, n, size=SZ, c=C, start=0, wide=False):
    os.makedirs(os.path.join(root, "img"), exist_ok=True)
    os.makedirs(os.path.join(root, "lat"), exist_ok=True)
    rs = np.random.RandomState(start)
    for i in range(start, start + n):
        w, h = (size * 2, size) if wide else (size, size)
        arr = (rs.rand(h, w, 3) * 255).astype(np.uint8)
        arr[:, : w // 2] = (arr[:, : w // 2] // 2 + 60)          # some structure
        Image.fromarray(arr).save(os.path.join(root, "img", f"{i:08d}.webp"), quality=90)
        np.save(os.path.join(root, "lat", f"{i:08d}.npy"),
                rs.randn(c, h // 8, w // 8).astype(np.float16))


def fast_args(root, **kw):
    argv = ["--root", str(root), "--min-pairs", "8", "--eval-n", "3", "--batch", "4",
            "--crop", "32", "--workers", "0", "--warmup", "2", "--no-upload",
            "--first-eval-s", "1000000", "--eval-every-s", "1000000",
            "--log-every-s", "1000000", "--epoch-min-steps", "2", "--poll-s", "0.01",
            "--amp", "off", "--device", "cpu"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return T.build_parser().parse_args(argv)


def markers(out, prefix="N1_TAE"):
    res = []
    for line in out.splitlines():
        head, _, rest = line.partition(" ")
        if head == prefix:
            res.append(json.loads(rest))
    return res


class Clock:
    """Fake clock: every read advances time."""

    def __init__(self, step=1.0, t=1000.0):
        self.t, self.step = t, step

    def __call__(self):
        self.t += self.step
        return self.t


# ---------------------------------------------------------------- model / loss

def test_param_count_in_range_and_not_hardcoded_channels():
    for c in (4, 16, 32):
        w, n = T.fit_widths(c)
        assert T.PARAM_MIN <= n <= T.PARAM_MAX
        assert T.count_params(T.TAE(c, w)) == n
    assert 2.0e6 <= T.count_params(T.TAE(16)) <= 3.2e6


def test_fit_widths_scales_when_default_does_not_fit():
    w, n = T.fit_widths(16, lo=3_300_000, hi=4_200_000)
    assert n >= 3_300_000 and all(x % 8 == 0 for x in w)


def test_forward_shapes_1_8_latent_and_back():
    m = T.TAE(C)
    x = torch.randn(2, 3, 64, 96)
    z = m.enc(x)
    assert z.shape == (2, C, 8, 12)
    assert m.dec(z).shape == x.shape


def test_sobel_ramp():
    ramp = torch.arange(16.0).view(1, 1, 1, 16).expand(1, 1, 16, 16).contiguous()
    g = T.sobel(ramp)
    assert g.shape == (1, 2, 16, 16)
    assert torch.allclose(g[0, 0, 4:12, 4:12], torch.full((8, 8), 8.0))
    assert torch.allclose(g[0, 1], torch.zeros(16, 16))


def test_lr_schedule():
    assert T.lr_at(0, 1e-3, 1000) == pytest.approx(1e-6)
    assert T.lr_at(999, 1e-3, 1000) == pytest.approx(1e-3)
    assert T.lr_at(1000, 1e-3, 1000) == pytest.approx(1e-3)
    mid = T.lr_at(1000 + (T.TOTAL_STEPS - 1000) // 2, 1e-3, 1000)
    assert 0.45e-3 < mid < 0.56e-3
    assert T.lr_at(10**9, 1e-3, 1000) == pytest.approx(1e-5)
    assert T.TOTAL_STEPS == 200_000


def test_one_step_losses_finite_and_decrease():
    torch.manual_seed(0)
    m = T.TAE(C)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.rand(4, 3, 32, 32) * 2 - 1
    z = torch.randn(4, C, 4, 4)
    first = None
    for _ in range(8):
        out = T.compute_losses(m, x, z)
        assert all(torch.isfinite(v) for v in out.values())
        assert float(out["loss"]) == pytest.approx(float(out["l_enc"] + out["l_dec"]), rel=1e-5)
        if first is None:
            first = float(out["loss"])
        assert T.optimizer_step(m, opt, out["loss"])
    assert float(T.compute_losses(m, x, z)["loss"]) < first


def test_nonfinite_step_is_skipped():
    m = T.TAE(C)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in m.parameters()]
    x = torch.rand(2, 3, 32, 32)
    x[0, 0, 0, 0] = float("nan")
    out = T.compute_losses(m, x, torch.randn(2, C, 4, 4))
    assert T.optimizer_step(m, opt, out["loss"]) is False
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_psnr():
    x = torch.zeros(1, 3, 8, 8)
    assert T.psnr(x, x) > 90
    y = torch.full_like(x, 0.2)             # in [0,1] terms error is 0.1
    assert T.psnr(y, x) == pytest.approx(20.0, abs=1e-3)


# ---------------------------------------------------------------- crops / data

def test_crop_alignment_and_flip():
    rng = random.Random(1)
    for _ in range(200):
        y, x = T.random_crop_origin(100, 250, 64, rng)
        assert y % 8 == 0 and x % 8 == 0 and y + 64 <= 100 and x + 64 <= 250
    arr = np.arange(40 * 48 * 3, dtype=np.uint8).reshape(40, 48, 3)
    lat = np.arange(C * 5 * 6, dtype=np.float16).reshape(C, 5, 6)
    a, z = T.crop_pair(arr, lat, 8, 16, 16, False)
    assert a.shape == (3, 16, 16) and z.shape == (C, 2, 2)
    assert np.array_equal(a, arr[8:24, 16:32].transpose(2, 0, 1))
    assert np.array_equal(z, lat[:, 1:3, 2:4])
    af, zf = T.crop_pair(arr, lat, 8, 16, 16, True)
    assert np.array_equal(af, a[:, :, ::-1]) and np.array_equal(zf, z[:, :, ::-1])
    with pytest.raises(ValueError):
        T.crop_pair(arr, lat, 3, 0, 16, False)
    with pytest.raises(ValueError):
        T.random_crop_origin(20, 20, 32, rng)


def test_list_pairs_ignores_tmp_and_halves(tmp_path):
    make_corpus(tmp_path, 5)
    (tmp_path / "img" / "00000099.webp.tmp").write_bytes(b"x")
    (tmp_path / "img" / "00000050.webp").write_bytes(b"x")      # img without lat
    (tmp_path / "lat" / "00000060.npy").write_bytes(b"x")       # lat without img
    assert T.list_pairs(str(tmp_path)) == [0, 1, 2, 3, 4]


def test_eval_indices_frozen_and_excluded(tmp_path):
    avail = list(range(100))
    assert T.choose_eval_indices(avail, 64) == list(range(36, 100))
    d = str(tmp_path)
    first = T.load_or_make_eval_indices(d, avail, 5)
    assert first == [95, 96, 97, 98, 99]
    again = T.load_or_make_eval_indices(d, avail + list(range(100, 150)), 5)
    assert again == first                                       # frozen even as the corpus grows


def test_epoch_sequence():
    rng = random.Random(0)
    seq = T.build_epoch_sequence(list(range(10)), 4, rng, min_steps=3, max_steps=5)
    assert len(seq) == 12 and set(seq) == set(range(10))
    seq = T.build_epoch_sequence(list(range(100)), 4, rng, min_steps=3, max_steps=5)
    assert len(seq) == 20
    assert T.build_epoch_sequence([], 4, rng) == []


def test_dataset_item_and_corrupt_items_skipped(tmp_path, capsys):
    make_corpus(tmp_path, 4)
    (tmp_path / "img" / "00000001.webp").write_bytes(b"not an image")
    np.save(tmp_path / "lat" / "00000002.npy", np.zeros((C + 1, 8, 8), np.float16))  # wrong channels
    ds = T.PairDataset(str(tmp_path), [0, 1, 2, 3], crop=32, crops_per_image=2, latent_channels=C, seed=3)
    for k in range(4):
        x, z = ds[k]
        assert x.shape == (2, 3, 32, 32) and x.dtype == torch.uint8
        assert z.shape == (2, C, 4, 4) and z.dtype == torch.float16
    assert "bad_item" in capsys.readouterr().out
    ds2 = T.PairDataset(str(tmp_path), [1], crop=32, latent_channels=C)
    assert ds2[0] is None
    assert T.collate([None, None]) is None
    x, z = T.collate([ds[0], None, ds[3]])
    assert x.shape[0] == 4


def test_dataset_is_deterministic(tmp_path):
    make_corpus(tmp_path, 3)
    a = T.PairDataset(str(tmp_path), [0, 1, 2], crop=32, seed=7)[1]
    b = T.PairDataset(str(tmp_path), [0, 1, 2], crop=32, seed=7)[1]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_too_small_image_rejected(tmp_path):
    make_corpus(tmp_path, 1, size=24)
    assert T.PairDataset(str(tmp_path), [0], crop=32)[0] is None


# ---------------------------------------------------------------- eval / ckpt

def test_evaluate_full_res_and_samples(tmp_path):
    make_corpus(tmp_path, 3, wide=True)
    es = T.EvalSet(str(tmp_path), [0, 1, 2], C)
    m = T.TAE(C)
    metrics, samples = T.evaluate(m, es, torch.device("cpu"))
    assert metrics["n"] == 3
    assert all(math.isfinite(metrics[k]) for k in ("psnr_dec_zt", "psnr_dec_enc", "latent_mse"))
    assert len(samples) == 3 and samples[0][0].shape == (SZ, 2 * SZ, 3)
    grid = T.make_sample_grid(samples, tile=64)
    assert grid.size == (3 * 64, 3 * 64)
    assert m.training                      # mode restored


def test_evaluate_skips_bad_items(tmp_path, capsys):
    make_corpus(tmp_path, 3)
    (tmp_path / "img" / "00000001.webp").write_bytes(b"junk")
    metrics, _ = T.evaluate(T.TAE(C), T.EvalSet(str(tmp_path), [0, 1, 2], C), torch.device("cpu"))
    assert metrics["n"] == 2
    assert "bad_eval_item" in capsys.readouterr().out
    none, _ = T.evaluate(T.TAE(C), T.EvalSet(str(tmp_path), [1], C), torch.device("cpu"))
    assert none is None


def test_sample_positions():
    assert T.sample_positions(64, 4) == [0, 21, 42, 63]
    assert T.sample_positions(2, 4) == [0, 1]
    assert T.sample_positions(0) == []


def test_checkpoint_atomic_roundtrip_and_corrupt(tmp_path):
    p = str(tmp_path / "ckpt_last.pt")
    m = T.TAE(C)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    T.optimizer_step(m, opt, T.compute_losses(m, torch.rand(2, 3, 32, 32), torch.randn(2, C, 4, 4))["loss"])
    T.save_atomic({"step": 7, "model": m.state_dict(), "opt": opt.state_dict(),
                   "config": m.config(), "best_psnr": 12.5}, p)
    assert os.listdir(tmp_path) == ["ckpt_last.pt"]             # no temp file left behind
    ck = T.load_ckpt(p)
    assert ck["step"] == 7 and ck["best_psnr"] == 12.5
    m2 = T.TAE(C)
    m2.load_state_dict(ck["model"])
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    opt2.load_state_dict(ck["opt"])
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters()))
    (tmp_path / "bad.pt").write_bytes(b"garbage")
    assert T.load_ckpt(str(tmp_path / "bad.pt")) is None
    assert T.load_ckpt(str(tmp_path / "missing.pt")) is None


def test_stage_upload_and_retry(tmp_path):
    tae = tmp_path / "tae"
    (tae / "samples").mkdir(parents=True)
    for n in ("ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl"):
        (tae / n).write_bytes(b"1")
    (tae / "samples" / "step_5.jpg").write_bytes(b"a")
    (tae / "samples" / "step_20.jpg").write_bytes(b"b")
    stage, names = T.stage_upload(str(tae))
    assert sorted(names) == ["ckpt_best.pt", "ckpt_last.pt", "metrics.jsonl", "sample_latest.jpg"]
    assert (open(os.path.join(stage, "sample_latest.jpg"), "rb").read()) == b"b"
    calls = []

    def flaky(folder, remote):
        calls.append(remote)
        if len(calls) < 3:
            raise RuntimeError("boom hf_abcdefghijklmnop")

    sleeps = []
    assert T.upload_with_retry(flaky, stage, "x/y", sleep=sleeps.append) is True
    assert len(calls) == 3 and len(sleeps) == 2

    def dead(folder, remote):
        raise RuntimeError("down")

    assert T.upload_with_retry(dead, stage, "x/y", sleep=lambda s: None) is False


def test_uploader_needs_token_and_scrub():
    assert T.make_uploader(None) is None and T.make_uploader("") is None
    assert callable(T.make_uploader("hf_dummytoken123456"))
    assert "hf_SECRET" not in T.scrub("err hf_SECRETSECRET99 here", token="hf_SECRETSECRET99")
    assert "abcdefghijkl" not in T.scrub("hf_abcdefghijkl")


def test_parse_deadline():
    assert T.parse_deadline(None) is None and T.parse_deadline("") is None
    assert T.parse_deadline("1700000000") == 1700000000.0
    assert T.parse_deadline("junk") is None


# ---------------------------------------------------------------- end to end

def run_main(argv, **deps):
    return T.main(argv, deps)


def test_train_resume_no_duplicates_and_markers(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETTOKEN123456")
    root = str(tmp_path)
    make_corpus(root, 12)
    (tmp_path / "img" / "00000003.webp").write_bytes(b"corrupt")       # one bad image mid-run
    uploads = []
    clock = Clock(step=0.0)

    args = ["--root", root, "--min-pairs", "8", "--eval-n", "3", "--batch", "4", "--crop", "32",
            "--workers", "0", "--warmup", "2", "--first-eval-s", "5", "--eval-every-s", "5",
            "--log-every-s", "3", "--epoch-min-steps", "2", "--amp", "off", "--device", "cpu",
            "--max-steps", "6", "--poll-s", "0.01"]

    def up(folder, remote):
        uploads.append((remote, sorted(os.listdir(folder))))

    class Tick(Clock):
        def __call__(self):
            self.t += 0.5
            return self.t

    assert run_main(args, uploader=up, now=Tick()) == 0
    out1 = capsys.readouterr().out
    assert "hf_SECRETTOKEN" not in out1
    ms = markers(out1)
    assert any("params" in m and m["in_range"] for m in ms)
    assert any("img_per_s" in m and "loss" in m and "l_enc" in m and "l_dec" in m for m in ms)
    evs = markers(out1, "N1_TAE_EVAL")
    assert evs and all(math.isfinite(e["psnr_dec_enc"]) for e in evs)
    done = markers(out1, "N1_TAE_DONE")[-1]
    assert done["status"] == "max_steps" and done["step"] == 6 and done["start_step"] == 0
    tae = tmp_path / "tae"
    assert (tae / "ckpt_last.pt").is_file() and (tae / "ckpt_best.pt").is_file()
    assert not [n for n in os.listdir(tae) if n.endswith(".tmp")]
    assert list((tae / "samples").glob("step_*.jpg"))
    lines = (tae / "metrics.jsonl").read_text().splitlines()
    assert lines and all(json.loads(x)["step"] >= 0 for x in lines)
    assert uploads and uploads[-1][0] == "ckpt/night1/tae_t"
    assert {"ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl"} <= set(uploads[-1][1])
    ev_idx = json.loads((tae / "eval_indices.json").read_text())
    assert len(ev_idx) == 3
    n_lines_1 = len(lines)

    # second run resumes at step 6 and continues to 10; eval set unchanged
    args2 = args[:-4] + ["--max-steps", "10", "--poll-s", "0.01"]
    assert run_main(args2, uploader=up, now=Tick()) == 0
    out2 = capsys.readouterr().out
    init = [m for m in markers(out2) if "resumed" in m][0]
    assert init["resumed"] is True and init["step"] == 6
    done2 = markers(out2, "N1_TAE_DONE")[-1]
    assert done2["start_step"] == 6 and done2["step"] == 10
    assert json.loads((tae / "eval_indices.json").read_text()) == ev_idx
    assert T.load_ckpt(str(tae / "ckpt_last.pt"))["step"] == 10
    assert len((tae / "metrics.jsonl").read_text().splitlines()) > n_lines_1


def test_no_token_logs_hf_token_no(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = str(tmp_path)
    make_corpus(root, 8)
    a = ["--root", root, "--min-pairs", "8", "--eval-n", "2", "--batch", "4", "--crop", "32",
         "--workers", "0", "--max-steps", "2", "--epoch-min-steps", "2", "--amp", "off",
         "--device", "cpu"]
    assert T.main(a) == 0
    start = markers(capsys.readouterr().out)[0]
    assert start["hf_token"] == "no"


def test_deadline_stops_training_cleanly(tmp_path, capsys):
    root = str(tmp_path)
    make_corpus(root, 10)
    clock = Clock(step=0.1, t=1000.0)
    a = ["--root", root, "--min-pairs", "8", "--eval-n", "2", "--batch", "4", "--crop", "32",
         "--workers", "0", "--epoch-min-steps", "2", "--amp", "off", "--device", "cpu",
         "--no-upload", "--deadline", "1006", "--warmup", "2", "--first-eval-s", "1000000",
         "--eval-every-s", "1000000", "--log-every-s", "1000000"]
    assert T.main(a, {"now": clock}) == 0
    done = markers(capsys.readouterr().out, "N1_TAE_DONE")[-1]
    assert done["status"] == "deadline" and 0 < done["step"] < 200
    assert T.load_ckpt(os.path.join(root, "tae", "ckpt_last.pt"))["step"] == done["step"]


def test_deadline_while_waiting_for_data(tmp_path, capsys):
    root = str(tmp_path)
    make_corpus(root, 3)
    clock = Clock(step=0.0, t=1000.0)
    slept = []

    def sleep(s):
        slept.append(s)
        clock.t += s

    a = ["--root", root, "--min-pairs", "1000", "--deadline", "1100", "--poll-s", "30", "--no-upload"]
    assert T.main(a, {"now": clock, "sleep": sleep}) == 0
    out = capsys.readouterr().out
    assert slept and sum(slept) == pytest.approx(100.0)
    assert markers(out)[1]["waiting_for_pairs"] == 3
    assert markers(out, "N1_TAE_DONE")[-1]["stop"] == "deadline_no_data"


def test_waits_then_starts_when_pairs_appear(tmp_path, capsys):
    root = str(tmp_path)
    make_corpus(root, 4)
    state = {"n": 0}

    def sleep(_s):
        state["n"] += 1
        if state["n"] == 2:
            make_corpus(root, 6, start=4)

    a = ["--root", root, "--min-pairs", "8", "--eval-n", "2", "--batch", "4", "--crop", "32",
         "--workers", "0", "--max-steps", "2", "--epoch-min-steps", "2", "--amp", "off",
         "--device", "cpu", "--no-upload"]
    assert T.main(a, {"sleep": sleep}) == 0
    assert state["n"] == 2
    assert markers(capsys.readouterr().out, "N1_TAE_DONE")[-1]["step"] == 2


def test_stop_event_saves_checkpoint(tmp_path, capsys):
    root = str(tmp_path)
    make_corpus(root, 8)
    ev = threading.Event()
    ev.set()
    a = ["--root", root, "--min-pairs", "8", "--eval-n", "2", "--batch", "4", "--crop", "32",
         "--workers", "0", "--amp", "off", "--device", "cpu", "--no-upload"]
    # stop requested before the first wait check: exits cleanly without crashing
    assert T.main(a, {"stop_event": ev}) == 0
    assert markers(capsys.readouterr().out, "N1_TAE_DONE")


def test_dataloader_workers_path(tmp_path, capsys):
    root = str(tmp_path)
    make_corpus(root, 8)
    a = ["--root", root, "--min-pairs", "8", "--eval-n", "2", "--batch", "4", "--crop", "32",
         "--crops-per-image", "2", "--workers", "2", "--max-steps", "3", "--epoch-min-steps", "2",
         "--amp", "off", "--device", "cpu", "--no-upload"]
    assert T.main(a) == 0
    assert markers(capsys.readouterr().out, "N1_TAE_DONE")[-1]["step"] == 3


def test_upload_failure_does_not_kill_run(tmp_path, capsys):
    root = str(tmp_path)
    make_corpus(root, 8)

    def dead(folder, remote):
        raise RuntimeError("network down")

    a = ["--root", root, "--min-pairs", "8", "--eval-n", "2", "--batch", "4", "--crop", "32",
         "--workers", "0", "--max-steps", "2", "--epoch-min-steps", "2", "--amp", "off",
         "--device", "cpu"]
    assert T.main(a, {"uploader": dead, "sleep": lambda s: None}) == 0
    out = capsys.readouterr().out
    assert "upload_failed" in out
    assert markers(out, "N1_TAE_DONE")[-1]["step"] == 2


# ------------------------------------------------- memory safety (shared GPU)

def _make_trainer(tmp_path, **kw):
    a = fast_args(tmp_path, **kw)
    t = T.Trainer(a, uploader=None)
    t.device = torch.device("cpu")
    t.model = T.TAE(C)
    t.opt = torch.optim.AdamW(t.model.parameters(), lr=1e-3)
    t.acc, t.acc_n, t.crops_seen = torch.zeros(3), 0, 0
    return t


def test_micro_batch_accumulation_matches_full_batch(tmp_path):
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32) * 2 - 1
    z = torch.randn(4, C, 4, 4)
    grads = {}
    for mb in (4, 1):
        t = _make_trainer(tmp_path)
        torch.manual_seed(1)
        t.model = T.TAE(C)
        t.opt = torch.optim.AdamW(t.model.parameters(), lr=1e-3)
        parts = t._accumulate(x, z, mb)
        grads[mb] = (parts, torch.cat([p.grad.flatten() for p in t.model.parameters()]))
    assert torch.allclose(grads[4][0], grads[1][0], rtol=1e-4, atol=1e-6)
    assert torch.allclose(grads[4][1], grads[1][1], rtol=1e-3, atol=1e-5)


def test_oom_halves_micro_batch_and_step_still_runs(tmp_path, monkeypatch, capsys):
    t = _make_trainer(tmp_path, batch=8)
    assert t.micro_batch == 8
    real = T.compute_losses
    seen = []

    def fake(model, x, z, amp=False, device_type="cpu"):
        seen.append(x.shape[0])
        if x.shape[0] > 2:
            raise torch.OutOfMemoryError("CUDA out of memory (fake)")
        return real(model, x, z, amp, device_type)

    monkeypatch.setattr(T, "compute_losses", fake)
    x = torch.randint(0, 255, (8, 3, 32, 32), dtype=torch.uint8)
    z = torch.randn(8, C, 4, 4)
    before = [p.detach().clone() for p in t.model.parameters()]
    assert t._train_step((x, z)) is True
    assert t.micro_batch == 2 and t.step == 1
    assert seen[:3] == [8, 4, 2]
    assert any(not torch.equal(b, p) for b, p in zip(before, t.model.parameters()))
    assert "oom_backoff" in capsys.readouterr().out
    assert t._ckpt_obj()["micro_batch"] == 2


def test_oom_at_micro_batch_one_propagates(tmp_path, monkeypatch):
    t = _make_trainer(tmp_path, batch=2)

    def fake(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory (fake)")

    monkeypatch.setattr(T, "compute_losses", fake)
    with pytest.raises(torch.OutOfMemoryError):
        t._train_step((torch.zeros(2, 3, 32, 32, dtype=torch.uint8), torch.zeros(2, C, 4, 4)))
    assert t.micro_batch == 1 and t.step == 0


def test_non_oom_errors_are_not_swallowed(tmp_path, monkeypatch):
    t = _make_trainer(tmp_path, batch=2)

    def fake(*a, **k):
        raise RuntimeError("something else")

    monkeypatch.setattr(T, "compute_losses", fake)
    with pytest.raises(RuntimeError, match="something else"):
        t._train_step((torch.zeros(2, 3, 32, 32, dtype=torch.uint8), torch.zeros(2, C, 4, 4)))
    assert t.micro_batch == 2


def test_resume_keeps_reduced_micro_batch(tmp_path):
    t = _make_trainer(tmp_path, batch=8)
    t.micro_batch = 2
    ck = t._ckpt_obj()
    t2 = _make_trainer(tmp_path, batch=8)
    t2.model.load_state_dict(ck["model"])
    assert t2._restore(ck) and t2.micro_batch == 2
    t3 = _make_trainer(tmp_path, batch=8, micro_batch=8)       # explicit CLI value wins
    assert t3._restore(ck) and t3.micro_batch == 8


def test_mem_fraction_and_defaults():
    assert T.mem_fraction(6, 32 * 2**30) == pytest.approx(6 / 32)
    assert T.mem_fraction(100, 32 * 2**30) == 1.0
    assert T.mem_fraction(0, 32 * 2**30) == 1.0
    assert T.apply_cuda_limits(torch.device("cpu"), 6) is None
    a = T.build_parser().parse_args([])
    assert a.mem_gb == pytest.approx(6.0) and a.loader_timeout_s == 120 and a.mp_context == "forkserver"
    assert a.batch == 32 and a.micro_batch == 0


def test_loader_gets_timeout_and_clean_start_method(tmp_path):
    root = str(tmp_path)
    make_corpus(root, 8)
    t = T.Trainer(fast_args(root, workers=2), uploader=None)
    t.device, t._C = torch.device("cpu"), C
    dl = t._make_loader(list(range(8)), 0, 2)
    assert dl.timeout == 120
    assert dl.multiprocessing_context.get_start_method() == "forkserver"
    dl0 = t._make_loader(list(range(8)), 0, 0)         # timeout only exists for worker processes
    assert dl0.timeout == 0


def test_eval_turns_cudnn_benchmark_off_and_restores(tmp_path, monkeypatch):
    root = str(tmp_path)
    make_corpus(root, 4)
    t = _make_trainer(root)
    t.device = torch.device("cuda")          # pretend; evaluate is stubbed so nothing touches CUDA
    t.evalset = None
    seen = {}

    def fake_eval(model, evalset, device):
        seen["bench"] = torch.backends.cudnn.benchmark
        return None, []

    monkeypatch.setattr(T, "evaluate", fake_eval)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: seen.setdefault("emptied", True))
    torch.backends.cudnn.benchmark = True
    try:
        t._evaluate_guarded()
        assert seen["bench"] is False and seen["emptied"] is True
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.benchmark = False
