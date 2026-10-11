"""CPU tests for m13_rigid.py (no GPU, no network, no diffusers). Uses the real buc.camera.rigs."""
import json
import os
import shutil
import signal
import sys
import threading

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import m13_rigid as M  # noqa: E402

torch.set_num_threads(1)     # tiny models: OpenMP spin-waiting only hurts on a shared/loaded CPU

TINY = json.dumps({"d": 32, "conv_ch": [16, 32, 32], "layers": 1, "heads": 2, "ff": 64})
TOKEN = "hf_AbCdEfGhIjKl0123456789"
NO_NOISE = M.NoiseCfg(rot_deg=(0.0, 0.0), trans_frac=(0.0, 0.0), missing_frac=0.0)


def fast_argv(out, **kw):
    argv = ["--out", str(out), "--root", str(out), "--train-per-rig", "3", "--val-per-rig", "2",
            "--procs", "1", "--epochs", "2", "--batch", "16", "--micro-batch", "8", "--warmup-steps", "2",
            "--device", "cpu", "--amp", "off", "--no-upload", "--model-cfg", TINY, "--log-every-s", "1000000"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def markers(out, prefix="D1_M13"):
    res = []
    for line in out.splitlines():
        head, _, rest = line.partition(" ")
        if head == prefix:
            res.append(json.loads(rest))
    return res


class Clock:
    """Fake clock: every read advances time."""

    def __init__(self, t=1000.0, step=1.0):
        self.t, self.step = t, step

    def __call__(self):
        self.t += self.step
        return self.t


@pytest.fixture(scope="module")
def cache_tpl(tmp_path_factory):
    """One tiny data cache (42 train + 28 val sequences) shared by the end-to-end tests."""
    d = tmp_path_factory.mktemp("tpl")
    rc = M.main(fast_argv(d) + ["--only-gen"])
    assert rc == 0
    return d


@pytest.fixture
def workdir(tmp_path, cache_tpl):
    shutil.copytree(cache_tpl / "cache", tmp_path / "cache")
    return tmp_path


# ---------------------------------------------------------------------------
# markers, secrets
# ---------------------------------------------------------------------------

def test_marker_never_exceeds_400_chars(capsys):
    M.log({"a": "x" * 2000, "b": 1})
    M.log({f"k{i}": "y" * 50 for i in range(40)})
    M.log({"epoch": 1, "val_acc": 0.5}, marker="D1_M13_DONE")
    out = capsys.readouterr().out
    for line in out.splitlines():
        assert len(line) <= 400
    assert out.splitlines()[0].startswith("D1_M13 ")
    assert markers(out, "D1_M13_DONE")[0]["val_acc"] == 0.5


def test_token_is_scrubbed(capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "some_secret_value_123")
    M.warn("boom", err="failed with some_secret_value_123 and " + TOKEN)
    M.log({"msg": TOKEN})
    out = capsys.readouterr().out
    assert "some_secret_value_123" not in out and TOKEN not in out and "hf_***" in out
    assert M.scrub("x " + TOKEN) == "x hf_***"


def test_parse_deadline():
    assert M.parse_deadline(None) is None and M.parse_deadline("") is None
    assert M.parse_deadline("1791723600") == 1791723600.0
    assert M.parse_deadline("nope") is None and M.parse_deadline("-5") is None


# ---------------------------------------------------------------------------
# rigs, synthetic samples
# ---------------------------------------------------------------------------

def test_rig_names_match_buc():
    from buc.camera import rigs
    assert M.rig_names() == tuple(rigs.RIG_NAMES) == M.RIG_NAMES_FALLBACK and len(rigs.RIG_NAMES) == M.N_RIGS


def test_each_real_rig_gives_a_valid_sample():
    for i in range(M.N_RIGS):                                  # the 14 rigs, simulated for real (short)
        s = M.gen_sample("train", i)
        assert s is not None, M.RIG_NAMES_FALLBACK[i]
        assert s["y"] == i
        assert s["x"].shape == (M.MAX_LEN, M.FEAT_DIM) and s["x"].dtype == np.float32
        assert np.isfinite(s["x"]).all()
        assert 3 * M.FPS < s["len"] <= M.MAX_LEN                # 3-10 s at 24 fps
        assert np.all(s["x"][s["len"]:] == 0)                   # zero padding
        assert s["aux"].shape == (M.N_AUX,) and s["aux"].min() >= 0 and s["aux"].max() <= 1
        assert s["aux_mask"][-1] and s["aux_mask"][0]            # mass and duration exist for every rig
        assert abs(s["x"][1:s["len"], -1].max() - 1.0) < 0.05 or s["len"] == M.MAX_LEN


def test_sample_is_deterministic_and_splits_are_disjoint():
    a, b = M.gen_sample("train", 17), M.gen_sample("train", 17)
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["aux"], b["aux"])
    v = M.gen_sample("val", 17)
    assert v["y"] == a["y"] and not np.array_equal(a["x"], v["x"])
    c = M.gen_sample("train", 17, base_seed=5)
    assert not np.array_equal(a["x"], c["x"])
    assert M.gen_sample("train", 17 + 14)["y"] == 3 and M.gen_sample("train", 3)["y"] == 3
    r1, r2 = M.sample_rng("train", 1, 2).random(4), M.sample_rng("val", 1, 2).random(4)
    assert not np.array_equal(r1, r2)


def test_intent_and_params_ranges():
    rigs = M._rigs()
    for name in ("slider", "steadicam", "techno_crane", "shoulder_handheld"):
        rig = rigs.make_rig(name)
        for k in range(20):
            rng = M.sample_rng("train", 0, k)
            intent, total = M.make_intent(rig.q_scale, rng)
            assert 3.0 <= total <= 10.0 + 1e-9 and abs(intent.duration_s - total) < 1e-9
            assert 2 <= len(intent.times) <= 5 and np.all(np.diff(intent.times) > 0)
            assert intent.n_q == rig.n_q
            params, aux, mask = M.draw_params(name, rng)
            rigs.make_rig(name, params=params, geom=M.draw_geom(name, rng))      # accepted by buc
            assert aux.min() >= 0 and aux.max() <= 1
    p, aux, mask = M.draw_params("slider", M.sample_rng("train", 2, 0), shrink=0.0)
    assert np.allclose(aux[mask], 0.5)                          # shrink 0 -> centre of every range
    assert not mask[M.AUX_KEYS.index("delay_s")]                # slider has no delay
    _, _, mask_h = M.draw_params("shoulder_handheld", M.sample_rng("train", 7, 0))
    assert mask_h[M.AUX_KEYS.index("delay_s")] and mask_h[M.AUX_KEYS.index("ou_sigma")]


# ---------------------------------------------------------------------------
# noise, normalisation, features
# ---------------------------------------------------------------------------

def _line_poses(m=60):
    P = np.tile(np.eye(4), (m, 1, 1))
    P[:, 0, 3] = np.linspace(0, 2.0, m)
    return P


def _rand_se3(rng):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=int(rng.integers(1000))).as_matrix()
    T[:3, 3] = rng.normal(size=3) * 3
    return T


def test_normalize_poses_is_invariant_to_start_pose_and_scale():
    rig = M._rigs().make_rig("crane_jib")
    intent, _ = M.make_intent(rig.q_scale, M.sample_rng("train", 3, 1))
    P = rig.simulate(intent).frame_poses(M.FPS)
    rng = np.random.default_rng(0)
    T = _rand_se3(rng)
    Q = T @ P
    Q[:, :3, 3] *= 7.3                                          # also a world scale change (after the rigid move)
    Q[:, :3, 3] = Q[:, :3, 3]
    A, B = M.normalize_poses(P), M.normalize_poses(Q)
    # the rigid move acts on the left: relative motion is unchanged, the scale only rescales translations
    np.testing.assert_allclose(A, B, atol=1e-9)
    np.testing.assert_allclose(A[0], np.eye(4), atol=1e-12)
    assert abs(M.path_length(A[:, :3, 3]) - 1.0) < 1e-9
    fa, ma = M.featurize(P)
    fb, mb = M.featurize(Q)
    np.testing.assert_allclose(fa, fb, atol=1e-6)
    assert ma.sum() == len(P)


def test_pose_features_layout_cut_and_pad():
    f, m = M.pose_features(M.normalize_poses(_line_poses(60)))
    assert f.shape == (M.MAX_LEN, M.FEAT_DIM) and m.sum() == 60 and np.all(f[60:] == 0)
    np.testing.assert_allclose(f[0], 0)                                    # frame 0 is all zeros
    np.testing.assert_allclose(f[1:60, :6], 0, atol=1e-9)                  # straight line: no rotation
    np.testing.assert_allclose(f[1:60, 6:9], np.tile([1.0, 0, 0], (59, 1)), atol=1e-9)   # unit mean step along x
    assert abs(f[59, 9] - 1.0) < 1e-9 and np.all(np.diff(f[:60, 9]) >= -1e-12)          # cumulative path -> 1
    f2, m2 = M.pose_features(M.normalize_poses(_line_poses(300)))
    assert m2.all() and f2.shape == (M.MAX_LEN, M.FEAT_DIM)               # cut to 240
    with pytest.raises(ValueError):
        M.normalize_poses(np.eye(4)[None])


def test_pose_features_rotation_term():
    from scipy.spatial.transform import Rotation
    P = np.tile(np.eye(4), (30, 1, 1))
    P[:, :3, :3] = Rotation.from_rotvec(np.outer(np.linspace(0, 0.6, 30), [0, 1, 0])).as_matrix()
    f, _ = M.pose_features(M.normalize_poses(P))                           # pure pan, no translation
    step = 0.6 / 29
    c, s = np.cos(step), np.sin(step)
    expect = (np.array([c, 0, -s, 0, 1, 0]) - np.array([1, 0, 0, 0, 1, 0])) * M.ROT_SCALE
    np.testing.assert_allclose(f[5, :6], expect, atol=1e-6)


def test_noise_free_config_is_identity_and_noise_properties():
    P = _line_poses(80)
    out = M.add_sfm_noise(P, np.random.default_rng(0), NO_NOISE)
    np.testing.assert_allclose(out, M.normalize_poses(P), atol=1e-9)
    noisy = M.add_sfm_noise(P, np.random.default_rng(1))
    assert noisy.shape == P.shape and np.isfinite(noisy).all()
    np.testing.assert_allclose(noisy[0], np.eye(4), atol=1e-9)             # start pose = origin
    assert abs(M.path_length(noisy[:, :3, 3]) - 1.0) < 1e-9                # unknown scale -> path length 1
    R = noisy[:, :3, :3]
    np.testing.assert_allclose(np.einsum("tij,tkj->tik", R, R), np.tile(np.eye(3), (80, 1, 1)), atol=1e-9)
    assert not np.allclose(noisy, out)
    again = M.add_sfm_noise(P, np.random.default_rng(1))
    np.testing.assert_array_equal(noisy, again)                           # same rng state -> same noise


def test_noise_magnitudes_follow_the_spec_ranges():
    P = _line_poses(200)
    rot_dev = []
    for seed in range(20):
        n = M.add_sfm_noise(P, np.random.default_rng(seed), M.NoiseCfg(missing_frac=0.0))
        ang = np.arccos(np.clip((np.trace(n[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1))
        rot_dev.append(np.degrees(ang).mean())
    assert 0.05 < np.mean(rot_dev) < 3.0                                   # sub-degree to ~1 deg of noise
    lo = M.add_sfm_noise(P, np.random.default_rng(3), M.NoiseCfg(trans_frac=(0.005, 0.005), missing_frac=0.0))
    hi = M.add_sfm_noise(P, np.random.default_rng(3), M.NoiseCfg(trans_frac=(0.02, 0.02), missing_frac=0.0))
    ideal = M.normalize_poses(P)[:, :3, 3]
    assert np.abs(hi[:, 1:3, 3]).mean() > np.abs(lo[:, 1:3, 3]).mean()    # larger fraction -> larger lateral error
    assert np.abs(ideal[:, 1:]).max() == 0


def test_pure_rotation_still_gets_a_noisy_path():
    from scipy.spatial.transform import Rotation
    P = np.tile(np.eye(4), (50, 1, 1))
    P[:, :3, :3] = Rotation.from_rotvec(np.outer(np.linspace(0, 0.5, 50), [0, 1, 0])).as_matrix()
    n = M.add_sfm_noise(P, np.random.default_rng(0))
    assert abs(M.path_length(n[:, :3, 3]) - 1.0) < 1e-9                    # no divide-by-zero, no leak of exact zeros


def test_missing_frames_are_interpolated():
    P = _line_poses(40)
    miss = np.zeros(40, dtype=bool)
    miss[[5, 6, 20]] = True
    P2 = P.copy()
    P2[miss, 0, 3] = 99.0                                                  # garbage that must be replaced
    R, p = M.interpolate_missing(P2[:, :3, :3], P2[:, :3, 3], miss)
    np.testing.assert_allclose(p[:, 0], P[:, 0, 3], atol=1e-12)
    np.testing.assert_allclose(R, P[:, :3, :3], atol=1e-12)
    cfg = M.NoiseCfg(rot_deg=(0, 0), trans_frac=(0, 0), missing_frac=0.05)
    n = M.add_sfm_noise(P, np.random.default_rng(2), cfg)                  # missing + no noise = exact line
    np.testing.assert_allclose(n, M.normalize_poses(P), atol=1e-9)


# ---------------------------------------------------------------------------
# cache: units, merge, resume, parallel
# ---------------------------------------------------------------------------

def test_gen_unit_deterministic_and_balanced():
    a, b = M.gen_unit("val", 0, 28), M.gen_unit("val", 0, 28)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert a["x"].dtype == np.float16 and a["x"].shape == (28, M.MAX_LEN, M.FEAT_DIM)
    assert list(np.bincount(a["y"], minlength=14)) == [2] * 14             # same number per rig
    part = M.gen_unit("val", 10, 20)
    np.testing.assert_array_equal(part["x"], a["x"][10:20])               # independent of unit boundaries


def test_ensure_split_resumes_and_skips_finished_work(tmp_path, monkeypatch):
    calls = []
    real = M.gen_unit
    monkeypatch.setattr(M, "gen_unit", lambda *a, **k: (calls.append(a[1]), real(*a, **k))[1])
    cdir = str(tmp_path / "c")
    path = M.ensure_split(cdir, "val", 2, 0, M.NoiseCfg(), 1, lambda: False, unit=7)
    assert sorted(calls) == [0, 7, 14, 21]                                 # 28 samples in 4 units
    z = M.load_split(path)
    assert len(z["y"]) == 28 and np.array_equal(z["idx"], np.arange(28))
    assert list(np.bincount(z["y"].astype(int), minlength=14)) == [2] * 14
    first = {k: v.copy() for k, v in z.items()}
    calls.clear()
    assert M.ensure_split(cdir, "val", 2, 0, M.NoiseCfg(), 1, lambda: False, unit=7) == path
    assert calls == []                                                     # complete merged file: nothing redone
    os.remove(path)
    os.remove(os.path.join(cdir, "val", "unit_00001.npz"))
    M.ensure_split(cdir, "val", 2, 0, M.NoiseCfg(), 1, lambda: False, unit=7)
    assert calls == [7]                                                    # only the missing unit is rebuilt
    z2 = M.load_split(path)
    for k in first:
        np.testing.assert_array_equal(first[k], z2[k])


def test_ensure_split_stop_then_continue(tmp_path):
    cdir = str(tmp_path / "c")
    n = {"k": 0}

    def stop():
        n["k"] += 1
        return n["k"] > 2

    assert M.ensure_split(cdir, "train", 2, 0, M.NoiseCfg(), 1, stop, unit=7) is None
    done = [f for f in os.listdir(os.path.join(cdir, "train")) if f.endswith(".npz")]
    assert 0 < len(done) < 4 and not os.path.exists(os.path.join(cdir, "train.npz"))
    path = M.ensure_split(cdir, "train", 2, 0, M.NoiseCfg(), 1, lambda: False, unit=7)
    assert len(M.load_split(path)["y"]) == 28


def test_parallel_pool_matches_serial(tmp_path):
    a = M.ensure_split(str(tmp_path / "a"), "val", 2, 0, M.NoiseCfg(), 1, lambda: False, unit=7)
    b = M.ensure_split(str(tmp_path / "b"), "val", 2, 0, M.NoiseCfg(), 2, lambda: False, unit=7)
    za, zb = M.load_split(a), M.load_split(b)
    for k in za:
        np.testing.assert_array_equal(za[k], zb[k])


def test_pool_terminates_even_with_flag_only_signal_handler(tmp_path):
    """main() installs a flag-only SIGTERM handler before the fork; workers must not inherit it (pool.join hung)."""
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    seen = []
    for s in old:
        signal.signal(s, lambda n, _f: seen.append(n))
    try:
        n = {"k": 0}

        def stop():
            n["k"] += 1
            return True

        res = {}
        t = threading.Thread(target=lambda: res.update(r=M.ensure_split(
            str(tmp_path / "c"), "train", 30, 0, M.NoiseCfg(), 2, stop, unit=7)), daemon=True)
        t.start()
        t.join(60)
        assert not t.is_alive(), "ensure_split hung: pool workers ignored terminate()"
        assert res["r"] is None
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def test_worker_init_restores_default_signals():
    import multiprocessing
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(1, initializer=M._worker_init) as p:
        assert p.apply(signal.getsignal, (signal.SIGTERM,)) == signal.SIG_DFL
        assert p.apply(signal.getsignal, (signal.SIGINT,)) == signal.SIG_IGN


def test_cgroup_cpu_limit_v2_v1_and_unlimited(tmp_path):
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cpu.max").write_text("1400000 100000\n")
    assert M.cgroup_cpu_limit(str(v2)) == pytest.approx(14.0)
    (v2 / "cpu.max").write_text("max 100000\n")
    assert M.cgroup_cpu_limit(str(v2)) is None
    v1 = tmp_path / "v1" / "cpu"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("1200000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    assert M.cgroup_cpu_limit(str(tmp_path / "v1")) == pytest.approx(12.0)
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    assert M.cgroup_cpu_limit(str(tmp_path / "v1")) is None
    assert M.cgroup_cpu_limit(str(tmp_path / "nothing")) is None


def test_default_procs_respects_quota_affinity_and_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(M.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(M.os, "sched_getaffinity", lambda _p: set(range(128)), raising=False)
    (tmp_path / "cpu.max").write_text("800000 100000\n")
    assert M.default_procs(str(tmp_path)) == 6                 # 8-core quota on a 128-core host -> 8 - 2
    (tmp_path / "cpu.max").write_text("1600000 100000\n")
    assert M.default_procs(str(tmp_path)) == 12                # capped
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert M.default_procs(str(tmp_path)) == 12
    monkeypatch.setattr(M.os, "sched_getaffinity", lambda _p: set(range(4)), raising=False)
    assert M.default_procs(str(tmp_path)) == 2                 # affinity mask wins
    monkeypatch.setattr(M.os, "cpu_count", lambda: 2)
    assert M.default_procs(str(tmp_path)) == 1                 # never below one


def test_cache_key_changes_with_config():
    k = M.cache_key(0, M.NoiseCfg())
    assert k == M.cache_key(0, M.NoiseCfg()) and k != M.cache_key(1, M.NoiseCfg()) \
        and k != M.cache_key(0, M.NoiseCfg(missing_frac=0.1))


def test_failed_samples_are_dropped_not_fatal(monkeypatch):
    real = M.gen_sample
    monkeypatch.setattr(M, "gen_sample", lambda split, i, *a, **k: None if i == 3 else real(split, i, *a, **k))
    u = M.gen_unit("val", 0, 8)
    assert len(u["y"]) == 7 and 3 not in u["idx"] and int(u["n_failed"]) == 1


def test_simulation_failure_is_redrawn(monkeypatch):
    rigs = M._rigs()
    real = rigs.Rig.simulate
    state = {"n": 0}

    def flaky(self, *a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError("boom")
        return real(self, *a, **k)

    monkeypatch.setattr(rigs.Rig, "simulate", flaky)
    assert M.gen_sample("val", 2) is not None and state["n"] >= 2


# ---------------------------------------------------------------------------
# model, loss, schedule, confusion
# ---------------------------------------------------------------------------

def test_param_count_in_range():
    n = M.count_params(M.RigIDNet())
    assert M.PARAM_MIN <= n <= M.PARAM_MAX
    assert 2_000_000 <= n <= 4_000_000


def test_forward_shapes_and_padding_invariance():
    torch.manual_seed(0)
    m = M.RigIDNet(**json.loads(TINY)).eval()
    x = torch.randn(3, M.MAX_LEN, M.FEAT_DIM)
    lens = torch.tensor([240, 100, 37])
    mask = M.lens_to_mask(lens)
    x = x * mask[..., None]
    lo, ax = m(x, mask)
    assert lo.shape == (3, 14) and ax.shape == (3, M.N_AUX) and ((ax >= 0) & (ax <= 1)).all()
    x2 = x.clone()
    x2[1, 100:] = 5.0                                                      # garbage in the padding
    x2[2, 37:] = -3.0
    lo2, ax2 = m(x2, mask)
    torch.testing.assert_close(lo, lo2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ax, ax2, atol=1e-5, rtol=1e-5)


def test_loss_masks_aux_and_is_finite():
    torch.manual_seed(0)
    m = M.RigIDNet(**json.loads(TINY))
    x = torch.randn(4, M.MAX_LEN, M.FEAT_DIM)
    mask = M.lens_to_mask(torch.tensor([240, 120, 80, 240]))
    y = torch.tensor([0, 5, 9, 13])
    aux = torch.rand(4, M.N_AUX)
    out0 = M.compute_loss(m, x, mask, y, aux, torch.zeros(4, M.N_AUX))
    assert torch.isfinite(out0["loss"]) and float(out0["aux"].detach()) == 0.0
    assert torch.allclose(out0["loss"], out0["ce"])
    out1 = M.compute_loss(m, x, mask, y, aux, torch.ones(4, M.N_AUX))
    assert float(out1["aux"].detach()) > 0 and float(out1["loss"].detach()) > float(out0["loss"].detach())
    out1["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_bf16_autocast_path_runs():
    torch.manual_seed(0)
    m = M.RigIDNet(**json.loads(TINY))
    d = M.gen_unit("val", 0, 14)
    x = torch.from_numpy(d["x"]).float()
    mask = M.lens_to_mask(torch.from_numpy(d["len"].astype(np.int64)))
    out = M.compute_loss(m, x, mask, torch.from_numpy(d["y"].astype(np.int64)), torch.from_numpy(d["aux"]),
                         torch.from_numpy(d["aux_mask"]), amp=True, device_type="cpu")
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and out["logits"].dtype == torch.float32


def test_lr_schedule():
    assert M.lr_at(0, 3e-4, 100, 1000) < M.lr_at(50, 3e-4, 100, 1000) < M.lr_at(99, 3e-4, 100, 1000) <= 3e-4
    assert abs(M.lr_at(100, 3e-4, 100, 1000) - 3e-4) < 1e-9
    assert M.lr_at(1000, 3e-4, 100, 1000) == pytest.approx(3e-4 * 0.02)
    assert M.lr_at(5000, 3e-4, 100, 1000) == pytest.approx(3e-4 * 0.02)


def test_confusion_matrix_and_png(tmp_path):
    cm = M.confusion_matrix(np.array([0, 1, 1, 13]), np.array([0, 1, 2, 13]))
    assert cm.shape == (14, 14) and cm[0, 0] == 1 and cm[2, 1] == 1 and cm.sum() == 4
    img = M.draw_confusion(cm, M.RIG_NAMES_FALLBACK, "t")
    assert img.width > 14 * 40 and img.height > 14 * 40
    assert len(img.getcolors(10**6)) > 3                                   # not blank
    img.save(tmp_path / "c.png")


def test_evaluate_counts():
    torch.manual_seed(0)
    model = M.RigIDNet(**json.loads(TINY))
    d = M.gen_unit("val", 0, 14)
    r = M.evaluate(model, d, torch.device("cpu"), bs=5)
    assert r["n"] == 14 and r["confusion"].sum() == 14 and 0 <= r["val_acc"] <= 1 and r["top2"] >= r["val_acc"]
    assert len(r["per_rig_acc"]) == 14 and np.isfinite(r["val_loss"]) and np.isfinite(r["aux_mae"])
    assert model.training                                                  # train/eval mode restored


def test_save_atomic_and_load(tmp_path):
    p = str(tmp_path / "ck.pt")
    M.save_atomic({"model": {"w": torch.ones(2)}, "step": 3}, p)
    assert not os.path.exists(p + ".tmp") and M.load_ckpt(p)["step"] == 3
    open(p, "wb").write(b"junk")
    assert M.load_ckpt(p) is None
    assert M.load_ckpt(str(tmp_path / "missing.pt")) is None
    open(str(tmp_path / "a.tmp"), "w").write("x")
    assert M.clean_tmp(str(tmp_path)) == 1


# ---------------------------------------------------------------------------
# end-to-end: training, checkpoints, resume, deadline, upload
# ---------------------------------------------------------------------------

def test_train_one_step_then_checkpoint_and_resume(workdir, capsys):
    rc = M.main(fast_argv(workdir))
    out = capsys.readouterr().out
    assert rc == 0
    for line in out.splitlines():
        assert len(line) <= 400
    done = markers(out, "D1_M13_DONE")[-1]
    assert done["status"] == "epochs_done" and done["epochs_done"] == 2 and done["step"] == 4   # 42 // 16 = 2 steps
    lines = [json.loads(x) for x in open(workdir / "metrics.jsonl")]
    assert [r["epoch"] for r in lines] == [1, 2]
    for r in lines:
        assert {"epoch", "loss", "val_acc", "top2"} <= set(r) and np.isfinite(r["loss"])
        assert 0 <= r["val_acc"] <= r["top2"] <= 1 and len(r["per_rig_acc"]) == 14
    assert (workdir / "ckpt_last.pt").exists() and (workdir / "ckpt_best.pt").exists()
    assert not any(n.endswith(".tmp") for n in os.listdir(workdir))
    cj = json.load(open(workdir / "confusion.json"))
    assert len(cj["confusion"]) == 14 and sum(map(sum, cj["confusion"])) == 28 and cj["names"] == list(M.RIG_NAMES_FALLBACK)
    from PIL import Image
    assert Image.open(workdir / "confusion.png").size[0] > 500
    ep = [m for m in markers(out) if "val_acc" in m]
    assert len(ep) == 2 and ep[0]["epoch"] == 1
    first = markers(out)[0]
    assert first["hf_token"] == "no"

    ck = M.load_ckpt(str(workdir / "ckpt_last.pt"))
    assert ck["epoch"] == 2 and ck["step"] == 4 and ck["config"]["d"] == 32

    # resume: more epochs continue from ckpt_last, finished work is not redone
    rc = M.main(fast_argv(workdir, epochs=3))
    out2 = capsys.readouterr().out
    assert rc == 0
    st = [m for m in markers(out2) if "resumed" in m][0]
    assert st["resumed"] is True and st["epoch"] == 2 and st["step"] == 4
    d2 = markers(out2, "D1_M13_DONE")[-1]
    assert d2["epochs_done"] == 3 and d2["step"] == 6 and d2["start_step"] == 4
    assert len(open(workdir / "metrics.jsonl").readlines()) == 3
    # already complete: no extra training, normal exit
    rc = M.main(fast_argv(workdir, epochs=3))
    d3 = markers(capsys.readouterr().out, "D1_M13_DONE")[-1]
    assert rc == 0 and d3["status"] == "complete" and d3["step"] == 6


def test_deadline_stops_training_cleanly(workdir, capsys):
    clk = Clock(t=1000.0, step=1.0)
    deadline = 1000 + 600 + 12                                             # stops a few reads in (margin 600 s)
    rc = M.main(fast_argv(workdir, epochs=50, deadline=deadline), deps={"now": clk})
    out = capsys.readouterr().out
    done = markers(out, "D1_M13_DONE")[-1]
    assert rc == 0 and done["status"] == "deadline" and done["step"] < 50 * 2
    assert (workdir / "ckpt_last.pt").exists()                             # last save happened before exit
    ck = M.load_ckpt(str(workdir / "ckpt_last.pt"))
    assert ck["step"] == done["step"]
    last = json.loads(open(workdir / "metrics.jsonl").readlines()[-1])
    assert np.isfinite(last["val_acc"])
    assert not any(n.endswith(".tmp") for n in os.listdir(workdir))
    # a rerun without the deadline continues where it stopped
    rc = M.main(fast_argv(workdir, epochs=ck["epoch"] + 1))
    assert rc == 0 and markers(capsys.readouterr().out, "D1_M13_DONE")[-1]["epochs_done"] == ck["epoch"] + 1


def test_deadline_during_data_generation_exits_normally(tmp_path, capsys):
    clk = Clock(t=1000.0, step=0.001)
    rc = M.main(fast_argv(tmp_path, deadline=1000 + 600 - 1), deps={"now": clk})   # already inside the margin
    done = markers(capsys.readouterr().out, "D1_M13_DONE")[-1]
    assert rc == 0 and done["status"] == "deadline_in_datagen"
    assert not (tmp_path / "ckpt_last.pt").exists()
    rc = M.main(fast_argv(tmp_path, epochs=1, train_per_rig=1, val_per_rig=1))      # a rerun builds and trains
    assert rc == 0 and markers(capsys.readouterr().out, "D1_M13_DONE")[-1]["epochs_done"] == 1


def test_signal_stop_event(workdir, capsys):
    ev = threading.Event()
    ev.set()
    rc = M.main(fast_argv(workdir, epochs=5), deps={"stop_event": ev})
    done = markers(capsys.readouterr().out, "D1_M13_DONE")[-1]
    assert rc == 0 and done["status"] in ("signal", "signal_in_datagen")


def test_uploads_and_token_never_printed(workdir, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    seen = []

    def fake_uploader(folder, remote):
        seen.append((remote, sorted(os.listdir(folder))))

    argv = [a for a in fast_argv(workdir) if a != "--no-upload"] + ["--upload-every-s", "0"]
    rc = M.main(argv, deps={"uploader": fake_uploader, "sleep": lambda s: None})
    out = capsys.readouterr().out
    assert rc == 0 and TOKEN not in out
    assert seen and all(r == "ckpt/day1/m13" for r, _ in seen)
    final = seen[-1][1]
    assert {"ckpt_best.pt", "ckpt_last.pt", "metrics.jsonl", "confusion.png"} <= set(final)

    # an uploader that fails (and leaks the token in its message) only produces a scrubbed warning
    def bad_uploader(folder, remote):
        raise RuntimeError(f"401 Unauthorized for {TOKEN}")

    shutil.rmtree(workdir / "upload_stage", ignore_errors=True)
    rc = M.main(argv + ["--epochs", "3"], deps={"uploader": bad_uploader, "sleep": lambda s: None})
    out = capsys.readouterr().out
    assert rc == 0 and TOKEN not in out and "upload_failed" in out
    assert markers(out, "D1_M13_DONE")[-1]["status"] == "epochs_done"


def test_token_from_env_selects_uploader_without_printing_it(workdir, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    got = {}

    def fake_make(token, repo_id=None):
        got["t"] = token
        return lambda folder, remote: None

    real_make = M.make_uploader
    monkeypatch.setattr(M, "make_uploader", fake_make)
    argv = [a for a in fast_argv(workdir, epochs=1) if a != "--no-upload"]
    assert M.main(argv) == 0
    out = capsys.readouterr().out
    assert got["t"] == TOKEN and TOKEN not in out and markers(out)[0]["hf_token"] == "yes"
    monkeypatch.delenv("HF_TOKEN")
    assert real_make(None) is None and real_make("") is None


def test_bad_step_is_skipped_and_run_goes_on(workdir, capsys, monkeypatch):
    real = M.compute_loss
    n = {"k": 0}

    def flaky(*a, **k):
        n["k"] += 1
        if n["k"] == 1:
            raise ValueError("synthetic failure")
        return real(*a, **k)

    monkeypatch.setattr(M, "compute_loss", flaky)
    rc = M.main(fast_argv(workdir))
    out = capsys.readouterr().out
    done = markers(out, "D1_M13_DONE")[-1]
    assert rc == 0 and "step_failed" in out and done["status"] == "epochs_done" and done["step"] == 3


def test_hundred_consecutive_failures_abort(workdir, capsys, monkeypatch):
    def always(*a, **k):
        raise ValueError("always broken")

    monkeypatch.setattr(M, "compute_loss", always)
    rc = M.main(fast_argv(workdir, epochs=100, batch=2))
    out = capsys.readouterr().out
    done = markers(out, "D1_M13_DONE")[-1]
    assert rc == 2 and done["fatal"] == "too_many_step_errors"
    assert all(len(line) <= 400 for line in out.splitlines())


def test_oom_backoff_keeps_the_effective_batch(workdir, monkeypatch):
    args = M.build_parser().parse_args(fast_argv(workdir, batch=16, micro_batch=16))
    tr = M.Trainer(args, uploader=None)
    tr.device = torch.device("cpu")
    tr.model = M.RigIDNet(**json.loads(TINY))
    tr.opt = torch.optim.AdamW(tr.model.parameters(), lr=1e-3)
    tr.acc, tr.acc_n = torch.zeros(3), 0
    real = M.compute_loss

    def oom_when_big(model, x, *a, **k):
        if x.shape[0] > 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(model, x, *a, **k)

    monkeypatch.setattr(M, "compute_loss", oom_when_big)
    data = M.gen_unit("val", 0, 16)
    batch = tr._batch_tensors(data, np.arange(16))
    assert tr._train_step(batch, 10) is True
    assert tr.micro_batch == 4 and tr.step == 1 and tr.acc_n == 1 and torch.isfinite(tr.acc).all()
    assert M.is_oom(RuntimeError("CUDA out of memory")) and not M.is_oom(ValueError("out of memory"))


def test_progress_marker_without_epoch_boundaries(workdir, capsys):
    # a tiny log period makes the in-epoch heartbeat fire (>= 1 marker per 5 minutes on the real box)
    clk = Clock(t=1000.0, step=100.0)
    rc = M.main(fast_argv(workdir, epochs=1, log_every_s=150), deps={"now": clk})
    out = capsys.readouterr().out
    assert rc == 0 and any(m.get("progress") == 1 for m in markers(out))
    assert M.build_parser().parse_args([]).log_every_s <= 300
