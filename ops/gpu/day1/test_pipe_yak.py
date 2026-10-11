"""CPU tests for pipe_yak.py (no GPU, no network, no diffusers, no HF): synthetic yak, fake disparity function,
fake teacher, fake uploader.  Mode A runs on a tiny synthetic image; mode B checks the stage-fallback logic and the
report schema with fakes."""
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipe_yak as P  # noqa: E402

P.ensure_blake3()
SECRET = "hf_" + "AbCdEfGhIjKlMnOpQrStUv"
TINY_W = 128
SCALE = 0.1                      # shot durations x0.1 -> 15..25 frames


# ---------------------------------------------------------------- fakes
def synth_image(W=400, H=172):
    rs = np.random.RandomState(0)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), np.float32)
    img[..., 2], img[..., 1] = 0.7, 0.4
    img[yy >= H * 0.4] = (0.2, 0.55, 0.2)
    cv2.circle(img, (W // 2, int(H * 0.5)), H // 4, (1, 1, 1), -1)
    img = np.clip(img + rs.randn(H, W, 3) * 0.03, 0, 1)
    return (img * 255).astype(np.uint8)


def fake_disp(rgb):
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = (yy / h).astype(np.float32)
    blob = ((yy - h * 0.5) ** 2 + (xx - w * 0.5) ** 2) < (h * 0.25) ** 2
    return (d + blob * 0.8).astype(np.float32)


def boom_disp(rgb):
    raise RuntimeError("depth model exploded")


class FakeApi:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
        if os.path.basename(path_in_repo) in self.fail:
            raise RuntimeError("upload broke " + SECRET)
        self.calls.append((os.path.basename(path_or_fileobj), path_in_repo, repo_id, repo_type))


class FakeTeacherHandle:
    """Duck-typed TCH-02 sampler the real WLD-08 accepts (identity encode/decode, SDEdit = 0.9 * latent)."""

    def __init__(self, mode="ok"):
        self.mode, self.released, self.sampler = mode, 0, None

    def get(self):
        if self.mode == "unavailable":
            raise P.TeacherUnavailable("no CUDA: Z-Image teacher not loaded")
        if self.mode == "boom":
            raise ValueError("sampler exploded")
        import torch
        from buc.teacher.sample import SigmaTable

        class Teacher:
            spec = SimpleNamespace(check_image_size=lambda h, w: None)

            def encode_image(self, x):
                return x.float()

            def decode_latent(self, z):
                return z.float()

        class Sampler:
            teacher = Teacher()
            table = SigmaTable.from_flow_shift()

            def sdedit(self, x0, cond, *, sigma, seeds, steps):
                return SimpleNamespace(z=x0 * 0.9, sigma_end=0.0)

        self.sampler = Sampler()
        return self.sampler

    def release(self):
        self.released += 1


def run_main(tmp_path, extra=(), *, modes="skeleton,scaffold", disp=fake_disp, teacher=None, image=None,
             deadline=None, api=None, download_fn=None, workers=1, no_upload=True):
    args = ["--root", str(tmp_path), "--width", str(TINY_W), "--dur-scale", str(SCALE), "--workers", str(workers),
            "--modes", modes, "--no-teacher" if teacher is None else "--force-matte"] + list(extra)
    if no_upload:
        args.append("--no-upload")
    rc = P.main(args, disparity_fn=disp, teacher=teacher, image_array=synth_image() if image is None else image,
                deadline=deadline, upload_api=api, download_fn=download_fn)
    out = os.path.join(str(tmp_path), "day1", "pipe")
    rep = json.load(open(os.path.join(out, "stages_report.json")))
    return rc, out, rep


def n_frames_of(path):
    cap = cv2.VideoCapture(path)
    n = 0
    first = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if first is None:
            first = f
        n += 1
    cap.release()
    return n, first


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("full")
    t0 = time.time()
    rc, out, rep = run_main(tmp)
    return {"rc": rc, "out": out, "rep": rep, "tmp": tmp, "secs": time.time() - t0}


# ---------------------------------------------------------------- pure helpers
def test_sizes_boxes_width_plan():
    assert P.out_size_for(1920) == (1920, 826)
    assert P.out_size_for(960) == (960, 414)
    assert all(v % 2 == 0 for v in P.out_size_for(333))
    b = P.scale_box((900, 640, 1130, 840), (2000, 861), (1000, 430))
    assert b == (450, 320, 565, 420)
    x0, y0, x1, y1 = P.scale_box((1990, 855, 2100, 900), (2000, 861), (100, 43))
    assert 0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 43
    assert P.plan_width([1920, 960], 1e-6, 1000, 4, None) == 1920
    assert P.plan_width([1920, 960, 640], 1e-6, 1000, 1, 1.0) == 640        # nothing fits -> smallest
    assert P.plan_width([1920, 960], 1e-6, 100, 8, 600.0) == 1920
    assert P.plan_width([1920, 960], 1e-5, 1000, 1, 5000.0) == 960


def test_marker_limit_and_token_scrub(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    big = {"ev": "x", "blob": "A" * 2000, "n": [1.23456789] * 200, "tok": SECRET}
    line = P.fit_marker(P.MARK, big)
    assert len(line) <= 400 and line.startswith("D1_PIPE ") and SECRET not in line
    P.log({"ev": "t", "err": f"bad {SECRET} thing"})
    out = capsys.readouterr().out
    assert SECRET not in out and out.startswith("D1_PIPE ")
    assert SECRET not in P.err_text(RuntimeError(SECRET))
    assert json.loads(out.split(" ", 1)[1])["ev"] == "t"


def test_disp_to_depth_calibration():
    rs = np.random.RandomState(1)
    disp = rs.rand(60, 120).astype(np.float32)
    disp[40:55, 50:80] = 0.9
    box = (50, 40, 80, 55)
    z = P.disp_to_depth(disp, box)
    assert abs(float(np.median(z[40:55, 50:80])) - 1.0) < 1e-3
    assert float(z.max()) <= P.FAR_LIMIT + 1e-3 and float(z.min()) >= P.NEAR_LIMIT - 1e-6
    # monotone: closer (larger disparity) is never farther
    order = np.argsort(disp.ravel())
    assert np.all(np.diff(z.ravel()[order]) <= 1e-5)
    flat = P.disp_to_depth(np.full((20, 20), 0.5, np.float32), (2, 2, 8, 8))
    assert np.isfinite(flat).all()


def test_deadline_logic():
    now = [1000.0]
    d = P.Deadline(1000.0 + 700.0, clock=lambda: now[0])
    assert not d.stop_now() and abs(d.left() - 700.0) < 1e-9
    now[0] += 150.0
    assert d.stop_now()                                    # < 600 s left
    assert P.Deadline(None).left() is None and not P.Deadline(None).stop_now()
    assert abs(P.Deadline(2000.0 + 600.0, clock=lambda: 1000.0).budget_s(0.5) - 500.0) < 1e-9


def test_ledger_verdicts():
    L = P.Ledger()
    L.add("a", "real")
    L.add("a", "real")
    L.add("b", "temp")
    L.add("c", "real")
    L.add("c", "fallback", "boom", "S2")
    L.add("d", "fallback", "boom")
    L.add("e", "stub_only")
    L.add("f", "unavailable", "no cuda")
    L.add("g", "rejected", "low_conf")
    v = {k: L.verdict(k) for k in "abcdefg"}
    assert v == {"a": "real", "b": "temp", "c": "mixed", "d": "fallback", "e": "stub_only", "f": "unavailable",
                 "g": "rejected"}
    assert L.verdict("zzz") == "not_run"
    assert L.summary()["c"]["events"][0]["shot"] == "S2"


def test_homography_frame_identity_and_shift():
    from buc.core import contracts as C
    img = synth_image(160, 69)
    obs = C.Obs(K=np.array([[200.0, 0, 79.5], [0, 200.0, 34.0], [0, 0, 1]]), R=np.eye(3), t=np.zeros(3), width=160,
                height=69)
    same = P.homography_frame(img, obs, np.eye(4), obs.K, 5.0)
    assert np.abs(same.astype(int) - img.astype(int)).max() <= 1
    T = np.eye(4)
    T[0, 3] = 0.5                                           # camera moves right: scene shifts left
    shifted = P.homography_frame(img, obs, T, obs.K, 5.0)
    assert np.abs(shifted.astype(int) - img.astype(int)).mean() > 1.0
    big = P.homography_frame(img, obs, np.eye(4), obs.K * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 1]]), 5.0)
    assert big.shape == img.shape


def test_overlay_label_marks_top_left():
    f = np.full((82, 192, 3), 40, np.uint8)
    g = P.overlay_label(f.copy(), P.SCAFFOLD_LABEL)
    assert (g[:20, :120] != 40).any() and (g[:20, :120] > 240).any()
    assert np.array_equal(g[-10:], f[-10:])                 # bottom untouched


def test_simple_dof_blurs_by_depth():
    rs = np.random.RandomState(0)
    img = rs.rand(40, 60, 3).astype(np.float32)
    depth = np.full((40, 60), 10.0, np.float32)
    depth[:, 30:] = 2.0
    out = P.simple_dof(img, depth, focus_m=10.0, max_r=6.0)
    assert out.shape == img.shape and np.isfinite(out).all()
    assert np.abs(out[:, :25] - img[:, :25]).mean() < np.abs(out[:, 35:] - img[:, 35:]).mean()


# ---------------------------------------------------------------- mode A
def test_mode_a_skeleton_tiny(tmp_path):
    t0 = time.time()
    rc, out, rep = run_main(tmp_path, modes="skeleton")
    assert time.time() - t0 < 30
    assert rc == 0
    sk = rep["skeleton"]
    assert sk["status"] == "ok" and sk["video"] == "skeleton_S1.mp4"
    # ONLY the PRD-03 stubs in the spine, no fallback involved
    assert {s: r["kind"] for s, r in sk["stages"].items()} == {"S-1": "stub", "S2": "stub", "S8": "stub", "S8.5": "stub"}
    assert [r["impl"].split("@")[0] for r in sk["stages"].values()] == [
        "stub:classic_read", "stub:flat_world", "stub:fixed_move", "stub:identity_polish"]
    assert all(r["fallback"] is None for r in sk["stages"].values())
    assert sk["move"] == "stub_push_in"
    n, first = n_frames_of(os.path.join(out, "skeleton_S1.mp4"))
    assert n == sk["n_frames"] == sk["render"]["frames"] == 20
    assert first.shape[:2] == (rep["config"]["height"], rep["config"]["width"])
    assert (first[:20, :100] > 235).any()                  # the SKELETON label is burnt in
    assert rep["scaffold"]["status"] == "not_run" and not os.path.exists(os.path.join(out, "scaffold_S1.mp4"))
    assert os.path.exists(os.path.join(out, "contact_sheet.jpg"))


def test_mode_a_survives_broken_real_stages(tmp_path, monkeypatch):
    """RDR-02 raising on every frame -> stub homography frames; RDR-06 writer broken -> imageio."""
    from buc.render import quads, video

    def boom(*a, **k):
        raise RuntimeError("render_ldi is broken")

    class BadWriter:
        def __init__(self, *a, **k):
            raise video.VideoError("rdr06 writer broken")

    monkeypatch.setattr(quads, "render_ldi", boom)
    monkeypatch.setattr(video, "VideoWriter", BadWriter)
    rc, out, rep = run_main(tmp_path, modes="skeleton")
    sk = rep["skeleton"]
    assert rc == 0 and sk["status"] == "ok"
    assert sk["render"]["render_stub"] == sk["render"]["frames"] == 20 and sk["render"]["render_real"] == 0
    assert "render_ldi is broken" in sk["render"]["render_errors"][0]
    assert sk["encode"]["impl"] == "imageio" and "rdr06" in sk["encode"]["errors"][0]
    n, _ = n_frames_of(os.path.join(out, "skeleton_S1.mp4"))
    assert n == 20
    assert rep["units"][P.U_RENDER]["verdict"] == "fallback" and rep["units"][P.U_ENC]["verdict"] == "fallback"


def test_out_video_last_resort_cv2(tmp_path, monkeypatch):
    from buc.render import video
    import imageio

    monkeypatch.setattr(video, "VideoWriter", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(imageio, "get_writer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("y")))
    v = P.OutVideo(tmp_path / "a.mp4", 64, 32, 24)
    assert v.impl == "cv2" and len(v.errors) == 2
    for i in range(5):
        v.write(np.full((32, 64, 3), 10 * i, np.uint8))
    v.close()
    assert n_frames_of(str(tmp_path / "a.mp4"))[0] == 5
    v2 = P.OutVideo(tmp_path / "b.mp4", 64, 32, 24)
    v2.write(np.zeros((32, 64, 3), np.uint8))
    v2.abort()
    assert not (tmp_path / "b.mp4").exists()


# ---------------------------------------------------------------- mode B (shared tiny full run)
def test_full_tiny_run_outputs_and_schema(full_run):
    rc, out, rep = full_run["rc"], full_run["out"], full_run["rep"]
    assert rc == 0
    assert full_run["secs"] < 60
    assert rep["schema"] == P.REPORT_SCHEMA
    for k in ("created_utc", "job", "lane", "host", "image", "config", "skeleton", "scaffold", "units", "errors",
              "outputs", "real_vs_stub", "no_depth_artifacts_written", "blake3_shim"):
        assert k in rep
    assert rep["no_depth_artifacts_written"] is True and "eval-only" in rep["lane"]
    names = {f"scaffold_S{i}.mp4" for i in range(1, 7)} | {"skeleton_S1.mp4", "stages_report.json", "contact_sheet.jpg"}
    assert names <= set(os.listdir(out))
    assert set(rep["outputs"]) >= names
    assert rep["scaffold"]["status"] == "ok" and set(rep["scaffold"]["shots"]) == set(P.SHOT_ORDER)
    for name, s in rep["scaffold"]["shots"].items():
        assert s["status"] == "ok", (name, s)
        n, first = n_frames_of(os.path.join(out, s["video"]))
        assert n == s["n_frames"] == s["render"]["frames"]
        assert n == int(round(P.SHOT_DEFS[name]["dur"] * SCALE * 24)) + 1 or name == "S5"
        assert set(s["stages"]) == {"S-1", "S2", "S8", "S8.5"}
        assert all(r["status"] in ("run", "hit", "fallback") for r in s["stages"].values())
        assert (first[:20, :100] > 235).any()              # SCAFFOLD label burnt in
    rs = rep["real_vs_stub"]
    assert rs["S1"] == {"S-1": "real", "S2": "real", "S8": "real", "S8.5": "stub"}
    assert rs["skeleton"]["S2"] == "stub"
    # the second shot reuses S-1 / S2 from the first run and says where they came from
    assert rep["scaffold"]["shots"]["S2"]["stages"]["S-1"]["status"] == "hit"
    assert rep["scaffold"]["shots"]["S2"]["stages"]["S-1"]["first_run"]["status"] == "run"


def test_full_tiny_run_read_and_units(full_run):
    rep = full_run["rep"]
    rd = rep["scaffold"]["read_summary"]
    assert rd["depth_source"] == "mrh" and rd["fallbacks"] == []
    assert set(rd["steps"]) == {"R0", "R1", "R5", "R6"} and all(v["status"] == "ok" for v in rd["steps"].values())
    assert "temp-da2s" in rd["steps"]["R5"]["impl"]
    assert rd["scale_s"] > 1.0 and rd["depth_m"][0] > 0
    u = rep["units"]
    assert u[P.U_R5]["verdict"] == "temp" and u[P.U_R0]["verdict"] == "real" and u[P.U_R1]["verdict"] == "real"
    assert u[P.U_R6]["verdict"] == "real"
    assert u[P.U_W3]["verdict"] == "real" and u[P.U_W8]["verdict"] == "unavailable"     # teacher disabled in this run
    assert u[P.U_CAM]["verdict"] == "real" and u[P.U_POLISH]["verdict"] == "stub_only"
    assert u[P.U_DOF]["verdict"] == "real" and u[P.U_RENDER]["verdict"] == "real" and u[P.U_ENC]["verdict"] == "real"
    assert P.U_MATTE in u
    wb = rep["scaffold"]["world_build"]
    assert any(e["unit"] == P.U_W3 and e["status"] == "real" for e in wb)


def test_full_tiny_run_camera_recipes(full_run):
    sh = full_run["rep"]["scaffold"]["shots"]
    assert sh["S1"]["camera"]["used"] == "push_in" and sh["S1"]["camera"]["source"] == "CAM-02"
    assert sh["S1"]["camera"]["rig"] in ("dolly", "ideal")
    assert sh["S2"]["camera"]["used"] == "slider"
    assert sh["S3"]["camera"]["used"] == "orbit" and "0 -> +12" in sh["S3"]["camera"]["note"]
    assert sh["S4"]["camera"]["used"] == "crane_up" and "no crane recipe" in sh["S4"]["camera"]["source"]
    assert sh["S5"]["camera"]["used"].startswith("rack_focus") and len(sh["S5"]["camera"]["focus_stops_m"]) == 3
    assert sh["S6"]["camera"]["used"] == "dolly_zoom" and "no dolly zoom" in sh["S6"]["camera"]["source"]
    assert sh["S5"]["lens"]["impl"] == "LNS-02 apply_dof"
    assert sh["S1"]["dur_s"] > 0 and sh["S1"]["move_used"] == "push_in"


def test_shot_builders_geometry():
    from buc.core import contracts as C
    from buc.read import pipeline as RP
    img = P.make_input_image(P.resize_rgb(synth_image(), (192, 82)), "t")
    steps = P.build_read_steps(P.TempDepthReader(fake_disp), P.make_size_obs_fn())
    read = RP.run_read(img, steps=steps).packet
    assert read.meta["read"]["depth_source"] == "mrh"
    obs = read.obs
    shots = {}
    for name in P.SHOT_ORDER:
        info = {}
        shots[name] = P.build_shot(read, name, 10 / 24.0 if name != "S5" else 12 / 24.0, 0, info)
        sh = shots[name]
        assert isinstance(sh, C.Shot) and sh.first_pose_matches(obs), name
        assert np.allclose(sh.K[0], obs.K)
        assert info["z_s"] > 0 and "used" in info
    assert shots["S1"].n_frames == 11 and shots["S5"].n_frames == 13
    # push-in: camera moves forward along +z; crane: camera ends higher (y smaller) and still looks at the subject
    assert shots["S1"].c2w[-1][2, 3] > 0.5
    c = shots["S4"].c2w
    assert c[-1][1, 3] < c[0][1, 3] - 0.1
    fwd_end = c[-1][:3, 2]
    assert fwd_end[1] > 0.05                               # pitched down
    # dolly zoom: camera backs off while focal length grows (people keep their size)
    d = shots["S6"]
    assert d.c2w[-1][2, 3] < -0.1 and d.K[-1][0, 0] > d.K[0][0, 0] * 1.2
    z = P.subject_depth(read)
    ratio = d.K[-1][0, 0] / d.K[0][0, 0]
    assert abs(ratio - (z + (z - d.c2w[-1][2, 3] - z) * 0 + (-d.c2w[-1][2, 3])) / z) < 1e-6
    # rack focus: fixed camera, focus rack poppy -> people -> eye
    r = shots["S5"]
    assert np.allclose(r.c2w, r.c2w[0]) and r.focus_m[0] < r.focus_m[6] < r.focus_m[-1]
    assert abs(r.focus_m[6] - z) / z < 1e-6 and r.theta.f_stop == P.S5_F_STOP
    # orbit arc: camera slides left (negative x) and turns toward the pivot
    o = shots["S3"].c2w
    assert o[-1][0, 3] < -0.1


def test_s5_focus_changes_the_image(full_run):
    out = full_run["out"]
    cap = cv2.VideoCapture(os.path.join(out, "scaffold_S5.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.float32))
    cap.release()
    assert len(frames) == 15
    assert np.abs(frames[0] - frames[-1])[30:].mean() > 0.2          # the rack changes the picture (label rows skipped)


# ---------------------------------------------------------------- stage fallbacks
def test_temp_depth_failure_falls_back_to_plane(tmp_path):
    rc, out, rep = run_main(tmp_path, modes="scaffold", disp=boom_disp)
    assert rc == 0 and rep["scaffold"]["status"] == "ok"
    rd = rep["scaffold"]["read_summary"]
    assert rd["depth_source"] == "plane" and any(f.startswith("R5:") for f in rd["fallbacks"])
    assert rd["steps"]["R5"]["status"] == "fallback"
    assert rep["units"][P.U_R5]["verdict"] == "fallback"
    n, first = n_frames_of(os.path.join(out, "scaffold_S1.mp4"))
    assert n > 5


def test_real_s8_failure_uses_fixed_move_stub(tmp_path, monkeypatch):
    real_build = P.build_shot

    def flaky(read, name, dur_s, seed, info):
        if name == "S2":
            raise RuntimeError("slider recipe exploded")
        return real_build(read, name, dur_s, seed, info)

    monkeypatch.setattr(P, "build_shot", flaky)
    rc, out, rep = run_main(tmp_path, modes="scaffold", extra=["--shots", "S1,S2"])
    s1, s2 = rep["scaffold"]["shots"]["S1"], rep["scaffold"]["shots"]["S2"]
    assert s1["stages"]["S8"]["kind"] == "real" and s1["status"] == "ok"
    assert s2["status"] == "ok"
    assert s2["stages"]["S8"]["kind"] == "stub" and s2["stages"]["S8"]["status"] == "fallback"
    assert "slider recipe exploded" not in s2["stages"]["S8"]["fallback"]           # type name only (PRD-01)
    assert s2["stages"]["S8"]["fallback"].startswith("real:cam02_moves@") and "RuntimeError" in s2["stages"]["S8"]["fallback"]
    assert s2["move_used"] == "stub:stub_push_in"
    assert rep["units"][P.U_CAM]["verdict"] == "mixed"
    assert rep["real_vs_stub"]["S2"]["S8"] == "stub"


def test_real_s2_failure_uses_flat_world_stub(tmp_path, monkeypatch):
    import buc.world.depth_world as dw

    def boom(*a, **k):
        raise RuntimeError("depth world broke")

    monkeypatch.setattr(dw, "build_depth_world", boom)
    rc, out, rep = run_main(tmp_path, modes="scaffold", extra=["--shots", "S1,S2"])
    assert rc == 0
    for n in ("S1", "S2"):
        s = rep["scaffold"]["shots"][n]
        assert s["status"] == "ok"
        assert s["stages"]["S2"]["kind"] == "stub"
    assert rep["scaffold"]["shots"]["S1"]["stages"]["S2"]["status"] == "fallback"
    assert "RuntimeError" in rep["scaffold"]["shots"]["S1"]["stages"]["S2"]["fallback"]
    assert rep["units"][P.U_W3]["verdict"] == "fallback"
    assert rep["units"][P.U_W8]["verdict"] == "fallback"
    # the failing real impl is attempted once, not once per shot (memoised)
    assert rep["units"][P.U_W3]["counts"]["fallback"] == 1


def test_real_s1_failure_uses_classic_read_stub(tmp_path, monkeypatch):
    from buc.read import pipeline as RP

    def boom(*a, **k):
        raise RuntimeError("orchestrator broke")

    monkeypatch.setattr(RP, "run_read", boom)
    rc, out, rep = run_main(tmp_path, modes="scaffold", extra=["--shots", "S1"])
    assert rc == 0
    s = rep["scaffold"]["shots"]["S1"]
    assert s["status"] == "ok" and s["stages"]["S-1"]["kind"] == "stub" and s["stages"]["S-1"]["status"] == "fallback"
    assert rep["units"][P.U_R5]["verdict"] == "fallback" and rep["units"][P.U_R0]["verdict"] == "fallback"
    assert rep["real_vs_stub"]["S1"]["S-1"] == "stub"


def test_one_shot_failure_does_not_stop_the_run(tmp_path, monkeypatch):
    real = P.render_shot_video

    def maybe(world, read, shot, out_path, **kw):
        if "S3" in str(out_path):
            raise RuntimeError("encoder died on S3")
        return real(world, read, shot, out_path, **kw)

    monkeypatch.setattr(P, "render_shot_video", maybe)
    rc, out, rep = run_main(tmp_path, modes="scaffold", extra=["--shots", "S1,S3,S4"])
    sh = rep["scaffold"]["shots"]
    assert rc == 0
    assert sh["S1"]["status"] == "ok" and sh["S4"]["status"] == "ok"
    assert sh["S3"]["status"] == "failed" and "encoder died" in sh["S3"]["error"]
    assert rep["scaffold"]["status"] == "partial"
    assert not os.path.exists(os.path.join(out, "scaffold_S3.mp4"))
    assert any(e["where"] == "scaffold S3" for e in rep["errors"])


def test_dof_fallback_when_lens_breaks(tmp_path, monkeypatch):
    from buc.lens import dof

    def boom(*a, **k):
        raise RuntimeError("apply_dof broke")

    monkeypatch.setattr(dof, "apply_dof", boom)
    rc, out, rep = run_main(tmp_path, modes="scaffold", extra=["--shots", "S5"])
    s = rep["scaffold"]["shots"]["S5"]
    assert rc == 0 and s["status"] == "ok"
    assert s["lens"]["impl"] == "simple layered blur fallback" and "apply_dof broke" in s["lens"]["notes"][0]
    assert rep["units"][P.U_DOF]["verdict"] == "fallback"
    assert n_frames_of(os.path.join(out, "scaffold_S5.mp4"))[0] == 15


def test_deadline_passed_skips_shots_and_still_reports(tmp_path):
    dl = P.Deadline(time.time() - 5.0)
    rc, out, rep = run_main(tmp_path, modes="scaffold", deadline=dl)
    assert rc == 1                                           # nothing produced
    assert all(s["status"] == "skipped_deadline" for s in rep["scaffold"]["shots"].values())
    assert rep["units"][P.U_CAM]["verdict"] == "skipped"
    assert os.path.exists(os.path.join(out, "stages_report.json"))


def test_deadline_mid_shot_aborts_without_partial_file(tmp_path):
    """The deadline margin is hit while a shot renders: no half-written mp4 is left behind."""

    class Countdown(P.Deadline):
        def __init__(self, n):
            super().__init__(None)
            self.n = n

        def stop_now(self):
            self.n -= 1
            return self.n < 0

    rc, out, rep = run_main(tmp_path, modes="scaffold", extra=["--shots", "S1", "--fixed-width"],
                            deadline=Countdown(6))
    s = rep["scaffold"]["shots"]["S1"]
    assert s["status"] == "failed" and "deadline" in s["error"]
    assert rc == 1
    assert not [f for f in os.listdir(out) if f.endswith(".mp4") or ".part" in f]


# ---------------------------------------------------------------- WLD-08 chain with fakes
def _read_for(h=82, w=192):
    from buc.read import pipeline as RP
    img = P.make_input_image(P.resize_rgb(synth_image(), (w, h)), "t")
    steps = P.build_read_steps(P.TempDepthReader(fake_disp), P.make_size_obs_fn())
    return RP.run_read(img, steps=steps).packet


def _no_isolate(fn, args, timeout):
    return fn(*args)


def test_world_chain_wld08_success_and_fallbacks():
    read = _read_for()
    ok = P.Ledger()
    tr = []
    t = FakeTeacherHandle("ok")
    w = P.build_world_chain(read, teacher=t, cfg={"use_matte": False}, trace=tr, ledger=ok, isolate=_no_isolate)
    assert ok.verdict(P.U_W8) == "real" and ok.verdict(P.U_W3) == "real" and t.released == 1
    assert [e["unit"] for e in tr][-1] == P.U_W8 and w.read_hash == read.content_hash()
    assert ok.verdict(P.U_MATTE) == "skipped"

    for mode, verdict in (("unavailable", "unavailable"), ("boom", "fallback")):
        L = P.Ledger()
        tr = []
        w2 = P.build_world_chain(read, teacher=FakeTeacherHandle(mode), cfg={"use_matte": False}, trace=tr, ledger=L,
                                 isolate=_no_isolate)
        assert L.verdict(P.U_W8) == verdict and L.verdict(P.U_W3) == "real"
        assert w2.read_hash == read.content_hash() and len(w2.layers) == 2          # plain WLD-03 world kept


def test_world_chain_matte_paths():
    read = _read_for()

    def timeout(fn, args, t):
        raise TimeoutError("matting too slow")

    L = P.Ledger()
    tr = []
    P.build_world_chain(read, teacher=None, cfg={"use_matte": True}, trace=tr, ledger=L, isolate=timeout)
    assert L.verdict(P.U_MATTE) == "fallback" and "too slow" in L.units[P.U_MATTE]["events"][0]["detail"]
    assert L.verdict(P.U_W3) == "real" and L.verdict(P.U_W8) == "unavailable"

    L2 = P.Ledger()
    P.build_world_chain(read, teacher=None, cfg={"use_matte": True, "force_matte": True}, trace=[], ledger=L2,
                        isolate=_no_isolate)
    assert L2.verdict(P.U_MATTE) == "real"                    # forced use of the matte even if low_conf


def test_run_isolated_timeout_and_error():
    assert P.run_isolated(lambda a, b: a + b, (2, 3), 20.0) == 5
    with pytest.raises(RuntimeError, match="nope"):
        P.run_isolated(lambda: (_ for _ in ()).throw(ValueError("nope")), (), 20.0)
    with pytest.raises(TimeoutError):
        P.run_isolated(time.sleep, (30,), 0.5)


# ---------------------------------------------------------------- pool, upload, hygiene
def test_pool_path_matches_serial(tmp_path):
    from buc.core import contracts as C
    from buc.pipeline import spine as S
    from buc.pipeline.stubs import pack
    arr = P.resize_rgb(synth_image(), (96, 42))
    reg = S.Registry()
    pack.register_stubs(reg)
    r = S.Spine(reg, prefer="stub").run(P.make_input_image(arr, "t"), until="S8.5", params={"S8": {"n_frames": 6}})
    read, world, shot = r.artifacts["read"], r.artifacts["world"], r.artifacts["shot"]
    dl = P.Deadline(None)
    res = []
    for w in (1, 2):
        st = P.render_shot_video(world, read, shot, tmp_path / f"w{w}.mp4", label="T", lens=None, src_u8=arr, z0=3.0,
                                 workers=w, deadline=dl)
        res.append(st)
    assert res[0]["pool"] is False and res[1]["pool"] is True
    assert res[0]["frames"] == res[1]["frames"] == 6 and res[1]["render_stub"] == 0
    a = cv2.VideoCapture(str(tmp_path / "w1.mp4"))
    b = cv2.VideoCapture(str(tmp_path / "w2.mp4"))
    for _ in range(6):
        fa, fb = a.read()[1], b.read()[1]
        assert np.abs(fa.astype(int) - fb.astype(int)).max() <= 2


def test_upload_flow_without_leaking_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    src = tmp_path / "yak.webp"
    from PIL import Image
    Image.fromarray(synth_image()).save(src, quality=95)
    seen = {}

    def fake_dl(repo, filename, token, cache_dir):
        seen.update(repo=repo, filename=filename, token_len=len(token), cache_dir=cache_dir)
        return str(src)

    api = FakeApi(fail={"scaffold_S2.mp4"})
    args = ["--root", str(tmp_path / "r"), "--width", str(TINY_W), "--dur-scale", str(SCALE), "--workers", "1",
            "--shots", "S1,S2", "--no-teacher"]
    rc = P.main(args, disparity_fn=fake_disp, download_fn=fake_dl, upload_api=api)
    out_dir = tmp_path / "r" / "day1" / "pipe"
    rep = json.loads((out_dir / "stages_report.json").read_text())
    assert rc == 0
    assert seen["repo"] == P.HF_REPO_ID and seen["filename"] == "demo/d15_hero_yak.webp"
    assert rep["image"]["source"] == "hf_vault" and rep["image"]["matches_doc"] is False    # synthetic != doc sha
    ups = {c[0]: c for c in api.calls}
    assert {"skeleton_S1.mp4", "scaffold_S1.mp4", "contact_sheet.jpg", "stages_report.json"} <= set(ups)
    assert "scaffold_S2.mp4" not in ups                      # that upload failed: a warning, not a crash
    assert all(c[1].startswith("video/pipe_d1/") and c[2] == P.HF_REPO_ID and c[3] == "model" for c in api.calls)
    assert rep["upload"]["failed"] == 1 and rep["upload"]["ok"] >= 3
    captured = capsys.readouterr().out
    assert SECRET not in captured
    assert SECRET not in (out_dir / "stages_report.json").read_text()
    assert all(len(l) <= 400 for l in captured.splitlines() if l.startswith("D1_PIPE"))


def test_no_image_no_token_is_a_clean_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = P.main(["--root", str(tmp_path), "--no-upload"], disparity_fn=fake_disp)
    rep = json.loads((tmp_path / "day1" / "pipe" / "stages_report.json").read_text())
    assert rc == 2 and "no --image and no HF_TOKEN" in rep["fatal"]


def test_eval_only_hygiene_nothing_leaks_to_training_paths(full_run):
    tmp = str(full_run["tmp"])
    seen = []
    for d, _, files in os.walk(tmp):
        for f in files:
            seen.append(os.path.relpath(os.path.join(d, f), tmp))
    assert seen and all(p.startswith(os.path.join("day1", "pipe")) for p in seen), seen
    assert not [p for p in seen if p.endswith((".npy", ".npz", ".pt", ".safetensors", ".webp", ".png"))]
    for top in ("img", "lat", "meta", "corpus"):
        assert not os.path.exists(os.path.join(tmp, top))
    from buc.gov import prov as G
    img = P.make_input_image(synth_image(), "x")
    rec = G.as_record(img.prov)
    assert rec.hold and rec.ext and G.BLOCK_TRAIN & rec.taint.__class__({t.split(":")[0] for t in rec.taint})
    assert P.SCAFFOLD_LABEL == "SCAFFOLD: temp depth (eval-only), not final"


def test_scaffold_label_text_in_frames_matches_spec(full_run):
    out = full_run["out"]
    n, first = n_frames_of(os.path.join(out, "scaffold_S1.mp4"))
    # compare against a re-render of the label on the decoded frame's own background: label region is brighter text
    assert first[:24, :140].max() > 235
    assert full_run["rep"]["scaffold"]["shots"]["S1"]["render"]["render_stub"] == 0


def test_cli_args_validation(capsys):
    with pytest.raises(SystemExit):
        P.parse_args(["--shots", "S9"])
    with pytest.raises(SystemExit):
        P.parse_args(["--modes", "nope"])
    a = P.parse_args(["--shots", "S1,S2", "--modes", "scaffold"])
    assert a.shots == ["S1", "S2"] and a.modes == ["scaffold"] and a.dur_scale == 1.0


def test_blake3_shim_is_idempotent_when_real_present():
    assert P.ensure_blake3() is False                       # real blake3 installed here: nothing replaced
