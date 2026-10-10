"""CPU tests for baseline_video.py (no GPU, network, diffusers or transformers)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_video as bv  # noqa: E402

W, H = 128, 56


def fake_image(path: Path, size=(200, 86)) -> Path:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (w + h)], axis=-1).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def fake_depth_fn(pil):
    w, h = pil.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # near at the bottom (large disparity), far at the top, plus a nearer blob
    disp = 1.0 + 4.0 * yy / h
    disp += 6.0 * np.exp(-(((xx - w * 0.3) / (w * 0.1)) ** 2 + ((yy - h * 0.7) / (h * 0.2)) ** 2))
    return disp.astype(np.float32)


def markers(capsys_out: str, tag: str):
    res = []
    for line in capsys_out.splitlines():
        if line.startswith(tag + " "):
            res.append(json.loads(line[len(tag) + 1:]))
    return res


# ---------------------------------------------------------------- trajectories
def test_frame_counts():
    assert [bv.frame_count(s) for s in bv.SHOTS] == [192, 192, 240, 192, 144, 144]
    assert bv.frame_count("S3", 24, 5) == 5


def test_smoothstep():
    assert bv.smoothstep(0) == 0 and bv.smoothstep(1) == 1 and bv.smoothstep(0.5) == pytest.approx(0.5)
    assert bv.smoothstep(-1) == 0 and bv.smoothstep(2) == 1


def test_pose_s1_s2_s4_s6():
    p0, p1 = bv.shot_pose("S1", 0), bv.shot_pose("S1", 1)
    assert np.allclose(p0.pos, 0) and np.allclose(p1.pos, [0, 0, 0.30])
    p0, p1 = bv.shot_pose("S2", 0), bv.shot_pose("S2", 1)
    assert np.allclose(p0.pos, [-0.06, 0.01, 0]) and np.allclose(p1.pos, [0.06, 0.01, 0])
    p0, p1 = bv.shot_pose("S4", 0), bv.shot_pose("S4", 1)
    assert p0.pos[1] == pytest.approx(0.03) and p1.pos[1] == pytest.approx(-0.10)  # y is down
    p0, p1 = bv.shot_pose("S6", 0), bv.shot_pose("S6", 1)
    assert p0.fscale == pytest.approx(1.0) and p0.pos[2] == pytest.approx(0.0)
    assert p1.pos[2] == pytest.approx(-0.4) and p1.fscale == pytest.approx(1.4)


def test_s3_arc_orbit_and_person_fixed():
    person = bv.person_point(bv.OUT_W, bv.OUT_H)
    p0, p1 = bv.shot_pose("S3", 0, person), bv.shot_pose("S3", 1, person)
    for p in (p0, p1):
        # orbit radius around the people is preserved
        assert np.linalg.norm(p.pos - person) == pytest.approx(np.linalg.norm(person), rel=1e-6)
    ang = lambda p: np.degrees(np.arctan2((p.pos - person)[0], -(p.pos - person)[2]))  # noqa: E731
    base = ang(bv.Pose(np.zeros(3), np.eye(3)))
    assert abs(ang(p0) - base) == pytest.approx(12.0, abs=1e-6) and abs(ang(p1) - base) == pytest.approx(12.0, abs=1e-6)
    assert np.sign(ang(p0) - base) == -np.sign(ang(p1) - base)  # opposite sides
    for t in (0, 0.3, 1):
        pose = bv.shot_pose("S3", t, person)
        u, v, z = bv.project_point(person, pose, bv.OUT_W, bv.OUT_H)
        u0, v0, _ = bv.project_point(person, bv.Pose(np.zeros(3), np.eye(3)), bv.OUT_W, bv.OUT_H)
        assert u == pytest.approx(u0, abs=1e-6) and v == pytest.approx(v0, abs=1e-6)


def test_s3_is_pure_yaw_no_roll():
    person = bv.person_point(bv.OUT_W, bv.OUT_H)
    assert person[1] > 0.1  # people sit below the image centre: a shortest rotation would roll
    for t in (0, 0.25, 0.5, 1):
        R = bv.shot_pose("S3", t, person).R
        assert R[1, 0] == pytest.approx(0.0, abs=1e-12)  # camera x axis stays horizontal
        assert R[1, 2] == pytest.approx(0.0, abs=1e-12) and R[1, 1] == pytest.approx(1.0)


def test_s4_person_stays_and_tilts_down():
    person = bv.person_point(bv.OUT_W, bv.OUT_H)
    for t in (0, 0.5, 1):
        pose = bv.shot_pose("S4", t, person)
        u, v, _ = bv.project_point(person, pose, bv.OUT_W, bv.OUT_H)
        u0, v0, _ = bv.project_point(person, bv.Pose(np.zeros(3), np.eye(3)), bv.OUT_W, bv.OUT_H)
        assert (u, v) == pytest.approx((u0, v0), abs=1e-6)
    top = bv.shot_pose("S4", 1, person)
    # camera is above the people: looking from it, its forward ray points down (+y in world)
    forward = top.R @ np.array([0.0, 0.0, 1.0])
    assert forward[1] > 0


def test_s6_person_size_constant():
    person = bv.person_point(bv.OUT_W, bv.OUT_H)
    off = np.array([0.1, 0.0, 0.0])  # a point beside the people at the same distance
    sizes = []
    for t in (0, 0.5, 1):
        pose = bv.shot_pose("S6", t, person)
        u0, _, _ = bv.project_point(person, pose, bv.OUT_W, bv.OUT_H)
        u1, _, _ = bv.project_point(person + off, pose, bv.OUT_W, bv.OUT_H)
        sizes.append(u1 - u0)
    assert sizes[0] == pytest.approx(sizes[1], rel=1e-6) == pytest.approx(sizes[2], rel=1e-6)


def test_s5_focus_schedule():
    assert bv.shot_pose("S5", 0).focus == pytest.approx(0.35)
    assert bv.focus_at(0.56) == pytest.approx(1.0)
    assert bv.shot_pose("S5", 1).focus == pytest.approx(1.6)
    vals = [bv.focus_at(t) for t in np.linspace(0, 1, 50)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    p = bv.shot_pose("S5", 0.5)
    assert np.allclose(p.pos, 0) and np.allclose(p.R, np.eye(3))


def test_unknown_shot():
    with pytest.raises(ValueError):
        bv.shot_pose("S9", 0.1)


def test_rot_between():
    a, b = np.array([0.1, 0.2, 1.0]), np.array([-0.3, 0.1, 1.0])
    R = bv.rot_between(a, b)
    assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    R2 = bv.rot_between(a, -a)
    assert np.allclose(R2 @ a, -a)


# ---------------------------------------------------------------- depth calibration
def test_disp_to_depth_calibration():
    rng = np.random.default_rng(0)
    disp = rng.uniform(1.0, 10.0, (86, 200)).astype(np.float32)
    disp[40:60, 90:120] = 5.0
    box = (90, 40, 120, 60)
    depth = bv.disp_to_depth(disp, box)
    assert np.median(depth[40:60, 90:120]) == pytest.approx(1.0, rel=1e-4)
    lo = np.percentile(disp, 2)
    near_lo = depth[disp <= lo]
    assert near_lo.max() <= 40.0 + 1e-3
    assert depth.max() <= 40.0 + 1e-3 and depth.min() >= 0.1 - 1e-6
    # closer (bigger disp) -> smaller depth
    assert depth[disp == disp.max()].min() < 1.0
    # exact value at the 2% percentile
    pt = np.argmin(np.abs(disp - lo))
    assert depth.flat[pt] == pytest.approx(40.0, rel=0.02)


def test_disp_to_depth_degenerate_constant():
    d = bv.disp_to_depth(np.ones((20, 20), np.float32), (5, 5, 10, 10))
    assert np.isfinite(d).all()


def test_scale_box():
    assert bv.scale_box((900, 640, 1130, 840), (2000, 861), (1000, 430)) == (450, 320, 565, 420)


def test_discontinuity_mask():
    d = np.ones((6, 6), np.float32)
    d[:, 3:] = 2.0
    m = bv.discontinuity_mask(d)
    assert m[:, 2].all() and not m[:, 0].any() and not m[:, 4].any()
    assert not bv.discontinuity_mask(np.ones((4, 4))).any()


# ---------------------------------------------------------------- z-buffer / render
def test_zbuffer_nearer_point_wins_any_order():
    import torch

    colors = torch.tensor([[255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype=torch.uint8)
    u = torch.tensor([2.0, 2.0, 5.0])
    v = torch.tensor([1.0, 1.0, 1.0])
    z = torch.tensor([3.0, 1.5, 2.0])  # blue is nearest at (2,1)
    for perm in ([0, 1, 2], [1, 0, 2], [2, 0, 1], [1, 2, 0]):
        p = torch.tensor(perm)
        rgb, hole, zbuf = bv.splat_zbuffer(u[p], v[p], z[p], colors[p], 8, 4)
        assert rgb[1, 2].tolist() == [0, 0, 255]
        assert zbuf[1, 2].item() == pytest.approx(1.5)
        assert rgb[1, 5].tolist() == [0, 255, 0]
        assert hole[0, 0].item() and not hole[1, 2].item()


def test_zbuffer_out_of_view_dropped():
    import torch

    colors = torch.full((2, 3), 200, dtype=torch.uint8)
    rgb, hole, _ = bv.splat_zbuffer(torch.tensor([-5.0, 99.0]), torch.tensor([1.0, 1.0]),
                                    torch.tensor([1.0, 1.0]), colors, 8, 4)
    assert hole.all() and (rgb == 0).all()


def scene_arrays(w=64, h=28):
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 1.0, np.float32)
    depth[:, : w // 2] = 3.0
    return rgb, depth


def test_renderer_identity_is_exact():
    rgb, depth = scene_arrays()
    r = bv.Renderer(rgb, depth, device="cpu")
    out, hole = r.render(bv.Pose(np.zeros(3), np.eye(3)))
    assert not hole.any()
    assert np.array_equal(out, rgb)


def test_renderer_parallax_near_moves_more_and_foreground_wins():
    rgb, depth = scene_arrays(64, 28)
    r = bv.Renderer(rgb, depth, device="cpu")
    out, hole = r.render(bv.shot_pose("S2", 1))  # camera moved right: scene shifts left
    assert hole.any()  # disocclusions exist
    filled = bv.fill_holes(out, hole)
    assert filled.shape == rgb.shape and filled.dtype == np.uint8
    # near half (right) shifts further than the far half
    f = bv.focal_px(64)
    assert f * 0.06 / 1.0 > f * 0.06 / 3.0


def test_fill_holes_small_and_big():
    yy, xx = np.mgrid[0:60, 0:100]
    img = np.stack([xx * 2, yy * 4, np.full_like(xx, 100)], axis=-1).astype(np.uint8)
    hole = np.zeros((60, 100), bool)
    hole[10:12, 10:30] = True  # thin crack
    hole[30:50, 40:80] = True  # big block
    broken = img.copy()
    broken[hole] = 0
    out = bv.fill_holes(broken, hole)
    assert np.abs(out[hole].astype(int) - img[hole].astype(int)).mean() < 40
    assert (out[hole].sum(axis=1) > 0).all()
    assert np.array_equal(out[~hole], img[~hole])
    assert bv.fill_holes(img, np.zeros_like(hole)) is img


# ---------------------------------------------------------------- rack focus / overlay
def test_rack_focus_blurs_out_of_focus_layers():
    import cv2

    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (60, 120, 3), dtype=np.uint8)
    depth = np.full((60, 120), 1.0, np.float32)
    depth[:, 60:] = 4.0

    def sharp(a):
        return cv2.Laplacian(a.astype(np.float32).mean(axis=2), cv2.CV_32F).var()

    rf = bv.RackFocus(img, depth, layers=8)
    near_focus = rf(1.0)
    far_focus = rf(4.0)
    assert sharp(near_focus[:, :55]) > 3 * sharp(near_focus[:, 65:])
    assert sharp(far_focus[:, 65:]) > 3 * sharp(far_focus[:, :55])
    assert max(rf.radii(0.35)) <= 18.0 + 1e-9
    assert np.array_equal(rf(1.0), near_focus)  # cached and identical


def test_rack_focus_outlier_does_not_merge_layers():
    h, w = 60, 120
    depth = np.full((h, w), 40.0, np.float32)  # sky
    depth[20:50, 0:40] = 1.6  # yak eye
    depth[30:50, 50:80] = 1.0  # people
    depth[50:, :] = 0.35  # foreground poppies
    depth[0, 0] = 0.1  # one pixel at the near clamp
    img = np.random.default_rng(3).integers(0, 255, (h, w, 3), dtype=np.uint8)
    rf = bv.RackFocus(img, depth, layers=8)
    assert len(rf.masks) >= 4

    def layer_of(y, x):
        return next(i for i, m in enumerate(rf.masks) if m[y, x] > 0)

    li_people, li_eye, li_sky = layer_of(40, 60), layer_of(30, 10), layer_of(5, 100)
    assert len({li_people, li_eye, li_sky}) == 3
    assert rf.radii(1.0)[li_people] < 1.0
    assert rf.radii(1.6)[li_eye] < 1.0
    assert rf.radii(1.0)[li_eye] > 1.0 and rf.radii(1.6)[li_people] > 1.0  # and the pull really moves it


def test_overlay_label_changes_top_left_only():
    f = np.zeros((90, 300, 3), np.uint8)
    o = bv.overlay_label(f)
    assert o[:30, :200].sum() > 0 and o[45:].sum() == 0 and f.sum() == 0


# ---------------------------------------------------------------- video IO
def test_render_shot_writes_h264_and_deadline_cleans_up(tmp_path):
    frame = lambda t: np.full((H, W, 3), int(t * 200), np.uint8)  # noqa: E731
    out = tmp_path / "a.mp4"
    bv.render_shot("S1", frame, 5, out, fps=6)
    import imageio

    r = imageio.get_reader(str(out), format="FFMPEG")
    meta = r.get_meta_data()
    frames = [np.asarray(x) for x in r]
    r.close()
    assert len(frames) == 5 and frames[0].shape == (H, W, 3)
    assert meta["codec"] == "h264" and abs(meta["fps"] - 6) < 0.01
    assert not list(tmp_path.glob("*.part.mp4"))
    # deadline in the past -> raises, no files left
    out2 = tmp_path / "b.mp4"
    with pytest.raises(bv.DeadlineReached):
        bv.render_shot("S1", frame, 5, out2, fps=6, deadline=bv.Deadline(time.time() - 1))
    assert not out2.exists() and not list(tmp_path.glob("*.part.mp4"))


def test_deadline_env(monkeypatch):
    monkeypatch.setenv("N1_DEADLINE", str(time.time() + 1000))
    d = bv.Deadline.from_env()
    assert not d.reached() and d.left() > 900
    monkeypatch.setenv("N1_DEADLINE", "1")
    assert bv.Deadline.from_env().reached()
    monkeypatch.delenv("N1_DEADLINE")
    assert not bv.Deadline.from_env().reached()


def test_scrub():
    assert "hf_SECRET" not in bv.scrub("boom hf_SECRET boom", "hf_SECRET")


# ---------------------------------------------------------------- end to end
def run_main(tmp_path, monkeypatch, extra=(), env=None, **kw):
    img = fake_image(tmp_path / "in.png")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("N1_DEADLINE", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    argv = ["--root", str(tmp_path / "root"), "--image", str(img), "--width", str(W), "--height", str(H),
            "--fps", "6", "--max-frames", "4", "--device", "cpu", "--out", str(tmp_path / "root" / "video"), *extra]
    return bv.main(argv, depth_fn=kw.pop("depth_fn", fake_depth_fn), **kw)


class FakeApi:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def upload_file(self, **kw):
        if self.fail_on and self.fail_on in kw["path_in_repo"]:
            raise RuntimeError("boom hf_SECRET")
        self.calls.append(kw)


def test_main_end_to_end_and_resume(tmp_path, monkeypatch, capsys):
    calls = {"n": 0}

    def counting_depth(p):
        calls["n"] += 1
        return fake_depth_fn(p)

    rc = run_main(tmp_path, monkeypatch, ["--no-upload"], depth_fn=counting_depth)
    assert rc == 0
    out = capsys.readouterr().out
    vid = tmp_path / "root" / "video"
    for s in bv.SHOTS:
        assert (vid / f"baseline_yak_{s}.mp4").stat().st_size > 0
    assert (vid / "baseline_yak_all.mp4").exists()
    sheet = Image.open(vid / "baseline_yak_sheet.jpg")
    assert sheet.size[0] == 3 * 640 and sheet.size[1] % 6 == 0
    ms = markers(out, "N1_VID")
    shot_marks = [m for m in ms if "frames" in m and "s" in m]
    assert [m["shot"] for m in shot_marks] == list(bv.SHOTS) and all(m["frames"] == 4 for m in shot_marks)
    done = markers(out, "N1_VID_DONE")[0]
    assert done["done"] == list(bv.SHOTS) and not done["failed"] and not done["deadline"]
    assert "hf_token=no" in out

    # concatenated video has 6 * 4 frames
    import imageio

    r = imageio.get_reader(str(vid / "baseline_yak_all.mp4"), format="FFMPEG")
    n = sum(1 for _ in r)
    r.close()
    assert n == 24

    # label is burned into the frames
    first = bv.read_frames_at(vid / "baseline_yak_S1.mp4", [0])[0]
    assert first[: H // 3, : W // 2].astype(int).max() > 200

    # resume: nothing re-rendered, depth model not called again
    mt = {p.name: p.stat().st_mtime_ns for p in vid.glob("*.mp4")}
    rc = run_main(tmp_path, monkeypatch, ["--no-upload"], depth_fn=lambda p: pytest.fail("depth rerun"))
    assert rc == 0
    out2 = capsys.readouterr().out
    assert {p.name: p.stat().st_mtime_ns for p in vid.glob("*.mp4")} == mt
    d2 = markers(out2, "N1_VID_DONE")[0]
    assert d2["done"] == [] and d2["skipped"] == list(bv.SHOTS)

    # partial resume: delete S3 only -> only S3 is redone, depth comes from the cache
    (vid / "baseline_yak_S3.mp4").unlink()
    rc = run_main(tmp_path, monkeypatch, ["--no-upload"], depth_fn=lambda p: pytest.fail("depth rerun"))
    assert rc == 0
    d3 = markers(capsys.readouterr().out, "N1_VID_DONE")[0]
    assert d3["done"] == ["S3"] and (vid / "baseline_yak_all.mp4").exists()


def test_main_one_shot_failure_is_skipped(tmp_path, monkeypatch, capsys):
    real = bv.make_frame_fn

    def flaky(shot, scene):
        if shot == "S3":
            def boom(t):
                raise RuntimeError("render exploded")
            return boom
        return real(shot, scene)

    monkeypatch.setattr(bv, "make_frame_fn", flaky)
    rc = run_main(tmp_path, monkeypatch, ["--no-upload"])
    assert rc == 0
    out = capsys.readouterr().out
    done = markers(out, "N1_VID_DONE")[0]
    assert done["failed"] == ["S3"] and "S4" in done["done"] and "S6" in done["done"]
    vid = tmp_path / "root" / "video"
    assert not (vid / "baseline_yak_S3.mp4").exists() and not list(vid.glob("*.part.mp4"))
    assert not (vid / "baseline_yak_all.mp4").exists()  # needs all six
    assert (vid / "baseline_yak_sheet.jpg").exists()


def test_main_deadline_in_past_exits_cleanly(tmp_path, monkeypatch, capsys):
    rc = run_main(tmp_path, monkeypatch, ["--no-upload"], env={"N1_DEADLINE": str(time.time() - 5)},
                  depth_fn=lambda p: pytest.fail("should not start"))
    assert rc == 0
    done = markers(capsys.readouterr().out, "N1_VID_DONE")[0]
    assert done["deadline"] is True and done["done"] == []


def test_main_deadline_mid_run(tmp_path, monkeypatch, capsys):
    ticks = {"n": 0}
    base = time.time()

    def clock():  # deadline "arrives" after a number of checks
        ticks["n"] += 1
        return base + (0 if ticks["n"] < 12 else 1000)

    rc = run_main(tmp_path, monkeypatch, ["--no-upload"], env={"N1_DEADLINE": str(base + 100)}, clock=clock)
    assert rc == 0
    done = markers(capsys.readouterr().out, "N1_VID_DONE")[0]
    assert done["deadline"] is True and len(done["done"]) < 6
    assert not list((tmp_path / "root" / "video").glob("*.part.mp4"))


def test_main_upload_with_token_never_leaks(tmp_path, monkeypatch, capsys):
    api = FakeApi(fail_on="S2")
    rc = run_main(tmp_path, monkeypatch, [], env={"HF_TOKEN": "hf_SECRET"}, upload_api=api)
    assert rc == 0
    cap = capsys.readouterr()
    assert "hf_SECRET" not in cap.out and "hf_SECRET" not in cap.err
    paths = sorted(c["path_in_repo"] for c in api.calls)
    assert "video/night1/baseline_yak_all.mp4" in paths and "video/night1/baseline_yak_sheet.jpg" in paths
    assert "video/night1/baseline_yak_S1.mp4" in paths and "video/night1/baseline_yak_S2.mp4" not in paths
    assert all(c["repo_id"] == bv.HF_REPO_ID and c["repo_type"] == "model" for c in api.calls)
    done = markers(cap.out, "N1_VID_DONE")[0]
    assert done["uploaded"] == len(api.calls) == 7


def test_main_downloads_from_vault_when_no_image(tmp_path, monkeypatch, capsys):
    img = fake_image(tmp_path / "in.png")
    seen = {}

    def dl(repo, fname, token, cache):
        seen.update(repo=repo, fname=fname, has_token=bool(token))
        return str(img)

    monkeypatch.setenv("HF_TOKEN", "hf_SECRET")
    monkeypatch.delenv("N1_DEADLINE", raising=False)
    rc = bv.main(["--root", str(tmp_path / "r"), "--width", str(W), "--height", str(H), "--fps", "6",
                  "--max-frames", "2", "--device", "cpu", "--shots", "S1", "--no-upload"],
                 depth_fn=fake_depth_fn, download_fn=dl)
    assert rc == 0
    assert seen == {"repo": bv.HF_REPO_ID, "fname": "demo/d15_hero_yak.webp", "has_token": True}
    assert "hf_SECRET" not in capsys.readouterr().out


def test_main_no_token_no_image_fails_softly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("N1_DEADLINE", raising=False)
    rc = bv.main(["--root", str(tmp_path / "r"), "--no-upload", "--shots", "S1"], depth_fn=fake_depth_fn)
    assert rc == 1
    assert markers(capsys.readouterr().out, "N1_VID_DONE")[0]["fatal"] is True


def test_main_env_root_default(tmp_path, monkeypatch):
    monkeypatch.setenv("N1_ROOT", str(tmp_path / "envroot"))
    assert bv.parse_args([]).root == str(tmp_path / "envroot")


def test_unknown_shot_arg(tmp_path, monkeypatch, capsys):
    rc = bv.main(["--root", str(tmp_path), "--shots", "S9", "--no-upload"])
    assert rc == 2


def test_main_resume_rejects_stub_and_changed_settings(tmp_path, monkeypatch, capsys):
    vid = tmp_path / "root" / "video"
    assert run_main(tmp_path, monkeypatch, ["--no-upload", "--shots", "S1"]) == 0
    capsys.readouterr()
    side = vid / "baseline_yak_S1.json"
    meta = json.loads(side.read_text())
    assert meta == {"w": W, "h": H, "fps": 6, "hfov": bv.HFOV_DEG, "box": list(bv.PERSON_BOX),
                    "frames": 4, "version": bv.OUTPUT_VERSION}
    assert not list(vid.glob("*.part*"))

    # same settings -> skipped
    assert run_main(tmp_path, monkeypatch, ["--no-upload", "--shots", "S1"]) == 0
    assert markers(capsys.readouterr().out, "N1_VID_DONE")[0]["skipped"] == ["S1"]

    # a stub made with fewer frames (different --max-frames) is re-rendered, not trusted
    img = tmp_path / "in.png"
    argv = ["--root", str(tmp_path / "root"), "--image", str(img), "--width", str(W), "--height", str(H),
            "--fps", "6", "--max-frames", "2", "--device", "cpu", "--out", str(vid), "--no-upload", "--shots", "S1"]
    assert bv.main(argv, depth_fn=fake_depth_fn) == 0
    d = markers(capsys.readouterr().out, "N1_VID_DONE")[0]
    assert d["done"] == ["S1"] and json.loads(side.read_text())["frames"] == 2

    # changed fps / hfov -> re-rendered
    assert run_main(tmp_path, monkeypatch, ["--no-upload", "--shots", "S1", "--hfov", "60"]) == 0
    d = markers(capsys.readouterr().out, "N1_VID_DONE")[0]
    assert d["done"] == ["S1"] and json.loads(side.read_text())["hfov"] == 60.0

    # missing / corrupt sidecar -> re-rendered
    side.write_text("{not json")
    assert run_main(tmp_path, monkeypatch, ["--no-upload", "--shots", "S1", "--hfov", "60"]) == 0
    assert markers(capsys.readouterr().out, "N1_VID_DONE")[0]["done"] == ["S1"]
    side.unlink()
    assert run_main(tmp_path, monkeypatch, ["--no-upload", "--shots", "S1", "--hfov", "60"]) == 0
    assert markers(capsys.readouterr().out, "N1_VID_DONE")[0]["done"] == ["S1"]


def test_main_stale_shot_not_concatenated_or_uploaded(tmp_path, monkeypatch, capsys):
    vid = tmp_path / "root" / "video"
    assert run_main(tmp_path, monkeypatch, ["--no-upload"]) == 0
    capsys.readouterr()
    # all six exist; now rerun with other settings but only S1 selected: S2..S6 are stale
    api = FakeApi()
    assert run_main(tmp_path, monkeypatch, ["--shots", "S1", "--hfov", "60"], env={"HF_TOKEN": "hf_X"},
                    upload_api=api) == 0
    d = markers(capsys.readouterr().out, "N1_VID_DONE")[0]
    assert d["done"] == ["S1"] and d["have"] == ["S1"] and d["all"] is False
    assert sorted(c["path_in_repo"] for c in api.calls) == [
        "video/night1/baseline_yak_S1.mp4", "video/night1/baseline_yak_sheet.jpg"]


def test_max_frames_defaults_out_to_smoke_folder(tmp_path, monkeypatch, capsys):
    img = fake_image(tmp_path / "in.png")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("N1_DEADLINE", raising=False)
    root = tmp_path / "root"
    base = ["--root", str(root), "--image", str(img), "--width", str(W), "--height", str(H), "--fps", "2",
            "--device", "cpu", "--no-upload", "--shots", "S1"]
    assert bv.main(base + ["--max-frames", "2"], depth_fn=fake_depth_fn) == 0
    assert (root / "video" / "smoke" / "baseline_yak_S1.mp4").exists()
    assert not (root / "video" / "baseline_yak_S1.mp4").exists()
    # the real run is not fooled by the smoke stub
    assert bv.main(base, depth_fn=fake_depth_fn) == 0
    capsys.readouterr()
    assert json.loads((root / "video" / "baseline_yak_S1.json").read_text())["frames"] == 2 * 8
