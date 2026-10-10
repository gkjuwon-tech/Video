"""RDR-06 실제 인코딩 계약(CPU, libx264): 프레임 수·fps·타임스탬프·색·결정론·스트리밍·원자성·타임라인 이음.
GPU(NVENC) 테스트는 마지막에 @pytest.mark.gpu 로 분리한다."""
import hashlib
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from buc.core.contracts import Frames
from buc.render import video as V


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _ffprobe_json(path, entries, extra=()):
    p = subprocess.run(["ffprobe", "-v", "error", *extra, "-show_entries", entries, "-of", "json", str(path)],
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


# ---------------------------------------------------------------------------------------------------------------
# 기본 왕복: 프레임 수·해상도·fps·색 태그·화질
# ---------------------------------------------------------------------------------------------------------------
def test_encode_frames_probe_and_result(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(24, 64, 96)
    fr = Frames(rgb=rgb, fps=24.0, shot_hash="s" * 8, world_hash="w" * 8)
    out = tmp_path / "sub" / "a.mp4"            # 상위 디렉터리는 자동 생성
    res = V.encode_frames(fr, out, encoder="cpu", crf=18, preset="veryfast")
    assert isinstance(res, V.EncodeResult)
    assert Path(res.path) == out and out.is_file()
    assert (res.n_frames, res.width, res.height, res.fps) == (24, 96, 64, 24.0)
    assert res.codec == "h264" and res.encoder == "libx264"
    assert res.bytes == out.stat().st_size > 0
    assert isinstance(res.cmd, tuple) and res.cmd[-1] != str(out)  # 최종 경로가 아니라 임시 파트 파일에 쓴다
    info = V.probe(out)
    assert isinstance(info, V.VideoInfo)
    assert info.codec == "h264" and (info.width, info.height) == (96, 64)
    assert info.n_frames == 24 and info.fps == 24.0
    assert abs(info.duration_s - 1.0) < 1e-3
    assert info.pix_fmt == "yuv420p" and info.color_space == "bt709" and info.color_range == "tv"


def test_decode_quality(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(24, 64, 96)
    V.encode_frames(rgb, tmp_path / "q.mp4", fps=24, encoder="cpu", crf=18, preset="veryfast")
    dec = V.decode_frames(tmp_path / "q.mp4")
    assert dec.dtype == np.uint8 and dec.shape == (24, 64, 96, 3)
    ref = V.to_uint8(rgb)
    assert _psnr(dec, ref) >= 36.0
    assert np.abs(dec.astype(int) - ref.astype(int)).max() <= 24    # 색 행렬이 틀리면 훨씬 크다
    # 프레임 순서 보존: 시간 t 에 따라 b 채널이 단조 감소한다
    means = dec[..., 2].mean(axis=(1, 2))
    assert np.all(np.diff(means) < 0)


def test_color_matrix_bt709_limited_range(ffmpeg_bin, tmp_path):
    """독립 검증: 인코더가 쓴 YUV 평면을 ffmpeg 로 그대로 꺼내 BT.709 제한 범위 공식과 비교한다."""
    cols = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0.5), (0.8, 0.4, 0.2), (0, 0, 0), (1, 1, 1)]
    h = w = 32
    for i, (r, g, b) in enumerate(cols):
        frames = np.empty((2, h, w, 3), np.float32)
        frames[...] = np.array([r, g, b], np.float32)
        p = tmp_path / f"c{i}.mp4"
        V.encode_frames(frames, p, fps=24, encoder="cpu", crf=0, preset="ultrafast")
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                             capture_output=True, timeout=30).stdout
        fs = h * w * 3 // 2
        assert len(raw) == 2 * fs
        y = np.frombuffer(raw[:h * w], np.uint8).astype(float).mean()
        u = np.frombuffer(raw[h * w:h * w + h * w // 4], np.uint8).astype(float).mean()
        v = np.frombuffer(raw[h * w + h * w // 4:fs], np.uint8).astype(float).mean()
        ey = 16 + 219 * (0.2126 * r + 0.7152 * g + 0.0722 * b)
        eu = 128 + 224 * (-0.1146 * r - 0.3854 * g + 0.5 * b)
        ev = 128 + 224 * (0.5 * r - 0.4542 * g - 0.0458 * b)
        assert abs(y - ey) <= 2.0, (cols[i], y, ey)
        assert abs(u - eu) <= 2.0, (cols[i], u, eu)
        assert abs(v - ev) <= 2.0, (cols[i], v, ev)


def test_fractional_fps(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(12, 32, 32)
    res = V.encode_frames(rgb, tmp_path / "f.mp4", fps=24000 / 1001, encoder="cpu", preset="ultrafast")
    assert abs(res.fps - 24000 / 1001) < 1e-9
    info = V.probe(tmp_path / "f.mp4")
    assert info.n_frames == 12 and abs(info.fps - 24000 / 1001) < 1e-6
    assert abs(info.duration_s - 12 * 1001 / 24000) < 2e-3


def test_frame_timestamps_follow_timeline(ffmpeg_bin, make_clip, tmp_path):
    n = 24
    V.encode_frames(make_clip(n, 32, 32), tmp_path / "t.mp4", fps=24, encoder="cpu", preset="veryfast")
    d = _ffprobe_json(tmp_path / "t.mp4", "frame=best_effort_timestamp_time", ("-select_streams", "v:0"))
    t = np.array([float(f["best_effort_timestamp_time"]) for f in d["frames"]])
    assert t.shape == (n,)
    np.testing.assert_allclose(t, V.frame_times(n, 24), rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------------------------------------------
# 결정론
# ---------------------------------------------------------------------------------------------------------------
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_libx264_output_is_bit_deterministic(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(24, 64, 96)
    V.encode_frames(rgb, tmp_path / "d1.mp4", fps=24, encoder="cpu", preset="veryfast")
    V.encode_frames(rgb, tmp_path / "other_name.mp4", fps=24, encoder="cpu", preset="veryfast")
    assert _sha(tmp_path / "d1.mp4") == _sha(tmp_path / "other_name.mp4")   # 파일명·실행 시각이 바이트에 새지 않는다


def test_input_not_mutated(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(6, 32, 32)
    rgb.flags.writeable = False
    h0 = hashlib.sha256(rgb.tobytes()).hexdigest()
    V.encode_frames(rgb, tmp_path / "m.mp4", fps=24, encoder="cpu", preset="ultrafast")   # 읽기 전용 입력 허용
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == h0


# ---------------------------------------------------------------------------------------------------------------
# 스트리밍 쓰기
# ---------------------------------------------------------------------------------------------------------------
def test_writer_streaming_equals_bulk(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(10, 32, 48)
    V.encode_frames(rgb, tmp_path / "bulk.mp4", fps=24, encoder="cpu", preset="veryfast")
    with V.VideoWriter(tmp_path / "stream.mp4", width=48, height=32, fps=24, encoder="cpu", preset="veryfast") as w:
        assert w.n_frames == 0
        w.write(rgb[0])                 # (H,W,3) 한 장
        assert w.n_frames == 1
        w.write(rgb[1:4])               # (k,H,W,3) 묶음
        for i in range(4, 10):
            w.write(rgb[i])
        assert w.n_frames == 10
    assert w.result is not None and w.result.n_frames == 10
    assert (tmp_path / "stream.mp4").is_file()
    assert _sha(tmp_path / "bulk.mp4") == _sha(tmp_path / "stream.mp4")
    assert V.probe(tmp_path / "stream.mp4").n_frames == 10


def test_writer_accepts_uint8_same_as_float(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(4, 32, 32)
    V.encode_frames(rgb, tmp_path / "f.mp4", fps=24, encoder="cpu", preset="ultrafast")
    V.encode_frames(V.to_uint8(rgb), tmp_path / "u.mp4", fps=24, encoder="cpu", preset="ultrafast")
    assert _sha(tmp_path / "f.mp4") == _sha(tmp_path / "u.mp4")


def test_writer_close_is_idempotent_and_blocks_writes(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(3, 32, 32)
    w = V.VideoWriter(tmp_path / "w.mp4", width=32, height=32, fps=24, encoder="cpu", preset="ultrafast")
    w.write(rgb)
    r1 = w.close()
    r2 = w.close()
    assert r1 == r2 and r1.n_frames == 3
    with pytest.raises(V.VideoError):
        w.write(rgb)


def test_1080p_and_4k_h264(ffmpeg_bin, tmp_path):
    for name, (wd, ht) in (("1080p", (1920, 1080)), ("4k", (3840, 2160))):
        assert V.resolution(name) == (wd, ht)
        base = np.zeros((ht, wd, 3), np.uint8)
        base[..., 0] = np.linspace(0, 255, wd, dtype=np.uint8)[None, :]
        base[..., 1] = np.linspace(0, 255, ht, dtype=np.uint8)[:, None]
        with V.VideoWriter(tmp_path / f"{name}.mp4", width=wd, height=ht, fps=24, encoder="cpu",
                           preset="ultrafast", crf=23) as w:
            w.write(base)
            w.write(base)
        info = V.probe(tmp_path / f"{name}.mp4")
        assert (info.width, info.height, info.n_frames, info.codec) == (wd, ht, 2, "h264")
        assert info.fps == 24.0


# ---------------------------------------------------------------------------------------------------------------
# 입력 검증·실패 시 파일 시스템
# ---------------------------------------------------------------------------------------------------------------
def test_wrong_frame_shape_aborts_cleanly(ffmpeg_bin, make_clip, tmp_path):
    out = tmp_path / "bad.mp4"
    with pytest.raises(ValueError):
        with V.VideoWriter(out, width=32, height=32, fps=24, encoder="cpu", preset="ultrafast") as w:
            w.write(make_clip(2, 32, 32))
            w.write(make_clip(1, 32, 34)[0])      # 크기 불일치
    assert not out.exists()
    assert os.listdir(tmp_path) == []             # 임시 파트 파일도 남기지 않는다


def test_exception_in_body_keeps_previous_file(ffmpeg_bin, make_clip, tmp_path):
    out = tmp_path / "keep.mp4"
    out.write_bytes(b"OLD-CONTENT")
    with pytest.raises(RuntimeError, match="boom"):
        with V.VideoWriter(out, width=32, height=32, fps=24, encoder="cpu", preset="ultrafast") as w:
            w.write(make_clip(2, 32, 32))
            raise RuntimeError("boom")
    assert out.read_bytes() == b"OLD-CONTENT"
    assert sorted(os.listdir(tmp_path)) == ["keep.mp4"]


def test_zero_frames_is_error(ffmpeg_bin, tmp_path):
    out = tmp_path / "z.mp4"
    w = V.VideoWriter(out, width=32, height=32, fps=24, encoder="cpu")
    with pytest.raises(ValueError):
        w.close()
    assert not out.exists() and os.listdir(tmp_path) == []


def test_bad_values_rejected_at_write(ffmpeg_bin, tmp_path):
    out = tmp_path / "n.mp4"
    with pytest.raises(ValueError):
        with V.VideoWriter(out, width=32, height=32, fps=24, encoder="cpu") as w:
            bad = np.full((32, 32, 3), 0.5, np.float32)
            bad[3, 3, 1] = np.nan
            w.write(bad)
    assert os.listdir(tmp_path) == []
    with pytest.raises(TypeError):
        with V.VideoWriter(out, width=32, height=32, fps=24, encoder="cpu") as w:
            w.write(np.zeros((32, 32, 3), np.int32))
    assert os.listdir(tmp_path) == []


def test_overwrite_flag(ffmpeg_bin, make_clip, tmp_path):
    out = tmp_path / "o.mp4"
    out.write_bytes(b"OLD")
    with pytest.raises(FileExistsError):
        V.encode_frames(make_clip(2, 32, 32), out, fps=24, encoder="cpu", overwrite=False)
    assert out.read_bytes() == b"OLD"
    V.encode_frames(make_clip(2, 32, 32), out, fps=24, encoder="cpu", preset="ultrafast")   # 기본 overwrite=True
    assert V.probe(out).n_frames == 2


def test_writer_constructor_validation(ffmpeg_bin, tmp_path):
    ok = dict(width=32, height=32, fps=24, encoder="cpu")
    with pytest.raises(ValueError):
        V.VideoWriter(tmp_path / "a.mp4", **{**ok, "width": 33})        # 홀수
    with pytest.raises(ValueError):
        V.VideoWriter(tmp_path / "a.gif", **ok)                          # 허용 확장자: .mp4 .mov .mkv
    with pytest.raises(ValueError):
        V.VideoWriter(tmp_path / "a.mp4", **{**ok, "fps": 0})
    with pytest.raises(ValueError):
        V.VideoWriter(tmp_path / "a.mp4", **{**ok, "crf": 99})
    assert os.listdir(tmp_path) == []
    for ext in (".mov", ".mkv"):
        with V.VideoWriter(tmp_path / f"a{ext}", **{**ok, "preset": "ultrafast"}) as w:
            w.write(np.zeros((32, 32, 3), np.uint8))
        assert V.probe(tmp_path / f"a{ext}").n_frames == 1


def test_encode_frames_fps_rules(ffmpeg_bin, make_clip, tmp_path):
    rgb = make_clip(2, 32, 32)
    with pytest.raises(ValueError):
        V.encode_frames(rgb, tmp_path / "x.mp4", encoder="cpu")                       # ndarray 는 fps 필수
    fr = Frames(rgb=rgb, fps=24.0)
    with pytest.raises(ValueError):
        V.encode_frames(fr, tmp_path / "x.mp4", fps=30, encoder="cpu")                # Frames.fps 와 충돌
    V.encode_frames(fr, tmp_path / "x.mp4", fps=24, encoder="cpu", preset="ultrafast")  # 같은 값이면 허용
    with pytest.raises(ValueError):
        V.encode_frames(rgb[0], tmp_path / "y.mp4", fps=24, encoder="cpu")            # (H,W,3) 은 encode_frames 불가
    assert V.probe(tmp_path / "x.mp4").n_frames == 2


def test_missing_binaries(make_clip, tmp_path):
    with pytest.raises(V.VideoError):
        V.encode_frames(make_clip(2, 32, 32), tmp_path / "m.mp4", fps=24, encoder="cpu",
                        ffmpeg="/nonexistent/dir/ffmpeg")
    assert os.listdir(tmp_path) == []
    some = tmp_path / "exists.mp4"
    some.write_bytes(b"x")
    with pytest.raises(V.VideoError):
        V.probe(some, ffprobe="/nonexistent/dir/ffprobe")
    assert V.find_ffmpeg("definitely-not-a-binary-xyz") is None


def test_probe_and_decode_errors(ffmpeg_bin, tmp_path):
    with pytest.raises(FileNotFoundError):
        V.probe(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all" * 10)
    with pytest.raises(V.VideoError):
        V.probe(junk)
    with pytest.raises(V.VideoError):
        V.decode_frames(junk)


# ---------------------------------------------------------------------------------------------------------------
# 타임라인 이음: concat_frames
# ---------------------------------------------------------------------------------------------------------------
def test_concat_frames_basic(make_clip):
    depth_a = np.arange(3 * 8 * 12, dtype=np.float32).reshape(3, 8, 12)
    a = Frames(rgb=make_clip(3, 8, 12), fps=24.0, aov={"depth": depth_a}, shot_hash="S1", world_hash="W")
    b = Frames(rgb=make_clip(5, 8, 12)[::-1].copy(), fps=24.0, aov={"depth": np.ones((5, 8, 12), np.float32)},
               shot_hash="S2", world_hash="W")
    c = V.concat_frames([a, b])
    assert isinstance(c, Frames)
    assert c.rgb.shape == (8, 8, 12, 3) and c.fps == 24.0
    np.testing.assert_array_equal(c.rgb[:3], a.rgb)
    np.testing.assert_array_equal(c.rgb[3:], b.rgb)
    assert set(c.aov) == {"depth"} and c.aov["depth"].shape == (8, 8, 12)
    np.testing.assert_array_equal(c.aov["depth"][:3], a.aov["depth"])
    assert c.shot_hash == "" and c.world_hash == "W"       # 같은 값만 유지, 다르면 ""
    assert V.concat_frames([a]).shot_hash == "S1"
    np.testing.assert_array_equal(V.concat_frames([a]).rgb, a.rgb)
    np.testing.assert_array_equal(a.rgb, make_clip(3, 8, 12))   # 입력 불변


def test_concat_frames_aov_intersection_and_errors(make_clip):
    a = Frames(rgb=make_clip(2, 8, 12), fps=24.0, aov={"depth": np.zeros((2, 8, 12), np.float32),
                                                        "id": np.zeros((2, 8, 12), np.int32)})
    b = Frames(rgb=make_clip(2, 8, 12), fps=24.0, aov={"depth": np.ones((2, 8, 12), np.float32)})
    c = V.concat_frames([a, b])
    assert set(c.aov) == {"depth"}                          # 모든 클립에 있는 키만
    assert V.concat_frames([a, Frames(rgb=make_clip(1, 8, 12), fps=24.0)]).aov == {}
    with pytest.raises(ValueError):
        V.concat_frames([])
    with pytest.raises(ValueError):
        V.concat_frames([a, Frames(rgb=make_clip(2, 8, 14), fps=24.0)])      # 크기 다름
    with pytest.raises(ValueError):
        V.concat_frames([a, Frames(rgb=make_clip(2, 8, 12), fps=30.0)])      # fps 다름


def test_concat_then_encode_keeps_frame_count(ffmpeg_bin, make_clip, tmp_path):
    a = Frames(rgb=make_clip(8, 32, 48), fps=24.0)
    b = Frames(rgb=make_clip(16, 32, 48), fps=24.0)
    res = V.encode_frames(V.concat_frames([a, b]), tmp_path / "c.mp4", encoder="cpu", preset="ultrafast")
    assert res.n_frames == 24
    assert V.probe(tmp_path / "c.mp4").n_frames == 24
    assert abs(V.probe(tmp_path / "c.mp4").duration_s - 1.0) < 1e-3
    assert V.fps_rational(24) == Fraction(24, 1)


# ---------------------------------------------------------------------------------------------------------------
# GPU (NVENC) — GPU 러너에서만
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("name", ["1080p", "4k"])
def test_nvenc_h264(ffmpeg_bin, make_clip, tmp_path, name):
    if not V.probe_encoder("h264_nvenc"):
        pytest.skip("이 러너의 ffmpeg/GPU 는 h264_nvenc 를 쓸 수 없다")
    wd, ht = V.resolution(name)
    base = np.zeros((ht, wd, 3), np.uint8)
    base[..., 0] = np.linspace(0, 255, wd, dtype=np.uint8)[None, :]
    base[..., 2] = np.linspace(0, 255, ht, dtype=np.uint8)[:, None]
    with V.VideoWriter(tmp_path / "n.mp4", width=wd, height=ht, fps=24, encoder="gpu", crf=20) as w:
        for _ in range(24):
            w.write(base)
    assert w.result.encoder == "h264_nvenc"
    info = V.probe(tmp_path / "n.mp4")
    assert (info.width, info.height, info.n_frames, info.codec, info.fps) == (wd, ht, 24, "h264", 24.0)
    assert info.color_space == "bt709"
    dec = V.decode_frames(tmp_path / "n.mp4")
    assert _psnr(dec[0], base) >= 30.0
