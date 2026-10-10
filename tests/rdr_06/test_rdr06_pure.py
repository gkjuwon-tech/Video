"""RDR-06 순수 함수 계약: 해상도·fps·타임라인·uint8 변환·인코더 선택·ffmpeg 명령행. ffmpeg 를 실행하지 않는다."""
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from buc.render import video as V

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _opt(cmd, flag, start=0):
    """cmd[start:] 에서 flag 다음 토큰."""
    i = cmd.index(flag, start)
    return cmd[i + 1]


# ---------------------------------------------------------------------------------------------------------------
# 임포트·상수
# ---------------------------------------------------------------------------------------------------------------
def test_import_is_cpu_light():
    code = "import sys, buc.render.video; assert 'torch' not in sys.modules, 'torch imported'"
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT}, timeout=30)
    assert p.returncode == 0, p.stderr


def test_error_hierarchy():
    assert issubclass(V.VideoError, RuntimeError)
    assert not issubclass(V.VideoError, ValueError)


def test_resolution_presets():
    assert V.resolution("1080p") == (1920, 1080)
    assert V.resolution("4K") == (3840, 2160) and V.resolution("4k") == (3840, 2160)
    assert V.resolution(" 1080P ") == (1920, 1080)
    for bad in ("720p", "8k", "", "1920x1080"):
        with pytest.raises(ValueError):
            V.resolution(bad)


def test_fps_rational():
    assert V.fps_rational(24) == Fraction(24, 1)
    assert V.fps_rational(24.0) == Fraction(24, 1)
    assert V.fps_rational(24000 / 1001) == Fraction(24000, 1001)
    assert V.fps_rational(30000 / 1001) == Fraction(30000, 1001)
    assert V.fps_rational(60) == Fraction(60, 1)
    assert isinstance(V.fps_rational(25), Fraction)
    for bad in (0, -24, float("nan"), float("inf"), 1000.5):
        with pytest.raises(ValueError):
            V.fps_rational(bad)


# ---------------------------------------------------------------------------------------------------------------
# 타임라인
# ---------------------------------------------------------------------------------------------------------------
def test_frame_count():
    assert V.frame_count(10.0, 24) == 240
    assert V.frame_count(5 / 24, 24) == 5
    assert V.frame_count(1.0, 24000 / 1001) == 24
    assert V.frame_count(0.0, 24) == 0
    assert isinstance(V.frame_count(2.0, 24), int)
    with pytest.raises(ValueError):
        V.frame_count(-1.0, 24)
    with pytest.raises(ValueError):
        V.frame_count(1.0, 0)


def test_frame_times():
    t = V.frame_times(5, 24)
    assert t.dtype == np.float64 and t.shape == (5,)
    np.testing.assert_allclose(t, np.arange(5) / 24, rtol=0, atol=1e-12)
    assert t[0] == 0.0
    np.testing.assert_allclose(np.diff(V.frame_times(240, 24)), 1 / 24, rtol=0, atol=1e-12)
    assert V.frame_times(0, 24).shape == (0,)
    with pytest.raises(ValueError):
        V.frame_times(-1, 24)


# ---------------------------------------------------------------------------------------------------------------
# float [0,1] -> uint8
# ---------------------------------------------------------------------------------------------------------------
def test_to_uint8_all_levels_roundtrip():
    k = np.arange(256)
    x = (k / 255.0).astype(np.float32).reshape(1, 1, 256, 1).repeat(3, axis=3)
    u = V.to_uint8(x)
    assert u.dtype == np.uint8 and u.shape == x.shape and u.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(u[0, 0, :, 0], k.astype(np.uint8))


def test_to_uint8_rounds_to_nearest():
    k = np.arange(0, 255)
    lo = ((k + 0.4) / 255).astype(np.float32)
    hi = ((k + 0.6) / 255).astype(np.float32)
    np.testing.assert_array_equal(V.to_uint8(lo), k.astype(np.uint8))
    np.testing.assert_array_equal(V.to_uint8(hi), (k + 1).astype(np.uint8))
    assert V.to_uint8(np.float64(0.5) * np.ones((1, 1, 3))).tolist() == [[[128, 128, 128]]]  # 127.5 -> 짝수로 128


def test_to_uint8_range_tolerance():
    assert V.to_uint8(np.array([1 + 5e-5, -5e-5], np.float32)).tolist() == [255, 0]
    for bad in (1 + 2e-4, -2e-4, 1.5, -0.5, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            V.to_uint8(np.array([0.5, bad], np.float32))


def test_to_uint8_dtypes():
    u = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    np.testing.assert_array_equal(V.to_uint8(u), u)
    assert V.to_uint8(np.zeros((2, 2, 3), np.float64)).dtype == np.uint8
    for dt in (np.int32, np.int64, np.float16, bool):
        with pytest.raises(TypeError):
            V.to_uint8(np.zeros((2, 2, 3), dt))
    ro = np.full((2, 2, 3), 0.25, np.float32)
    ro.flags.writeable = False
    assert V.to_uint8(ro)[0, 0, 0] == 64  # 63.75 -> 64; 읽기 전용 입력 허용
    assert ro.flags.writeable is False and float(ro[0, 0, 0]) == 0.25


# ---------------------------------------------------------------------------------------------------------------
# 인코더 선택 (available 를 주면 프로브 없이 순수 함수)
# ---------------------------------------------------------------------------------------------------------------
def test_select_encoder_h264():
    both = {"libx264", "h264_nvenc"}
    assert V.select_encoder("h264", "auto", available=both) == "h264_nvenc"
    assert V.select_encoder("h264", "gpu", available=both) == "h264_nvenc"
    assert V.select_encoder("h264", "cpu", available=both) == "libx264"
    assert V.select_encoder("h264", "auto", available={"libx264"}) == "libx264"
    assert V.select_encoder("h264", "cpu", available={"libx264"}) == "libx264"
    with pytest.raises(V.VideoError):
        V.select_encoder("h264", "gpu", available={"libx264"})  # 조용한 CPU 폴백 금지
    with pytest.raises(V.VideoError):
        V.select_encoder("h264", "cpu", available={"h264_nvenc"})
    with pytest.raises(V.VideoError):
        V.select_encoder("h264", "auto", available=set())
    assert V.select_encoder("h264", available=frozenset({"libx264", "other"})) == "libx264"  # 기본 prefer="auto"


def test_select_encoder_hevc_and_validation():
    both = ["libx265", "hevc_nvenc", "libx264"]
    assert V.select_encoder("hevc", "auto", available=both) == "hevc_nvenc"
    assert V.select_encoder("hevc", "cpu", available=both) == "libx265"
    with pytest.raises(V.VideoError):
        V.select_encoder("hevc", "auto", available=["libx264"])
    with pytest.raises(ValueError):
        V.select_encoder("vp9", "auto", available=both)
    with pytest.raises(ValueError):
        V.select_encoder("h264", "turbo", available=both)


def test_probe_encoder_and_real_selection(ffmpeg_bin):
    assert V.probe_encoder("libx264") is True
    assert V.probe_encoder("no_such_encoder_xyz") is False
    assert V.probe_encoder("libx264", ffmpeg="/nonexistent/dir/ffmpeg") is False  # 예외 없이 False
    enc = V.select_encoder("h264", "auto")
    assert enc in ("libx264", "h264_nvenc") and V.probe_encoder(enc)
    assert V.select_encoder("h264", "cpu") == "libx264"


# ---------------------------------------------------------------------------------------------------------------
# ffmpeg 명령행 (순수, 셸 문자열 금지)
# ---------------------------------------------------------------------------------------------------------------
def test_build_cmd_libx264_layout():
    cmd = V.build_ffmpeg_cmd("/tmp/out.mp4", width=1920, height=1080, fps=Fraction(24, 1), encoder="libx264",
                             crf=18, preset="medium", ffmpeg="ffmpeg")
    assert isinstance(cmd, list) and all(isinstance(c, str) for c in cmd)
    assert cmd[0] == "ffmpeg" and cmd[-1] == "/tmp/out.mp4"
    i_in = cmd.index("-i")
    assert cmd[i_in + 1] == "-"                       # stdin 에서 원시 프레임
    inp, out = cmd[:i_in], cmd[i_in + 2:-1]
    assert _opt(inp, "-f") == "rawvideo" and _opt(inp, "-pix_fmt") == "rgb24"
    assert _opt(inp, "-s") == "1920x1080" and _opt(inp, "-r") == "24/1"
    assert _opt(out, "-c:v") == "libx264" and _opt(out, "-crf") == "18" and _opt(out, "-preset") == "medium"
    assert _opt(out, "-pix_fmt") == "yuv420p"
    assert "-an" in out
    assert "-y" in cmd
    # 색: BT.709 행렬 + 제한 범위(tv), 컨테이너 태그
    vf = _opt(out, "-vf")
    assert "out_color_matrix=bt709" in vf and "out_range=tv" in vf
    assert _opt(out, "-colorspace") == "bt709" and _opt(out, "-color_primaries") == "bt709"
    assert _opt(out, "-color_trc") == "bt709" and _opt(out, "-color_range") == "tv"
    # 결정론·메타데이터
    assert _opt(out, "-map_metadata") == "-1"
    assert "+bitexact" in _opt(out, "-fflags")
    assert _opt(out, "-movflags") == "+faststart"


def test_build_cmd_fractional_fps_and_4k():
    cmd = V.build_ffmpeg_cmd("o.mp4", width=3840, height=2160, fps=Fraction(24000, 1001), encoder="libx264")
    inp = cmd[:cmd.index("-i")]
    assert _opt(inp, "-s") == "3840x2160" and _opt(inp, "-r") == "24000/1001"
    assert _opt(cmd, "-crf") == "18" and _opt(cmd, "-preset") == "medium"  # 기본값


def test_build_cmd_nvenc():
    cmd = V.build_ffmpeg_cmd("o.mp4", width=1920, height=1080, fps=Fraction(24, 1), encoder="h264_nvenc",
                             crf=20, preset="slow")
    out = cmd[cmd.index("-i") + 2:]
    assert _opt(out, "-c:v") == "h264_nvenc"
    assert _opt(out, "-cq") == "20"                    # crf 값이 -cq 로 간다
    assert _opt(out, "-rc") == "vbr" and _opt(out, "-b:v") == "0"
    p = _opt(out, "-preset")
    assert p in {f"p{i}" for i in range(1, 8)}         # x264 이름이 아니라 NVENC p1..p7
    assert "-crf" not in out
    assert _opt(out, "-pix_fmt") == "yuv420p"


def test_build_cmd_hevc_tag():
    cmd = V.build_ffmpeg_cmd("o.mp4", width=64, height=64, fps=Fraction(24, 1), encoder="libx265")
    assert _opt(cmd, "-c:v") == "libx265" and _opt(cmd, "-tag:v") == "hvc1"
    cmd = V.build_ffmpeg_cmd("o.mp4", width=64, height=64, fps=Fraction(24, 1), encoder="libx264")
    assert "-tag:v" not in cmd


@pytest.mark.parametrize("kw", [
    dict(width=63, height=64),          # 홀수 크기: yuv420p 불가
    dict(width=64, height=65),
    dict(width=0, height=64),
    dict(width=-2, height=64),
    dict(width=64, height=64, crf=-1),
    dict(width=64, height=64, crf=52),
    dict(width=64, height=64, preset="warp9"),
    dict(width=64, height=64, encoder="mpeg9"),
    dict(width=64, height=64, fps=Fraction(0, 1)),
])
def test_build_cmd_rejects_bad_params(kw):
    args = dict(width=64, height=64, fps=Fraction(24, 1), encoder="libx264")
    args.update(kw)
    with pytest.raises(ValueError):
        V.build_ffmpeg_cmd("o.mp4", **args)
