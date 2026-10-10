"""RDR-06 영상 인코더·타임라인 v0: Frames -> H.264 mp4 (ffmpeg, libx264 / NVENC, 24 fps, 1080p/4K).

정형 래퍼다. numpy·표준 라이브러리·시스템 ffmpeg/ffprobe 만 쓴다(torch 금지, CPU import 가능).
원자성: 인코딩은 같은 디렉터리의 임시 파트 파일(<이름>.part<확장자>)에 쓰고, 성공 시에만 os.replace 한다.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from buc.core.contracts import Frames

__all__ = [
    "VideoError", "resolution", "fps_rational", "frame_count", "frame_times", "to_uint8", "find_ffmpeg",
    "probe_encoder", "select_encoder", "build_ffmpeg_cmd", "EncodeResult", "VideoInfo", "VideoWriter",
    "encode_frames", "probe", "decode_frames", "concat_frames",
]


class VideoError(RuntimeError):
    """ffmpeg 없음·인코더 없음·프로세스 실패·닫힌 writer 사용."""


_RESOLUTIONS = {"1080p": (1920, 1080), "4k": (3840, 2160)}
_X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
_NVENC_PRESETS = dict(zip(_X264_PRESETS, ("p1", "p2", "p3", "p3", "p4", "p5", "p6", "p7", "p7")))
_CODECS = ("h264", "hevc")
_PREFERS = ("auto", "cpu", "gpu")
_CANDIDATES = {"h264": {"gpu": "h264_nvenc", "cpu": "libx264"}, "hevc": {"gpu": "hevc_nvenc", "cpu": "libx265"}}
_ENCODER_CODEC = {"libx264": "h264", "h264_nvenc": "h264", "libx265": "hevc", "hevc_nvenc": "hevc"}
_EXTS = (".mp4", ".mov", ".mkv")
_probe_cache: dict = {}


# ---------------------------------------------------------------------------------------------------------------
# 순수 함수
# ---------------------------------------------------------------------------------------------------------------
def resolution(name: str) -> Tuple[int, int]:
    """'1080p' -> (1920, 1080), '4k' -> (3840, 2160). 대소문자·앞뒤 공백 무시."""
    if not isinstance(name, str):
        raise ValueError(f"해상도 이름은 문자열이어야 한다: {name!r}")
    key = name.strip().lower()
    if key not in _RESOLUTIONS:
        raise ValueError(f"모르는 해상도: {name!r} (지원: 1080p, 4k)")
    return _RESOLUTIONS[key]


def fps_rational(fps) -> Fraction:
    """fps 를 유리수로. 유한하고 0 < fps <= 1000 이어야 한다. 24000/1001 은 그대로 보존된다."""
    try:
        f = float(fps)
    except (TypeError, ValueError) as e:
        raise ValueError(f"fps 가 숫자가 아니다: {fps!r}") from e
    if not (math.isfinite(f) and 0.0 < f <= 1000.0):
        raise ValueError(f"fps 는 유한하고 0 < fps <= 1000 이어야 한다: {fps!r}")
    return Fraction(f).limit_denominator(1001)


def _fraction_fps(fps) -> Fraction:
    """Fraction 은 정확한 값을 그대로 쓰고(범위만 검사), 그 밖의 숫자는 fps_rational 로 바꾼다."""
    if isinstance(fps, Fraction):
        if not (0 < fps <= 1000):
            raise ValueError(f"fps 는 0 < fps <= 1000 이어야 한다: {fps}")
        return fps
    return fps_rational(fps)


def frame_count(duration_s: float, fps) -> int:
    """int(round(duration_s * fps))."""
    fr = fps_rational(fps)
    d = float(duration_s)
    if not math.isfinite(d) or d < 0:
        raise ValueError(f"duration_s 는 유한하고 0 이상이어야 한다: {duration_s!r}")
    return int(round(d * float(fr)))


def frame_times(n: int, fps) -> np.ndarray:
    """프레임 i 의 PTS = i / fps (float64, 길이 n)."""
    fr = fps_rational(fps)
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError(f"n 은 0 이상의 정수여야 한다: {n!r}")
    return np.arange(int(n), dtype=np.float64) / float(fr)


def to_uint8(x) -> np.ndarray:
    """uint8 은 그대로. float32/float64 [0,1](허용 오차 1e-4) 은 rint(clip(x,0,1)*255) (반올림 짝수)."""
    a = np.asarray(x)
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if a.dtype in (np.float32, np.float64):
        if a.size:
            if not np.all(np.isfinite(a)):
                raise ValueError("to_uint8: NaN/inf 입력")
            if float(a.min()) < -1e-4 or float(a.max()) > 1.0 + 1e-4:
                raise ValueError("to_uint8: [0,1] 범위를 벗어난 값")
        y = np.rint(np.clip(a.astype(np.float64), 0.0, 1.0) * 255.0)
        return np.ascontiguousarray(y.astype(np.uint8))
    raise TypeError(f"to_uint8: 지원하지 않는 dtype {a.dtype} (uint8 또는 float32/float64)")


def find_ffmpeg(name: str = "ffmpeg") -> Optional[str]:
    return shutil.which(name)


def probe_encoder(encoder: str, *, ffmpeg: Optional[str] = None) -> bool:
    """256x256 한 프레임을 실제로 인코딩해 본다. 목록에 있어도 쓸 수 없으면 False. 예외 없이 bool, 프로세스 내 캐시."""
    exe = ffmpeg if ffmpeg is not None else find_ffmpeg("ffmpeg")
    if exe is None:
        return False
    key = (str(exe), str(encoder))
    if key in _probe_cache:
        return _probe_cache[key]
    ok = False
    try:
        p = subprocess.run(
            [str(exe), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1:r=24",
             "-frames:v", "1", "-pix_fmt", "yuv420p", "-c:v", str(encoder), "-f", "null", "-"],
            capture_output=True, timeout=120)
        ok = p.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError):
        ok = False
    _probe_cache[key] = ok
    return ok


def select_encoder(codec: str = "h264", prefer: str = "auto", *, available=None, ffmpeg: Optional[str] = None) -> str:
    """auto=GPU 후 CPU, cpu=CPU만, gpu=GPU만. available 이 주어지면 프로브 없이 그 집합만 쓴다."""
    if codec not in _CODECS:
        raise ValueError(f"codec 은 h264|hevc 이어야 한다: {codec!r}")
    if prefer not in _PREFERS:
        raise ValueError(f"prefer 는 auto|cpu|gpu 이어야 한다: {prefer!r}")
    order = {"auto": ("gpu", "cpu"), "cpu": ("cpu",), "gpu": ("gpu",)}[prefer]
    names = [_CANDIDATES[codec][k] for k in order]
    avail = None if available is None else set(available)
    for name in names:
        ok = (name in avail) if avail is not None else probe_encoder(name, ffmpeg=ffmpeg)
        if ok:
            return name
    raise VideoError(f"쓸 수 있는 {codec} 인코더가 없다 (prefer={prefer}, 후보={names})")


def _check_params(width, height, crf, preset) -> None:
    for nm, v in (("width", width), ("height", height)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{nm} 은 정수여야 한다: {v!r}")
        if v < 2 or v % 2:
            raise ValueError(f"{nm}={v}: 2 이상의 짝수여야 한다(yuv420p)")
    if isinstance(crf, bool) or not isinstance(crf, (int, np.integer)) or not (0 <= crf <= 51):
        raise ValueError(f"crf 는 0..51 정수여야 한다: {crf!r}")
    if preset not in _X264_PRESETS:
        raise ValueError(f"모르는 preset: {preset!r} (지원: {', '.join(_X264_PRESETS)})")


def build_ffmpeg_cmd(out_path, *, width: int, height: int, fps, encoder: str, crf: int = 18,
                     preset: str = "medium", ffmpeg: str = "ffmpeg") -> list:
    """순수 명령행(셸 문자열 아님). 모든 원소는 str."""
    if encoder not in _ENCODER_CODEC:
        raise ValueError(f"모르는 인코더: {encoder!r}")
    _check_params(width, height, crf, preset)
    fr = _fraction_fps(fps)
    w, h = int(width), int(height)
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fr.numerator}/{fr.denominator}", "-i", "-",
           "-an", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv",
           "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-c:v", encoder]
    if encoder in ("libx264", "libx265"):
        cmd += ["-crf", str(int(crf)), "-preset", preset]
    else:
        cmd += ["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0", "-preset", _NVENC_PRESETS[preset]]
    if encoder in ("libx265", "hevc_nvenc"):
        cmd += ["-tag:v", "hvc1"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    return cmd


# ---------------------------------------------------------------------------------------------------------------
# 결과 타입
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EncodeResult:
    path: Path
    n_frames: int
    width: int
    height: int
    fps: float
    codec: str
    encoder: str
    bytes: int
    cmd: tuple


@dataclass(frozen=True)
class VideoInfo:
    codec: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    pix_fmt: str
    color_space: str
    color_range: str


# ---------------------------------------------------------------------------------------------------------------
# 스트리밍 writer
# ---------------------------------------------------------------------------------------------------------------
class VideoWriter:
    """프레임을 ffmpeg stdin 으로 즉시 흘려보낸다. 첫 write 에서 ffmpeg 를 띄운다(생성자는 아무 파일도 만들지 않는다)."""

    def __init__(self, path, *, width: int, height: int, fps, codec: str = "h264", encoder: str = "auto",
                 crf: int = 18, preset: str = "medium", overwrite: bool = True, ffmpeg: Optional[str] = None):
        p = Path(path)
        if p.suffix.lower() not in _EXTS:
            raise ValueError(f"확장자는 {_EXTS} 중 하나여야 한다: {p.suffix!r}")
        if codec not in _CODECS:
            raise ValueError(f"codec 은 h264|hevc 이어야 한다: {codec!r}")
        fr = _fraction_fps(fps)
        _check_params(width, height, crf, preset)
        if not overwrite and p.exists():
            raise FileExistsError(f"이미 있는 파일(overwrite=False): {p}")
        exe = ffmpeg if ffmpeg is not None else find_ffmpeg("ffmpeg")
        if exe is None:
            raise VideoError("ffmpeg 를 찾을 수 없다")
        if encoder in _PREFERS:
            enc = select_encoder(codec, encoder, ffmpeg=exe)
        elif encoder in _ENCODER_CODEC:
            if _ENCODER_CODEC[encoder] != codec:
                raise ValueError(f"인코더 {encoder} 는 codec={codec} 과 맞지 않는다")
            enc = encoder
        else:
            raise ValueError(f"encoder 는 auto|cpu|gpu 또는 인코더 이름이어야 한다: {encoder!r}")
        self._path = p
        self._part = p.with_name(f"{p.stem}.part{p.suffix}")
        self._W, self._H = int(width), int(height)
        self._fr = fr
        self._codec = codec
        self._encoder = enc
        self._overwrite = bool(overwrite)
        self._cmd = build_ffmpeg_cmd(self._part, width=self._W, height=self._H, fps=fr, encoder=enc,
                                     crf=crf, preset=preset, ffmpeg=str(exe))
        self._proc: Optional[subprocess.Popen] = None
        self._stderr = None
        self._n = 0
        self._closed = False
        self._result: Optional[EncodeResult] = None

    # 속성
    @property
    def n_frames(self) -> int:
        return self._n

    @property
    def result(self) -> Optional[EncodeResult]:
        return self._result

    # 내부
    def _start(self) -> None:
        try:
            self._part.parent.mkdir(parents=True, exist_ok=True)
            self._stderr = tempfile.TemporaryFile()
            self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                          stderr=self._stderr)
        except OSError as e:
            self._abort()
            raise VideoError(f"ffmpeg 실행 실패: {e}") from e

    def _stderr_tail(self) -> str:
        try:
            if self._stderr is None:
                return ""
            self._stderr.seek(0)
            return self._stderr.read()[-2000:].decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _remove_part(self) -> None:
        try:
            if self._part.exists():
                self._part.unlink()
        except OSError:
            pass

    def _abort(self) -> None:
        self._closed = True
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                if proc.stdin:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=60)
            except (OSError, subprocess.SubprocessError):
                pass
        if self._stderr is not None:
            try:
                self._stderr.close()
            except OSError:
                pass
            self._stderr = None
        self._remove_part()

    # 공개
    def write(self, frames) -> None:
        """(H,W,3) 한 장 또는 (k,H,W,3) 묶음. uint8 또는 float [0,1]. 검증이 끝난 뒤에만 쓴다."""
        if self._closed:
            raise VideoError("닫힌 VideoWriter 에는 쓸 수 없다")
        a = np.asarray(frames)
        if a.ndim == 3:
            a = a[None]
        if a.ndim != 4 or a.shape[1:] != (self._H, self._W, 3):
            raise ValueError(f"프레임 shape {a.shape} 가 ({self._H},{self._W},3) 와 맞지 않는다")
        u = to_uint8(a)
        k = int(u.shape[0])
        if k == 0:
            return
        if self._proc is None:
            self._start()
        try:
            self._proc.stdin.write(u.reshape(-1).data)
        except (OSError, ValueError) as e:
            tail = self._stderr_tail()
            self._abort()
            raise VideoError(f"ffmpeg 파이프 쓰기 실패: {tail}") from e
        self._n += k

    def close(self) -> EncodeResult:
        """멱등. 프레임 0개면 ValueError(파일·파트 없음), ffmpeg 실패면 VideoError(파트 삭제)."""
        if self._result is not None:
            return self._result
        if self._closed:
            raise VideoError("이미 실패했거나 닫힌 VideoWriter 다")
        self._closed = True
        if self._n == 0:
            raise ValueError("프레임이 0개라 영상을 만들 수 없다")
        proc = self._proc
        self._proc = None
        try:
            try:
                proc.stdin.close()
            except OSError:
                pass
            rc = proc.wait()
            tail = self._stderr_tail()
        finally:
            if self._stderr is not None:
                try:
                    self._stderr.close()
                except OSError:
                    pass
                self._stderr = None
        if rc != 0:
            self._remove_part()
            raise VideoError(f"ffmpeg 실패(rc={rc}): {tail}")
        if not self._overwrite and self._path.exists():
            self._remove_part()
            raise FileExistsError(f"이미 있는 파일(overwrite=False): {self._path}")
        if not self._part.is_file() or self._part.stat().st_size == 0:
            self._remove_part()
            raise VideoError(f"ffmpeg 가 결과 파일을 만들지 않았다: {tail}")
        os.replace(self._part, self._path)
        self._result = EncodeResult(
            path=self._path, n_frames=self._n, width=self._W, height=self._H, fps=float(self._fr),
            codec=self._codec, encoder=self._encoder, bytes=int(self._path.stat().st_size), cmd=tuple(self._cmd))
        return self._result

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self._abort()
        return False


def encode_frames(frames, path, *, fps=None, codec: str = "h264", encoder: str = "auto", crf: int = 18,
                  preset: str = "medium", overwrite: bool = True, ffmpeg: Optional[str] = None) -> EncodeResult:
    """Frames 또는 (N,H,W,3) ndarray(fps 필수) 를 mp4 로. 내부는 VideoWriter, 프레임을 하나씩 흘린다."""
    if isinstance(frames, Frames):
        if fps is not None and fps_rational(fps) != fps_rational(frames.fps):
            raise ValueError(f"fps={fps} 가 Frames.fps={frames.fps} 와 다르다")
        rate, arr = frames.fps, frames.rgb
    else:
        if fps is None:
            raise ValueError("ndarray 입력에는 fps 가 필요하다")
        arr = np.asarray(frames)
        if arr.ndim != 4:
            raise ValueError(f"ndarray 는 (N,H,W,3) 이어야 한다: shape {arr.shape}")
        rate = fps
    n, h, w = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    with VideoWriter(path, width=w, height=h, fps=rate, codec=codec, encoder=encoder, crf=crf, preset=preset,
                     overwrite=overwrite, ffmpeg=ffmpeg) as writer:
        for i in range(n):
            writer.write(arr[i])
    return writer.result


# ---------------------------------------------------------------------------------------------------------------
# 읽기·검사
# ---------------------------------------------------------------------------------------------------------------
def _sibling_ffprobe(ffmpeg: Optional[str]) -> Optional[str]:
    if ffmpeg is None:
        return find_ffmpeg("ffprobe")
    cand = os.path.join(os.path.dirname(str(ffmpeg)), "ffprobe")
    return cand if os.path.isfile(cand) else find_ffmpeg("ffprobe")


def probe(path, *, ffprobe: Optional[str] = None) -> VideoInfo:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"파일이 없다: {p}")
    exe = ffprobe if ffprobe is not None else find_ffmpeg("ffprobe")
    if exe is None:
        raise VideoError("ffprobe 를 찾을 수 없다")
    cmd = [str(exe), "-v", "error", "-count_frames", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt,color_space,color_range",
           "-of", "json", str(p)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as e:
        raise VideoError(f"ffprobe 실행 실패: {e}") from e
    if r.returncode != 0:
        raise VideoError(f"ffprobe 실패: {r.stderr.strip()[-500:]}")
    try:
        streams = (json.loads(r.stdout or "{}").get("streams") or [])
    except (ValueError, AttributeError) as e:
        raise VideoError("ffprobe 출력을 해석할 수 없다") from e
    if not streams:
        raise VideoError(f"비디오 스트림이 없다: {p}")
    s = streams[0]
    try:
        num, den = str(s["r_frame_rate"]).split("/")
        fr = Fraction(int(num), int(den))
        if fr <= 0:
            raise ValueError("frame rate 0")
        n = int(s["nb_read_frames"])
        info = VideoInfo(
            codec=str(s["codec_name"]), width=int(s["width"]), height=int(s["height"]), fps=float(fr),
            n_frames=n, duration_s=float(Fraction(n) / fr), pix_fmt=str(s.get("pix_fmt", "")),
            color_space=str(s.get("color_space", "")), color_range=str(s.get("color_range", "")))
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
        raise VideoError(f"비디오 메타데이터를 읽을 수 없다: {p}") from e
    return info


def decode_frames(path, *, ffmpeg: Optional[str] = None) -> np.ndarray:
    """rgb24 로 디코드(태그된 색 행렬 사용). 반환 uint8 (N,H,W,3)."""
    exe = ffmpeg if ffmpeg is not None else find_ffmpeg("ffmpeg")
    if exe is None:
        raise VideoError("ffmpeg 를 찾을 수 없다")
    info = probe(path, ffprobe=_sibling_ffprobe(ffmpeg))
    cmd = [str(exe), "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as e:
        raise VideoError(f"ffmpeg 디코드 실행 실패: {e}") from e
    if r.returncode != 0:
        raise VideoError(f"ffmpeg 디코드 실패: {r.stderr.decode('utf-8', errors='replace')[-500:]}")
    fs = info.width * info.height * 3
    if fs <= 0 or len(r.stdout) % fs:
        raise VideoError("디코드 출력 크기가 프레임 크기와 맞지 않는다")
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, info.height, info.width, 3).copy()


def concat_frames(clips: Sequence[Frames]) -> Frames:
    """같은 (H,W)·fps 클립을 이어 붙인다. aov 는 모든 클립에 있는 키만, 해시는 전부 같을 때만 유지."""
    clips = list(clips)
    if not clips:
        raise ValueError("concat_frames: 빈 입력")
    for c in clips:
        if not isinstance(c, Frames):
            raise TypeError(f"concat_frames: Frames 가 아니다: {type(c).__name__}")
    first = clips[0]
    hw = first.rgb.shape[1:3]
    for c in clips[1:]:
        if c.rgb.shape[1:3] != hw:
            raise ValueError(f"concat_frames: 크기 불일치 {c.rgb.shape[1:3]} vs {hw}")
        if c.fps != first.fps:
            raise ValueError(f"concat_frames: fps 불일치 {c.fps} vs {first.fps}")
    rgb = np.concatenate([c.rgb for c in clips], axis=0)
    keys = [k for k in first.aov if all(k in c.aov for c in clips)]
    aov = {k: np.concatenate([c.aov[k] for c in clips], axis=0) for k in keys}
    shot = first.shot_hash if all(c.shot_hash == first.shot_hash for c in clips) else ""
    world = first.world_hash if all(c.world_hash == first.world_hash for c in clips) else ""
    return Frames(rgb=rgb, fps=first.fps, aov=aov, shot_hash=shot, world_hash=world)
