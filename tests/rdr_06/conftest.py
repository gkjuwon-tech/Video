"""RDR-06 공용 픽스처: ffmpeg 존재 확인과 작은 결정론 클립 생성기."""
import numpy as np
import pytest

from buc.render import video as V


@pytest.fixture(scope="session")
def ffmpeg_bin():
    """ffmpeg·ffprobe 가 없으면 테스트를 건너뛰지 않고 실패시킨다(건너뛰면 성공처럼 보인다)."""
    ff, fp = V.find_ffmpeg("ffmpeg"), V.find_ffmpeg("ffprobe")
    if ff is None or fp is None:
        pytest.fail("ffmpeg/ffprobe 를 PATH 에서 찾을 수 없다 (RDR-06 테스트는 CPU 컨테이너의 ffmpeg 를 전제한다)")
    return ff


@pytest.fixture
def make_clip():
    """make_clip(n, h, w) -> float32 (n,h,w,3) [0,1]. 부드러운 그라디언트가 프레임마다 움직인다(결정론)."""

    def _make(n=24, h=64, w=96):
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, None, :]
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None]
        t = (np.arange(n, dtype=np.float32) / max(n - 1, 1))[:, None, None]
        r = np.clip(0.15 + 0.6 * x + 0.2 * t, 0, 1) * np.ones((n, h, w), np.float32)
        g = np.clip(0.1 + 0.7 * y, 0, 1) * np.ones((n, h, w), np.float32)
        b = np.clip(0.8 - 0.5 * t + 0.1 * x, 0, 1) * np.ones((n, h, w), np.float32)
        out = np.stack([r, g, b], axis=-1).astype(np.float32)
        assert out.min() >= 0.0 and out.max() <= 1.0
        return out

    return _make
