"""EVL-02 공용 설정: 레포 루트를 sys.path 에 넣고(어디서 돌려도 buc 를 찾는다) ffmpeg 가 없으면 실패시킨다."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session", autouse=True)
def _ffmpeg_present():
    from buc.render import video as V

    if V.find_ffmpeg("ffmpeg") is None or V.find_ffmpeg("ffprobe") is None:
        pytest.fail("ffmpeg/ffprobe 를 PATH 에서 찾을 수 없다 (EVL-02 테스트는 CPU 컨테이너의 ffmpeg 를 전제한다)")
