"""EVL-04 테스트 공용 도우미. 평가 모듈은 파일 경로로 적재한다(eval/ 는 패키지가 아니다)."""
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "eval" / "first_frame.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # buc 를 어디서 돌려도 찾게 한다


@lru_cache(maxsize=1)
def ff():
    """eval/first_frame.py 를 적재해 돌려준다(dataclass 가 모듈을 찾도록 sys.modules 에 등록)."""
    spec = importlib.util.spec_from_file_location("first_frame_evl04", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------- 합성 영상
def smooth_noise(h, w, sigma, seed):
    r = np.random.default_rng(seed)
    n = ndimage.gaussian_filter(r.standard_normal((h, w)), sigma, mode="wrap")
    n = (n - n.mean()) / n.std()
    return n


def texture(h, w, seed=0, sigma=2.5, amp=0.18, base=0.45):
    """색이 있는 매끈한 무작위 질감, 표시값 [0,1] float32 (H,W,3)."""
    ch = [base + amp * smooth_noise(h, w, sigma, seed * 7 + c) for c in range(3)]
    return np.clip(np.stack(ch, -1), 0.02, 0.98).astype(np.float32)


def white_noise_gray(h, w, mean, std, seed):
    r = np.random.default_rng(seed)
    g = mean + std * r.standard_normal((h, w))
    return np.repeat(np.clip(g, 0, 1)[..., None], 3, axis=2).astype(np.float32)


def shift_img(img, dy, dx):
    """out[p] = img[p - (dy,dx)]  =>  img(p) = out(p + (dx,dy)),  즉 I -> out 의 흐름은 (+dx, +dy)."""
    return np.stack([ndimage.shift(img[..., c].astype(np.float64), (dy, dx), order=3, mode="reflect")
                     for c in range(3)], -1).astype(np.float32)


def noisy(img, std, seed):
    r = np.random.default_rng(seed)
    return np.clip(img + std * r.standard_normal(img.shape), 0, 1).astype(np.float32)


class FakeClient:
    """LPIPS 서버 대역: 평균 절대 차이를 거리로 쓰고, 받은 배열을 기록한다."""
    backend = "fake"
    is_perceptual = False

    def __init__(self):
        self.calls = []

    def distance(self, a, b):
        self.calls.append((np.array(a, copy=True), np.array(b, copy=True)))
        return float(np.mean(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

    def close(self):
        pass


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
