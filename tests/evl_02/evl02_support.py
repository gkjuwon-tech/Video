"""EVL-02 테스트 공용 도우미. 러너는 파일 경로로 적재한다(eval/ 는 패키지가 아니다).
렌더러는 대역(결정론 합성 Frames)이다 — 이 테스트는 러너의 장부·인코딩·시트·재개 로직을 검증한다."""
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "eval" / "daily_ab.py"
QA_TOOL = ROOT / "tools" / "inputs_qa.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # buc 를 어디서 돌려도 찾게 한다

from buc.core import contracts as C  # noqa: E402
from buc.render import video as V  # noqa: E402

W, H, N = 64, 48, 6  # 합성 클립 크기(짝수), 프레임 수


@lru_cache(maxsize=1)
def ab():
    """eval/daily_ab.py 를 적재해 돌려준다(dataclass 가 모듈을 찾도록 sys.modules 에 등록)."""
    spec = importlib.util.spec_from_file_location("daily_ab_evl02", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def qa():
    spec = importlib.util.spec_from_file_location("inputs_qa_evl02", QA_TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fake_sha(slot):
    return hashlib.sha256(f"evl02-test-{slot}".encode()).hexdigest()


def make_manifest(n_ready=4):
    """실제 매니페스트(24칸)에서 슬롯 1..n_ready 만 ready 로 만든 Manifest. 나머지는 planned."""
    m = qa().load_manifest()
    items = []
    for it in m.items:
        if it.slot <= n_ready and it.status != "ready":
            it = replace(it, status="ready", vault_path=f"tests/{it.id}.png", sha256=fake_sha(it.slot), size=(W, H),
                         crop_aspect="native", crop_box=(0, 0, W, H))
        items.append(it)
    return qa().Manifest(items=tuple(items), moves=m.moves, selection=dict(m.selection), vault_repo=m.vault_repo)


def tag_of(item_id):
    return int(item_id[2:])


def make_frames(item_id, offset=0.0, n=N, h=H, w=W, fps=24.0):
    """항목마다 다르고 시간에 따라 움직이는 부드러운 클립. 값은 [0.05, 0.85] 안(오프셋 0.1 이 잘리지 않는다)."""
    k = tag_of(item_id)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, None, :]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :, None]
    t = (np.arange(n, dtype=np.float32) / max(n - 1, 1))[:, None, None]
    ph = (k % 7) / 7.0
    r = 0.10 + 0.45 * x * (0.5 + 0.5 * ph) + 0.20 * t
    g = 0.10 + 0.50 * y * (1.0 - 0.4 * ph) + 0.10 * t
    b = 0.70 - 0.40 * t + 0.10 * x * ph
    z = np.zeros((n, h, w), np.float32)
    rgb = np.stack([r + z, g + z, b + z], axis=-1).astype(np.float32) + np.float32(offset)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    return C.Frames(rgb=rgb, fps=fps, shot_hash=hashlib.sha256(item_id.encode()).hexdigest(),
                    world_hash=hashlib.sha256(b"w" + item_id.encode()).hexdigest())


class FakeRender:
    """호출을 기록하는 렌더러 대역. offsets[id] 로 항목별 출력을 바꾸고 fail[id] 로 예외를 낸다."""

    def __init__(self, offsets=None, fail=None, interrupt_at=None, with_root=True, bad_type=()):
        self.offsets = dict(offsets or {})
        self.fail = dict(fail or {})
        self.interrupt_at = interrupt_at
        self.with_root = with_root
        self.bad_type = set(bad_type)
        self.calls = []

    def __call__(self, job):
        self.calls.append((job.id, job.seed, job.move))
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt
        if job.id in self.fail:
            raise self.fail[job.id]
        if job.id in self.bad_type:
            return np.zeros((2, 2, 2, 3), np.float32)
        fr = make_frames(job.id, self.offsets.get(job.id, 0.0))
        return ab().Rendered(fr, f"root-{job.id}") if self.with_root else fr

    @property
    def ids(self):
        return [c[0] for c in self.calls]


def file_hashes(d):
    out = {}
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def last_frame(mp4):
    return V.decode_frames(str(mp4))[-1]


def crop(img, box):
    x0, y0, x1, y1 = box
    return np.asarray(img.crop((x0, y0, x1, y1)).convert("RGB"))


def listing(d):
    return sorted(os.listdir(d))


def cli_renderer():
    """CLI `--renderer evl02_support:cli_renderer` 가 부르는 공장 함수."""
    return FakeRender()
