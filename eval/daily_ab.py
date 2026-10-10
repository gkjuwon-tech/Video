"""EVL-02 일일 A/B 러너 v0: 24장 고정 무브 -> 전 구간 -> mp4 -> 전일 대비 시트 (야간 잡).

렌더 자체는 주입된 ``render(job)`` 가 한다(스파인 연결은 ``spine_renderer``). 이 모듈은 장부(run.json) +
인코딩 + 시트 층이다. 한 항목이 터져도 밤 전체가 죽지 않고, 끊겨도 ``resume`` 으로 이어서 돈다.
평가 모델(lpips 등)은 싣지 않는다(R-EVAL). import 부작용 없음.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

if __name__ == "__main__":  # 스크립트로 실행될 때만 레포 루트를 경로에 넣는다
    _R = str(Path(__file__).resolve().parents[1])
    if _R not in sys.path:
        sys.path.insert(0, _R)

import numpy as np
from PIL import Image, ImageDraw

from buc.core import det
from buc.core.contracts import Frames
from buc.render import video as V

SCHEMA_VERSION = 1
DEFAULT_MOVE = "I1"
SEED_TAG = "buc.daily_ab.v1"
RECORD_NAME = "run.json"
SHEET_NAME = "sheet.png"
PSNR_CAP_DB = 99.0
NO_PREV_RGB = (70, 70, 70)
FAILED_RGB = (120, 16, 16)
BG_RGB = (24, 24, 28)
STATUS_RGB = {
    "new": (60, 120, 220),
    "same": (50, 150, 80),
    "changed": (230, 170, 30),
    "failed": (190, 50, 50),
    "regressed": (200, 60, 200),
}
PAD = 4
HEADER_H = 28
LABEL_H = 16

_ROOT = Path(__file__).resolve().parents[1]
_QA_PATH = _ROOT / "tools" / "inputs_qa.py"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SECRET_RE = re.compile(r"h" + r"f_[A-Za-z0-9_]{6,}|sk-[A-Za-z0-9_\-]{10,}")
_STATUSES = ("new", "same", "changed", "failed", "regressed")
_ITEM_KEYS = ("id", "slot", "seed", "status", "mp4", "n_frames", "fps", "width", "height", "frames_hash",
              "mp4_sha256", "root", "error", "wall_s", "delta")


class DailyABError(RuntimeError):
    """일일 러너 오류(계획·장부·금고 대조)."""


@dataclass(frozen=True)
class Job:
    item: object
    move: str
    seed: int
    crop_box: tuple

    @property
    def id(self) -> str:
        return self.item.id

    @property
    def slot(self) -> int:
        return self.item.slot


@dataclass(frozen=True)
class Rendered:
    frames: Frames
    root: Optional[str] = None


Render = Callable[[Job], object]


@dataclass(frozen=True)
class PrevDay:
    date: str
    dir: Path
    record: dict


@dataclass(frozen=True)
class DailyResult:
    record: dict
    day_dir: Path
    record_path: Path
    sheet_path: Path
    reused: tuple

    @property
    def ok(self) -> bool:
        return self.record["summary"]["failed"] == 0


# ------------------------------------------------------------------------------------------------ 지연 적재
_QA_MOD = None


def _qa():
    """tools/inputs_qa.py 를 파일 경로로 지연 적재한다(tools/ 는 패키지가 아니다)."""
    global _QA_MOD
    if _QA_MOD is None:
        name = "inputs_qa_daily_ab"
        spec = importlib.util.spec_from_file_location(name, _QA_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _QA_MOD = mod
    return _QA_MOD


def load_inputs(path=None, *, partial: bool = False):
    return _qa().load_manifest(path, require_ready=not partial)


# ------------------------------------------------------------------------------------------------ 검증 도우미
def _check_date(date) -> str:
    if not isinstance(date, str) or not _DATE_RE.match(date):
        raise ValueError(f"date 는 YYYY-MM-DD 여야 한다: {date!r}")
    try:
        _dt.date.fromisoformat(date)
    except ValueError as e:
        raise ValueError(f"달력상 유효하지 않은 date: {date!r}") from e
    return date


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_thumb(thumb_size):
    try:
        tw, th = thumb_size
    except (TypeError, ValueError) as e:
        raise ValueError(f"thumb_size 는 (w, h) 여야 한다: {thumb_size!r}") from e
    if not (_is_int(tw) and _is_int(th)) or tw < 1 or th < 1:
        raise ValueError(f"thumb_size 는 양의 정수 쌍이어야 한다: {thumb_size!r}")
    return int(tw), int(th)


# ------------------------------------------------------------------------------------------------ 시드·계획
def job_seed(base_seed, item_id, move) -> int:
    return det.derive_seed(base_seed, SEED_TAG, item_id, move)


def plan_jobs(m, *, move=DEFAULT_MOVE, base_seed=0, partial=False) -> list:
    if not isinstance(move, str) or move not in m.moves:
        raise ValueError(f"move {move!r} 는 매니페스트 moves {tuple(m.moves)} 에 없다")
    items = sorted(m.items, key=lambda x: x.slot)
    if partial:
        chosen = [x for x in items if x.status == "ready"]
        if not chosen:
            raise DailyABError("ready 항목이 하나도 없다")
    else:
        planned = [x.id for x in items if x.status != "ready"]
        if planned:
            raise DailyABError("planned 항목이 있어 24장 모드를 돌릴 수 없다(--partial 필요): " + ", ".join(planned))
        chosen = items
    return [Job(item=x, move=move, seed=job_seed(base_seed, x.id, move), crop_box=x.crop_box) for x in chosen]


def check_sha256(item, data: bytes) -> None:
    want = getattr(item, "sha256", None)
    if not want:
        raise DailyABError(f"{item.id}: 대조할 sha256 이 없다")
    if hashlib.sha256(bytes(data)).hexdigest() != str(want).lower():
        raise DailyABError(f"{item.id}: 금고 바이트 sha256 불일치")


def spine_renderer(spine, load_image, *, params=None, until="S9-10", move_stage="S8") -> Render:
    def render(job: Job) -> Rendered:
        image = load_image(job)
        P = copy.deepcopy(dict(params)) if params else {}
        stage = dict(P.get(move_stage) or {})
        stage["move"] = job.move
        stage["crop_box"] = list(job.crop_box)
        P[move_stage] = stage
        result = spine.run(image, seed=job.seed, params=P, until=until)
        frames = (getattr(result, "artifacts", None) or {}).get("frames")
        if frames is None:
            raise DailyABError(f"{job.id}: 스파인 결과에 frames 아티팩트가 없다(until={until})")
        return Rendered(frames, result.root)

    return render


# ------------------------------------------------------------------------------------------------ 장부 읽기
def read_record(day_dir) -> dict:
    d = Path(day_dir)
    p = d / RECORD_NAME
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DailyABError(f"{p}: 장부를 읽을 수 없다 ({type(e).__name__})") from e
    if not isinstance(rec, dict):
        raise DailyABError(f"{p}: 최상위가 객체가 아니다")
    if rec.get("schema") != SCHEMA_VERSION:
        raise DailyABError(f"{p}: 알 수 없는 schema")
    if not isinstance(rec.get("complete"), bool):
        raise DailyABError(f"{p}: complete 가 bool 이 아니다")
    if not isinstance(rec.get("items"), list):
        raise DailyABError(f"{p}: items 가 list 가 아니다")
    if rec.get("date") != d.name:
        raise DailyABError(f"{p}: date 가 디렉터리 이름과 다르다")
    return rec


def find_previous(out_root, date) -> Optional[PrevDay]:
    _check_date(date)
    root = Path(out_root)
    try:
        names = sorted(os.listdir(root), reverse=True)
    except OSError:
        return None
    for name in names:
        if not _DATE_RE.match(name) or name >= date:
            continue
        try:
            _dt.date.fromisoformat(name)
        except ValueError:
            continue
        d = root / name
        if not d.is_dir():
            continue
        try:
            rec = read_record(d)
        except DailyABError:
            continue
        if rec["complete"] is True:
            return PrevDay(name, d, rec)
    return None


def _strip_wall(x):
    if isinstance(x, dict):
        return {k: _strip_wall(v) for k, v in x.items() if k != "wall_s"}
    if isinstance(x, (list, tuple)):
        return [_strip_wall(v) for v in x]
    return x


def canonical_digest(record) -> str:
    s = json.dumps(_strip_wall(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ 클립 델타
def clip_delta(prev_mp4, cur_mp4) -> dict:
    a = V.decode_frames(str(prev_mp4))
    b = V.decode_frames(str(cur_mp4))
    if a.shape != b.shape:
        return {"comparable": False, "mad": None, "psnr_db": None}
    diff = a.astype(np.float64) - b.astype(np.float64)
    mad = float(np.abs(diff).mean())
    mse = float((diff ** 2).mean())
    psnr = PSNR_CAP_DB if mse <= 0.0 else min(PSNR_CAP_DB, 10.0 * math.log10(255.0 ** 2 / mse))
    return {"comparable": True, "mad": round(mad, 6), "psnr_db": round(psnr, 6)}


# ------------------------------------------------------------------------------------------------ 시트
@dataclass(frozen=True)
class SheetGeometry:
    n: int
    thumb_size: tuple
    cols: int
    rows: int
    cell_w: int
    cell_h: int
    size: tuple

    def _origin(self, i):
        if not _is_int(i) or not 0 <= i < self.n:
            raise IndexError(f"항목 인덱스 범위 밖: {i!r}")
        return (i % self.cols) * self.cell_w, HEADER_H + (i // self.cols) * self.cell_h

    def thumb_box(self, i, which) -> tuple:
        if which not in ("prev", "cur"):
            raise ValueError(f"which 는 'prev'|'cur': {which!r}")
        x, y = self._origin(i)
        tw, th = self.thumb_size
        x0 = x + PAD if which == "prev" else x + 2 * PAD + tw
        return (x0, y + PAD, x0 + tw, y + PAD + th)

    def label_box(self, i) -> tuple:
        x, y = self._origin(i)
        th = self.thumb_size[1]
        return (x + PAD, y + PAD + th, x + self.cell_w - PAD, y + PAD + th + LABEL_H)


def sheet_geometry(n_items, thumb_size=(160, 90), cols=4) -> SheetGeometry:
    if not _is_int(n_items):
        raise TypeError(f"n_items 는 int 여야 한다: {n_items!r}")
    if n_items < 1:
        raise ValueError("n_items >= 1")
    tw, th = _check_thumb(thumb_size)
    if not _is_int(cols):
        raise TypeError(f"cols 는 int 여야 한다: {cols!r}")
    if cols < 1:
        raise ValueError("cols >= 1")
    ce = min(cols, n_items)
    rows = -(-n_items // ce)
    cw, ch = 3 * PAD + 2 * tw, 2 * PAD + th + LABEL_H
    return SheetGeometry(n_items, (tw, th), ce, rows, cw, ch, (ce * cw, HEADER_H + rows * ch))


def _ascii(s) -> str:
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(s))


def _text_rgb(bg):
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _last_frame(mp4):
    try:
        return V.decode_frames(str(mp4))[-1]
    except Exception:  # noqa: BLE001 - 디코드 실패는 칸 대체색으로
        return None


def _thumb(arr, tw, th):
    if arr.shape[0] == th and arr.shape[1] == tw:
        return arr
    return np.asarray(Image.fromarray(arr).resize((tw, th), Image.BILINEAR))


def _paste(img, box, arr, flat_rgb):
    x0, y0, x1, y1 = box
    if arr is None:
        img[y0:y1, x0:x1] = np.array(flat_rgb, np.uint8)
    else:
        img[y0:y1, x0:x1] = _thumb(arr, x1 - x0, y1 - y0)


def _write_atomic(path: Path, writer) -> None:
    tmp = path.with_name(".tmp_" + path.name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _render_sheet(path, record, items, day_dir, prev, geo) -> None:
    W, H = geo.size
    img = np.empty((H, W, 3), np.uint8)
    img[:] = np.array(BG_RGB, np.uint8)
    tw, th = geo.thumb_size
    for i, e in enumerate(items):
        d = e["delta"]
        # 오른쪽 = 오늘
        cur = None
        if e["status"] == "ok" and e["mp4"]:
            cur = _last_frame(day_dir / e["mp4"])
        _paste(img, geo.thumb_box(i, "cur"), cur, FAILED_RGB)
        # 왼쪽 = 전일
        pv = None
        if prev is not None and d["status"] in ("same", "changed", "regressed"):
            pe = _prev_item(prev, e["id"])
            if pe is not None:
                pv = _last_frame(prev.dir / pe["mp4"])
        _paste(img, geo.thumb_box(i, "prev"), pv, NO_PREV_RGB)
        lx0, ly0, lx1, ly1 = geo.label_box(i)
        col = STATUS_RGB[d["status"]]
        img[ly0:ly1, lx0:lx1] = np.array(col, np.uint8)
    pil = Image.fromarray(img, "RGB")
    dr = ImageDraw.Draw(pil)
    for i, e in enumerate(items):
        lx0, ly0, _, _ = geo.label_box(i)
        d = e["delta"]
        txt = f"{e['id']} {d['status']}"
        if d.get("mad") is not None:
            txt += f" {d['mad']:.1f}"
        dr.text((lx0 + 2, ly0 + 2), _ascii(txt), fill=_text_rgb(STATUS_RGB[d["status"]]))
    head1 = f"{record['date']}  prev {record['prev_date'] or '-'}  tag {record['tag'] or '-'}  move {record['move']}"
    dr.text((4, 2), _ascii(head1), fill=(235, 235, 235))
    dr.text((4, 14), _ascii(record.get("disclosure") or ""), fill=(170, 170, 175))
    _write_atomic(path, lambda t: pil.save(str(t), format="PNG"))


# ------------------------------------------------------------------------------------------------ 전일 대비
def _prev_item(prev, item_id):
    for pe in prev.record.get("items", []):
        if isinstance(pe, dict) and pe.get("id") == item_id:
            if pe.get("status") == "ok" and isinstance(pe.get("mp4"), str) and (prev.dir / pe["mp4"]).is_file():
                return pe
            return None
    return None


def _delta(prev, e, day_dir) -> dict:
    ok = e["status"] == "ok"
    pe = _prev_item(prev, e["id"]) if prev is not None else None
    if pe is None:
        return {"status": "new" if ok else "failed", "mad": None, "psnr_db": None, "prev_date": None}
    if not ok:
        return {"status": "regressed", "mad": None, "psnr_db": None, "prev_date": prev.date}
    if pe.get("frames_hash") == e["frames_hash"]:
        return {"status": "same", "mad": 0.0, "psnr_db": PSNR_CAP_DB, "prev_date": prev.date}
    try:
        cd = clip_delta(prev.dir / pe["mp4"], day_dir / e["mp4"])
    except Exception:  # noqa: BLE001
        cd = {"comparable": False, "mad": None, "psnr_db": None}
    return {"status": "changed", "mad": cd["mad"], "psnr_db": cd["psnr_db"], "prev_date": prev.date}


# ------------------------------------------------------------------------------------------------ 실행
def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _err_text(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}".replace("\n", " ").replace("\r", " ")
    return _SECRET_RE.sub("[redacted]", s)[:300]


def _try_reuse(old, job, day_dir):
    e = old.get(job.id)
    if not isinstance(e, dict) or any(k not in e for k in _ITEM_KEYS):
        return None
    if e["status"] != "ok" or e["seed"] != job.seed or not isinstance(e["mp4"], str):
        return None
    f = day_dir / e["mp4"]
    try:
        if not f.is_file() or _sha_file(f) != e["mp4_sha256"]:
            return None
    except OSError:
        return None
    return dict(e)


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def _do_job(job, render, day_dir, encoder, crf) -> dict:
    t0 = time.monotonic()
    name = f"{job.id}_{job.move}.mp4"
    path = day_dir / name
    e = {"id": job.id, "slot": job.slot, "seed": job.seed, "status": "failed", "mp4": None, "n_frames": None,
         "fps": None, "width": None, "height": None, "frames_hash": None, "mp4_sha256": None, "root": None,
         "error": None, "wall_s": 0.0, "delta": None}
    try:
        out = render(job)
        if isinstance(out, Frames):
            out = Rendered(out, None)
        if not isinstance(out, Rendered) or not isinstance(out.frames, Frames):
            raise TypeError(f"render 는 Frames 또는 Rendered 를 돌려줘야 한다: {type(out).__name__}")
        frames = out.frames
        V.encode_frames(frames, str(path), encoder=encoder, crf=crf)
        info = V.probe(str(path))
        if info.n_frames != int(frames.rgb.shape[0]):
            raise DailyABError(f"프레임 수 불일치: 인코딩 {info.n_frames} != 렌더 {int(frames.rgb.shape[0])}")
        e.update(status="ok", mp4=name, n_frames=int(info.n_frames), fps=float(info.fps), width=int(info.width),
                 height=int(info.height), frames_hash=frames.content_hash(), mp4_sha256=_sha_file(path),
                 root=None if out.root is None else str(out.root))
    except Exception as ex:  # noqa: BLE001 - 한 항목의 실패는 그 항목만
        _unlink(path)
        e["error"] = _err_text(ex)
    except BaseException:
        _unlink(path)
        raise
    e["wall_s"] = round(time.monotonic() - t0, 3)
    return e


def _summary(items, n_jobs) -> dict:
    deltas = {k: 0 for k in _STATUSES}
    for e in items:
        deltas[e["delta"]["status"]] += 1
    ok = sum(1 for e in items if e["status"] == "ok")
    return {"n_jobs": n_jobs, "ok": ok, "failed": len(items) - ok, "deltas": deltas}


def run_daily(manifest, render, out_root, date, *, move=DEFAULT_MOVE, base_seed=0, partial=False, tag="",
              encoder="cpu", crf=18, resume=True, thumb_size=(160, 90), cols=4) -> DailyResult:
    _check_date(date)
    tw, th = _check_thumb(thumb_size)
    jobs = plan_jobs(manifest, move=move, base_seed=base_seed, partial=partial)
    geo = sheet_geometry(len(jobs), (tw, th), cols)
    tag = "" if tag is None else str(tag)
    qa = _qa()
    matrix = qa.coverage(manifest)["matrix"]

    t_start = time.monotonic()
    root = Path(out_root)
    prev = find_previous(root, date)
    day_dir = root / date
    day_dir.mkdir(parents=True, exist_ok=True)
    record_path = day_dir / RECORD_NAME
    sheet_path = day_dir / SHEET_NAME

    old = {}
    if resume:
        try:
            orec = read_record(day_dir)
            if orec.get("move") == move and orec.get("tag") == tag:
                old = {x.get("id"): x for x in orec["items"] if isinstance(x, dict)}
        except DailyABError:
            old = {}

    head = {
        "schema": SCHEMA_VERSION, "date": date, "move": move, "base_seed": base_seed, "tag": tag,
        "prev_date": prev.date if prev is not None else None,
        "vault_repo": manifest.vault_repo,
        "disclosure": manifest.selection.get("disclosure"),
        "matrix": matrix,
        "n_planned_skipped": len(manifest.items) - len(jobs),
    }

    def build(items, complete):
        rec = dict(head)
        rec.update(complete=complete, items=list(items), summary=_summary(items, len(jobs)),
                   wall_s=round(time.monotonic() - t_start, 3))
        return rec

    def write_record(rec):
        text = json.dumps(rec, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
        _write_atomic(record_path, lambda t: t.write_text(text, encoding="utf-8"))

    items, reused = [], []
    for job in jobs:
        e = _try_reuse(old, job, day_dir) if resume else None
        if e is not None:
            e["slot"] = job.slot
            reused.append(job.id)
        else:
            e = _do_job(job, render, day_dir, encoder, crf)
        e["delta"] = _delta(prev, e, day_dir)
        items.append(e)
        write_record(build(items, False))

    # 장부와 디스크 정합: mp4 존재 <=> status ok
    keep = {e["mp4"] for e in items if e["status"] == "ok"}
    for f in day_dir.iterdir():
        if f.is_file() and ((f.suffix == ".mp4" and f.name not in keep) or f.name.startswith(".tmp")):
            _unlink(f)

    final = build(items, True)
    _render_sheet(sheet_path, final, items, day_dir, prev, geo)
    write_record(final)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return DailyResult(record=record, day_dir=day_dir, record_path=record_path, sheet_path=sheet_path,
                       reused=tuple(reused))


# ------------------------------------------------------------------------------------------------ CLI
def _resolve_renderer(spec: str):
    mod, sep, attr = spec.partition(":")
    if not sep or not mod or not attr:
        raise ValueError(f"--renderer 는 MOD:ATTR 형식이어야 한다: {spec!r}")
    factory = getattr(importlib.import_module(mod), attr)
    return factory()


def main(argv=None, *, render=None) -> int:
    ap = argparse.ArgumentParser(description="EVL-02 일일 A/B 러너")
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--move", default=DEFAULT_MOVE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--renderer", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    try:
        _check_date(args.date)
        m = load_inputs(args.manifest, partial=args.partial)
        plan_jobs(m, move=args.move, base_seed=args.seed, partial=args.partial)
        if render is None:
            if not args.renderer:
                raise ValueError("렌더러가 없다(--renderer MOD:ATTR)")
            render = _resolve_renderer(args.renderer)
    except Exception as e:  # noqa: BLE001
        print(f"error: {_err_text(e)}", file=sys.stderr)
        return 2
    try:
        res = run_daily(m, render, args.out, args.date, move=args.move, base_seed=args.seed, partial=args.partial,
                        tag=args.tag)
    except (ValueError, DailyABError) as e:
        print(f"error: {_err_text(e)}", file=sys.stderr)
        return 2
    s = res.record["summary"]
    print(f"{args.date} ok={s['ok']} failed={s['failed']}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
