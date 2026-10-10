"""EVL-02 순수·준순수 부분: 작업 계획, 시드, 전일 탐색, 시트 기하, 클립 델타, 금고 바이트 대조, 스파인 렌더러."""
import dataclasses
import json
import math

import numpy as np
import pytest

from buc.core import det
from buc.render import video as V
from evl02_support import FakeRender, ab, make_frames, make_manifest, sha_file

M = ab()


# ------------------------------------------------------------------------------------------------ 시드
def test_job_seed_formula_and_spread():
    s = M.job_seed(7, "in05", "I1")
    assert s == det.derive_seed(7, "buc.daily_ab.v1", "in05", "I1")
    assert isinstance(s, int) and 0 <= s < 1 << 64
    assert s == M.job_seed(7, "in05", "I1")  # 결정론
    others = {M.job_seed(7, "in06", "I1"), M.job_seed(8, "in05", "I1"), M.job_seed(7, "in05", "I2")}
    assert s not in others and len(others) == 3
    assert len({M.job_seed(0, f"in{k:02d}", "I1") for k in range(1, 25)}) == 24


# ------------------------------------------------------------------------------------------------ 계획
def test_plan_jobs_full_manifest_is_24_in_slot_order():
    m = make_manifest(24)
    jobs = M.plan_jobs(m, base_seed=11)
    assert [j.slot for j in jobs] == list(range(1, 25))
    assert [j.id for j in jobs] == [f"in{k:02d}" for k in range(1, 25)]
    assert {j.move for j in jobs} == {"I1"} == {M.DEFAULT_MOVE}
    for j, it in zip(jobs, m.items):
        assert j.item is it or j.item == it
        assert j.seed == M.job_seed(11, it.id, "I1")
        assert tuple(j.crop_box) == tuple(it.crop_box)  # 같은 크롭 파일(EVL-01 규약)
    assert M.plan_jobs(m, move="I3")[0].move == "I3"


def test_plan_jobs_input_order_does_not_matter():
    m = make_manifest(24)
    shuffled = dataclasses.replace(m, items=tuple(reversed(m.items)))
    assert [j.slot for j in M.plan_jobs(shuffled)] == list(range(1, 25))


def test_plan_jobs_refuses_planned_unless_partial():
    m = make_manifest(4)
    with pytest.raises(M.DailyABError) as ei:
        M.plan_jobs(m)
    msg = str(ei.value)
    assert "in05" in msg and "in24" in msg and "in01" not in msg  # planned 항목만 이름을 든다
    jobs = M.plan_jobs(m, partial=True)
    assert [j.id for j in jobs] == ["in01", "in02", "in03", "in04"]
    assert issubclass(M.DailyABError, RuntimeError)


def test_plan_jobs_partial_with_nothing_ready_and_bad_move():
    m = make_manifest(0)
    m = dataclasses.replace(m, items=tuple(dataclasses.replace(x, status="planned") for x in m.items))
    with pytest.raises(M.DailyABError):
        M.plan_jobs(m, partial=True)
    for bad in ("I9", "push_in", "", None):
        with pytest.raises(ValueError):
            M.plan_jobs(make_manifest(2), move=bad, partial=True)


# ------------------------------------------------------------------------------------------------ 전일 탐색
def _rec(date, complete=True, items=None):
    return {"schema": 1, "date": date, "complete": complete, "items": items or []}


def _put(root, name, rec=None, raw=None):
    d = root / name
    d.mkdir(parents=True)
    if raw is not None:
        (d / "run.json").write_text(raw, encoding="utf-8")
    elif rec is not None:
        (d / "run.json").write_text(json.dumps(rec), encoding="utf-8")
    return d


def test_find_previous_picks_latest_valid_complete_earlier_day(tmp_path):
    _put(tmp_path, "2026-10-01", _rec("2026-10-01"))
    _put(tmp_path, "2026-10-03", raw="{ not json")
    _put(tmp_path, "2026-10-04", _rec("2026-10-04", complete=False))      # 끊긴 날은 기준선이 아니다
    _put(tmp_path, "2026-10-05", _rec("2026-10-06"))                     # 디렉터리 이름과 date 불일치
    _put(tmp_path, "2026-10-07", dict(_rec("2026-10-07"), schema=99))    # 알 수 없는 스키마
    _put(tmp_path, "notes")                                              # 날짜 이름이 아닌 디렉터리
    (tmp_path / "2026-10-08").mkdir()                                    # run.json 없음
    _put(tmp_path, "2026-10-09", _rec("2026-10-09"))
    _put(tmp_path, "2026-10-10", _rec("2026-10-10"))                     # 같은 날은 전일이 아니다
    _put(tmp_path, "2026-10-12", _rec("2026-10-12"))                     # 미래도 아니다
    (tmp_path / "2026-10-02").write_text("file, not dir")
    p = M.find_previous(tmp_path, "2026-10-10")
    assert (p.date, p.dir.name, p.record["date"]) == ("2026-10-09", "2026-10-09", "2026-10-09")
    assert M.find_previous(tmp_path, "2026-10-09").date == "2026-10-01"
    assert M.find_previous(tmp_path, "2026-10-01") is None
    assert M.find_previous(tmp_path / "missing", "2026-10-10") is None


def test_find_previous_and_read_record_validation(tmp_path):
    with pytest.raises(ValueError):
        M.find_previous(tmp_path, "2026-13-40")
    d = _put(tmp_path, "2026-10-01", _rec("2026-10-01"))
    assert M.read_record(d)["date"] == "2026-10-01"
    for name, raw in {"2026-10-02": "[]", "2026-10-03": "null", "2026-10-04": json.dumps(dict(_rec("2026-10-04"), items={}))}.items():
        dd = _put(tmp_path, name, raw=raw)
        with pytest.raises(M.DailyABError):
            M.read_record(dd)
    with pytest.raises(M.DailyABError):
        M.read_record(tmp_path / "nonexistent")


# ------------------------------------------------------------------------------------------------ 시트 기하
def test_sheet_geometry_numbers():
    g = M.sheet_geometry(24, (64, 48), 4)
    assert (M.PAD, M.HEADER_H, M.LABEL_H) == (4, 28, 16)
    assert (g.cell_w, g.cell_h, g.cols, g.rows) == (3 * 4 + 128, 2 * 4 + 48 + 16, 4, 6) == (140, 72, 4, 6)
    assert tuple(g.size) == (560, 28 + 6 * 72) == (560, 460)
    # 슬롯 6(인덱스 5) = 1행 1열
    assert tuple(g.thumb_box(5, "prev")) == (144, 104, 208, 152)
    assert tuple(g.thumb_box(5, "cur")) == (212, 104, 276, 152)
    assert tuple(g.label_box(5)) == (144, 152, 276, 168)
    assert tuple(g.thumb_box(0, "prev")) == (4, 32, 68, 80)
    # 열 수보다 항목이 적으면 폭이 줄고, 마지막 행이 덜 차도 높이는 행 수로
    assert tuple(M.sheet_geometry(2, (64, 48), 4).size) == (280, 28 + 72)
    assert tuple(M.sheet_geometry(5, (64, 48), 4).size) == (560, 28 + 2 * 72)
    # 기본값
    assert tuple(M.sheet_geometry(24).size) == (4 * (12 + 320), 28 + 6 * (8 + 90 + 16))


def test_sheet_geometry_boxes_disjoint_and_inside():
    g = M.sheet_geometry(24, (64, 48), 4)
    W, H = g.size
    cells = []
    for i in range(24):
        for box in (g.thumb_box(i, "prev"), g.thumb_box(i, "cur"), g.label_box(i)):
            x0, y0, x1, y1 = box
            assert 0 <= x0 < x1 <= W and 28 <= y0 < y1 <= H
            cells.append(box)
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            A, B = cells[a], cells[b]
            assert A[2] <= B[0] or B[2] <= A[0] or A[3] <= B[1] or B[3] <= A[1], (A, B)
    for i in (-1, 24):
        with pytest.raises(IndexError):
            g.thumb_box(i, "cur")
    with pytest.raises(ValueError):
        g.thumb_box(0, "both")


def test_sheet_geometry_validation():
    for args in ((0,), (-3,), (24, (0, 48)), (24, (64, 0)), (24, (64, 48), 0), (2.5,)):
        with pytest.raises((ValueError, TypeError)):
            M.sheet_geometry(*args)


def test_status_colors_are_distinct_and_cover_all_statuses():
    assert set(M.STATUS_RGB) == {"new", "same", "changed", "failed", "regressed"}
    assert len(set(map(tuple, M.STATUS_RGB.values()))) == 5
    special = {tuple(M.NO_PREV_RGB), tuple(M.FAILED_RGB), tuple(M.BG_RGB)}
    assert len(special) == 3
    assert M.PSNR_CAP_DB == 99.0


# ------------------------------------------------------------------------------------------------ 클립 델타
def _enc(tmp_path, name, frames):
    p = tmp_path / name
    V.encode_frames(frames, str(p), encoder="cpu")
    return p


def test_clip_delta_identical_changed_and_incomparable(tmp_path):
    a = _enc(tmp_path, "a.mp4", make_frames("in03"))
    a2 = _enc(tmp_path, "a2.mp4", make_frames("in03"))
    b = _enc(tmp_path, "b.mp4", make_frames("in03", offset=0.1))
    d0 = M.clip_delta(a, a2)  # 같은 입력·같은 인코더 = 같은 바이트(RDR-06 불변식 3)
    assert d0["comparable"] is True and d0["mad"] == 0.0 and d0["psnr_db"] == M.PSNR_CAP_DB
    d1 = M.clip_delta(a, b)
    da, db = V.decode_frames(str(a)).astype(np.float64), V.decode_frames(str(b)).astype(np.float64)
    mad = float(np.abs(da - db).mean())
    mse = float(((da - db) ** 2).mean())
    assert d1["comparable"] is True
    assert d1["mad"] == pytest.approx(mad, abs=1e-4)
    assert d1["psnr_db"] == pytest.approx(10 * math.log10(255.0 ** 2 / mse), abs=1e-3)
    assert mad == pytest.approx(25.5, abs=2.0)  # 오프셋 0.1 = 25.5/255, 손실 압축 여유
    assert d1["mad"] == pytest.approx(M.clip_delta(b, a)["mad"], abs=1e-6)  # 대칭
    c = _enc(tmp_path, "c.mp4", make_frames("in03", n=4))            # 프레임 수가 다르면 비교 불가
    d = _enc(tmp_path, "d.mp4", make_frames("in03", h=32, w=48))     # 크기가 다르면 비교 불가
    for other in (c, d):
        x = M.clip_delta(a, other)
        assert x["comparable"] is False and x["mad"] is None and x["psnr_db"] is None


# ------------------------------------------------------------------------------------------------ 금고 바이트
def test_check_sha256_accepts_match_and_rejects_mismatch_without_dumping_data():
    import hashlib
    data = b"photo-bytes-" * 100
    it = dataclasses.replace(make_manifest(2).items[1], sha256=hashlib.sha256(data).hexdigest())
    assert M.check_sha256(it, data) is None
    with pytest.raises(M.DailyABError) as ei:
        M.check_sha256(it, data + b"x")
    assert it.id in str(ei.value) and "photo-bytes" not in str(ei.value)
    with pytest.raises(M.DailyABError):
        M.check_sha256(dataclasses.replace(it, sha256=None), data)  # planned 항목은 대조할 해시가 없다


# ------------------------------------------------------------------------------------------------ 스파인 렌더러
class _FakeResult:
    def __init__(self, artifacts, root):
        self.artifacts, self.root = artifacts, root


class _FakeSpine:
    def __init__(self, with_frames=True):
        self.calls = []
        self.with_frames = with_frames

    def run(self, image, **kw):
        self.calls.append((image, kw))
        arts = {"frames": make_frames("in02")} if self.with_frames else {}
        return _FakeResult(arts, "r" * 64)


def test_spine_renderer_passes_seed_move_crop_and_returns_frames_and_root():
    sp = _FakeSpine()
    loaded = []

    def load_image(job):
        loaded.append(job.id)
        return f"IMG-{job.id}"

    user_params = {"S9-10": {"spp": 4}, "S8": {"rig": "dolly"}}
    snapshot = json.loads(json.dumps(user_params))
    render = M.spine_renderer(sp, load_image, params=user_params)
    job = M.plan_jobs(make_manifest(3), base_seed=5, partial=True)[2]
    out = render(job)
    assert isinstance(out, M.Rendered) and out.root == "r" * 64
    assert out.frames.content_hash() == make_frames("in02").content_hash()
    (image, kw), = sp.calls
    assert image == "IMG-in03" and loaded == ["in03"]
    assert kw["seed"] == job.seed == M.job_seed(5, "in03", "I1")
    assert kw["until"] == "S9-10"
    assert kw["params"]["S8"] == {"rig": "dolly", "move": "I1", "crop_box": [0, 0, 64, 48]}  # 사용자 값 보존 + 무브·크롭 주입
    assert kw["params"]["S9-10"] == {"spp": 4}
    assert user_params == snapshot  # 입력 params 불변
    assert kw.get("force", ()) in ((), None, [])  # 일일 러너는 캐시를 끄지 않는다(PRD-01 부분 재실행을 쓴다)


def test_spine_renderer_without_frames_raises_runner_error():
    render = M.spine_renderer(_FakeSpine(with_frames=False), lambda job: "img")
    with pytest.raises(M.DailyABError):
        render(M.plan_jobs(make_manifest(1), partial=True)[0])
