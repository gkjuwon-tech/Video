"""EVL-02 일일 러너 끝에서 끝까지: 24장 고정 무브 -> mp4 -> 장부(run.json) -> 전일 대비 시트. 렌더러는 대역이다."""
import re
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from buc.render import video as V
from evl02_support import (H, N, W, FakeRender, ab, file_hashes, last_frame, listing, make_frames, make_manifest,
                           qa, read_json, sha_file)

M = ab()
D1, D2 = "2026-10-10", "2026-10-11"
THUMB = (W, H)  # 썸네일 = 클립 크기 -> 시트 칸 픽셀이 디코드한 프레임과 정확히 같다
DELTA_KEYS = {"new", "same", "changed", "failed", "regressed"}


def sheet_of(res, n, thumb=THUMB, cols=4):
    img = np.asarray(Image.open(res.sheet_path).convert("RGB"))
    geo = M.sheet_geometry(n, thumb, cols)
    assert img.shape == (geo.size[1], geo.size[0], 3)
    return img, geo


def box(img, b):
    x0, y0, x1, y1 = b
    return img[y0:y1, x0:x1]


def flat(img, b, rgb):
    return bool((box(img, b) == np.array(rgb, np.uint8)).all())


@pytest.fixture(scope="module")
def two_days(tmp_path_factory):
    root = tmp_path_factory.mktemp("ab_two_days")
    m = make_manifest(4)
    r1 = FakeRender()
    res1 = M.run_daily(m, r1, root, D1, partial=True, base_seed=3, tag="c1", thumb_size=THUMB)
    snap1 = file_hashes(res1.day_dir)
    r2 = FakeRender(offsets={"in03": 0.1}, fail={"in04": RuntimeError("boom: CUDA OOM")})
    res2 = M.run_daily(m, r2, root, D2, partial=True, base_seed=3, tag="c2", thumb_size=THUMB)
    return SimpleNamespace(root=root, m=m, r1=r1, r2=r2, res1=res1, res2=res2, snap1=snap1)


# ------------------------------------------------------------------------------------------------ 첫날
def test_day1_outputs_exactly_mp4s_record_and_sheet(two_days):
    d = two_days.res1.day_dir
    assert d == two_days.root / D1
    assert listing(d) == sorted([f"in0{k}_I1.mp4" for k in range(1, 5)] + ["run.json", "sheet.png"])
    assert two_days.res1.record_path == d / "run.json" and two_days.res1.sheet_path == d / "sheet.png"
    assert two_days.r1.ids == ["in01", "in02", "in03", "in04"]  # 슬롯 순서로 정확히 한 번씩
    assert two_days.res1.reused == ()
    assert two_days.res1.record == read_json(d / "run.json") == M.read_record(d)


def test_day1_record_fields(two_days):
    rec = M.read_record(two_days.res1.day_dir)
    m = two_days.m
    assert rec["schema"] == 1 and rec["date"] == D1 and rec["complete"] is True
    assert rec["move"] == "I1" and rec["base_seed"] == 3 and rec["tag"] == "c1" and rec["prev_date"] is None
    assert rec["vault_repo"] == m.vault_repo
    assert rec["disclosure"] == m.selection["disclosure"] and rec["disclosure"]  # EVL-01 선별 공개 문구가 시트 장부에 붙는다
    assert rec["matrix"] == qa().coverage(m)["matrix"]                         # 커버리지 매트릭스
    assert rec["n_planned_skipped"] == 20
    assert rec["wall_s"] >= 0
    assert rec["summary"] == {"n_jobs": 4, "ok": 4, "failed": 0,
                              "deltas": {"new": 4, "same": 0, "changed": 0, "failed": 0, "regressed": 0}}
    assert [e["id"] for e in rec["items"]] == ["in01", "in02", "in03", "in04"]
    assert [e["slot"] for e in rec["items"]] == [1, 2, 3, 4]


def test_day1_items_match_the_files_on_disk(two_days):
    d = two_days.res1.day_dir
    for e in M.read_record(d)["items"]:
        f = d / e["mp4"]
        assert e["mp4"] == f"{e['id']}_I1.mp4" and e["status"] == "ok" and e["error"] is None
        info = V.probe(str(f))
        assert (e["n_frames"], e["width"], e["height"]) == (info.n_frames, info.width, info.height) == (N, W, H)
        assert e["fps"] == pytest.approx(24.0, abs=1e-3) == pytest.approx(info.fps, abs=1e-3)
        assert e["mp4_sha256"] == sha_file(f)
        assert e["frames_hash"] == make_frames(e["id"]).content_hash()  # 렌더가 낸 Frames 의 내용 해시(PRD-01/FND-02)
        assert e["root"] == f"root-{e['id']}"
        assert e["seed"] == M.job_seed(3, e["id"], "I1")
        assert e["wall_s"] >= 0
        assert e["delta"] == {"status": "new", "mad": None, "psnr_db": None, "prev_date": None}


def test_day1_sheet_shows_todays_last_frame_beside_a_placeholder(two_days):
    img, geo = sheet_of(two_days.res1, 4)
    for i in range(4):
        cur = last_frame(two_days.res1.day_dir / f"in0{i + 1}_I1.mp4")
        assert np.array_equal(box(img, geo.thumb_box(i, "cur")), cur)
        assert flat(img, geo.thumb_box(i, "prev"), M.NO_PREV_RGB)
        x0, y0, _, _ = geo.label_box(i)
        assert tuple(img[y0, x0]) == tuple(M.STATUS_RGB["new"])
    assert tuple(img[0, 0]) == tuple(M.BG_RGB)


# ------------------------------------------------------------------------------------------------ 이튿날 (전일 대비)
def test_day2_statuses_deltas_and_isolation_of_failure(two_days):
    rec = M.read_record(two_days.res2.day_dir)
    assert rec["prev_date"] == D1 and rec["tag"] == "c2" and rec["complete"] is True
    assert two_days.r2.ids == ["in01", "in02", "in03", "in04"]  # in04 가 터져도 나머지는 다 돌았고 순서 유지
    assert [e["delta"]["status"] for e in rec["items"]] == ["same", "same", "changed", "regressed"]
    assert rec["summary"] == {"n_jobs": 4, "ok": 3, "failed": 1,
                              "deltas": {"new": 0, "same": 2, "changed": 1, "failed": 0, "regressed": 1}}
    assert two_days.res2.ok is False and two_days.res1.ok is True
    for e in rec["items"][:2]:
        assert e["delta"] == {"status": "same", "mad": 0.0, "psnr_db": M.PSNR_CAP_DB, "prev_date": D1}
    bad = rec["items"][3]
    assert bad["status"] == "failed" and bad["mp4"] is None and bad["frames_hash"] is None and bad["mp4_sha256"] is None
    assert bad["n_frames"] is None and bad["fps"] is None
    assert bad["error"].startswith("RuntimeError") and "boom: CUDA OOM" in bad["error"] and len(bad["error"]) <= 300
    assert bad["delta"] == {"status": "regressed", "mad": None, "psnr_db": None, "prev_date": D1}
    assert not (two_days.res2.day_dir / "in04_I1.mp4").exists()  # 실패하면 mp4 를 남기지 않는다(mp4 있음 <=> status ok)


def test_day2_changed_delta_matches_independent_oracle(two_days):
    rec = M.read_record(two_days.res2.day_dir)
    e = rec["items"][2]
    a = V.decode_frames(str(two_days.res1.day_dir / "in03_I1.mp4")).astype(np.float64)
    b = V.decode_frames(str(two_days.res2.day_dir / "in03_I1.mp4")).astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    d = e["delta"]
    assert d["status"] == "changed" and d["prev_date"] == D1
    assert d["mad"] == pytest.approx(float(np.abs(a - b).mean()), abs=1e-4)
    assert d["psnr_db"] == pytest.approx(10 * np.log10(255.0 ** 2 / mse), abs=1e-3)
    assert d["mad"] == pytest.approx(25.5, abs=2.0)  # 입력 오프셋 0.1
    assert e["frames_hash"] != M.read_record(two_days.res1.day_dir)["items"][2]["frames_hash"]


def test_day2_sheet_pairs_yesterday_and_today_and_marks_status(two_days):
    img, geo = sheet_of(two_days.res2, 4)
    want = ["same", "same", "changed", "regressed"]
    for i in range(4):
        iid = f"in0{i + 1}"
        prev = last_frame(two_days.res1.day_dir / f"{iid}_I1.mp4")
        assert np.array_equal(box(img, geo.thumb_box(i, "prev")), prev)  # 왼쪽 = 전일
        if want[i] == "regressed":
            assert flat(img, geo.thumb_box(i, "cur"), M.FAILED_RGB)
        else:
            cur = last_frame(two_days.res2.day_dir / f"{iid}_I1.mp4")
            assert np.array_equal(box(img, geo.thumb_box(i, "cur")), cur)   # 오른쪽 = 오늘
        x0, y0, _, _ = geo.label_box(i)
        assert tuple(img[y0, x0]) == tuple(M.STATUS_RGB[want[i]])
    # 바뀐 칸은 전일과 눈에 띄게 다르고 같은 칸은 정확히 같다
    p2, c2 = (box(img, geo.thumb_box(2, s)).astype(int) for s in ("prev", "cur"))
    p0, c0 = (box(img, geo.thumb_box(0, s)).astype(int) for s in ("prev", "cur"))
    assert np.abs(p2 - c2).mean() > 15 and np.array_equal(p0, c0)


def test_previous_day_is_never_modified(two_days):
    assert file_hashes(two_days.res1.day_dir) == two_days.snap1


def test_outputs_have_no_secrets_and_only_expected_file_types(two_days):
    for day in (two_days.res1.day_dir, two_days.res2.day_dir):
        for f in day.iterdir():
            assert f.suffix in (".mp4", ".json", ".png"), f
        txt = (day / "run.json").read_text(encoding="utf-8")
        assert not re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}", txt)


def test_canonical_digest_ignores_only_timings(two_days):
    rec = M.read_record(two_days.res1.day_dir)
    d0 = M.canonical_digest(rec)
    assert isinstance(d0, str) and re.fullmatch(r"[0-9a-f]{64}", d0)
    t = M.read_record(two_days.res1.day_dir)
    t["wall_s"] += 123.0
    for e in t["items"]:
        e["wall_s"] += 7.0
    assert M.canonical_digest(t) == d0
    for mutate in (lambda r: r["items"][0].__setitem__("frames_hash", "0" * 64),
                   lambda r: r["items"][1]["delta"].__setitem__("status", "changed"),
                   lambda r: r.__setitem__("tag", "other"),
                   lambda r: r["items"].pop()):
        u = M.read_record(two_days.res1.day_dir)
        mutate(u)
        assert M.canonical_digest(u) != d0
    assert M.read_record(two_days.res1.day_dir) == rec  # 입력 불변


# ------------------------------------------------------------------------------------------------ 첫날 실패 종류
def test_first_day_failures_are_isolated_and_recorded(tmp_path):
    m = make_manifest(3)
    r = FakeRender(fail={"in02": ValueError("x" * 1000)}, bad_type={"in03"})
    res = M.run_daily(m, r, tmp_path, D1, partial=True, thumb_size=THUMB)
    rec = res.record
    assert r.ids == ["in01", "in02", "in03"]
    assert [e["status"] for e in rec["items"]] == ["ok", "failed", "failed"]
    assert [e["delta"]["status"] for e in rec["items"]] == ["new", "failed", "failed"]  # 전일이 없으면 퇴행이 아니라 그냥 실패
    assert rec["items"][1]["error"].startswith("ValueError") and len(rec["items"][1]["error"]) <= 300
    assert rec["items"][2]["error"].startswith("TypeError")  # Frames 가 아닌 반환은 계약 위반
    assert rec["summary"]["ok"] == 1 and rec["summary"]["failed"] == 2 and rec["summary"]["deltas"]["failed"] == 2
    assert res.ok is False
    assert listing(res.day_dir) == ["in01_I1.mp4", "run.json", "sheet.png"]  # 파트·임시 파일 없음
    img, geo = sheet_of(res, 3)
    for i in (1, 2):
        assert flat(img, geo.thumb_box(i, "cur"), M.FAILED_RGB) and flat(img, geo.thumb_box(i, "prev"), M.NO_PREV_RGB)
        x0, y0, _, _ = geo.label_box(i)
        assert tuple(img[y0, x0]) == tuple(M.STATUS_RGB["failed"])


def test_rerun_without_resume_removes_stale_mp4_of_a_now_failing_item(tmp_path):
    m = make_manifest(2)
    M.run_daily(m, FakeRender(), tmp_path, D1, partial=True, thumb_size=THUMB)
    assert (tmp_path / D1 / "in02_I1.mp4").exists()
    res = M.run_daily(m, FakeRender(fail={"in02": RuntimeError("regressed")}), tmp_path, D1, partial=True,
                      resume=False, thumb_size=THUMB)
    assert listing(res.day_dir) == ["in01_I1.mp4", "run.json", "sheet.png"]  # 장부와 디스크가 어긋나지 않는다
    assert [e["status"] for e in M.read_record(res.day_dir)["items"]] == ["ok", "failed"]


def test_baseline_with_missing_mp4_is_treated_as_new(tmp_path):
    m = make_manifest(2)
    M.run_daily(m, FakeRender(), tmp_path, D1, partial=True, thumb_size=THUMB)
    (tmp_path / D1 / "in02_I1.mp4").unlink()  # 전일 산출물이 정리됐다
    res = M.run_daily(m, FakeRender(), tmp_path, D2, partial=True, thumb_size=THUMB)
    assert [e["delta"]["status"] for e in res.record["items"]] == ["same", "new"]
    img, geo = sheet_of(res, 2)
    assert flat(img, geo.thumb_box(1, "prev"), M.NO_PREV_RGB)


# ------------------------------------------------------------------------------------------------ 재개 · 중단 · 결정론
def test_interrupt_leaves_valid_partial_record_and_resume_finishes_identically(tmp_path):
    m = make_manifest(3)
    root, root_fresh = tmp_path / "a", tmp_path / "fresh"
    with pytest.raises(KeyboardInterrupt):
        M.run_daily(m, FakeRender(interrupt_at=3), root, D1, partial=True, tag="t", thumb_size=THUMB)
    part = M.read_record(root / D1)  # 끊긴 장부도 읽힌다(작업마다 원자적으로 갱신)
    assert part["complete"] is False and [e["id"] for e in part["items"]] == ["in01", "in02"]
    names = listing(root / D1)
    assert "in01_I1.mp4" in names and "in02_I1.mp4" in names and "in03_I1.mp4" not in names
    assert not [n for n in names if ".part" in n or n.startswith(".tmp")]
    # 끊긴 날은 전일 기준선이 되지 못한다
    nxt = M.run_daily(m, FakeRender(), root, D2, partial=True, tag="t", thumb_size=THUMB)
    assert nxt.record["prev_date"] is None and {e["delta"]["status"] for e in nxt.record["items"]} == {"new"}
    # 재개: 끝난 두 개는 건너뛰고 마지막 하나만 렌더
    r = FakeRender()
    res = M.run_daily(m, r, root, D1, partial=True, tag="t", thumb_size=THUMB)
    assert r.ids == ["in03"] and res.reused == ("in01", "in02") and res.record["complete"] is True
    fresh = M.run_daily(m, FakeRender(), root_fresh, D1, partial=True, tag="t", thumb_size=THUMB)
    assert M.canonical_digest(res.record) == M.canonical_digest(fresh.record)  # 재개 결과 = 처음부터 돈 결과
    ia, _ = sheet_of(res, 3)
    ib, _ = sheet_of(fresh, 3)
    assert np.array_equal(ia, ib)
    for k in (1, 2, 3):
        assert sha_file(root / D1 / f"in0{k}_I1.mp4") == sha_file(root_fresh / D1 / f"in0{k}_I1.mp4")  # libx264 결정론


def test_resume_reuses_only_what_is_still_valid(tmp_path):
    m = make_manifest(3)
    kw = dict(partial=True, tag="t", base_seed=1, thumb_size=THUMB)
    first = M.run_daily(m, FakeRender(), tmp_path, D1, **kw)
    digest = M.canonical_digest(first.record)
    r = FakeRender()
    again = M.run_daily(m, r, tmp_path, D1, **kw)
    assert r.calls == [] and again.reused == ("in01", "in02", "in03") and M.canonical_digest(again.record) == digest
    for change in (dict(tag="u"), dict(base_seed=2), dict(resume=False)):  # 코드 식별자·시드가 다르면 재사용 금지
        r = FakeRender()
        out = M.run_daily(m, r, tmp_path, D1, **{**kw, **change})
        assert r.ids == ["in01", "in02", "in03"] and out.reused == ()
    # 마지막 반복(resume=False, 같은 tag/seed)이 장부를 kw 설정으로 다시 썼다 -> 이제 파일 하나를 훼손한다
    kw2 = kw
    with open(tmp_path / D1 / "in02_I1.mp4", "ab") as f:  # 파일이 훼손되면 그것만 다시
        f.write(b"\0")
    r = FakeRender()
    fixed = M.run_daily(m, r, tmp_path, D1, **kw2)
    assert r.ids == ["in02"] and fixed.reused == ("in01", "in03")
    assert sha_file(tmp_path / D1 / "in02_I1.mp4") == fixed.record["items"][1]["mp4_sha256"]


def test_two_independent_runs_are_identical_except_timings(tmp_path):
    m = make_manifest(3)
    outs = []
    for name in ("x", "y"):
        base = tmp_path / name
        M.run_daily(m, FakeRender(), base, D1, partial=True, thumb_size=THUMB)
        outs.append(M.run_daily(m, FakeRender(offsets={"in02": 0.1}), base, D2, partial=True, thumb_size=THUMB))
    a, b = outs
    assert M.canonical_digest(a.record) == M.canonical_digest(b.record)
    assert np.array_equal(sheet_of(a, 3)[0], sheet_of(b, 3)[0])


# ------------------------------------------------------------------------------------------------ 24장 전부
def test_full_24_item_night_run(tmp_path):
    m = make_manifest(24)
    t0 = time.time()
    res = M.run_daily(m, FakeRender(with_root=False), tmp_path, D1, thumb_size=THUMB)  # Frames 를 그대로 반환해도 된다
    assert time.time() - t0 < 50  # 야간 잡 오버헤드가 합성 클립 24개에 수십 초를 먹지 않는다
    rec = res.record
    assert rec["n_planned_skipped"] == 0 and rec["summary"]["n_jobs"] == 24 and res.ok
    assert [e["slot"] for e in rec["items"]] == list(range(1, 25))
    assert all(e["root"] is None and e["status"] == "ok" for e in rec["items"])
    assert len({e["frames_hash"] for e in rec["items"]}) == 24  # 항목마다 다른 클립
    assert len([n for n in listing(res.day_dir) if n.endswith(".mp4")]) == 24
    img, geo = sheet_of(res, 24)
    assert tuple(geo.size) == (560, 460) and img.shape == (460, 560, 3)
    for i in range(24):
        cur = last_frame(res.day_dir / f"in{i + 1:02d}_I1.mp4")
        assert np.array_equal(box(img, geo.thumb_box(i, "cur")), cur)
        x0, y0, _, _ = geo.label_box(i)
        assert tuple(img[y0, x0]) == tuple(M.STATUS_RGB["new"])


def test_thumbnails_are_resized_when_the_clip_is_larger(tmp_path):
    m = make_manifest(2)
    res = M.run_daily(m, FakeRender(), tmp_path, D1, partial=True, thumb_size=(32, 24), cols=1)
    img, geo = sheet_of(res, 2, thumb=(32, 24), cols=1)
    assert geo.cols == 1 and geo.rows == 2
    full = last_frame(res.day_dir / "in02_I1.mp4").astype(np.float64)
    ref = full.reshape(24, 2, 32, 2, 3).mean(axis=(1, 3))
    got = box(img, geo.thumb_box(1, "cur")).astype(np.float64)
    assert got.shape == (24, 32, 3) and np.abs(got - ref).mean() < 3.0  # 면적 평균과 거의 같다(이중선형 축소)


# ------------------------------------------------------------------------------------------------ 입력 검증 · 준비 안 된 매니페스트
def test_planned_items_without_partial_fail_before_touching_disk(tmp_path):
    r = FakeRender()
    with pytest.raises(M.DailyABError):
        M.run_daily(make_manifest(4), r, tmp_path / "out", D1)
    assert r.calls == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", ["2026-13-01", "2026-02-30", "10/10/2026", "2026-1-1", "20261010", "", None, 20261010])
def test_bad_dates_rejected_before_touching_disk(tmp_path, bad):
    with pytest.raises(ValueError):
        M.run_daily(make_manifest(2), FakeRender(), tmp_path / "out", bad, partial=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kw", [dict(thumb_size=(0, 10)), dict(thumb_size=(10,)), dict(cols=0), dict(move="I9")])
def test_bad_arguments_rejected_before_touching_disk(tmp_path, kw):
    with pytest.raises((ValueError, TypeError)):
        M.run_daily(make_manifest(2), FakeRender(), tmp_path / "out", D1, partial=True, **kw)
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------------ GPU 러너
@pytest.mark.gpu
def test_nvenc_encoder_path_on_gpu_runner(tmp_path):
    try:
        V.select_encoder("h264", "gpu")
    except V.VideoError:
        pytest.skip("NVENC 사용 불가")
    res = M.run_daily(make_manifest(2), FakeRender(), tmp_path, D1, partial=True, encoder="gpu", thumb_size=THUMB)
    assert res.ok and [e["n_frames"] for e in res.record["items"]] == [N, N]
