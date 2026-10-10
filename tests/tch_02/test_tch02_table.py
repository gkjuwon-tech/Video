"""TCH-02 σ 표: 기본값 수치, 스냅 규칙, 창, 스케줄, 키, 오류, import 안전성."""
import dataclasses
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from buc.core import det
from buc.teacher import sample as S
from buc.teacher.zimage import TeacherSpecError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = (1.0, 21 / 22, 0.9, 5 / 6, 0.75, 9 / 14, 0.5, 0.3)


def test_default_table_matches_shift3_flow_schedule():
    t = S.DEFAULT_TABLE
    assert len(t) == 8 == S.N_STEPS and S.FLOW_SHIFT == 3.0
    assert t.points == pytest.approx(DEFAULT, abs=1e-12)
    assert all(a > b for a, b in zip(t.points, t.points[1:]))
    assert S.SigmaTable.from_flow_shift().points == t.points


def test_from_flow_shift_other_parameters():
    lin = S.SigmaTable.from_flow_shift(4, 1.0)           # shift=1 -> 선형
    assert lin.points == pytest.approx((1.0, 0.75, 0.5, 0.25), abs=1e-12)
    t = S.SigmaTable.from_flow_shift(5, 3.0)
    u = [1 - i / 5 for i in range(5)]
    assert t.points == pytest.approx(tuple(3 * x / (1 + 2 * x) for x in u), abs=1e-12)
    assert t.points[0] == 1.0


@pytest.mark.parametrize("pts", [(), (0.5, 0.5), (0.3, 0.5), (1.2, 0.5), (1.0, 0.0), (1.0, -0.1),
                                 (1.0, float("nan")), (float("inf"),)])
def test_table_validation(pts):
    with pytest.raises(S.SigmaError):
        S.SigmaTable(pts)


def test_sigma_error_hierarchy():
    assert issubclass(S.SigmaError, TeacherSpecError) and issubclass(S.SigmaError, ValueError)


def test_table_is_frozen_dataclass():
    assert dataclasses.is_dataclass(S.DEFAULT_TABLE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        S.DEFAULT_TABLE.points = (1.0,)


@pytest.mark.parametrize("req,idx", [(1.0, 0), (0.99, 0), (0.96, 1), (0.9, 2), (0.8, 3), (0.76, 4), (0.7, 4),
                                     (0.65, 5), (0.55, 6), (0.5, 6), (0.35, 7), (0.25, 7), (0.2, 7),
                                     (0.01, 7)])
def test_snap_values(req, idx):
    s = S.DEFAULT_TABLE.snap(req)
    assert s.index == idx
    assert s.sigma == pytest.approx(DEFAULT[idx], abs=1e-12)
    assert s.requested == req
    assert S.snap_sigma(req).index == idx                   # 모듈 함수는 기본 표


def test_snap_pipeline_anchor_points_land_on_table():
    # 파이프라인이 쓰는 요청값(.2/.25/.35/.6)은 전부 표의 점으로 나온다.
    for req in (0.2, 0.25, 0.35, 0.6):
        assert S.DEFAULT_TABLE.snap(req).sigma in S.DEFAULT_TABLE.points


def test_snap_matches_bruteforce_nearest_with_ties_to_higher_sigma():
    t = S.DEFAULT_TABLE
    p = t.points
    for i in range(1, 201):
        s = i / 200
        d = sorted((abs(p[j] - s), j) for j in range(len(p)))
        if d[1][0] - d[0][0] < 1e-9:
            continue                                        # 동률 근방은 아래 전용 테스트가 다룬다
        want = d[0][1]
        got = t.snap(s)
        assert got.index == want, s
        assert got.sigma == p[want]


def test_snap_tie_goes_to_larger_sigma():
    t = S.SigmaTable((1.0, 0.5, 0.25))
    assert t.snap(0.375).index == 1                         # |0.375-0.5| == |0.375-0.25| 정확히 동률
    assert t.snap(0.75).index == 0                          # 0.75: 1.0과 0.5 동률
    assert t.snap(0.3).index == 2


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0000001, 2.0, float("nan"), float("inf"), -float("inf")])
def test_snap_rejects_bad_sigma(bad):
    with pytest.raises(S.SigmaError):
        S.DEFAULT_TABLE.snap(bad)


def test_index_of_exact_points_only():
    t = S.DEFAULT_TABLE
    for i, p in enumerate(t.points):
        assert t.index_of(p) == i
        assert t.index_of(p - 5e-10) == i                   # tol=1e-9
    for bad in (0.31, 0.25, 0.0, 1.5, float("nan")):
        with pytest.raises(S.SigmaError):
            t.index_of(bad)
    with pytest.raises(S.SigmaError):
        t.index_of(0.3 + 1e-6)
    assert t.index_of(0.3 + 1e-6, tol=1e-5) == 7


def test_window_inclusive_and_descending():
    t = S.SigmaTable((1.0, 0.5, 0.25))
    assert t.window(0.25, 0.5) == (1, 2)
    assert t.window(0.0 + 1e-9, 1.0) == (0, 1, 2)
    d = S.DEFAULT_TABLE
    assert d.window(0.29, 0.76) == (4, 5, 6, 7)
    assert d.window(0.45, 0.55) == (6,)


def test_window_empty_falls_back_to_snap_of_midpoint():
    d = S.DEFAULT_TABLE
    assert d.window(0.32, 0.46) == (7,)                     # 중앙 0.39 -> 0.3
    assert d.window(0.52, 0.62) == (6,)                     # 중앙 0.57 -> 0.5
    with pytest.raises(S.SigmaError):
        d.window(0.6, 0.4)


def test_schedule_pairs_end_at_zero():
    d = S.DEFAULT_TABLE
    assert d.schedule(5, 2) == ((d.points[5], d.points[6]), (d.points[6], d.points[7]))
    assert d.schedule(6, 2) == ((d.points[6], d.points[7]), (d.points[7], 0.0))
    assert d.schedule(7, 1) == ((d.points[7], 0.0),)
    full = d.schedule(0, 8)
    assert len(full) == 8 and full[0][0] == 1.0 and full[-1][1] == 0.0
    assert [a for a, _ in full] == list(d.points)
    assert [b for _, b in full[:-1]] == list(d.points[1:])


@pytest.mark.parametrize("start,steps", [(8, 1), (-1, 1), (0, 0), (0, -1), (0, 9), (7, 2), (5, 4), (0.5, 1), (0, 1.5)])
def test_schedule_rejects_out_of_range(start, steps):
    with pytest.raises(S.SigmaError):
        S.DEFAULT_TABLE.schedule(start, steps)


def test_table_key_recipe_and_sensitivity():
    t = S.DEFAULT_TABLE
    want = det.hash_parts("buc.tch02.table.v1", t.name, *[struct.pack("<d", p) for p in t.points]).hex()
    assert t.key == want and len(want) == 64
    assert S.SigmaTable(t.points, name=t.name).key == t.key                              # 같은 값 -> 같은 키
    assert S.SigmaTable(t.points, name=t.name + "x").key != t.key                # 이름이 키에 든다
    bumped = (t.points[0],) + tuple(p - 1e-9 for p in t.points[1:])
    assert S.SigmaTable(bumped, name=t.name).key != t.key                                 # 비트 하나 바뀌면 다른 키


def test_import_is_light_and_side_effect_free():
    code = ("import sys, torch; a=torch.are_deterministic_algorithms_enabled(); import buc.teacher.sample as s;"
            "assert 'diffusers' not in sys.modules and 'transformers' not in sys.modules;"
            "assert 'huggingface_hub' not in sys.modules;"
            "assert torch.are_deterministic_algorithms_enabled()==a;"
            "assert not torch.cuda.is_initialized();"
            "print('ok', s.TeacherSampler.__name__)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr
    assert "ok TeacherSampler" in r.stdout


def test_math_helpers_use_python_floats():
    s = S.DEFAULT_TABLE.snap(0.25)
    assert isinstance(s.index, int) and isinstance(s.sigma, float) and not math.isnan(s.sigma)
