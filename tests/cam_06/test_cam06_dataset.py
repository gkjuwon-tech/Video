"""CAM-06: 증강 ×5.5 데이터셋 구성(예산 · 순서 · 출처 · 결정론) · 윈도 · 모듈 위생."""
import copy
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cam06_helpers import imu, make_clip

RATE = 200.0
REPO = Path(__file__).resolve().parents[2]
PUBLIC = ["RATE_HZ", "AUG_FACTOR", "G_STD", "MAX_GAP_S", "WINDOW_LEN", "GRAVITY_LP_HZ", "RIG_KINDS", "Series", "RawImu", "ImuClip",
          "parse_gpmf", "parse_phone_csv", "merge_raw", "split_segments", "resample_uniform", "rotate_frame", "shift_clip",
          "estimate_offset", "AlignEstimate", "align_pair", "gravity_estimate", "warped_length", "time_warp", "rig_filter",
          "perturb_sensor", "AugmentConfig", "sample_augmentation", "apply_augmentation", "augment_clip", "augment_dataset",
          "make_windows"]


def budget(n, factor=5.5):
    return int(math.floor((factor - 1.0) * n + 0.5))


def _clips(lengths, **meta):
    return [make_clip(n, seed=10 + i, t0=100.0 * i, meta={"dataset": "TUM VI", "license": "CC-BY-4.0", **meta}, device=f"d{i}")
            for i, n in enumerate(lengths)]


def _group(outs):
    by = {}
    for o in outs:
        by.setdefault(o.meta["origin"], []).append(o)
    return by


# ------------------------------------------------------------------ 예산 · 순서
def test_dataset_budget_is_exactly_five_and_a_half_times():
    m = imu()
    lengths = [2000, 3333, 777]
    clips = _clips(lengths)
    outs = m.augment_dataset(clips, seed=3)
    n = len(clips)
    assert [o.meta["variant"] for o in outs[:n]] == [0] * n and [o.meta["origin"] for o in outs[:n]] == list(range(n))   # 원본이 먼저
    for i in range(n):
        assert outs[i].meta["aug"] is None
        assert np.array_equal(outs[i].gyro, clips[i].gyro) and np.array_equal(outs[i].accel, clips[i].accel) and outs[i].t0 == clips[i].t0
    key = [(o.meta["origin"], o.meta["variant"]) for o in outs[n:]]
    assert key == sorted(key)                                                                            # 출처 순, 변형 번호 순
    by = _group(outs[n:])
    for i, N in enumerate(lengths):
        vs = by[i]
        assert [v.meta["variant"] for v in vs] == list(range(1, len(vs) + 1))
        assert sum(v.n for v in vs) == budget(N)                                                         # 변형 길이 합 = round(4.5 N)
        assert 4 <= len(vs) <= 7
        for v in vs[:-1]:
            assert v.n == m.warped_length(N, v.meta["aug"]["time_scale"]) and v.meta["aug"]["crop"] is None   # 마지막 빼고 전체
        last = vs[-1]
        wl = m.warped_length(N, last.meta["aug"]["time_scale"])
        if last.meta["aug"]["crop"] is not None:
            st, ln = last.meta["aug"]["crop"]
            assert ln == last.n and 0 <= st and st + ln <= wl
        else:
            assert last.n == wl
        assert all(v.rate_hz == RATE for v in vs)
    assert sum(o.n for o in outs) == sum(lengths) + sum(budget(N) for N in lengths)
    assert sum(o.n for o in outs) / sum(lengths) == pytest.approx(5.5, abs=2e-3)


def test_dataset_every_variant_is_reproducible_from_its_recorded_params():
    m = imu()
    clips = _clips([1800, 2500])
    outs = m.augment_dataset(clips, seed=11)
    for o in outs[2:]:
        src = clips[o.meta["origin"]]
        again = m.apply_augmentation(src, o.meta["aug"])
        assert np.array_equal(again.gyro, o.gyro) and np.array_equal(again.accel, o.accel)
        assert again.t0 == o.t0 and again.rate_hz == o.rate_hz
        assert o.source == src.source and o.device == src.device
        assert o.meta["dataset"] == "TUM VI" and o.meta["license"] == "CC-BY-4.0"                     # 출처·라이선스 꼬리표가 따라간다


def test_dataset_variants_differ_and_respect_config():
    m = imu()
    cfg = m.AugmentConfig(time_scale=(0.95, 1.05), rig_prob=1.0, rig_kinds=("lowpass",), rig_fn_hz=(5.0, 6.0), rot_deg_max=10.0)
    outs = m.augment_dataset(_clips([4000]), cfg=cfg, seed=2)[1:]
    augs = [o.meta["aug"] for o in outs]
    assert len({a["noise"]["seed"] for a in augs}) == len(augs)
    assert len({tuple(a["rotvec"]) for a in augs}) == len(augs)
    for a in augs:
        assert 0.95 <= a["time_scale"] <= 1.05 and a["rig"]["kind"] == "lowpass" and 5.0 <= a["rig"]["fn_hz"] <= 6.0
        assert np.linalg.norm(a["rotvec"]) <= math.radians(10.0) + 1e-9
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            k = min(outs[i].n, outs[j].n)
            assert not np.allclose(outs[i].gyro[:k], outs[j].gyro[:k])


def test_dataset_identity_config_gives_four_full_copies_and_a_half_crop():
    m = imu()
    cfg = m.AugmentConfig(rot_deg_max=0.0, time_scale=(1.0, 1.0), rig_prob=0.0, gyro_sigma=(0.0, 0.0), accel_sigma=(0.0, 0.0),
                          gyro_bias_max=0.0, accel_bias_max=0.0)
    c = make_clip(2000, seed=5)
    outs = m.augment_dataset([c], cfg=cfg, seed=1)
    assert [o.n for o in outs] == [2000, 2000, 2000, 2000, 2000, 1000]
    for o in outs[1:5]:
        assert np.max(np.abs(o.gyro - c.gyro)) < 1e-9 and np.max(np.abs(o.accel - c.accel)) < 1e-9
    st, ln = outs[5].meta["aug"]["crop"]
    assert ln == 1000 and 0 <= st <= 1000
    assert np.max(np.abs(outs[5].gyro - c.gyro[st:st + 1000])) < 1e-9
    assert outs[5].t0 == pytest.approx(c.t0 + st / RATE, abs=1e-9)


def test_dataset_factor_edge_cases():
    m = imu()
    clips = _clips([1000, 1200])
    one = m.augment_dataset(clips, factor=1.0)
    assert len(one) == 2 and all(o.meta["variant"] == 0 for o in one)
    three = m.augment_dataset(clips, factor=3.0, seed=4)
    for i, N in enumerate((1000, 1200)):
        assert sum(v.n for v in _group(three[2:])[i]) == budget(N, 3.0) == 2 * N
    assert m.augment_dataset([], seed=1) == []
    tiny = m.augment_dataset([make_clip(7)], seed=1)
    assert sum(o.n for o in tiny[1:]) == budget(7)
    for bad in (0.99, 0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.augment_dataset(clips, factor=bad)
    assert m.AUG_FACTOR == 5.5 and sum(o.n for o in m.augment_dataset(clips)) == sum(
        c.n + budget(c.n) for c in clips)                                                              # 기본 factor = AUG_FACTOR


# ------------------------------------------------------------------ 결정론 · 부작용
def test_dataset_determinism_and_no_side_effects():
    m = imu()
    clips = _clips([1500, 2100])
    snap = [(c.gyro.copy(), c.accel.copy(), copy.deepcopy(c.meta)) for c in clips]
    st = np.random.get_state()[1].copy()
    a = m.augment_dataset(clips, seed=9)
    b = m.augment_dataset(clips, seed=9)
    c = m.augment_dataset(clips, seed=10)
    assert np.array_equal(np.random.get_state()[1], st)                                                  # 전역 RNG 불변
    assert len(a) == len(b) and all(np.array_equal(x.gyro, y.gyro) and np.array_equal(x.accel, y.accel) and x.meta == y.meta
                                    for x, y in zip(a, b))                                               # 비트 동일
    assert [o.meta["aug"] for o in a[2:]] != [o.meta["aug"] for o in c[2:]]                              # 시드가 다르면 증강도 다르다
    for cl, (g, ac, mt) in zip(clips, snap):
        assert np.array_equal(cl.gyro, g) and np.array_equal(cl.accel, ac) and cl.meta == mt            # 입력 불변
    # 한 클립의 변형은 다른 클립이 있든 없든 같다(출처 번호만 같으면): 클립 순서 바꿔 넣는 실수를 막는 앵커
    alone = m.augment_dataset(clips[:1], seed=9)
    mine = [o for o in a if o.meta["origin"] == 0 and o.meta["variant"] > 0]
    assert len(alone) - 1 == len(mine)
    assert all(np.array_equal(x.gyro, y.gyro) for x, y in zip(alone[1:], mine))


# ------------------------------------------------------------------ 윈도 (MDL-17 이음매)
def test_make_windows_shapes_and_content():
    m = imu()
    c = make_clip(1000, seed=6)
    w = m.make_windows(c)                                                                                # 길이 400, hop 200
    assert w.dtype == np.float32 and w.shape == (4, 400, 6)
    arr = c.as_array()
    for k in range(4):
        assert np.allclose(w[k], arr[k * 200:k * 200 + 400].astype(np.float32), atol=0, rtol=0)
    w2 = m.make_windows(c, length=100, hop=300)
    assert w2.shape == ((1000 - 100) // 300 + 1, 100, 6) and np.array_equal(w2[2], arr[600:700].astype(np.float32))
    assert m.make_windows(c, length=1000).shape == (1, 1000, 6) and m.make_windows(c, length=1000, hop=1).shape == (1, 1000, 6)
    assert m.make_windows(make_clip(399)).shape == (0, 400, 6)
    assert m.WINDOW_LEN == 400
    for kw in (dict(length=0), dict(hop=0), dict(length=-5), dict(hop=-1)):
        with pytest.raises(ValueError):
            m.make_windows(c, **kw)


# ------------------------------------------------------------------ 모듈 위생
def test_public_api_is_exported():
    m = imu()
    assert set(PUBLIC) <= set(m.__all__)
    for name in PUBLIC:
        assert hasattr(m, name), name


def test_module_is_torch_free_and_import_has_no_side_effects():
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "st = np.random.get_state()[1].copy()\n"
        "import importlib\n"
        "m = importlib.import_module('buc.camera.imu')\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert np.array_equal(np.random.get_state()[1], st), 'global RNG touched'\n"
        "print('ok')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), p.stderr[-800:]
