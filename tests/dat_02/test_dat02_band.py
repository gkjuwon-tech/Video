"""결정 곡선 대역 마스크 + 저대역 판정의 의미 검사 (무노이즈 체커 교사).

무노이즈(ε=0) 가짜 교사는 x̂0 = x0 + γ·δ·checker(잠재) 를 낸다. 블록 상수 영상은 가짜 VAE 왕복이 정확하므로
x′ - x 는 주기 16px 정사각파뿐이고 옥타브 ≤3 에만 에너지가 있다(64x64, n_oct=6)."""
import numpy as np
import pytest

import dat02_fakes as F
from buc.data import realify as R


def _items(n=2):
    return [R.RealifyItem(key=f"b{i}", image=F.block_image(40 + i), prov=F.ENGINE, caption="room") for i in range(n)]


def _const_curve(sampler, row):
    return R.DecisionCurve(sigmas=sampler.table.points, D=[list(row)] * len(sampler.table), table_key=sampler.table.key,
                           name="test")


def _run(row, tau, sigma=0.5):
    t, s, bus = F.make_rig(zero_noise=True)
    its = _items()
    res = R.realify(s, bus, its, curve=_const_curve(s, row), sigma=sigma, seed=3, token_offset=F.OFF, tau=tau)
    return its, res


def test_difference_is_confined_to_fine_octaves():
    its, res = _run([0.5, 0.5, 0.5, 0.5, 0.0, 0.0], tau=1e-4)
    for it, r in zip(its, res):
        d = (r.image.astype(np.float64) - it.image).transpose(2, 0, 1)
        assert float(np.abs(d).max()) > 1e-3                        # 실제로 바뀌었다
        coarse = F.oracle_band_project(d, [False] * 4 + [True] * 2)
        assert float(np.abs(coarse).max()) < 1e-5                   # 옥타브 4,5 는 보존
        assert r.band_valid == (False, False, False, False, True, True)
        assert r.lowband_err < 1e-5
        assert r.keep is True and r.reject_reason is None


def test_octave3_inclusive_threshold_triggers_lowband_reject():
    its, res = _run([0.5, 0.5, 0.5, 0.1, 0.0, 0.0], tau=1e-4)     # D=0.1 은 유효(경계 포함)
    for it, r in zip(its, res):
        assert r.band_valid == (False, False, False, True, True, True)
        want = F.oracle_lowband_err(r.image, it.image, r.band_valid)
        assert want > 1e-3
        assert r.lowband_err == pytest.approx(want, rel=1e-6)
        assert r.keep is False and r.reject_reason == "lowband"


def test_tau_boundary_keeps_at_equality():
    its, res = _run([0.5, 0.5, 0.5, 0.1, 0.0, 0.0], tau=10.0)
    err = res[0].lowband_err
    assert res[0].keep
    _, res2 = _run([0.5, 0.5, 0.5, 0.1, 0.0, 0.0], tau=err)          # err ≤ tau → keep
    assert res2[0].keep and res2[0].lowband_err == err
    _, res3 = _run([0.5, 0.5, 0.5, 0.1, 0.0, 0.0], tau=err * (1 - 1e-6))
    assert not res3[0].keep and res3[0].reject_reason == "lowband"


def test_no_valid_band_rejects():
    its, res = _run([1.0] * 6, tau=10.0)
    for r in res:
        assert r.band_valid == (False,) * 6
        assert r.lowband_err == 0.0
        assert r.keep is False and r.reject_reason == "no_band"


def test_d_thresh_parameter():
    t, s, bus = F.make_rig(zero_noise=True)
    c = _const_curve(s, [0.5, 0.5, 0.5, 0.3, 0.2, 0.0])
    r = R.realify(s, bus, _items(1), curve=c, sigma=0.5, seed=3, token_offset=F.OFF, tau=1.0, d_thresh=0.25)[0]
    assert r.band_valid == (False, False, False, False, True, True)
    r = R.realify(s, bus, _items(1), curve=c, sigma=0.5, seed=3, token_offset=F.OFF, tau=1.0)[0]
    assert r.band_valid == (False, False, False, False, False, True)


def test_mask_uses_snapped_start_sigma():
    t, s, bus = F.make_rig(zero_noise=True)
    rows = [[1.0] * 6 for _ in s.table.points]
    i5 = s.table.index_of(9 / 14)
    rows[i5] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    rows[i5 + 1] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]                  # tap σ=0.5 행과 구분
    c = R.DecisionCurve(sigmas=s.table.points, D=rows, table_key=s.table.key)
    r = R.realify(s, bus, _items(1), curve=c, sigma=0.62, steps=2, seed=3, token_offset=F.OFF, tau=1.0)[0]
    assert r.sigma == pytest.approx(9 / 14) and r.tap_sigma == 0.5
    assert r.band_valid == (False, False, False, False, False, True)
