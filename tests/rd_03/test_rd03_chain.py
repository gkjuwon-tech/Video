"""전 스택 통합: LNS-04 체인(비네팅·할레이션·그레인·센서 잡음·12비트 ADC)과 횡색수차를 입힌 영상에서 R2 를 한꺼번에 읽는다.
각 효과가 서로를 오염시키지 않는지, to_meta/theta_fields 이음매, LNS-08 순방향으로 되돌려 본 자기 일관, 실행 시간."""
import json
import time
from functools import lru_cache

import numpy as np
import pytest

import rd03_helpers as H
from buc.lens import chain
from buc.lens import sensor_film as sf
from buc.read import screen_fx as fx

HH, WW = H.H0, H.W0
HD = H.half_diag(HH, WW)
A_HAL, S_VIG, S_GRAIN = 0.2, 0.7, 0.5
Y_SRC = 2.2
DISC = (150.0, 90.0, 9.0)
CA_PX = (0.7, -0.45)


@lru_cache(maxsize=None)
def case(seed):
    x = H.mixed_scene(seed=seed)
    cx, cy, r = DISC
    x = x.copy()
    x[int(cy) - 60:int(cy) + 60, int(cx) - 60:int(cx) + 60] = 0.25            # 조용한 하늘 같은 구역(평균 밝기)
    cov = H.disc_cov(HH, WW, cx, cy, r)[..., None]
    x = x * (1.0 - cov) + Y_SRC * cov
    x_ca = H.with_ca(x, CA_PX[0] / HD, CA_PX[1] / HD)
    th = H.theta(grain=S_GRAIN, vignette=S_VIG, halation=A_HAL, lut="none")
    y = H.full_chain(x_ca, th, seed=seed, noise=True)
    t0 = time.perf_counter()
    res = fx.estimate_screen_fx(y)
    return x_ca.astype(np.float32), th, y, res, time.perf_counter() - t0


@pytest.mark.parametrize("seed", [0, 3])
def test_all_effects_recovered_together(seed):
    x, th, y, r, _ = case(seed)
    # 비네팅
    assert abs(float(r.vignette.gain_at(1.0)) - (1.0 - 0.8 * S_VIG)) <= 0.15
    assert abs(r.vignette.strength - S_VIG) <= 0.2
    # 그레인(LNS-04 체인의 상관 반경은 max(0.75, 0.0012 W) px)
    assert abs(r.grain.strength - S_GRAIN) <= 0.15
    assert 0.4 <= r.grain.radius_px <= 1.5
    assert r.grain.nonphysical is False
    # 횡색수차
    assert abs(r.ca.shift_corner_px[0] - CA_PX[0]) <= 0.3 and abs(r.ca.shift_corner_px[1] - CA_PX[1]) <= 0.3
    # 할레이션: 포화 광원 하나, 헤일로 곱 P = amount·(Y·V(ρ)-0.8)  (할레이션은 비네팅 뒤에 더해진다)
    h = r.halation
    rho = np.hypot(DISC[0] - (WW - 1) / 2.0, DISC[1] - (HH - 1) / 2.0) / HD
    p_true = A_HAL * (Y_SRC * (1.0 - 0.8 * S_VIG * rho * rho) - H.HALATION_THRESHOLD)
    assert h.detected is True and h.n_blobs == 1
    assert 0.7 * p_true <= h.halo_gain <= 1.3 * p_true
    assert 0.012 <= h.sigma_frac <= 0.03
    assert 0.2 <= h.tint[1] <= 0.5 and h.tint[2] <= 0.25
    assert abs(h.blobs[0].x - DISC[0]) <= 2.0 and abs(h.blobs[0].y - DISC[1]) <= 2.0
    # 고스트: 광원 위치는 맞다(질감 바탕에서 반점 유무는 식별성이 낮아 단정하지 않는다)
    assert r.ghost.source is not None
    assert abs(r.ghost.source[0] - DISC[0]) <= 2.0 and abs(r.ghost.source[1] - DISC[1]) <= 2.0
    assert r.reliable is True


@pytest.mark.parametrize("seed", [0])
def test_screen_fx_equals_individual_estimators(seed):
    _, _, y, r, _ = case(seed)
    assert r.vignette == fx.estimate_vignette(y)
    assert r.grain == fx.estimate_grain(y)
    assert r.halation == fx.estimate_halation(y)
    assert r.ghost == fx.estimate_ghost(y)
    assert r.size == (HH, WW)


def test_theta_fields_and_meta_roundtrip():
    _, _, _, r, _ = case(0)
    tf = r.theta_fields()
    assert abs(tf["vignette"] - S_VIG) <= 0.2
    assert abs(tf["grain"] - S_GRAIN) <= 0.15
    # 광원 휘도 ≈ 가정(2.0): amount ≈ A_HAL (±35 %)
    assert abs(tf["halation"] - A_HAL) <= 0.35 * A_HAL + 0.02
    m = r.to_meta()
    s = json.dumps(m, allow_nan=False)
    assert json.loads(s) == m
    assert m["halation"]["detected"] is True and m["ghost"]["n"] == len(r.ghost.ghosts)
    assert m["warnings"] == sorted(set(m["warnings"]))


def test_self_consistency_with_lns08_forward():
    """θ̂ 로 같은 세계를 LNS-08 순방향에 태우면 입력과 저역에서 맞는다(r_look 의 전제). 기본 θ 보다 훨씬 가깝다."""
    x, th, y, r, _ = case(0)
    from buc.core import contracts as C
    th_hat = C.Theta(lut="none", **r.theta_fields())
    ref = chain.look(x, th, noise=False)
    got = chain.look(x, th_hat, noise=False)
    base = chain.look(x, C.Theta(lut="none"), noise=False)
    err = float(np.abs(got.astype(np.float64) - ref).mean())
    err0 = float(np.abs(base.astype(np.float64) - ref).mean())
    assert err <= 0.03
    assert err <= 0.5 * err0


def test_division_by_estimated_gain_matches_noise_free_world():
    """ExplainAway 이음매: 영상/gain 이 비네팅 없는 같은 체인 영상과 중간 대역에서 맞는다."""
    x, th, y, r, _ = case(0)
    th_nov = H.theta(grain=0.0, vignette=0.0, halation=A_HAL, lut="none")
    clean = sf.apply_sensor_film(x, th_nov, noise=False).astype(np.float64)
    from buc.core import color
    clean_lin = color.srgb_to_linear(clean.astype(np.float32)).astype(np.float64)
    flat = y.astype(np.float64) / r.vignette.gain()[..., None].astype(np.float64)
    yy, xx = np.mgrid[0:HH, 0:WW]
    rr = np.hypot(xx - (WW - 1) / 2, yy - (HH - 1) / 2) / HD
    ring = rr > 0.7
    ratio = flat[ring].mean() / clean_lin[ring].mean()
    assert abs(ratio - 1.0) <= 0.15
    raw = y.astype(np.float64)[ring].mean() / clean_lin[ring].mean()
    assert raw < 0.7                                    # 되돌리기 전에는 실제로 어둡다


def test_runtime_budget():
    *_, dt = case(0)
    assert dt < 15.0, dt
