"""RD-14 build_obs_map: 화소 지도(φ, l_obs 경/부분, ν_obs), 블록 요약, MRH 거부권, 스케일 여유, 소유 판정."""
import json
import math
import time

import numpy as np
import pytest

import rd14_helpers as H

om = H.om
NAN = float("nan")


# ------------------------------------------------------------------ 오라클 (계약서 §수학 그대로, 화소 루프)
def expected(depth, K, *, r_ratio=1.0, coc=None, mot=None, jpeg_q=None, normal=None, valid=None, mrh=None,
             sigma_log_s=0.0, k_margin=0.0, cos_min=0.1, part_extra=0.0):
    Hh, Ww = depth.shape
    f = 0.5 * (K[0, 0] + K[1, 1])
    ok = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        ok &= valid
    cos = np.ones_like(depth)
    if normal is not None:
        d = H.rays(K, Hh, Ww)
        n = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        cos = np.clip(np.abs((n * d).sum(-1)), cos_min, 1.0)
    z = np.where(ok, depth, 1.0)
    phi = z / (f * cos) * math.exp(k_margin * sigma_log_s)
    qj = H.jpeg_factor(jpeg_q)
    c = np.zeros_like(depth) if coc is None else np.broadcast_to(coc, depth.shape).astype(np.float64)
    m = np.zeros_like(depth) if mot is None else np.broadcast_to(mot, depth.shape).astype(np.float64)
    cap = np.full(depth.shape, 0.5)
    if mrh is not None:
        h, w = mrh.shape[1:]
        si = np.minimum(h - 1, np.floor((np.arange(Hh) + 0.5) * h / Hh).astype(int))
        sj = np.minimum(w - 1, np.floor((np.arange(Ww) + 0.5) * w / Ww).astype(int))
        sel = mrh[:, si][:, :, sj] > 0.5
        for i in range(Hh):
            for j in range(Ww):
                cap[i, j] = H.mrh_cap_oracle([bool(v) for v in sel[:, i, j]])
    pe = H.phi_eff(phi, r_ratio=r_ratio, coc=c, mot=m, qj=qj, nu_cap=cap if mrh is not None else None)
    lh = np.full(depth.shape, H.NO_LEVEL, np.int64)
    lp = lh.copy()
    nu = np.full(depth.shape, np.nan)
    cache = {}
    for i in range(Hh):
        for j in range(Ww):
            if not ok[i, j]:
                continue
            lh[i, j], lp[i, j] = H.owned_levels(pe[i, j], part_extra)
            key = (float(c[i, j]), float(m[i, j]))
            if key not in cache:
                cache[key] = H.nu_obs_oracle(r_ratio, key[0], key[1], qj)
            nu[i, j] = min(cache[key], cap[i, j])
    return dict(ok=ok, phi=phi, phi_eff=pe, lh=lh, lp=lp, nu=nu, cap=cap, c=c, m=m, qj=qj)


def scene(Hh=48, Ww=64, seed=0):
    rng = np.random.default_rng(seed)
    K = H.K_from_fov(Ww, Hh, 60.0)
    yy, xx = np.mgrid[0:Hh, 0:Ww]
    z = 2.3 + 0.03 * xx + 0.02 * yy + 0.4 * rng.random((Hh, Ww))
    return K, z


def build(depth, K, **kw):
    return om.build_obs_map(depth, K, **kw)


# ------------------------------------------------------------------ 균일 평면: 손으로 푼 값
def test_uniform_plane_hand_values():
    K = H.K_from_fov(64, 48, 60.0)
    z = np.full((48, 64), 3.137)
    m = build(z, K)
    f = K[0, 0]
    phi = 3.137 / f                                    # ≈ 0.0566 m, a = log2(4/φ) = 6.14
    assert m.size == (48, 64) and m.valid.all() and m.valid.dtype == bool
    assert np.allclose(m.phi_in, phi, rtol=1e-6) and np.allclose(m.phi_eff, phi, rtol=1e-6)
    assert (m.l_hard == 6).all() and (m.l_part == 7).all()
    assert m.l_hard.dtype == np.int16 and m.l_part.dtype == np.int16
    assert np.allclose(m.nu_obs, 0.5)
    assert m.block_px == 16 and m.blk_valid.shape == (3, 4) and m.blk_valid.all()
    assert (m.blk_l_hard == 6).all() and (m.blk_l_part == 7).all() and np.allclose(m.blk_nu_obs, 0.5)
    assert m.blk_w.shape == (4, 3, 4) and (m.blk_w > 0.9).all() and (m.blk_w <= 1.0).all()
    assert m.warnings == () and m.used_mrh is False and m.r_ratio == 1.0 and m.q_jpeg == 1.0
    assert m.obs_levels() == (6.0, 7.0)


def test_block_grid_shapes_non_divisible():
    K = H.K_from_fov(70, 50, 60.0)
    z = np.full((50, 70), 4.0)
    m = build(z, K, opts=om.ObsOpts(block_px=16))
    assert m.blk_valid.shape == (4, 5) and m.blk_w.shape == (4, 4, 5) and m.blk_l_hard.shape == (4, 5)
    m1 = build(z, K, opts=om.ObsOpts(block_px=1))
    assert m1.blk_valid.shape == (50, 70) and (m1.blk_l_hard == m1.l_hard).all()


# ------------------------------------------------------------------ 전 요인 오라클 대조
def test_full_oracle_all_factors():
    K, z = scene()
    Hh, Ww = z.shape
    coc = np.zeros(z.shape)
    coc[:, 32:] = 3.0
    coc[:16, :16] = 6.0
    mot = np.zeros(z.shape)
    mot[30:, :20] = 4.5
    n = np.zeros(z.shape + (3,))
    n[..., 0], n[..., 2] = math.sin(0.4), -math.cos(0.4)
    valid = np.ones(z.shape, bool)
    valid[40:, 50:] = False
    kw = dict(r_ratio=0.75, coc_px=coc, motion_px=mot, jpeg_quality=60.0, normal_cam=n, valid=valid)
    m = build(z, K, **kw)
    e = expected(z, K, r_ratio=0.75, coc=coc, mot=mot, jpeg_q=60.0, normal=n, valid=valid)
    assert (m.valid == e["ok"]).all()
    assert (m.l_hard[e["ok"]] == e["lh"][e["ok"]]).all() and (m.l_part[e["ok"]] == e["lp"][e["ok"]]).all()
    assert (m.l_hard[~e["ok"]] == H.NO_LEVEL).all() and (m.l_part[~e["ok"]] == H.NO_LEVEL).all()
    assert np.allclose(m.phi_in[e["ok"]], e["phi"][e["ok"]], rtol=1e-6)
    assert np.allclose(m.phi_eff[e["ok"]], e["phi_eff"][e["ok"]], rtol=1e-6)
    assert np.isnan(m.phi_in[~e["ok"]]).all() and np.isnan(m.nu_obs[~e["ok"]]).all()
    assert np.allclose(m.nu_obs[e["ok"]], e["nu"][e["ok"]], atol=3e-3, rtol=0.03)
    assert m.q_jpeg == pytest.approx(H.jpeg_factor(60.0)) and m.r_ratio == 0.75
    # 열화는 단조: 같은 깊이에서 열화 영역의 레벨이 깨끗한 영역보다 낮다
    clean = build(z, K, normal_cam=n)
    assert (m.l_hard[e["ok"]] <= clean.l_hard[e["ok"]]).all() and (m.l_hard < clean.l_hard)[e["ok"]].any()


def test_cos_theta_clamp_and_normal_optional():
    K, z = scene(seed=3)
    # 법선 없음 = cosθ 1
    assert np.allclose(build(z, K).phi_in, z / K[0, 0], rtol=1e-6)
    # 시선과 거의 수직인 법선: cosθ 는 cos_min 으로 잘린다
    n = H.rays(K, *z.shape).copy()
    side = np.cross(n, np.array([0.0, 1.0, 0.0]))
    side /= np.linalg.norm(side, axis=-1, keepdims=True)
    nrm = 0.999 * side + 0.0447 * n                       # |n·d| ≈ 0.0447 < cos_min
    m = build(z, K, normal_cam=nrm)
    assert np.allclose(m.phi_in, z / (K[0, 0] * 0.1), rtol=1e-5)
    m2 = build(z, K, normal_cam=nrm, opts=om.ObsOpts(cos_min=0.04))               # 덜 자르면 φ 가 더 크다
    e2 = expected(z, K, normal=nrm, cos_min=0.04)
    assert np.allclose(m2.phi_in, e2["phi"], rtol=1e-5) and (m2.phi_in > m.phi_in).all()
    assert (m2.l_hard == e2["lh"]).all() and (m2.l_hard <= m.l_hard).all()
    # 법선의 부호(앞/뒤)는 무관하고 단위가 아니어도 된다
    a = build(z, K, normal_cam=3.0 * nrm)
    b = build(z, K, normal_cam=-nrm)
    assert np.array_equal(a.l_hard, b.l_hard) and np.allclose(a.phi_in, b.phi_in)


def test_invalid_depth_and_no_valid_pixels():
    K = H.K_from_fov(32, 32, 60.0)
    z = np.full((32, 32), 3.3)
    z[0, 0], z[0, 1], z[0, 2], z[0, 3] = NAN, 0.0, -1.0, np.inf
    m = build(z, K)
    assert (~m.valid[0, :4]).all() and m.valid.sum() == 32 * 32 - 4
    assert (m.l_hard[0, :4] == H.NO_LEVEL).all() and m.blk_valid.all()
    allbad = build(np.full((32, 32), NAN), K)
    assert not allbad.valid.any() and not allbad.blk_valid.any() and "no_valid_depth" in allbad.warnings
    assert (allbad.blk_l_hard == H.NO_LEVEL).all() and (allbad.blk_w == 0).all()
    assert allbad.obs_levels() == (0.0, 0.0)
    assert allbad.owner_at(5).sum() == 0
    json.dumps(allbad.to_meta(), allow_nan=False)


# ------------------------------------------------------------------ 블록 요약: 하위 분위수, 정렬 순서 v[⌊q(n-1)⌋]
def test_block_lower_quantile_rule():
    K = H.K_from_fov(16, 16, 60.0)
    z = np.full((16, 16), 2.0)
    far = np.zeros((16, 16), bool)
    far[0, 0] = True                                      # 블록 (0,0): 64 화소 중 1 개
    far[0:2, 8:13] = True                                 # 블록 (0,1): 10 개
    far[8, 0:7] = True                                    # 블록 (1,0): 7 개 (= ⌊0.1·63⌋+1: 인덱스 6 이 아직 '먼' 쪽)
    far[8, 8:14] = True                                   # 블록 (1,1): 6 개 (인덱스 6 은 '가까운' 쪽)
    z[far] = 40.0
    near_l = int(build(np.full((16, 16), 2.0), K).l_hard[0, 0])
    far_l = int(build(np.full((16, 16), 40.0), K).l_hard[0, 0])
    assert far_l < near_l - 3
    default = build(z, K, opts=om.ObsOpts(block_px=8))                       # q = 0.1 -> v[⌊0.1·63⌋] = v[6]
    assert default.blk_l_hard[0, 0] == near_l and default.blk_l_hard[0, 1] == far_l
    assert default.blk_l_hard[1, 0] == far_l and default.blk_l_hard[1, 1] == near_l
    assert default.blk_l_part[0, 1] == far_l + 1
    zero = build(z, K, opts=om.ObsOpts(block_px=8, block_quantile=0.0))      # 최소
    assert zero.blk_l_hard[0, 0] == far_l and zero.blk_l_hard[1, 1] == far_l and zero.blk_l_hard[0, 1] == far_l
    half = build(z, K, opts=om.ObsOpts(block_px=8, block_quantile=0.5))      # 중앙(하한) v[31]
    assert half.blk_l_hard[0, 0] == near_l and half.blk_l_hard[0, 1] == near_l
    # ν_obs 블록값도 같은 규칙 (여기서는 전부 0.5)
    assert np.allclose(default.blk_nu_obs, 0.5)


def test_block_nu_obs_lower_quantile_with_defocus_patch():
    K = H.K_from_fov(16, 16, 60.0)
    z = np.full((16, 16), 3.0)
    coc = np.zeros((16, 16))
    coc[0:3, 0:4] = 6.0                                    # 블록 (0,0): 12/64 화소 -> 하위 분위수에 잡힌다
    m = build(z, K, coc_px=coc, opts=om.ObsOpts(block_px=8))
    assert m.blk_nu_obs[0, 0] == pytest.approx(H.nu_obs_oracle(1.0, 6.0, 0.0), abs=3e-3)
    assert m.blk_nu_obs[0, 1] == pytest.approx(0.5) and m.blk_l_hard[0, 0] < m.blk_l_hard[0, 1]


def test_block_w_summary_is_mean_of_band_trust():
    K = H.K_from_fov(32, 32, 60.0)
    z = np.full((32, 32), 3.0)
    coc = np.zeros((32, 32))
    coc[:, 16:] = 6.0                                      # 왼쪽 선명, 오른쪽 흐림
    m = build(z, K, coc_px=coc, opts=om.ObsOpts(block_px=16))
    for b in range(4):
        sharp = float(H.w_full(H.OCT_CENTER[b], 1.0, 0.0))
        blur = float(H.w_full(H.OCT_CENTER[b], 1.0, 6.0))
        assert m.blk_w[b, 0, 0] == pytest.approx(sharp, abs=0.02)
        assert m.blk_w[b, 0, 1] == pytest.approx(blur, abs=0.02)
    assert m.blk_w[0, 0, 1] < 0.3 < 0.9 < m.blk_w[3, 0, 1]     # 흐린 곳은 고대역 신뢰가 없다
    # 블록이 두 반으로 갈리면 평균
    mixed = build(z, K, coc_px=coc, opts=om.ObsOpts(block_px=32))
    for b in range(4):
        sharp = float(H.w_full(H.OCT_CENTER[b], 1.0, 0.0))
        blur = float(H.w_full(H.OCT_CENTER[b], 1.0, 6.0))
        assert mixed.blk_w[b, 0, 0] == pytest.approx(0.5 * (sharp + blur), abs=0.02)
    # r_eff 절단: 대역 0(중심 0.354)과 1(0.177) 사이에서 ν_c=0.25 가 자른다
    cut = build(z, K, r_ratio=0.5)
    assert (cut.blk_w[0] == 0).all() and (cut.blk_w[1] > 0.9).all()


# ------------------------------------------------------------------ MRH.obs 거부권
def test_mrh_veto_only_lowers_and_matches_oracle():
    K, z = scene(seed=5)
    h, w = 12, 16                                          # 48x64 의 1/4 격자: 최근접 사상이 블록 상수
    mrh = np.full((4, h, w), 0.9, np.float32)
    mrh[0, :, 8:] = 0.1                                    # 오른쪽 절반: 최고 대역 아님 -> cap 0.25
    mrh[3, 8:, :4] = 0.2                                   # 왼쪽 아래: 가장 낮은 대역도 아님 -> cap 1/32
    base = build(z, K)
    m = build(z, K, mrh_obs=mrh)
    e = expected(z, K, mrh=mrh)
    assert m.used_mrh is True and "mrh_veto" in m.warnings and base.used_mrh is False
    assert (m.l_hard == e["lh"]).all() and (m.l_part == e["lp"]).all()
    assert np.allclose(m.nu_obs, e["nu"], atol=1e-6)
    assert (m.l_hard <= base.l_hard).all() and (m.nu_obs <= base.nu_obs + 1e-9).all()
    assert (m.l_hard[:, 32:] == base.l_hard[:, 32:] - 1).all()          # cap 1/4 -> φ_eff ×2 -> 정확히 한 단
    assert (m.l_hard[:32, :32] == base.l_hard[:32, :32]).all()
    assert (m.l_hard[32:, :16] == base.l_hard[32:, :16] - 4).all()      # cap 1/32 -> ×16 -> 네 단
    # w 요약: 거부된 대역은 0(중심 ν_b > cap), 나머지 대역은 MRH 가 없을 때와 같다
    assert (m.blk_w[0, :, 2:] == 0).all() and (m.blk_w[:, 2, 0] == 0).all()
    assert np.array_equal(m.blk_w[1:, :2, 2:], base.blk_w[1:, :2, 2:]) and np.array_equal(m.blk_w[:, :2, :2], base.blk_w[:, :2, :2])
    assert np.allclose(m.nu_obs[:, 32:], 0.25) and np.allclose(m.nu_obs[32:, :16], 1 / 32)
    # 전부 관측이면 비트 동일 (MRH 는 한 번도 소유를 늘리지 못한다)
    full = build(z, K, mrh_obs=np.ones((4, h, w), np.float32))
    for name in ("l_hard", "l_part", "phi_eff", "nu_obs", "blk_l_hard", "blk_w"):
        assert np.array_equal(getattr(full, name), getattr(base, name), equal_nan=True), name
    assert "mrh_veto" not in full.warnings
    # 고전 열화가 이미 큰 곳에서 MRH 가 '관측'이라 해도 늘지 않는다
    coc = np.full(z.shape, 6.0)
    assert np.array_equal(build(z, K, coc_px=coc, mrh_obs=np.ones((4, h, w))).l_hard, build(z, K, coc_px=coc).l_hard)
    # 임계 mrh_tau
    weak = np.full((4, h, w), 0.4, np.float32)
    assert (build(z, K, mrh_obs=weak, opts=om.ObsOpts(mrh_tau=0.3)).l_hard == base.l_hard).all()
    assert (build(z, K, mrh_obs=weak).l_hard < base.l_hard).all()


def test_mrh_nearest_resample_non_divisible():
    K, z = scene(seed=6)
    mrh = np.full((4, 5, 7), 0.9, np.float32)
    mrh[0, 2, :] = 0.0                                    # 격자 행 2 만 최고 대역 거부
    m = build(z, K, mrh_obs=mrh)
    e = expected(z, K, mrh=mrh)
    assert (m.l_hard == e["lh"]).all()
    rows = np.minimum(4, np.floor((np.arange(48) + 0.5) * 5 / 48).astype(int))
    assert ((rows == 2) == (m.l_hard < build(z, K).l_hard).all(axis=1)).all()


# ------------------------------------------------------------------ 스케일 불확실도 여유
def test_scale_margin_and_sigma_reporting():
    K, z = scene(seed=7)
    s = 0.3
    base = build(z, K, sigma_log_s=s)
    assert np.array_equal(base.l_hard, build(z, K).l_hard) and base.sigma_log_s == s            # 기본 k = 0: 불변
    assert base.l_sigma_octaves == pytest.approx(s / math.log(2.0))
    assert "scale_uncertain" not in base.warnings
    marg = build(z, K, sigma_log_s=s, opts=om.ObsOpts(scale_margin_k=2.0))
    e = expected(z, K, sigma_log_s=s, k_margin=2.0)
    assert np.allclose(marg.phi_in, e["phi"], rtol=1e-6) and (marg.l_hard == e["lh"]).all()
    assert (marg.l_hard <= base.l_hard).all() and (marg.l_hard < base.l_hard).any()
    assert "scale_uncertain" in build(z, K, sigma_log_s=0.36).warnings
    assert "scale_uncertain" not in build(z, K, sigma_log_s=0.35).warnings


def test_part_extra_option_flows_through():
    K, z = scene(seed=8)
    a = build(z, K)
    b = build(z, K, opts=om.ObsOpts(part_extra=0.5))
    e = expected(z, K, part_extra=0.5)
    assert (b.l_hard == a.l_hard).all() and (b.l_part == e["lp"]).all() and (b.l_part >= a.l_part).all()
    assert (b.l_part > a.l_part).any()


# ------------------------------------------------------------------ 소유 판정 / 요약
def test_owner_at_matches_levels_and_phi():
    K, z = scene(seed=9)
    coc = np.zeros(z.shape)
    coc[:, :20] = 4.0
    valid = np.ones(z.shape, bool)
    valid[:3] = False
    m = build(z, K, coc_px=coc, valid=valid)
    for l in range(0, 11):
        own = m.owner_at(l)
        assert own.shape == z.shape and own.dtype == np.uint8
        assert (own[~valid] == 0).all()
        ref = np.where(m.l_hard >= l, 1, np.where(m.l_part >= l, 2, 0))
        assert (own[valid] == ref[valid]).all()
        # 정의에 직접: τ_l ≥ 2φ_eff -> 1, τ_l ≥ φ_eff -> 2 (float32 저장 오차 경계는 제외)
        tau = 8.0 * 2.0 ** (-l)
        safe = valid & (np.abs(np.log2(tau / (2 * m.phi_eff.astype(np.float64)))) > 1e-5) & \
            (np.abs(np.log2(tau / m.phi_eff.astype(np.float64))) > 1e-5)
        exp = np.where(tau >= 2 * m.phi_eff, 1, np.where(tau >= m.phi_eff, 2, 0))
        assert (own[safe] == exp[safe]).all()
        bo = m.block_owner_at(l)
        assert bo.shape == m.blk_valid.shape and ((bo > 0) <= m.blk_valid).all()
        ref_b = np.where(m.blk_l_hard >= l, 1, np.where(m.blk_l_part >= l, 2, 0)) * m.blk_valid
        assert (bo == ref_b).all()
    for bad in (-9, 25, 1.5):
        with pytest.raises((ValueError, TypeError)):
            m.owner_at(bad)


def test_obs_levels_feed_core_obs_contract():
    from buc.core import contracts as C
    K, z = scene(seed=10)
    coc = np.zeros(z.shape)
    coc[:, 40:] = 5.0
    m = build(z, K, coc_px=coc)
    hard, part = m.obs_levels()
    ok = m.valid
    assert hard == pytest.approx(max(0.0, float(np.median(m.l_hard[ok]))))
    assert part == pytest.approx(max(hard, max(0.0, float(np.median(m.l_part[ok])))))
    assert m.to_obs_fields() == {"l_obs_hard": hard, "l_obs_part": part}
    obs = C.Obs(K=K, R=np.eye(3), t=np.zeros(3), width=64, height=48, **m.to_obs_fields())     # 0 ≤ hard ≤ part
    assert obs.l_obs_hard == hard and obs.l_obs_part == part
    far = build(np.full((48, 64), 1e6), K)                  # 레벨이 음수여도 Obs 에 넣을 수 있게 0 으로 자른다
    h2, p2 = far.obs_levels()
    assert 0.0 <= h2 <= p2
    C.Obs(K=K, R=np.eye(3), t=np.zeros(3), width=64, height=48, **far.to_obs_fields())


# ------------------------------------------------------------------ 메타·결정론·입력 불변
def test_meta_json_and_determinism_and_no_mutation():
    K, z = scene(seed=11)
    coc = np.zeros(z.shape)
    coc[:, :30] = 2.0
    mrh = np.full((4, 12, 16), 0.8, np.float32)
    mrh[0, :, :4] = 0.2
    args = dict(coc_px=coc, motion_px=1.0, jpeg_quality=70.0, mrh_obs=mrh, r_ratio=0.9, sigma_log_s=0.1)
    snap = (z.copy(), K.copy(), coc.copy(), mrh.copy())
    state = np.random.get_state()[1].copy()
    a = build(z, K, **args)
    b = build(z, K, **args)
    assert np.array_equal(np.random.get_state()[1], state)             # 전역 RNG 불변
    for arr, ref in zip((z, K, coc, mrh), snap):
        assert np.array_equal(arr, ref)                                # 입력 불변
    for name in ("valid", "phi_in", "phi_eff", "l_hard", "l_part", "nu_obs", "blk_valid", "blk_l_hard", "blk_l_part",
                 "blk_nu_obs", "blk_w"):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    meta = a.to_meta()
    text = json.dumps(meta, allow_nan=False, sort_keys=True)
    assert text == json.dumps(b.to_meta(), allow_nan=False, sort_keys=True)
    for key in ("size", "block_px", "r_ratio", "q_jpeg", "mrh", "warnings"):
        assert key in meta
    assert meta["size"] == [48, 64] and meta["mrh"] is True and isinstance(meta["warnings"], list)
    # 반환 배열은 호출자 입력과 메모리를 공유하지 않는다
    assert not np.shares_memory(a.phi_in, z)


def test_scalar_and_array_degradation_equivalence():
    K, z = scene(seed=12)
    a = build(z, K, coc_px=3.0, motion_px=2.0)
    b = build(z, K, coc_px=np.full(z.shape, 3.0), motion_px=np.full(z.shape, 2.0))
    assert np.array_equal(a.l_hard, b.l_hard) and np.array_equal(a.nu_obs, b.nu_obs, equal_nan=True)
    assert build(z, K, coc_px=None, motion_px=None).l_hard.tolist() == build(z, K, coc_px=0.0, motion_px=0.0).l_hard.tolist()


def test_custom_noise_model_changes_nu_obs_not_levels():
    K, z = scene(seed=13)
    coc = np.full(z.shape, 3.0)
    base = build(z, K, coc_px=coc)
    noisy = build(z, K, coc_px=coc, opts=om.ObsOpts(noise=om.NoiseModel(nsr_ref=0.3)))
    assert (noisy.nu_obs < base.nu_obs).all() and np.array_equal(noisy.l_hard, base.l_hard)     # 레벨은 φ_eff 식만 쓴다
    assert (noisy.blk_w <= base.blk_w + 1e-6).all()


# ------------------------------------------------------------------ 검증
@pytest.mark.parametrize("mut", [
    lambda a: dict(depth_m=a["z"][0]),                                       # 1차원
    lambda a: dict(depth_m=a["z"].astype(np.float32)[..., None]),            # 3차원
    lambda a: dict(depth_m=np.zeros((0, 4))),
    lambda a: dict(K=np.eye(2)),
    lambda a: dict(K=np.array([[0.0, 0, 1], [0, 1, 1], [0, 0, 1]])),         # fx = 0
    lambda a: dict(K=np.full((3, 3), np.nan)),
    lambda a: dict(r_ratio=0.0), lambda a: dict(r_ratio=1.5), lambda a: dict(r_ratio=NAN),
    lambda a: dict(coc_px=-1.0), lambda a: dict(coc_px=NAN), lambda a: dict(coc_px=np.zeros((3, 3))),
    lambda a: dict(motion_px=-0.5), lambda a: dict(motion_px=np.full((48, 64), np.inf)),
    lambda a: dict(jpeg_quality=0.0), lambda a: dict(jpeg_quality=101.0),
    lambda a: dict(normal_cam=np.zeros((48, 64, 2))), lambda a: dict(normal_cam=np.zeros((48, 64, 3))),
    lambda a: dict(normal_cam=np.full((48, 64, 3), np.nan)),
    lambda a: dict(valid=np.ones((3, 3), bool)),
    lambda a: dict(mrh_obs=np.zeros((3, 4, 4))), lambda a: dict(mrh_obs=np.full((4, 4, 4), 2.0)),
    lambda a: dict(sigma_log_s=-0.1), lambda a: dict(sigma_log_s=NAN), lambda a: dict(sigma_log_s=np.inf),
])
def test_build_rejects_bad_input(mut):
    K, z = scene()
    args = dict(depth_m=z, K=K)
    args.update(mut(dict(z=z, K=K)))
    d, k = args.pop("depth_m"), args.pop("K")
    with pytest.raises(ValueError):
        om.build_obs_map(d, k, **args)


@pytest.mark.parametrize("kw", [
    dict(depth_m="x"), dict(depth_m=[[1.0, 2.0]]), dict(depth_m=np.ones((4, 4), dtype=bool)),
    dict(depth_m=np.ones((4, 4), dtype=complex)), dict(K=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    dict(valid=np.ones((48, 64), dtype=np.float64)), dict(opts=object()), dict(jpeg_quality="90"),
])
def test_build_rejects_bad_types(kw):
    K, z = scene()
    args = dict(depth_m=z, K=K)
    args.update(kw)
    d, k = args.pop("depth_m"), args.pop("K")
    with pytest.raises(TypeError):
        om.build_obs_map(d, k, **args)


@pytest.mark.parametrize("kw", [
    dict(block_px=0), dict(block_px=2.5), dict(block_px=True), dict(block_quantile=-0.1), dict(block_quantile=0.51),
    dict(cos_min=0.0), dict(cos_min=1.5), dict(part_extra=-0.1), dict(part_extra=1.1), dict(scale_margin_k=-1.0),
    dict(mrh_tau=0.0), dict(mrh_tau=1.0), dict(noise=1.0),
])
def test_opts_validation(kw):
    with pytest.raises((ValueError, TypeError)):
        om.ObsOpts(**kw)


# ------------------------------------------------------------------ 비용 (R10 은 <1 s 목표; 여기서는 12 s 상한)
def test_720p_time_budget():
    Hh, Ww = 720, 1280
    K = H.K_from_fov(Ww, Hh, 60.0)
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    z = (2.5 + 0.002 * xx + 0.001 * yy).astype(np.float32)
    coc = (0.005 * xx).astype(np.float32)                    # 0..9.6 px 연속 지도
    mot = np.where(yy > 360, 2.0, 0.0).astype(np.float32)       # 모션은 두 영역(고유 (c,m) 쌍이 폭발하지 않게)
    mrh = np.ones((4, Hh // 4, Ww // 4), np.float32)
    mrh[0, :, 200:] = 0.0
    t0 = time.perf_counter()
    m = om.build_obs_map(z, K, coc_px=coc, motion_px=mot, jpeg_quality=80.0, mrh_obs=mrh, r_ratio=0.8)
    dt = time.perf_counter() - t0
    assert dt < 12.0, dt
    assert m.l_hard.shape == (Hh, Ww) and m.valid.all()
    # 연속 지도에서도 스칼라 함수와 맞는다 (표본 몇 점)
    for (i, j) in ((10, 10), (400, 600), (700, 1250), (200, 1000)):
        qj = H.jpeg_factor(80.0)
        exp = min(H.nu_obs_oracle(0.8, float(coc[i, j]), float(mot[i, j]), qj), 0.25 if j >= 800 else 0.5)
        assert m.nu_obs[i, j] == pytest.approx(exp, abs=3e-3, rel=0.03)
