"""RD-10 부품: transport / render_shading / solve_intensities(NNLS) / cmaes_minimize / energy_report(CL4) / cheat_shadow_index(CL5)."""
import math

import numpy as np
import pytest

import rd10_helpers as H
from buc.read import fit_lights as F
from buc.read import intrinsics as X

N, VALID = H.ball()
W1 = VALID.astype(np.float64)


def L(name, d, k, role="motivated", kind="sun", **kw):
    if role == "cheat" and kind == "sun":
        kind = "area"
    return F.Light(name=name, role=role, kind=kind, direction=tuple(H.unit(d)), intensity=k, **kw)


# ------------------------------------------------------------------ transport
def test_transport_formula_and_rd09_seam():
    d = H.unit([0.3, -0.6, -0.5])
    t = F.transport(N, tuple(d))
    assert t.shape == N.shape[:2] and t.dtype == np.float64
    assert np.allclose(t, np.maximum(N @ d, 0.0), atol=1e-13)
    assert t.min() >= 0.0
    # RD-09 geometric_shading 과 같은 모델(ambient 0, intensity 1)
    g = X.geometric_shading(N, X.GeoLight(direction=tuple(d), intensity=1.0, ambient=0.0))
    assert np.allclose(t, g, atol=1e-12)


def test_transport_visibility_and_normal_not_normalized():
    d = tuple(H.unit([0.3, -0.6, -0.5]))
    vis = np.linspace(0.0, 1.0, N.shape[0] * N.shape[1]).reshape(N.shape[:2])
    assert np.allclose(F.transport(N, d, visibility=vis), vis * np.maximum(N @ np.array(d), 0.0), atol=1e-13)
    n2 = N * 2.0
    assert np.allclose(F.transport(n2, d), 2.0 * F.transport(N, d), atol=1e-12)     # 정규화하지 않는다
    z = N.copy(); z[3, 3] = 0.0
    assert F.transport(z, d)[3, 3] == 0.0
    with pytest.raises(ValueError):
        F.transport(N, (0.0, 0.0, 2.0))
    with pytest.raises(ValueError):
        F.transport(N, d, visibility=vis * 1.5)             # [0,1] 밖
    with pytest.raises(ValueError):
        F.transport(N, d, visibility=vis[:-1])


# ------------------------------------------------------------------ render_shading
def test_render_shading_sum_and_flags():
    a, b = H.KEY_DIR, H.RIM_DIR
    lights = [L("key", a, 0.6), L("cheat1", b, 0.4, role="cheat")]
    s = F.render_shading(lights, N, ambient=0.1)
    assert np.allclose(s, H.shading_from(N, [(a, 0.6), (b, 0.4)], 0.1), atol=1e-12)
    cov = np.zeros(N.shape[:2]); cov[:, :12] = 0.75
    fl = F.Flag(name="f1", light="cheat1", coverage=cov)
    s2 = F.render_shading(lights, N, ambient=0.1, flags=[fl])
    ref = H.shading_from(N, [(a, 0.6), (b, 0.4)], 0.1, cover=[None, cov])
    assert np.allclose(s2, ref, atol=1e-12)                 # 플래그 = 지명한 광원만 (1-coverage) 감쇠. 다른 광원은 그대로
    vis = {"key": np.where(np.arange(N.shape[1])[None, :] < 8, 0.0, 1.0) * np.ones(N.shape[:2])}
    s3 = F.render_shading(lights, N, ambient=0.1, visibility=vis)
    assert np.allclose(s3, H.shading_from(N, [(a, 0.6), (b, 0.4)], 0.1, vis=[vis["key"], None]), atol=1e-12)
    assert np.allclose(F.render_shading([], N, ambient=0.3), 0.3)
    s4 = F.render_shading(lights, N, ambient=0.1, flags=[F.Flag(name="x", light="nobody", coverage=cov)])
    assert np.allclose(s4, s, atol=1e-13)                   # 없는 광원을 지명한 플래그는 무시
    with pytest.raises(ValueError):
        F.render_shading([L("key", a, 0.6), L("key", b, 0.1)], N, ambient=0.0)      # 이름 중복
    with pytest.raises(ValueError):
        F.render_shading(lights, N, ambient=-0.1)
    many = [F.Flag(name=f"f{i}", light="key", coverage=cov) for i in range(F.MAX_FLAGS + 1)]
    with pytest.raises(ValueError):
        F.render_shading(lights, N, ambient=0.1, flags=many)


# ------------------------------------------------------------------ solve_intensities (NNLS)
def _problem(seed, J=3, nneg=False):
    rng = np.random.default_rng(seed)
    dirs = [H.unit(rng.normal(size=3) * [1, 1, 1]) for _ in range(J)]
    dirs = [d if d[2] < 0 else d * np.array([1, 1, -1]) for d in dirs]
    T = np.stack([F.transport(N, tuple(d)) for d in dirs])
    return rng, T


def test_nnls_exact_recovery_and_scale_invariants():
    rng, T = _problem(1)
    x = np.array([0.5, 0.2, 0.9]); a = 0.12
    S = a + np.tensordot(x, T, 1)
    f = F.solve_intensities(T, S, weight=W1)
    assert np.allclose(f.intensities, x, atol=1e-9) and f.ambient == pytest.approx(a, abs=1e-9)
    assert f.r2 == pytest.approx(1.0, abs=1e-12) and f.rms < 1e-9 and f.n_used == int(W1.sum())
    assert f.nmse == pytest.approx(0.0, abs=1e-12)
    # 영상 c배 -> 세기 c배, R^2 불변
    g = F.solve_intensities(T, 3.0 * S, weight=W1)
    assert np.allclose(g.intensities, 3.0 * x, atol=1e-8) and g.r2 == pytest.approx(1.0, abs=1e-12)
    # 가중 c배 -> 같은 해
    h = F.solve_intensities(T, S, weight=7.0 * W1)
    assert np.allclose(h.intensities, x, atol=1e-9)
    # 가중 0 화소의 값은 무관
    S2 = S.copy(); S2[~VALID] = 99.0
    k = F.solve_intensities(T, S2, weight=W1)
    assert np.allclose(k.intensities, x, atol=1e-9)
    # ambient=False -> ambient 정확히 0 이고 해는 (T 만으로) 최소제곱
    m = F.solve_intensities(T, np.tensordot(x, T, 1), weight=W1, ambient=False)
    assert m.ambient == 0.0 and np.allclose(m.intensities, x, atol=1e-9)


@pytest.mark.parametrize("seed", [2, 3, 4, 5])
def test_nnls_kkt_nonnegative_and_optimal(seed):
    rng, T = _problem(seed)
    x = np.array([0.6, 0.0, 0.3]); a = 0.2
    S = a + np.tensordot(x, T, 1) - 0.35 * T[1] + 0.02 * rng.normal(size=T[0].shape)     # 음의 몫을 요구하는 자료
    S = np.maximum(S, 0.0)
    w = W1 * (0.5 + rng.random(W1.shape))
    f = F.solve_intensities(T, S, weight=w)
    assert f.ambient >= 0.0 and np.all(f.intensities >= 0.0)                              # CL2: 음의 광원 없음
    A = np.concatenate([T.reshape(3, -1), np.ones((1, T[0].size))], 0).T
    p = np.concatenate([f.intensities, [f.ambient]])
    r = S.reshape(-1) - A @ p
    g = A.T @ (w.reshape(-1) * r)
    scale = float(np.sum(w))
    for j, pj in enumerate(p):
        if pj > 1e-12:
            assert abs(g[j]) < 1e-7 * scale, (j, g[j])            # 양수 성분: 기울기 0
        else:
            assert g[j] < 1e-7 * scale, (j, g[j])                 # 0 성분: 기울기 <= 0 (KKT)
    assert f.intensities[1] == 0.0 or f.intensities[1] < 1e-9     # 음이어야 할 성분은 0 으로 눌린다
    wf = w.reshape(-1)
    ref_sse = float(np.sum(wf * r * r))
    assert f.rms == pytest.approx(math.sqrt(ref_sse / wf.sum()), rel=1e-6)
    mean = float(np.sum(w * S) / w.sum())
    den = float(np.sum(w * (S - mean) ** 2))
    assert f.r2 == pytest.approx(1.0 - ref_sse / den, abs=1e-9)
    assert f.nmse == pytest.approx(ref_sse / den, rel=1e-6)


def test_huber_resists_unmodeled_shadows():
    n = H.dome(24, 32)
    d = H.DOME_DIR
    T = np.stack([F.transport(n, tuple(d))])
    S = 0.25 + 0.75 * T[0]
    rng = np.random.default_rng(0)
    m = rng.random(S.shape) < 0.12
    S2 = S.copy(); S2[m] -= 0.2                                # 모형 밖 어둠(드리운 그림자)
    l2 = F.solve_intensities(T, S2, loss="l2")
    hb = F.solve_intensities(T, S2, loss="huber")
    err = lambda f: abs(f.intensities[0] - 0.75) + abs(f.ambient - 0.25)
    assert err(l2) > 0.01
    assert err(hb) < 1e-3 and err(hb) < 0.2 * err(l2)
    # 이상치가 없으면 두 손실이 일치
    c = F.solve_intensities(T, S, loss="huber")
    assert np.allclose(c.intensities, [0.75], atol=1e-9) and c.ambient == pytest.approx(0.25, abs=1e-9)


def test_solve_intensities_degenerate_and_errors():
    _, T = _problem(1)
    S = 0.2 + T[0]
    with pytest.raises(F.FitLightsError):
        F.solve_intensities(T, S, weight=np.zeros_like(S))     # 사용 화소 부족
    w = np.zeros_like(S); w.flat[:3] = 1.0
    with pytest.raises(F.FitLightsError):
        F.solve_intensities(T, S, weight=w)                    # 미지수(4)보다 화소(3) 적음
    f = F.solve_intensities(T, np.full_like(S, 0.4), weight=W1)         # 분산 0 -> R^2 정의 안 됨
    assert math.isnan(f.r2) and math.isnan(f.nmse)
    with pytest.raises(ValueError):
        F.solve_intensities(T, S, loss="l1")
    with pytest.raises(ValueError):
        F.solve_intensities(T[0], S)                           # T 는 (J,H,W)
    with pytest.raises(ValueError):
        F.solve_intensities(T, S[:-1])
    with pytest.raises(ValueError):
        F.solve_intensities(T, S, weight=-W1)
    with pytest.raises(TypeError):
        F.solve_intensities(T.tolist(), S)


# ------------------------------------------------------------------ CMA-ES
def test_cmaes_converges_on_ellipsoid_and_sphere():
    f = lambda x: float(x[0] ** 2 + 10.0 * x[1] ** 2)
    for seed in range(3):
        r = F.cmaes_minimize(f, [3.0, -2.0], 1.0, popsize=16, generations=30, seed=seed)
        assert r.fun < 1e-3 and np.all(np.abs(r.x) < 0.05), (seed, r.fun)
    tgt = np.arange(5) * 0.5
    g = lambda x: float(np.sum((x - tgt) ** 2))
    r = F.cmaes_minimize(g, np.full(5, 3.0), 1.0, popsize=16, generations=80, seed=1)
    assert r.fun < 1e-3 and np.allclose(r.x, tgt, atol=0.05)


def test_cmaes_budget_history_determinism():
    calls = []
    f = lambda x: (calls.append(np.array(x)), float(np.sum((x - 1.0) ** 2)))[1]
    r = F.cmaes_minimize(f, [0.0, 0.0, 0.0], 0.5, popsize=12, generations=7, seed=5)
    assert r.n_evals == 12 * 7 == len(calls)                   # 정확히 popsize x generations 회
    assert len(r.history) == 7
    assert all(b <= a + 1e-15 for a, b in zip(r.history, r.history[1:]))         # 최선값은 비증가
    assert r.history[-1] == r.fun and r.fun == pytest.approx(min(float(np.sum((c - 1.0) ** 2)) for c in calls))
    assert float(np.sum((r.x - 1.0) ** 2)) == pytest.approx(r.fun)
    r2 = F.cmaes_minimize(lambda x: float(np.sum((x - 1.0) ** 2)), [0.0, 0.0, 0.0], 0.5, popsize=12, generations=7, seed=5)
    assert np.array_equal(r.x, r2.x) and r.fun == r2.fun and r.history == r2.history      # 같은 시드 -> 비트 동일
    r3 = F.cmaes_minimize(lambda x: float(np.sum((x - 1.0) ** 2)), [0.0, 0.0, 0.0], 0.5, popsize=12, generations=7, seed=6)
    assert not np.array_equal(r.x, r3.x)
    x0 = np.array([0.0, 0.0, 0.0])
    F.cmaes_minimize(lambda x: 0.0, x0, 0.5, generations=2)
    assert np.array_equal(x0, np.zeros(3))                     # 입력 불변


def test_cmaes_bounds_respected():
    seen = []
    f = lambda x: (seen.append(np.array(x)), float(np.sum((x - 2.0) ** 2)))[1]
    r = F.cmaes_minimize(f, [0.2, 0.2], 1.0, lower=[-1.0, -1.0], upper=[1.0, 1.0], popsize=16, generations=30, seed=0)
    S = np.stack(seen)
    assert S.min() >= -1.0 and S.max() <= 1.0                  # 평가한 모든 점이 경계 안
    assert np.allclose(r.x, [1.0, 1.0], atol=1e-3) and r.fun == pytest.approx(2.0, abs=1e-3)
    r = F.cmaes_minimize(lambda x: float(np.sum(x ** 2)), [0.5, 0.5], 1.0, lower=0.25, upper=[3.0, 3.0], generations=20)
    assert np.all(r.x >= 0.25 - 1e-12) and np.allclose(r.x, 0.25, atol=1e-3)         # 스칼라 경계는 방송


def test_cmaes_errors():
    f = lambda x: float(np.sum(x ** 2))
    for kw in (dict(popsize=3), dict(generations=0), dict(sigma0=0.0), dict(sigma0=-1.0), dict(sigma0=float("nan"))):
        base = dict(sigma0=0.5)
        base.update(kw)
        s0 = base.pop("sigma0")
        with pytest.raises(ValueError):
            F.cmaes_minimize(f, [0.0, 0.0], s0, **base)
    with pytest.raises(ValueError):
        F.cmaes_minimize(f, [0.0, np.nan], 0.5)
    with pytest.raises(ValueError):
        F.cmaes_minimize(f, [[0.0, 0.0]], 0.5)
    with pytest.raises(ValueError):
        F.cmaes_minimize(f, [0.0, 0.0], 0.5, lower=[1.0, 1.0], upper=[0.0, 0.0])
    with pytest.raises(TypeError):
        F.cmaes_minimize(f, [0.0, 0.0], 0.5, popsize=16.0)
    with pytest.raises(F.FitLightsError):
        F.cmaes_minimize(lambda x: float("nan"), [0.0, 0.0], 0.5, generations=2)       # 비유한 목적값


# ------------------------------------------------------------------ CL4: 에너지 몫과 r_light
def test_energy_report_matches_oracle():
    lights = [L("key", H.KEY_DIR, 0.6), L("cheat1", H.RIM_DIR, 0.5, role="cheat"),
              L("cheat2", [0.2, -0.3, -0.9], 0.2, role="cheat")]
    rep = F.energy_report(lights, N, ambient=0.15, weight=W1)
    e, sh, cs = H.energy_oracle(N, [(H.KEY_DIR, 0.6), (H.RIM_DIR, 0.5), (H.unit([0.2, -0.3, -0.9]), 0.2)], 0.15,
                                ["motivated", "cheat", "cheat"], W1)
    assert np.allclose(rep.energies, e, rtol=1e-9) and len(rep.energies) == 3
    assert np.allclose(rep.shares, sh, rtol=1e-9) and rep.cheat_share == pytest.approx(cs, rel=1e-9)
    assert rep.ambient_energy == pytest.approx(0.15)
    assert rep.n_cheat == 2 and rep.n_flags == 0
    assert rep.r_light == pytest.approx(max(cs / 0.35, 2 / 4, 0 / 4)) and rep.red is False
    # 가시성·플래그가 조도에 들어간다
    cov = np.zeros(N.shape[:2]); cov[:, :16] = 1.0
    rep2 = F.energy_report(lights, N, ambient=0.15, weight=W1, flags=[F.Flag(name="f", light="cheat2", coverage=cov)])
    assert rep2.energies[2] < rep.energies[2] and rep2.energies[0] == pytest.approx(rep.energies[0])
    assert rep2.n_flags == 1


def test_energy_report_limits_cl4():
    base = [L("key", H.KEY_DIR, 0.4)]
    # 치트 몫 > 35% -> 적색
    big = base + [L("cheat1", [-0.3, -0.3, -0.9], 1.2, role="cheat")]
    rep = F.energy_report(big, N, ambient=0.05, weight=W1)
    assert rep.cheat_share > F.CHEAT_SHARE_MAX and rep.r_light == pytest.approx(rep.cheat_share / 0.35) and rep.red is True
    # 광원 개수 5 (치트 5개, 몫은 작음) -> 적색
    five = base + [L(f"cheat{i}", [0.1 * i, -0.3, -0.9], 0.01, role="cheat") for i in range(1, 6)]
    rep = F.energy_report(five, N, ambient=0.3, weight=W1)
    assert rep.n_cheat == 5 and rep.cheat_share < 0.35 and rep.r_light == pytest.approx(1.25) and rep.red is True
    # 정확히 한계(치트 4 + 플래그 4)는 적색이 아니다 (r_light == 1.0)
    four = base + [L(f"cheat{i}", [0.1 * i, -0.3, -0.9], 0.01, role="cheat") for i in range(1, 5)]
    fl = [F.Flag(name=f"f{i}", light="key", coverage=np.zeros(N.shape[:2])) for i in range(4)]
    rep = F.energy_report(four, N, ambient=0.3, weight=W1, flags=fl)
    assert rep.n_cheat == 4 and rep.n_flags == 4 and rep.r_light == pytest.approx(1.0) and rep.red is False
    # 치트 없음: 몫 0, r_light 0
    rep = F.energy_report(base, N, ambient=0.2, weight=W1)
    assert rep.cheat_share == 0.0 and rep.r_light == 0.0 and rep.red is False
    # 전체 에너지 0 이면 몫 0 (0 나눗셈 없음)
    rep = F.energy_report([L("key", H.KEY_DIR, 0.0)], N, ambient=0.0, weight=W1)
    assert rep.cheat_share == 0.0 and all(s == 0.0 for s in rep.shares)


def test_cheat_shadow_index_cl5():
    d = H.unit([0.4, -0.2, -0.85])
    lights = [L("key", H.KEY_DIR, 0.6), L("cheat1", d, 0.3, role="cheat")]
    v = np.ones(N.shape[:2]); v[:, 20:] = 0.2                  # 치트 광원의 그림자가 오른쪽에 드리움
    idx = F.cheat_shadow_index(lights, N, {"cheat1": v, "key": np.zeros(N.shape[:2])}, valid=VALID)
    lit = (N @ d > 0) & VALID & (v < 0.5)
    assert idx == pytest.approx(float(lit.sum()) / float(VALID.sum()))       # 동기 광원의 그림자는 세지 않는다
    assert 0.0 < idx < 1.0
    assert F.cheat_shadow_index(lights, N, {}, valid=VALID) == 0.0           # 가시성 정보 없음 = 그림자 없음
    assert F.cheat_shadow_index([lights[0]], N, {"key": np.zeros(N.shape[:2])}, valid=VALID) == 0.0
    idx2 = F.cheat_shadow_index(lights, N, {"cheat1": v}, valid=VALID, vis_thresh=0.1)
    assert idx2 == 0.0
