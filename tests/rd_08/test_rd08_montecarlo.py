"""RD-08 통계 검증(고정 시드, 실제 DAT-16 표): 2σ 포함률, 공통모드 FOV 오차, chi2 보정, 저술 오탐·검출력."""
import math

import numpy as np
import pytest

import rd08_helpers as H

ms = H.ms
N_TRIALS = 3000
SIG_H = 0.08


@pytest.fixture(scope="module")
def table():
    return H.real_table()


def _draw(rng, table, names, n, s_true, f_err=0.0, saliency=None):
    picks = [names[int(i)] for i in rng.integers(0, len(names), n)]
    obs = []
    for k, nm in enumerate(picks):
        c = table.classes[nm]
        h_true = math.exp(c.mu_log + c.sigma_log * rng.standard_normal())
        h_hat = h_true / s_true * math.exp(SIG_H * rng.standard_normal() - f_err)
        obs.append(ms.SizeObs(cls=nm, h_rel=h_hat, sigma_h=SIG_H, id=str(k),
                              saliency=0.0 if saliency is None else saliency[k]))
    return obs


def _run(table, seed, *, sigma_f_true=0.0, sigma_log_f=0.0, with_saliency=False):
    rng = np.random.default_rng(seed)
    names = list(table.core())
    covered, zs, chis, incons, authored = 0, [], [], 0, 0
    for _ in range(N_TRIALS):
        n = int(rng.integers(3, 9))
        s_true = math.exp(rng.uniform(math.log(0.1), math.log(10)))
        f_err = float(rng.normal(0, sigma_f_true)) if sigma_f_true else 0.0
        sal = None
        if with_saliency:
            sal = [float(x) for x in rng.permutation(n) / n + 0.01]  # 서로 다른 saliency, 유일한 최대
        obs = _draw(rng, table, names, n, s_true, f_err, sal)
        r = ms.estimate_metric_scale(obs, table=table, sigma_log_f=sigma_log_f)
        z = (r.log_s - math.log(s_true)) / r.sigma_s
        zs.append(z)
        covered += abs(z) <= 2.0
        if r.chi2_red is not None:
            chis.append(r.chi2_red)
        incons += "scale_inconsistent" in r.warnings
        authored += bool(r.authored)
    return dict(cover=covered / N_TRIALS, z_std=float(np.std(zs)), chi=float(np.mean(chis)),
                incons=incons / N_TRIALS, authored=authored / N_TRIALS)


def test_two_sigma_coverage_with_real_table(table):
    out = _run(table, 20240801)
    assert 0.93 <= out["cover"] <= 0.975, out
    assert 0.95 <= out["z_std"] <= 1.05, out
    # 일관된 자료에서 E[chi2_red] = 1, 오탐(경고) 드묾
    assert 0.9 <= out["chi"] <= 1.1, out
    assert out["incons"] < 0.04, out


def test_common_mode_fov_error_needs_sigma_log_f(table):
    naive = _run(table, 777, sigma_f_true=0.10, sigma_log_f=0.0)
    aware = _run(table, 777, sigma_f_true=0.10, sigma_log_f=0.10)
    assert naive["cover"] < 0.85, naive  # N 이 늘어도 줄지 않는 오차를 무시하면 과신한다
    assert 0.93 <= aware["cover"] <= 0.975, aware


def test_authored_false_alarm_rate_is_low_on_consistent_scenes(table):
    out = _run(table, 31337, with_saliency=True)
    assert out["authored"] < 0.01, out
    assert 0.93 <= out["cover"] <= 0.975, out


def test_authored_hero_detection_power_and_factor(table):
    rng = np.random.default_rng(99)
    firm = [n for n in table.core() if table.classes[n].sigma_log <= 0.10]
    all_names = list(table.core())
    detected, errs, trials = 0, [], 600
    for _ in range(trials):
        n = int(rng.integers(3, 7))
        s_true = math.exp(rng.uniform(math.log(0.1), math.log(10)))
        sal = [0.1] * n + [0.9]
        obs = _draw(rng, table, all_names, n, s_true, saliency=sal[:n])
        hn = firm[int(rng.integers(len(firm)))]
        c = table.classes[hn]
        h_true = math.exp(c.mu_log + c.sigma_log * rng.standard_normal()) * 3.0
        obs.append(ms.SizeObs(cls=hn, h_rel=h_true / s_true * math.exp(SIG_H * rng.standard_normal()),
                              sigma_h=SIG_H, saliency=0.9, id="hero"))
        r = ms.estimate_metric_scale(obs, table=table)
        if [a.id for a in r.authored] == ["hero"]:
            detected += 1
            # 사전 중앙값 대비가 아니라 실제 배수 3 × 개체 편차 exp(σε): 로그 오차는 σ_c 규모
            errs.append(abs(math.log(r.authored[0].factor / 3.0)))
    assert detected / trials >= 0.90, detected / trials
    assert float(np.median(errs)) < 0.15
    assert not math.isnan(sum(errs))
