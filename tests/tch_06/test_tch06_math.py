"""TCH-06 순수 수식: 잔차, v-오차, DR_rel, 합격 판정, AUC, top-k. 변환기 없이 해석값과 대조한다."""
import dataclasses
import math

import numpy as np
import pytest
import torch

from buc.teacher import dca as D
from buc.teacher import sample as S
from buc.teacher.zimage import TeacherSpecError


def _x(b=3, seed=0, shape=(16, 4, 6)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, *shape, generator=g)


# ----------------------------------------------------------------------------- 상수·예외

def test_constants_and_error_hierarchy():
    assert D.DR_REL_MAX == 1.2 and D.DR_REL_MAX_OS == 1.1
    assert D.DCA_SIGMAS == (0.75, 0.5) and D.DR_SIGMAS == (0.5,)
    assert 0 < D.DR_FLOOR <= 1e-9
    for s in D.DCA_SIGMAS + D.DR_SIGMAS:      # 기본 σ는 기본 표의 점이다(스냅 없이 정확히)
        S.DEFAULT_TABLE.index_of(s)
    assert issubclass(D.DCAError, TeacherSpecError) and issubclass(D.DCAError, ValueError)


# ----------------------------------------------------------------------------- residual / velocity_error

def test_residual_hand_values():
    x0 = _x(2, 1)
    x0h = x0 + torch.tensor([1.0, 2.0]).view(2, 1, 1, 1)
    r = D.residual(x0h, x0)
    assert r.dtype == torch.float32 and tuple(r.shape) == (2,)
    torch.testing.assert_close(r, torch.tensor([1.0, 4.0]), rtol=1e-5, atol=1e-6)


def test_residual_matches_float64_reference_and_is_per_view():
    x0, x0h = _x(4, 2), _x(4, 3)
    ref = ((x0h.double() - x0.double()) ** 2).mean(dim=(1, 2, 3))
    r = D.residual(x0h, x0)
    torch.testing.assert_close(r.double(), ref, rtol=2e-5, atol=1e-7)
    # 뷰 i의 값은 다른 뷰에 의존하지 않는다
    for i in range(4):
        torch.testing.assert_close(D.residual(x0h[i:i + 1], x0[i:i + 1])[0], r[i], rtol=1e-6, atol=1e-8)
    assert float(D.residual(x0, x0).abs().max()) == 0.0


def test_residual_detaches_and_validates():
    x0 = _x(2, 4).requires_grad_(True)
    x0h = _x(2, 5).requires_grad_(True)
    assert not D.residual(x0h, x0).requires_grad
    with pytest.raises(D.DCAError):
        D.residual(_x(2)[:, :, :, :5], _x(2))                 # 모양 불일치
    with pytest.raises(D.DCAError):
        D.residual(_x(2)[0], _x(2)[0])                       # 4차원 아님
    bad = _x(2).clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(D.DCAError):
        D.residual(bad, _x(2))
    bad[0, 0, 0, 0] = float("inf")
    with pytest.raises(D.DCAError):
        D.residual(_x(2), bad)


@pytest.mark.parametrize("sigma", [1.0, 0.75, 0.5, 0.3])
def test_velocity_error_is_residual_over_sigma_squared(sigma):
    x0, x0h = _x(3, 6), _x(3, 7)
    torch.testing.assert_close(D.velocity_error(x0h, x0, sigma), D.residual(x0h, x0) / sigma ** 2,
                               rtol=1e-6, atol=1e-8)


def test_velocity_error_equals_mean_squared_velocity_gap():
    """정의 확인: v_hat=(z-x0_hat)/σ, v=ε-x0 이면 mean(v_hat-v)² == r/σ²."""
    sigma = 0.5
    x0, eps, x0h = _x(3, 8), _x(3, 9), _x(3, 10)
    z = (1 - sigma) * x0 + sigma * eps
    v_hat, v = (z - x0h) / sigma, eps - x0
    want = ((v_hat - v).double() ** 2).mean(dim=(1, 2, 3))
    torch.testing.assert_close(D.velocity_error(x0h, x0, sigma).double(), want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("sigma", [0.0, -0.1, 1.5, float("nan"), float("inf"), True, "0.5", None])
def test_velocity_error_rejects_bad_sigma(sigma):
    with pytest.raises(S.SigmaError):
        D.velocity_error(_x(2), _x(2, 1), sigma)


# ----------------------------------------------------------------------------- dr_rel / dr_verdict

def test_dr_rel_values_dtype_and_broadcast():
    r = D.dr_rel(torch.tensor([1.0, 3.0, 0.0]), 2.0)
    assert r.dtype == torch.float64
    torch.testing.assert_close(r, torch.tensor([0.5, 1.5, 0.0], dtype=torch.float64), rtol=0, atol=0)
    r2 = D.dr_rel(torch.tensor([1.0, 3.0]), torch.tensor([4.0, 2.0]))
    torch.testing.assert_close(r2, torch.tensor([0.25, 1.5], dtype=torch.float64), rtol=0, atol=0)
    r3 = D.dr_rel(torch.tensor([1.0, 3.0]), torch.tensor(2.0))          # 0차원 텐서도 스칼라
    torch.testing.assert_close(r3, torch.tensor([0.5, 1.5], dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("dr,ref", [
    ([1.0, 2.0], 0.0),                      # 기준 0
    ([1.0, 2.0], 1e-13),                    # 바닥 아래
    ([1.0, 2.0], -1.0),
    ([1.0, 2.0], float("nan")),
    ([1.0, 2.0], float("inf")),
    ([1.0, -0.5], 1.0),                     # 음수 DR
    ([1.0, float("nan")], 1.0),
    ([1.0, 2.0], [1.0, 2.0, 3.0]),          # 길이 불일치
    ([1.0, 2.0], [1.0, 0.0]),               # 원소별 기준 0
])
def test_dr_rel_rejects_bad_inputs(dr, ref):
    ref = torch.tensor(ref) if isinstance(ref, list) else ref
    with pytest.raises(D.DCAError):
        D.dr_rel(torch.tensor(dr), ref)


def test_dr_verdict_boundary_is_inclusive():
    v = D.dr_verdict(torch.tensor([3.0, 3.001, 0.0, 2.999]), 2.0, max_rel=1.5)   # 1.5 는 정확히 표현된다
    assert v.accept.dtype == torch.bool and v.rel.dtype == torch.float64
    assert v.accept.tolist() == [True, False, True, True]
    assert v.max_rel == 1.5
    assert dataclasses.is_dataclass(v)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.max_rel = 2.0


def test_dr_verdict_default_and_os_thresholds():
    dr = torch.tensor([119.0, 121.0, 100.0, 109.0, 111.0])
    assert D.dr_verdict(dr, 100.0).accept.tolist() == [True, False, True, True, False]          # 1.2
    assert D.dr_verdict(dr, 100.0, max_rel=D.DR_REL_MAX_OS).accept.tolist() == [False, False, True, True, False]
    # 기준이 원소별일 때
    v = D.dr_verdict(torch.tensor([1.0, 5.0]), torch.tensor([1.0, 10.0]))
    assert v.accept.tolist() == [True, True]
    np.testing.assert_allclose(v.rel.numpy(), [1.0, 0.5], rtol=0, atol=0)


@pytest.mark.parametrize("m", [0.0, -1.0, float("nan"), float("inf")])
def test_dr_verdict_rejects_bad_max_rel(m):
    with pytest.raises(D.DCAError):
        D.dr_verdict(torch.tensor([1.0]), 1.0, max_rel=m)


# ----------------------------------------------------------------------------- dr_auc

def test_dr_auc_hand_values():
    assert D.dr_auc([1.0, 2.0, 3.0], [4.0, 5.0]) == 1.0
    assert D.dr_auc([4.0, 5.0], [1.0, 2.0, 3.0]) == 0.0
    assert D.dr_auc([1.0, 1.0], [1.0, 1.0, 1.0]) == 0.5
    # 손계산: real=[1,2,3], fake=[2,3,4]: fake>real 6쌍, 동률 2쌍 -> (6+1)/9
    assert D.dr_auc(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 4.0])) == pytest.approx(7 / 9, abs=1e-12)
    assert isinstance(D.dr_auc([1.0], [2.0]), float)


def test_dr_auc_matches_bruteforce_and_symmetries():
    g = np.random.default_rng(0)
    real = np.round(g.normal(0.0, 1.0, 17), 1)          # 반올림으로 동률을 만든다
    fake = np.round(g.normal(0.6, 1.0, 23), 1)
    brute = sum((f > r) + 0.5 * (f == r) for r in real for f in fake) / (len(real) * len(fake))
    a = D.dr_auc(torch.tensor(real), torch.tensor(fake))
    assert a == pytest.approx(brute, abs=1e-12)
    assert D.dr_auc(torch.tensor(fake), torch.tensor(real)) == pytest.approx(1.0 - brute, abs=1e-12)
    # 단조 증가 변환에 불변
    assert D.dr_auc(torch.tensor(np.exp(real)), torch.tensor(np.exp(fake))) == pytest.approx(brute, abs=1e-12)
    assert 0.5 < a < 1.0


@pytest.mark.parametrize("real,fake", [
    ([], [1.0]), ([1.0], []), ([1.0, float("nan")], [1.0]), ([1.0], [float("inf")]),
    ([[1.0, 2.0]], [1.0]),
])
def test_dr_auc_rejects_bad_inputs(real, fake):
    with pytest.raises(D.DCAError):
        D.dr_auc(torch.tensor(real, dtype=torch.float64), torch.tensor(fake, dtype=torch.float64))


# ----------------------------------------------------------------------------- rank_topk

def test_rank_topk_ties_prefer_lower_index():
    s = torch.tensor([0.5, 2.0, 2.0, -1.0, 2.0])
    r = D.rank_topk(s, 3)
    assert r.dtype == torch.int64 and r.tolist() == [1, 2, 4]
    assert D.rank_topk(s, 10).tolist() == [1, 2, 4, 0, 3]          # k > n 이면 전부
    assert D.rank_topk(s, 1).tolist() == [1]
    assert D.rank_topk(torch.tensor([7.0]), 16).tolist() == [0]


def test_rank_topk_matches_bruteforce_with_many_ties():
    g = torch.Generator().manual_seed(4)
    s = torch.randint(0, 6, (50,), generator=g).float()
    want = sorted(range(50), key=lambda i: (-float(s[i]), i))[:16]
    assert D.rank_topk(s, 16).tolist() == want
    assert D.rank_topk(s, 16).tolist() == D.rank_topk(s.clone(), 16).tolist()


@pytest.mark.parametrize("scores,k", [
    (torch.tensor([1.0, 2.0]), 0), (torch.tensor([1.0, 2.0]), -3), (torch.tensor([1.0, float("nan")]), 1),
    (torch.tensor([1.0, float("inf")]), 1), (torch.ones(2, 2), 1), (torch.zeros(0), 1),
    (torch.tensor([1.0, 2.0]), 1.5), (torch.tensor([1.0, 2.0]), True),
])
def test_rank_topk_rejects_bad_inputs(scores, k):
    with pytest.raises(D.DCAError):
        D.rank_topk(scores, k)
