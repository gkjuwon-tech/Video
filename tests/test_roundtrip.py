"""LNS-08 얇은 스모크: 공개 API 로 explain(look(x)) 왕복이 소유 100 % 이고 error_bound 이내인지 한 건만 본다."""
import numpy as np

from buc.core.contracts import Theta
from buc.lens import chain


def test_default_theta_roundtrip_is_owned_and_within_bound():
    rng = np.random.default_rng(0)
    x = (0.02 + 0.4 * rng.random((24, 40, 3))).astype(np.float32)
    th = Theta()
    r = chain.explain(chain.look(x, th, noise=False), th)
    assert r.owned.all()
    err = np.abs(r.linear.astype(np.float64) - x)
    assert np.all(err <= r.error_bound.astype(np.float64) + 1e-7)
