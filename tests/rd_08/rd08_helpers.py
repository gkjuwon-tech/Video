"""RD-08 테스트 공용 도우미: 가짜 크기 사전 표, 독립 오라클(계약서 수식 그대로), 관측 생성기.
RD-08 구현을 오라클에 쓰지 않는다(numpy·math만). 표는 resolve(text)->{name,dim,mu_log,sigma_log} 덕 타이핑이다."""
import importlib.util
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "size_prior_check.py"

from buc.read import metric_scale as ms  # noqa: E402  (구현 전에는 여기서 실패한다: 정상)


@dataclass(frozen=True)
class Prior:
    name: str
    dim: str
    median_m: float
    sigma_log: float

    @property
    def mu_log(self):
        return math.log(self.median_m)


class FakeTable:
    """DAT-16 SizePriorTable 의 resolve 계약만 흉내 낸다: 정규화(strip·소문자·공백/하이픈->_) 후 정본, 그다음 별칭."""

    def __init__(self, classes, aliases=None):
        self.classes = dict(classes)
        self.alias = dict(aliases or {})
        self.calls = []

    def resolve(self, text):
        self.calls.append(text)
        if not isinstance(text, str):
            raise KeyError(text)
        key = re.sub(r"[\s\-]+", "_", text.strip().lower())
        if key in self.classes:
            return self.classes[key]
        if key in self.alias:
            return self.classes[self.alias[key]]
        raise KeyError(text)


def std_table():
    c = {
        "door_interior": Prior("door_interior", "height", 2.0, 0.04),
        "person_adult": Prior("person_adult", "height", 1.7, 0.06),
        "chair_seat": Prior("chair_seat", "height", 0.45, 0.06),
        "dining_table": Prior("dining_table", "height", 0.75, 0.04),
        "car": Prior("car", "height", 1.5, 0.06),
        "dog": Prior("dog", "height", 0.5, 0.30),
        "tree": Prior("tree", "height", 15.0, 0.40),
    }
    return FakeTable(c, {"door": "door_interior", "person": "person_adult", "human": "person_adult"})


def obs_at(table, cls, s_true, *, dev=0.0, factor=1.0, **kw):
    """클래스 사전 중앙값 × factor × exp(dev) 크기의 객체를 진짜 스케일 s_true 에서 본 상대 크기 h_rel 로 만든다."""
    p = table.resolve(cls)
    h_rel = p.median_m * factor * math.exp(dev) / s_true
    return ms.SizeObs(cls=cls, h_rel=h_rel, **kw)


def oracle(obs, table, anchors=(), sigma_log_f=0.0):
    """계약서 수식. 모든 객체가 아는 클래스이고 저술이 없다고 가정한다."""
    a, w, v = [], [], []
    for o in obs:
        p = table.resolve(o.cls)
        vk = p.sigma_log ** 2 + o.sigma_h ** 2
        a.append(p.mu_log - math.log(o.h_rel))
        w.append(o.q / vk)
        v.append(vk)
    for an in anchors:
        a.append(an.log_s)
        w.append(1.0 / an.sigma ** 2)
        v.append(an.sigma ** 2)
    a, w, v = np.array(a), np.array(w), np.array(v)
    log_s = float((w * a).sum() / w.sum())
    sig_stat = float(w.sum() ** -0.5)
    n = len(a)
    chi2 = float((((a - log_s) ** 2) / v).sum() / (n - 1)) if n > 1 else None
    return dict(log_s=log_s, sigma_stat=sig_stat, sigma_s=math.hypot(sig_stat, sigma_log_f), chi2_red=chi2,
                z=((a - log_s) / np.sqrt(v)))


@lru_cache(maxsize=1)
def real_tool():
    spec = importlib.util.spec_from_file_location("size_prior_check_rd08", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def real_table():
    return real_tool().load_table()
