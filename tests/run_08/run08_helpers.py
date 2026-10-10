"""RUN-08 테스트 공용: 모듈 적재(파일 경로), 가짜 torch(cuda), 주입 시계, 케이스 공장."""
import importlib.util
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ops" / "t4" / "shot_on_t4.py"
GIB = 2**30


def load_module():
    name = "buc_run08_shot_on_t4"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


class OutOfMemoryError(RuntimeError):
    """torch.cuda.OutOfMemoryError 흉내(이름이 중요하다)."""


class Clock:
    """호출하면 현재 시각을 돌려준다. events에 "clock"을 남긴다."""

    def __init__(self, t=100.0, events=None):
        self.t = float(t)
        self.events = events if events is not None else []

    def __call__(self):
        self.events.append("clock")
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeCuda:
    def __init__(self, *, available=True, name="Tesla T4", total=15 * GIB, capability=(7, 5), events=None):
        self.available = available
        self.name = name
        self.total = total
        self.capability = capability
        self.events = events if events is not None else []
        self.alloc = 1 * GIB
        self.peak_alloc = 6 * GIB
        self.peak_reserved = int(7.5 * GIB)
        self.calls = []

    def _log(self, what):
        self.calls.append(what)
        self.events.append(what)

    def is_available(self):
        return self.available

    def get_device_name(self, i=0):
        return self.name

    def get_device_properties(self, i=0):
        return types.SimpleNamespace(total_memory=self.total)

    def get_device_capability(self, i=0):
        return self.capability

    def synchronize(self, *a, **k):
        self._log("sync")

    def empty_cache(self):
        self._log("empty_cache")

    def reset_peak_memory_stats(self, *a, **k):
        self._log("reset_peak")

    def memory_allocated(self, *a, **k):
        self._log("memory_allocated")
        return self.alloc

    def max_memory_allocated(self, *a, **k):
        self._log("max_memory_allocated")
        return self.peak_alloc

    def max_memory_reserved(self, *a, **k):
        self._log("max_memory_reserved")
        return self.peak_reserved


class FakeTorch:
    def __init__(self, **kw):
        self.events = kw.pop("events", [])
        self.cuda = FakeCuda(events=self.events, **kw)
        self.__version__ = "2.9.1+cu128"
        self.version = types.SimpleNamespace(cuda="12.8" if self.cuda.available else None)


def fake_t4(**kw):
    return FakeTorch(**kw)


def fake_5090():
    return FakeTorch(name="NVIDIA GeForce RTX 5090", total=32 * GIB, capability=(12, 0))


def fake_cpu():
    return FakeTorch(available=False)


def make_case(mod, name="c", *, clock=None, dur=0.25, params=None, behavior=None, builds=None):
    """fn 한 번 호출마다 clock을 dur만큼 진행시키는 케이스. behavior(i)로 i번째 호출에서 예외를 던질 수 있다.
    builds 리스트에 build 호출 인자(device_kind)를 쌓고, fn 호출은 case.calls에 센다."""
    state = {"calls": 0}
    builds = builds if builds is not None else []
    params = params if params is not None else {"mode": "fwd", "width": 64, "height": 48, "n_prims": 10}

    def build(kind):
        builds.append(kind)

        def fn():
            i = state["calls"]
            state["calls"] += 1
            if clock is not None:
                clock.advance(dur)
            if behavior is not None:
                behavior(i)
            return i
        return fn

    case = mod.Case(name=name, params=dict(params), build=build)
    return case, state, builds


GATE_PARAMS = {"mode": "fwd", "width": 3840, "height": 2160, "n_prims": 3_000_000}
