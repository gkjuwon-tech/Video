"""FND-05 테스트 공용 도구. tools/ 는 패키지가 아니므로 파일 경로로 import 한다."""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "ledger_check.py"
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"
SCHEMA = "buc.number-ledger/v1"
_CACHE = {}


def load():
    if "m" not in _CACHE:
        spec = importlib.util.spec_from_file_location("ledger_check_under_test", TOOL)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ledger_check_under_test"] = mod  # dataclass + __future__.annotations 가 필요로 한다
        spec.loader.exec_module(mod)
        _CACHE["m"] = mod
    return _CACHE["m"]


def E(id, value, unit="TFU", owner="pipeline", **kw):
    d = {"id": id, "value": value, "unit": unit, "owner": owner}
    d.update(kw)
    return d


def D(id, value, formula, inputs, unit="TFU", owner="pipeline", **kw):
    return E(id, value, unit, owner, formula=formula, inputs=list(inputs), **kw)


def doc(*entries, **top):
    d = {"schema": SCHEMA, "entries": list(entries)}
    d.update(top)
    return d


def checks(report, name=None):
    return [v for v in report.violations if name is None or v.check == name]
