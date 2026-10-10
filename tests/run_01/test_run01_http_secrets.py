"""HTTP 어댑터(가짜 urlopen): 헤더·경로·오류 매핑·키 위생, 키 해석, import 순수성, 실 API 검색(GPU 러너)."""
import io
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from buc.rt import vast as V

KEY = "vk" + "TESTONLY9f8e7d6c5b4a3210fedcba98"  # 테스트 전용 가짜 값
ROOT = Path(V.__file__).parents[2]


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self, *a):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeUrlopen:
    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else {}
        self.error = error
        self.reqs = []
        self.timeouts = []

    def __call__(self, req, timeout=None):
        self.reqs.append(req)
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return _Resp(self.payload)


def _api(op, base="https://example.invalid/api/v0"):
    return V.HttpVastAPI(KEY, base_url=base, timeout=7.0, urlopen=op)


def _check_auth(req, base="https://example.invalid/api/v0"):
    assert req.full_url.startswith(base)
    assert KEY not in req.full_url
    assert req.get_header("Authorization") == "Bearer " + KEY


def test_search_offers_request_and_parse():
    op = FakeUrlopen({"offers": [{"id": 1, "dph_total": 0.3}, {"id": 2, "dph_total": 0.4}]})
    out = _api(op).search_offers(V.OfferQuery().to_api_query())
    assert [o["id"] for o in out] == [1, 2]
    assert len(op.reqs) == 1 and op.timeouts == [7.0]
    _check_auth(op.reqs[0])
    body = op.reqs[0].data or b""
    assert KEY.encode() not in body


def test_create_instance_returns_int():
    op = FakeUrlopen({"success": True, "new_contract": 123})
    iid = _api(op).create_instance(77, {"image": V.DEFAULT_IMAGE, "disk": 90, "label": "buc-gpu",
                                        "onstart": None, "env": {}})
    assert iid == 123 and isinstance(iid, int)
    _check_auth(op.reqs[0])
    assert "77" in op.reqs[0].full_url


def test_set_state_and_destroy_paths():
    op = FakeUrlopen({"success": True})
    api = _api(op)
    api.set_state(77, "stopped")
    req = op.reqs[-1]
    _check_auth(req)
    assert req.get_method() == "PUT" and req.full_url.rstrip("/").endswith("/instances/77")
    assert json.loads(req.data.decode())["state"] == "stopped"
    api.destroy_instance(77)
    req = op.reqs[-1]
    _check_auth(req)
    assert req.get_method() == "DELETE" and req.full_url.rstrip("/").endswith("/instances/77")


def test_list_instances_parse():
    op = FakeUrlopen({"instances": [{"id": 5, "label": "buc-a"}]})
    assert [d["id"] for d in _api(op).list_instances()] == [5]


def _http_error(code):
    body = io.BytesIO(json.dumps({"msg": "bad key " + KEY}).encode())
    return urllib.error.HTTPError("https://example.invalid/api/v0/x?api_key=" + KEY, code, "err", {}, body)


@pytest.mark.parametrize("code,exc", [(401, V.VastAuthError), (403, V.VastAuthError),
                                      (404, V.VastNotFound), (500, V.VastAPIError), (429, V.VastAPIError)])
def test_http_error_mapping_and_redaction(code, exc):
    op = FakeUrlopen(error=_http_error(code))
    with pytest.raises(exc) as ei:
        _api(op).list_instances()
    assert isinstance(ei.value, V.VastError)
    assert KEY not in str(ei.value) and KEY not in repr(ei.value)
    if exc is V.VastAPIError:
        assert ei.value.status == code


def test_repr_hides_key():
    api = _api(FakeUrlopen())
    assert KEY not in repr(api) and KEY not in str(api)


def test_resolve_api_key(tmp_path):
    f = tmp_path / "vast_api_key"
    f.write_text("  filekey123\n")
    assert V.resolve_api_key(env={"VAST_API_KEY": " envkey "}, key_path=f) == "envkey"
    assert V.resolve_api_key(env={"VAST_API_KEY": "   "}, key_path=f) == "filekey123"
    assert V.resolve_api_key(env={}, key_path=f) == "filekey123"
    assert V.resolve_api_key(env={}, key_path=tmp_path / "missing") is None
    with pytest.raises(V.VastAuthError):
        V.HttpVastAPI.from_env(env={}, key_path=tmp_path / "missing")
    api = V.HttpVastAPI.from_env(env={"VAST_API_KEY": KEY}, key_path=tmp_path / "missing")
    assert isinstance(api, V.HttpVastAPI) and KEY not in repr(api)


def test_constants():
    assert V.API_BASE == "https://console.vast.ai/api/v0"
    assert V.KEY_ENV == "VAST_API_KEY"
    assert V.DEFAULT_IMAGE == "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"
    for e in (V.VastAuthError, V.VastAPIError, V.VastNotFound, V.NoOfferError, V.RentLimitError, V.BudgetRefused):
        assert issubclass(e, V.VastError)
    assert issubclass(V.VastError, RuntimeError)


def test_import_is_pure_and_offline():
    code = (
        "import sys, socket; sys.path.insert(0, %r)\n"
        "def _no(*a, **k): raise AssertionError('network at import')\n"
        "socket.socket.connect = _no\n"
        "import buc.rt.vast as m\n"
        "bad = [n for n in ('torch', 'numpy', 'requests') if n in sys.modules]\n"
        "assert not bad, bad\n"
        "print('OK')\n" % str(ROOT)
    )
    env = {k: v for k, v in os.environ.items() if k != "VAST_API_KEY"}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env)
    assert p.returncode == 0 and p.stdout.strip() == "OK", p.stderr[-2000:]


def test_source_hygiene():
    src = Path(V.__file__).read_text(encoding="utf-8")
    assert "verify=False" not in src and "CERT_NONE" not in src
    assert "?api_key" not in src and "&api_key" not in src  # 키를 URL 쿼리로 보내지 않는다


@pytest.mark.gpu
def test_live_offer_search_only():
    if not os.environ.get("VAST_API_KEY"):
        pytest.skip("VAST_API_KEY 없음")
    api = V.HttpVastAPI.from_env()
    raws = api.search_offers(V.OfferQuery(max_dph=2.0).to_api_query())
    assert isinstance(raws, list)
    offers = [V.Offer.from_api(r) for r in raws if r.get("dph_total") is not None]
    for o in V.select_offers(offers, V.OfferQuery(max_dph=2.0)):
        assert o.dph_total <= 2.0 and o.country != "CN"
