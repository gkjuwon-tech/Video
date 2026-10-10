"""FND-07 config.py: 검증·우선순위·정규 JSON·해시 안정성·stdlib 전용."""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import blake3
import pytest

from buc.core.config import Config, ConfigError, canonical_json, load_config

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- canonical_json
def test_canonical_json_exact_bytes():
    obj = {"b": [1, 2.5, None, True], "a": {"z": "한", "y": (1, 2)}}
    assert canonical_json(obj) == '{"a":{"y":[1,2],"z":"한"},"b":[1,2.5,null,true]}'.encode("utf-8")
    assert isinstance(canonical_json({}), bytes) and canonical_json({}) == b"{}"


def test_canonical_json_keeps_int_float_bool_distinct():
    assert canonical_json(1) != canonical_json(1.0)
    assert canonical_json(1) != canonical_json(True)
    assert canonical_json(1.0) == b"1.0" and canonical_json(1) == b"1"


def test_canonical_json_is_order_independent():
    a = canonical_json({"x": 1, "y": {"p": 1, "q": 2}})
    b = canonical_json({"y": {"q": 2, "p": 1}, "x": 1})
    assert a == b


@pytest.mark.parametrize("bad", [{1: 2}, object(), {"a": object()}, {"a": {1, 2}}, {"a": b"x"}])
def test_canonical_json_rejects_non_json(bad):
    with pytest.raises(TypeError):
        canonical_json(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {"a": [float("-inf")]}])
def test_canonical_json_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        canonical_json(bad)


# ---------------------------------------------------------------- defaults / hash
EXPECTED_DEFAULT_JSON = (
    b'{"budget_daily_gpu_h":null,"budget_daily_tfu":null,"budget_daily_usd":10.0,'
    b'"budget_total_gpu_h":null,"budget_total_tfu":null,"budget_total_usd":35.0,'
    b'"extra":{},"ledger_path":"runs/ledger.sqlite","runs_dir":"runs","seed":0}'
)


def test_default_config_layout_is_pinned():
    cfg = Config()
    assert canonical_json(cfg.to_dict()) == EXPECTED_DEFAULT_JSON
    assert cfg.hash() == blake3.blake3(EXPECTED_DEFAULT_JSON).hexdigest()
    assert len(cfg.hash()) == 64 and cfg.hash() == cfg.hash()


def test_hash_changes_with_every_field():
    base = Config()
    variants = [
        dict(seed=1), dict(runs_dir="r2"), dict(ledger_path="x.sqlite"), dict(budget_daily_usd=11.0),
        dict(budget_total_usd=36.0), dict(budget_daily_gpu_h=1.0), dict(budget_total_gpu_h=1.0),
        dict(budget_daily_tfu=1.0), dict(budget_total_tfu=1.0), dict(extra={"k": 1}),
    ]
    hashes = {base.hash()} | {Config(**v).hash() for v in variants}
    assert len(hashes) == len(variants) + 1


def test_int_and_float_numbers_normalise_to_same_hash():
    a, b = Config(budget_daily_usd=10), Config(budget_daily_usd=10.0)
    assert a.hash() == b.hash() and a == b
    assert isinstance(a.budget_daily_usd, float)
    assert isinstance(Config(budget_daily_tfu=4).budget_daily_tfu, float)


def test_extra_key_order_does_not_change_hash():
    a = Config(extra={"a": 1, "b": {"c": 2, "d": 3}})
    b = Config(extra={"b": {"d": 3, "c": 2}, "a": 1})
    assert a.hash() == b.hash()


def test_seed_boundaries():
    assert Config(seed=2**64 - 1).seed == 2**64 - 1
    with pytest.raises(ConfigError):
        Config(seed=2**64)


@pytest.mark.parametrize("kw", [
    dict(seed=-1), dict(seed=True), dict(seed=1.5), dict(seed="3"),
    dict(budget_daily_usd=-1), dict(budget_daily_usd=float("nan")), dict(budget_total_usd=float("inf")),
    dict(budget_daily_usd="5"), dict(budget_daily_usd=True), dict(budget_daily_usd=None),
    dict(budget_total_usd=None), dict(budget_daily_tfu=-0.1), dict(budget_daily_gpu_h=float("nan")),
    dict(budget_total_tfu="1"), dict(runs_dir=""), dict(ledger_path=""), dict(runs_dir=5),
    dict(extra={"a": object()}), dict(extra={1: 2}), dict(extra=[1]), dict(extra={"a": float("nan")}),
])
def test_invalid_values_raise_config_error(kw):
    with pytest.raises(ConfigError):
        Config(**kw)


def test_config_error_is_value_error():
    assert issubclass(ConfigError, ValueError)


def test_optional_budgets_accept_none_and_zero():
    c = Config(budget_daily_gpu_h=None, budget_daily_usd=0)
    assert c.budget_daily_gpu_h is None and c.budget_daily_usd == 0.0


def test_frozen():
    with pytest.raises(AttributeError):
        Config().seed = 3  # type: ignore[misc]


def test_no_secret_like_fields():
    names = [f.name for f in dataclasses.fields(Config)]
    for n in names:
        for bad in ("key", "token", "secret", "password", "passwd"):
            assert bad not in n.lower(), n


# ---------------------------------------------------------------- copy semantics
def test_to_dict_is_deep_copy():
    cfg = Config(extra={"a": {"b": [1]}})
    d = cfg.to_dict()
    d["extra"]["a"]["b"].append(2)
    d["seed"] = 99
    assert cfg.to_dict()["extra"] == {"a": {"b": [1]}} and cfg.seed == 0


def test_extra_is_copied_at_construction():
    src = {"a": [1]}
    cfg = Config(extra=src)
    h = cfg.hash()
    src["a"].append(2)
    src["new"] = 1
    assert cfg.hash() == h and cfg.to_dict()["extra"] == {"a": [1]}


def test_from_dict_roundtrip_and_strictness():
    cfg = Config(seed=5, budget_total_tfu=100, extra={"k": [1, 2]})
    again = Config.from_dict(cfg.to_dict())
    assert again == cfg and again.hash() == cfg.hash()
    assert Config.from_dict({"seed": 3}) == Config(seed=3)  # 빠진 키는 기본값
    with pytest.raises(ConfigError):
        Config.from_dict({"nope": 1})
    with pytest.raises(ConfigError):
        Config.from_dict({"seed": -2})


def test_replace():
    cfg = Config(seed=1)
    c2 = cfg.replace(seed=5, budget_daily_usd=2)
    assert (c2.seed, c2.budget_daily_usd, cfg.seed) == (5, 2.0, 1)
    assert c2.runs_dir == cfg.runs_dir
    with pytest.raises(ConfigError):
        cfg.replace(seed=-1)
    with pytest.raises(ConfigError):
        cfg.replace(nope=1)


def test_caps():
    c = Config(budget_daily_usd=2, budget_total_usd=7, budget_daily_tfu=3, budget_total_tfu=9)
    assert c.caps("usd") == (2.0, 7.0)
    assert c.caps("tfu") == (3.0, 9.0)
    assert c.caps("gpu_h") == (None, None)
    with pytest.raises(ValueError):
        c.caps("eur")


# ---------------------------------------------------------------- load_config
def test_load_defaults():
    assert load_config(env={}) == Config()
    assert load_config(None, env={}, overrides=None) == Config()


def test_precedence_file_env_overrides(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"seed": 1, "budget_daily_usd": 3, "runs_dir": "fromfile"}), encoding="utf-8")
    c = load_config(f, env={})
    assert (c.seed, c.budget_daily_usd, c.runs_dir) == (1, 3.0, "fromfile")
    c = load_config(str(f), env={"BUC_SEED": "2"})
    assert (c.seed, c.budget_daily_usd, c.runs_dir) == (2, 3.0, "fromfile")
    c = load_config(f, env={"BUC_SEED": "2", "BUC_BUDGET_TOTAL_USD": "12.5"}, overrides={"seed": 3})
    assert (c.seed, c.budget_total_usd, c.budget_daily_usd) == (3, 12.5, 3.0)
    # 기본값 < 파일: 파일에 없는 키는 기본값
    assert load_config(f, env={}).ledger_path == Config().ledger_path


def test_env_parsing_and_none_values(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"budget_daily_tfu": 5}', encoding="utf-8")
    c = load_config(env={"BUC_BUDGET_DAILY_TFU": "5", "BUC_BUDGET_TOTAL_GPU_H": "2.5", "BUC_RUNS_DIR": "rr"})
    assert (c.budget_daily_tfu, c.budget_total_gpu_h, c.runs_dir) == (5.0, 2.5, "rr")
    for none_text in ("", "none", "None", "NONE"):
        assert load_config(f, env={}).budget_daily_tfu == 5.0
        c = load_config(f, env={"BUC_BUDGET_DAILY_TFU": none_text})
        assert c.budget_daily_tfu is None, none_text
    with pytest.raises(ConfigError):
        load_config(env={"BUC_BUDGET_DAILY_USD": "none"})  # None 불가 필드
    with pytest.raises(ConfigError):
        load_config(env={"BUC_SEED": "abc"})
    with pytest.raises(ConfigError):
        load_config(env={"BUC_BUDGET_TOTAL_USD": "x1"})
    with pytest.raises(ConfigError):
        load_config(env={"BUC_SEED": "-3"})


def test_unknown_env_ignored_and_extra_not_settable_by_env():
    c = load_config(env={"BUC_FOO": "1", "BUC_EXTRA": '{"a": 1}', "OTHER": "x", "buc_seed": "9"})
    assert c == Config()


def test_file_errors(tmp_path):
    bad_key = tmp_path / "a.json"
    bad_key.write_text('{"seeed": 1}', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_key, env={})
    lst = tmp_path / "b.json"
    lst.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(lst, env={})
    junk = tmp_path / "c.json"
    junk.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(junk, env={})
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json", env={})
    with pytest.raises(ConfigError):
        load_config(env={}, overrides={"nope": 1})
    with pytest.raises(ConfigError):
        load_config(env={}, overrides={"seed": -1})


def test_load_config_reads_os_environ_by_default_and_never_mutates_env(monkeypatch):
    monkeypatch.setenv("BUC_SEED", "7")
    assert load_config().seed == 7
    env = {"BUC_SEED": "4"}
    snapshot = dict(env)
    environ_before = dict(os.environ)
    load_config(env=env)
    assert env == snapshot and dict(os.environ) == environ_before


def test_extra_from_file_and_overrides(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"extra": {"vast": {"max_idle_min": 10}}}), encoding="utf-8")
    c = load_config(f, env={})
    assert c.extra == {"vast": {"max_idle_min": 10}}
    c2 = load_config(f, env={}, overrides={"extra": {"z": 1}})
    assert c2.extra == {"z": 1}  # overrides는 필드 단위로 통째 교체


# ---------------------------------------------------------------- 가벼운 import
def test_config_and_ledger_import_is_light_and_side_effect_free(tmp_path):
    code = (
        "import sys, os\n"
        "import buc.core.config, buc.core.ledger\n"
        "heavy = [m for m in ('torch', 'numpy') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "assert os.listdir('.') == [], os.listdir('.')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=50)
    assert p.returncode == 0, p.stdout + p.stderr
