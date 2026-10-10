"""FND-07 이음매: config 캡 -> 장부, manifest_id -> run_id, FND-03 해시와의 결합."""
import pytest

from buc.core import det
from buc.core.config import Config, load_config
from buc.core.ledger import BudgetExceeded, LedgerEvent, open_ledger, tfu_for_forward
from buc.core.manifest import make_manifest

DAY = 86400
D0 = 20000 * DAY
SHA = "a" * 40


def test_config_caps_drive_ledger_enforcement(tmp_path):
    cfg = Config(ledger_path=str(tmp_path / "led" / "l.sqlite"), budget_daily_usd=2, budget_total_usd=3)
    daily, total = cfg.caps("usd")
    with open_ledger(cfg.ledger_path) as led:
        led.record_within_budget(LedgerEvent(kind="usd", amount=2.0, run_id="r", ts=D0 + 1),
                                 daily_cap=daily, total_cap=total)
        with pytest.raises(BudgetExceeded) as e:
            led.record_within_budget(LedgerEvent(kind="usd", amount=0.5, run_id="r", ts=D0 + 2),
                                     daily_cap=daily, total_cap=total)
        assert e.value.reason == "daily"
        led.record_within_budget(LedgerEvent(kind="usd", amount=1.0, run_id="r", ts=D0 + DAY + 1),
                                 daily_cap=daily, total_cap=total)
        with pytest.raises(BudgetExceeded) as e2:
            led.record_within_budget(LedgerEvent(kind="usd", amount=0.5, run_id="r", ts=D0 + 2 * DAY + 1),
                                     daily_cap=daily, total_cap=total)
        assert e2.value.reason == "total" and led.total("usd") == 3.0


def test_env_driven_config_reaches_the_ledger_path(tmp_path):
    cfg = load_config(env={"BUC_LEDGER_PATH": str(tmp_path / "x" / "env.sqlite"), "BUC_BUDGET_TOTAL_TFU": "100"})
    d, t = cfg.caps("tfu")
    assert (d, t) == (None, 100.0)
    with open_ledger(cfg.ledger_path) as led:
        for _ in range(4):
            led.record_within_budget(LedgerEvent(kind="tfu", amount=tfu_for_forward(512, 25), run_id="r", ts=D0),
                                     total_cap=t)  # 25 * 0.25 = 6.25 TFU 씩
        assert led.total("tfu") == 25.0
        led.record_within_budget(LedgerEvent(kind="tfu", amount=75.0, run_id="r", ts=D0), total_cap=t)
        with pytest.raises(BudgetExceeded):
            led.record_within_budget(LedgerEvent(kind="tfu", amount=0.01, run_id="r", ts=D0), total_cap=t)


def test_manifest_id_is_a_valid_run_id_and_cache_key_material(tmp_path):
    cfg = Config(seed=3)
    m = make_manifest("exp-9", cfg, git_commit=SHA, created_utc="2026-10-10T00:00:00Z")
    m2 = make_manifest("exp-9", Config(seed=3), git_commit=SHA, created_utc="2027-01-01T00:00:00Z")
    assert m.manifest_id == m2.manifest_id
    with open_ledger(tmp_path / "l.sqlite") as led:
        led.record(LedgerEvent(kind="gpu_h", amount=0.5, run_id=m.manifest_id, stage="train"))
        led.record(LedgerEvent(kind="gpu_h", amount=0.25, run_id=m2.manifest_id, stage="eval"))
        assert led.totals_by_run("gpu_h") == {m.manifest_id: 0.75}
    # PRD-01 이음매: str 재료로 FND-03 해시에 들어간다. 입력이 같으면 같고, 설정이 다르면 다르다.
    k1 = det.hash_parts("stage:S4", m.manifest_id, cfg.hash())
    k2 = det.hash_parts("stage:S4", m2.manifest_id, Config(seed=3).hash())
    k3 = det.hash_parts("stage:S4", make_manifest("exp-9", Config(seed=4), git_commit=SHA).manifest_id, Config(seed=4).hash())
    assert k1 == k2 and k1 != k3 and len(k1) == 32
