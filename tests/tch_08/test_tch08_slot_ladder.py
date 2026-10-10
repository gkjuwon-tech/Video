"""RefSlot(카드당 상주 하나, 교체형 참조 슬롯)과 L2b 사다리 선택."""
import pytest
import torch
from torch import nn

from buc.teacher import ref_slot as R

DEV, PARK = "gpu1", "host"


class Mover:
    def __init__(self, fail_on=None):
        self.where = {}
        self.log = []
        self.fail_on = fail_on

    def __call__(self, obj, device):
        if self.fail_on is not None and self.fail_on(obj, device):
            raise RuntimeError("injected move failure")
        self.where[id(obj)] = device
        self.log.append((obj.name, device))
        on_dev = [k for k, v in self.where.items() if v == DEV]
        assert len(on_dev) <= 1, f"장치에 교사 둘: {self.log}"
        return obj


class Obj:
    def __init__(self, name):
        self.name = name


def _slot(mover, factory_calls, factory=None):
    resident = Obj("z")
    mover.where[id(resident)] = DEV

    def make():
        factory_calls.append(1)
        return factory() if factory else Obj("klein")

    return R.RefSlot(resident, make, device=DEV, park_device=PARK, mover=mover), resident


def test_swap_order_and_lazy_single_load():
    m, calls = Mover(), []
    slot, res = _slot(m, calls)
    assert slot.state == "resident" and slot.ref is None and calls == []
    ref = slot.swap_in()
    assert slot.state == "ref" and ref is slot.ref and len(calls) == 1
    assert m.log == [("z", PARK), ("klein", DEV)]
    slot.swap_out()
    assert slot.state == "resident"
    assert m.log[2:] == [("klein", PARK), ("z", DEV)]
    ref2 = slot.swap_in()
    assert ref2 is ref and len(calls) == 1                        # 두 번째는 재적재 없음
    slot.swap_out()
    assert slot.stats == {"loads": 1, "swaps_in": 2, "swaps_out": 2}


def test_double_swap_errors():
    m, calls = Mover(), []
    slot, _ = _slot(m, calls)
    with pytest.raises(R.RefSlotError):
        slot.swap_out()
    slot.swap_in()
    with pytest.raises(R.RefSlotError):
        slot.swap_in()
    slot.swap_out()


def test_use_context_restores_on_exception():
    m, calls = Mover(), []
    slot, res = _slot(m, calls)
    with pytest.raises(KeyError):
        with slot.use() as ref:
            assert ref.name == "klein" and slot.state == "ref"
            raise KeyError("boom")
    assert slot.state == "resident" and m.where[id(res)] == DEV
    with slot.use():
        pass
    assert slot.stats["swaps_in"] == 2 and slot.stats["swaps_out"] == 2


def test_factory_failure_restores_resident():
    m, calls = Mover(), []

    def bad():
        raise OSError("no weights")

    slot, res = _slot(m, calls, factory=bad)
    with pytest.raises(OSError):
        slot.swap_in()
    assert slot.state == "resident" and m.where[id(res)] == DEV and slot.ref is None


def test_move_failure_restores_resident():
    m, calls = Mover(fail_on=lambda o, d: o.name == "klein" and d == DEV), []
    slot, res = _slot(m, calls)
    with pytest.raises(RuntimeError):
        slot.swap_in()
    assert slot.state == "resident" and m.where[id(res)] == DEV


def test_close_releases_and_reloads():
    m, calls = Mover(), []
    slot, res = _slot(m, calls)
    slot.swap_in()
    slot.close()
    assert slot.state == "resident" and slot.ref is None and m.where[id(res)] == DEV
    slot.swap_in()
    assert len(calls) == 2 and slot.stats["loads"] == 2
    slot.swap_out()


def test_default_mover_moves_modules():
    resident = nn.Linear(2, 2)
    ref = nn.Linear(3, 3)
    slot = R.RefSlot(resident, lambda: ref, device="cpu", park_device="cpu")
    with slot.use() as r:
        assert r is ref
        assert r.weight.device.type == "cpu"
    assert slot.state == "resident" and resident.weight.device == torch.device("cpu")


@pytest.mark.parametrize("level", ["L1", "L1.5", "L2a"])
def test_ladder_non_l2b(level):
    assert R.choose_l2b_path(level, ring_r_green=False, klein_ok=True, adapter_ok=True) == "none"


@pytest.mark.parametrize("green,klein,adapter,want", [
    (True, True, True, "ring_r"), (True, False, False, "ring_r"),
    (False, True, True, "klein_ref"), (False, True, False, "klein_ref"),
    (False, False, True, "ref_adapter"), (False, False, False, "cards_2_5d")])
def test_ladder_l2b(green, klein, adapter, want):
    assert R.choose_l2b_path("L2b", ring_r_green=green, klein_ok=klein, adapter_ok=adapter) == want


@pytest.mark.parametrize("level", ["L2", "l2b", "", None, "L3"])
def test_ladder_unknown_level(level):
    with pytest.raises(ValueError):
        R.choose_l2b_path(level, ring_r_green=True, klein_ok=True, adapter_ok=True)
