"""PRD-03 스텁 팩 묶음: S-1·S2·S8 스텁, S8.5 항등 폴리시, 레지스트리 등록.

계약서: ops/harness/contracts/PRD-03.md. S9-10(렌더)·S11(KPI) 스텁은 이 팩에 없다.
"""
from buc.pipeline import spine as S
from buc.pipeline.stubs import move_fixed, read_classic, world_flat
from buc.pipeline.stubs._common import check_keys

VERSION = "0.1.0"
POLISH_NAME = "identity_polish"


def identity_polish(inputs: dict, ctx) -> dict:
    check_keys(ctx.params, ())
    return {"world": inputs["world"]}


def stub_impls() -> dict:
    """키 ("S-1","S2","S8","S8.5") 순서의 스텁 구현 목록."""
    return {
        "S-1": S.Impl(stage="S-1", kind="stub", name=read_classic.NAME, version=read_classic.VERSION,
                      fn=read_classic.classic_read),
        "S2": S.Impl(stage="S2", kind="stub", name=world_flat.NAME, version=world_flat.VERSION,
                     fn=world_flat.flat_world),
        "S8": S.Impl(stage="S8", kind="stub", name=move_fixed.NAME, version=move_fixed.VERSION,
                     fn=move_fixed.fixed_move),
        "S8.5": S.Impl(stage="S8.5", kind="stub", name=POLISH_NAME, version=VERSION, fn=identity_polish),
    }


def register_stubs(registry, *, replace: bool = False):
    for impl in stub_impls().values():
        registry.register(impl, replace=replace)
    return registry
