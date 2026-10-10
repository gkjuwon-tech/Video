"""TCH-11 순수 함수: 프롬프트 합성·금칙·시드·키·uint8 변환 (교사 없이)."""
import re

import numpy as np
import pytest
import torch

from buc.core import det
from buc.teacher import entry_gen as E

KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")   # DAT-05 key 정규식


def test_modes_and_suffix_table():
    assert E.MODES == ("survey", "cinematic", "plain")
    assert set(E.MODE_SUFFIX) == set(E.MODES)
    assert E.MODE_SUFFIX["plain"] == ""
    s, c = E.MODE_SUFFIX["survey"], E.MODE_SUFFIX["cinematic"]
    assert isinstance(s, str) and s.strip() and isinstance(c, str) and c.strip()
    assert s != c and s not in c and c not in s
    assert E.ENTRY_TAG == "buc.tch11.entry.v1"
    assert E.DEFAULT_TAINT == ("ENGINE",)


def test_error_hierarchy():
    assert issubclass(E.EntryGenError, ValueError)
    assert issubclass(E.BannedPromptError, E.EntryGenError)


@pytest.mark.parametrize("mode", ["survey", "cinematic", "plain"])
def test_compose_prompt_format(mode):
    raw = "  a mossy stone wall  "
    out = E.compose_prompt(raw, mode)
    suffix = E.MODE_SUFFIX[mode]
    assert out == ("a mossy stone wall" if suffix == "" else f"a mossy stone wall, {suffix}")
    # 기본 모드는 survey
    assert E.compose_prompt(raw) == E.compose_prompt(raw, "survey")


def test_compose_prompt_keeps_unicode_and_inner_text():
    p = "시장 골목의 국수 가게,\n김이 오른다"
    assert E.compose_prompt(p, "plain") == p
    assert E.compose_prompt(p, "cinematic").startswith(p + ", ")


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_compose_prompt_rejects_blank(bad):
    with pytest.raises(E.EntryGenError):
        E.compose_prompt(bad, "survey")


def test_compose_prompt_type_and_mode_errors():
    for bad in (None, 3, b"bytes", ["a"]):
        with pytest.raises(TypeError):
            E.compose_prompt(bad, "survey")
    for m in ("weird", "", "SURVEY", None):
        with pytest.raises(E.EntryGenError):
            E.compose_prompt("a wall", m)


def test_check_prompts_strips_and_returns_tuple():
    out = E.check_prompts([" a ", "b\n"], banned=())
    assert out == ("a", "b") and isinstance(out, tuple)
    assert E.check_prompts([]) == ()


def test_check_prompts_banned_regex_case_insensitive():
    ps = ["a calm lake", "a MICKEY mouse statue"]
    with pytest.raises(E.BannedPromptError) as ei:
        E.check_prompts(ps, banned=["mickey"])
    assert "mickey" in str(ei.value).lower()
    # 컴파일된 패턴과 단어 경계
    assert E.check_prompts(["nikes are shoes"], banned=[r"\bnike\b"]) == ("nikes are shoes",)
    with pytest.raises(E.BannedPromptError):
        E.check_prompts(["a nike sign"], banned=[re.compile(r"\bnike\b", re.I)])
    # 한글도 걸린다
    with pytest.raises(E.BannedPromptError):
        E.check_prompts(["우주 소년 아톰 포스터"], banned=["아톰"])


def test_check_prompts_type_and_blank():
    with pytest.raises(TypeError):
        E.check_prompts("a single string")
    with pytest.raises(TypeError):
        E.check_prompts(["ok", 5])
    with pytest.raises(E.EntryGenError):
        E.check_prompts(["ok", "   "])


def test_entry_seed_oracle_and_distinctness():
    for seed, idx in [(0, 0), (7, 3), (2**64 - 1, 10**6)]:
        s = E.entry_seed(seed, idx)
        assert s == det.derive_seed(seed, E.ENTRY_TAG, idx)
        assert isinstance(s, int) and 0 <= s < 2**64
    seeds = {E.entry_seed(11, i) for i in range(200)}
    assert len(seeds) == 200
    assert E.entry_seed(11, 5) != E.entry_seed(12, 5)
    assert E.entry_seed(11, 5) == E.entry_seed(11, 5)


@pytest.mark.parametrize("seed,idx", [(-1, 0), (2**64, 0), (0, -1)])
def test_entry_seed_range_errors(seed, idx):
    with pytest.raises(ValueError):
        E.entry_seed(seed, idx)


def test_entry_key_oracle_and_sensitivity():
    base = dict(prompt="a wall", mode="survey", seed=5, height=32, width=48, table_key="T" * 8)
    k = E.entry_key(**base)
    assert re.fullmatch(r"[0-9a-f]{32}", k) and KEY_RE.fullmatch(k)
    want = det.hash_parts(E.ENTRY_TAG, "key", "a wall", "survey", 5, 32, 48, "T" * 8, 0, 0).hex()[:32]
    assert k == want
    assert E.entry_key(**base) == k
    for name, other in [("prompt", "a wall "), ("mode", "plain"), ("seed", 6), ("height", 48), ("width", 64),
                        ("table_key", "U" * 8), ("level", 1), ("k", 1)]:
        assert E.entry_key(**{**base, name: other}) != k, name


def test_to_uint8_rgb_all_levels_roundtrip():
    lv = torch.arange(256, dtype=torch.float32) / 127.5 - 1.0          # 정확히 256 단계
    x = torch.zeros(1, 3, 1, 256)
    x[0, 0, 0] = lv
    x[0, 1, 0] = lv.flip(0)
    x[0, 2, 0] = -1.0
    out = E.to_uint8_rgb(x)
    assert out.dtype == np.uint8 and out.shape == (1, 1, 256, 3)
    np.testing.assert_array_equal(out[0, 0, :, 0], np.arange(256))
    np.testing.assert_array_equal(out[0, 0, :, 1], np.arange(256)[::-1])
    assert (out[0, 0, :, 2] == 0).all()


def test_to_uint8_rgb_clamp_and_midpoint():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 9.0]).view(1, 1, 1, 5).expand(1, 3, 1, 5)
    out = E.to_uint8_rgb(x)
    np.testing.assert_array_equal(out[0, 0, :, 0], [0, 0, 128, 255, 255])   # floor(v*127.5+127.5+0.5)


def test_to_uint8_rgb_random_matches_float64_oracle():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(2, 3, 9, 11, generator=g) * 2.4 - 1.2
    ref = np.floor((np.clip(x.double().numpy(), -1, 1) + 1) * 127.5 + 0.5).astype(np.uint8).transpose(0, 2, 3, 1)
    out = E.to_uint8_rgb(x)
    assert out.shape == ref.shape
    diff = np.abs(out.astype(int) - ref.astype(int))
    assert diff.max() <= 1 and (diff == 0).mean() >= 0.999


def test_to_uint8_rgb_rejects_bad_input():
    with pytest.raises(E.EntryGenError):
        E.to_uint8_rgb(torch.zeros(3, 4, 4))                 # ndim 3
    with pytest.raises(E.EntryGenError):
        E.to_uint8_rgb(torch.zeros(1, 4, 4, 4))              # 채널 4
    bad = torch.zeros(1, 3, 2, 2)
    bad[0, 1, 0, 0] = float("nan")
    with pytest.raises(E.EntryGenError):
        E.to_uint8_rgb(bad)
    bad[0, 1, 0, 0] = float("inf")
    with pytest.raises(E.EntryGenError):
        E.to_uint8_rgb(bad)
