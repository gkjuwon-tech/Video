"""TCH-11 생성 루프: 진짜 load_teacher + TeacherSampler + 아핀 가짜 변환기 (CPU)."""
import inspect

import numpy as np
import pytest
import torch

from buc.gov import prov as P
from buc.teacher import entry_gen as E
from buc.teacher.sample import DEFAULT_TABLE, SigmaTable
from tch11_fakes import IMG_H, IMG_W, make_sampler

PROMPTS = ["a mossy stone wall", "시장 골목의 국수 가게", "a red kite over a wheat field",
           "an empty corridor with tiled floor", "a ceramic teapot on a wooden table"]
KW = dict(height=IMG_H, width=IMG_W)


def gen(tmp_path, sub, prompts=PROMPTS, *, table=None, **kw):
    sampler, teacher, spy = make_sampler(tmp_path / sub, table=table)
    out = E.generate_entries(sampler, prompts, **{**KW, **kw})
    return out, sampler, teacher, spy


def test_signature_is_pinned():
    sig = inspect.signature(E.generate_entries)
    names = list(sig.parameters)
    assert names[:2] == ["sampler", "prompts"]
    exp = dict(seed=0, height=512, width=512, mode="survey", batch_size=4, level=0, k=0, banned=(),
               taint=E.DEFAULT_TAINT)
    for n, d in exp.items():
        p = sig.parameters[n]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default == d, n


def test_images_match_independent_sampler_oracle(tmp_path):
    seed = 7
    out, sampler, teacher, _ = gen(tmp_path, "a", PROMPTS[:3], seed=seed, mode="survey", batch_size=3)
    assert len(out) == 3 and all(isinstance(e, E.EntryImage) for e in out)
    for i, (p, e) in enumerate(zip(PROMPTS[:3], out)):
        cap = E.compose_prompt(p, "survey")
        s_i = E.entry_seed(seed, i)
        res = sampler.generate([cap], seeds=[s_i], height=IMG_H, width=IMG_W)
        want = E.to_uint8_rgb(teacher.decode_latent(res.z))[0]
        np.testing.assert_array_equal(e.image, want)
        assert e.image.dtype == np.uint8 and e.image.shape == (IMG_H, IMG_W, 3)
        assert e.image.flags.c_contiguous and not e.image.flags.writeable
        assert (e.index, e.prompt, e.caption, e.mode, e.seed) == (i, p, cap, "survey", s_i)
        assert (e.height, e.width, e.nfe) == (IMG_H, IMG_W, 8)
        assert e.table_key == sampler.table.key
        assert e.key == E.entry_key(p, "survey", s_i, IMG_H, IMG_W, sampler.table.key)
        assert e.image.std() > 3.0           # 가짜 VAE 출력이 상수 이미지가 아님


def test_exactly_eight_nfe_per_chunk_with_expected_t_and_captions(tmp_path):
    out, sampler, teacher, spy = gen(tmp_path, "a", PROMPTS[:3], seed=1, batch_size=3)
    calls = teacher.transformer.calls
    assert len(calls) == 8 == len(DEFAULT_TABLE.points) == out[0].nfe
    want_t = torch.tensor([1.0 - s for s in DEFAULT_TABLE.points], dtype=torch.float32)
    for call, wt in zip(calls, want_t):
        assert call.n == 3
        torch.testing.assert_close(call.t, wt.repeat(3), rtol=0, atol=1e-6)
    assert spy.prompts == [E.compose_prompt(p, "survey") for p in PROMPTS[:3]]


def test_batch_size_does_not_change_results_and_chunks_calls(tmp_path):
    runs = {}
    for bs in (1, 2, 5):
        out, _, teacher, _ = gen(tmp_path, f"bs{bs}", seed=3, batch_size=bs)
        runs[bs] = (out, [c.n for c in teacher.transformer.calls])
    assert runs[1][1] == [1] * 40
    assert runs[2][1] == [2] * 8 + [2] * 8 + [1] * 8
    assert runs[5][1] == [5] * 8
    ref = runs[5][0]
    for bs in (1, 2):
        got = runs[bs][0]
        assert [e.key for e in got] == [e.key for e in ref]
        for a, b in zip(got, ref):
            np.testing.assert_array_equal(a.image, b.image)
            assert (a.index, a.seed, a.caption) == (b.index, b.seed, b.caption)


def test_deterministic_and_seed_sensitive(tmp_path):
    a, *_ = gen(tmp_path, "a", seed=9)
    b, *_ = gen(tmp_path, "b", seed=9)
    c, *_ = gen(tmp_path, "c", seed=10)
    for x, y, z in zip(a, b, c):
        np.testing.assert_array_equal(x.image, y.image)
        assert x.key == y.key and x.png_bytes() == y.png_bytes()
        assert not np.array_equal(x.image, z.image)
        assert x.key != z.key and x.seed != z.seed


def test_duplicate_prompts_get_distinct_seeds_images_and_keys(tmp_path):
    out, *_ = gen(tmp_path, "a", ["a wall", "a wall", "a wall"], seed=4)
    assert len({e.seed for e in out}) == 3 and len({e.key for e in out}) == 3
    assert not np.array_equal(out[0].image, out[1].image)
    assert not np.array_equal(out[1].image, out[2].image)


def test_explicit_seed_sequence_and_order_purity(tmp_path):
    seeds = [11, 22, 33, 44, 55]
    fwd, *_ = gen(tmp_path, "f", seed=seeds, batch_size=2)
    rev, *_ = gen(tmp_path, "r", PROMPTS[::-1], seed=seeds[::-1], batch_size=3)
    assert [e.seed for e in fwd] == seeds
    for i in range(5):
        np.testing.assert_array_equal(fwd[i].image, rev[4 - i].image)
        assert fwd[i].key == rev[4 - i].key             # 키는 index에 의존하지 않는다
        assert (fwd[i].index, rev[4 - i].index) == (i, 4 - i)


def test_mode_level_k_change_results(tmp_path):
    base, _, _, spy0 = gen(tmp_path, "m0", PROMPTS[:2], seed=2, mode="survey")
    cine, _, _, spy1 = gen(tmp_path, "m1", PROMPTS[:2], seed=2, mode="cinematic")
    plain, _, _, spy2 = gen(tmp_path, "m2", PROMPTS[:2], seed=2, mode="plain")
    assert spy2.prompts == PROMPTS[:2]
    assert spy0.prompts != spy1.prompts != spy2.prompts
    assert all(b.mode == "survey" and c.mode == "cinematic" and p.mode == "plain"
               for b, c, p in zip(base, cine, plain))
    for b, c, p in zip(base, cine, plain):
        assert len({b.key, c.key, p.key}) == 3
        assert not np.array_equal(b.image, c.image) and not np.array_equal(b.image, p.image)
    lvl, *_ = gen(tmp_path, "l1", PROMPTS[:2], seed=2, level=1)
    kk, *_ = gen(tmp_path, "k1", PROMPTS[:2], seed=2, k=1)
    for b, l_, k_ in zip(base, lvl, kk):
        assert not np.array_equal(b.image, l_.image)
        assert not np.array_equal(b.image, k_.image)
        assert len({b.key, l_.key, k_.key}) == 3      # level/k는 키 재료다
        assert (l_.level, l_.k, k_.level, k_.k) == (1, 0, 0, 1)


def test_custom_table_changes_nfe_and_key_material(tmp_path):
    tbl = SigmaTable.from_flow_shift(n=4)
    out, sampler, teacher, _ = gen(tmp_path, "t4", PROMPTS[:2], seed=1, batch_size=2, table=tbl)
    calls = teacher.transformer.calls
    assert len(calls) == 4 and all(e.nfe == 4 and e.table_key == tbl.key for e in out)
    for c, s in zip(calls, tbl.points):
        torch.testing.assert_close(c.t, torch.full((2,), 1.0 - s), rtol=0, atol=1e-6)
    ref, *_ = gen(tmp_path, "t8", PROMPTS[:2], seed=1, batch_size=2)
    assert out[0].key != ref[0].key and not np.array_equal(out[0].image, ref[0].image)


def test_default_and_custom_provenance(tmp_path):
    out, *_ = gen(tmp_path, "a", PROMPTS[:2], seed=1)
    for e in out:
        assert isinstance(e.prov, P.ProvRecord)
        assert e.prov.taint == frozenset({"ENGINE"}) and not e.prov.ext
        assert P.train_ok(e.prov) and P.dev_ok(e.prov)
        assert e.meta()["prov"] == e.prov.to_dict()
    tag = "OPEN-GEN:z-image:abc123:train"
    out2, *_ = gen(tmp_path, "b", PROMPTS[:1], seed=1, taint=(tag,))
    assert out2[0].prov.taint == frozenset({tag})
    # 태그는 이미지·키를 바꾸지 않는다
    np.testing.assert_array_equal(out2[0].image, out[0].image)
    assert out2[0].key == out[0].key


def test_empty_prompt_list_runs_nothing(tmp_path):
    out, _, teacher, spy = gen(tmp_path, "a", [], seed=1)
    assert out == [] and teacher.transformer.calls == [] and spy.prompts == []


@pytest.mark.parametrize("name,kw,exc", [
    ("banned_last_chunk", dict(banned=["teapot"], batch_size=2), E.BannedPromptError),
    ("height_not_16", dict(height=30), ValueError),
    ("width_zero", dict(width=0), ValueError),
    ("batch_zero", dict(batch_size=0), ValueError),
    ("bad_mode", dict(mode="weird"), E.EntryGenError),
    ("ext_taint_user", dict(taint=("USER",)), E.EntryGenError),
    ("hold_taint", dict(taint=("HOLDOUT",)), E.EntryGenError),
    ("seed_negative", dict(seed=-1), ValueError),
    ("seed_len_mismatch", dict(seed=[1, 2, 3]), ValueError),
])
def test_validation_fails_before_any_forward(tmp_path, name, kw, exc):
    sampler, teacher, spy = make_sampler(tmp_path / name)
    with pytest.raises(exc):
        E.generate_entries(sampler, PROMPTS, **{**KW, **kw})
    assert teacher.transformer.calls == [] and spy.prompts == []


@pytest.mark.parametrize("prompts,exc", [("a single string", TypeError), (["ok", 3], TypeError),
                                         (["ok", "  "], E.EntryGenError)])
def test_bad_prompts_fail_before_any_forward(tmp_path, prompts, exc):
    sampler, teacher, spy = make_sampler(tmp_path / "p")
    with pytest.raises(exc):
        E.generate_entries(sampler, prompts, **KW)
    assert teacher.transformer.calls == [] and spy.prompts == []


def test_no_global_side_effects(tmp_path):
    sampler, teacher, _ = make_sampler(tmp_path / "g")
    teacher.encode_text(["warm up"])             # 가짜 TE의 지연 적재가 전역 RNG를 쓰므로 먼저 적재해 둔다
    rng, gr = torch.get_rng_state(), torch.is_grad_enabled()
    out = E.generate_entries(sampler, PROMPTS[:2], seed=1, **KW)
    assert torch.equal(torch.get_rng_state(), rng) and torch.is_grad_enabled() == gr
    assert all(not p.requires_grad for p in teacher.transformer.parameters())
    assert len(out) == 2
