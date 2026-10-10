"""TCH-05 X1-slim 실행기: 호출 규약, 기준 영상(코덱 왕복), 검증 순서, 알려진 결정 구조를 가진 오라클, TCH-02 실물 연동."""
import importlib.util
import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from buc.teacher import decision_curve as dc
from buc.teacher.sample import TeacherSampler
from tch05_support import TABLE, ExplodingSampler, IdCodec, OracleSampler, PoolCodec, RecSampler, batch


def _run(sampler=None, codec=None, n=2, **kw):
    sampler = RecSampler() if sampler is None else sampler
    codec = IdCodec() if codec is None else codec
    kw.setdefault("seeds", [11, 22])
    return dc.run_x1_slim(sampler, codec, batch(n), ["a"], **kw), sampler, codec


def test_identity_teacher_gives_zero_decision_and_zero_spread():
    cur, _, _ = _run()
    assert cur.decision.shape == (7, 4) and cur.spread.shape == (7, 4)
    assert np.all(cur.decision == 0.0) and np.allclose(cur.spread, 0.0, atol=1e-15)
    assert cur.indices == tuple(range(7))                      # 기본: j + steps <= len(표)
    assert cur.sigmas == tuple(float(p) for p in TABLE.points[:7])
    assert (cur.levels, cur.steps, cur.n_images, cur.n_seeds, cur.table_key) == (4, 2, 2, 2, TABLE.key)
    assert cur.input_energy.shape == (4,) and np.all(cur.input_energy > 0)


def test_call_protocol_order_and_arguments():
    cur, s, c = _run(seeds=[5, 9, 13], steps=2, level=3, k=1, indices=(1, 4))
    assert [x.sigma for x in s.calls] == [TABLE.points[1]] * 3 + [TABLE.points[4]] * 3     # σ 바깥, 시드 안쪽
    assert [x.seeds for x in s.calls] == [5, 9, 13] * 2 and all(isinstance(x.seeds, int) for x in s.calls)
    assert all(x.steps == 2 and x.level == 3 and x.k == 1 and x.cond == ["a"] for x in s.calls)
    assert all(x.shape == (2, 3, 64, 64) and x.extra == {} for x in s.calls)    # 한 호출에 전 이미지 lockstep
    assert c.enc == 1                                                            # 인코딩은 한 번(같은 잠재 재사용)
    assert c.dec == 1 + 6                                                        # 기준 1 + (σ 2 x 시드 3)
    assert cur.indices == (1, 4) and cur.n_seeds == 3


def test_default_indices_follow_steps():
    cur, _, _ = _run(steps=3)
    assert cur.indices == tuple(range(6)) and cur.steps == 3
    cur1, _, _ = _run(steps=1)
    assert cur1.indices == tuple(range(8))


def test_forward_count_formula():
    assert dc.forward_count(n_images=3, n_seeds=4, n_indices=7, steps=2) == 168
    assert dc.forward_count(1, 2, 1, 1) == 2
    assert dc.forward_count(400, 4, 9, 1) == 14400
    with pytest.raises(dc.DecisionCurveError):
        dc.forward_count(0, 2, 1, 1)
    with pytest.raises(dc.DecisionCurveError):
        dc.forward_count(1, 2, 1, True)


def test_reference_is_codec_round_trip_not_raw_image():
    imgs = batch(2)
    codec = PoolCodec()
    assert not torch.allclose(codec.decode(codec.encode(imgs)), imgs, atol=1e-3)       # 코덱은 손실
    cur = dc.run_x1_slim(RecSampler(), codec, imgs, ["a"], seeds=[1, 2])
    assert np.all(cur.decision == 0.0)                         # 원본이 아니라 왕복 영상과 비교하므로 0


def test_oracle_teacher_structure():
    cur, _, _ = _run(OracleSampler(), n=3, seeds=[11, 22, 33])
    d, sp = cur.decision, cur.spread
    assert np.all(d[:, 0] > 0.75) and np.all(d[:, -1] < 0.25)                 # 미세 옥타브는 교체, 굵은 옥타브는 보존
    assert np.all(d[:-1] >= d[1:] - 0.02)                                     # σ가 작아질수록(인덱스 증가) D 감소
    assert np.all(np.abs(sp[:, 0] - (1 - 1 / 3)) < 0.1)                       # 독립 시드 3개: 미세 퍼짐 ≈ 2/3
    assert np.all(sp[:, -1] < 0.05)
    ed = np.array(cur.edges(0.5))
    assert np.all(np.diff(ed) < 0)                                            # 결정 경계가 단조롭게 미세 쪽으로
    assert all(w < dc.WIDTH_MAX for w in cur.widths() if math.isfinite(w))
    # 곡선은 해석 가능한 상태로 나온다
    it = dc.interpret(dc.run_x1_slim(OracleSampler(), IdCodec(), batch(3), ["a"], seeds=[11, 22, 33], levels=5))
    assert it.monotone and it.e2a_pass and len(it.adopted) >= 2 and it.fallback == "none"


def test_run_is_deterministic_and_seed_order_invariant():
    a, _, _ = _run(OracleSampler(), seeds=[11, 22, 33])
    b, _, _ = _run(OracleSampler(), seeds=[11, 22, 33])
    assert np.array_equal(a.decision, b.decision) and np.array_equal(a.spread, b.spread)
    r, _, _ = _run(OracleSampler(), seeds=[33, 22, 11])
    assert np.allclose(a.decision, r.decision, rtol=1e-9) and np.allclose(a.spread, r.spread, rtol=1e-9)
    other, _, _ = _run(OracleSampler(), seeds=[1, 2, 3])
    assert not np.array_equal(a.decision, other.decision)                      # 시드가 다르면 표본이 다르다


def test_curve_serializes_after_run():
    cur, _, _ = _run(OracleSampler(), seeds=[1, 2])
    r = dc.DecisionCurve.from_dict(__import__("json").loads(__import__("json").dumps(cur.to_dict())))
    assert np.array_equal(r.decision, cur.decision)


@pytest.mark.parametrize("bad", ["rank3", "channels4", "nan", "tiny", "one_seed", "dup_seed", "float_seed", "bool_seed",
                                 "levels0", "steps0", "steps_too_long", "idx_unsorted", "idx_dup", "idx_range",
                                 "idx_neg", "idx_empty", "not_tensor"])
def test_validation_precedes_any_sampler_or_codec_call(bad):
    imgs = batch(2)
    kw = dict(seeds=[1, 2])
    if bad == "rank3":
        imgs = imgs[0]
    elif bad == "channels4":
        imgs = torch.cat([imgs, imgs[:, :1]], 1)
    elif bad == "nan":
        imgs = imgs.clone()
        imgs[0, 0, 0, 0] = float("nan")
    elif bad == "tiny":
        imgs = batch(2, 16, 16)
    elif bad == "one_seed":
        kw["seeds"] = [1]
    elif bad == "dup_seed":
        kw["seeds"] = [4, 4]
    elif bad == "float_seed":
        kw["seeds"] = [1.0, 2.0]
    elif bad == "bool_seed":
        kw["seeds"] = [True, False]
    elif bad == "levels0":
        kw["levels"] = 0
    elif bad == "steps0":
        kw["steps"] = 0
    elif bad == "steps_too_long":
        kw["steps"] = 9
    elif bad == "idx_unsorted":
        kw["indices"] = (3, 1)
    elif bad == "idx_dup":
        kw["indices"] = (1, 1)
    elif bad == "idx_range":
        kw["indices"] = (0, 7)                                  # 7 + 2 > 8
    elif bad == "idx_neg":
        kw["indices"] = (-1, 2)
    elif bad == "idx_empty":
        kw["indices"] = ()
    elif bad == "not_tensor":
        imgs = imgs.numpy()
    codec = IdCodec()
    with pytest.raises(dc.DecisionCurveError):
        dc.run_x1_slim(ExplodingSampler(), codec, imgs, ["a"], **kw)
    assert codec.enc == 0 and codec.dec == 0


def test_wrong_shaped_prediction_is_rejected():
    s = RecSampler(fn=lambda x0, sigma, seeds: x0[..., :-1])
    with pytest.raises(dc.DecisionCurveError):
        dc.run_x1_slim(s, IdCodec(), batch(2), ["a"], seeds=[1, 2])


def test_flat_prediction_yields_nan_cells_not_zero():
    s = RecSampler(fn=lambda x0, sigma, seeds: torch.zeros_like(x0))
    cur = dc.run_x1_slim(s, IdCodec(), batch(2), ["a"], seeds=[1, 2], indices=(0,))
    assert np.allclose(cur.decision, 1.0, atol=1e-9) and np.all(np.isnan(cur.spread))


def test_does_not_use_grad_or_modify_images():
    imgs = batch(2)
    before = imgs.clone()
    seen = []

    def fn(x0, sigma, seeds):
        seen.append((torch.is_grad_enabled(), x0.requires_grad))
        return x0.clone()
    dc.run_x1_slim(RecSampler(fn=fn), IdCodec(), imgs, ["a"], seeds=[1, 2], indices=(0,))
    assert torch.equal(imgs, before)
    assert len(seen) == 2 and all(g is False and r is False for g, r in seen)       # 전체가 no_grad


# ----------------------------------------------------------------------------- 코덱 어댑터

def test_teacher_codec_delegates_to_teacher():
    log = []

    class T:
        def encode_image(self, x):
            log.append(("enc", x))
            return x + 1

        def decode_latent(self, z, clamp=True):
            log.append(("dec", z, clamp))
            return z - 1

    c = dc.TeacherCodec(T())
    x = torch.zeros(1, 3, 32, 32)
    z = c.encode(x)
    assert torch.equal(z, x + 1) and log[0][0] == "enc"
    y = c.decode(z)
    assert torch.equal(y, x) and log[1][0] == "dec" and log[1][2] is True      # 기본 clamp=True 그대로


# ----------------------------------------------------------------------------- TCH-02 실물 샘플러 연동(가짜 변환기)

def _load_fakes():
    path = pathlib.Path(__file__).resolve().parents[1] / "tch_02" / "tch02_fakes.py"
    name = "tch05_loaded_tch02_fakes"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class LatentCodec:
    """픽셀 [N,3,8h,8w] <-> 잠재 [N,16,h,w]: 8x8 평균 풀링 + 채널 순환 복제 / 앞 3채널 + 최근접 8배 확대."""

    def encode(self, x):
        p = torch.nn.functional.avg_pool2d(x, 8)
        return torch.cat([p[:, i % 3:i % 3 + 1] for i in range(16)], 1).contiguous()

    def decode(self, z):
        return z[:, :3].repeat_interleave(8, dim=2).repeat_interleave(8, dim=3).contiguous()


def _real(tmp_path, **fake_kw):
    F = _load_fakes()
    teacher = F.make_teacher(tmp_path, **fake_kw)
    return TeacherSampler(teacher), teacher


def _imgs():
    return batch(2, 32, 48, seed0=40)


def test_real_sampler_perfect_denoiser_has_no_spread_and_flat_rows(tmp_path):
    sampler, teacher = _real(tmp_path, mode="perfect")
    cur = dc.run_x1_slim(sampler, LatentCodec(), _imgs(), ["a", "bb"], seeds=[3, 4, 5])
    assert cur.decision.shape == (7, 4) and np.all(np.isfinite(cur.decision))
    assert np.allclose(cur.spread, 0.0, atol=1e-6)             # x̂0 가 σ·시드와 무관
    assert np.allclose(cur.decision, cur.decision[0], atol=1e-5)    # 모든 σ 행이 같다
    assert len(teacher.transformer.calls) == 7 * 3 * 2          # σ 점 x 시드 x step, 이미지 2장은 한 호출


def test_real_sampler_affine_denoiser_spread_follows_sigma(tmp_path):
    sampler, _ = _real(tmp_path, mode="affine", a=0.5)
    a = dc.run_x1_slim(sampler, LatentCodec(), _imgs(), ["a", "bb"], seeds=[3, 4])
    assert np.all(np.isfinite(a.decision)) and np.all(a.spread >= 0) and np.all(a.spread <= 1)
    tot = a.spread.sum(axis=1)
    assert tot[0] > tot[-1] > 0                                 # σ 가 클수록 시드 퍼짐이 크다
    sampler2, _ = _real(tmp_path / "again", mode="affine", a=0.5)
    b = dc.run_x1_slim(sampler2, LatentCodec(), _imgs(), ["a", "bb"], seeds=[3, 4])
    assert np.array_equal(a.decision, b.decision) and np.array_equal(a.spread, b.spread)       # 비트 결정론
