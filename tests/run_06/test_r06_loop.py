"""Trainer: 설정 검증, 스텝 의미(건너뛰기·누적·스케줄러·EMA), 로그, 경보, 발산 방어, 시간 예산, 멱등."""
import logging

import pytest
import torch
from run06_helpers import (Y, X, assert_sd_equal, batch_fn, losses, make_model, make_trainer, mse,  # noqa: F401
                           train_logs)

from buc.train.amp import AmpConfig
from buc.train.loop import FitResult, LoopConfig, ModelEMA, TrainingDiverged, Trainer, to_device


def base_cfg(tmp_path, **kw):
    d = dict(run_id="r06", ckpt_dir=tmp_path, num_samples=48, batch_size=8, max_steps=10)
    d.update(kw)
    return d


# ---- LoopConfig --------------------------------------------------------------------------------
def test_config_defaults_follow_the_recipe(tmp_path):
    c = LoopConfig(**base_cfg(tmp_path))
    assert (c.seed, c.grad_accum, c.grad_clip, c.ema_decay) == (0, 1, 1.0, None)
    assert (c.log_every, c.ckpt_every_steps, c.ckpt_every_seconds, c.upload_every_seconds) == (100, 1000, 900.0, 1800.0)
    assert (c.keep_last, c.max_abs_every, c.max_abs_threshold, c.max_consecutive_skips) == (3, 100, 2e4, 64)
    assert c.time_budget_s is None and c.deterministic is True and c.git_commit is None and dict(c.hparams) == {}
    assert c.amp == AmpConfig()
    with pytest.raises(Exception):
        c.seed = 3  # frozen


@pytest.mark.parametrize("kw", [
    dict(run_id="../x"), dict(run_id=""), dict(num_samples=0), dict(batch_size=0), dict(max_steps=-1),
    dict(grad_accum=0), dict(batch_size=8, grad_accum=3), dict(grad_clip=0.0), dict(ema_decay=1.0),
    dict(ema_decay=0.0), dict(log_every=0), dict(ckpt_every_steps=0), dict(keep_last=0), dict(max_abs_every=0),
    dict(max_consecutive_skips=0), dict(ckpt_every_seconds=0), dict(upload_every_seconds=0), dict(time_budget_s=0.0),
])
def test_config_validation(tmp_path, kw):
    with pytest.raises(ValueError):
        LoopConfig(**base_cfg(tmp_path, **kw))


def test_config_hash_covers_identity_not_schedule(tmp_path):
    h = LoopConfig(**base_cfg(tmp_path)).config_hash()
    assert len(h) == 64
    same = [dict(max_steps=999), dict(ckpt_every_steps=7), dict(log_every=3), dict(keep_last=9), dict(time_budget_s=5.0),
            dict(ckpt_dir=tmp_path / "other"), dict(upload_every_seconds=1.0), dict(max_abs_threshold=5.0)]
    for kw in same:
        assert LoopConfig(**base_cfg(tmp_path, **kw)).config_hash() == h, kw
    diff = [dict(seed=1), dict(batch_size=16), dict(num_samples=49), dict(grad_accum=2), dict(grad_clip=0.5),
            dict(ema_decay=0.99), dict(run_id="other"), dict(hparams={"lr": 1}), dict(amp=AmpConfig(dtype="bf16"))]
    for kw in diff:
        assert LoopConfig(**base_cfg(tmp_path, **kw)).config_hash() != h, kw
    a = LoopConfig(**base_cfg(tmp_path, hparams={"a": 1, "b": 2})).config_hash()
    assert a == LoopConfig(**base_cfg(tmp_path, hparams={"b": 2, "a": 1})).config_hash()


# ---- 보조 --------------------------------------------------------------------------------------
def test_to_device_recurses():
    obj = {"a": torch.ones(2), "b": [torch.zeros(1), (torch.ones(1), 3)], "c": "s"}
    out = to_device(obj, "cpu")
    assert out["c"] == "s" and out["b"][1][1] == 3 and isinstance(out["b"][1], tuple)
    assert torch.equal(out["a"], torch.ones(2))


def test_model_ema_numerics_and_state():
    m = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema = ModelEMA(m, 0.9)
    with torch.no_grad():
        m.weight.fill_(2.0)
    ema.update(m)
    assert torch.allclose(ema.state_dict()["weight"], torch.full((1, 2), 1.1), rtol=1e-6)
    with torch.no_grad():
        m.weight.fill_(3.0)
    ema.update(m)
    assert torch.allclose(ema.state_dict()["weight"], torch.full((1, 2), 0.9 * 1.1 + 0.3), rtol=1e-6)
    sd = ema.state_dict()
    ema2 = ModelEMA(torch.nn.Linear(2, 1, bias=False), 0.9)
    ema2.load_state_dict(sd)
    assert torch.equal(ema2.state_dict()["weight"], sd["weight"])
    assert sd["weight"].dtype == torch.float32
    tgt = torch.nn.Linear(2, 1, bias=False)
    ema.copy_to(tgt)
    assert torch.equal(tgt.weight.detach(), sd["weight"])
    sd["weight"].zero_()  # state_dict는 복사본
    assert ema.state_dict()["weight"].abs().sum() > 0


# ---- 기본 동작 ---------------------------------------------------------------------------------
def test_fit_result_logging_and_final_checkpoint(tmp_path):
    tr, logs = make_trainer(tmp_path, max_steps=6, log_every=2, ckpt_every_steps=100)
    res = tr.fit()
    assert isinstance(res, FitResult)
    assert (res.step, res.stop_reason, res.skipped_steps, res.resumed_from) == (6, "max_steps", 0, None)
    assert res.alarms == () and res.ckpt_path == tmp_path / "step_000000006.pt" and res.ckpt_path.exists()
    t = train_logs(logs)
    assert [d["step"] for d in t] == [2, 4, 6]  # 완료한 스텝 수가 log_every의 배수일 때
    assert set(t[0]) >= {"event", "step", "loss", "lr", "grad_norm", "loss_scale", "skipped_steps", "metrics"}
    assert t[0]["loss_scale"] == 1.0 and t[0]["skipped_steps"] == 0 and t[0]["grad_norm"] > 0
    assert res.last_log == t[-1]
    # StepLR(step=3, gamma=0.8): 완료 스텝 2 뒤 lr=1e-2, 4 뒤 0.8e-2
    assert t[0]["lr"] == pytest.approx(1e-2) and t[1]["lr"] == pytest.approx(0.8e-2)
    assert [p.name for p in tmp_path.iterdir()] == ["step_000000006.pt"]


def test_training_learns_and_masters_stay_fp32(tmp_path):
    tr, logs = make_trainer(tmp_path, max_steps=80, dropout=0.0, ckpt_every_steps=1000, lr=5e-2, sched=False,
                            grad_clip=None)
    first = mse(tr.model, batch_fn(torch.arange(48).numpy())).item()
    tr.fit()
    last = mse(tr.model, batch_fn(torch.arange(48).numpy())).item()
    assert last < 0.7 * first
    assert all(p.dtype == torch.float32 for p in tr.model.parameters())


def test_metrics_dict_from_compute_loss_is_logged(tmp_path):
    def cl(model, batch):
        loss = mse(model, batch)
        return loss, {"mse": loss.detach(), "const": 2.0}

    tr, logs = make_trainer(tmp_path, max_steps=2, compute_loss=cl, dropout=0.0)
    tr.fit()
    d = train_logs(logs)[-1]
    assert d["metrics"]["const"] == 2.0 and d["metrics"]["mse"] == pytest.approx(d["loss"], rel=1e-6)


def test_deterministic_flag(tmp_path):
    assert torch.are_deterministic_algorithms_enabled() is False
    make_trainer(tmp_path / "a", max_steps=1)[0].fit()
    assert torch.are_deterministic_algorithms_enabled() is True
    torch.use_deterministic_algorithms(False)
    make_trainer(tmp_path / "b", max_steps=1, deterministic=False)[0].fit()
    assert torch.are_deterministic_algorithms_enabled() is False


def test_seed_controls_dropout_and_sampler(tmp_path):
    def final(seed, d):
        tr, _ = make_trainer(tmp_path / d, max_steps=5, seed=seed)
        tr.fit()
        return tr.model.state_dict()

    a, a2, b = final(1, "a"), final(1, "a2"), final(2, "b")
    assert_sd_equal(a, a2)
    assert not torch.equal(a["0.weight"], b["0.weight"])


def test_non_fp32_model_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="fp32"):
        make_trainer(tmp_path, max_steps=1, model=make_model().half())


def test_batches_follow_the_stateless_sampler(tmp_path):
    from buc.train.sampler import StatelessSampler
    seen = []

    def spy(idx):
        seen.append(idx.tolist())
        return batch_fn(idx)

    tr, _ = make_trainer(tmp_path, max_steps=5, ckpt_every_steps=100, data_fn=spy)
    tr.fit()
    ref = StatelessSampler(48, 8, 7)
    assert seen == [ref.indices(t).tolist() for t in range(5)]


# ---- 누적 --------------------------------------------------------------------------------------
def test_grad_accum_equals_big_batch(tmp_path):
    def run(d, accum):
        tr, logs = make_trainer(tmp_path / d, max_steps=5, dropout=0.0, sched=False, ema=None, opt="sgd",
                                grad_accum=accum, ckpt_every_steps=100, grad_clip=None)
        tr.fit()
        return tr.model.state_dict(), losses(logs)

    (a, la), (b, lb) = run("a1", 1), run("a2", 2)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6, rtol=1e-5), k
    for s in la:
        assert la[s] == pytest.approx(lb[s], rel=1e-5)


def test_grad_accum_splits_indices_into_equal_chunks(tmp_path):
    calls = []

    def spy(idx):
        calls.append(len(idx))
        return batch_fn(idx)

    tr, _ = make_trainer(tmp_path, max_steps=2, grad_accum=4, ckpt_every_steps=100, data_fn=spy)
    tr.fit()
    assert calls == [2] * 8


# ---- 비유한 스텝 -------------------------------------------------------------------------------
def test_nonfinite_step_is_skipped_but_counted(tmp_path):
    bad_steps, calls = {2}, {"n": 0}

    def cl(model, batch):
        calls["n"] += 1
        loss = mse(model, batch)
        return loss * float("nan") if (calls["n"] - 1) in bad_steps else loss

    stepped_at = []
    holder = {}

    def log(d):  # 로그는 해당 스텝이 끝난 뒤의 상태를 본다
        tr = holder["tr"]
        if d.get("event") == "train":
            after[d["step"]] = ({k: v.clone() for k, v in tr.model.state_dict().items()}, tr.scheduler.last_epoch,
                                {k: v.clone() for k, v in tr.ema.state_dict().items()}, d["skipped_steps"])

    after = {}
    tr, logs = make_trainer(tmp_path, max_steps=5, compute_loss=cl, dropout=0.0, ckpt_every_steps=100, log_fn=log)
    holder["tr"] = tr
    orig = tr.optimizer.step

    def spy_step(*a, **k):
        stepped_at.append(calls["n"] - 1)
        return orig(*a, **k)

    tr.optimizer.step = spy_step
    res = tr.fit()
    assert res.step == 5 and res.skipped_steps == 1  # 스텝 카운터는 전진한다
    assert stepped_at == [0, 1, 3, 4]  # 건너뛴 스텝에서는 optimizer.step이 호출되지 않는다
    # 인덱스 2 스텝(완료 수 2 -> 3)에서 파라미터·EMA·스케줄러가 모두 그대로
    assert_sd_equal(after[2][0], after[3][0])
    assert_sd_equal(after[2][2], after[3][2])
    assert after[2][1] == after[3][1] == 2 and after[4][1] == 3
    assert (after[2][3], after[3][3], after[5][3]) == (0, 1, 1)
    # 그 외 스텝에서는 파라미터가 실제로 움직인다
    assert not torch.equal(after[1][0]["0.weight"], after[2][0]["0.weight"])
    assert not torch.equal(after[3][0]["0.weight"], after[4][0]["0.weight"])


def test_diverged_after_too_many_consecutive_skips(tmp_path):
    def cl(model, batch):
        return mse(model, batch) * float("nan")

    tr, _ = make_trainer(tmp_path, max_steps=50, compute_loss=cl, max_consecutive_skips=3, ckpt_every_steps=100)
    w0 = {k: v.clone() for k, v in tr.model.state_dict().items()}
    with pytest.raises(TrainingDiverged):
        tr.fit()
    assert tr.step == 4  # 한도(3)를 넘은 4번째 연속 건너뛰기에서 중단
    assert_sd_equal(w0, tr.model.state_dict())
    assert issubclass(TrainingDiverged, RuntimeError)


def test_isolated_skips_do_not_accumulate_toward_divergence(tmp_path):
    n = {"i": 0}

    def cl(model, batch):
        n["i"] += 1
        loss = mse(model, batch)
        return loss * float("nan") if n["i"] % 2 == 0 else loss

    tr, _ = make_trainer(tmp_path, max_steps=10, compute_loss=cl, max_consecutive_skips=1, ckpt_every_steps=100)
    res = tr.fit()
    assert res.step == 10 and res.skipped_steps == 5


def test_fp16_autocast_run_on_cpu_stays_finite_with_fp32_masters(tmp_path):
    tr, logs = make_trainer(tmp_path, max_steps=30, dropout=0.0, ckpt_every_steps=100, lr=5e-2, sched=False,
                            grad_clip=None, amp=AmpConfig(dtype="fp16", device_type="cuda", init_scale=1024.0, growth_interval=5))
    assert tr.amp.cfg.device_type == "cpu"  # 장치에 맞춰 덮어쓴다
    assert tr.amp.scaling and tr.amp.scale_value == 1024.0
    res = tr.fit()
    t = train_logs(logs)
    assert all(torch.isfinite(torch.tensor(d["loss"])) for d in t)
    assert sum(d["loss"] for d in t[-5:]) < 0.8 * sum(d["loss"] for d in t[:5])
    assert all(p.dtype == torch.float32 for p in tr.model.parameters())
    assert res.skipped_steps == 0 and t[-1]["loss_scale"] > 1024.0  # 5스텝마다 성장


# ---- max-abs 경보 ------------------------------------------------------------------------------
def big_model():
    m = make_model(dropout=0.0)
    with torch.no_grad():
        m[0].weight.mul_(1e5)  # 첫 층 활성이 2e4를 넘는다
    return m


def test_alarm_is_reported_logged_and_does_not_stop_training(tmp_path, caplog):
    got = []
    tr, logs = make_trainer(tmp_path, max_steps=4, model=big_model(), max_abs_every=2, ckpt_every_steps=100,
                            grad_clip=1.0, ema=None, on_alarm=got.append)
    with caplog.at_level(logging.WARNING, logger="buc.train"):
        res = tr.fit()
    assert res.step == 4 and res.stop_reason == "max_steps"
    assert {a.step for a in res.alarms} == {0, 2}  # 실행 스텝 인덱스가 every의 배수일 때만 측정(0 포함)
    assert {a.layer for a in res.alarms} >= {"0"} and all(a.value > 2e4 and a.threshold == 2e4 for a in res.alarms)
    assert got == list(res.alarms)
    assert any("max-abs" in r.getMessage() and "0" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    ev = [d for d in logs if d.get("event") == "max_abs"]
    assert [d["step"] for d in ev] == [0, 2]
    assert ev[0]["n_alarms"] >= 1 and ev[0]["worst"] > 2e4 and ev[0]["worst_layer"] in ("0", "1")


def test_no_alarm_for_healthy_model_and_threshold_is_configurable(tmp_path):
    tr, logs = make_trainer(tmp_path / "a", max_steps=3, dropout=0.0, max_abs_every=1)
    res = tr.fit()
    assert res.alarms == ()
    ev = [d for d in logs if d.get("event") == "max_abs"]
    assert [d["step"] for d in ev] == [0, 1, 2] and all(d["n_alarms"] == 0 for d in ev)
    tr2, _ = make_trainer(tmp_path / "b", max_steps=1, dropout=0.0, max_abs_every=1, max_abs_threshold=1e-6)
    assert tr2.fit().alarms  # 임계값을 낮추면 건강한 모델도 걸린다


def test_monitor_hooks_are_removed_after_fit(tmp_path):
    tr, _ = make_trainer(tmp_path, max_steps=2)
    tr.fit()
    assert all(len(m._forward_hooks) == 0 for m in tr.model.modules())
    tr2, _ = make_trainer(tmp_path / "x", max_steps=3, compute_loss=lambda m, b: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        tr2.fit()
    assert all(len(m._forward_hooks) == 0 for m in tr2.model.modules())  # 예외 경로에서도 정리


# ---- 시간 예산·멱등 ----------------------------------------------------------------------------
def test_time_budget_stops_gracefully_with_checkpoint(tmp_path):
    t = [0.0]

    def cl(model, batch):
        t[0] += 10.0
        return mse(model, batch)

    tr, _ = make_trainer(tmp_path, max_steps=20, compute_loss=cl, clock=lambda: t[0], time_budget_s=35.0,
                         ckpt_every_steps=100, dropout=0.0)
    res = tr.fit()
    assert (res.step, res.stop_reason) == (4, "time_budget")  # 4스텝째 끝에 40초 >= 35초
    assert res.ckpt_path == tmp_path / "step_000000004.pt" and res.ckpt_path.exists()


def test_refit_after_completion_is_a_noop(tmp_path):
    tr, _ = make_trainer(tmp_path, max_steps=6)
    tr.fit()
    files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    calls = []
    tr2, _ = make_trainer(tmp_path, max_steps=6, model_seed=5, data_fn=lambda idx: calls.append(1) or batch_fn(idx))
    res = tr2.fit()
    assert calls == [] and (res.step, res.stop_reason, res.resumed_from) == (6, "max_steps", 6)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == files
    assert_sd_equal(tr.model.state_dict(), tr2.model.state_dict())  # 체크포인트에서 복원됨


def test_max_steps_zero_trains_nothing(tmp_path):
    tr, logs = make_trainer(tmp_path, max_steps=0)
    res = tr.fit()
    assert res.step == 0 and res.ckpt_path is None and logs == []
    assert list(tmp_path.iterdir()) == []


def test_periodic_checkpoints_are_pruned(tmp_path):
    tr, _ = make_trainer(tmp_path, max_steps=13, ckpt_every_steps=3, keep_last=2)
    tr.fit()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_000000012.pt", "step_000000013.pt"]


def test_ema_follows_the_recurrence_in_the_loop(tmp_path):
    hist, holder = [], {}

    def log(d):
        if d["event"] == "train":
            hist.append({k: v.clone() for k, v in holder["tr"].model.state_dict().items()})

    tr, _ = make_trainer(tmp_path, max_steps=5, dropout=0.0, ckpt_every_steps=100, ema=0.9, log_fn=log)
    holder["tr"] = tr
    init = {k: v.clone() for k, v in tr.model.state_dict().items()}
    tr.fit()
    ema = {k: v.clone() for k, v in init.items()}
    for snap in hist:  # 스텝이 끝날 때마다 한 번씩 갱신
        for k in ema:
            ema[k] = 0.9 * ema[k] + 0.1 * snap[k]
    got = tr.ema.state_dict()
    for k in ema:
        assert torch.allclose(got[k], ema[k], atol=1e-6), k
    assert not torch.equal(got["0.weight"], tr.model.state_dict()["0.weight"])
