"""재개 정확성: 중간에 멈췄다 이어간 학습은 끝까지 한 번에 돈 학습과 비트 동일하다(RNG·옵티마이저·EMA·스케줄러·샘플러)."""
import pytest
import torch
from run06_helpers import FakeHub, assert_sd_equal, losses, make_sync, make_trainer, mse

from buc.train import ckpt as C

TOTAL = 12


def snapshot(tr):
    return (
        {k: v.clone() for k, v in tr.model.state_dict().items()},
        {k: v.clone() for k, v in tr.ema.state_dict().items()},
        tr.optimizer.state_dict()["param_groups"][0]["lr"],
        tr.scheduler.last_epoch,
        tr.amp.state_dict(),
    )


def assert_snap_equal(a, b):
    assert_sd_equal(a[0], b[0])
    assert_sd_equal(a[1], b[1])
    assert a[2:] == b[2:]


@pytest.fixture
def straight(tmp_path):
    tr, logs = make_trainer(tmp_path / "straight", max_steps=TOTAL)  # dropout 0.2, AdamW, StepLR, EMA
    tr.fit()
    return snapshot(tr), losses(logs)


def test_resume_after_clean_stop_is_bit_identical(tmp_path, straight):
    d = tmp_path / "ck"
    tr1, _ = make_trainer(d, max_steps=7)
    tr1.fit()
    tr2, logs2 = make_trainer(d, max_steps=TOTAL, model_seed=123)  # 초기 가중치가 달라도 체크포인트가 덮어쓴다
    res = tr2.fit()
    assert res.resumed_from == 7 and res.step == TOTAL
    assert_snap_equal(snapshot(tr2), straight[0])
    assert losses(logs2) == {s: v for s, v in straight[1].items() if s > 7}


def test_resume_after_crash_between_checkpoints_is_bit_identical(tmp_path, straight):
    d = tmp_path / "ck"
    n = {"i": 0}

    def crashing(model, batch):
        n["i"] += 1
        if n["i"] == 10:  # 인덱스 9 스텝에서 죽는다. 마지막 체크포인트는 스텝 8
            raise RuntimeError("preempted")
        return mse(model, batch)

    tr1, _ = make_trainer(d, max_steps=TOTAL, compute_loss=crashing)
    with pytest.raises(RuntimeError, match="preempted"):
        tr1.fit()
    assert sorted(p.name for p in d.iterdir()) == ["step_000000004.pt", "step_000000008.pt"]
    tr2, logs2 = make_trainer(d, max_steps=TOTAL, model_seed=5)
    res = tr2.fit()
    assert res.resumed_from == 8 and res.step == TOTAL
    assert_snap_equal(snapshot(tr2), straight[0])
    assert losses(logs2) == {s: v for s, v in straight[1].items() if s > 8}


def test_resume_chain_of_many_short_runs(tmp_path, straight):
    d = tmp_path / "ck"
    for i, upto in enumerate((3, 4, 9, TOTAL)):
        tr, _ = make_trainer(d, max_steps=upto, model_seed=i)
        tr.fit()
    assert_snap_equal(snapshot(tr), straight[0])


def test_resume_with_time_budget_stop(tmp_path, straight):
    t = [0.0]

    def cl(model, batch):
        t[0] += 1.0
        return mse(model, batch)

    d = tmp_path / "ck"
    tr1, _ = make_trainer(d, max_steps=TOTAL, compute_loss=cl, clock=lambda: t[0], time_budget_s=5.0)
    r1 = tr1.fit()
    assert r1.stop_reason == "time_budget" and r1.step == 5
    tr2, _ = make_trainer(d, max_steps=TOTAL)
    assert tr2.fit().step == TOTAL
    assert_snap_equal(snapshot(tr2), straight[0])


def test_resume_preserves_amp_scaler_state(tmp_path):
    from buc.train.amp import AmpConfig
    amp = AmpConfig(dtype="fp16", device_type="cpu", init_scale=64.0, growth_interval=4)
    full, _ = make_trainer(tmp_path / "f", max_steps=10, dropout=0.0, amp=amp, ema=None)
    full.fit()
    d = tmp_path / "ck"
    a, _ = make_trainer(d, max_steps=6, dropout=0.0, amp=amp, ema=None)
    a.fit()
    b, _ = make_trainer(d, max_steps=10, dropout=0.0, amp=amp, ema=None, model_seed=9)
    b.fit()
    assert b.amp.state_dict() == full.amp.state_dict()
    assert b.amp.scale_value > 64.0  # 성장 카운터가 재개를 가로질러 이어졌다
    assert_sd_equal(full.model.state_dict(), b.model.state_dict())


def test_config_change_blocks_resume_but_schedule_change_does_not(tmp_path):
    d = tmp_path / "ck"
    make_trainer(d, max_steps=4)[0].fit()
    tr, _ = make_trainer(d, max_steps=8, batch_size=16)
    with pytest.raises(C.CheckpointMismatchError):
        tr.fit()
    tr, _ = make_trainer(d, max_steps=8, seed=8)
    with pytest.raises(C.CheckpointMismatchError):
        tr.fit()
    tr, _ = make_trainer(d, max_steps=8, ckpt_every_steps=2, keep_last=5, log_every=3)  # 일정 변경은 허용
    assert tr.fit().step == 8


def test_corrupt_latest_checkpoint_falls_back_and_still_matches(tmp_path, straight):
    d = tmp_path / "ck"
    make_trainer(d, max_steps=8)[0].fit()
    (d / "step_000000008.pt").write_bytes(b"truncated")  # 마지막 파일이 깨졌다
    tr, _ = make_trainer(d, max_steps=TOTAL, model_seed=2)
    res = tr.fit()
    assert res.resumed_from == 4
    assert_snap_equal(snapshot(tr), straight[0])


# ---- RUN-03 허브와의 이음매 --------------------------------------------------------------------
def test_hub_roundtrip_resume_on_a_fresh_machine(tmp_path, straight):
    hub = FakeHub()
    tr1, _ = make_trainer(tmp_path / "gpu_a", max_steps=8, hub=make_sync(hub))
    tr1.fit()
    # 최종 체크포인트는 주기와 무관하게 업로드된다(완성본 = 데이터 + 사이드카)
    assert "ckpt/r06/step_000000008.pt" in hub.files and "ckpt/r06/step_000000008.pt.sha256.json" in hub.files
    # 다른 서버: 로컬 디스크가 비어 있고 허브만 있다
    tr2, _ = make_trainer(tmp_path / "gpu_b", max_steps=TOTAL, model_seed=77, hub=make_sync(hub))
    res = tr2.fit()
    assert res.resumed_from == 8 and res.step == TOTAL
    assert_snap_equal(snapshot(tr2), straight[0])
    assert "ckpt/r06/step_000000012.pt" in hub.files  # 이어간 끝도 올라간다


def test_training_continues_when_the_hub_is_down(tmp_path):
    from buc.rt import hf_sync as H

    class Down(FakeHub):
        def upload_file(self, local, remote):
            raise H.TransientHubError("down")

        def list_files(self, prefix):
            raise H.TransientHubError("down")

    tr, _ = make_trainer(tmp_path, max_steps=6, hub=make_sync(Down()))
    res = tr.fit()
    assert res.step == 6 and res.ckpt_path.exists()


def test_no_token_hub_is_a_silent_noop(tmp_path):
    from buc.rt import hf_sync as H
    tr, _ = make_trainer(tmp_path, max_steps=5, hub=H.HubSync(None))
    assert tr.fit().step == 5


def test_periodic_upload_uses_injected_clock(tmp_path):
    t = [0.0]
    hub = FakeHub()

    def cl(model, batch):
        t[0] += 10.0
        return mse(model, batch)

    tr, _ = make_trainer(tmp_path, max_steps=9, compute_loss=cl, clock=lambda: t[0], hub=make_sync(hub),
                         ckpt_every_steps=3, upload_every_seconds=50.0, keep_last=10)
    tr.fit()
    uploaded = sorted(k for k in hub.files if k.endswith(".pt"))
    # 저장: 3,6 (스텝 주기), 9(최종). 업로드: 6스텝째(경과 60s >= 50s)와 최종 9. 3스텝째(30s)는 이르다.
    assert uploaded == ["ckpt/r06/step_000000006.pt", "ckpt/r06/step_000000009.pt"]
