"""TCH-01 load_teacher: 가짜 로더로 적재 순서·dtype·동결·지연 TE·오프라인·결정론을 검증한다."""
import gc
import json
import socket

import pytest
import torch

from buc.teacher import zimage as Z
from tch01_fakes import SPEC, Spy, block_image, collect, make_hub, make_loaders


def _load(tmp_path, spy=None, loaders=None, **kw):
    hub = kw.pop("hub_cache", None) or make_hub(tmp_path)
    if loaders is None:
        loaders, spy = make_loaders(spy)
    kw.setdefault("device", "cpu")
    kw.setdefault("te_cache_dir", tmp_path / "tecache")
    kw.setdefault("deterministic", False)
    t = Z.load_teacher(SPEC, hub_cache=hub, loaders=loaders, **kw)
    return t, spy


def test_happy_path_order_dtype_freeze_and_layers(tmp_path):
    t, spy = _load(tmp_path)
    assert isinstance(t, Z.ZImageTeacher)
    names = [c[0] for c in spy.calls]
    assert sorted(names) == ["transformer", "vae"]                       # TE·토크나이저는 아직 안 읽는다(지연)
    for name, subdir, dtype in spy.calls:
        assert subdir == name and dtype == torch.bfloat16
    assert t.snapshot.name == "a" * 40 and t.spec is SPEC
    assert t.device == torch.device("cpu") and t.dtype == torch.bfloat16
    assert len(t.layers) == SPEC.num_layers and t.layers[0] is t.transformer.layers[0]
    for m in (t.transformer,):
        assert not m.training
        for p in m.parameters():
            assert p.dtype == torch.bfloat16 and not p.requires_grad
    assert isinstance(t.vae, Z.VaeCodec) and t.vae.scaling_factor == 0.5 and t.vae.shift_factor == 0.25
    assert isinstance(t.text, Z.TextEmbedCache) and not t.text.unloaded
    assert t.fingerprints is None
    assert t.determinism is None                                          # deterministic=False면 모드를 켜지 않았다


def test_vae_module_is_frozen_eval_bf16(tmp_path):
    t, spy = _load(tmp_path)
    x = block_image(b=1, hb=2, wb=2, scale=0.5)
    z = t.encode_image(x)
    assert z.dtype == torch.float32 and tuple(z.shape) == (1, 16, 2, 2) and not z.requires_grad
    y = t.decode_latent(z)
    assert torch.allclose(y, x, atol=0.05)                                # bf16 VAE: 왕복 오차 허용
    assert torch.equal(t.encode_image(x), z)


def test_vae_shift_factor_none_means_zero(tmp_path):
    loaders, spy = make_loaders(vae_kwargs={"shift": None})
    t, _ = _load(tmp_path, loaders=loaders, spy=spy)
    assert t.vae.shift_factor == 0.0


def test_dtype_policy_fp16_needs_explicit_allow_and_runs_before_any_load(tmp_path):
    loaders, spy = make_loaders()
    with pytest.raises(Z.TeacherSpecError):
        Z.load_teacher(SPEC, device="cpu", dtype=torch.float16, hub_cache=make_hub(tmp_path), loaders=loaders,
                       deterministic=False)
    assert spy.calls == []
    t = Z.load_teacher(SPEC, device="cpu", dtype=torch.float16, hub_cache=make_hub(tmp_path / "x"), loaders=loaders,
                       deterministic=False, allow_fp16=True, te_cache_dir=tmp_path / "c")
    assert t.dtype == torch.float16 and all(p.dtype == torch.float16 for p in t.transformer.parameters())


def test_loader_returning_wrong_dtype_is_rejected_not_cast(tmp_path):
    loaders, spy = make_loaders(tr_dtype_override=torch.float32)
    with pytest.raises(Z.TeacherSpecError):
        _load(tmp_path, loaders=loaders, spy=spy)


def test_structure_checks(tmp_path):
    loaders, spy = make_loaders(tr_layers=3)                             # SPEC는 4층
    with pytest.raises(Z.TeacherSpecError):
        _load(tmp_path / "a", loaders=loaders, spy=spy)
    loaders, spy = make_loaders(vae_kwargs={"channels": 8})
    with pytest.raises(Z.TeacherSpecError):
        _load(tmp_path / "b", loaders=loaders, spy=spy)
    loaders, spy = make_loaders()
    bad = Z.Loaders(tokenizer=loaders.tokenizer, text_encoder=loaders.text_encoder, vae=loaders.vae,
                    encode_prompts=loaders.encode_prompts, transformer=lambda p, d: torch.nn.Linear(2, 2).to(d))
    with pytest.raises(Z.TeacherSpecError):                              # layers 속성 없음
        Z.load_teacher(SPEC, device="cpu", hub_cache=make_hub(tmp_path / "c"), loaders=bad, deterministic=False)


def test_validation_failures_happen_before_any_loader_call(tmp_path):
    loaders, spy = make_loaders()
    with pytest.raises(Z.TeacherSpecError):                              # 제3자 repo: 허브가 없어도 스펙 오류가 먼저
        Z.load_teacher(Z.TeacherSpec(repo="evil/repo"), device="cpu", hub_cache=tmp_path / "nohub", loaders=loaders)
    with pytest.raises(Z.TeacherWeightsMissing):
        Z.load_teacher(SPEC, device="cpu", hub_cache=tmp_path / "nohub", loaders=loaders)
    with pytest.raises(Z.TeacherWeightsMissing):
        Z.load_teacher(SPEC, device="cpu", hub_cache=make_hub(tmp_path / "h", subdirs=("transformer",)),
                       loaders=loaders)
    assert spy.calls == []


def test_device_placement_transformer_and_vae_follow_device(tmp_path):
    t, _ = _load(tmp_path, device="meta")
    assert all(p.device.type == "meta" for p in t.transformer.parameters())
    assert t.device == torch.device("meta")
    assert t.info()["weights_bytes"] > 0                                 # meta에서도 바이트는 센다


def test_te_is_lazy_loaded_on_cpu_once_then_cached(tmp_path):
    t, spy = _load(tmp_path)
    a = t.encode_text(["hello", "hello", "wor"])
    assert [c[0] for c in spy.calls].count("text_encoder") == 1
    assert [c[0] for c in spy.calls].count("tokenizer") == 1
    assert spy.te_device_seen == "cpu"                                   # 인코딩은 CPU(te_device)에서
    assert spy.encode_calls == [["hello", "wor"]]
    assert len(a) == 3 and all(x.dtype == torch.bfloat16 and x.shape[-1] == SPEC.text_dim for x in a)
    assert a[0].shape[0] == 5 and a[2].shape[0] == 3
    b = t.encode_text(["wor", "hello"])
    assert spy.encode_calls == [["hello", "wor"]] and torch.equal(b[0], a[2])      # 적중: TE 재적재/재인코딩 없음
    assert [c[0] for c in spy.calls].count("text_encoder") == 1
    assert t.text.stats["hits"] == 2


def test_text_encoder_passes_max_len_and_namespace_pins_repo_revision_dtype(tmp_path):
    t, spy = _load(tmp_path)
    long = "x" * 100
    out = t.encode_text([long])[0]
    assert out.shape[0] == SPEC.max_text_len                             # 가짜 인코더가 max_len에서 자른다
    key = t.text.key("p")
    other, _ = _load(tmp_path / "o", dtype=torch.float16, allow_fp16=True)
    assert key != other.text.key("p")                                    # dtype이 키 네임스페이스에 들어 있다
    assert SPEC.repo.encode() not in key.encode()                        # 해시이므로 평문 노출 없음


def test_release_text_encoder_drops_every_reference(tmp_path):
    t, spy = _load(tmp_path)
    t.encode_text(["a"])
    assert spy.te_weak() is not None
    t.release_text_encoder()
    collect()
    assert spy.te_weak() is None                                         # TE 모듈이 실제로 메모리에서 사라졌다
    assert t.text.unloaded
    assert t.encode_text(["a"])[0].shape[-1] == SPEC.text_dim            # 캐시 적중은 계속
    n_te = [c[0] for c in spy.calls].count("text_encoder")
    with pytest.raises(Z.TEUnloadedError):
        t.encode_text(["never seen"])
    assert [c[0] for c in spy.calls].count("text_encoder") == n_te       # 다시 적재하지 않는다


def test_release_before_first_use_never_loads_te(tmp_path):
    t, spy = _load(tmp_path)
    t.release_text_encoder()
    assert "text_encoder" not in [c[0] for c in spy.calls]
    with pytest.raises(Z.TEUnloadedError):
        t.encode_text(["a"])


def test_warm_disk_cache_means_te_is_never_loaded(tmp_path):
    t1, spy1 = _load(tmp_path)
    a = t1.encode_text(["scene", "castle"])
    t1.release_text_encoder()
    loaders2, spy2 = make_loaders()
    t2, _ = _load(tmp_path, hub_cache=t1.snapshot.parents[2], loaders=loaders2, spy=spy2)
    b = t2.encode_text(["scene", "castle"])
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert sorted(c[0] for c in spy2.calls) == ["transformer", "vae"]
    assert spy2.encode_calls == []


def test_te_hidden_size_mismatch_raises_spec_error_on_first_miss(tmp_path):
    loaders, spy = make_loaders(te_hidden=SPEC.text_dim + 1)
    t, _ = _load(tmp_path, loaders=loaders, spy=spy)                     # 적재 자체는 성공(지연)
    with pytest.raises(Z.TeacherSpecError):
        t.encode_text(["a"])


def test_no_network_during_load_and_encode(tmp_path, monkeypatch):
    def deny(*a, **k):
        raise AssertionError("TCH-01은 네트워크를 쓰지 않는다")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    t, _ = _load(tmp_path)
    t.encode_text(["a"])
    t.release_text_encoder()


def test_deterministic_flag_controls_global_state_and_seed(tmp_path):
    torch.use_deterministic_algorithms(False)
    _load(tmp_path / "off", deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    t, _ = _load(tmp_path / "on", deterministic=True, seed=5)
    assert torch.are_deterministic_algorithms_enabled()
    assert t.determinism["deterministic_algorithms"] is True and t.determinism["seed"] == 5
    assert t.determinism["tf32_matmul"] is False and t.determinism["tf32_cudnn"] is False


def test_make_noise_and_info_and_fingerprints(tmp_path):
    t1, _ = _load(tmp_path / "1", fingerprint=True)
    t2, _ = _load(tmp_path / "2", fingerprint=True)
    n1 = t1.make_noise(9, 2, 64, 96)
    assert tuple(n1.shape) == (2, 16, 8, 12) == SPEC.latent_shape(64, 96, batch=2)
    assert n1.dtype == torch.float32 and n1.device.type == "cpu"
    assert torch.equal(n1, Z.make_noise(9, (2, 16, 8, 12))) and torch.equal(n1, t2.make_noise(9, 2, 64, 96))
    with pytest.raises(ValueError):
        t1.make_noise(9, 1, 60, 96)                                      # 16의 배수 아님
    assert set(t1.fingerprints) == {"transformer", "vae"} and all(len(v) == 32 for v in t1.fingerprints.values())
    assert t1.fingerprints == t2.fingerprints
    assert t1.fingerprints["transformer"] == Z.module_fingerprint(t1.transformer)
    loaders3, spy3 = make_loaders(tr_seed=5)
    t3, _ = _load(tmp_path / "3", loaders=loaders3, spy=spy3, fingerprint=True)
    assert t3.fingerprints["transformer"] != t1.fingerprints["transformer"]
    assert t3.fingerprints["vae"] == t1.fingerprints["vae"]
    info = t1.info()
    json.dumps(info)                                                     # 직렬화 가능
    assert info["repo"] == SPEC.repo and info["dtype"] == "torch.bfloat16" and info["device"] == "cpu"
    assert info["weights_bytes"] == Z.module_bytes(t1.transformer) + Z.module_bytes(t1.vae.vae)
    assert info["te_unloaded"] is False and info["snapshot"] == str(t1.snapshot)
    assert info["fingerprints"]["transformer"] == t1.fingerprints["transformer"].hex()
    t1.release_text_encoder()
    assert t1.info()["te_unloaded"] is True
    assert _load(tmp_path / "4")[0].info()["fingerprints"] is None


def test_default_spec_is_used_when_none(tmp_path):
    # 기본 스펙은 30층: 가짜 transformer를 30층으로
    loaders30, spy30 = make_loaders(tr_layers=30)
    t = Z.load_teacher(None, device="cpu", hub_cache=make_hub(tmp_path), loaders=loaders30, deterministic=False,
                       te_cache_dir=tmp_path / "c")
    assert spy30.calls
    assert t.spec == Z.TeacherSpec() and len(t.layers) == 30
