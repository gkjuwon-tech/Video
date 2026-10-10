"""RUN-08: 한국어 부록의 정확 일치, 비T4 수치 은닉, 비밀 가림."""
import pytest
from run08_helpers import GATE_PARAMS, fake_5090, fake_cpu, fake_t4

HEADER_ROW = "| 케이스 | 상태 | 중앙값(s) | p95(s) | 최대 메모리(GiB) | 판정 | 비고 |"
SEP_ROW = "|---|---|---|---|---|---|---|"
SECRET = "hf_" + "C" * 24


def m(t4, name, status, params, verdict, *, samples=None, reserved=None, reason=None):
    tm = t4.summarize_samples(samples, warmup=1) if samples else None
    mem = (t4.MemStats(baseline_gib=1.0, peak_allocated_gib=2.0, peak_reserved_gib=reserved)
           if reserved is not None else None)
    return t4.Measurement(name=name, params=dict(params), status=status, timing=tm, memory=mem, verdict=verdict,
                          reason=reason)


def build(t4, tt, *, gate_verdict="fail", ok_verdict="info"):
    p4k = {"mode": "fwd", "width": 3840, "height": 2160, "n_prims": 500_000}
    pfb = {"mode": "fwdbwd", "width": 1024, "height": 1024, "n_prims": 1_000_000}
    ms = (
        m(t4, "splat_fwd_3840x2160_500000", "ok", p4k, ok_verdict, samples=[0.25, 0.5, 0.75], reserved=7.5),
        m(t4, "splat_fwd_3840x2160_3000000", "oom", GATE_PARAMS, gate_verdict, reason="CUDA out of memory | big"),
        m(t4, "splat_fwdbwd_1024x1024_1000000", "ok", pfb, ok_verdict, samples=[0.125], reserved=3.25),
        m(t4, "splat_fwd_7680x4320_500000", "unmeasured",
          {"mode": "fwd", "width": 7680, "height": 4320, "n_prims": 500_000}, "info", reason="time-budget"),
    )
    return t4.Report(schema=1, device=t4.detect_device(tt), measurements=ms, created_utc="2026-10-24T01:02:03Z",
                     budget_s=None)


def test_appendix_exact_on_t4(t4):
    rep = build(t4, fake_t4(), gate_verdict="fail")
    expected = "\n".join([
        "# Shot on T4 실측 부록",
        "> 측정 장치: Tesla T4 · 15.0 GiB · torch 2.9.1+cu128 · CUDA 12.8",
        "",
        HEADER_ROW,
        SEP_ROW,
        "| splat_fwd_3840x2160_500000 | 측정 | 0.500 | 0.725 | 7.50 | - |  |",
        "| splat_fwd_3840x2160_3000000 | OOM | - | - | - | 미달 | CUDA out of memory / big |",
        "| splat_fwdbwd_1024x1024_1000000 | 측정 | 0.125 | 0.125 | 3.25 | - |  |",
        "| splat_fwd_7680x4320_500000 | 미측정 | - | - | - | - | time-budget |",
        "",
        "게이트: 4K ≤0.6 s @3M 프리미티브 — 미달",
        "",
    ])
    assert t4.render_appendix(rep) == expected


@pytest.mark.parametrize("verdict,word", [("pass", "통과"), ("fail", "미달"), ("unmeasured", "미측정")])
def test_appendix_gate_line_words(t4, verdict, word):
    ok_gate = m(t4, "g", "ok", GATE_PARAMS, verdict if verdict != "fail" else "fail", samples=[0.5], reserved=5.0)
    if verdict == "unmeasured":
        ok_gate = m(t4, "g", "unmeasured", GATE_PARAMS, "unmeasured", reason="time-budget")
    rep = t4.Report(schema=1, device=t4.detect_device(fake_t4()), measurements=(ok_gate,), created_utc=None,
                    budget_s=None)
    text = t4.render_appendix(rep)
    assert text.splitlines()[-1] == f"게이트: 4K ≤0.6 s @3M 프리미티브 — {word}"
    assert rep.gate_verdict == verdict


def test_appendix_hides_numbers_when_not_t4(t4):
    rep = build(t4, fake_5090(), gate_verdict="unmeasured")
    text = t4.render_appendix(rep)
    lines = text.splitlines()
    assert lines[0] == "# Shot on T4 실측 부록"
    assert lines[1] == "> 미측정: 이 장치는 T4가 아니다 (NVIDIA GeForce RTX 5090). T4 수치는 없다."
    assert "| splat_fwd_3840x2160_500000 | 측정 | 미측정 | 미측정 | 미측정 | - |  |" in lines
    assert "| splat_fwdbwd_1024x1024_1000000 | 측정 | 미측정 | 미측정 | 미측정 | - |  |" in lines
    assert "| splat_fwd_3840x2160_3000000 | OOM | - | - | - | 미측정 | CUDA out of memory / big |" in lines
    for leaked in ("0.500", "0.725", "7.50", "3.25", "0.125", "측정 장치", "15.0 GiB", "32.0 GiB"):
        assert leaked not in text
    assert lines[-1] == "게이트: 4K ≤0.6 s @3M 프리미티브 — 미측정"


def test_appendix_cpu_report_is_all_unmeasured(t4):
    tt = fake_cpu()
    ms = tuple(
        t4.Measurement(name=n, params=dict(p), status="unmeasured", timing=None, memory=None, verdict=v,
                       reason="no-cuda")
        for n, p, v in (("a", GATE_PARAMS, "unmeasured"),
                        ("b", {"mode": "fwdbwd", "width": 1024, "height": 1024, "n_prims": 5}, "info")))
    rep = t4.Report(schema=1, device=t4.detect_device(tt), measurements=ms, created_utc=None, budget_s=None)
    assert t4.render_appendix(rep).splitlines() == [
        "# Shot on T4 실측 부록",
        "> 미측정: 이 장치는 T4가 아니다 (cpu). T4 수치는 없다.",
        "",
        HEADER_ROW,
        SEP_ROW,
        "| a | 미측정 | - | - | - | 미측정 | no-cuda |",
        "| b | 미측정 | - | - | - | - | no-cuda |",
        "",
        "게이트: 4K ≤0.6 s @3M 프리미티브 — 미측정",
    ]


def test_appendix_error_row_and_newline_escape_and_redaction(t4):
    err = m(t4, "e", "error", {"mode": "fwd", "width": 8, "height": 8, "n_prims": 1}, "info",
            reason=f"ValueError: a|b\nline2 token={SECRET}")
    rep = t4.Report(schema=1, device=t4.detect_device(fake_t4()), measurements=(err,), created_utc=None,
                    budget_s=None)
    text = t4.render_appendix(rep)
    row = [ln for ln in text.splitlines() if ln.startswith("| e |")]
    assert len(row) == 1
    assert row[0] == "| e | 오류 | - | - | - | - | ValueError: a/b line2 token=*** |"
    assert SECRET not in text


def test_appendix_is_deterministic_and_newline_terminated(t4):
    rep = build(t4, fake_t4())
    a = t4.render_appendix(rep)
    b = t4.render_appendix(t4.Report.from_dict(rep.to_dict()))
    assert a == b
    assert a.endswith("\n") and not a.endswith("\n\n")
    assert len(a.splitlines()) <= 30


def test_appendix_cuda_version_missing_prints_na(t4):
    tt = fake_t4()
    d = t4.detect_device(tt)
    d2 = t4.DeviceInfo(kind=d.kind, name=d.name, total_mem_gib=d.total_mem_gib, capability=d.capability,
                       torch_version=d.torch_version, cuda_version=None, is_t4=True)
    rep = t4.Report(schema=1, device=d2, measurements=(), created_utc=None, budget_s=None)
    assert "· CUDA n/a" in t4.render_appendix(rep).splitlines()[1]
    assert t4.render_appendix(rep).splitlines()[-1] == "게이트: 4K ≤0.6 s @3M 프리미티브 — 미측정"
