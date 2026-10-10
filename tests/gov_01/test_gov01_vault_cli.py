"""GOV-01 vault 경로·마운트 규칙 2, CLI(ci/lineage_check.py) 종료 코드·JSON, 저장소 기본 검사."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from buc.gov import lineage as L
from buc.gov import prov as P

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "ci" / "lineage_check.py"


def test_constants():
    assert L.VAULT_NAMES == ("user_vault", "ext_vault", "comp_vault", "pub_bench", "pub_bench_sealed",
                             "hold_vault", "train_ok")
    assert set(L.FORBIDDEN_TRAIN_MOUNTS) == {"user_vault", "ext_vault", "comp_vault", "pub_bench",
                                             "pub_bench_sealed", "hold_vault", "soul_vault"}
    assert L.STUDENT_DIRS == ("buc/models", "buc/train")
    assert "checkpoint" in L.ARTIFACT_KINDS and "source" not in L.ARTIFACT_KINDS


def test_vault_relpath():
    assert L.vault_relpath(["USER"], tenant="t1", project="p-9") == "user_vault/t1/p-9/"
    assert L.vault_relpath(["USER-DERIVED", "EXTGEN:soul"], tenant="t", project="p") == "user_vault/t/p/"
    assert L.vault_relpath(["BIO"], tenant="t", project="p") == "user_vault/t/p/"
    for kw in ({}, {"tenant": "t"}, {"tenant": "../x", "project": "p"}, {"tenant": "t", "project": "a/b"}):
        with pytest.raises(ValueError):
            L.vault_relpath(["USER"], **kw)
    assert L.vault_relpath(["EXTGEN:soul"]) == "ext_vault/soul/"
    assert L.vault_relpath(["SOUL-DERIVED", "ENGINE"]) == "ext_vault/soul/"
    assert L.vault_relpath(["EXTGEN:soul", "EXTGEN-DERIVED:gpt-image"]) == "ext_vault/_mixed/"
    assert L.vault_relpath(["COMP:vendor"]) == "comp_vault/"
    assert L.vault_relpath(["EVAL-ONLY"]) == "comp_vault/"
    assert L.vault_relpath(["HOLDOUT"]) == "pub_bench/"
    assert L.vault_relpath(["PUB-SEALED", "HOLDOUT"]) == "pub_bench_sealed/"
    assert L.vault_relpath(["ENGINE", "CAPTURE"]) == "train_ok/"
    # 명시 ns가 더 엄격하면 그것을 따른다
    rec = P.ProvRecord(taint=frozenset({"ENGINE"}), ns="comp_vault")
    assert L.vault_relpath(rec) == "comp_vault/"


def test_check_mounts():
    ok = ["/data/train_ok/shards", "/scratch", "train_ok/"]
    assert L.check_mounts(ok) == ()
    bad = ["/mnt/USER_VAULT/t/p", "/data/ext_vault/soul", "s3://b/comp_vault/", "/x/pub_bench_sealed/r1",
           "/x/hold_vault", "/old/soul_vault", "/data/pub_bench/in", "/mnt/user_vault_backup"]
    got = L.check_mounts(ok + bad + bad[:2], purpose="train")
    assert got == tuple(sorted(set(bad)))
    assert L.check_mounts(bad[:1], purpose="dev") == (bad[0],)
    with pytest.raises(ValueError):
        L.check_mounts(ok, purpose="render")


def _run_sub(args, cwd=None):
    """실제 `python3 ci/lineage_check.py` 실행(다른 cwd에서도 저장소를 찾아야 한다)."""
    p = subprocess.run([sys.executable, str(CLI), *args], cwd=cwd or str(REPO), capture_output=True,
                       text=True, timeout=120)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
    out = json.loads(lines[-1]) if p.returncode in (0, 1) else None
    return p.returncode, out


def _cli():
    spec = importlib.util.spec_from_file_location("gov01_lineage_check", str(CLI))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(args):
    """같은 프로세스에서 main(argv) 호출(빠름). argparse 오류는 SystemExit 코드로."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            code = _cli().main(list(args))
        except SystemExit as e:
            code = e.code
    lines = [ln for ln in buf.getvalue().strip().splitlines() if ln.strip()]
    out = json.loads(lines[-1]) if code in (0, 1) else None
    return code, out


def _ledger(tmp_path, bad_tag=None):
    lg = L.LineageLedger()
    lg.add("eng", kind="source", prov=["ENGINE"])
    lg.add("shard", kind="shard", parents=["eng"], content_hash="a" * 64)
    if bad_tag:
        lg.add("bad", kind="source", prov=[bad_tag])
        lg.add("ckpt", kind="checkpoint", parents=["shard", "bad"])
    else:
        lg.add("ckpt", kind="checkpoint", parents=["shard"])
    p = tmp_path / "ledger.json"
    p.write_text(lg.to_json())
    return p


def test_cli_ledger_clean_and_canaries(tmp_path):
    code, out = _run(["--ledger", str(_ledger(tmp_path))])
    assert code == 0 and out == {"ok": True, "violations": []}
    for tag in ("USER", "EXTGEN:test"):
        p = _ledger(tmp_path, tag)
        code, out = _run(["--ledger", str(p), "--artifact", "ckpt"])
        assert code == 1 and out["ok"] is False
        assert all(v["check"] == "ancestry" for v in out["violations"])
        assert {"check": "ancestry", "where": "ckpt", "why": f"bad:{tag}"} in out["violations"]
        assert out["violations"] == [{"check": "ancestry", "where": "ckpt", "why": w}
                                     for w in L.LineageLedger.from_json(p.read_text()).why_not("ckpt")]
        code, out = _run(["--ledger", str(p), "--artifact", "shard"])
        assert code == 0
        # 기본(--artifact 없음)은 check_all
        code, out = _run(["--ledger", str(p)])
        assert code == 1 and {v["where"] for v in out["violations"]} == {"ckpt"}
    code, out = _run(["--ledger", str(_ledger(tmp_path, "EXTGEN:test")), "--purpose", "dev"])
    assert code == 1


def test_cli_tampered_ledger_and_usage(tmp_path):
    p = _ledger(tmp_path)
    d = json.loads(p.read_text())
    d["nodes"][0]["prov"]["taint"] = ["CAPTURE"]
    p.write_text(json.dumps(d))
    code, out = _run(["--ledger", str(p)])
    assert code == 1 and out["violations"][0]["check"] == "ledger"
    code, _ = _run(["--ledger", str(tmp_path / "missing.json")])
    assert code == 1
    code, _ = _run(["--purpose", "bogus"])
    assert code == 2
    code, _ = _run(["--no-such-flag"])
    assert code == 2


def test_cli_src_ckpt_mount_denylist(tmp_path):
    src = tmp_path / "student.py"
    src.write_text("import torchvision\nm = torchvision.models.raft_large(weights='DEFAULT')\n")
    code, out = _run(["--src", str(src)])
    assert code == 1
    assert out["violations"] == [{"check": "pretrained", "where": f"{src}:2", "why": "WEIGHTS"}]
    src.write_text("import torchvision\nm = torchvision.models.raft_large(weights=None)\n")
    assert _run(["--src", str(src)])[0] == 0

    torch.manual_seed(0)
    sd = torch.nn.Linear(3, 2).state_dict()
    ck = tmp_path / "init.pt"
    torch.save(sd, ck)
    dl = tmp_path / "deny.json"
    dl.write_text(L.Denylist({L.state_dict_hash(sd): "tv_raft"}).to_json())
    code, out = _run(["--ckpt", str(ck), "--denylist", str(dl)])
    assert code == 1
    assert out["violations"] == [{"check": "denylist", "where": str(ck), "why": "STATE_DICT:tv_raft"}]
    bad_dl = tmp_path / "bad.json"
    bad_dl.write_text("{")
    code, out = _run(["--ckpt", str(ck), "--denylist", str(bad_dl)])
    assert code == 1 and out["violations"][0]["check"] == "ledger"
    assert _run(["--ckpt", str(ck)])[0] == 0  # 빈 금지목록

    code, out = _run(["--mount", "/data/train_ok", "--mount", "/mnt/user_vault/t/p"])
    assert code == 1
    assert out["violations"] == [{"check": "mount", "where": "/mnt/user_vault/t/p", "why": "FORBIDDEN_MOUNT"}]
    assert _run(["--mount", "/data/train_ok"])[0] == 0


def test_cli_main_importable_and_default_scan(tmp_path, capsys):
    mod = _cli()
    root = tmp_path / "r"
    (root / "buc" / "models").mkdir(parents=True)
    (root / "buc" / "models" / "m.py").write_text("x = Foo.from_pretrained('a')\n")
    (root / "buc" / "teacher").mkdir(parents=True)
    (root / "buc" / "teacher" / "t.py").write_text("x = Foo.from_pretrained('a')\n")  # 교사는 범위 밖
    assert mod.main(["--root", str(root)]) == 1
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert [v["check"] for v in out["violations"]] == ["pretrained"]
    assert "m.py" in out["violations"][0]["where"]
    (root / "buc" / "models" / "m.py").write_text("x = 1\n")
    assert mod.main(["--root", str(root)]) == 0


def test_subprocess_entrypoint(tmp_path):
    code, out = _run_sub(["--ledger", str(_ledger(tmp_path, "USER")), "--artifact", "ckpt"],
                         cwd=str(tmp_path))
    assert code == 1 and {"check": "ancestry", "where": "ckpt", "why": "bad:USER"} in out["violations"]
    code, _ = _run_sub(["--purpose", "bogus"], cwd=str(tmp_path))
    assert code == 2


def test_repo_student_dirs_are_clean():
    code, out = _run_sub([], cwd=os.path.join(str(REPO), "ops"))
    assert code == 0, out
