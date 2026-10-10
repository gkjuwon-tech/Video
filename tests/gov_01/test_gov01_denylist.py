"""GOV-01 체크포인트 해시 금지목록: file_sha256, state_dict_hash(독립 오라클), check_checkpoint."""
import hashlib
import json
import struct

import numpy as np
import pytest
import torch

from buc.gov import lineage as L
from buc.gov import prov as P


def oracle_sd_hash(sd):
    h = hashlib.sha256(b"buc.sd.v1")
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        kb = k.encode("utf-8")
        db = str(t.dtype).encode("utf-8")
        h.update(struct.pack("<I", len(kb)) + kb)
        h.update(struct.pack("<I", len(db)) + db)
        h.update(struct.pack("<I", t.dim()))
        for d in t.shape:
            h.update(struct.pack("<Q", d))
        raw = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
        h.update(struct.pack("<Q", len(raw)) + raw)
    return h.hexdigest()


def make_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.GroupNorm(2, 4), torch.nn.Linear(4, 2))


def test_constants():
    assert L.SD_TAG == b"buc.sd.v1"
    assert L.DENYLIST_TAG == "buc.denylist.v1"


def test_state_dict_hash_matches_oracle_and_invariances():
    sd = make_model(0).state_dict()
    sd["buf_i64"] = torch.arange(5, dtype=torch.int64)
    sd["half"] = torch.linspace(0, 1, 7).half()
    sd["empty"] = torch.zeros(0, 3)
    h = L.state_dict_hash(sd)
    assert h == oracle_sd_hash(sd)
    assert len(h) == 64 and h == h.lower()
    rev = dict(reversed(list(sd.items())))
    assert L.state_dict_hash(rev) == h
    # requires_grad·비연속 뷰와 무관
    sd2 = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v.clone()) for k, v in sd.items()}
    assert L.state_dict_hash(sd2) == h
    w = sd["0.weight"]
    sd3 = dict(sd)
    sd3["0.weight"] = w.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)
    assert not sd3["0.weight"].is_contiguous()
    assert L.state_dict_hash(sd3) == h


def test_state_dict_hash_sensitive():
    sd = make_model(0).state_dict()
    h = L.state_dict_hash(sd)
    assert L.state_dict_hash(make_model(0).state_dict()) == h  # 같은 시드 결정론
    assert L.state_dict_hash(make_model(1).state_dict()) != h
    one = {k: v.clone() for k, v in sd.items()}
    flat = one["2.bias"].view(-1)
    flat[0] = torch.nextafter(flat[0], torch.tensor(1e9))  # 1 ulp
    assert L.state_dict_hash(one) != h
    ren = {("x." + k if k == "2.bias" else k): v for k, v in sd.items()}
    assert L.state_dict_hash(ren) != h
    dt = dict(sd)
    dt["2.bias"] = sd["2.bias"].double()
    assert L.state_dict_hash(dt) != h
    rs = dict(sd)
    rs["2.bias"] = sd["2.bias"].reshape(1, 2)
    assert L.state_dict_hash(rs) != h


def test_denylist_match_and_json(tmp_path):
    dl = L.Denylist({"ff5fadd5": "raft_large", "AB" * 32: "hrnet_imagenet"})
    assert dl.match("ff5fadd5" + "1" * 56) == "raft_large"
    assert dl.match("FF5FADD5" + "1" * 56) == "raft_large"
    assert dl.match("ab" * 32) == "hrnet_imagenet"
    assert dl.match("ff5fadd4" + "1" * 56) is None
    assert dl.match("ab" * 31 + "ac") is None
    p = tmp_path / "deny.json"
    p.write_text(dl.to_json())
    d = json.loads(p.read_text())
    assert d["v"] == "buc.denylist.v1"
    assert {(e["sha256"], e["name"]) for e in d["entries"]} == {("ff5fadd5", "raft_large"),
                                                                  ("ab" * 32, "hrnet_imagenet")}
    dl2 = L.Denylist.load(str(p))
    assert dl2.match("ff5fadd5" + "0" * 56) == "raft_large"
    for bad in ({"v": "x", "entries": []}, {"v": "buc.denylist.v1", "entries": [{"sha256": "abc", "name": "s"}]},
                {"v": "buc.denylist.v1", "entries": [{"sha256": "zz" * 8, "name": "s"}]},
                {"v": "buc.denylist.v1", "entries": [{"sha256": "a" * 65, "name": "s"}]}):
        p.write_text(json.dumps(bad))
        with pytest.raises(L.LineageError):
            L.Denylist.load(str(p))
    p.write_text("{not json")
    with pytest.raises(L.LineageError):
        L.Denylist.load(str(p))
    with pytest.raises(L.LineageError):
        L.Denylist({"1234567": "too short"})


def test_check_checkpoint(tmp_path):
    m = make_model(0)
    sd = m.state_dict()
    p1, p2, p3 = tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "c.pt"
    torch.save(sd, p1)
    torch.save({"state_dict": sd, "step": 3}, p2)  # 다시 감싼 파일: 파일 해시는 달라도 내용은 같다
    torch.save(make_model(5).state_dict(), p3)
    fh = hashlib.sha256(p1.read_bytes()).hexdigest()
    assert L.file_sha256(str(p1)) == fh
    sh = L.state_dict_hash(sd)

    assert L.check_checkpoint(str(p1), L.Denylist()) == ()
    assert L.check_checkpoint(str(p1), L.Denylist({fh[:12]: "pre"})) == ("FILE:pre",)
    dl = L.Denylist({sh: "imagenet_init"})
    assert L.check_checkpoint(str(p1), dl) == ("STATE_DICT:imagenet_init",)
    assert L.check_checkpoint(str(p2), dl) == ("STATE_DICT:imagenet_init",)
    assert L.check_checkpoint(str(p3), dl) == ()
    both = L.Denylist({sh: "s", fh: "f"})
    assert L.check_checkpoint(str(p1), both) == ("FILE:f", "STATE_DICT:s")

    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"\x00not a checkpoint")
    assert L.check_checkpoint(str(junk), L.Denylist()) == ("UNREADABLE",)
    torch.save([1, 2, 3], tmp_path / "list.pt")
    assert L.check_checkpoint(str(tmp_path / "list.pt"), L.Denylist()) == ("UNREADABLE",)
    assert L.check_checkpoint(str(tmp_path / "missing.pt"), L.Denylist()) == ("UNREADABLE",)


def test_check_checkpoint_never_unpickles_code(tmp_path):
    class Evil:
        def __reduce__(self):
            return (exec, ("open(%r, 'w').write('pwned')" % str(tmp_path / "pwned"),))

    p = tmp_path / "evil.pt"
    torch.save({"w": torch.ones(2), "x": Evil()}, p)
    assert L.check_checkpoint(str(p), L.Denylist()) == ("UNREADABLE",)
    assert not (tmp_path / "pwned").exists()


def test_assert_from_scratch():
    m = make_model(0)
    L.assert_from_scratch(m, L.Denylist())
    L.assert_from_scratch(m.state_dict(), L.Denylist({"0" * 64: "other"}))
    dl = L.Denylist({L.state_dict_hash(m.state_dict()): "torchvision_resnet18"})
    with pytest.raises(P.FirewallError) as ei:
        L.assert_from_scratch(m, dl)
    assert "torchvision_resnet18" in str(ei.value)
    np.testing.assert_equal(L.state_dict_hash(m.state_dict()), oracle_sd_hash(m.state_dict()))
