"""TCH-11 산출물: PNG·메타·샤드 필드·디스크 쓰기·불변성·import 순수성."""
import dataclasses
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from buc.teacher import entry_gen as E
from tch11_fakes import IMG_H, IMG_W, make_sampler

ROOT = Path(__file__).resolve().parents[2]
META_KEYS = {"v", "key", "index", "prompt", "caption", "mode", "seed", "level", "k", "height", "width", "nfe",
             "table_key", "prov"}


@pytest.fixture(scope="module")
def entries(tmp_path_factory):
    sampler, _, _ = make_sampler(tmp_path_factory.mktemp("io"))
    return E.generate_entries(sampler, ["a mossy stone wall", "시장 골목의 국수 가게", "a red kite"], seed=5,
                              height=IMG_H, width=IMG_W, mode="cinematic", batch_size=2)


def test_png_roundtrip_is_lossless_and_deterministic(entries):
    for e in entries:
        b = e.png_bytes()
        assert b[:8] == b"\x89PNG\r\n\x1a\n" and b == e.png_bytes()
        im = Image.open(io.BytesIO(b))
        assert im.mode == "RGB" and im.size == (IMG_W, IMG_H)
        np.testing.assert_array_equal(np.asarray(im), e.image)


def test_image_f32_view(entries):
    for e in entries:
        f = e.image_f32()
        assert f.dtype == np.float32 and f.shape == (IMG_H, IMG_W, 3)
        np.testing.assert_array_equal(f, e.image.astype(np.float32) / np.float32(255.0))
        assert 0.0 <= f.min() and f.max() <= 1.0


def test_meta_fields(entries):
    for e in entries:
        m = e.meta()
        assert set(m) == META_KEYS
        json.dumps(m)                                  # 직렬화 가능
        assert m["v"] == E.ENTRY_TAG and m["key"] == e.key and m["index"] == e.index
        assert m["prompt"] == e.prompt and m["caption"] == e.caption and m["mode"] == "cinematic"
        assert (m["seed"], m["level"], m["k"]) == (e.seed, 0, 0)
        assert (m["height"], m["width"], m["nfe"], m["table_key"]) == (IMG_H, IMG_W, 8, e.table_key)
        assert m["prov"] == e.prov.to_dict()
        assert isinstance(m["seed"], int) and not isinstance(m["seed"], bool)


def test_shard_fields_seam_for_dat05(entries):
    for e in entries:
        f = e.shard_fields()
        assert set(f) == {"rgb.png", "meta.json"} and all(isinstance(v, bytes) for v in f.values())
        assert f["rgb.png"] == e.png_bytes()
        want = json.dumps(e.meta(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        assert f["meta.json"] == want and json.loads(f["meta.json"]) == e.meta()


def test_write_entries_files_and_idempotence(entries, tmp_path):
    out = tmp_path / "nested" / "dev_inputs"
    paths = E.write_entries(entries, out)
    assert [p.name for p in paths] == [f"{e.key}.png" for e in entries] and all(p.parent == out for p in paths)
    assert sorted(os.listdir(out)) == sorted([f"{e.key}.png" for e in entries] + [f"{e.key}.json" for e in entries])
    for e, p in zip(entries, paths):
        f = e.shard_fields()
        assert p.read_bytes() == f["rgb.png"]
        assert (out / f"{e.key}.json").read_bytes() == f["meta.json"]
    before = {n: (out / n).read_bytes() for n in os.listdir(out)}
    paths2 = E.write_entries(entries, out)                     # 같은 내용 재기록은 오류 없이 같은 결과
    assert paths2 == paths and {n: (out / n).read_bytes() for n in os.listdir(out)} == before
    assert E.write_entries([], tmp_path / "empty") == []


def test_entries_are_immutable(entries):
    e = entries[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.seed = 1
    with pytest.raises(ValueError):
        e.image[0, 0, 0] = 7                                   # 읽기 전용 배열


def test_import_is_pure_and_cpu_only():
    code = ("import sys; sys.path.insert(0, %r)\n"
            "import buc.teacher.entry_gen\n"
            "bad = [m for m in ('diffusers', 'transformers', 'huggingface_hub', 'accelerate') if m in sys.modules]\n"
            "print('BAD', bad)" % str(ROOT))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip() == "BAD []"
