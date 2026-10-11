"""RD-15 계약 표면: 상수, 공개 이름, 불변 자료형, import 부작용 없음, GPU 불필요."""
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import rd15_helpers as H

ROOT = Path(__file__).resolve().parents[2]

PUBLIC = ["FacesError", "TemplateHit", "HeadRender", "HeadFit", "FaceOpts", "Face", "FaceMap", "FaceRecall",
          "skin_prob", "detect_template", "sem_face_prob", "template_prob_map", "reread_window", "render_head",
          "fit_head", "combine_sigma_z", "face_map", "box_iou", "face_recall"]


def test_public_names_and_constants():
    fm = H.fm()
    assert set(PUBLIC) <= set(fm.__all__) and all(hasattr(fm, n) for n in fm.__all__)
    assert (fm.STRIDE, fm.P_ACCEPT, fm.P_SURE, fm.P_CAND, fm.FACE_H_MIN_PX) == (4, 0.2, 0.5, 0.1, 16)
    assert (fm.NMS_IOU, fm.FACE_ASPECT, fm.SEM_TEMP) == (0.3, 0.75, 0.07)
    assert tuple(fm.WIDTH_RANGE_M) == (0.12, 0.19) and fm.MIN_FIT_CELLS == 24 and fm.SIGMA_Z_FLOOR_M == 0.002
    assert fm.FIT_MAX_NORMAL_DEG == 45.0 and fm.FIT_MIN_IOU == 0.3
    assert (fm.REREAD_PAD, fm.REREAD_UP, fm.REREAD_SIGMA) == (0.25, 2, 0.25)
    assert issubclass(fm.FacesError, ValueError)


def test_value_types_are_frozen():
    fm = H.fm()
    img, gt = H.head_image(fh=60, cy=100, cx=140)
    hit = fm.detect_template(img)[0]
    ro, table, tr = H.head_readout()
    m = fm.face_map(img, ro, f_px=H.F_PX, sem_table=table, opts=fm.FaceOpts(face_ids=(H.FACE_CLS,)))
    for obj, attr in ((hit, "score"), (m, "cell_px"), (m.faces[0], "p"), (fm.FaceOpts(), "accept")):
        assert dataclasses.is_dataclass(obj)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attr, 0)


def test_import_is_cpu_only_and_has_no_side_effects():
    code = (
        "import sys, numpy as np\n"
        "s0 = np.random.get_state()[1].copy()\n"
        "import buc.read.faces\n"
        "assert (np.random.get_state()[1] == s0).all()\n"
        "bad = [m for m in ('diffusers', 'transformers', 'accelerate') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "if 'torch' in sys.modules:\n"
        "    import torch\n"
        "    assert not torch.cuda.is_initialized()\n"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert p.returncode == 0, p.stderr[-2000:]


def test_owned_file_only_depends_on_allowed_packages():
    src = (ROOT / "buc" / "read" / "faces.py").read_text(encoding="utf-8")
    for banned in ("import cv2", "import yaml", "import diffusers", "import transformers", "import mediapipe",
                   "import dlib", "import insightface", "import face_recognition"):
        assert banned not in src, banned                                    # 사전학습 얼굴 모델 금지(파라미터 0 템플릿)
    assert "cuda(" not in src and '"cuda"' not in src
