"""RD-11 공용 설정: 레포 루트와 이 폴더를 sys.path 에 넣는다(어디서 돌려도 buc, rd11_helpers 를 찾는다)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
