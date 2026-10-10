"""GOV-01 Lineage CI CLI.

원장 조상 검사(BLOCK_TRAIN/DEV, 금지목록), 체크포인트 금지목록, weights=None 린트,
학습 마운트 금지 검사를 한 번에 돈다. stdout 마지막 줄은 JSON
`{"ok": bool, "violations": [{"check","where","why"}...]}`.

종료 코드: 0 = 위반 없음, 1 = 위반(깨진·변조 원장·금지목록 포함), 2 = 사용법 오류.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from buc.gov import lineage as L  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lineage_check", allow_abbrev=False,
                                description="BUC Lineage Ledger CI")
    p.add_argument("--ledger", help="원장 JSON 경로")
    p.add_argument("--artifact", action="append", default=[], help="검사할 아티팩트 id (반복)")
    p.add_argument("--purpose", choices=L._PURPOSES, default="train")
    p.add_argument("--branch", choices=L._BRANCHES, default="main")
    p.add_argument("--denylist", help="체크포인트 금지목록 JSON 경로")
    p.add_argument("--ckpt", action="append", default=[], help="체크포인트 경로 (반복)")
    p.add_argument("--src", action="append", default=[], help="weights=None 린트 대상 파일·디렉터리 (반복)")
    p.add_argument("--mount", action="append", default=[], help="학습 잡 마운트 (반복)")
    p.add_argument("--root", default=_REPO, help="저장소 루트 (기본: 이 저장소)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.artifact and not args.ledger:
        parser.error("--artifact 는 --ledger 와 함께 써야 한다")
    if args.purpose == "dev" and args.branch != "main":
        parser.error("--purpose dev 는 --branch main 만 가능하다")

    found: set = set()

    def add(check: str, where: str, why: str) -> None:
        found.add((check, where, why))

    denylist = L.Denylist()
    if args.denylist:
        try:
            denylist = L.Denylist.load(args.denylist)
        except L.LineageError as e:
            add("ledger", args.denylist, str(e))

    src = list(args.src)
    if not (args.ledger or args.ckpt or src or args.mount):
        src = [os.path.join(args.root, d) for d in L.STUDENT_DIRS if os.path.isdir(os.path.join(args.root, d))]

    if args.ledger:
        try:
            with open(args.ledger, "r", encoding="utf-8") as f:
                ledger = L.LineageLedger.from_json(f.read())
        except (OSError, ValueError) as e:  # LineageError 는 ValueError 하위
            add("ledger", args.ledger, str(e))
        else:
            if args.artifact:
                for aid in args.artifact:
                    try:
                        for why in ledger.why_not(aid, args.purpose, args.branch, denylist):
                            add("ancestry", aid, why)
                    except KeyError:
                        add("ledger", args.ledger, f"원장에 없는 아티팩트: {aid}")
            else:
                for aid, whys in ledger.check_all(args.purpose, args.branch, denylist).items():
                    for why in whys:
                        add("ancestry", aid, why)

    for ck in args.ckpt:
        for why in L.check_checkpoint(ck, denylist):
            add("denylist", ck, why)

    if src:
        for v in L.scan_paths(src):
            add("pretrained", f"{v.path}:{v.line}", v.rule)

    for m in L.check_mounts(args.mount, args.purpose):
        add("mount", m, "FORBIDDEN_MOUNT")

    violations = [{"check": c, "where": w, "why": y} for c, w, y in sorted(found)]
    for v in violations:
        print(f"VIOLATION {v['check']} {v['where']} {v['why']}", file=sys.stderr)
    print(json.dumps({"ok": not violations, "violations": violations}, ensure_ascii=False,
                     separators=(",", ":")))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
