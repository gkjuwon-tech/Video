"""하네스 작업 선택기: 의존성이 끝난 작업 중 우선순위(수직 슬라이스 → 순번)대로, 소유 경로가 겹치지 않게 k개를 고른다.
사용: python3 ops/harness/pick.py --k 2 [--claim] [--slice-only]   → JSON 출력"""
import argparse, json, os, time
P = os.path.join(os.path.dirname(__file__), "backlog.json")

def top(p):  # 경로 충돌 판정 단위: 파일이면 파일 자체, 디렉터리면 그 디렉터리
    return p.strip().rstrip("/")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=2); ap.add_argument("--claim", action="store_true"); ap.add_argument("--reset-doing", action="store_true"); ap.add_argument("--slice-only", action="store_true", help="수직 슬라이스 단위와 그 미완료 조상만 고른다(사용량 절약용)")
    a = ap.parse_args()
    B = json.load(open(P, encoding="utf-8")); items = B["items"]; by = {x["id"]: x for x in items}
    if a.reset_doing:
        for x in items:
            if x["status"] == "doing": x["status"] = "todo"; x["log"].append({"ts": int(time.time()), "event": "reset"})
        json.dump(B, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    busy = {top(p) for x in items if x["status"] == "doing" for p in x["paths"]}
    ready = [x for x in items if x["status"] == "todo" and all(by.get(d, {}).get("status") == "done" for d in x["deps"])]
    if a.slice_only:
        keep = set()
        def up(i):
            for d in by[i]["deps"]:
                if by.get(d, {}).get("status") != "done" and d not in keep:
                    keep.add(d); up(d)
        for x in items:
            if x.get("slice") and x["status"] != "done":
                keep.add(x["id"]); up(x["id"])
        ready = [x for x in ready if x["id"] in keep]
    ready.sort(key=lambda x: (not x["slice"], x["rank"]))
    chosen = []
    for x in ready:
        ps = {top(p) for p in x["paths"]}
        if ps & busy: continue
        chosen.append(x); busy |= ps
        if len(chosen) >= a.k: break
    if a.claim:
        for x in chosen:
            x["status"] = "doing"; x["log"].append({"ts": int(time.time()), "event": "claimed"})
        json.dump(B, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    keys = ["id", "name", "paths", "deps", "diff", "risk", "screen", "verify", "impl_model", "impl_effort", "design", "review", "slice", "why"]
    done = sum(x["status"] == "done" for x in items)
    print(json.dumps({"items": [{k: x[k] for k in keys} for x in chosen], "ready": len(ready), "done": done, "total": len(items)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
