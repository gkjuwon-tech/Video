"""작업 상태 갱신 + 장부 기록. 사용: python3 ops/harness/update.py ID {done|blocked|todo} [--models a,b] [--note "..."]"""
import argparse, json, os, time
D = os.path.dirname(__file__)
ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("status", choices=["done", "blocked", "todo", "doing"])
ap.add_argument("--models", default=""); ap.add_argument("--note", default="")
a = ap.parse_args()
B = json.load(open(os.path.join(D, "backlog.json"), encoding="utf-8"))
x = next(i for i in B["items"] if i["id"] == a.id)
x["status"] = a.status; x["attempts"] += 1
ev = {"ts": int(time.time()), "event": a.status, "models": a.models, "note": a.note[:500]}
x["log"].append(ev)
json.dump(B, open(os.path.join(D, "backlog.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(D, "ledger.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps({"id": a.id, **ev}, ensure_ascii=False) + "\n")
print(json.dumps({"id": a.id, "status": a.status}))
