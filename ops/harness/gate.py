"""기계 게이트. 모델 토큰을 쓰기 전에 기계로 거른다.
  --freeze ID : 계약 작성자가 테스트를 쓴 뒤 테스트 파일 해시를 동결한다(구현자의 테스트 조작 방지).
  ID          : 동결 해시 일치 + 소유 경로 존재 + 문법 + ruff(치명 규칙) + pytest(GPU 마크 제외)를 검사하고 JSON을 출력한다."""
import glob, hashlib, json, os, shutil, subprocess, sys
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(R, "ops", "harness")
def tdir(i): return os.path.join(R, "tests", i.lower().replace("-", "_"))
def files(i): return sorted(glob.glob(os.path.join(tdir(i), "**", "*.py"), recursive=True))
def digest(i):
    h = {}
    for f in files(i):
        h[os.path.relpath(f, R)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    return h
def item(i):
    return next(x for x in json.load(open(os.path.join(D, "backlog.json"), encoding="utf-8"))["items"] if x["id"] == i)
def run(cmd):
    p = subprocess.run(cmd, cwd=R, capture_output=True, text=True, timeout=1200)
    return p.returncode, (p.stdout + p.stderr)[-4000:]
def main():
    if sys.argv[1] == "--freeze":
        i = sys.argv[2]; h = digest(i)
        if not h: print(json.dumps({"ok": False, "why": f"테스트 없음: {tdir(i)}"})); sys.exit(1)
        json.dump(h, open(os.path.join(D, "contracts", f"{i}.tests.json"), "w"), indent=1)
        print(json.dumps({"ok": True, "frozen": list(h)})); return
    i = sys.argv[1]; out = {"id": i, "checks": {}}
    fz = os.path.join(D, "contracts", f"{i}.tests.json")
    if not os.path.exists(fz): out["checks"]["frozen"] = "동결 해시 없음"
    else:
        exp = json.load(open(fz)); now = digest(i)
        bad = [f for f in exp if now.get(f) != exp[f]]
        out["checks"]["frozen"] = "ok" if not bad else f"테스트 변경됨: {bad}"
    it = item(i)
    missing = [p for p in it["paths"] if not glob.glob(os.path.join(R, p.split(" ")[0]))]
    out["checks"]["paths"] = "ok" if not missing else f"없는 경로: {missing}"
    pys = [p for p in it["paths"] if p.endswith(".py") and os.path.exists(os.path.join(R, p))]
    if pys:
        c, o = run([sys.executable, "-m", "py_compile", *pys]); out["checks"]["compile"] = "ok" if c == 0 else o
        c, o = run([shutil.which("ruff") or "ruff", "check", "--select", "E9,F63,F7,F82", *pys]); out["checks"]["ruff"] = "ok" if c == 0 else o
    if files(i):
        c, o = run([sys.executable, "-m", "pytest", "-q", "-m", "not gpu", "-x", tdir(i)])
        out["checks"]["pytest"] = "ok" if c == 0 else o
    else:
        out["checks"]["pytest"] = "테스트 없음"
    out["pass"] = all(v == "ok" for v in out["checks"].values())
    print(json.dumps(out, ensure_ascii=False)); sys.exit(0 if out["pass"] else 1)
if __name__ == "__main__":
    main()
