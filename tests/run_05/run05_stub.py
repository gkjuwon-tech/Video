"""RUN-05 가짜 `$BUC_PYTHON`. 실제 GPU·네트워크 없이 bootstrap.sh 의 파이썬 이음매를 흉내 낸다.

환경:
  STUB_SPEC : JSON 파일. {"probe": {...}, "results": {"<key>": [{"rc":0,"out":"","sleep":0}, ...]}}
              key = BOOT_STEP, prefetch 는 "prefetch:<BOOT_MODEL>". 목록을 호출 순서대로 쓰고 마지막 것을 반복.
  STUB_LOG  : 호출마다 JSON 한 줄을 덧붙인다(step, model, argv, env 일부, constraints 내용).
비밀 값(HF_TOKEN)은 기록하지 않는다(있음/없음만).
"""
import json
import os
import sys
import time

ENV_KEYS = ("BOOT_MODEL", "HF_HOME", "TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR", "MAX_JOBS",
            "HF_HUB_ENABLE_HF_TRANSFER")


def main():
    if not sys.stdin.isatty():
        try:
            sys.stdin.read()
        except Exception:
            pass
    spec = json.load(open(os.environ["STUB_SPEC"]))
    log = os.environ["STUB_LOG"]
    step = os.environ.get("BOOT_STEP")
    model = os.environ.get("BOOT_MODEL")
    key = f"prefetch:{model}" if step == "prefetch" else step
    prior = 0
    if os.path.exists(log):
        for line in open(log):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("key") == key:
                prior += 1
    argv = sys.argv[1:]
    cons = None
    if "-c" in argv:
        i = argv.index("-c")
        if i + 1 < len(argv) and os.path.isfile(argv[i + 1]):
            cons = open(argv[i + 1]).read()
    rec = {"key": key, "step": step, "model": model, "argv": argv,
           "env": {k: os.environ.get(k) for k in ENV_KEYS},
           "has_token": bool(os.environ.get("HF_TOKEN")), "constraints": cons}
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")

    results = spec.get("results", {}).get(key) or [{}]
    r = results[min(prior, len(results) - 1)]
    if r.get("sleep"):
        time.sleep(r["sleep"])
    rc = int(r.get("rc", 0))
    if step == "probe" and "out" not in r:
        print("torch probe noise line")
        print("PROBE " + json.dumps(spec["probe"]))
    elif r.get("out"):
        print(r["out"])
    if step == "gsplat_jit" and rc == 0:
        d = os.environ.get("TORCH_EXTENSIONS_DIR")
        if d:
            os.makedirs(os.path.join(d, "gsplat_cuda"), exist_ok=True)
            with open(os.path.join(d, "gsplat_cuda", "gsplat_cuda.so"), "w") as f:
                f.write("stub")
    sys.stdout.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
