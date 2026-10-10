"""GPU 인스턴스가 기한 안에 실제로 뜨는지 지킨다. 안 뜨면 파기하고 대체 5090을 빌려 같은 부팅 스크립트로 다시 띄운다.
키는 vastai CLI 기본 위치(~/.config/vastai/vast_api_key)에서 CLI가 직접 읽는다. 키를 출력하지 않는다.
사용: python3 ops/gpu/ensure_up.py INSTANCE_ID [첫 기한(분, 생성 시각 기준)=50] [재시도 기한(분)=25] [최대 대체 횟수=3]
결과: ops/gpu/instance.json 에 현재 인스턴스 기록. 성공 시 'UP <id>' 출력 후 종료 코드 0."""
import json, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REC = os.path.join(REPO, "ops/gpu/instance.json")
IMAGE = "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"
BOOT = ("until python -c \"import urllib.request,sys; sys.stdout.write(urllib.request.urlopen("
        "'https://raw.githubusercontent.com/gkjuwon-tech/Video/refs/heads/claude/video-generation-new-paradigm-qriszj/"
        "ops/gpu/e0_bringup.sh').read().decode())\" > /tmp/e0.sh; do echo waiting-for-script; sleep 15; done; bash /tmp/e0.sh")
MAX_DPH = 0.50


def vast(*args, timeout=60):
    # --args는 나머지 인자를 전부 먹으므로 --raw는 하위 명령 바로 뒤에 둔다.
    r = subprocess.run(["vastai", *args[:2], "--raw", *args[2:]], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''} 실패: {r.stderr.strip()[:200]}")
    return json.loads(r.stdout)


def record(**kw):
    kw["updated"] = int(time.time())
    json.dump(kw, open(REC, "w"), ensure_ascii=False, indent=1)


def pick_offer(exclude_machines):
    q = ("gpu_name=RTX_5090 num_gpus=1 rentable=true verified=true reliability>=0.98 cpu_cores_effective>=12 "
         "cpu_ram>=30 inet_down>=300 disk_space>=100 cuda_vers>=12.8")
    offers = vast("search", "offers", q, "-o", "dph_total")
    for x in offers:
        loc = str(x.get("geolocation") or "")
        if "CN" in loc or "China" in loc or x.get("machine_id") in exclude_machines:
            continue
        if x.get("dph_total", 9) <= MAX_DPH:
            return x
    return None


def wait_up(iid, deadline_ts):
    while time.time() < deadline_ts:
        try:
            d = vast("show", "instance", str(iid))
            st = d.get("actual_status")
            print(time.strftime("%H:%M:%SZ", time.gmtime()), iid, "actual_status=", st, "msg=", (d.get("status_msg") or "")[:80], flush=True)
            if st == "running":
                return True
            if st in ("exited", "offline"):
                return False
        except Exception as e:
            print("poll error:", e, flush=True)
        time.sleep(120)
    return False


def main():
    iid = int(sys.argv[1])
    first_min = float(sys.argv[2]) if len(sys.argv) > 2 else 50
    retry_min = float(sys.argv[3]) if len(sys.argv) > 3 else 25
    max_swaps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    d = vast("show", "instance", str(iid))
    bad = {d.get("machine_id")}
    deadline = d["start_date"] + first_min * 60
    for swap in range(max_swaps + 1):
        record(id=iid, machine=d.get("machine_id"), dph=d.get("dph_total"), state="waiting", swap=swap)
        if wait_up(iid, deadline):
            record(id=iid, machine=d.get("machine_id"), dph=d.get("dph_total"), state="running", swap=swap)
            print("UP", iid, flush=True)
            return 0
        if swap == max_swaps:
            break
        print("기한 초과: 인스턴스", iid, "파기 후 대체", flush=True)
        try:
            vast("destroy", "instance", str(iid))
        except Exception as e:
            print("destroy 실패:", e, flush=True)
        offer = pick_offer(bad)
        if offer is None:
            record(id=None, state="no_offer", swap=swap)
            print("대체 매물 없음 (5090, 검증됨, ≤$%.2f/h)" % MAX_DPH, flush=True)
            return 3
        r = vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", "90",
                 "--label", "buc-gpu-%d" % (swap + 1), "--entrypoint", "/bin/bash", "--args", "-c", BOOT)
        iid = int(r["new_contract"])
        bad.add(offer.get("machine_id"))
        print(f"대체 임대: {iid} machine={offer.get('machine_id')} ${offer['dph_total']:.3f}/h {offer.get('geolocation')}", flush=True)
        time.sleep(20)
        d = vast("show", "instance", str(iid))
        deadline = time.time() + retry_min * 60
    record(id=iid, state="failed")
    print("FAILED", iid, flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
