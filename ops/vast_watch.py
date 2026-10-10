"""Vast.ai 꿀매물 감시: 조건을 만족하는 매물이 나오면 출력하고 종료한다(공개 API, 키 불필요).
조건: (a) 한국 5090 꿀매물(machine 35043)이 다시 렌트 가능해짐, 또는
      (b) 온디맨드 RTX 5090 ≤ $0.26/h, 신뢰도 ≥ 0.98, 중국 제외, CPU ≥ 12, RAM ≥ 30GB.
사용: python3 ops/vast_watch.py [최대 분] [간격 초]"""
import json, sys, time, urllib.request, urllib.parse
MAX_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 110
EVERY = float(sys.argv[2]) if len(sys.argv) > 2 else 180
def q(query):
    url = "https://console.vast.ai/api/v0/bundles/?q=" + urllib.parse.quote(json.dumps(query))
    return json.load(urllib.request.urlopen(url, timeout=40)).get("offers", [])
t0 = time.time()
while (time.time() - t0) / 60 < MAX_MIN:
    try:
        hits = []
        for x in q({"machine_id": {"eq": 35043}, "type": "on-demand", "limit": 20}):
            if x.get("rentable"):
                hits.append(("한국 꿀매물 복귀", x))
        for x in q({"gpu_name": {"eq": "RTX 5090"}, "rentable": {"eq": True}, "rented": {"eq": False}, "type": "on-demand",
                    "reliability2": {"gte": 0.98}, "dph_total": {"lte": 0.26}, "order": [["dph_total", "asc"]], "limit": 50}):
            loc = str(x.get("geolocation") or "")
            if "CN" in loc or "China" in loc: continue
            if (x.get("cpu_cores_effective") or 0) < 12 or (x.get("cpu_ram") or 0) < 30000: continue
            hits.append(("싼 5090", x))
        if hits:
            for tag, x in hits:
                print(f"[{tag}] id={x['id']} machine={x.get('machine_id')} {x['num_gpus']}x{x['gpu_name']} ${x['dph_total']:.3f}/h "
                      f"rel={x.get('reliability2',0):.4f} {x.get('geolocation')} cpu={x.get('cpu_cores_effective')} "
                      f"ram={(x.get('cpu_ram') or 0)/1024:.0f}GB disk={x.get('disk_space',0):.0f}GB verif={x.get('verification')}")
            sys.exit(0)
    except Exception as e:
        print("poll error:", e, file=sys.stderr)
    time.sleep(EVERY)
print("NO_HIT")
