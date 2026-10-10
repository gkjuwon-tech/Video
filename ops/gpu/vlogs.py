"""Vast 인스턴스 로그를 가져온다(CLI의 9초 대기 한계를 우회해 결과 URL을 최대 120초 폴링). 키는 ~/.config/vastai/vast_api_key 에서 읽고 출력하지 않는다.
사용: python3 ops/gpu/vlogs.py INSTANCE_ID [tail]"""
import json, os, sys, time, urllib.request
iid, tail = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "400")
key = open(os.path.expanduser("~/.config/vastai/vast_api_key")).read().strip()
req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/request_logs/{iid}/", method="PUT",
                             data=json.dumps({"tail": tail}).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=60))
url = r.get("result_url")
if not url:
    print("no result_url:", {k: v for k, v in r.items() if k != "result_url"}); sys.exit(1)
for _ in range(60):
    try:
        print(urllib.request.urlopen(url, timeout=30).read().decode(errors="replace")); sys.exit(0)
    except Exception:
        time.sleep(2)
print("logs not ready after 120s"); sys.exit(2)
