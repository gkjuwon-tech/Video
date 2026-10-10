#!/usr/bin/env bash
# E0 기동: GPU 서버에서 엔진 교사(Z-Image-Turbo)를 BF16 그대로 올리고 속도·메모리·출력 정상 여부를 잰다.
# 결과는 표준출력의 E0_* 줄로만 낸다(우리 컨테이너는 `vastai logs`로 읽는다). 비밀 값은 쓰지 않는다.
set -u
echo "E0_START $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h / | tail -1
python - <<'PY'
import json, os, torch
print("E0_ENV", json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                            "cap": list(torch.cuda.get_device_capability(0)), "cpus": os.cpu_count(), "bf16": torch.cuda.is_bf16_supported()}))
PY
pip install -q --no-cache-dir -U diffusers transformers accelerate safetensors hf_transfer sentencepiece 2>&1 | tail -2
python -c "from diffusers import ZImagePipeline" 2>/dev/null || pip install -q --no-cache-dir git+https://github.com/huggingface/diffusers
python -c "import diffusers; print('E0_DIFFUSERS', diffusers.__version__)"
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
import base64, io, json, time
import numpy as np, torch
from huggingface_hub import snapshot_download
from diffusers import ZImagePipeline
res = {}
t = time.time(); path = snapshot_download("Tongyi-MAI/Z-Image-Turbo", ignore_patterns=["assets/*"]); res["download_s"] = round(time.time() - t, 1)
t = time.time(); pipe = ZImagePipeline.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False); pipe.to("cuda")
res["load_s"] = round(time.time() - t, 1); res["vram_after_load_GB"] = round(torch.cuda.memory_allocated() / 1e9, 2)
prompt = ("A quiet old restaurant interior in Euljiro, Seoul, overcast afternoon light from the window, "
          "stainless steel table with three green soju bottles covered in condensation, handwritten Korean menu on the wall")
def run(h, w, seed):
    torch.cuda.synchronize(); t0 = time.time()
    img = pipe(prompt=prompt, height=h, width=w, num_inference_steps=9, guidance_scale=0.0,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    torch.cuda.synchronize(); return img, time.time() - t0
run(1024, 1024, 0)  # 예열
ts = [run(1024, 1024, s)[1] for s in (1, 2, 3)]
res["t_1024_s"] = [round(x, 2) for x in ts]; res["t_per_fwd_1024_s"] = round(sum(ts) / len(ts) / 8, 3)
res["t_512_s"] = round(run(512, 512, 4)[1], 2)
img, _ = run(1024, 1024, 7); a = np.asarray(img).astype(np.float32)
res["img_ok"] = bool(np.isfinite(a).all() and a.std() > 1.0)
res["peak_vram_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print("E0_RESULT", json.dumps(res), flush=True)
buf = io.BytesIO(); img.resize((320, 320)).save(buf, format="JPEG", quality=82)
b = base64.b64encode(buf.getvalue()).decode()
for i in range(0, len(b), 1500): print(f"E0_IMG {i // 1500:03d} {b[i:i + 1500]}", flush=True)
PY
echo "E0_DONE $(date -u +%FT%TZ)"
# ---- 상주 루프: 러너 갱신 + 데드맨 스위치 ----
# 1) 레포에 ops/gpu/runner.sh 가 생기면 내려받아 실행한다(배포 = 깃허브 푸시).
# 2) 오케스트레이터 하트비트(ops/gpu/heartbeat.json)가 IDLE_MAX초 넘게 갱신되지 않고 실행 중인 잡이 없으면 스스로 정지한다.
python -m pip install -q --no-cache-dir vastai >/dev/null 2>&1 || true
echo "E0_ENV_KEYS has_container_key=$([ -n "${CONTAINER_API_KEY:-}" ] && echo yes || echo no) has_container_id=$([ -n "${CONTAINER_ID:-}" ] && echo yes || echo no)"
BASE='https://raw.githubusercontent.com/gkjuwon-tech/Video/refs/heads/claude/video-generation-new-paradigm-qriszj'
IDLE_MAX=${IDLE_MAX:-9000}
fetch() { python -c "import urllib.request,sys; sys.stdout.write(urllib.request.urlopen(sys.argv[1]+'?t='+sys.argv[2], timeout=30).read().decode())" "$BASE/$1" "$(date +%s)"; }
self_stop() {
  echo "WATCHDOG_STOP $(date -u +%FT%TZ) reason=$1"
  [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" || true
  sleep 30; exit 0
}
while true; do
  if fetch ops/gpu/runner.sh > /tmp/runner.sh 2>/dev/null && [ -s /tmp/runner.sh ]; then
    echo "RUNNER_FOUND $(date -u +%FT%TZ)"; exec bash /tmp/runner.sh
  fi
  hb=$(fetch ops/gpu/heartbeat.json 2>/dev/null | python -c "import json,sys; print(int(json.load(sys.stdin).get('epoch',0)))" 2>/dev/null || echo 0)
  now=$(date +%s); age=$(( now - hb ))
  echo "HEARTBEAT age=${age}s"
  [ "$age" -gt "$IDLE_MAX" ] && self_stop "heartbeat_stale_${age}s"
  sleep 300
done
