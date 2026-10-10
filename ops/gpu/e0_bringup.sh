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
sleep infinity
