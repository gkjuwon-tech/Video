#!/usr/bin/env bash
# 중간보스 LTX-2.5로 D15 야크 6샷 I2V 생성 (경쟁 비교용, COMP 태그: 평가 전용, 학습 금지).
# 별도 GPU 인스턴스(A100 80GB)에서 부트 루프가 이 파일을 깃허브에서 받아 실행한다. 멱등: 끝난 샷은 건너뛴다.
# 비밀 값은 출력하지 않는다(HF_TOKEN은 있음/없음만).
set -u
W=/workspace/comp; mkdir -p "$W/out" "$W/models" "$W/logs"; cd "$W" || exit 1
REPO_ID="sdsdsdsdsdsdsd123/buc-artifacts"
echo "C_START $(date -u +%FT%TZ) hf_token=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1) 도구
command -v git >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq git >/dev/null; }
python -m pip install -q --no-cache-dir uv huggingface_hub hf_transfer vastai >/dev/null 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1 PATH="$HOME/.local/bin:$PATH"

# 2) LTX-2 코드 + 환경 (한 번만)
if [ ! -f "$W/.env_ok" ]; then
  [ -d LTX-2 ] || git clone -q --depth 1 https://github.com/Lightricks/LTX-2.git
  (cd LTX-2 && (uv sync --extra natten >"$W/logs/uv.log" 2>&1 || uv sync >>"$W/logs/uv.log" 2>&1)) \
    && touch "$W/.env_ok" && echo "C_ENV ok" || { echo "C_ENV fail"; tail -5 "$W/logs/uv.log" | cut -c1-300; exit 1; }
fi

# 3) 가중치 (이어받기 가능)
python - <<'PY' || { echo "C_DL fail"; exit 1; }
import os
from huggingface_hub import hf_hub_download
files = ["diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
         "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
         "vae/ltx-2.5-video-vae-bf16.safetensors", "vae/ltx-2.5-audio-vae-bf16.safetensors",
         "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"]
for f in files:
    hf_hub_download("Lightricks/LTX-2.5", f, local_dir="/workspace/comp/models", token=os.environ.get("HF_TOKEN"))
    print("C_DL ok", f.split("/")[-1], flush=True)
hf_hub_download("sdsdsdsdsdsdsd123/buc-artifacts", "demo/d15_hero_yak.webp", repo_type="model",
                local_dir="/workspace/comp", token=os.environ.get("HF_TOKEN"))
from PIL import Image
Image.open("/workspace/comp/demo/d15_hero_yak.webp").convert("RGB").save("/workspace/comp/yak.png")
print("C_DL ok yak.png", flush=True)
PY

M="$W/models"
FROZEN="Everything in the scene is perfectly still, frozen in time: no wind, the giant yak, the two people, the long hair and the grass do not move at all. Only the camera moves. Photorealistic, cinematic, sharp and highly detailed, natural light."
declare -A P N
P[S1]="The camera slowly dollies forward, pushing in toward the two small people standing beneath the face of the giant white yak, revealing fine detail in the long white hair. $FROZEN"; N[S1]=193
P[S2]="The camera trucks slowly from left to right at a low height just above the grass, creating strong parallax between the blue poppies in the foreground, the curtain of long yak hair, the yak's face, the black yaks and the distant mountains. $FROZEN"; N[S2]=193
P[S3]="The camera arcs slowly around the two people, orbiting about 25 degrees from left to right while keeping them centered, revealing the side of the giant white yak's face and its long curved horns. $FROZEN"; N[S3]=241
P[S4]="The camera cranes up smoothly from just above the grass to the height of the yak's horns, tilting down slightly to keep the two people in frame, revealing the herd of black yaks and the green hills beyond. $FROZEN"; N[S4]=193
P[S5]="Static camera, no camera movement. A slow rack focus: the focus starts on the blurry blue poppies in the foreground, then shifts to the two people, then to the eye of the giant white yak. $FROZEN"; N[S5]=145
P[S6]="Dolly zoom vertigo effect: the camera moves backward while zooming in, so the two people stay the same size while the giant white yak and the hills behind them appear to grow and stretch. $FROZEN"; N[S6]=145

run_shot() {  # run_shot S extra_flags...
  local s=$1; shift
  (cd "$W/LTX-2" && timeout 3600 uv run python -m ltx_pipelines.distilled \
    --transformer-path "$M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$M/vae/ltx-2.5-video-vae-bf16.safetensors" \
    --audio-vae-path "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --spatial-upsampler-path "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --image "$W/yak.png" 0 1.0 --width 1920 --height 832 --num-frames "${N[$s]}" --frame-rate 24 --seed 42 \
    --prompt "${P[$s]}" --output-path "$W/out/ltx25_yak_$s.part.mp4" "$@") >"$W/logs/$s.log" 2>&1
}
upload() {
  python - "$1" <<'PY' || echo "C_UP fail $(basename "$1")"
import os, sys
from huggingface_hub import HfApi
p = sys.argv[1]
HfApi(token=os.environ.get("HF_TOKEN")).upload_file(path_or_fileobj=p, path_in_repo="comp/ltx25/" + os.path.basename(p),
    repo_id="sdsdsdsdsdsdsd123/buc-artifacts", repo_type="model", commit_message="COMP eval-only: LTX-2.5 yak")
print("C_UP ok", os.path.basename(p), flush=True)
PY
}

for s in S1 S2 S3 S4 S5 S6; do
  out="$W/out/ltx25_yak_$s.mp4"
  [ -s "$out" ] && { echo "C_SHOT skip $s"; continue; }
  t0=$(date +%s)
  if run_shot "$s"; then :; else
    echo "C_SHOT retry_offload $s"; tail -3 "$W/logs/$s.log" | cut -c1-300
    run_shot "$s" --offload cpu || { echo "C_SHOT fail $s"; tail -6 "$W/logs/$s.log" | cut -c1-300; continue; }
  fi
  mv "$W/out/ltx25_yak_$s.part.mp4" "$out" && echo "C_SHOT ok $s s=$(( $(date +%s) - t0 ))" && upload "$out"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/C_VRAM /'
done
n=$(ls "$W"/out/ltx25_yak_S?.mp4 2>/dev/null | wc -l)
for f in "$W"/logs/S?.log; do upload "$f"; done
echo "C_DONE shots=$n $(date -u +%FT%TZ)"
[ "$n" -ge 6 ] && touch "$W/.all_done"
