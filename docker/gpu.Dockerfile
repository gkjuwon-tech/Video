# GPU 서버 이미지 (RTX 5090 = sm_120 → CUDA 12.8 이상 필요. 호스트 드라이버가 12.8까지만 지원하는 경우가 있어 12.8 고정)
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir -U diffusers transformers accelerate safetensors huggingface_hub hf_transfer sentencepiece vastai
ENV HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /workspace
