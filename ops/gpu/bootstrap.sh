#!/usr/bin/env bash
# RUN-05 GPU 서버 부트스트랩: 진단 -> 의존성 -> HF 가중치 prefetch -> gsplat 설치/JIT 캐시 빌드.
# 계약서: ops/harness/contracts/RUN-05.md. 모든 설정은 환경 변수. 멱등(스탬프 캐시).
# 주의: set -x 금지(HF_TOKEN 유출). GPU/네트워크 작업은 전부 $BUC_PYTHON 호출로만 한다.
set -u

BUC_ROOT="${BUC_ROOT:-/workspace/buc}"
BUC_PYTHON="${BUC_PYTHON:-python}"
BUC_MODELS="${BUC_MODELS:-Tongyi-MAI/Z-Image-Turbo}"
BUC_PIP_PKGS="${BUC_PIP_PKGS:-diffusers transformers accelerate safetensors huggingface_hub hf_transfer sentencepiece ninja jaxtyping rich}"
BUC_GSPLAT_VERSION="${BUC_GSPLAT_VERSION:-1.5.3}"
BUC_RETRIES="${BUC_RETRIES:-3}"
BUC_RETRY_SLEEP="${BUC_RETRY_SLEEP:-15}"
BUC_GSPLAT_TIMEOUT="${BUC_GSPLAT_TIMEOUT:-2400}"
BUC_FORCE="${BUC_FORCE:-0}"
case "$BUC_RETRIES" in ''|*[!0-9]*) BUC_RETRIES=3 ;; esac
[ "$BUC_RETRIES" -lt 1 ] && BUC_RETRIES=1
case "$BUC_RETRY_SLEEP" in ''|*[!0-9]*) BUC_RETRY_SLEEP=15 ;; esac
case "$BUC_GSPLAT_TIMEOUT" in ''|*[!0-9]*) BUC_GSPLAT_TIMEOUT=2400 ;; esac

mkdir -p "$BUC_ROOT" "$BUC_ROOT/logs" "$BUC_ROOT/stamps" 2>/dev/null
STAMPS="$BUC_ROOT/stamps"
LOG="$BUC_ROOT/logs/bootstrap.log"

# ---- 잠금 (다른 프로세스가 쥐고 있으면 75) ----
if command -v flock >/dev/null 2>&1; then
  exec 9>"$BUC_ROOT/bootstrap.lock"
  if ! flock -n 9; then
    echo "BOOT_LOCKED another bootstrap holds $BUC_ROOT/bootstrap.lock"
    exit 75
  fi
fi

# ---- 모든 출력을 비밀 제거 필터 -> 화면 + 로그 ----
_scrub() {
  python3 -u -c '
import os, sys
t = os.environ.get("HF_TOKEN") or ""
for line in sys.stdin:
    if len(t) >= 4:
        line = line.replace(t, "***")
    sys.stdout.write(line)
    sys.stdout.flush()
' 2>/dev/null || cat
}
exec > >(_scrub | tee -a "$LOG") 2>&1

bootline() { local m="$*"; echo "${m:0:400}"; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
hkey() { python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:20])' "$1"; }

export HF_HOME="${HF_HOME:-$BUC_ROOT/hf}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BUC_ROOT/torch_extensions}"
if [ -z "${MAX_JOBS:-}" ]; then
  _n="$(nproc 2>/dev/null || echo 4)"
  [ "$_n" -gt 16 ] && _n=16
  export MAX_JOBS="$_n"
else
  export MAX_JOBS
fi
mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR" 2>/dev/null

T_START="$(now)"
ENV_STATUS=fail; ENV_REASON=""; P_TORCH=""; P_CUDA=""; P_GPU=""; P_CAP0=""; P_CAP1=""; ARCH=""
DEPS_STATUS=skip; PF_STATUS=skip; PF_MODELS=""
GS_STATUS=skip; GS_SOURCE=""; GS_ATTEMPTS=""
CODE=0
HF_TOKEN_STATE=no
[ -n "${HF_TOKEN:-}" ] && HF_TOKEN_STATE=yes

bootline "BOOT_START root=$BUC_ROOT python=$BUC_PYTHON models=\"$BUC_MODELS\" gsplat=$BUC_GSPLAT_VERSION force=$BUC_FORCE hf_token=$HF_TOKEN_STATE"

write_json() {
  export T_START T_END="$(now)" ENV_STATUS ENV_REASON P_TORCH P_CUDA P_GPU P_CAP0 P_CAP1 ARCH \
         DEPS_STATUS PF_STATUS PF_MODELS GS_STATUS GS_SOURCE GS_ATTEMPTS CODE HF_TOKEN_STATE \
         BUC_ROOT BUC_GSPLAT_VERSION
  python3 - <<'PY'
import json, os
E = os.environ
def nz(k):
    v = E.get(k, "")
    return v if v != "" else None
models = []
for line in E.get("PF_MODELS", "").splitlines():
    p = line.split("\t")
    if len(p) == 3:
        models.append({"repo": p[0], "status": p[1], "attempts": int(p[2])})
cap = None
if E.get("P_CAP0", "") != "" and E.get("P_CAP1", "") != "":
    cap = [int(E["P_CAP0"]), int(E["P_CAP1"])]
code = int(E["CODE"])
doc = {
    "v": 1, "ok": code == 0, "code": code,
    "ts_start": E["T_START"], "ts_end": E["T_END"],
    "hf_token": E["HF_TOKEN_STATE"],
    "paths": {"root": E["BUC_ROOT"], "hf_home": E["HF_HOME"],
              "torch_extensions_dir": E["TORCH_EXTENSIONS_DIR"]},
    "env": {"status": E["ENV_STATUS"], "reason": nz("ENV_REASON"), "torch": nz("P_TORCH"),
            "cuda": nz("P_CUDA"), "gpu": nz("P_GPU"), "cap": cap, "arch": nz("ARCH")},
    "deps": {"status": E["DEPS_STATUS"]},
    "prefetch": {"status": E["PF_STATUS"], "models": models},
    "gsplat": {"status": E["GS_STATUS"], "version": E["BUC_GSPLAT_VERSION"],
               "source": nz("GS_SOURCE"), "attempts": E.get("GS_ATTEMPTS", "").split()},
}
root = E["BUC_ROOT"]
tmp = os.path.join(root, "bootstrap.json.tmp.%d" % os.getpid())
with open(tmp, "w") as f:
    json.dump(doc, f, indent=1)
    f.write("\n")
os.replace(tmp, os.path.join(root, "bootstrap.json"))
PY
}

finish() {
  write_json
  local ok=0
  [ "$CODE" -eq 0 ] && ok=1
  bootline "BOOT_RESULT ok=$ok code=$CODE env=$ENV_STATUS deps=$DEPS_STATUS prefetch=$PF_STATUS gsplat=$GS_STATUS json=$BUC_ROOT/bootstrap.json"
  exit "$CODE"
}

# ---- 파이썬 이음매 ----
py_probe() {
  ( export BOOT_STEP=probe
    "$BUC_PYTHON" - <<'PY'
import json, sys
try:
    import torch
except Exception as e:
    print("import torch failed: %s: %s" % (type(e).__name__, e), file=sys.stderr)
    sys.exit(1)
ok = bool(torch.cuda.is_available())
d = {"cuda_available": ok, "torch": str(torch.__version__), "cuda": torch.version.cuda,
     "gpu": "", "cap": [0, 0], "arch_list": list(torch.cuda.get_arch_list()), "bf16": False}
if ok:
    d["gpu"] = torch.cuda.get_device_name(0)
    d["cap"] = list(torch.cuda.get_device_capability(0))
    d["bf16"] = bool(torch.cuda.is_bf16_supported())
print("PROBE " + json.dumps(d))
PY
  )
}

parse_probe() {  # $1=rc $2=출력 파일 -> TSV(key\tvalue)
  python3 - "$1" "$2" <<'PY'
import json, sys
rc = int(sys.argv[1])
reason = ""
d = {}
try:
    lines = [l for l in open(sys.argv[2], errors="replace").read().splitlines() if l.startswith("PROBE ")]
    if rc != 0 or not lines:
        raise ValueError("no probe")
    d = json.loads(lines[-1][len("PROBE "):])
    cap = d["cap"]
    M, m = int(cap[0]), int(cap[1])
    torch_v = str(d["torch"])
    cuda = d.get("cuda")
    arch_list = [str(x) for x in d["arch_list"]]
    avail = bool(d["cuda_available"])
    gpu = str(d.get("gpu", ""))
except Exception:
    print("reason\tprobe_failed")
    sys.exit(0)

def ver(s):
    try:
        p = str(s).split(".")
        return (int(p[0]), int(p[1]) if len(p) > 1 else 0)
    except Exception:
        return (0, 0)

if not avail:
    reason = "no_cuda"
elif M >= 12 and (cuda is None or ver(cuda) < (12, 8)):
    reason = "cuda_too_old_for_sm120"
elif "sm_%d%d" % (M, m) not in arch_list:
    reason = "torch_lacks_sm_%d%d" % (M, m)
print("reason\t" + reason)
print("torch\t" + torch_v)
print("cuda\t" + ("" if cuda is None else str(cuda)))
print("gpu\t" + gpu.replace("\t", " ").replace("\n", " "))
print("cap0\t%d" % M)
print("cap1\t%d" % m)
PY
}

py_prefetch() {  # $1=repo
  ( export BOOT_STEP=prefetch BOOT_MODEL="$1" HF_HOME HF_HUB_ENABLE_HF_TRANSFER=1
    "$BUC_PYTHON" - </dev/null <<'PY'
import os, sys
repo = os.environ["BOOT_MODEL"]
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=repo)  # 토큰은 HF_TOKEN 환경에서 라이브러리가 직접 읽는다
    print("prefetched %s -> %s" % (repo, path))
except Exception as e:
    print("prefetch failed: %s: %s" % (type(e).__name__, str(e)[:300]), file=sys.stderr)
    sys.exit(1)
PY
  )
}

py_gsplat_jit() {
  ( export BOOT_STEP=gsplat_jit TORCH_CUDA_ARCH_LIST TORCH_EXTENSIONS_DIR MAX_JOBS
    timeout -k 10 "$BUC_GSPLAT_TIMEOUT" "$BUC_PYTHON" - <<'PY'
import sys
import torch
try:
    import gsplat
    from gsplat import rasterization
    dev = "cuda"
    means = torch.tensor([[0.0, 0.0, 2.0]], device=dev, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    scales = torch.full((1, 3), 0.3, device=dev)
    opac = torch.tensor([1.0], device=dev)
    colors = torch.tensor([[1.0, 0.5, 0.25]], device=dev)
    viewmats = torch.eye(4, device=dev)[None]
    K = torch.tensor([[[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]], device=dev)
    img, alpha, _ = rasterization(means, quats, scales, opac, colors, viewmats, K, 64, 64)
    img.sum().backward()
    torch.cuda.synchronize()
    assert tuple(img.shape) == (1, 64, 64, 3), img.shape
    assert bool(torch.isfinite(img).all()), "non-finite render"
    assert float(alpha.max()) > 0.5, "empty render"
    assert bool(torch.isfinite(means.grad).all()), "non-finite grad"
    print("gsplat %s jit ok" % getattr(gsplat, "__version__", "?"))
except Exception as e:
    print("gsplat jit failed: %s: %s" % (type(e).__name__, str(e)[:400]), file=sys.stderr)
    sys.exit(1)
PY
  )
}

pip_install() {  # $1=step, 나머지=pip 인자 (timeout 은 gsplat 만)
  local step="$1"; shift
  ( export BOOT_STEP="$step" TORCH_CUDA_ARCH_LIST TORCH_EXTENSIONS_DIR MAX_JOBS
    if [ "$step" = deps ]; then
      "$BUC_PYTHON" -m pip install --disable-pip-version-check -c "$CONSTRAINTS" "$@" </dev/null
    else
      timeout -k 10 "$BUC_GSPLAT_TIMEOUT" "$BUC_PYTHON" -m pip install --disable-pip-version-check "$@" </dev/null
    fi
  )
}

# ---- 1. probe (매번) ----
if command -v nvidia-smi >/dev/null 2>&1; then
  timeout 20 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n 4 || true
fi
PROBE_OUT="$BUC_ROOT/logs/probe.out"
py_probe >"$PROBE_OUT"
PROBE_RC=$?
cat "$PROBE_OUT"
ENV_REASON="probe_failed"
while IFS=$'\t' read -r k v; do
  case "$k" in
    reason) ENV_REASON="$v" ;;
    torch) P_TORCH="$v" ;;
    cuda) P_CUDA="$v" ;;
    gpu) P_GPU="$v" ;;
    cap0) P_CAP0="$v" ;;
    cap1) P_CAP1="$v" ;;
  esac
done < <(parse_probe "$PROBE_RC" "$PROBE_OUT")

if [ -n "$P_CAP0" ]; then
  ARCH="${P_CAP0}.${P_CAP1}"
fi
if [ -n "$ENV_REASON" ]; then
  ENV_STATUS=fail
  CODE=10
  rm -f "$BUC_ROOT/env.sh"
  bootline "BOOT_ENV status=fail reason=$ENV_REASON torch=${P_TORCH:-?} cuda=${P_CUDA:-?} gpu=\"${P_GPU:-?}\" arch=${ARCH:-?}"
  finish
fi
ENV_STATUS=ok
ENV_REASON=""
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH}"
ARCH="$TORCH_CUDA_ARCH_LIST"
bootline "BOOT_ENV status=ok torch=$P_TORCH cuda=$P_CUDA gpu=\"$P_GPU\" cap=${P_CAP0}.${P_CAP1} arch=$ARCH max_jobs=$MAX_JOBS"

# torch 보호: pip 가 cu128 torch 를 갈아엎지 못하게 constraints 로 고정
CONSTRAINTS="$BUC_ROOT/constraints.txt"
printf 'torch==%s\n' "$P_TORCH" >"$CONSTRAINTS.tmp.$$" && mv -f "$CONSTRAINTS.tmp.$$" "$CONSTRAINTS"

# env.sh (비밀 없음)
{
  printf 'export HF_HOME=%q TORCH_EXTENSIONS_DIR=%q TORCH_CUDA_ARCH_LIST=%q HF_HUB_ENABLE_HF_TRANSFER=1\n' \
    "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCH_CUDA_ARCH_LIST"
} >"$BUC_ROOT/env.sh.tmp.$$" && mv -f "$BUC_ROOT/env.sh.tmp.$$" "$BUC_ROOT/env.sh"

stamp_hit() { [ "$BUC_FORCE" != 1 ] && [ -f "$1" ]; }

# ---- 2. deps ----
PKGS=""
for p in $BUC_PIP_PKGS; do
  if [[ "$p" =~ ^(torch|torchvision|torchaudio)($|[][=\<\>!~]) ]]; then
    bootline "BOOT_DEPS dropped torch-family requirement: $p"
    continue
  fi
  PKGS="$PKGS $p"
done
if [ "${BUC_SKIP_DEPS:-0}" = 1 ]; then
  DEPS_STATUS=skip
else
  DEPS_STAMP="$STAMPS/deps-$(hkey "$PKGS|$P_TORCH")"
  if stamp_hit "$DEPS_STAMP"; then
    DEPS_STATUS=cached
  else
    # shellcheck disable=SC2086
    pip_install deps $PKGS
    if [ $? -eq 0 ]; then
      DEPS_STATUS=ok
      : >"$DEPS_STAMP"
    else
      DEPS_STATUS=fail
      CODE=11
    fi
  fi
  bootline "BOOT_DEPS status=$DEPS_STATUS"
fi
if [ "$DEPS_STATUS" = fail ]; then
  finish
fi

# ---- 3. prefetch ----
if [ "${BUC_SKIP_PREFETCH:-0}" = 1 ]; then
  PF_STATUS=skip
else
  n_fail=0; n_ok=0; n_cached=0; n_total=0
  for repo in $BUC_MODELS; do
    n_total=$((n_total + 1))
    mstamp="$STAMPS/prefetch-$(hkey "$repo")"
    if stamp_hit "$mstamp"; then
      PF_MODELS="${PF_MODELS}${repo}"$'\t'"cached"$'\t'"0"$'\n'
      n_cached=$((n_cached + 1))
      bootline "BOOT_PREFETCH repo=$repo status=cached"
      continue
    fi
    attempts=0; done_ok=0
    while [ "$attempts" -lt "$BUC_RETRIES" ]; do
      attempts=$((attempts + 1))
      py_prefetch "$repo" </dev/null
      if [ $? -eq 0 ]; then done_ok=1; break; fi
      if [ "$attempts" -lt "$BUC_RETRIES" ]; then sleep "$BUC_RETRY_SLEEP"; fi
    done
    if [ "$done_ok" -eq 1 ]; then
      : >"$mstamp"
      PF_MODELS="${PF_MODELS}${repo}"$'\t'"ok"$'\t'"${attempts}"$'\n'
      n_ok=$((n_ok + 1))
      bootline "BOOT_PREFETCH repo=$repo status=ok attempts=$attempts"
    else
      PF_MODELS="${PF_MODELS}${repo}"$'\t'"fail"$'\t'"${attempts}"$'\n'
      n_fail=$((n_fail + 1))
      bootline "BOOT_PREFETCH repo=$repo status=fail attempts=$attempts"
    fi
  done
  if [ "$n_total" -eq 0 ]; then
    PF_STATUS=skip
  elif [ "$n_fail" -gt 0 ]; then
    PF_STATUS=fail
    [ "$CODE" -eq 0 ] && CODE=20
  elif [ "$n_ok" -eq 0 ]; then
    PF_STATUS=cached
  else
    PF_STATUS=ok
  fi
fi

# ---- 4. gsplat ----
gs_ext_nonempty() { [ -d "$TORCH_EXTENSIONS_DIR" ] && [ -n "$(ls -A "$TORCH_EXTENSIONS_DIR" 2>/dev/null)" ]; }

if [ "${BUC_SKIP_GSPLAT:-0}" = 1 ]; then
  GS_STATUS=skip
else
  GS_STAMP="$STAMPS/gsplat-$(hkey "$BUC_GSPLAT_VERSION|${P_CAP0}.${P_CAP1}|$P_TORCH|$P_CUDA")"
  if stamp_hit "$GS_STAMP" && gs_ext_nonempty; then
    GS_STATUS=cached
    GS_SOURCE="$(head -c 16 "$GS_STAMP" 2>/dev/null | tr -d '\n')"
    case "$GS_SOURCE" in pypi|git) ;; *) GS_SOURCE="" ;; esac
  else
    GS_STATUS=fail
    GS_ATTEMPTS="pypi"
    pip_install gsplat_install -c "$CONSTRAINTS" "gsplat==${BUC_GSPLAT_VERSION}"
    if [ $? -eq 0 ]; then
      py_gsplat_jit </dev/null
      [ $? -eq 0 ] && { GS_STATUS=ok; GS_SOURCE=pypi; }
    fi
    if [ "$GS_STATUS" != ok ]; then
      GS_ATTEMPTS="pypi git"
      pip_install gsplat_install --no-build-isolation -c "$CONSTRAINTS" \
        "git+https://github.com/nerfstudio-project/gsplat@v${BUC_GSPLAT_VERSION}"
      if [ $? -eq 0 ]; then
        py_gsplat_jit </dev/null
        [ $? -eq 0 ] && { GS_STATUS=ok; GS_SOURCE=git; }
      fi
    fi
    if [ "$GS_STATUS" = ok ]; then
      printf '%s\n' "$GS_SOURCE" >"$GS_STAMP"
    else
      GS_SOURCE=""
      [ "$CODE" -eq 0 ] && CODE=30
    fi
  fi
  bootline "BOOT_GSPLAT status=$GS_STATUS version=$BUC_GSPLAT_VERSION source=${GS_SOURCE:-none} attempts=${GS_ATTEMPTS:-none}"
fi

finish
