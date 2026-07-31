#!/usr/bin/env bash
# Launch the SGLang + FlashInfer-ROCm dev container.
#
#   bash docker/rocm-flashinfer-run.sh                 # baked-in SGLang source
#   MOUNT_SRC=1 bash docker/rocm-flashinfer-run.sh     # live-edit host source
#   MOUNT_FLASHINFER=~/devel/flashinfer bash docker/rocm-flashinfer-run.sh
#
# MOUNT_SRC=1 bind-mounts the host repo over the image's copy so Python edits are
# live. The trade-off: it also hides the Rust extension (sglang.srt.multimodal._core)
# that was compiled into the image's tree, so the script reinstalls editable on
# entry when that extension is missing.
set -euo pipefail

IMAGE="${IMAGE:-sglang:rocm-flashinfer}"
NAME="${NAME:-sglang-rocm-flashinfer}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

args=(
  --rm -it
  --name "$NAME"
  --network host --ipc host --shm-size 32g
  --device /dev/kfd --device /dev/dri
  --group-add video --group-add render
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined
  -v "$HF_CACHE:/root/.cache/huggingface"
  -e "HF_TOKEN=${HF_TOKEN:-}"
)

if [[ "${MOUNT_SRC:-0}" == "1" ]]; then
  args+=(-v "$REPO_ROOT:/sgl-workspace/sglang")
fi

if [[ -n "${MOUNT_FLASHINFER:-}" ]]; then
  args+=(-v "$(cd "$MOUNT_FLASHINFER" && pwd):/sgl-workspace/flashinfer")
fi

# `--privileged` is deliberately omitted; the device + group flags above are
# sufficient for GPU access and keep the container unprivileged.
exec docker run "${args[@]}" "$IMAGE" bash -lc '
  if [ -d /sgl-workspace/flashinfer ] && [ -f /sgl-workspace/flashinfer/pyproject.toml ]; then
    echo "[run] installing bind-mounted FlashInfer editable"
    pip install --no-build-isolation --no-deps -ve /sgl-workspace/flashinfer
  fi
  if ! python3 -c "import sglang.srt.multimodal._core" >/dev/null 2>&1; then
    echo "[run] Rust extension missing (bind-mounted source shadowed it); reinstalling editable"
    cd /sgl-workspace/sglang \
      && [ -f python/pyproject_other.toml ] \
      && cp python/pyproject_other.toml /tmp/pyproject_hip.toml \
      && cp /tmp/pyproject_hip.toml python/pyproject.toml
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 pip install -e "/sgl-workspace/sglang/python[srt_hip]"
  fi
  exec bash
'
