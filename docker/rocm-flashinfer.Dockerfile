# =============================================================================
# SGLang + FlashInfer-ROCm development image
#
# Builds an SGLang runtime on top of AMD's *validated* FlashInfer+ROCm image, so
# ROCm / PyTorch / FlashInfer are the exact combination AMD tests against rather
# than a set of pins we maintain ourselves.
#
#   Base: rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
#         -> ROCm 7.2, Ubuntu 24.04, Python 3.12, torch 2.9.1, amd-flashinfer 0.5.3+amd.1
#
# AITER is installed too: it is both the FlashInfer AITER dispatch target and the
# baseline that `--attention-backend flashinfer_rocm` is benchmarked against, so
# both must live in the same image for the comparison to be meaningful.
#
# Build (from repo root, so the SGLang source is in the build context):
#   docker build -f docker/rocm-flashinfer.Dockerfile -t sglang:rocm-flashinfer .
#
# Build against a local FlashInfer checkout instead of the base image's wheel
# (for kernel-side debugging):
#   docker build -f docker/rocm-flashinfer.Dockerfile \
#     --build-arg FLASHINFER_SRC=1 -t sglang:rocm-flashinfer . \
#     && docker run ... -v ~/devel/flashinfer:/sgl-workspace/flashinfer ...
#
# Run: see docker/rocm-flashinfer-run.sh
# =============================================================================

ARG BASE_IMAGE=rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
FROM ${BASE_IMAGE}

# gfx942 = CDNA3 (MI300X / MI325X). Add gfx950 for CDNA4 (MI355X).
ARG GPU_ARCH=gfx942
# FlashInfer source. The base image ships the only *published* wheel,
# 0.5.3+amd.1, which does NOT export BatchMLAPagedAttentionWrapper and predates
# AITER decode CUDA-graph capture — both landed in 0.5.3+amd.2 and are needed
# here. So the default is to build the pinned amd-integration ref over the top.
# Set FLASHINFER_MODE=preinstalled to keep the base image's wheel (MHA only).
ARG FLASHINFER_MODE=source
ARG FLASHINFER_REPO=https://github.com/AMD-Ecosystem/flashinfer.git
ARG FLASHINFER_REF=cd6b40d2a857934f0cbc658b8d59aa905093e81d
# AITER. Built from source at SGLang's pinned commit -- keep this in sync with
# AITER_COMMIT_DEFAULT in docker/rocm.Dockerfile. The base image bundles an older
# AITER wheel that SGLang main cannot import (sglang.srt.layers.quantization.fp8
# needs aiter.ops.shuffle.shuffle_scale, absent from amd_aiter 0.1.x), so the
# wheel is not a usable fallback for this image.
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_COMMIT=d9e5ef7ce08ee7045d583aed768cff41aa9210fe

ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH=${GPU_ARCH}
ENV GPU_ARCH_LIST=${GPU_ARCH}
ENV PATH="${ROCM_PATH}/bin:${PATH}"
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /sgl-workspace

# ---- FlashInfer (ROCm) -------------------------------------------------------
# Installed editable and JIT-first (no FLASHINFER_ROCM_ARCH_LIST AOT pass), so
# the build stays short and kernels compile on first use. Bind-mounting a host
# checkout over /sgl-workspace/flashinfer at runtime keeps the install valid.
RUN if [ "${FLASHINFER_MODE}" = "source" ]; then \
        git clone "${FLASHINFER_REPO}" /sgl-workspace/flashinfer \
        && cd /sgl-workspace/flashinfer \
        && git checkout -f "${FLASHINFER_REF}" \
        && pip install --no-build-isolation --no-deps -ve . ; \
    else \
        echo "Keeping the base image's amd-flashinfer wheel (MHA only; no MLA)." ; \
    fi
# NOTE: no `import flashinfer` smoke test here. `docker build` has no GPU, and
# flashinfer resolves its JIT workspace via torch.cuda at import time, so any
# import during build dies on "No HIP GPUs are available". Verification lives in
# docker/rocm-flashinfer-smoke.py, run inside the container.

# ---- AITER -------------------------------------------------------------------
# Installed before SGLang so the srt_hip extra resolves against it.
# AITER_USE_SYSTEM_TRITON=1 keeps the ROCm triton from the base image (aiter
# would otherwise pull its own, breaking the pinned stack). No PREBUILD_KERNELS:
# kernels JIT on first use, which keeps the image build to minutes instead of
# an hour -- same choice scripts/ci/amd/amd_ci_install_dependency.sh makes.
RUN pip uninstall -y aiter amd_aiter || true; \
    git clone "${AITER_REPO}" /sgl-workspace/aiter \
    && cd /sgl-workspace/aiter \
    && git checkout -f "${AITER_COMMIT}" \
    && git submodule update --init --recursive \
    && SETUPTOOLS_SCM_PRETEND_VERSION= AITER_USE_SYSTEM_TRITON=1 \
       GPU_ARCHS=${GPU_ARCH_LIST} \
       pip install --config-settings editable_mode=compat -e .
ENV PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH}

# ---- Rust --------------------------------------------------------------------
# setuptools-rust builds the sglang-mm extension (sglang.srt.multimodal._core)
# during the editable install below, so the toolchain must precede it.
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_BUILD_JOBS=4
RUN command -v cargo >/dev/null 2>&1 \
    || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y) \
    && rustc --version && cargo --version

# ---- Pin the ROCm torch stack ------------------------------------------------
# `srt_hip` leaves torch unpinned and pulls timm, which needs torchvision. The
# base image ships no torchvision, so pip resolves it from PyPI -- and PyPI
# torchvision pins an exact torch, which silently replaced the ROCm build with
# torch 2.10.0+cu128 plus the whole nvidia-*-cu12 stack. Same for triton.
#
# So: install the matching ROCm torchvision first, then freeze the three
# packages to whatever is actually installed and hand that to pip as a
# constraint for every later install.
ARG TORCHVISION_VERSION=0.24.0+rocm7.2.0.gitb919bd0c
ARG ROCM_WHEEL_INDEX=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
RUN pip install "torchvision==${TORCHVISION_VERSION}" -f "${ROCM_WHEEL_INDEX}"

# The base image ships ROCm triton 3.5.1, but SGLang's pinned AITER refuses to
# import below 3.6.0 ("aiter gluon kernels require triton>=3.6.0"). Take the
# ROCm 3.6.0 wheel rather than PyPI's CUDA triton. torch 2.9.1 declares
# triton==3.5.1, so --no-deps keeps pip from dragging torch along with it;
# docker/rocm.Dockerfile solves the same conflict by relaxing that pin.
ARG TRITON_VERSION=3.6.0+rocm7.2.0.gitba5c1517
RUN pip install --no-deps "triton==${TRITON_VERSION}" -f "${ROCM_WHEEL_INDEX}"

# torch 2.9.1 declares an exact `triton==3.5.1`, so with 3.6.0 installed the
# next resolve is ResolutionImpossible. Relax that one pin in torch's installed
# metadata to `triton` (unversioned) -- docker/rocm.Dockerfile does the same
# thing for the same reason, by rewriting the wheel before install.
RUN META="$(python3 -c "import importlib.metadata as md, pathlib; print(pathlib.Path(md.distribution('torch')._path) / 'METADATA')")" \
    && sed -i -E 's/^(Requires-Dist: triton)[=<>!~].*/\1/' "$META" \
    && grep -iE "^Requires-Dist: triton" "$META"

ENV PIP_CONSTRAINT=/sgl-workspace/rocm-constraints.txt
RUN python3 -c "\
import importlib.metadata as md;\
open('/sgl-workspace/rocm-constraints.txt','w').write(\
''.join(f'{n}=={md.version(n)}\n' for n in ('torch','torchvision','triton')))" \
    && cat /sgl-workspace/rocm-constraints.txt

# ---- SGLang ------------------------------------------------------------------
# Any pre-existing sglang in the base image would shadow the editable install.
RUN pip uninstall -y sgl_kernel sglang || true

COPY . /sgl-workspace/sglang

# Two source-tree mutations mirror docker/rocm.Dockerfile: the AOT kernel package
# and the top-level package each ship a ROCm pyproject that must be swapped into
# place before installing. They happen inside the image, never on the host tree.
RUN cd /sgl-workspace/sglang/python/sglang/kernels/aot \
    && rm -f pyproject.toml \
    && mv pyproject_rocm.toml pyproject.toml \
    && AMDGPU_TARGET=${GPU_ARCH_LIST} python3 setup_rocm.py install

RUN cd /sgl-workspace/sglang \
    && rm -f python/pyproject.toml \
    && mv python/pyproject_other.toml python/pyproject.toml \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 \
       pip install -e "python[srt_hip]"

# Guard the two ways this install can silently end up on the wrong stack. Both
# checks read package metadata rather than importing, so they run without a GPU.
#   1. NVIDIA flashinfer_python collides with amd-flashinfer (same import name).
#   2. A CUDA torch/torchvision/triton replacing the ROCm build -- the failure
#      that motivated the constraints file above.
RUN ! pip show flashinfer-python >/dev/null 2>&1 \
    || (echo "ERROR: NVIDIA flashinfer_python installed; collides with amd-flashinfer" && exit 1)
RUN python3 -c "\
import importlib.metadata as md, sys;\
bad=[f'{n}=={md.version(n)}' for n in ('torch','torchvision','triton') if 'rocm' not in md.version(n)];\
sys.exit(f'ERROR: non-ROCm build(s) installed: {bad}') if bad else print('ROCm torch stack intact')"

# ---- Runtime defaults --------------------------------------------------------
# Mirrors docker/rocm.Dockerfile so aiter-baseline runs behave identically.
ENV SGLANG_USE_AITER=1 \
    SGLANG_MOE_PADDING=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    SGLANG_DISABLE_CUDNN_CHECK=1 \
    NCCL_MIN_NCHANNELS=112

WORKDIR /sgl-workspace/sglang
CMD ["/bin/bash"]
