#!/usr/bin/env bash
# Throughput comparison across ROCm attention backends.
#
#   bash benchmark/flashinfer_rocm/bench_backends.sh [model] [gpu]
#
# Defaults to Mistral-7B-Instruct-v0.3 (32/8 heads -> GQA group 4, the
# Llama-3.1-8B attention shape) on GPU 0. Run inside the container built from
# docker/rocm-flashinfer.Dockerfile.
#
# Pin to an IDLE GPU. This box is shared, and the first round of numbers in
# README.md was collected while another process held the same device. Check
# with `rocm-smi --showmeminfo vram` first -- an idle CDNA3 card sits at ~0.3 GB.
set -uo pipefail

MODEL="${1:-mistralai/Mistral-7B-Instruct-v0.3}"
export HIP_VISIBLE_DEVICES="${2:-0}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
BATCH_SIZES="${BATCH_SIZES:-1 8 32}"
BACKENDS="${BACKENDS:-flashinfer_rocm aiter}"

echo "model=$MODEL gpu=$HIP_VISIBLE_DEVICES input=$INPUT_LEN output=$OUTPUT_LEN"
echo

for backend in $BACKENDS; do
  for bs in $BATCH_SIZES; do
    # one_batch prints a warmup pass then a benchmark pass; keep the latter.
    line=$(python3 -m sglang.benchmark.one_batch \
        --model-path "$MODEL" \
        --attention-backend "$backend" \
        --batch-size "$bs" --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
        --cuda-graph-max-bs-decode 32 2>&1 \
      | grep -E "^(Prefill|Decode\.)" | tail -2 | tr '\n' '|')
    echo "RESULT backend=$backend bs=$bs $line"
  done
done
