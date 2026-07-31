# flashinfer_rocm vs aiter on MI300X

Baseline numbers for the `flashinfer_rocm` attention backend against `aiter`,
the established ROCm backend, so later changes have something to move.

## Method

```bash
docker build -f docker/rocm-flashinfer.Dockerfile -t sglang:rocm-flashinfer .

# Per backend, per batch size:
python3 -m sglang.benchmark.one_batch \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --attention-backend {flashinfer_rocm|aiter} \
  --batch-size {1|8|32} --input-len 512 --output-len 32 \
  --cuda-graph-max-bs-decode 32
```

`one_batch` prints a warmup pass and then a benchmark pass; the tables below use
the **benchmark** pass. CUDA graphs are on for both backends (decode only for
`flashinfer_rocm` — it does not implement prefill graphs).

Select the kernel family behind `flashinfer_rocm` with
`SGLANG_FLASHINFER_ROCM_BACKEND` (`fa2` default / `aiter` / `auto`).

## Environment

| | |
| --- | --- |
| GPU | 1× MI300X (gfx942) |
| Image | `sglang:rocm-flashinfer` (ROCm 7.2, Ubuntu 24.04, Python 3.12) |
| torch | 2.9.1+rocm7.2.0 |
| FlashInfer | 0.5.3+amd.2.dev52 (AMD-Ecosystem/flashinfer @ `cd6b40d2`) |
| AITER | ROCm/aiter @ `d9e5ef7c` (SGLang's `AITER_COMMIT_DEFAULT`) |
| Model | TinyLlama-1.1B-Chat-v1.0 (32 Q heads / 4 KV heads → GQA group 8) |
| Date | 2026-07-31 |

## Results

Throughput, tokens/s (higher is better):

| Batch | Phase | `flashinfer_rocm` (fa2) | `aiter` | ratio |
| ---: | --- | ---: | ---: | ---: |
| 1 | prefill | 36,061 | 38,224 | 0.94 |
| 1 | decode | 427 | 542 | 0.79 |
| 8 | prefill | 157,160 | 184,783 | 0.85 |
| 8 | decode | 3,258 | 4,198 | 0.78 |
| 32 | prefill | 176,415 | 221,499 | 0.80 |
| 32 | decode | 11,923 | 15,199 | 0.78 |

Median decode latency, seconds (lower is better):

| Batch | `flashinfer_rocm` (fa2) | `aiter` |
| ---: | ---: | ---: |
| 1 | 0.00234 | 0.00185 |
| 8 | 0.00246 | 0.00191 |
| 32 | 0.00268 | 0.00211 |

### Where the decode gap actually is

AITER leads on every graph-enabled row, but the shape of the decode gap says it
is **not** the kernel:

| Batch | `flashinfer_rocm` (ms) | `aiter` (ms) | absolute gap (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 2.34 | 1.85 | 0.49 |
| 8 | 2.46 | 1.91 | 0.55 |
| 32 | 2.68 | 2.11 | 0.57 |

The gap is ~0.5 ms and essentially flat from batch 1 to batch 32 — 32× the work
for the same extra cost. That is fixed per-step overhead, not per-token kernel
cost, which would scale with the batch.

Eager mode (graphs off, batch 8, same shapes) confirms it — with the graph path
removed, the in-tree HIP kernel is *ahead* of AITER on decode:

| Config | prefill tok/s | decode tok/s |
| --- | ---: | ---: |
| `flashinfer_rocm`, `fa2` kernels | 154,139 | **710** |
| `flashinfer_rocm`, `aiter` kernels | 161,124 | 564 |
| `aiter` backend | 186,336 | 671 |

So `fa2` decode is competitive; CUDA graphs just help AITER more (6.3× vs 4.6×).
The prime suspect is the plan path: this backend calls the public ROCm `plan()`
on every decode step, while SGLang's NVIDIA FlashInfer backend uses
`fast_decode_plan` specifically to skip the device-to-host copies inside
`plan()`. The ROCm port exports no equivalent. Making the plan path sync-free is
therefore the highest-value optimization, ahead of anything kernel-side.

Prefill is a separate story: `aiter` leads there in every mode, and part of that
is `page_size=1` (see caveats).

## Caveats

- **1.1B model.** Attention is a smaller share of total time than it would be at
  8B, and prefill numbers at this size are dominated by launch overhead. Re-run
  on Llama-3.1-8B (32/8 → GQA group 4) before drawing conclusions about
  production shapes.
- **`page_size=1`.** `flashinfer_rocm` supports nothing else today, and AITER's
  CK FMHA prefers `{128, 256, 1024}`. Part of the prefill gap is likely the extra
  gather this forces; sweeping page size is the first optimization to try.
- **Prefill CUDA graphs are off for `flashinfer_rocm`** and available to `aiter`,
  which flatters `aiter` on the prefill rows.
- Single run per cell, no clock locking, on a shared machine. Treat differences
  under ~5% as noise.
- **`SGLANG_FLASHINFER_ROCM_BACKEND=aiter` is impractical under CUDA graphs
  today.** AITER JIT-builds a kernel variant per captured graph shape; a
  capture ladder up to batch 32 was still compiling after 30 minutes of solid
  CPU on batch size 1 alone. The `aiter`-kernel row above is therefore eager
  only. Prebuilding the AITER kernel cache into the image would be needed to
  measure that combination under graphs.
- FlashInfer's AITER prefill path dlopen's an AITER `.so` it cannot build
  itself; `warm_aiter_mha.py`-style prewarming (calling `aiter.ops.mha.
  mha_varlen_fwd` with matching dtype/causal/lse) is required first, otherwise
  it fails with "AITER .so not found".

## MLA (DeepSeek-V2-Lite)

MLA is **decode-only** on this backend, and not production-viable today.

`flashinfer_rocm` cannot serve MLA *prefill* at all. DeepSeek prefill is ragged
attention at `head_dim_qk=192` / `head_dim_vo=128`, and neither ROCm path
supports it: the in-tree `fa2` kernel fails to **compile** for that shape (its
`k128B_16Row` swizzle `static_assert`s on a power-of-2 stride, and 192 is not
one), while AITER rejects `head_dim_qk != head_dim_vo` outright. SGLang now
rejects the configuration at startup instead of failing several minutes into a
JIT build.

MLA decode does work, in a hybrid split:

```bash
--prefill-attention-backend aiter --decode-attention-backend flashinfer_rocm
```

DeepSeek-V2-Lite, batch 2, input 128, output 8:

| Config | Decode tok/s |
| --- | ---: |
| `aiter`, CUDA graphs | **351.4** |
| `aiter`, eager | 39.9 |
| `flashinfer_rocm` MLA decode (hybrid), eager | 28.3 |

Two separate effects, worth keeping apart:

- **Eager vs eager, the gap is 0.71×** (28.3 / 39.9) — the same order as the MHA
  plan-overhead gap, and consistent with `mla_rocm`'s `plan()` calling
  `qo_lens.max().item()`, a host sync, on every step of every layer.
- **In practice the gap is 12×**, because `flashinfer_rocm` MLA cannot use CUDA
  graphs at all (the ROCm `BatchMLAPagedAttentionWrapper` takes no
  `use_cuda_graph` argument) while `aiter` can. Graphs are worth ~8.8× here.

So MLA on this backend is a plumbing-correctness result, not a usable
configuration. Closing the gap needs a CUDA-graph-capable MLA wrapper in the
ROCm port, which is an upstream change.

## Correctness

Greedy decoding (`temperature=0`, 48 new tokens) over 4 prompts, CUDA graphs on
for both: **4/4 outputs byte-identical between `flashinfer_rocm` and `aiter`.**

Kernel-level checks live in
`test/registered/kernels/test_flashinfer_rocm_attention_kernels.py`; end-to-end
accuracy and the aiter equivalence check live in
`test/registered/attention/test_flashinfer_rocm_attention_backend.py`.
