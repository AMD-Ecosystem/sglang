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
| Models | Mistral-7B-Instruct-v0.3 (32/8 → GQA group 4, the Llama-3.1-8B shape); TinyLlama-1.1B-Chat-v1.0 (32/4 → group 8) |
| Date | 2026-07-31 |

`meta-llama/*` is gated for this account, so Mistral-7B stands in for
Llama-3.1-8B; its attention shape is identical.

## Results — Mistral-7B-Instruct-v0.3 (production-shaped)

The headline numbers. 32/8 heads → GQA group 4, `head_dim=128`, bf16, no sliding
window — the same attention shape as Llama-3.1-8B, which is gated. Batch 1/8/32,
input 512, output 32, CUDA graphs on for both.

| Batch | Phase | `flashinfer_rocm` (fa2) | `aiter` | ratio |
| ---: | --- | ---: | ---: | ---: |
| 1 | prefill | 23,018 | 24,354 | 0.95 |
| 1 | decode | 151 | 170 | 0.89 |
| 8 | prefill | 34,346 | 36,709 | 0.94 |
| 8 | decode | 1,136 | 1,267 | 0.90 |
| 32 | prefill | 34,841 | 38,687 | 0.90 |
| 32 | decode | 3,934 | 4,228 | 0.93 |

At production scale `flashinfer_rocm` lands within **5–10% on prefill and
7–11% on decode**. The 1.1B numbers further down look far worse (0.78 decode);
that difference is itself the evidence for where the cost is.

### The gap is fixed per-step overhead

Decode gap in absolute terms, both models:

| Batch | TinyLlama-1.1B | Mistral-7B |
| ---: | ---: | ---: |
| 1 | 0.49 ms | 0.74 ms |
| 8 | 0.55 ms | 0.72 ms |
| 32 | 0.57 ms | 0.56 ms |

Roughly **0.6 ms regardless of model size or batch size** — 6× the parameters
and 32× the batch, same absolute cost. The *ratio* improves from 0.78 to ~0.90
purely because each step does more work for the fixed cost to hide behind.

That is what a fixed per-step overhead predicts. A per-token kernel deficit
would scale with batch and with model size, and would leave the ratio flat
instead of the absolute gap flat.

The overhead is the plan path. This backend calls the public ROCm `plan()` on
every decode step; SGLang's NVIDIA FlashInfer backend uses `fast_decode_plan`
specifically to skip the device-to-host copies inside `plan()`, and the ROCm
port exports no equivalent. **Making the plan path sync-free is the whole
optimization**, and it is worth about 0.6 ms/step wherever it lands.

Eager mode corroborates this: with the graph path removed, the in-tree HIP
kernel is *ahead* of AITER on decode (see the TinyLlama eager table below).

## Results — TinyLlama-1.1B (bring-up)

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

Median decode latency here is 2.34 / 2.46 / 2.68 ms against AITER's
1.85 / 1.91 / 2.11 ms — the ~0.5 ms flat gap analysed above.

Eager mode (graphs off, batch 8, same shapes) is the other half of the evidence:
with the graph path removed, the in-tree HIP kernel is *ahead* of AITER on
decode, which rules out the kernel as the cause.

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

Greedy decoding (`temperature=0`, 48 new tokens), CUDA graphs on for both:

| Model | Identical outputs |
| --- | --- |
| TinyLlama-1.1B | 4/4 |
| Mistral-7B-Instruct-v0.3 | 3/4 |

The one Mistral divergence is a **tie-break, not a wrong result**, and it is
worth understanding before treating exact-match as the bar.

Both completions were correct; they just worded the answer differently. Running
the same prompt with per-token logprobs locates the split at output token 7,
where the two candidates are:

| Backend | top-1 | top-2 | margin |
| --- | --- | --- | ---: |
| `aiter` | tok 1070 @ -2.19410 | tok 12928 @ -2.19410 | **0.00000** |
| `flashinfer_rocm` | tok 12928 @ -2.18112 | tok 1070 @ -2.24362 | 0.06250 |

Same two candidate tokens, and under `aiter` they are **exactly tied** to the
last bit. Which one greedy picks is decided by tie-break order, not by attention
math. Logprob agreement over the preceding 7 tokens is within 0.030, and the
top-5 distribution for the first token matches in both membership and order
(largest disagreement 0.063 logprob).

Once greedy takes different tokens at position 7, the sequences separate
completely — so a single tie flips the whole rest of the output. This is why
`test_flashinfer_rocm_attention_backend.py` reports every mismatched prompt with
both texts rather than asserting on the first: **exact match across kernels is
not a sound correctness bar**, and a failure there needs this same logprob check
before it means anything.

Kernel-level agreement is the load-bearing check instead: causal paged prefill,
non-causal prefill, and decode each match a torch SDPA reference to <0.01 in
`test/registered/kernels/test_flashinfer_rocm_attention_kernels.py`.

Kernel-level checks live in
`test/registered/kernels/test_flashinfer_rocm_attention_kernels.py`; end-to-end
accuracy and the aiter equivalence check live in
`test/registered/attention/test_flashinfer_rocm_attention_backend.py`.
