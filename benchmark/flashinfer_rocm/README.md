# flashinfer_rocm vs aiter on MI300X

Baseline numbers for the `flashinfer_rocm` attention backend against `aiter`,
the established ROCm backend, so later changes have something to move.

## Method

```bash
docker build -f docker/rocm-flashinfer.Dockerfile -t sglang:rocm-flashinfer .

# Throughput, all backends and batch sizes. Second arg pins the GPU.
bash benchmark/flashinfer_rocm/bench_backends.sh \
  mistralai/Mistral-7B-Instruct-v0.3 3

# Numerical agreement, three-way. Re-run after any kernel change.
python3 benchmark/flashinfer_rocm/compare_backends.py --gpu 3
```

**Pin to an idle GPU.** This box is shared; check `rocm-smi --showmeminfo vram`
first, where an idle CDNA3 card sits at ~0.3 GB. The first round of these
numbers was collected on a device another process was using.

That turned out not to matter — re-running the whole Mistral-7B sweep on an idle
GPU reproduced the contended run within ~2% on every cell (decode ratio 0.888 →
0.898, 0.897 → 0.917, 0.930 → 0.929; the absolute decode gap stayed 0.67 / 0.58
/ 0.57 ms against 0.74 / 0.72 / 0.56 ms). Both backends were competing on equal
terms, so the comparison held. Worth re-checking rather than assuming, since a
contended baseline that penalised one side unequally would have invalidated the
central claim.

`one_batch` prints a warmup pass and then a benchmark pass; the tables below use
the **benchmark** pass. Decode CUDA graphs are on for both backends; prefill
graphs are off for both, because ROCm disables that phase for every attention
backend (see caveats).

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

### Attributing the gap — measured, not inferred

An earlier revision of this file blamed the plan path outright. That was wrong,
and worth recording because the reasoning was superficially sound: SGLang's
NVIDIA backend carries a `fast_decode_plan` to dodge device-to-host copies
inside `plan()`, the ROCm port exports no equivalent, and the gap had the right
shape. Direct measurement says otherwise.

Timing the pieces at batch 8 on TinyLlama (22 layers), where the step gap is
0.50 ms:

| Component | `flashinfer_rocm` | `aiter` | gap | share |
| --- | ---: | ---: | ---: | ---: |
| `init_forward_metadata_out_graph` (per step) | 0.190 ms | 0.080 ms | 0.109 ms | **22%** |
| everything else (attention kernels) | — | — | ~0.39 ms | **78%** |

And within that metadata figure, the D2H sync is minor: a whole `plan()` call
costs 0.070 ms, dropping to 0.029 ms when handed CPU-resident inputs, so the
sync is worth **0.039 ms** — about 8% of the step gap, not the whole of it.

The remaining ~0.39 ms is the decode kernel. `fa2` decode `run()` measures
0.0405 ms/call at these shapes, i.e. 0.891 ms/step across 22 layers; closing
0.39 ms would require AITER's kernel to be about **1.78x faster per call**.
That is an unremarkable margin between a hand-tuned CDNA3 assembly/CK kernel
and a HIP port of a CUDA kernel.

**Correction to the eager result below:** eager mode showing `fa2` ahead of
AITER does *not* clear the kernel. Eager pays large per-layer Python and launch
overhead on both sides, which swamps a 0.02 ms/call kernel difference. Only the
graph-captured measurement isolates kernel time, and it puts AITER ahead.

So the optimization splits, and the larger half is not in SGLang:

1. **~0.39 ms/step — kernel work in the ROCm fork.** Out of scope here; belongs
   with the FlashInfer HIP decode kernel.
2. **~0.11 ms/step — metadata prep in this backend.** Addressable: pass
   CPU-resident `indptr` / `last_page_len` / `seq_lens` into `plan()` (worth
   0.039 ms), and cut the cost of building kv-indices, where AITER's equivalent
   is 2.4x cheaper.

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

So `fa2` decode is competitive here; CUDA graphs just help AITER more (6.3× vs
4.6×). Read this table only as evidence that the eager kernel is not obviously
deficient — **not** as evidence about where the graph-mode gap comes from. See
"Attributing the gap" above: the plan path is ~8% of it and the kernel ~78%, and
eager cannot resolve a 0.02 ms/call difference through per-layer Python overhead.

Prefill is a separate story: `aiter` leads there in every mode, and part of that
is `page_size=1` (see caveats).

## Caveats

- **1.1B model.** Attention is a smaller share of total time than it would be at
  8B, and prefill numbers at this size are dominated by launch overhead. Re-run
  on Llama-3.1-8B (32/8 → GQA group 4) before drawing conclusions about
  production shapes.
- **`page_size=1`, and it costs nothing.** It is this backend's restriction
  rather than the kernel's — the ROCm `fa2` path runs correctly at page sizes 1
  through 128 — but lifting it would not help. Measured at batch 8, seq 1024,
  identical total KV, only the paging granularity varying:

  | page_size | prefill ms | decode ms |
  | ---: | ---: | ---: |
  | 1 | 1.0546 | 0.0327 |
  | 16 | 1.0583 | 0.0327 |
  | 64 | 1.0605 | 0.0328 |
  | 128 | 1.0608 | 0.0328 |
  | 256 | 1.0612 | 0.0525 |

  Flat within 0.6% up to 128, and page 256 makes decode 38% *worse*. An earlier
  revision of this file called page-size sweeping "the first prefill
  optimization to try", reasoning that AITER's CK FMHA prefers
  `{128, 256, 1024}` and page-size 1 must therefore force a gather. That
  reasoning does not apply: the default kernel family is `fa2`, not AITER, and
  `fa2` is indifferent to page size. The page-granular kv-indices work has no
  payoff and was abandoned on this evidence.
- **Prefill CUDA graphs are off for every backend on ROCm**, so they do not skew
  the comparison. An earlier revision of this file claimed they were available to
  `aiter` but not to `flashinfer_rocm` and therefore flattered `aiter`; that was
  wrong. The HIP default is `tc_piecewise`, and it is subsequently disabled —
  resolving `ServerArgs` for `flashinfer_rocm`, `aiter` and `triton` on this host
  yields `prefill=disabled` and `decode=full` for all three. `flashinfer_rocm`
  also disables the phase explicitly, which is therefore redundant rather than a
  handicap.
- Single run per cell, no clock locking, on a shared machine. Treat differences
  under ~5% as noise.
- **`SGLANG_FLASHINFER_ROCM_BACKEND=aiter` is impractical under CUDA graphs
  today.** AITER JIT-builds a kernel variant per captured graph shape; a
  capture ladder up to batch 32 was still compiling after 30 minutes of solid
  CPU on batch size 1 alone. The `aiter`-kernel row above is therefore eager
  only. Prebuilding the AITER kernel cache into the image would be needed to
  measure that combination under graphs.
- FlashInfer's AITER prefill path dlopen's an AITER `.so` it cannot build
  itself. This used to require `warm_aiter_mha.py`-style prewarming and
  otherwise failed with "AITER .so not found"; upstream PRs #275 and #276 now
  bootstrap the library and probe for native paging instead of trusting the
  AITER version, so `backend=auto` resolves to AITER and runs unaided at page
  sizes 1 through 256. Page size 1024 still fails a `dlsym` lookup.

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

### Prefill: both optimization hypotheses tested, both weak

Prefill trails `aiter` by 5-10%, and the two candidate explanations were
measured rather than argued. Neither holds up.

**Page size** — no effect; see the caveat below.

**Paged vs ragged prefill.** This backend always uses the paged wrapper: KV goes
to the pool first, then attention reads it through `kv_indices`, which costs a
gather. The ragged wrapper takes contiguous k/v with no indirection, and for
prefill with no prefix cache both compute the same thing. Measured:

| batch | seq | paged ms | ragged ms | ragged speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.0736 | 0.0675 | 1.09x |
| 8 | 512 | 0.3429 | 0.3700 | **0.93x** |
| 8 | 2048 | 3.7860 | 3.5332 | 1.07x |
| 32 | 512 | 1.3169 | 1.2410 | 1.06x |

~4% on average, inconsistent, and paged is *faster* at batch 8 / seq 512. Adding
the ragged + paged + `merge_state` split would buy little and cost a
`merge_state` dependency on HIP.

So prefill has the same shape as decode: the residual is the kernel, not the
plumbing around it. There is no significant prefill headroom inside SGLang.

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

Kernel-level agreement is one load-bearing check: causal paged prefill,
non-causal prefill, and decode each match a torch SDPA reference to <0.01 in
`test/registered/kernels/test_flashinfer_rocm_attention_kernels.py`. That only
covers a single forward pass, though — the three-way comparison below covers the
autoregressive loop.

### Three-way agreement, and which backend is the outlier

Comparing two backends tells you they differ, not which one is right. Max
absolute difference in the **probability** assigned to the chosen token, over
the tokens both backends selected identically (Mistral-7B, 32 tokens, 4 prompts):

| pair | max Δp | token agreement |
| --- | ---: | --- |
| `flashinfer_rocm` vs `triton` | **0.031** | 32/32 |
| `flashinfer_rocm` vs `aiter` | 0.101 | 32/32 |
| `aiter` vs `triton` | **0.109** | 32/32 |

**`flashinfer_rocm` is 3.5x closer to the `triton` reference than `aiter` is.**
The `flashinfer_rocm`-vs-`aiter` figure (0.101) is essentially `aiter`'s own
deviation from `triton` (0.109), so that gap is dominated by `aiter`, not by
this backend. All three pick identical tokens at every position; they differ
only in how much probability mass sits on them, which is the room bf16 attention
has where the model is uncertain.

Two methodology notes, both learned the hard way here:

- **Compare probabilities, not logprobs.** `|Δ log p|` is scale-distorted: a
  step where two backends put 8.3% vs 6.5% on the chosen token — a difference of
  0.019 that matters to nobody — reads as 0.2575 in log space. An earlier
  revision of the equivalence test used log space and failed 2/4 prompts on
  what was almost entirely this artifact.
- **`flashinfer_rocm` is bit-deterministic across server launches** (0.0000
  delta, 32/32 tokens, all prompts), so any cross-backend delta is a real
  implementation difference rather than run-to-run noise. Worth re-establishing
  before reading anything into a comparison.

Kernel-level checks live in
`test/registered/kernels/test_flashinfer_rocm_attention_kernels.py`; end-to-end
accuracy and the aiter equivalence check live in
`test/registered/attention/test_flashinfer_rocm_attention_backend.py`.
