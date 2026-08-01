# Handoff: two ROCm FlashInfer items (gfx942)

1. **head_dim 192 does not compile** — blocks DeepSeek/MLA prefill outright.
2. **The HIP fa2 decode kernel is ~1.8x slower than AITER** — a 7-11%
   end-to-end deficit at 7B.

Plus a release item: the published `amd-flashinfer` wheel has a causal-prefill
correctness bug that is already fixed in git but not on the index.

# Item 1: prefill kernels do not compile at head_dim 192

Written 2026-07-31 from the SGLang `flashinfer_rocm` backend work, and kept in
the SGLang tree because that is where the evidence was produced. The work it
describes belongs in **AMD-Ecosystem/flashinfer**, not here.

All file paths below are relative to a `flashinfer` checkout.

## First, a correction: there are two different "causal prefill" issues

Do not go hunting for the one that is already fixed.

### 1. Causal prefill returns wrong values — **ALREADY FIXED**, not published

The published wheel `amd-flashinfer 0.5.3+amd.1` (the only version on
`pypi.amd.com`, and the one in `rocm/flashinfer:flashinfer-0.5.3.amd1_*`)
computes **causal** paged prefill incorrectly on gfx942: some output rows are
exact, others are badly wrong, while the output stays right-shaped and finite.

Measured against a torch SDPA reference, max abs error:

| Case | amd.1 | amd.2 |
| --- | ---: | ---: |
| MHA bs1 seq8 h1 causal | 2.36 | 0.0006 |
| MHA bs1 seq8 h4 causal | 2.45 | 0.0010 |
| GQA bs1 seq8 4/1 causal | 3.29 | 0.0010 |
| MHA bs1 seq64 h4 causal | 2.56 | 0.0009 |
| MHA bs1 seq8 non-causal | 0.0005 | 0.0005 |

Reproduces on both fp16 and bf16, so it is *not* the bf16 softmax-scaling half
of the fix. Per-token error for `seq8 h1` was
`[0.0, 0.529, 2.047, 0.966, 2.359, 0.0, 0.169, 0.0]` — the some-rows-right,
some-rows-wrong pattern of a row-index bug.

**Fixed by `a7a2f0b6` (PR #214), "fix(hip): correct bf16 rowsum MMA and
custom-mask MFMA row indexing on CDNA3"**, which touches
`include/gpu_iface/backend/hip/mma_hip.h` and two locations in
`include/flashinfer/attention/generic/prefill.cuh`. Tagged `v0.5.3+amd.2`.

**So there is no kernel work here.** The only open item is that `amd.2` is not
on `pypi.amd.com` — `pip install amd-flashinfer` still yields `amd.1` and its
silently-wrong causal prefill. That is a release/packaging action, and arguably
an urgent one: the failure mode is wrong numbers, not a crash.

### 2. head_dim 192 fails to compile — **THE ACTUAL OPEN BUG**

This is what needs fixing, and it blocks DeepSeek/MLA prefill entirely.

## Symptom

Any prefill kernel with `head_dim == 192` fails to **build** (JIT, at
`amdclang++` time — not a runtime error):

```
include/flashinfer/attention/generic/permuted_smem.cuh:125:63:
  error: static assertion failed due to requirement
  'stride > 0u && (stride & (stride - 1u)) == 0u'
  "k128B_16Row advance_offset_by_column<2/4> requires a non-zero power-of-2
   stride template argument"
  note: expression evaluates to '32 == 0'
```

## Scope — it is the head dim, not the mask and not the asymmetry

Verified at `cd6b40d2` (`0.5.3+amd.2.dev52`) on gfx942, bf16, ragged prefill:

| head_dim_qk | head_dim_vo | causal | result |
| ---: | ---: | --- | --- |
| 128 | 128 | true | OK |
| 128 | 128 | false | OK |
| 192 | 128 | false | **FAIL** |
| 192 | 128 | true | **FAIL** |
| 192 | 192 | true | **FAIL** |
| 256 | 256 | true | OK |

Three things this rules out:

- **Not causal-specific** — non-causal 192 fails identically.
- **Not the `head_dim_qk != head_dim_vo` asymmetry** — symmetric 192/192 fails too.
- **Not "large head dims"** — 256 compiles and runs fine.

192 is simply the only tested head dim that is not a power of two.

## Likely root cause: a regression from PR #207

`a01dea2f` (PR #207, "New k128_16row_swizzle mode to reduce LDS bank conflicts
for prefill kernels") flipped both swizzle modes **unconditionally**:

```c++
// include/flashinfer/attention/generic/prefill.cuh:122-123  (current)
static constexpr SwizzleMode SWIZZLE_MODE_Q  = SwizzleMode::k128B_16Row;
static constexpr SwizzleMode SWIZZLE_MODE_KV = SwizzleMode::k128B_16Row;
```

Before #207 both were `SwizzleMode::k128B`. The offending `static_assert` lives
only in the `k128B_16Row` branch of `advance_offset_by_column<step>` for
`step ∈ {2, 4}` (`permuted_smem.cuh:125`) — `k128B` has no power-of-2 stride
requirement.

**Not yet verified**, and the first thing to check: build a 192/128 prefill
kernel at `a01dea2f^`. If it compiles there, this is a straightforward
regression and the fix is scoping, not new math.

## Suggested directions, cheapest first

1. **Select the swizzle by stride.** Make `SWIZZLE_MODE_Q/KV` fall back to
   `k128B` when the derived stride is not a power of two, keeping `k128B_16Row`
   (and its LDS-bank-conflict win) for 128/256. Restores 192 at the cost of the
   #207 speedup on that one shape.
2. **Generalise the exact XOR formula.** The comment at `permuted_smem.cuh:120`
   explains the `step ∈ {2,4}` path needs the exact formula because the simple
   `+8 / ^8` shortcut is not universally correct. Making it work for non-power-of-2
   strides is the real fix, but needs the swizzle math re-derived.
3. **Pad the head dim to 256 in shared memory** for the 192 case. Wastes LDS and
   likely hurts occupancy; only worth it if 1 and 2 both fail.

Whichever route: add 192/128 to the AOT/JIT test matrix. `aot_hip.py:283`
currently lists `fa2_head_dim = [(64,64), (128,128), (256,256)]`, so no test
covers 192 today and nothing would have caught this.

## Why it matters downstream

DeepSeek MLA prefill is ragged attention at `head_dim_qk=192` /
`head_dim_vo=128`. With `fa2` unable to compile it and AITER rejecting
`head_dim_qk != head_dim_vo`, **there is no MLA prefill path on ROCm
FlashInfer at all.** SGLang's `flashinfer_rocm` backend currently refuses that
configuration at startup and directs users to a hybrid split
(`--prefill-attention-backend aiter --decode-attention-backend flashinfer_rocm`).

Fixing 192 would let MLA run end to end on this backend. A CUDA-graph-capable
`BatchMLAPagedAttentionWrapper` would then be the next blocker — the ROCm
wrapper takes no `use_cuda_graph` argument, which costs ~9x on MLA decode.

## Reproducer

```python
import torch, flashinfer
dev, dtype, H, S = "cuda", torch.bfloat16, 16, 64
ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
q = torch.randn(S, H, 192, dtype=dtype, device=dev)
k = torch.randn(S, H, 192, dtype=dtype, device=dev)
v = torch.randn(S, H, 128, dtype=dtype, device=dev)
ip = torch.tensor([0, S], dtype=torch.int32, device=dev)
w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
w.plan(ip, ip, H, H, 192, head_dim_vo=128, causal=True,
       q_data_type=dtype, kv_data_type=dtype)
w.run(q, k, v)   # fails during JIT compile
```

Swap `192 -> 128` or `-> 256` (with matching vo) to see it pass.

---

# Second item: the HIP fa2 decode kernel is ~1.8x slower than AITER

Independent of the head_dim-192 bug, and the larger of the two in end-user
terms. This is a performance gap, not a correctness one.

## The measurement

SGLang's `flashinfer_rocm` backend trails `--attention-backend aiter` on decode.
Locating that gap at batch 8 on TinyLlama-1.1B (22 layers, 32/4 heads,
`head_dim=64`), where the end-to-end step gap is 0.50 ms:

| Component | flashinfer_rocm | aiter | gap | share |
| --- | ---: | ---: | ---: | ---: |
| attention metadata prep, per step | 0.190 ms | 0.080 ms | 0.109 ms | 22% |
| attention kernels | — | — | ~0.39 ms | **78%** |

Direct kernel timing, `BatchDecodeWithPagedKVCacheWrapper(backend="fa2",
use_tensor_cores=False)` at batch 8, seq 544, `page_size=1`:

```
fa2 decode run(): 0.0405 ms/call
x22 layers      : 0.891 ms/step of attention
```

Closing the 0.39 ms residual requires AITER to be about **1.78x faster per
call** (0.0405 -> 0.0228 ms). That is an unremarkable margin between hand-tuned
CDNA3 assembly/CK and a HIP port of a CUDA kernel, and it is consistent across
model sizes: the absolute gap holds near 0.6 ms/step on both TinyLlama-1.1B and
Mistral-7B, and across batch 1 / 8 / 32.

At 7B this leaves `flashinfer_rocm` within 7-11% of AITER on decode — so the
gap is not alarming, but it is the single largest remaining item.

## What this rules out

- **Not the plan path.** A full ROCm `plan()` call is 0.070 ms, dropping to
  0.029 ms when handed host-resident `indptr` / `last_page_len` / `seq_lens`.
  The device-to-host sync is therefore ~0.039 ms, about 8% of the step gap.
  SGLang now passes host-resident inputs, which banked exactly that.
- **Not visible in eager mode.** With CUDA graphs off, `fa2` measures *ahead*
  of AITER on decode (710 vs 671 tok/s at batch 8). That is misleading: eager
  pays per-layer Python and launch overhead on both sides that swamps a
  0.02 ms/call kernel difference. Only graph-captured runs isolate kernel time.
  Do not use eager numbers to evaluate this.

## Do not trade away accuracy to close this

`fa2` is currently the *more* accurate of the two, so a naive chase of AITER's
speed could regress something that is presently better. Max absolute difference
in the probability assigned to the chosen token, over tokens each pair selected
identically (Mistral-7B, 32 tokens, 4 prompts, via SGLang):

| pair | max Δp |
| --- | ---: |
| `fa2` (flashinfer_rocm) vs `triton` | **0.031** |
| `fa2` vs AITER | 0.101 |
| AITER vs `triton` | **0.109** |

`fa2` sits 3.5x closer to the `triton` reference than AITER does, and the
fa2-vs-AITER figure is essentially AITER's own deviation from `triton`. All
three pick identical tokens at every position; they differ only in how much
probability mass sits on them.

Two implications:

1. Some of AITER's speed may be bought with numerical shortcuts. Matching its
   *time* is the goal; matching its *numerics* is not.
2. Whatever changes here, re-run the three-way comparison rather than a pairwise
   one. A pairwise delta cannot tell you which side moved — that mistake nearly
   sent this investigation after a non-existent bug in `fa2`.

Compare probabilities, not logprobs: `|Δ log p|` is scale-distorted, and a step
where two backends put 8.3% vs 6.5% on the chosen token reads as 0.2575 in log
space while being irrelevant.

## Where to look

`BatchDecodeWithPagedKVCacheWrapper` with `backend="fa2"`,
`use_tensor_cores=False`, `page_size=1`, `kv_layout="NHD"`, bf16 — the
configuration SGLang drives. Worth checking first:

1. **`page_size=1` overhead.** SGLang's ROCm default is 1, so every KV index is
   its own page. If the HIP decode kernel has a per-page cost, this is the
   worst case for it. Comparing 0.0405 ms/call at page_size 1 against larger
   page sizes at equal total KV would isolate that quickly.
2. **`use_tensor_cores=False`.** SGLang picks this because AITER decode
   requires it, but the in-tree HIP path may do better with MFMA enabled.
   Cheap to test.
3. **GQA group 4/8 specifically.** The dispatch instantiates group sizes
   1/2/3/4/8 (`include/gpu_iface/dispatch.cuh`); 4 and 8 are what real models
   hit (Llama-3.x and Mistral are 4, TinyLlama is 8).

## Reproducer

```python
import time, torch, flashinfer
dev, dtype = "cuda", torch.bfloat16
HQ, HKV, HD, BS, SEQ = 32, 4, 64, 8, 544
ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
n = BS * SEQ
k = torch.randn(n, HKV, HD, dtype=dtype, device=dev)
v = torch.randn(n, HKV, HD, dtype=dtype, device=dev)
kv = (k.view(n, 1, HKV, HD), v.view(n, 1, HKV, HD))
indptr = torch.arange(BS + 1, dtype=torch.int32, device=dev) * SEQ
indices = torch.arange(n, dtype=torch.int32, device=dev)
lpl = torch.ones(BS, dtype=torch.int32, device=dev)
q = torch.randn(BS, HQ, HD, dtype=dtype, device=dev)
w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend="fa2",
                                                  use_tensor_cores=False)
w.plan(indptr, indices, lpl, HQ, HKV, HD, 1,
       q_data_type=dtype, kv_data_type=dtype, non_blocking=True)
for _ in range(50):
    w.run(q, kv)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(500):
    w.run(q, kv)
torch.cuda.synchronize()
print(f"{(time.perf_counter()-t0)/500*1000:.4f} ms/call")   # 0.0405 on MI300X
```

## Priority against the head_dim-192 bug

They serve different ends. **head_dim 192 is a capability gap** — it blocks
DeepSeek/MLA prefill entirely, so no amount of tuning substitutes for it. **The
decode kernel is a performance gap** of 7-11% at 7B, which is a real but
survivable deficit against AITER. If only one gets attention, 192 unblocks more.
