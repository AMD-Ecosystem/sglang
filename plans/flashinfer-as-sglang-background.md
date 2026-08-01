# Integrate ROCm FlashInfer as an SGLang attention backend

## Context

SGLang on AMD GPUs today runs attention through **AITER** (default for MHA and
for MLA when kv-head count is 16/128), or **Triton** otherwise. SGLang's
existing `flashinfer` backend is NVIDIA-only — `is_flashinfer_available()` in
`python/sglang/srt/utils/common.py:350` literally ends with `and is_cuda()`.

Meanwhile AMD ships a mature ROCm port of FlashInfer
([AMD-Ecosystem/flashinfer](https://github.com/AMD-Ecosystem/flashinfer),
branch `amd-integration`, released as `amd-flashinfer`), which is already
cloned at `~/devel/flashinfer`. It provides in-tree HIP kernels plus optional
AITER dispatch for prefill, decode, MLA, paged-KV append, RoPE, norm, and
sampling on gfx942/gfx950.

**Goal:** add `flashinfer_rocm` as a selectable SGLang attention backend on
ROCm, prove it is numerically aligned with the AITER backend, benchmark it
against AITER, and ship a reproducible container + tests so it stays working.

**Non-goal:** vLLM. (Earlier drafts scoped a vLLM cross-check; that is dropped —
this plan is SGLang-only.)

## Status (2026-07-31)

Implemented on branch `feat/flashinfer-rocm-attention-backend`, committed
locally, **not pushed**.

| Milestone | State | Evidence |
| --- | --- | --- |
| M0 container | Done | `docker/rocm-flashinfer.Dockerfile`; `rocm-flashinfer-smoke.py` green |
| M1 registration | Done | `--attention-backend flashinfer_rocm` dispatches; startup validation |
| M2 MHA eager | Done | 69K tok/s prefill, 1650 tok/s decode (TinyLlama-1.1B) |
| M3 CUDA graphs | Done (decode only) | 4.6× over eager; output still matches aiter |
| M4 MLA | **Decode-only; prefill impossible** | see "MLA is decode-only" below |
| M5 tests | Kernel 3/3 green; E2E MMLU green; **one class still unrun** | see "What is not exercised" |
| M6 benchmark | Done | `benchmark/flashinfer_rocm/README.md`; TinyLlama + Mistral-7B |
| M7 optimization | In-scope part done | 78% of the gap is upstream kernel work — see below |
| M8 docs | Done | `attention_backend.mdx` matrix row + launch section |

**Headline correctness result:** greedy decoding with CUDA graphs on for both —
4/4 outputs byte-identical to `aiter` on TinyLlama, 3/4 on Mistral-7B. The one
Mistral divergence is a tie-break, not a wrong result: at output token 7 aiter's
top two candidates are exactly equal (-2.19410 both), so greedy picks between
them on tie-break order and the sequences separate from there. Exact output
match is therefore **not** a sound cross-kernel correctness bar; the kernel-level
SDPA comparison (<0.01) is the load-bearing check.

### Where the time goes — measured (M7)

An earlier revision of this section blamed the plan path for the whole decode
gap. That was **wrong**, and the correction is the main M7 finding. The
reasoning was superficially sound — the gap is a flat ~0.6 ms across model
sizes and batch sizes, which is the signature of fixed per-step overhead, and
SGLang's NVIDIA backend carries a `fast_decode_plan` that the ROCm port lacks —
but the attribution was a guess written up as a conclusion.

Measured at batch 8 on TinyLlama (22 layers), step gap 0.50 ms:

| Component | flashinfer_rocm | aiter | gap | share |
| --- | ---: | ---: | ---: | ---: |
| `init_forward_metadata_out_graph` | 0.190 ms | 0.080 ms | 0.109 ms | 22% |
| attention kernels | — | — | ~0.39 ms | **78%** |

Within the metadata, a whole `plan()` call is 0.070 ms, falling to 0.029 ms on
host-resident inputs — so the D2H sync is 0.039 ms, ~8% of the step gap. That
part is **done**: `plan()` now receives pinned host mirrors plus the scheduler's
`seq_lens_cpu`, measured 0.190 → 0.146 ms/step with output unchanged.

The remaining ~0.39 ms is the decode kernel. `fa2` decode `run()` is
0.0405 ms/call, i.e. 0.891 ms/step over 22 layers; closing 0.39 ms needs AITER
to be ~1.78× faster per call. **That is kernel work in AMD-Ecosystem/flashinfer,
not something this backend can fix** — see `HANDOFF-head-dim-192.md`.

Also retracted: eager mode showing `fa2` ahead of AITER does *not* clear the
kernel. Eager pays per-layer Python and launch overhead on both sides that
swamps a 0.02 ms/call difference. Only graph-captured runs isolate kernel time.

Still open in-scope: metadata is 0.146 ms vs aiter's 0.074 ms, so ~0.072 ms
remains in the kv-indices build. Low value against a gap that is already only
7–11% at 7B.

### MLA is decode-only

DeepSeek prefill is ragged attention at `head_dim_qk=192` / `head_dim_vo=128`.
Neither ROCm path can do it: `fa2` fails to **compile** for that shape (its
`k128B_16Row` swizzle `static_assert`s on a power-of-2 stride; 192 is not one),
and AITER rejects `head_dim_qk != head_dim_vo`. Not fixable in SGLang — it needs
an upstream change in the fork. `_validate_flashinfer_rocm_backend` now rejects
the config at startup instead of failing minutes into a JIT build.

Decode works hybrid (`--prefill-attention-backend aiter
--decode-attention-backend flashinfer_rocm`) but is not production-viable:
28.3 tok/s vs aiter's 351.4 with graphs. Eager-to-eager the gap is only 0.71×;
the rest is that ROCm's `BatchMLAPagedAttentionWrapper` takes no
`use_cuda_graph` argument at all, and graphs are worth ~8.8× here.

### Model constraints discovered by running

- **GQA group size must be 1, 2, 3, 4, or 8** — the ROCm decode kernel
  instantiates no others (`include/gpu_iface/dispatch.cuh`). Qwen2.5-0.5B (14/2)
  and Qwen2.5-7B (28/4) abort mid-decode with `Unsupported group_size: 7`;
  the backend now rejects them at construction. Llama-3.x (32/8 = 4) is fine.
- **`SGLANG_FLASHINFER_ROCM_BACKEND=aiter` is impractical under CUDA graphs** —
  AITER JIT-builds a kernel variant per captured shape; a ladder to batch 32 was
  still compiling after 30 minutes on batch size 1 alone.
- **FlashInfer's AITER prefill path `dlopen`s an AITER `.so` it cannot build
  itself** — fails with "AITER .so not found" until `aiter.ops.mha.
  mha_varlen_fwd` is called separately with matching dtype/causal/lse.

### What is not exercised

- **`TestFlashInferRocmMatchesAiter` has never run.** The equivalent comparison
  was done by hand (4/4 TinyLlama, 3/4 Mistral with the tie explained), but the
  test itself is committed and unexecuted — the same situation that produced a
  false failure in the MMLU test.
- **Prefill was never investigated.** All M7 work was decode. Prefill sits 5–10%
  behind aiter and two hypotheses are untested: `page_size > 1`, and the
  ragged + paged + `merge_state` split deliberately skipped in M2. Prefill
  dominates at long context, so this may matter more than the decode remainder.

### Open decisions

- **Push target.** `origin` is `sgl-project/sglang` itself, so pushing publishes
  a branch on the official repo rather than backing anything up.
- **The `amd.1` disclosure.** The published wheel's causal-prefill bug is
  documented in `benchmark/flashinfer_rocm/README.md`, `plans/`, and the error
  message in `docker/rocm-flashinfer-smoke.py`. Accurate and verified, but no
  fixed wheel is on the index yet.
- **Upstream PR or internal branch?** Changes what else is needed.

### Model access notes

`meta-llama/*` is gated for this account and `lmsys/sglang-ci-dsv3-test` is
private, so: Mistral-7B-Instruct-v0.3 substitutes for Llama-3.1-8B (identical
32/8 attention shape) via `SGLANG_TEST_ATTENTION_MODEL`, and
`deepseek-ai/DeepSeek-V2-Lite` substitutes for the MLA CI model.

## Ground truth established during exploration

Local environment:

| Thing | Value |
| --- | --- |
| GPU | 1× MI300X, `gfx942` |
| Host ROCm | 6.4.1 (too old — container supplies 7.2) |
| SGLang | `~/devel/sglang`, `sgl-project/sglang` `main` @ `425349b799` |
| AMD FlashInfer | `~/devel/flashinfer`, `amd-integration` @ `cd6b40d2`, version `0.5.3+amd.2` |
| Reference image | `rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1` |
| Working recipe | `~/devel/flashinfer-bench/docker/rocm/Dockerfile` (ROCm 7.2 / py3.12 / torch 2.9.1 / amd-flashinfer 0.5.3 / amd_aiter 0.1.10) |

Critical constraints from the AMD fork's README feature matrix:

- **MLA is AITER-only** — no HIP kernel, `bfloat16` only, `page_size=1` only.
  So "FlashInfer MLA vs AITER MLA" compares plumbing, not kernels.
- **Decode CUDA-graph capture on the AITER route is opt-in** (`backend="aiter"`)
  and fixes grid + `.so` at capture-time shapes → **must capture at max seq len**.
  Under `backend="auto"` capture silently uses the HIP `fa2` path instead.
- **AITER route silently ignores** ALiBi slopes, attention sinks, fp8 dequant
  scales, `use_fp16_qk_reduction`, and multimodal prefix-cache pointers. Any
  model needing those must be forced onto `backend="fa2"`.
- **Custom attention masks force the HIP path** — relevant if EAGLE is added later.
- AITER CK FMHA native page sizes are `{128, 256, 1024}` on `amd-aiter==0.1.10`;
  `page_size=1` works but pays an extra GPU gather.

Version gap: SGLang pins `flashinfer_python[cu13]==0.6.15.post1`; the AMD fork
is based on upstream **v0.5.3**. This is the main reason for a separate backend
rather than branching the existing one.

No packaging conflict on ROCm: SGLang's ROCm extras live in
`python/pyproject_other.toml` (`srt_hip`), which does **not** pull FlashInfer.
Both packages import as `flashinfer`, so they must never be co-installed.

## Findings from execution (update 2026-07-31)

Things that only surfaced by running against real kernels, recorded so they are
not rediscovered:

1. **The published `amd-flashinfer` wheel is unusable here.** `0.5.3+amd.1` is
   the only version on `pypi.amd.com`, and on gfx942 its *causal* paged prefill
   returns wrong values (max abs err 2.4–4.1 vs a torch SDPA reference) while
   decode and non-causal prefill are exact and the output stays right-shaped and
   finite. Reproduced on both bf16 and fp16, so it is the amd.2 "MFMA C/D row
   indexing for custom masks" fix, not the bf16 softmax-scaling one. It also does
   not export `BatchMLAPagedAttentionWrapper`. → the image builds FlashInfer from
   a pinned `amd-integration` ref (`cd6b40d2`).
2. **`docker build` has no GPU**, and FlashInfer resolves its JIT workspace via
   `torch.cuda` at *import* time. Any `import flashinfer` in a build step dies
   with "No HIP GPUs are available"; all verification must be import-free at
   build time and run in the container.
3. **`pip install -e "python[srt_hip]"` silently replaces the ROCm torch.**
   `srt_hip` leaves `torch` unpinned and pulls `timm` → `torchvision`; PyPI
   torchvision pins an exact torch, so the resolver installed `torch 2.10.0+cu128`
   plus the whole `nvidia-*-cu12` stack over `2.9.1+rocm7.2.0`. Fixed by
   installing ROCm `torchvision 0.24.0+rocm7.2.0` first, then freezing
   torch/torchvision/triton into a `PIP_CONSTRAINT` file.
4. **The base image's bundled AITER is too old for SGLang main** —
   `sglang.srt.layers.quantization.fp8` imports `aiter.ops.shuffle.shuffle_scale`,
   absent from `amd_aiter 0.1.x`. AITER is built from SGLang's pinned
   `AITER_COMMIT_DEFAULT` instead.
5. **Verified working**: the `(k_buffer, v_buffer)` tuple viewed as
   `[num_pages, 1, H, D]` at `page_size=1` is the correct paged-KV layout for
   both the prefill and decode wrappers, matching SDPA to ~1e-3.

## Approach

Add a **new, self-contained backend** — `python/sglang/srt/layers/attention/flashinfer_rocm_backend.py`,
registered as `flashinfer_rocm`. It targets only the API subset the AMD fork
implements, leaves the NVIDIA `flashinfer_backend.py` untouched, and can be
iterated and benchmarked independently.

Milestones below are ordered so each one is independently verifiable.

---

## M0 — Container and environment

Create `docker/rocm-flashinfer.Dockerfile`, modelled on
`~/devel/flashinfer-bench/docker/rocm/Dockerfile` (known-good pins) and on
SGLang's existing `docker/rocm.Dockerfile` (for the SGLang install steps).

- Base `rocm/dev-ubuntu-24.04:7.2-complete`; py3.12; `torch==2.9.1` from
  `-f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` (find-links, **not**
  `--index-url` — that repo is a flat listing, not a PEP 503 index).
- `amd-flashinfer==0.5.3` from `--extra-index-url https://pypi.amd.com/rocm-7.2.0/simple/`.
- `amd_aiter==0.1.10` from `https://pypi.amd.com/rocm-7.1.1/simple/` (keeps the
  AITER backend installed so it remains the benchmark baseline in the same image).
- Install SGLang the way `docker/rocm.Dockerfile` does: build the AOT kernels via
  `python/sglang/kernels/aot/setup_rocm.py`, swap `pyproject_other.toml` into
  place, then `pip install -e "python[srt_hip]"`.
- Add a `--build-arg FLASHINFER_SRC=` escape hatch that instead does
  `pip install --no-build-isolation -ve /workspace/flashinfer` so `~/devel/flashinfer`
  can be bind-mounted for kernel-side debugging.
- Bind-mount SGLang at `/sgl-workspace/sglang` so edits are live.

Optionally add a `srt_hip_flashinfer` extra in `python/pyproject_other.toml`
(= `srt_hip` + `amd-flashinfer`) so the dependency is declared rather than
Docker-only. Caveat: `amd-flashinfer` lives on `pypi.amd.com`, not public PyPI,
so the extra only resolves with `--extra-index-url` — document that next to the
extra or keep the install Docker-side. Existing extras start at
`python/pyproject_other.toml:97`.

**Verify:** in the container, `python3 -c "import flashinfer, torch; print(flashinfer.__version__, torch.__version__, torch.version.hip)"`
and run a couple of the fork's own ROCm tests (`~/devel/flashinfer/tests/rocm_tests/test_*_hip.py`).

## M1 — Registration and flag plumbing

Four small edits, all following the existing `aiter` / `wave` precedent:

1. `python/sglang/srt/server_args.py:174` — add `"flashinfer_rocm"` to the
   `# AMD specific` section of `ATTENTION_BACKEND_CHOICES`.
2. `python/sglang/srt/layers/attention/attention_registry.py` — add a
   `@register_attention_backend("flashinfer_rocm")` factory that branches on
   `runner.use_mla_backend` (mirroring `create_flashinfer_backend`) and raises a
   clear error when `not _is_hip`. `_is_hip` is already imported there.
3. `python/sglang/srt/utils/common.py:350` — do **not** relax
   `is_flashinfer_available()`; it gates the NVIDIA default path. Add a separate
   `is_flashinfer_rocm_available()` (`find_spec("flashinfer") is not None and is_hip()`).
4. Leave `_get_default_attn_backend` (`server_args.py:5552`) alone for now —
   AITER stays the ROCm default; `flashinfer_rocm` is strictly opt-in until it
   beats AITER. Add validation in `_handle_attention_backend_compatibility`
   (`server_args.py:5628`) rejecting the backend on non-HIP and on non-gfx942/gfx950.

**Verify:** `--attention-backend flashinfer_rocm` reaches the factory and fails
with the intended "not implemented yet" error, not an "invalid backend" error.

## M2 — MHA/GQA: eager prefill + decode

Implement the `AttentionBackend` contract from
`python/sglang/srt/layers/attention/base_attn_backend.py`. Note this tree uses
the **new 3-method contract** — `init_forward_metadata_out_graph(fb, in_capture)`
and `init_forward_metadata_in_graph(fb)`; the legacy
`init_forward_metadata_capture_cuda_graph` / `_replay_cuda_graph` are removed
from the ABC. Do not copy the old shape from stale examples.

Minimum surface: `init_forward_metadata`, `init_forward_metadata_out_graph`,
`forward_extend`, `forward_decode`, `get_cuda_graph_seq_len_fill_value`.

Structure to mirror from `flashinfer_backend.py`: an indices-updater object that
owns the `.plan()` side, with `forward_*` doing only `.run()`. Wrappers:

- `BatchPrefillWithRaggedKVCacheWrapper` — prefill with no prefix cache
- `BatchPrefillWithPagedKVCacheWrapper` — prefill with prefix cache
- `BatchDecodeWithPagedKVCacheWrapper` — decode

Three concrete deltas from the NVIDIA backend (see the compatibility section):
call `run()` / `run_return_lse()` on the ragged wrapper (it has no `forward`);
pass `window_left` / `sm_scale` / `logits_soft_cap` to ragged `plan()` rather
than per call; and omit `fixed_split_size` from the paged-prefill `plan()` call.
Do **not** copy `fast_prefill_plan` / `fast_decode_plan` — see M3.

> **As built:** MHA prefill uses the **paged wrapper only**. KV is written to the
> pool first, then attended over as one causal block, which is correct with or
> without a prefix. The ragged + paged + `merge_state` split the NVIDIA backend
> does is a throughput optimization, not a correctness requirement, and dropping
> it also avoids depending on `merge_state` on HIP. Revisit under M7.
> The ragged wrapper *is* used, but only by the MLA class.

Start at `page_size=1` and pass KV as the `(k_buffer, v_buffer)` tuple from
`token_to_kv_pool.get_kv_buffer(layer_id)`, viewed as
`[num_pages, 1, num_heads, head_dim]` — the same trick the NVIDIA backend uses,
and it matches MLA's `page_size=1` requirement.

Pin `backend="fa2"` (in-tree HIP) for the first correctness pass, not `"auto"`.
Rationale: `"auto"` silently reroutes to AITER, which would make this a
comparison of AITER against itself and would silently drop unsupported kwargs.
Expose `SGLANG_FLASHINFER_ROCM_BACKEND` (env var, per
[`env-var-conventions`](../.claude/skills/env-var-conventions/SKILL.md)) to switch
between `fa2` / `aiter` / `auto` for later A/B runs.

**Verify:** `--attention-backend flashinfer_rocm --disable-cuda-graph` on
Llama-3.1-8B; greedy outputs match `--attention-backend aiter` token-for-token
on a fixed prompt set; then MMLU ≥ 0.65 via the harness in
`test/registered/attention/test_triton_attention_backend.py`.

## M3 — CUDA graphs

Implement `init_cuda_graph_state(max_bs, max_num_tokens)` and the capture/replay
halves of `init_forward_metadata_out_graph(fb, in_capture=True/False)`.

Preallocate `kv_indptr`, `kv_indices`, `kv_last_page_len`, `qo_indptr` buffers and
construct decode wrappers with `use_cuda_graph=True` plus the `paged_kv_*_buffer`
kwargs, exactly as `_prepare_cuda_graph_metadata` does in `flashinfer_backend.py`.
Those constructor signatures are **identical** in the fork, so this part ports
cleanly for decode and paged prefill.

**Skip the `fast_*_plan` fast paths entirely.** SGLang's `fast_decode_plan` /
`fast_prefill_plan` reach into flashinfer privates and call
`_cached_module.plan()` with 19 positional args; the ROCm modules take 15 and
lay them out differently, and `fast_decode_plan` isn't even exported under HIP.
Use the public `plan()` and accept the host-sync cost for now — revisit in M7.

Also note **full-CG chunked prefill is out of scope here**: SGLang's design
relies on split-kv's `block_valid_mask` (`flashinfer_backend.py:1099`), and
`disable_split_kv` is a documented no-op on ROCm. Leave
`supports_full_cuda_graph_chunked_prefix = False`.

Two AMD-specific hazards to design around from the start (both documented in the
fork's `v0.5.3+amd.2` changelog entry):

- **`fa2` is the safe graph path.** Its graph support is *capacity-based and
  capture-order-independent*, which matches how SGLang captures a descending
  batch-size ladder. This is a second reason to default the backend to `"fa2"`.
- **The AITER decode graph path is opt-in and shape-fixed.** `backend="aiter"`
  under `use_cuda_graph=True` fixes the launch grid and the `.so` variant at
  capture-time shapes (the kernel early-exits per sequence on `context_lens`), so
  it must be **captured at max sequence length**; shorter replays are correct,
  longer are not. `backend="auto"` silently uses `fa2` under capture instead —
  make choosing AITER an explicit, logged decision rather than an accident.

Reference implementation and tests for this exact path already exist in the fork:
`benchmarks/rocm_benchmarks/bench_decode_graph_hip.py` plus its HIP graph-replay
tests. Mirror their capture discipline rather than rediscovering it.

**Verify:** same greedy-output and MMLU checks as M2 but with graphs enabled;
plus a replay-at-shorter-seq-len test to confirm the capture-at-max rule holds.

## M4 — MLA (DeepSeek)

Add the MLA sibling using `BatchMLAPagedAttentionWrapper`, dispatched from the
registry factory when `runner.use_mla_backend`.

Hard constraints to assert up front with clear errors: `bfloat16` only,
`page_size=1` only, AITER-only (`backend="fa2"` raises), and **eager only**.

**MLA cannot use CUDA graphs on this fork.** `mla_rocm.BatchMLAPagedAttentionWrapper.__init__`
is `(self, float_workspace_buffer, backend="auto")` — the `use_cuda_graph`,
`qo_indptr`, `kv_indptr`, `kv_indices`, `kv_len_arr` kwargs that the CUDA version
takes simply do not exist (`mla_rocm.py:106` vs `mla.py:132`). This is a real
carve-out from "CUDA graphs from the start": MHA gets graphs in M3, MLA runs
eager until the fork grows a CG constructor. Force `--disable-cuda-graph` (or
skip graph capture for MLA layers) and say so in the error message.

Two further MLA gaps: `run(return_lse=True)` raises `NotImplementedError`
(`mla_rocm.py:255`), so **DCP is unsupported**; and `plan()` does
`qo_lens.max().item()` (`:216`), a host sync every step — a known M7 target.

**Set expectations:** because the fork's MLA *is* AITER underneath, this
milestone validates wrapper plumbing and metadata correctness, not kernel
performance. Expect parity with the AITER backend, not a win.

> **As built — the plan was wrong here in two ways:**
>
> 1. MLA needs **two** wrappers, not one. Decode uses
>    `BatchMLAPagedAttentionWrapper` over the compressed latent cache; prefill
>    uses the **ragged** wrapper over uncompressed q/k/v at 192/128, because
>    absorption does not pay off during prefill and SGLang's DeepSeek model
>    hands this backend full q/k/v. My first implementation used the MLA wrapper
>    for both and failed in the kernel with
>    `mla_prefill_asm_fwd: only support head_size == KV.size(3)`.
> 2. **MLA prefill turns out to be impossible on this port** — see the Status
>    section. It is rejected at startup with a pointer to the hybrid config.
>
> `lmsys/sglang-ci-dsv3-test` is gated; `deepseek-ai/DeepSeek-V2-Lite` (MLA,
> bf16, ungated, ~30 GB) is the working substitute.

**Verify:** MLA decode runs under
`--prefill-attention-backend aiter --decode-attention-backend flashinfer_rocm`
on DeepSeek-V2-Lite. Greedy-output match against `aiter` is still outstanding.

## M5 — Tests

Add `test/registered/attention/test_flashinfer_rocm_attention_backend.py`,
copying the shape of `test/registered/attention/test_triton_attention_backend.py`:
`register_amd_ci(...)` only (no `register_cuda_ci`), a `test_latency` using
`run_bench_offline_throughput`, and a `test_mmlu` using `popen_launch_server` +
`run_eval`.

Add a backend-equivalence test built on
`sglang.test.server_fixtures.hybrid_attn_backend_fixture` (the existing
cross-backend comparison harness with an `accuracy_threshold`) comparing
`flashinfer_rocm` against `aiter`.

Per [`unit-test-admission`](../.claude/rules/unit-test-admission.md), each case must
guard a distinct failure mode — the capture-at-max-seq-len rule and the
"auto silently reroutes to AITER" trap are the two worth pinning explicitly.

Register in the AMD CI suites used by `.github/workflows/pr-test-amd.yml`
(`stage-b-test-1-gpu-small-amd`), gated so it skips when `amd-flashinfer` is absent.

**Verify:** `cd test && python3 run_suite.py --hw amd --suite stage-b-test-1-gpu-small-amd`.

## M6 — Benchmark against AITER

Same container, same model, same seeds; only `--attention-backend` changes across
`flashinfer_rocm` (`fa2`), `flashinfer_rocm` (`aiter` via env), and `aiter`.

- Kernel/step latency: `python3 -m sglang.benchmark.one_batch --model-path ... --batch-size 1 8 32 --input-len 128 1024 4096 --output-len 8 --run-name ...`
- Offline throughput: `python3 -m sglang.benchmark.offline_throughput --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 500`
- Online serving: `python3 -m sglang.benchmark.serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024`

Sweep `--page-size` 1 / 128 / 256 — AITER CK FMHA prefers `{128, 256, 1024}`, so
`page_size=1` likely understates the AITER-route numbers. Lock GPU clocks
(`amd-smi`) before timing runs.

Record TTFT, TPOT, and output throughput into a small results table checked in
under `benchmark/` so regressions are visible over time.

## M7 — Optimization

**Status: the in-scope part is done.** See "Where the time goes" above for the
measurement that reorganised this list — 78% of the decode gap is kernel work in
the ROCm fork, not anything reachable from SGLang.

Done:

- **Host-resident `plan()` inputs.** `plan()` did `indptr.to("cpu")` and
  `last_page_len.to("cpu")` unconditionally; it now gets pinned host mirrors plus
  the scheduler's `seq_lens_cpu`. Measured 0.190 → 0.146 ms/step of metadata,
  output unchanged.

Handed off to AMD-Ecosystem/flashinfer (`HANDOFF-head-dim-192.md`):

- **Decode kernel**, ~0.39 ms/step of the 0.50 ms gap. `fa2` decode `run()` is
  0.0405 ms/call; AITER would need ~1.78× per call to close it.

Still open here, in rough value order:

1. **Prefill.** Never investigated — all M7 work was decode. Prefill is 5–10%
   behind aiter, and two hypotheses are untested: `page_size > 1`, and the
   ragged + paged + `merge_state` split skipped in M2. Prefill dominates at long
   context, so this plausibly outranks the decode remainder.
2. **Remaining metadata**, 0.146 ms vs aiter's 0.074 ms — the kv-indices build.
   Low value against a 7–11% total gap at 7B.
3. **MLA `plan()` host sync** — `qo_lens.max().item()` (`mla_rocm.py:216`) every
   step. Only matters if MLA becomes usable, which needs the head_dim-192 fix
   *and* a CUDA-graph-capable MLA wrapper upstream.
4. **fp8 KV cache** — supported on the HIP decode path (E4M3FNUZ); unmeasured.

## M8 — Docs and environment maintenance

- Add `flashinfer_rocm` rows to the MHA and MLA matrices in
  `docs_new/docs/advanced_features/attention_backend.mdx` (which already has
  AITER and Wave rows), documenting the bf16/page_size/CUDA-graph caveats.
- Note the container recipe in `docs_new/docs/hardware-platforms/amd_gpu.mdx`.
- Pin `amd-flashinfer` and `amd_aiter` versions in one place in the Dockerfile,
  mirroring how `AITER_COMMIT_DEFAULT` is pinned in `docker/rocm.Dockerfile` and
  re-checked by `scripts/ci/amd/amd_ci_install_dependency.sh`.
- Because the AMD fork tracks upstream v0.5.3 while SGLang pins 0.6.15, add a
  short compatibility note recording which fork release the backend was written
  against, so the next bump has a starting point.

---

## API compatibility: fork `0.5.3+amd.2` vs SGLang's NVIDIA backend

Measured directly against `~/devel/flashinfer` @ `cd6b40d2`. This confirms the
separate-backend decision and pins down exactly what the new backend may call.

**Import-time blockers** (why `flashinfer_backend.py` cannot simply be reused):

- `from flashinfer import fast_decode_plan` — **missing** under `IS_HIP`. Defined
  only in the CUDA `flashinfer/decode.py`. Hard `ImportError`.
- `is_flashinfer_available()` is False on ROCm (`is_cuda()` → `torch.version.cuda`),
  so the entire import block is skipped and every wrapper name is undefined.

**Drop-in as-is** (identical signatures — the good news):

| API | Fork location |
| --- | --- |
| `BatchDecodeWithPagedKVCacheWrapper.__init__` incl. all `use_cuda_graph` buffer kwargs | `decode_rocm.py:795` |
| `...decode.plan()` incl. `data_type`, `q_data_type`, `non_blocking` | `decode_rocm.py:973` |
| `...decode.forward()` incl. `sm_scale`, `logits_soft_cap`, `k_scale`, `v_scale` | `decode_rocm.py:1390` |
| `BatchPrefillWithPagedKVCacheWrapper.__init__` incl. all CG buffer kwargs | `prefill_rocm.py:1536` |
| `...paged prefill .forward()` / `.forward_return_lse()` | `prefill_rocm.py:2169`, `:2461` |
| `BatchPrefillWithRaggedKVCacheWrapper.__init__` / `.plan()` | `prefill_rocm.py:2597`, `:2745` |
| `BatchMLAPagedAttentionWrapper.plan()` (same 12 positional args) | `mla_rocm.py:127` |

**Differences the new backend must design around:**

1. **Ragged prefill has no `forward` / `forward_return_lse`** — only `run()` and
   `run_return_lse()`. Call those instead.
2. **Ragged prefill takes `window_left` / `sm_scale` / `logits_soft_cap` at
   *plan* time, not per call.** SGLang's NVIDIA backend passes them per call.
   The indices-updater must move them into `plan()`.
3. **Paged prefill `plan()` has no `fixed_split_size` / `disable_split_kv`** —
   simply don't pass them (SGLang passes `fixed_split_size` unconditionally).
4. **`fixed_split_size` / `disable_split_kv` are accepted but no-ops on ROCm
   *decode*** (`decode_rocm.py:1044`). Deterministic-inference split control
   silently does nothing — do not advertise determinism guarantees.
5. **`prefix_len_ptr` / `token_pos_in_items_ptr` are accepted by `plan()` but
   ignored in `run()`** (`prefill_rocm.py:2291`). Multi-item scoring would be
   silently wrong — reject it explicitly rather than let it run.
6. **`merge_state` exists and is exported under HIP** (`cascade.py:40`), but the
   HIP cascade path is partly behind `FLASHINFER_HIP_FUSED_CASCADE=1`. SGLang
   already has a Triton fallback (`_safe_merge_state`, `flashinfer_backend.py:97`)
   — reuse it and smoke-test the HIP path before trusting it.

**`backend=` values accepted on ROCm:** decode and both prefill wrappers take
`"auto"` / `"fa2"` / `"aiter"`. MLA takes only `"auto"` / `"aiter"` — passing
`"fa2"` raises. Prefill silently downgrades unknown values (e.g. `"cutlass"`) to
`"fa2"` with a warning.

## Verification summary

| Milestone | Check |
| --- | --- |
| M0 | `import flashinfer` in container; fork's own `tests/rocm_tests` pass |
| M1 | `--attention-backend flashinfer_rocm` dispatches to the new factory |
| M2 | Greedy-token match vs `aiter`, eager; MMLU ≥ 0.65 on Llama-3.1-8B |
| M3 | Same, with CUDA graphs (MHA only); short-replay-after-max-capture correctness |
| M4 | Greedy-token match vs `aiter` on an MLA model, **eager only** |
| M5 | `run_suite.py --hw amd --suite stage-b-test-1-gpu-small-amd` green |
| M6 | Benchmark table: `flashinfer_rocm` vs `aiter`, TTFT/TPOT/throughput |
