from __future__ import annotations

"""Attention backend built on the ROCm port of FlashInfer.

Targets AMD-Ecosystem/flashinfer (the ``amd-flashinfer`` wheel), which tracks
upstream FlashInfer v0.5.3 and ships HIP kernels plus optional AITER dispatch
for gfx942 / gfx950.

Deliberately separate from ``flashinfer_backend.py`` rather than sharing it:
SGLang pins ``flashinfer_python==0.6.15`` on NVIDIA, and the two wheels differ
enough that the NVIDIA backend cannot import here at all (``fast_decode_plan``
is not exported under HIP). Keeping them apart also means the NVIDIA path takes
no risk from ROCm changes.

Scope of this backend, and why each limit exists:

  * ``page_size`` must be 1. This is a limitation of *this backend*, not of the
    ROCm kernels: the fa2 MHA path runs prefill and decode correctly at page
    sizes 1, 16, 64 and 128 (verified on gfx942). The restriction is here
    because MLA genuinely requires page_size=1, and because it makes the
    paged-KV view a free reshape — the pool hands over ``[num_tokens, H, D]``,
    which is already ``[num_pages, 1, H, D]``. Lifting it for MHA needs
    page-granular kv-indices, and is the first prefill optimization to try:
    AITER's CK FMHA prefers ``{128, 256, 1024}``, so page_size=1 forces a
    gather on that path.
  * No sliding-window models — the ROCm wrappers take ``window_left`` at plan
    time, so per-layer windows would need one wrapper per distinct window (what
    the NVIDIA backend's ``WrapperDispatch`` does). Rejected rather than
    silently applying the wrong window.
  * No cross-attention / encoder-only, no multi-item scoring, no speculative
    decoding. Multi-item scoring in particular is rejected rather than ignored:
    the ROCm ``plan()`` accepts ``prefix_len_ptr`` / ``token_pos_in_items_ptr``
    and then drops them in ``run()``, which would silently produce wrong output.
  * MLA is eager-only — ``mla_rocm.BatchMLAPagedAttentionWrapper`` has no
    ``use_cuda_graph`` support at all.
  * Prefill CUDA graphs are not implemented. Note this costs nothing today:
    ROCm resolves ``prefill`` to ``disabled`` for *every* attention backend
    (aiter and triton included), so the explicit disable in
    ``_validate_flashinfer_rocm_backend`` is redundant rather than a handicap.
"""

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_flashinfer_rocm_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

if is_flashinfer_rocm_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchMLAPagedAttentionWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )

# 128 MB, same order as the NVIDIA backend's default workspace. The ROCm plan
# path stores its scheduling metadata here; too small surfaces as a plan-time
# failure rather than a silent slowdown.
_WORKSPACE_BYTES = 128 * 1024 * 1024

# Values the ROCm wrappers accept for `backend=`. "fa2" is the in-tree HIP
# kernel, "aiter" the AITER dispatch, "auto" picks AITER when its compatibility
# conditions hold. We default to "fa2" rather than "auto" for two reasons:
# benchmarking against SGLang's aiter backend must not silently compare AITER
# with itself, and "auto" drops kwargs (alibi, sinks, fp8 scales) without error.
_VALID_BACKENDS = ("fa2", "aiter", "auto")
_DEFAULT_BACKEND = "fa2"

# GQA ratios the HIP fa2 decode kernel instantiates
# (include/gpu_iface/dispatch.cuh). Anything else reaches the kernel and aborts
# with "Unsupported group_size: N" partway through the first decode, so the
# backend checks it at construction instead. Llama-3.x (32/8=4) and
# TinyLlama (32/4=8) are fine; Qwen2.5-0.5B (14/2=7) and Qwen2.5-7B (28/4=7)
# are not.
_SUPPORTED_DECODE_GROUP_SIZES = (1, 2, 3, 4, 8)


def _resolve_flashinfer_backend() -> str:
    """Read the `backend=` string to hand the ROCm wrappers."""
    requested = envs.SGLANG_FLASHINFER_ROCM_BACKEND.get()
    if requested not in _VALID_BACKENDS:
        raise ValueError(
            f"SGLANG_FLASHINFER_ROCM_BACKEND={requested!r} is not one of "
            f"{_VALID_BACKENDS}."
        )
    return requested


class FlashInferRocmMetadata(msgspec.Struct, omit_defaults=True):
    """Per-forward index tensors handed to the ROCm wrappers' ``plan()``."""

    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    qo_indptr: Optional[torch.Tensor] = None


def _reject_unsupported_layer(layer: RadixAttention) -> None:
    """Fail loudly on layer shapes this backend does not implement.

    Checked per layer rather than once at init because sliding window and
    attention type are per-layer properties.
    """
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        raise NotImplementedError(
            "flashinfer_rocm does not support sliding-window attention: the ROCm "
            "wrappers take window_left at plan time, so a per-layer window would "
            "need one wrapper per window size. Use --attention-backend aiter or "
            "triton for this model."
        )
    if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
        raise NotImplementedError(
            "flashinfer_rocm does not support cross-attention or encoder-only "
            "layers yet. Use --attention-backend aiter or triton for this model."
        )


class FlashInferRocmAttnBackend(AttentionBackend):
    """MHA / GQA attention on ROCm FlashInfer.

    Prefill always goes through the paged wrapper (KV is written to the pool
    first, then attended over as one causal block). The NVIDIA backend splits
    this into a ragged pass over new tokens plus a paged pass over the prefix,
    merged with ``merge_state``; that is a throughput optimization, not a
    correctness requirement, and it is deferred until the paged path is proven.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        if not is_flashinfer_rocm_available():
            raise ImportError(
                "flashinfer_rocm requires the amd-flashinfer package. Install "
                "it with: pip install amd-flashinfer --extra-index-url "
                "https://pypi.amd.com/rocm-7.2.0/simple/"
            )

        self.model_runner = model_runner
        self.device = model_runner.device
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.page_size = model_runner.page_size
        if self.page_size != 1:
            raise NotImplementedError(
                f"flashinfer_rocm requires --page-size 1 (got {self.page_size})."
            )

        attn_tp_size = get_parallel().attn_tp_size
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(attn_tp_size)
        self.head_dim = model_runner.model_config.head_dim

        group_size = self.num_qo_heads // self.num_kv_heads
        if group_size not in _SUPPORTED_DECODE_GROUP_SIZES:
            raise NotImplementedError(
                f"flashinfer_rocm cannot serve this model: its GQA group size is "
                f"{group_size} ({self.num_qo_heads} query heads / "
                f"{self.num_kv_heads} KV heads at attn_tp_size={attn_tp_size}), but "
                f"the ROCm decode kernel only instantiates "
                f"{_SUPPORTED_DECODE_GROUP_SIZES}. Use --attention-backend aiter "
                f"or triton for this model."
            )
        self.q_data_type = model_runner.dtype
        self.kv_data_type = model_runner.kv_cache_dtype

        self.fi_backend = _resolve_flashinfer_backend()
        logger.info(
            "flashinfer_rocm: using FlashInfer backend=%r (set "
            "SGLANG_FLASHINFER_ROCM_BACKEND to override)",
            self.fi_backend,
        )

        self.workspace_buffer = torch.empty(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )

        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=self.fi_backend
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend=self.fi_backend,
            use_tensor_cores=False,
        )

        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.qo_indptr = torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device)
        # page_size == 1, so every page is exactly full.
        self.kv_last_page_len = torch.ones(
            max_bs, dtype=torch.int32, device=self.device
        )

        # Host mirrors handed to plan() instead of the device tensors. plan()
        # does `indptr.to("cpu")` and `last_page_len.to("cpu")` unconditionally
        # -- blocking device-to-host copies that stall on whatever is still in
        # flight. Passing tensors that already live on the host turns both into
        # no-ops. Measured at ~0.039 ms/step. Pinned so the paired host-to-device
        # copy plan() makes into the graph buffers can be async.
        self.kv_indptr_host = torch.zeros(
            max_bs + 1, dtype=torch.int32, device="cpu"
        ).pin_memory()
        self.kv_last_page_len_host = torch.ones(
            max_bs, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # forward_decode reads this; eager and CUDA-graph planning each point it
        # at their own wrapper. Never assign self.decode_wrapper itself -- the
        # eager wrapper has no graph buffers and must stay reusable.
        self._active_decode_wrapper = self.decode_wrapper

        self.max_context_len = model_runner.model_config.context_len
        self.forward_metadata: Optional[FlashInferRocmMetadata] = None

        # CUDA-graph state, populated by init_cuda_graph_state.
        self.cuda_graph_kv_indices: Optional[torch.Tensor] = None
        self.decode_cuda_graph_wrappers: dict[int, object] = {}

    def _build_kv_indices(
        self,
        *,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        seq_lens_sum: int,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        out_kv_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Flatten req_to_token into the (kv_indptr, kv_indices) pair plan() wants.

        Also returns a host copy of kv_indptr when the scheduler supplied CPU
        seq lens, so plan() can be given host-resident inputs and skip its
        device-to-host copies. None when unavailable -- callers fall back to the
        device tensor, which is correct, just slower.
        """
        bs = req_pool_indices.shape[0]
        kv_indptr = self.kv_indptr
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]

        kv_indptr_host = None
        if seq_lens_cpu is not None:
            # Same cumsum, host side. Cheap at these sizes and it costs no sync,
            # whereas copying the device result back would.
            host = self.kv_indptr_host
            torch.cumsum(seq_lens_cpu, dim=0, out=host[1 : bs + 1])
            kv_indptr_host = host[: bs + 1]

        if out_kv_indices is None:
            out_kv_indices = torch.empty(
                seq_lens_sum, dtype=torch.int32, device=self.device
            )

        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            out_kv_indices,
            self.req_to_token.shape[1],
        )
        return kv_indptr, out_kv_indices, kv_indptr_host

    def _plan_decode_wrapper(
        self,
        wrapper,
        *,
        bs: int,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_indptr_host: Optional[torch.Tensor],
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        """Call plan(), preferring host-resident inputs.

        plan() copies indptr and last_page_len to the host unconditionally and
        derives kv lens from them. Handing it tensors already on the host makes
        those copies free; passing seq_lens skips the derivation too. Falls back
        to the device tensors when the scheduler gave us no CPU seq lens.
        """
        use_host = kv_indptr_host is not None
        wrapper.plan(
            kv_indptr_host if use_host else kv_indptr,
            kv_indices,
            self.kv_last_page_len_host[:bs] if use_host else self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            seq_lens=seq_lens_cpu if use_host else None,
            q_data_type=self.q_data_type,
            kv_data_type=self.kv_data_type,
            non_blocking=True,
        )

    def _plan_decode(self, forward_batch: ForwardBatch) -> FlashInferRocmMetadata:
        bs = forward_batch.batch_size
        kv_indptr, kv_indices, kv_indptr_host = self._build_kv_indices(
            req_pool_indices=forward_batch.req_pool_indices,
            paged_kernel_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )
        self._active_decode_wrapper = self.decode_wrapper
        self._plan_decode_wrapper(
            self.decode_wrapper,
            bs=bs,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_indptr_host=kv_indptr_host,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )
        return FlashInferRocmMetadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=self.kv_last_page_len[:bs],
        )

    def _plan_extend(self, forward_batch: ForwardBatch) -> FlashInferRocmMetadata:
        bs = forward_batch.batch_size
        kv_indptr, kv_indices, _ = self._build_kv_indices(
            req_pool_indices=forward_batch.req_pool_indices,
            paged_kernel_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
        )

        qo_indptr = self.qo_indptr
        qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        self.prefill_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            causal=True,
            q_data_type=self.q_data_type,
            kv_data_type=self.kv_data_type,
            non_blocking=True,
        )
        return FlashInferRocmMetadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=self.kv_last_page_len[:bs],
            qo_indptr=qo_indptr,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.spec_info is not None:
            raise NotImplementedError(
                "flashinfer_rocm does not support speculative decoding yet."
            )

        if forward_batch.forward_mode.is_decode():
            self.forward_metadata = self._plan_decode(forward_batch)
        elif forward_batch.forward_mode.is_idle():
            self.forward_metadata = None
        else:
            self.forward_metadata = self._plan_extend(forward_batch)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---- CUDA graphs (decode only) ----------------------------------------
    #
    # init_forward_metadata above is an independent eager body, so the three
    # methods below are reached only from the graph runner: at capture with
    # in_capture=True, and before each replay with in_capture=False.
    #
    # Prefill graphs are not supported; _validate_flashinfer_rocm_backend
    # disables them up front, and _require_decode_mode re-checks here so a
    # future default flip fails loudly instead of capturing a wrong graph.

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # page_size == 1, so one KV index per token of context per sequence.
        self.cuda_graph_kv_indices = torch.zeros(
            max_num_tokens * self.max_context_len, dtype=torch.int32, device=self.device
        )

    def _decode_cuda_graph_wrapper(self, bs: int):
        """One graph-enabled decode wrapper per batch size, created on demand.

        The buffers handed in here are the ones the captured graph will read
        from forever, so they must be the same tensors every replay.
        """
        wrapper = self.decode_cuda_graph_wrappers.get(bs)
        if wrapper is not None:
            return wrapper

        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend=self.fi_backend,
            use_cuda_graph=True,
            use_tensor_cores=False,
            paged_kv_indptr_buffer=self.kv_indptr[: bs + 1],
            paged_kv_indices_buffer=self.cuda_graph_kv_indices,
            paged_kv_last_page_len_buffer=self.kv_last_page_len[:bs],
        )
        self.decode_cuda_graph_wrappers[bs] = wrapper
        return wrapper

    def _require_decode_mode(self, forward_batch: ForwardBatch) -> None:
        if not forward_batch.forward_mode.is_decode():
            raise NotImplementedError(
                "flashinfer_rocm supports CUDA graphs for decode only, but graph "
                f"capture was requested for {forward_batch.forward_mode}. Re-run "
                "with --cuda-graph-backend-prefill=disabled."
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        self._require_decode_mode(forward_batch)
        if forward_batch.spec_info is not None:
            raise NotImplementedError(
                "flashinfer_rocm does not support speculative decoding yet."
            )

        bs = forward_batch.batch_size
        wrapper = self._decode_cuda_graph_wrapper(bs)

        # Write straight into the graph-visible indices buffer. Capture and
        # replay are identical here: plan() refreshes the wrapper's scheduling
        # metadata, and run() inside the graph reads these same addresses.
        kv_indptr, kv_indices, kv_indptr_host = self._build_kv_indices(
            req_pool_indices=forward_batch.req_pool_indices,
            paged_kernel_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            out_kv_indices=self.cuda_graph_kv_indices,
        )
        self._plan_decode_wrapper(
            wrapper,
            bs=bs,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_indptr_host=kv_indptr_host,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )
        self._active_decode_wrapper = wrapper
        self.forward_metadata = FlashInferRocmMetadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=self.kv_last_page_len[:bs],
        )

    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        _reject_unsupported_layer(layer)

        if k is not None and save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(forward_batch.out_cache_loc),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        o = self.prefill_wrapper.run(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    @debug_kernel_api
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        _reject_unsupported_layer(layer)

        if k is not None and save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(forward_batch.out_cache_loc),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        o = self._active_decode_wrapper.run(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)



class FlashInferRocmMlaAttnBackend(AttentionBackend):
    """MLA (DeepSeek-style) attention on ROCm FlashInfer.

    Two different wrappers, because MLA prefill and MLA decode are different
    problems:

      * **Decode** uses ``BatchMLAPagedAttentionWrapper`` over the compressed
        latent cache (``kv_lora_rank`` + ``qk_rope_head_dim``). ``mla_rocm``
        dispatches this to AITER unconditionally -- there is no HIP MLA kernel --
        so it is bf16 + ``page_size=1`` only, and expect parity with SGLang's
        aiter backend rather than a speedup.
      * **Prefill** uses the ragged prefill wrapper over uncompressed q/k/v with
        ``head_dim_qk=192`` and ``head_dim_vo=128``. Absorption does not pay off
        during prefill, so SGLang's DeepSeek model hands this backend full q/k/v
        (``flashinfer_rocm`` is deliberately absent from
        ``FORWARD_ABSORB_CORE_ATTENTION_BACKENDS``). Note ``head_dim_qk !=
        head_dim_vo`` is one of the documented AITER-incompatibility conditions,
        so this path always runs the in-tree HIP ``fa2`` kernel.

    Eager only: the ROCm ``BatchMLAPagedAttentionWrapper`` takes no
    ``use_cuda_graph`` kwarg at all, and ``run(return_lse=True)`` raises, so DCP
    is unsupported too. ``_validate_flashinfer_rocm_backend`` disables graph
    capture up front.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        if not is_flashinfer_rocm_available():
            raise ImportError(
                "flashinfer_rocm requires the amd-flashinfer package. Install "
                "it with: pip install amd-flashinfer --extra-index-url "
                "https://pypi.amd.com/rocm-7.2.0/simple/"
            )

        self.model_runner = model_runner
        self.device = model_runner.device
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.page_size = model_runner.page_size
        if self.page_size != 1:
            raise NotImplementedError(
                "flashinfer_rocm MLA requires --page-size 1 "
                f"(got {self.page_size}); the ROCm MLA kernel is page_size=1 only."
            )
        if model_runner.dtype != torch.bfloat16:
            raise NotImplementedError(
                "flashinfer_rocm MLA requires bfloat16 "
                f"(got {model_runner.dtype}); the ROCm MLA kernel is bf16 only."
            )

        config = model_runner.model_config
        self.num_local_heads = config.num_attention_heads // get_parallel().attn_tp_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.scaling = config.scaling
        self.q_data_type = model_runner.dtype
        self.kv_data_type = model_runner.kv_cache_dtype

        self.workspace_buffer = torch.empty(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )
        # backend="auto" resolves to "aiter" for MLA; "fa2" would raise.
        self.decode_wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, backend="auto"
        )
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=_resolve_flashinfer_backend()
        )

        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device)
        self.qo_indptr = torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device)

        self.forward_metadata: Optional[FlashInferRocmMetadata] = None

    def _plan_ragged_prefill(self, forward_batch: ForwardBatch) -> None:
        """Plan uncompressed q/k/v prefill over the new tokens only.

        Prefix reuse would need the paged MLA prefill path, which is a separate
        implementation; reject it rather than silently attend over new tokens
        alone and return wrong output for cached requests.
        """
        if any(forward_batch.extend_prefix_lens_cpu):
            raise NotImplementedError(
                "flashinfer_rocm MLA does not implement prefix-cached prefill "
                "yet (it would need the paged MLA prefill path). Re-run with "
                "--disable-radix-cache, or use --attention-backend aiter."
            )

        bs = forward_batch.batch_size
        qo_indptr = self.qo_indptr
        qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        config = self.model_runner.model_config
        head_dim_qk = config.qk_nope_head_dim + config.qk_rope_head_dim
        # ROCm takes causal / sm_scale / logits_soft_cap at plan time; the
        # NVIDIA wrapper takes them per call.
        self.prefill_wrapper_ragged.plan(
            qo_indptr,
            qo_indptr,
            self.num_local_heads,
            self.num_local_heads,
            head_dim_qk,
            head_dim_vo=config.v_head_dim,
            causal=True,
            sm_scale=self.scaling,
            q_data_type=self.q_data_type,
            kv_data_type=self.q_data_type,
        )
        self.forward_metadata = FlashInferRocmMetadata(
            kv_indptr=qo_indptr,
            kv_indices=qo_indptr,
            kv_last_page_len=qo_indptr,
            qo_indptr=qo_indptr,
        )

    def _plan_mla_decode(self, forward_batch: ForwardBatch) -> None:
        bs = forward_batch.batch_size
        kv_lens = forward_batch.seq_lens.to(torch.int32)

        kv_indptr = self.kv_indptr
        kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]

        kv_indices = torch.empty(
            forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.shape[1],
        )

        # One query token per sequence during decode.
        qo_indptr = self.qo_indptr
        qo_indptr[: bs + 1] = torch.arange(
            bs + 1, dtype=torch.int32, device=self.device
        )
        qo_indptr = qo_indptr[: bs + 1]

        self.decode_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            self.num_local_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            False,  # causal: a single query token attends over the whole prefix
            self.scaling,
            self.q_data_type,
            self.kv_data_type,
        )
        self.forward_metadata = FlashInferRocmMetadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_lens,
            qo_indptr=qo_indptr,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.spec_info is not None:
            raise NotImplementedError(
                "flashinfer_rocm MLA does not support speculative decoding yet."
            )
        if forward_batch.forward_mode.is_idle():
            self.forward_metadata = None
        elif forward_batch.forward_mode.is_decode():
            self._plan_mla_decode(forward_batch)
        else:
            self._plan_ragged_prefill(forward_batch)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _save_kv(self, layer, forward_batch, k, v, k_rope):
        if k is None:
            return
        if k_rope is not None:
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(forward_batch.out_cache_loc), k, v
            )

    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if save_kv_cache:
            self._save_kv(layer, forward_batch, k, v, k_rope)

        # Absorbed layouts arrive split; stitch them back into full head_dim.
        if q_rope is not None:
            q = torch.cat(
                [
                    q.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    ),
                ],
                dim=-1,
            )
        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)

        o = self.prefill_wrapper_ragged.run(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
            v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    @debug_kernel_api
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if save_kv_cache:
            self._save_kv(layer, forward_batch, k, v, k_rope)

        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshaped = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshaped[:, :, : layer.v_head_dim]
            q_rope = reshaped[:, :, layer.v_head_dim :]

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
        o = self.decode_wrapper.run(
            q_nope,
            q_rope,
            k_buffer[:, :, : layer.v_head_dim],
            k_buffer[:, :, layer.v_head_dim :],
            out=q_nope.new_empty(q_nope.shape),
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
