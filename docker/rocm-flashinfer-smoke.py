"""Smoke-test the ROCm FlashInfer install inside the container.

    python3 docker/rocm-flashinfer-smoke.py

Checks the three things that have actually broken here, in order of how quietly
they fail:

  1. flashinfer imports, torch is a ROCm build, GPU is gfx942/gfx950.
  2. BatchMLAPagedAttentionWrapper is exported. The published 0.5.3+amd.1 wheel
     does not export it; MLA needs >= 0.5.3+amd.2.
  3. Causal paged prefill matches a torch SDPA reference. 0.5.3+amd.1 returns
     wrong values on *some* rows for causal prefill on gfx942 while non-causal
     and decode are exact — so an import check alone will not catch it.

Exits non-zero on the first failure.
"""

import sys

import torch
import torch.nn.functional as F

SUPPORTED_ARCHS = ("gfx942", "gfx950")
TOLERANCE = 0.02


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_environment() -> None:
    import flashinfer

    if not torch.version.hip:
        _fail(f"expected a ROCm torch build, got {torch.__version__}")

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in SUPPORTED_ARCHS:
        _fail(f"GPU is {arch}; ROCm flashinfer supports {SUPPORTED_ARCHS}")

    print(
        f"PASS environment: flashinfer {flashinfer.__version__} | "
        f"torch {torch.__version__} | {arch}"
    )


def check_mla_exported() -> None:
    import flashinfer

    if not hasattr(flashinfer, "BatchMLAPagedAttentionWrapper"):
        _fail(
            "BatchMLAPagedAttentionWrapper is not exported. The published "
            "0.5.3+amd.1 wheel predates MLA support; rebuild the image with "
            "FLASHINFER_MODE=source (the default) to get >= 0.5.3+amd.2."
        )
    print("PASS MLA wrapper exported")


def _reference(q, k, v, *, causal: bool):
    """Torch SDPA reference for [L, H, D] inputs with GQA expansion."""
    repeat = q.shape[1] // k.shape[1]
    to_bhld = lambda t: t.transpose(0, 1).unsqueeze(0).float()
    out = F.scaled_dot_product_attention(
        to_bhld(q),
        to_bhld(k.repeat_interleave(repeat, dim=1)),
        to_bhld(v.repeat_interleave(repeat, dim=1)),
        is_causal=causal,
    )
    return out.squeeze(0).transpose(0, 1)


def check_causal_prefill() -> None:
    """The 0.5.3+amd.1 regression: right shape, wrong values on some rows."""
    import flashinfer

    device, dtype = "cuda", torch.bfloat16
    num_qo, num_kv, head_dim, seq = 32, 8, 128, 64
    torch.manual_seed(0)

    k = torch.randn(seq, num_kv, head_dim, dtype=dtype, device=device)
    v = torch.randn(seq, num_kv, head_dim, dtype=dtype, device=device)
    q = torch.randn(seq, num_qo, head_dim, dtype=dtype, device=device)

    # SGLang's pool hands over [num_tokens, H, D] per side; page_size=1 makes
    # that exactly [num_pages, page_size, H, D].
    kv_cache = (k.view(seq, 1, num_kv, head_dim), v.view(seq, 1, num_kv, head_dim))
    indptr = torch.tensor([0, seq], dtype=torch.int32, device=device)
    indices = torch.arange(seq, dtype=torch.int32, device=device)
    last_page_len = torch.ones(1, dtype=torch.int32, device=device)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        "NHD",
        backend="fa2",
    )
    wrapper.plan(
        indptr,
        indptr,
        indices,
        last_page_len,
        num_qo,
        num_kv,
        head_dim,
        1,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        non_blocking=True,
    )
    error = (
        (wrapper.run(q, kv_cache).float() - _reference(q, k, v, causal=True))
        .abs()
        .max()
        .item()
    )
    if error > TOLERANCE:
        _fail(
            f"causal paged prefill disagrees with SDPA (max abs err {error:.4f} "
            f"> {TOLERANCE}). Known-broken on 0.5.3+amd.1; needs >= 0.5.3+amd.2."
        )
    print(f"PASS causal prefill vs SDPA (max abs err {error:.4f})")


if __name__ == "__main__":
    check_environment()
    check_mla_exported()
    check_causal_prefill()
    print("\nAll smoke checks passed.")
