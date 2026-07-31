"""Kernel-level correctness for the ROCm FlashInfer wrappers.

Usage:
python3 -m unittest test_flashinfer_rocm_attention_kernels.TestFlashInferRocmKernels

Regression guard. The published amd-flashinfer 0.5.3+amd.1 wheel returns wrong
values from *causal* paged prefill on gfx942 while decode and non-causal prefill
are exact, and while the output stays right-shaped and finite. It was fixed in
0.5.3+amd.2 ("MFMA C/D row indexing for custom masks with GQA"). An import or
shape check does not catch that class of failure, so compare against a torch
SDPA reference and pin the (k, v) tuple layout that flashinfer_rocm_backend
hands the wrappers at the same time.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_flashinfer_rocm_available
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")

_WORKSPACE_BYTES = 128 * 1024 * 1024
# bf16 attention against an fp32 SDPA reference; the observed error is ~1e-3,
# while the amd.1 regression produced 2.4-4.1.
_TOLERANCE = 0.02


def _sdpa_reference(q, k, v, *, causal: bool) -> torch.Tensor:
    """Reference for [L, H, D] inputs, expanding KV heads for GQA."""
    repeat = q.shape[1] // k.shape[1]
    to_bhld = lambda t: t.transpose(0, 1).unsqueeze(0).float()
    out = F.scaled_dot_product_attention(
        to_bhld(q),
        to_bhld(k.repeat_interleave(repeat, dim=1)),
        to_bhld(v.repeat_interleave(repeat, dim=1)),
        is_causal=causal,
    )
    return out.squeeze(0).transpose(0, 1)


@unittest.skipUnless(
    is_flashinfer_rocm_available(),
    "requires the amd-flashinfer package on a ROCm GPU",
)
class TestFlashInferRocmKernels(CustomTestCase):
    seq_len = 64
    head_dim = 128

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        cls.dtype = torch.bfloat16
        cls.workspace = torch.empty(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=cls.device
        )

    def _make_inputs(self, num_qo_heads: int, num_kv_heads: int):
        """Build one sequence plus the paged views the backend passes through.

        SGLang's KV pool exposes [num_tokens, H, D] per side. With page_size=1
        that is exactly [num_pages, page_size, H, D], which is the layout
        flashinfer_rocm_backend relies on.
        """
        torch.manual_seed(0)
        shape = (self.seq_len, num_kv_heads, self.head_dim)
        k = torch.randn(shape, dtype=self.dtype, device=self.device)
        v = torch.randn(shape, dtype=self.dtype, device=self.device)
        q = torch.randn(
            self.seq_len, num_qo_heads, self.head_dim, dtype=self.dtype, device=self.device
        )
        paged = (
            k.view(self.seq_len, 1, num_kv_heads, self.head_dim),
            v.view(self.seq_len, 1, num_kv_heads, self.head_dim),
        )
        indptr = torch.tensor([0, self.seq_len], dtype=torch.int32, device=self.device)
        indices = torch.arange(self.seq_len, dtype=torch.int32, device=self.device)
        last_page_len = torch.ones(1, dtype=torch.int32, device=self.device)
        return q, k, v, paged, indptr, indices, last_page_len

    def _run_prefill(self, *, num_qo_heads, num_kv_heads, causal):
        import flashinfer

        q, k, v, paged, indptr, indices, last_page_len = self._make_inputs(
            num_qo_heads, num_kv_heads
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            indptr,
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            self.head_dim,
            1,
            causal=causal,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        actual = wrapper.run(q, paged).float()
        return (actual - _sdpa_reference(q, k, v, causal=causal)).abs().max().item()

    def test_causal_paged_prefill_matches_sdpa(self):
        """The amd.1 regression: some output rows exact, others badly wrong."""
        for num_qo_heads, num_kv_heads in ((4, 4), (32, 8)):
            with self.subTest(num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads):
                error = self._run_prefill(
                    num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, causal=True
                )
                self.assertLess(error, _TOLERANCE, f"max abs error {error:.4f}")

    def test_non_causal_paged_prefill_matches_sdpa(self):
        """Stays exact across the amd.1 regression — isolates it to masking.

        Without this, a future causal-only failure looks like a generic layout
        or scaling bug and sends the next person down the wrong path.
        """
        error = self._run_prefill(num_qo_heads=32, num_kv_heads=8, causal=False)
        self.assertLess(error, _TOLERANCE, f"max abs error {error:.4f}")

    def test_decode_matches_sdpa(self):
        """Pins the paged (k, v) tuple layout independently of the mask path."""
        import flashinfer

        num_qo_heads, num_kv_heads = 32, 8
        _, k, v, paged, indptr, indices, last_page_len = self._make_inputs(
            num_qo_heads, num_kv_heads
        )
        q = torch.randn(
            1, num_qo_heads, self.head_dim, dtype=self.dtype, device=self.device
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, "NHD", backend="fa2", use_tensor_cores=False
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        actual = wrapper.run(q, paged).float()
        error = (actual - _sdpa_reference(q, k, v, causal=False)).abs().max().item()
        self.assertLess(error, _TOLERANCE, f"max abs error {error:.4f}")


if __name__ == "__main__":
    unittest.main()
