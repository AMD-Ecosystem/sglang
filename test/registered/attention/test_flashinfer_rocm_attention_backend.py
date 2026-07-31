"""End-to-end tests for the flashinfer_rocm attention backend.

Usage:
python3 -m unittest test_flashinfer_rocm_attention_backend.TestFlashInferRocmAttnBackend.test_mmlu

AMD-only: the backend is registered under is_hip() and needs the amd-flashinfer
wheel, so there is no register_cuda_ci here.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import is_flashinfer_rocm_available, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=1800, suite="stage-b-test-1-gpu-small-amd")

# Prompts chosen to span the two paths that broke during bring-up: a long-ish
# prompt exercises causal paged prefill over many rows, short ones exercise the
# decode path with little prefix.
_EQUIVALENCE_PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue during the day.",
    "List the first five prime numbers in ascending order:",
    (
        "A train leaves station A at 9:00 am travelling at 60 km/h. A second "
        "train leaves station B, 240 km away, at 10:00 am travelling toward "
        "station A at 80 km/h. Work through the problem step by step and state "
        "when and where the two trains meet."
    ),
]


def _greedy_completions(base_url: str) -> list[str]:
    """Greedy-decode every equivalence prompt and return the raw texts."""
    outputs = []
    for prompt in _EQUIVALENCE_PROMPTS:
        response = requests.post(
            base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 32},
            },
        )
        response.raise_for_status()
        outputs.append(response.json()["text"])
    return outputs


@unittest.skipUnless(
    is_flashinfer_rocm_available(),
    "requires the amd-flashinfer package on a ROCm GPU",
)
class TestFlashInferRocmAttnBackend(CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "flashinfer_rocm"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )
        )
        self.assertGreaterEqual(metrics["score"], 0.65)


@unittest.skipUnless(
    is_flashinfer_rocm_available(),
    "requires the amd-flashinfer package on a ROCm GPU",
)
class TestFlashInferRocmMatchesAiter(CustomTestCase):
    """flashinfer_rocm must agree with aiter, the established ROCm backend.

    Guards the failure mode that motivated this comparison: a wrong attention
    result is right-shaped and finite, so it does not crash and it barely moves
    an aggregate eval score. Comparing greedy text against the backend SGLang
    already trusts on this hardware is what actually catches it.
    """

    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def _collect(cls, backend: str) -> list[str]:
        process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", backend],
        )
        try:
            return _greedy_completions(cls.base_url)
        finally:
            kill_process_tree(process.pid)

    def test_greedy_output_matches_aiter(self):
        flashinfer_outputs = self._collect("flashinfer_rocm")
        aiter_outputs = self._collect("aiter")

        mismatches = [
            (prompt, fi, ai)
            for prompt, fi, ai in zip(
                _EQUIVALENCE_PROMPTS, flashinfer_outputs, aiter_outputs
            )
            if fi != ai
        ]
        # Greedy decoding on identical weights should agree token for token.
        # Kernel-level fp differences can still diverge after a near-tie argmax,
        # so report every mismatch rather than just the first.
        self.assertEqual(
            mismatches,
            [],
            f"{len(mismatches)}/{len(_EQUIVALENCE_PROMPTS)} prompts diverged "
            f"between flashinfer_rocm and aiter:\n"
            + "\n".join(
                f"  prompt: {p!r}\n    flashinfer_rocm: {f!r}\n    aiter:           {a!r}"
                for p, f, a in mismatches
            ),
        )


if __name__ == "__main__":
    unittest.main()
