"""End-to-end tests for the flashinfer_rocm attention backend.

Usage:
python3 -m unittest test_flashinfer_rocm_attention_backend.TestFlashInferRocmAttnBackend.test_mmlu

AMD-only: the backend is registered under is_hip() and needs the amd-flashinfer
wheel, so there is no register_cuda_ci here.

The model can be overridden with SGLANG_TEST_ATTENTION_MODEL, which matters for
two reasons: the default is a gated repo, and this backend refuses GQA group
sizes outside 1/2/3/4/8. Any substitute must keep the default's attention shape
or the run says nothing useful -- Mistral-7B-Instruct-v0.3 (32/8, group 4)
matches Llama-3.1-8B exactly.

Set SGLANG_TEST_ATTENTION_MMLU_THRESHOLD too. The default pass mark only holds
for the default model; the harness scores an answer solely when the model emits
"Answer: X", so a model with other formatting habits scores near the 4-choice
floor while being entirely correct. Verified locally:

    SGLANG_TEST_ATTENTION_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
    SGLANG_TEST_ATTENTION_MMLU_THRESHOLD=0.28 \
    python3 -m unittest \
      registered.attention.test_flashinfer_rocm_attention_backend.TestFlashInferRocmAttnBackend
"""

import math
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
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


def _model_under_test() -> str:
    return envs.SGLANG_TEST_ATTENTION_MODEL.get() or DEFAULT_MODEL_NAME_FOR_TEST


def _mmlu_threshold(default: float) -> float:
    override = envs.SGLANG_TEST_ATTENTION_MMLU_THRESHOLD.get()
    return default if override is None else override

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


# Tokens to generate per prompt when comparing backends. Long enough to get past
# the prompt-shaped prefix and into free-running decode.
_EQUIVALENCE_TOKENS = 32

# Cross-backend agreement is compared in PROBABILITY space, not log space.
# |delta log p| is scale-distorted: for an unlikely token a negligible change in
# probability is a large change in log-probability. Measured here, a step where
# the two backends put 8.3% vs 6.5% on the chosen token -- a 0.019 difference
# that matters to nobody -- shows up as 0.2575 in log space.
#
# The threshold is calibrated against how far two *trusted* backends already sit
# from each other on this hardware, rather than picked. Max |delta p| over the
# agreeing prefix, Mistral-7B, 32 tokens, four prompts:
#
#     flashinfer_rocm vs triton   0.031
#     flashinfer_rocm vs aiter    0.101
#     aiter           vs triton   0.109
#
# All three pick identical tokens at all 32 positions; they disagree only on how
# much probability mass sits on them, and bf16 attention has room for that where
# the model is uncertain. Since aiter deviates 0.109 from the triton reference on
# its own, anything below that would fail known-good backends.
_PROB_TOLERANCE = 0.15

# A logprob margin at or under this counts as a tie for tie-break purposes.
# Observed real ties are exactly 0.0.
_TIE_MARGIN = 0.05


def _greedy_with_logprobs(base_url: str, prompt: str) -> tuple[list[int], list[float], list[float]]:
    """Greedy-decode one prompt, returning per-step (token, logprob, top2 margin).

    Token identity alone is not a usable cross-backend signal: greedy breaks an
    exact logprob tie arbitrarily, and one flipped tie rewrites the whole tail.
    The margin lets a divergence be classified rather than just detected.
    """
    response = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": _EQUIVALENCE_TOKENS},
            "return_logprob": True,
            "top_logprobs_num": 2,
        },
    )
    response.raise_for_status()
    meta = response.json()["meta_info"]
    tokens, logprobs, margins = [], [], []
    for (lp, tid, *_), top in zip(
        meta["output_token_logprobs"], meta["output_top_logprobs"]
    ):
        tokens.append(tid)
        logprobs.append(lp)
        ranked = sorted((e[0] for e in top), reverse=True)
        margins.append(ranked[0] - ranked[1] if len(ranked) > 1 else float("inf"))
    return tokens, logprobs, margins


def _compare_prompt(prompt, a, b) -> str | None:
    """Return a failure description, or None when the two runs agree.

    Agreement means the probability the two backends assign to each chosen token
    matches, not that greedy happened to pick the same tokens. A token
    divergence is accepted only when the decision there was a near-tie: greedy
    breaks an exact tie arbitrarily and one flip rewrites the whole tail.
    """
    (a_tok, a_lp, a_margin), (b_tok, b_lp, b_margin) = a, b

    agree_len = 0
    for x, y in zip(a_tok, b_tok):
        if x != y:
            break
        agree_len += 1

    worst = max(
        (
            abs(math.exp(x) - math.exp(y))
            for x, y in zip(a_lp[:agree_len], b_lp[:agree_len])
        ),
        default=0.0,
    )
    if worst > _PROB_TOLERANCE:
        return (
            f"{prompt!r}: probability assigned to the chosen token differs by "
            f"{worst:.4f} over the {agree_len} tokens both backends agreed on, "
            f"above the {_PROB_TOLERANCE} tolerance. Cross-check against a third "
            f"backend (--attention-backend triton) before concluding which side "
            f"is wrong; aiter itself sits ~0.109 from triton."
        )

    if agree_len == len(a_tok) == len(b_tok):
        return None

    margin = min(a_margin[agree_len], b_margin[agree_len])
    if margin <= _TIE_MARGIN:
        return None  # tie-break; both choices were defensible

    return (
        f"{prompt!r}: tokens diverge at index {agree_len} with a decision margin "
        f"of {margin:.4f} logprob, far above the {_TIE_MARGIN} that would make it "
        f"a tie-break. One backend picked a token the other considered clearly "
        f"worse."
    )


@unittest.skipUnless(
    is_flashinfer_rocm_available(),
    "requires the amd-flashinfer package on a ROCm GPU",
)
class TestFlashInferRocmAttnBackend(CustomTestCase):
    base_url = DEFAULT_URL_FOR_TEST
    # Tied to DEFAULT_MODEL_NAME_FOR_TEST; override alongside the model.
    mmlu_score_threshold = 0.65

    @classmethod
    def setUpClass(cls):
        cls.model = _model_under_test()
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
        threshold = _mmlu_threshold(self.mmlu_score_threshold)
        self.assertGreaterEqual(
            metrics["score"],
            threshold,
            f"MMLU {metrics['score']:.3f} < {threshold} on {self.model}. If this "
            f"is a substituted model, check the score against another backend "
            f"before suspecting attention -- the default threshold is calibrated "
            f"for {DEFAULT_MODEL_NAME_FOR_TEST} and does not transfer.",
        )


@unittest.skipUnless(
    is_flashinfer_rocm_available(),
    "requires the amd-flashinfer package on a ROCm GPU",
)
class TestFlashInferRocmMatchesAiter(CustomTestCase):
    """flashinfer_rocm must agree with aiter, the established ROCm backend.

    Guards the failure mode that motivated this comparison: a wrong attention
    result is right-shaped and finite, so it does not crash and it barely moves
    an aggregate eval score. Comparing against the backend SGLang already trusts
    on this hardware is what catches it.

    The comparison is on logprobs, not on output text. An earlier version
    asserted byte-identical greedy output and passed -- but only because its
    prompts happened to miss a tie. On a neighbouring prompt the two backends
    split at output token 7 where aiter's top two candidates were equal to the
    last bit (-2.19410 both), and one flipped tie rewrites the entire tail. That
    made exact match a coin toss dressed up as a correctness bar.
    """

    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.model = _model_under_test()

    @classmethod
    def _collect(cls, backend: str) -> list[tuple]:
        process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", backend],
        )
        try:
            return [
                _greedy_with_logprobs(cls.base_url, prompt)
                for prompt in _EQUIVALENCE_PROMPTS
            ]
        finally:
            kill_process_tree(process.pid)

    def test_logprobs_match_aiter(self):
        flashinfer_runs = self._collect("flashinfer_rocm")
        aiter_runs = self._collect("aiter")

        failures = [
            msg
            for prompt, a, b in zip(_EQUIVALENCE_PROMPTS, flashinfer_runs, aiter_runs)
            if (msg := _compare_prompt(prompt, a, b)) is not None
        ]
        # Report every failing prompt: a systematic error usually shows on more
        # than one, and which ones it spares is diagnostic.
        self.assertEqual(
            failures,
            [],
            f"{len(failures)}/{len(_EQUIVALENCE_PROMPTS)} prompts disagree between "
            f"flashinfer_rocm and aiter:\n" + "\n".join(f"  {f}" for f in failures),
        )


if __name__ == "__main__":
    unittest.main()
