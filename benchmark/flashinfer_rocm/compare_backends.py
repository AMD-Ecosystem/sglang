"""Three-way numerical comparison of ROCm attention backends.

    python3 benchmark/flashinfer_rocm/compare_backends.py [--gpu 3] [--model ...]

Answers "do these backends compute the same thing", which a pairwise diff
cannot: it tells you two backends differ, not which one moved. Running a third
backend as reference is what distinguishes "my backend is wrong" from "the
baseline is."

That distinction is not academic here. Measured on MI300X, `flashinfer_rocm`
sits ~3.5x closer to `triton` than `aiter` does, so a two-way comparison against
`aiter` makes the more accurate backend look like the broken one.

Two methodology points, both learned by getting them wrong:

  * Compare probabilities, not logprobs. |delta log p| is scale-distorted -- a
    step where two backends put 8.3% vs 6.5% on the chosen token, a difference
    nobody cares about, reads as 0.2575 in log space.
  * Ignore token identity as a pass/fail signal. Greedy breaks an exact logprob
    tie arbitrarily and one flip rewrites the whole tail, so byte-comparison of
    output text is a coin toss dressed up as a correctness check.

Re-run this after any change to the ROCm attention kernels.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os

PROMPTS = [
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
MAX_NEW_TOKENS = 32


def collect(model: str, backend: str) -> list[tuple[list[int], list[float]]]:
    """Greedy-decode every prompt, returning (token ids, chosen-token logprobs)."""
    import requests

    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import (
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        DEFAULT_URL_FOR_TEST,
        popen_launch_server,
    )

    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=["--attention-backend", backend],
    )
    try:
        runs = []
        for prompt in PROMPTS:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                    "return_logprob": True,
                },
            )
            response.raise_for_status()
            entries = response.json()["meta_info"]["output_token_logprobs"]
            runs.append(([e[1] for e in entries], [e[0] for e in entries]))
        return runs
    finally:
        kill_process_tree(process.pid)


def worst_probability_delta(run_a, run_b) -> tuple[float, int]:
    """Max |delta p| on the chosen token, over the tokens both runs selected.

    Restricted to the agreeing prefix because once the two diverge they are
    conditioning on different text, and comparing them past that point measures
    nothing.
    """
    worst, min_agree = 0.0, MAX_NEW_TOKENS
    for (tok_a, lp_a), (tok_b, lp_b) in zip(run_a, run_b):
        agree = 0
        for x, y in zip(tok_a, tok_b):
            if x != y:
                break
            agree += 1
        min_agree = min(min_agree, agree)
        for x, y in zip(lp_a[:agree], lp_b[:agree]):
            worst = max(worst, abs(math.exp(x) - math.exp(y)))
    return worst, min_agree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--gpu", default="0", help="pin to an IDLE GPU; this box is shared")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["flashinfer_rocm", "aiter", "triton"],
        help="three or more; a pairwise run cannot say which side moved",
    )
    args = parser.parse_args()
    os.environ["HIP_VISIBLE_DEVICES"] = args.gpu

    if len(args.backends) < 3:
        print(
            f"WARNING: {len(args.backends)} backends given. A pairwise delta shows "
            f"that two differ, not which is closer to correct. Add a reference "
            f"backend (triton)."
        )

    print(f"model={args.model} gpu={args.gpu} tokens={MAX_NEW_TOKENS}\n")
    runs = {b: collect(args.model, b) for b in args.backends}

    print(f"{'pair':>40}  {'max delta p':>11}  {'token agreement':>15}")
    results = {}
    for a, b in itertools.combinations(args.backends, 2):
        delta, agree = worst_probability_delta(runs[a], runs[b])
        results[(a, b)] = delta
        print(f"{a + ' vs ' + b:>40}  {delta:>11.4f}  {agree:>12d}/{MAX_NEW_TOKENS}")

    closest = min(results, key=results.get)
    print(
        f"\nClosest pair: {closest[0]} vs {closest[1]} ({results[closest]:.4f}). "
        f"A backend that is far from the other two is the outlier; a pair that is "
        f"far apart while each sits close to the third is just those two drifting "
        f"in opposite directions."
    )


if __name__ == "__main__":
    main()
