"""CLI: compare the latent 4-agent pipeline against a single-LLM baseline.

Examples:
    python main.py --list
    python main.py --problem-index 2
    python main.py --problem "Find the remainder when 7^100 is divided by 1000."
    python main.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from textwrap import shorten
from typing import Optional

from psiduck.config import get_settings
from psiduck.data import SAMPLE_PROBLEMS, get_problem
from psiduck.pipeline import Comparison, compare


def _hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _build_backbone():
    settings = get_settings()
    print(f"Loading shared backbone: {settings.model_name} (CPU, fp32) ...")
    from psiduck.backbone import SharedModel

    t0 = time.time()
    model = SharedModel(
        settings.model_name,
        temperature=settings.temperature,
        seed=settings.seed,
        torch_threads=settings.torch_threads,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s (hidden size {model.hidden_size})")
    return model, settings


def _print_comparison(cmp: Comparison) -> None:
    b, l = cmp.baseline, cmp.latent

    _hr("BASELINE (single LLM, reasons in English)")
    print(shorten(b.raw_text.replace("\n", " "), width=600))
    _hr("LATENT 4-AGENT (agents exchange hidden-state vectors, no English between them)")
    print(f"Decoded final answer text: {l.raw_text!r}")
    print(f"Shared backbone id: {l.detail['shared_backbone_id']}  "
          f"agents: {l.detail['num_agents']}  "
          f"latent messages exchanged: {l.detail['latent_messages']}")

    _hr("RESULT")

    def verdict(ans: Optional[int]) -> str:
        if cmp.expected is None:
            return "n/a"
        return "CORRECT" if ans == cmp.expected else "wrong"

    rows = [
        ("approach", "answer", "correct", "gen_tokens", "fwd_passes", "wall_s"),
        (
            b.approach,
            str(b.answer),
            verdict(b.answer),
            str(b.compute.generated_tokens),
            str(b.compute.forward_passes),
            f"{b.wall_time_s:.1f}",
        ),
        (
            l.approach,
            str(l.answer),
            verdict(l.answer),
            str(l.compute.generated_tokens),
            str(l.compute.forward_passes),
            f"{l.wall_time_s:.1f}",
        ),
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r_i, row in enumerate(rows):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if r_i == 0:
            print("  ".join("-" * w for w in widths))

    if cmp.expected is not None:
        print(f"\nExpected answer: {cmp.expected}")
    ratio = cmp.generated_token_ratio
    if ratio is not None:
        print(
            f"Latent pipeline generated {l.compute.generated_tokens} tokens vs "
            f"{b.compute.generated_tokens} for the baseline "
            f"({ratio:.1%} as many; lower = cheaper English articulation)."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Psiduck-AI latent vs baseline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--problem", type=str, help="A custom problem statement.")
    group.add_argument("--problem-index", type=int, default=0)
    group.add_argument("--all", action="store_true", help="Run all sample problems.")
    parser.add_argument("--list", action="store_true", help="List sample problems.")
    parser.add_argument(
        "--self-consistency",
        type=int,
        default=1,
        metavar="N",
        help="Decode the latent answer N times and majority-vote (default 1 = off). "
        "Needs PSIDUCK_TEMPERATURE > 0 to get diverse samples. This is how the "
        "latent pipeline narrows the accuracy gap against the baseline.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for i, p in enumerate(SAMPLE_PROBLEMS):
            print(f"[{i}] ({p.difficulty}) {p.problem_id}: "
                  f"{shorten(p.statement, width=70)}")
        return 0

    model, settings = _build_backbone()

    def run_one(problem_text: str, expected: Optional[int]) -> Comparison:
        _hr("PROBLEM")
        print(problem_text)
        if expected is not None:
            print(f"(known answer: {expected})")
        cmp = compare(
            model,
            problem_text,
            expected=expected,
            baseline_max_new_tokens=settings.baseline_max_new_tokens,
            answer_max_new_tokens=settings.answer_max_new_tokens,
            num_samples=args.self_consistency,
        )
        _print_comparison(cmp)
        return cmp

    comparisons = []
    if args.all:
        for p in SAMPLE_PROBLEMS:
            comparisons.append(run_one(p.statement, p.answer))
    elif args.problem:
        comparisons.append(run_one(args.problem, None))
    else:
        p = get_problem(args.problem_index)
        comparisons.append(run_one(p.statement, p.answer))

    if len(comparisons) > 1:
        _hr("AGGREGATE COMPUTE")
        tot_b = sum(c.baseline.compute.generated_tokens for c in comparisons)
        tot_l = sum(c.latent.compute.generated_tokens for c in comparisons)
        corr_b = sum(
            1 for c in comparisons if c.expected is not None and c.baseline.answer == c.expected
        )
        corr_l = sum(
            1 for c in comparisons if c.expected is not None and c.latent.answer == c.expected
        )
        n = sum(1 for c in comparisons if c.expected is not None)
        print(f"Problems: {len(comparisons)} (scored: {n})")
        print(f"Baseline correct: {corr_b}/{n} | Latent correct: {corr_l}/{n}")
        print(f"Total generated tokens -> baseline: {tot_b}, latent: {tot_l}"
              + (f" ({tot_l / tot_b:.1%} as many)" if tot_b else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
