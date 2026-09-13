"""Benchmark: single-LLM baseline vs. the latent 4-agent pipeline.

Runs every sample AIME/AMC-style problem (see ``psiduck/data.py``) through
both approaches, writes a CSV of the raw results, and renders a chart
comparing accuracy and generated-token cost. This is the script behind the
chart in the README and the "Run benchmark" button in the web demo
(``webapp/app.py``).

Usage:
    python scripts/benchmark.py                    # simulated backbone (no downloads)
    python scripts/benchmark.py --backbone real     # real Qwen model (needs torch + weights)
    python scripts/benchmark.py --num-samples 7     # more self-consistency samples
    python scripts/benchmark.py --difficulties aime,amc

By default this uses ``SimulatedBackbone`` (see ``psiduck/sim_backbone.py``)
so it runs anywhere with no network access, no GPU, and no multi-GB model
download -- useful for CI and for trying the product before committing to a
real model download. Pass ``--backbone real`` to use the actual shared HF
model described in the README.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psiduck.data import problems_by_difficulty
from psiduck.pipeline import compare


def _build_backbone(kind: str, seed: int, realtime: bool = True):
    if kind == "real":
        from psiduck.backbone import SharedModel
        from psiduck.config import get_settings

        settings = get_settings()
        print(f"Loading real shared backbone: {settings.model_name} ...")
        return SharedModel(
            settings.model_name,
            temperature=settings.temperature,
            seed=settings.seed,
            torch_threads=settings.torch_threads,
        ), None

    from psiduck.sim_backbone import SimulatedBackbone

    print("Using SimulatedBackbone (no model download; see psiduck/sim_backbone.py)")
    return SimulatedBackbone(seed=seed, realtime=realtime), "simulated"


def run_benchmark(
    *,
    backbone_kind: str = "simulated",
    difficulties: tuple[str, ...] = ("amc", "aime"),
    num_samples: int = 5,
    seed: int = 0,
    baseline_max_new_tokens: int = 256,
    answer_max_new_tokens: int = 24,
    realtime: bool = True,
) -> list[dict]:
    backbone, sim_flag = _build_backbone(backbone_kind, seed, realtime=realtime)
    problems = problems_by_difficulty(*difficulties) if difficulties else problems_by_difficulty()
    if not problems:
        raise SystemExit(f"No sample problems match difficulties={difficulties!r}")

    rows: list[dict] = []
    for p in problems:
        if sim_flag:  # SimulatedBackbone needs to know the target answer
            backbone.set_problem(p.statement, p.answer)
        print(f"  running {p.problem_id} ({p.difficulty}) ...")
        cmp = compare(
            backbone,
            p.statement,
            expected=p.answer,
            baseline_max_new_tokens=baseline_max_new_tokens,
            answer_max_new_tokens=answer_max_new_tokens,
            num_samples=num_samples,
        )
        for result in (cmp.baseline, cmp.latent):
            rows.append(
                {
                    "problem_id": p.problem_id,
                    "difficulty": p.difficulty,
                    "approach": result.approach,
                    "answer": result.answer,
                    "expected": p.answer,
                    "correct": result.answer == p.answer,
                    "generated_tokens": result.compute.generated_tokens,
                    "forward_passes": result.compute.forward_passes,
                    "wall_time_s": round(result.wall_time_s, 4),
                }
            )
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _aggregate(rows: list[dict]) -> dict:
    by_approach: dict[str, dict] = {}
    for r in rows:
        # Bucket latent approach names ("latent-4-agent+sc5") under one label
        # for aggregation, keyed by whether it's the baseline or the latent
        # pipeline, so different --num-samples runs stay comparable.
        bucket = "single-llm-english" if r["approach"] == "single-llm-english" else "latent-4-agent"
        d = by_approach.setdefault(
            bucket, {"n": 0, "correct": 0, "generated_tokens": 0, "wall_time_s": 0.0}
        )
        d["n"] += 1
        d["correct"] += int(bool(r["correct"]))
        d["generated_tokens"] += r["generated_tokens"]
        d["wall_time_s"] += r["wall_time_s"]
    return by_approach


def render_chart(rows: list[dict], path: str, *, title_suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = _aggregate(rows)
    approaches = [a for a in ("single-llm-english", "latent-4-agent") if a in agg]
    labels = {"single-llm-english": "Single LLM\n(English CoT)", "latent-4-agent": "Latent 4-agent\n(+ self-consistency)"}
    colors = {"single-llm-english": "#6b7280", "latent-4-agent": "#2563eb"}

    accuracy = [100.0 * agg[a]["correct"] / agg[a]["n"] for a in approaches]
    tokens = [agg[a]["generated_tokens"] / agg[a]["n"] for a in approaches]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.suptitle(f"Psiduck-AI: single LLM vs. latent 4-agent{title_suffix}", fontsize=12)

    bars1 = ax1.bar(approaches, accuracy, color=[colors[a] for a in approaches])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 112)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title("Accuracy on AMC/AIME-style problems")
    ax1.set_xticks(range(len(approaches)))
    ax1.set_xticklabels([labels[a] for a in approaches])
    for b, v in zip(bars1, accuracy):
        ax1.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}%", ha="center")

    bars2 = ax2.bar(approaches, tokens, color=[colors[a] for a in approaches])
    ax2.set_ylabel("Generated tokens (avg / problem)")
    ax2.set_title("Generation cost (lower = cheaper)")
    ax2.set_xticks(range(len(approaches)))
    ax2.set_xticklabels([labels[a] for a in approaches])
    for b, v in zip(bars2, tokens):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Wrote {path}")


def print_summary(rows: list[dict]) -> None:
    agg = _aggregate(rows)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for approach, d in agg.items():
        print(
            f"{approach:22s} accuracy {d['correct']}/{d['n']} "
            f"({100.0 * d['correct'] / d['n']:.0f}%)   "
            f"avg generated tokens {d['generated_tokens'] / d['n']:.1f}   "
            f"total wall time {d['wall_time_s']:.2f}s"
        )
    if "single-llm-english" in agg and "latent-4-agent" in agg:
        b, l = agg["single-llm-english"], agg["latent-4-agent"]
        b_tok = b["generated_tokens"] / b["n"]
        l_tok = l["generated_tokens"] / l["n"]
        if b_tok:
            print(f"\nLatent pipeline used {l_tok / b_tok:.1%} as many generated tokens as baseline.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone", choices=["simulated", "real"], default="simulated",
        help="'simulated' (default, no downloads) or 'real' (actual shared HF model).",
    )
    parser.add_argument(
        "--difficulties", default="amc,aime",
        help="Comma-separated difficulty tags to include (default: amc,aime). Use 'all' for every sample problem.",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Self-consistency samples for the latent decoder.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-max-new-tokens", type=int, default=256)
    parser.add_argument("--answer-max-new-tokens", type=int, default=24)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip SimulatedBackbone's simulated latency (no-op for --backbone real). Useful in CI.",
    )
    args = parser.parse_args(argv)

    difficulties = () if args.difficulties.strip().lower() == "all" else tuple(
        d.strip() for d in args.difficulties.split(",") if d.strip()
    )

    t0 = time.time()
    rows = run_benchmark(
        backbone_kind=args.backbone,
        difficulties=difficulties,
        num_samples=args.num_samples,
        seed=args.seed,
        baseline_max_new_tokens=args.baseline_max_new_tokens,
        answer_max_new_tokens=args.answer_max_new_tokens,
        realtime=not args.fast,
    )
    print(f"\nRan {len(rows)} solver runs in {time.time() - t0:.1f}s")

    csv_path = os.path.join(args.out_dir, "benchmark_results.csv")
    chart_path = os.path.join(args.out_dir, "benchmark_chart.png")
    write_csv(rows, csv_path)
    suffix = " (simulated)" if args.backbone == "simulated" else ""
    render_chart(rows, chart_path, title_suffix=suffix)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
