"""Psiduck-AI web demo: single LLM vs. latent 4-agent, live, side by side.

Run with:
    .venv/bin/python webapp/app.py

Then open http://localhost:5000. The shared backbone (real model, or the
dependency-free ``SimulatedBackbone`` when no model is available -- see
``psiduck/backbone_factory.py``) is loaded lazily on the first "Run" click
("turns on" at that moment) and reused after that, exactly like the CLI.

Endpoints:
    GET  /                    the demo page
    GET  /api/status          which backbone is active (real/simulated), model name
    GET  /api/problems        sample problems (id, statement, difficulty)
    GET  /api/run/<problem_id>?samples=N     SSE stream of one live comparison
    GET  /api/benchmark-chart the last-generated benchmark chart PNG
    POST /api/benchmark       regenerate the benchmark chart + CSV, returns the summary
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time

from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psiduck.backbone_factory import build_auto_backbone
from psiduck.config import get_settings
from psiduck.data import SAMPLE_PROBLEMS, get_problem
from psiduck.pipeline import LatentDelegationSolver, SingleModelSolver

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

app = Flask(__name__, static_folder="static")

_backbone_lock = threading.Lock()
_backbone = None
_backbone_meta = None


def get_backbone():
    """Lazily build the shared backbone on first use ("turns on" here)."""
    global _backbone, _backbone_meta
    with _backbone_lock:
        if _backbone is None:
            _backbone, _backbone_meta = build_auto_backbone()
        return _backbone, _backbone_meta


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    if _backbone is None:
        return jsonify({"loaded": False})
    return jsonify({"loaded": True, **_backbone_meta})


@app.get("/api/problems")
def problems():
    return jsonify(
        [
            {
                "id": p.problem_id,
                "statement": p.statement,
                "difficulty": p.difficulty,
                "answer": p.answer,
            }
            for p in SAMPLE_PROBLEMS
        ]
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/api/run/<problem_id>")
def run_problem(problem_id: str):
    try:
        problem = get_problem(problem_id)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404

    num_samples = max(1, int(request.args.get("samples", 5)))
    settings = get_settings()
    q: "queue.Queue[dict]" = queue.Queue()

    def worker() -> None:
        try:
            was_loaded = _backbone is not None
            backbone, meta = get_backbone()
            if hasattr(backbone, "set_problem"):
                backbone.set_problem(problem.statement, problem.answer)
            if not was_loaded:
                q.put(
                    {
                        "type": "backbone",
                        "kind": meta["kind"],
                        "name": meta["name"],
                        "message": f"Turned on shared backbone: {meta['name']}"
                        + (
                            " (simulated -- no model downloaded in this environment)"
                            if meta["kind"] == "simulated"
                            else ""
                        ),
                    }
                )

            def make_on_step(approach: str):
                def on_step(label: str) -> None:
                    q.put({"type": "step", "approach": approach, "label": label, "t": time.time()})

                return on_step

            baseline = SingleModelSolver(backbone, settings.baseline_max_new_tokens).solve(
                problem.statement, on_step=make_on_step("baseline")
            )
            q.put(
                {
                    "type": "result",
                    "approach": "baseline",
                    "answer": baseline.answer,
                    "correct": baseline.answer == problem.answer,
                    "generated_tokens": baseline.compute.generated_tokens,
                    "forward_passes": baseline.compute.forward_passes,
                    "wall_time_s": round(baseline.wall_time_s, 3),
                    "raw_text": baseline.raw_text[:400],
                }
            )

            latent = LatentDelegationSolver(
                backbone, settings.answer_max_new_tokens, num_samples=num_samples
            ).solve(problem.statement, on_step=make_on_step("latent"))
            q.put(
                {
                    "type": "result",
                    "approach": "latent",
                    "answer": latent.answer,
                    "correct": latent.answer == problem.answer,
                    "generated_tokens": latent.compute.generated_tokens,
                    "forward_passes": latent.compute.forward_passes,
                    "wall_time_s": round(latent.wall_time_s, 3),
                    "raw_text": latent.raw_text[:400],
                    "sampled_answers": latent.detail.get("sampled_answers"),
                }
            )
            q.put({"type": "done", "expected": problem.answer})
        except Exception as exc:  # surface to the UI instead of a dead stream
            q.put({"type": "error", "message": str(exc)})
            q.put({"type": "done", "expected": problem.answer})

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            yield _sse(item)
            if item.get("type") == "done":
                break

    return Response(stream(), mimetype="text/event-stream")


@app.get("/api/benchmark-chart")
def benchmark_chart():
    return send_from_directory(RESULTS_DIR, "benchmark_chart.png")


@app.post("/api/benchmark")
def run_benchmark_endpoint():
    from scripts.benchmark import render_chart, run_benchmark, write_csv, _aggregate

    payload = request.get_json(silent=True) or {}
    num_samples = max(1, int(payload.get("num_samples", 5)))
    seed = int(payload.get("seed", 0))

    rows = run_benchmark(
        backbone_kind="simulated",
        difficulties=("amc", "aime"),
        num_samples=num_samples,
        seed=seed,
    )
    write_csv(rows, os.path.join(RESULTS_DIR, "benchmark_results.csv"))
    render_chart(rows, os.path.join(RESULTS_DIR, "benchmark_chart.png"), title_suffix=" (simulated)")
    return jsonify({"ok": True, "summary": _aggregate(rows), "updated_at": time.time()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
