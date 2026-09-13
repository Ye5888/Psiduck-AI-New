"""Smoke tests for the Flask web demo, using SimulatedBackbone (realtime off)
so the SSE run endpoint completes instantly instead of sleeping through the
simulated latency that makes the live demo feel real.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import webapp.app as webapp_app
from psiduck.sim_backbone import SimulatedBackbone


def _client():
    webapp_app.app.testing = True
    return webapp_app.app.test_client()


def test_problems_endpoint_lists_all_sample_problems():
    client = _client()
    res = client.get("/api/problems")
    assert res.status_code == 200
    data = res.get_json()
    assert any(p["difficulty"] == "amc" for p in data)
    assert any(p["difficulty"] == "aime" for p in data)


def test_status_before_and_after_backbone_loads():
    webapp_app._backbone = None
    webapp_app._backbone_meta = None
    client = _client()
    assert client.get("/api/status").get_json() == {"loaded": False}

    webapp_app._backbone = SimulatedBackbone(seed=0, realtime=False)
    webapp_app._backbone_meta = {"kind": "simulated", "name": webapp_app._backbone.name}
    status = client.get("/api/status").get_json()
    assert status["loaded"] is True
    assert status["kind"] == "simulated"


def test_run_endpoint_streams_steps_and_a_result_for_each_approach():
    # Force a fast, non-sleeping backbone so this test doesn't take seconds.
    webapp_app._backbone = SimulatedBackbone(seed=0, realtime=False)
    webapp_app._backbone_meta = {"kind": "simulated", "name": webapp_app._backbone.name}
    client = _client()

    res = client.get("/api/run/amc-style-odd-sum?samples=3")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    events = [
        json.loads(line[len("data: "):])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]

    types = [e["type"] for e in events]
    assert "step" in types
    assert types[-1] == "done"

    results = {e["approach"]: e for e in events if e["type"] == "result"}
    assert set(results) == {"baseline", "latent"}
    assert results["baseline"]["answer"] == 15
    assert results["latent"]["answer"] == 15
    # The headline claim, reproduced end-to-end through the HTTP layer.
    assert results["latent"]["generated_tokens"] < results["baseline"]["generated_tokens"]
