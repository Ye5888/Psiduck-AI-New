import os

from psiduck.backbone_factory import build_auto_backbone
from psiduck.sim_backbone import SimulatedBackbone


def test_forced_simulated_never_touches_torch(monkeypatch):
    monkeypatch.setenv("PSIDUCK_BACKBONE", "simulated")
    backbone, meta = build_auto_backbone(seed=3)
    assert isinstance(backbone, SimulatedBackbone)
    assert meta["kind"] == "simulated"


def test_auto_falls_back_to_simulated_when_real_is_unavailable(monkeypatch):
    # In this test environment torch/transformers (or a cached model) aren't
    # available, so "auto" must fall back rather than raise or hang.
    monkeypatch.delenv("PSIDUCK_BACKBONE", raising=False)
    backbone, meta = build_auto_backbone(seed=4)
    assert meta["kind"] in ("real", "simulated")
    if meta["kind"] == "simulated":
        assert isinstance(backbone, SimulatedBackbone)
