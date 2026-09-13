"""Pick a real or simulated backbone without ever blocking on a download.

The web demo (and anything else that wants to "just work") should use the
real shared model when it's actually available, and fall back to
``SimulatedBackbone`` otherwise -- without hanging for a minute retrying a
blocked network call. So the real-backbone probe never fetches anything: it
only checks whether the configured model is *already* in the local Hugging
Face cache (``local_files_only=True``), which is a filesystem check.

Controlled by the ``PSIDUCK_BACKBONE`` env var:
    "auto" (default) -- real if importable + cached, else simulated
    "real"           -- force real (raises if unavailable/uncached)
    "simulated"      -- force the simulated backbone
"""

from __future__ import annotations

import os
from typing import Tuple


def _try_real():
    from .backbone import SharedModel
    from .config import get_settings
    from .hf_env import ensure_hf_home

    ensure_hf_home()
    from transformers import AutoConfig  # noqa: PLC0415

    settings = get_settings()
    # Filesystem-only check: raises immediately (no network) if the model
    # isn't already cached, instead of hanging on a blocked/slow download.
    AutoConfig.from_pretrained(settings.model_name, local_files_only=True)

    model = SharedModel(
        settings.model_name,
        temperature=settings.temperature,
        seed=settings.seed,
        torch_threads=settings.torch_threads,
    )
    return model, settings.model_name


def build_auto_backbone(*, seed: int = 0) -> Tuple[object, dict]:
    """Return ``(backbone, meta)``. ``meta`` has ``kind`` and ``name`` keys."""
    forced = os.environ.get("PSIDUCK_BACKBONE", "auto").strip().lower()

    if forced == "real":
        model, name = _try_real()
        return model, {"kind": "real", "name": name}

    if forced != "simulated":  # "auto" or anything unrecognized
        try:
            model, name = _try_real()
            return model, {"kind": "real", "name": name}
        except Exception:
            pass  # not importable, or model not already cached -> simulate

    from .sim_backbone import SimulatedBackbone

    sim = SimulatedBackbone(seed=seed)
    return sim, {"kind": "simulated", "name": sim.name}
