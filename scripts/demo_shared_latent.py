"""Demonstrate that all agents operate in one shared latent space.

Loads the model once and shows that a pooled hidden-state vector (a "latent
message") can be read from the shared backbone. Because every agent in the
pipeline holds this same backbone object, the latent messages they exchange all
live in one space -- that is what makes zero-English AI-to-AI communication
possible.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psiduck.backbone import SharedModel
from psiduck.config import get_settings
from psiduck.pipeline import LatentDelegationSolver, SingleModelSolver


def main() -> int:
    settings = get_settings()
    print(f"Loading shared backbone: {settings.model_name}")
    model = SharedModel(settings.model_name, torch_threads=settings.torch_threads)

    latent_solver = LatentDelegationSolver(model)
    baseline_solver = SingleModelSolver(model)

    same = (latent_solver.backbone is model) and (baseline_solver.backbone is model)
    print(f"\nHidden size (latent dimension): {model.hidden_size}")
    print(f"Latent solver agents share the backbone: {same}")
    print(f"Shared backbone object id: {id(model)}")

    vec = model.latent_vector("Find the remainder when 7^100 is divided by 1000.")
    print(f"\nPooled latent message dimension: {len(vec)}")
    print(f"First 8 dims: {[round(x, 4) for x in vec[:8]]}")

    assert same, "agents do not share a single backbone!"
    print("\nOK: one backbone, one latent space, shared by all agents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
