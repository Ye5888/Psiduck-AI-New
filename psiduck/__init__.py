"""Psiduck-AI: latent-space multi-agent solving for AIME problems.

One open-weights model is loaded once and shared by a controller/encoder, four
specialist delegates, and a decoder. The agents communicate by exchanging
*latent vectors* (pooled hidden states injected as soft tokens) rather than
English text, so no natural-language tokens are generated between agents -- only
the final integer answer is decoded. The goal is to show this style of
"AI-to-AI" communication is scalable and can use less generation compute than a
single model reasoning in English.
"""

from .config import Settings, get_settings
from .metrics import ComputeMeter, ComputeStats
from .pipeline import (
    Comparison,
    LatentDelegationSolver,
    SingleModelSolver,
    SolveResult,
    compare,
)

__all__ = [
    "Settings",
    "get_settings",
    "ComputeMeter",
    "ComputeStats",
    "Comparison",
    "LatentDelegationSolver",
    "SingleModelSolver",
    "SolveResult",
    "compare",
]
