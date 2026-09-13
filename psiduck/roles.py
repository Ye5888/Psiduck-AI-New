"""Specialist role descriptors for the four delegates.

Each role is a short natural-language tag that is embedded (not generated) and
prepended to the shared context to steer that delegate. Because these are only
ever fed through the embedding layer as soft tokens, no English is *generated*
for inter-agent communication.
"""

from __future__ import annotations

DELEGATE_ROLES: tuple[str, ...] = (
    "Role: algebra and equation manipulation specialist.",
    "Role: combinatorics and counting specialist.",
    "Role: number theory and modular arithmetic specialist.",
    "Role: geometry and verification specialist.",
)
