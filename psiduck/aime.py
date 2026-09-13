"""Helpers for AIME-style answers.

AIME answers are always integers in the range 0-999. These utilities extract a
candidate answer from free-form model text and normalize it into that range.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional


_BOXED_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_ANSWER_LABEL_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*(-?\d{1,4})",
    re.IGNORECASE,
)
_INT_RE = re.compile(r"-?\d+")


def extract_boxed(text: str) -> Optional[int]:
    """Return the integer inside the last ``\\boxed{...}`` if present."""
    matches = _BOXED_RE.findall(text or "")
    if not matches:
        return None
    return int(matches[-1])


def normalize_aime_answer(value: int) -> int:
    """Coerce any integer into the valid AIME range [0, 999] via mod 1000."""
    return value % 1000


def extract_answer(text: str) -> Optional[int]:
    """Best-effort extraction of an AIME answer from model output.

    Priority: ``\\boxed{}`` > an "answer: N" style label > the last integer in
    the text. Returns ``None`` when no integer can be found.
    """
    if not text:
        return None

    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_aime_answer(boxed)

    label = _ANSWER_LABEL_RE.findall(text)
    if label:
        return normalize_aime_answer(int(label[-1]))

    ints = _INT_RE.findall(text)
    if ints:
        return normalize_aime_answer(int(ints[-1]))

    return None


def majority_vote(answers: Iterable[Optional[int]]) -> Optional[int]:
    """Return the most common non-None answer, or None if there are none.

    Ties are broken by the smallest value for determinism.
    """
    counts: Counter[int] = Counter(a for a in answers if a is not None)
    if not counts:
        return None
    top = max(counts.values())
    winners: List[int] = sorted(v for v, c in counts.items() if c == top)
    return winners[0]
