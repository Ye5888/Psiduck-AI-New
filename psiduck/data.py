"""Sample problems with known integer answers, for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AimeProblem:
    problem_id: str
    statement: str
    answer: int
    difficulty: str  # "easy" | "amc" | "aime"


SAMPLE_PROBLEMS: List[AimeProblem] = [
    AimeProblem(
        problem_id="easy-sum",
        statement="What is 1 + 2 + 3 + ... + 10? Give the integer.",
        answer=55,
        difficulty="easy",
    ),
    AimeProblem(
        problem_id="easy-mod",
        statement="What is the remainder when 100 is divided by 7?",
        answer=2,
        difficulty="easy",
    ),
    AimeProblem(
        problem_id="amc-style-odd-sum",
        statement=(
            "The sum of the first n positive odd integers is 225. What is n?"
        ),
        answer=15,
        difficulty="amc",
    ),
    AimeProblem(
        problem_id="2023-I-1",
        statement=(
            "Five men and nine women stand equally spaced around a circle in "
            "random order. The probability that every man stands diametrically "
            "opposite a woman is m/n, where m and n are relatively prime positive "
            "integers. Find m + n."
        ),
        answer=191,
        difficulty="aime",
    ),
    AimeProblem(
        problem_id="2022-I-1",
        statement=(
            "Quadratic polynomials P(x) and Q(x) have leading coefficients 2 and "
            "-2, respectively. The graphs of both polynomials pass through the two "
            "points (16,54) and (20,53). Find P(0) + Q(0)."
        ),
        answer=116,
        difficulty="aime",
    ),
    AimeProblem(
        problem_id="aime-style-pow-mod",
        statement="Find the remainder when 7^100 is divided by 1000.",
        answer=1,
        difficulty="aime",
    ),
    AimeProblem(
        problem_id="aime-style-divisor-sum",
        statement=(
            "Let S be the sum of the positive divisors of 2023. Find the "
            "remainder when S is divided by 1000."
        ),
        answer=456,
        difficulty="aime",
    ),
]


def get_problem(index_or_id: str | int) -> AimeProblem:
    if isinstance(index_or_id, int):
        return SAMPLE_PROBLEMS[index_or_id]
    for p in SAMPLE_PROBLEMS:
        if p.problem_id == index_or_id:
            return p
    try:
        return SAMPLE_PROBLEMS[int(index_or_id)]
    except (ValueError, IndexError) as exc:
        raise KeyError(f"No sample problem matching {index_or_id!r}") from exc


def problems_by_difficulty(*difficulties: str) -> List[AimeProblem]:
    """Return sample problems matching any of the given difficulty tags.

    With no arguments, returns every sample problem.
    """
    if not difficulties:
        return list(SAMPLE_PROBLEMS)
    wanted = set(difficulties)
    return [p for p in SAMPLE_PROBLEMS if p.difficulty in wanted]
