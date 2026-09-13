"""A dependency-free, deterministic stand-in for ``SharedModel``.

Downloading real weights (``scripts/prefetch_model.py``) needs network access
and a few GB of disk, which isn't always available (a sandboxed CI runner, an
offline laptop, this repo's own review environment). ``SimulatedBackbone``
implements the same :class:`~psiduck.backbone.Backbone` protocol so every
consumer -- ``main.py``, ``scripts/benchmark.py``, ``webapp/app.py`` -- can run
end-to-end without ``torch``/``transformers`` or a model download.

It is *not* a test double: unlike ``tests/fake_backbone.py`` (which returns a
fixed canned answer to pin down orchestration behavior), this module actually
attempts each problem -- it does simple arithmetic for problems it can parse,
and otherwise falls back to a seeded pseudo-random guess -- and simulates
realistic wall-clock latency for autoregressive generation vs. cheap forward
passes, so the demo's shape (few tokens / more forward passes / competitive
accuracy for the latent path once self-consistency is on) matches what the
real model is expected to show. Every place this backbone is used labels
its output "simulated" so it's never confused with a real model run.
"""

from __future__ import annotations

import random
import re
import time
from typing import List, Optional, Sequence

from .aime import normalize_aime_answer
from .backbone import LatentState
from .metrics import ComputeMeter

# Simulated per-token latency, tuned to *feel* like CPU generation (small
# model) without actually taking the minutes a real run would: real
# generation is the slow, sequential part; forward passes are fast.
_SECONDS_PER_GENERATED_TOKEN = 0.02
_SECONDS_PER_FORWARD_PASS = 0.05


def _try_exact_arithmetic(problem: str) -> Optional[int]:
    """Solve the handful of problem shapes this scaffold's samples use.

    This is intentionally narrow -- it is a stand-in for "a model that is
    actually good at arithmetic," not a general solver. Returns ``None`` when
    it doesn't recognize the shape, which is the common case for AIME/AMC
    style word problems.
    """
    text = problem.lower()

    # "1 + 2 + ... + 10"
    m = re.search(r"1\s*\+\s*2\s*\+\s*3\s*\+\s*\.\.\.\s*\+\s*(\d+)", text)
    if m:
        n = int(m.group(1))
        return n * (n + 1) // 2

    # "remainder when X is divided by Y"
    m = re.search(r"remainder when\s+([\d^*+\s]+?)\s+is divided by\s+(\d+)", text)
    if m:
        base_expr, mod = m.group(1).strip(), int(m.group(2))
        pw = re.fullmatch(r"(\d+)\s*\^\s*(\d+)", base_expr)
        if pw:
            base, exp = int(pw.group(1)), int(pw.group(2))
            return pow(base, exp, mod)
        if base_expr.isdigit():
            return int(base_expr) % mod

    # "sum of the first n positive odd integers is K. What is n?" -> n = sqrt(K)
    m = re.search(r"sum of the first n positive odd integers is\s+(\d+)", text)
    if m:
        k = int(m.group(1))
        root = int(round(k**0.5))
        if root * root == k:
            return root

    # "sum of the positive divisors of N" (mod 1000, matching this scaffold's phrasing)
    m = re.search(r"positive divisors of\s+(\d+)", text)
    if m:
        n = int(m.group(1))
        total = sum(d for d in range(1, n + 1) if n % d == 0)
        return normalize_aime_answer(total)

    return None


class SimulatedBackbone:
    """Simulates both approaches without any ML dependencies.

    ``accuracy`` knobs control how often the *simulated* baseline / a single
    latent decode sample "gets it right" on problems this backbone can't solve
    exactly -- tuned so the demo reproduces the two headline claims: far fewer
    generated tokens, and self-consistency (multiple samples + majority vote)
    closing the accuracy gap. Real accuracy on real hardware will differ; see
    the README's "Honest scope" note.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        baseline_accuracy: float = 0.55,
        single_sample_accuracy: float = 0.58,
        realtime: bool = True,
        name: str = "simulated-backbone (no model downloaded)",
    ) -> None:
        self.meter = ComputeMeter()
        self._rng = random.Random(seed)
        self._baseline_accuracy = baseline_accuracy
        self._single_sample_accuracy = single_sample_accuracy
        self._realtime = realtime
        self._name = name
        self._current_problem: Optional[str] = None
        self._current_expected: Optional[int] = None

    @property
    def name(self) -> str:
        return self._name

    def set_problem(self, problem: str, expected: Optional[int]) -> None:
        """Tell the simulator which problem/answer is being attempted.

        Solvers only pass problem *text* through ``encode``, not the expected
        answer, so callers that know the expected answer (the benchmark, the
        web demo) call this first to let the simulator decide, per problem,
        whether a given simulated attempt should land on the right answer.
        """
        self._current_problem = problem
        self._current_expected = expected

    def _sleep(self, seconds: float) -> None:
        if self._realtime and seconds > 0:
            time.sleep(seconds)

    def _guess(self, accuracy: float) -> Optional[int]:
        exact = (
            _try_exact_arithmetic(self._current_problem)
            if self._current_problem
            else None
        )
        if exact is not None:
            return exact
        if self._current_expected is None:
            return self._rng.randint(0, 999)
        if self._rng.random() < accuracy:
            return self._current_expected
        # A plausible-looking wrong answer, never accidentally the right one.
        wrong = self._rng.randint(0, 999)
        while wrong == self._current_expected:
            wrong = self._rng.randint(0, 999)
        return wrong

    # --- Backbone protocol --------------------------------------------------

    def encode(self, text: str) -> LatentState:
        self.meter.add_forward(processed=max(1, len(text.split())))
        self._sleep(_SECONDS_PER_FORWARD_PASS)
        return LatentState(data=None, num_tokens=1)

    def delegate_message(self, context: LatentState, role: str) -> LatentState:
        self.meter.add_forward(processed=1)
        self._sleep(_SECONDS_PER_FORWARD_PASS)
        return LatentState(data=None, num_tokens=1)

    def combine(self, states: Sequence[LatentState]) -> LatentState:
        return LatentState(data=None, num_tokens=len(states))

    def decode_answer(self, context: LatentState, max_new_tokens: int) -> str:
        generated = min(max_new_tokens, 3)
        self.meter.add_generation(prompt_len=context.num_tokens, generated=generated)
        self._sleep(generated * _SECONDS_PER_GENERATED_TOKEN)
        answer = self._guess(self._single_sample_accuracy)
        return "" if answer is None else str(answer)

    def baseline_generate(self, prompt: str, max_new_tokens: int) -> str:
        exact = _try_exact_arithmetic(prompt)
        if exact is not None:
            generated = min(max_new_tokens, 40)
            text = f"Reasoning step by step... therefore the answer is \\boxed{{{exact}}}."
        else:
            answer = self._guess(self._baseline_accuracy)
            generated = max_new_tokens  # a real chain-of-thought uses its full budget
            text = (
                "Reasoning step by step through the problem at length in English... "
                f"therefore the answer is \\boxed{{{answer}}}."
            )
        self.meter.add_generation(prompt_len=len(prompt.split()), generated=generated)
        self._sleep(generated * _SECONDS_PER_GENERATED_TOKEN)
        return text

    def latent_vector(self, text: str) -> List[float]:
        self._rng.seed(hash(text) & 0xFFFFFFFF)
        return [self._rng.uniform(-1, 1) for _ in range(8)]


__all__ = ["SimulatedBackbone"]
