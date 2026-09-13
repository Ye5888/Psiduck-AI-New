"""A deterministic, dependency-free backbone for fast unit tests.

It implements the same high-level latent API as ``SharedModel`` without torch or
transformers, and updates a real ``ComputeMeter`` so we can assert the compute
accounting (e.g. latent communication generates far fewer tokens than the
English baseline).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from psiduck.backbone import LatentState
from psiduck.metrics import ComputeMeter


class FakeBackbone:
    def __init__(
        self,
        *,
        baseline_text: str = r"Reasoning ... therefore \boxed{116}.",
        answer_text: str = "116",
        answer_texts: Optional[Sequence[str]] = None,
        baseline_tokens: int = 200,
        name: str = "fake",
    ) -> None:
        self._baseline_text = baseline_text
        self._answer_text = answer_text
        # When set, decode_answer() cycles through these instead of always
        # returning `answer_text` -- lets tests exercise self-consistency
        # (multiple samples that don't all agree).
        self._answer_texts = list(answer_texts) if answer_texts else None
        self._decode_calls = 0
        self._baseline_tokens = baseline_tokens
        self._name = name
        self.meter = ComputeMeter()
        self.encode_calls: List[str] = []
        self.delegate_roles: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    def encode(self, text: str) -> LatentState:
        self.encode_calls.append(text)
        self.meter.add_forward(processed=len(text.split()))
        return LatentState(data=[float(len(text) % 7)], num_tokens=1)

    def delegate_message(self, context: LatentState, role: str) -> LatentState:
        self.delegate_roles.append(role)
        self.meter.add_forward(processed=1 + len(role.split()))
        return LatentState(data=[float(len(role) % 5)], num_tokens=1)

    def combine(self, states: Sequence[LatentState]) -> LatentState:
        return LatentState(data=[x for s in states for x in s.data], num_tokens=len(states))

    def decode_answer(self, context: LatentState, max_new_tokens: int) -> str:
        # Short, cheap decode -- this is the ONLY generation in the latent path.
        generated = min(max_new_tokens, 3)
        self.meter.add_generation(prompt_len=context.num_tokens, generated=generated)
        if self._answer_texts:
            text = self._answer_texts[self._decode_calls % len(self._answer_texts)]
            self._decode_calls += 1
            return text
        return self._answer_text

    def baseline_generate(self, prompt: str, max_new_tokens: int) -> str:
        generated = min(max_new_tokens, self._baseline_tokens)
        self.meter.add_generation(prompt_len=len(prompt.split()), generated=generated)
        return self._baseline_text
