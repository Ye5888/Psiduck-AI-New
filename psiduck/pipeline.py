"""Solvers and the head-to-head comparison.

``LatentDelegationSolver`` — the Psiduck-AI approach. One shared backbone:
    1. encode the problem                -> context latent   (no generation)
    2. 4 delegates exchange latent msgs  -> 4 latent messages (no generation)
    3. controller fuses + decodes the final integer answer (short generation)

Optionally it decodes the answer ``num_samples`` times (self-consistency: cheap
because each decode is only a few tokens) and takes a majority vote, which is
how this scaffold narrows the accuracy gap against the baseline without adding
any new English generation between agents.

``SingleModelSolver`` — the baseline. One model reasons in English (long
generation) and we parse the boxed integer.

``compare()`` runs both on a problem and returns answers + compute stats so we
can test the thesis that latent communication uses fewer generated tokens.

Both solvers accept an optional ``on_step`` callback -- ``on_step(label: str)``
-- invoked right before each backbone call, so a caller (a CLI progress line,
or the web demo's live activity feed) can show which agent/model is "on" at
that moment without the solvers knowing anything about UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .aime import extract_answer, majority_vote
from .backbone import Backbone, LatentState
from .metrics import ComputeStats
from .roles import DELEGATE_ROLES

OnStep = Optional[Callable[[str], None]]


def _emit(on_step: OnStep, label: str) -> None:
    if on_step is not None:
        on_step(label)


@dataclass
class SolveResult:
    approach: str
    answer: Optional[int]
    raw_text: str
    compute: ComputeStats
    wall_time_s: float
    detail: dict


class LatentDelegationSolver:
    def __init__(
        self,
        backbone: Backbone,
        answer_max_new_tokens: int = 24,
        *,
        num_samples: int = 1,
    ) -> None:
        self.backbone = backbone
        self.answer_max_new_tokens = answer_max_new_tokens
        # Self-consistency: decode the (cheap) answer this many times and
        # majority-vote. Each extra sample costs only ~answer_max_new_tokens
        # generated tokens, so a handful of samples still generates far fewer
        # tokens than one English chain-of-thought baseline.
        self.num_samples = max(1, num_samples)

    @property
    def num_agents(self) -> int:
        return 1 + len(DELEGATE_ROLES) + 1  # encoder/controller + delegates + decoder

    def solve(self, problem: str, *, on_step: OnStep = None) -> SolveResult:
        before = self.backbone.meter.snapshot()
        t0 = time.time()

        _emit(on_step, "encoder")
        context = self.backbone.encode(problem)

        messages: List[LatentState] = []
        for role in DELEGATE_ROLES:
            _emit(on_step, role)
            messages.append(self.backbone.delegate_message(context, role))

        fused = self.backbone.combine([context, *messages])

        raw_samples: List[str] = []
        sampled_answers: List[Optional[int]] = []
        for i in range(self.num_samples):
            _emit(
                on_step,
                "decoder" if self.num_samples == 1 else f"decoder (sample {i + 1}/{self.num_samples})",
            )
            raw = self.backbone.decode_answer(fused, self.answer_max_new_tokens)
            raw_samples.append(raw)
            sampled_answers.append(extract_answer(raw))

        answer = majority_vote(sampled_answers)
        # Report the first raw decode as the representative text, but keep all
        # samples/votes visible in `detail` for transparency.
        raw = raw_samples[0]

        elapsed = time.time() - t0
        compute = self.backbone.meter.snapshot().minus(before)
        return SolveResult(
            approach="latent-4-agent"
            if self.num_samples == 1
            else f"latent-4-agent+sc{self.num_samples}",
            answer=answer,
            raw_text=raw,
            compute=compute,
            wall_time_s=elapsed,
            detail={
                "shared_backbone_id": id(self.backbone),
                "num_agents": self.num_agents,
                "delegates": len(DELEGATE_ROLES),
                "latent_messages": len(messages),
                "num_samples": self.num_samples,
                "sampled_answers": sampled_answers,
            },
        )


class SingleModelSolver:
    def __init__(self, backbone: Backbone, max_new_tokens: int = 256) -> None:
        self.backbone = backbone
        self.max_new_tokens = max_new_tokens

    def solve(self, problem: str, *, on_step: OnStep = None) -> SolveResult:
        before = self.backbone.meter.snapshot()
        t0 = time.time()
        _emit(on_step, "single-llm")
        raw = self.backbone.baseline_generate(problem, self.max_new_tokens)
        elapsed = time.time() - t0
        compute = self.backbone.meter.snapshot().minus(before)
        return SolveResult(
            approach="single-llm-english",
            answer=extract_answer(raw),
            raw_text=raw,
            compute=compute,
            wall_time_s=elapsed,
            detail={},
        )


@dataclass
class Comparison:
    problem: str
    expected: Optional[int]
    baseline: SolveResult
    latent: SolveResult

    @property
    def generated_token_ratio(self) -> Optional[float]:
        b = self.baseline.compute.generated_tokens
        if b == 0:
            return None
        return self.latent.compute.generated_tokens / b


def compare(
    backbone: Backbone,
    problem: str,
    *,
    expected: Optional[int] = None,
    baseline_max_new_tokens: int = 256,
    answer_max_new_tokens: int = 24,
    num_samples: int = 1,
    on_step: OnStep = None,
) -> Comparison:
    baseline = SingleModelSolver(backbone, baseline_max_new_tokens).solve(
        problem, on_step=on_step
    )
    latent = LatentDelegationSolver(
        backbone, answer_max_new_tokens, num_samples=num_samples
    ).solve(problem, on_step=on_step)
    return Comparison(
        problem=problem, expected=expected, baseline=baseline, latent=latent
    )
