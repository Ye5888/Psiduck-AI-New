"""Compute accounting for comparing the latent pipeline against a baseline.

The dominant cost of LLM inference is the *autoregressive generation* of tokens:
each generated token is one sequential forward step. Encoding a prompt or running
a single forward pass over a short sequence is comparatively cheap and
parallelizable. We therefore track, separately:

* ``generated_tokens`` — sequential decode steps (the expensive part), and
* ``forward_passes`` — non-generating forward calls (cheap, batchable), and
* ``processed_tokens`` — total tokens/soft-tokens fed through the model.

The thesis behind Psiduck-AI is that agents communicating in *latent space* skip
the English articulation between each other, so the multi-agent pipeline should
generate far fewer tokens than a single model reasoning in English.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class ComputeMeter:
    forward_passes: int = 0
    generated_tokens: int = 0
    processed_tokens: int = 0

    def reset(self) -> None:
        self.forward_passes = 0
        self.generated_tokens = 0
        self.processed_tokens = 0

    def add_forward(self, processed: int) -> None:
        self.forward_passes += 1
        self.processed_tokens += max(0, processed)

    def add_generation(self, prompt_len: int, generated: int) -> None:
        self.generated_tokens += max(0, generated)
        self.processed_tokens += max(0, prompt_len)

    def snapshot(self) -> "ComputeStats":
        return ComputeStats(
            forward_passes=self.forward_passes,
            generated_tokens=self.generated_tokens,
            processed_tokens=self.processed_tokens,
        )


@dataclass(frozen=True)
class ComputeStats:
    forward_passes: int
    generated_tokens: int
    processed_tokens: int

    def minus(self, other: "ComputeStats") -> "ComputeStats":
        return replace(
            self,
            forward_passes=self.forward_passes - other.forward_passes,
            generated_tokens=self.generated_tokens - other.generated_tokens,
            processed_tokens=self.processed_tokens - other.processed_tokens,
        )
