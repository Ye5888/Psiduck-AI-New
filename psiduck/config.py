"""Runtime configuration, driven by environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class Settings:
    """Configuration for the shared model and the two solvers.

    Overridable via environment variables so the same code runs against a small
    CPU model here and a heavier model on capable hardware.
    """

    model_name: str = DEFAULT_MODEL
    baseline_max_new_tokens: int = 256
    answer_max_new_tokens: int = 24
    temperature: float = 0.0
    seed: int = 0
    torch_threads: int = 0  # 0 => let torch decide / use all cores


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_settings() -> Settings:
    return Settings(
        model_name=os.environ.get("PSIDUCK_MODEL", DEFAULT_MODEL),
        baseline_max_new_tokens=_int_env("PSIDUCK_BASELINE_MAX_NEW_TOKENS", 256),
        answer_max_new_tokens=_int_env("PSIDUCK_ANSWER_MAX_NEW_TOKENS", 24),
        temperature=_float_env("PSIDUCK_TEMPERATURE", 0.0),
        seed=_int_env("PSIDUCK_SEED", 0),
        torch_threads=_int_env("PSIDUCK_TORCH_THREADS", 0),
    )
