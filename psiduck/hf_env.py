"""Point the Hugging Face cache at an in-repo directory by default.

The Cloud Agent environment persists the workspace volume but not the user home
directory, so we cache model weights under ``<repo>/.hf_cache`` (gitignored) so a
prefetched model survives into environment snapshots/builds. Call
``ensure_hf_home()`` *before* importing ``transformers``.
"""

from __future__ import annotations

import os


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_hf_home() -> str:
    cache = os.path.join(repo_root(), ".hf_cache")
    os.environ.setdefault("HF_HOME", cache)
    return os.environ["HF_HOME"]
