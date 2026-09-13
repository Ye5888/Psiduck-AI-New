"""Download the configured model into the in-repo HF cache.

Run during environment `install` so the weights are baked into the snapshot and
no download is needed at agent runtime. Idempotent: a cached model is a fast
no-op.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psiduck.hf_env import ensure_hf_home


def main() -> int:
    cache = ensure_hf_home()
    from psiduck.config import get_settings

    model_name = get_settings().model_name
    print(f"Prefetching model into HF cache ({cache}): {model_name}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        print(f"transformers not importable: {exc}", file=sys.stderr)
        return 1

    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    print("Prefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
