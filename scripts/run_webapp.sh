#!/usr/bin/env bash
# Launch the Psiduck-AI web demo (webapp/app.py).
#
# Uses the real shared model automatically if it's importable AND already
# cached locally (no download attempt, no hang on a blocked network) --
# otherwise falls back to the dependency-free SimulatedBackbone. Force one or
# the other with PSIDUCK_BACKBONE=real|simulated.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="./.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

export HF_HOME="$(pwd)/.hf_cache"
echo "==> Starting Psiduck-AI web demo on http://localhost:${PORT:-5000}"
echo "    (PSIDUCK_BACKBONE=${PSIDUCK_BACKBONE:-auto}; set to 'simulated' to force the offline demo backbone)"
exec "$PY" webapp/app.py
