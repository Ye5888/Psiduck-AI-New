#!/usr/bin/env bash
# Idempotent environment bootstrap for Psiduck-AI.
#
# Everything is installed into the repository (a virtualenv at .venv and the
# Hugging Face cache at .hf_cache) rather than the user home directory, because
# the Cloud Agent environment persists the workspace volume but not $HOME. This
# lets a prebuilt environment build bake the deps + model into its snapshot.
#
# Safe to run repeatedly: pip is a no-op when satisfied and the model prefetch
# is skipped when the weights are already cached.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="$ROOT/.venv"
export HF_HOME="$ROOT/.hf_cache"

# Debian/Ubuntu ships venv support separately; install it if missing.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "==> Installing python venv support"
  PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  sudo apt-get update -y
  sudo apt-get install -y "python${PYVER}-venv" || sudo apt-get install -y python3-venv
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "==> Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"

echo "==> Upgrading pip"
"$PY" -m pip install --upgrade pip

echo "==> Installing Python dependencies"
"$PY" -m pip install -r requirements.txt

echo "==> Prefetching model weights into $HF_HOME"
"$PY" scripts/prefetch_model.py

echo "==> Install complete."
echo "    Run commands with: $VENV/bin/python  (or: source .venv/bin/activate)"
