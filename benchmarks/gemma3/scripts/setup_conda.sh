#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="vai"
PYTHON_VERSION="3.12"

log() { printf '\n==> %s\n' "$*"; }
fail() { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }

log "Checking conda..."
if ! command -v conda >/dev/null 2>&1; then
  fail "conda command not found. Install Anaconda/Miniconda first, then reopen your shell."
fi

# Make 'conda activate' available in non-interactive shell scripts.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

log "Creating or updating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists. Ensuring Python ${PYTHON_VERSION} is installed..."
  conda install -n "$ENV_NAME" -y "python=${PYTHON_VERSION}"
else
  conda create -n "$ENV_NAME" -y "python=${PYTHON_VERSION}"
fi

log "Activating environment: ${ENV_NAME}"
conda activate "$ENV_NAME"

log "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

log "Installing Hugging Face Hub and Transformers"
python -m pip install --upgrade huggingface_hub transformers

log "Verifying installation"
python - <<'PY'
import sys
import importlib

checks = {
    "python": sys.version.split()[0],
}

for mod in ("huggingface_hub", "transformers"):
    m = importlib.import_module(mod)
    checks[mod] = getattr(m, "__version__", "unknown")

print("\n[INSTALLATION SUMMARY]")
for k, v in checks.items():
    print(f"  {k:16s}: {v}")

major, minor = sys.version_info[:2]
if (major, minor) != (3, 12):
    raise SystemExit(f"Python version mismatch: expected 3.12, got {major}.{minor}")

print("\nRESULT: SUCCESS")
PY

log "Environment is ready"
echo "Activate it later with: conda activate ${ENV_NAME}"
