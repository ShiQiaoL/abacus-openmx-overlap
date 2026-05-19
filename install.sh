#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

"$PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -e .

echo "Installed. Activate with:"
echo "  source $VENV/bin/activate"
echo "Then run:"
echo "  direct-overlap --help"
