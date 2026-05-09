#!/usr/bin/env bash
# Creates a virtual environment in .venv, installs dependencies from
# requirements.txt, and registers the nbstripout git filter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 was not found on PATH. Install Python and try again." >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "Creating virtual environment in .venv ..."
    python3 -m venv .venv
else
    echo ".venv already exists, skipping creation."
fi

echo "Upgrading pip ..."
.venv/bin/python -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt ..."
.venv/bin/python -m pip install -r requirements.txt

echo "Registering nbstripout git filter ..."
.venv/bin/nbstripout --install

echo
echo "Setup complete. Activate the environment with:"
echo "    source .venv/bin/activate"
