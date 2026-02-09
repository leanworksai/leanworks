#!/usr/bin/env bash
# Run tests inside a virtual environment.
# Creates .venv if missing, installs deps, runs pytest.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Creating .venv..."
  python3 -m venv "$ROOT/.venv"
fi
source "$ROOT/.venv/bin/activate"
echo "Installing dependencies..."
pip install -q -r requirements.txt -r requirements-dev.txt
echo "Running pytest..."
python -m pytest tests/ "$@"
