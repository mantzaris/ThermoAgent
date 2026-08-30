#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m pytest --import-mode=importlib "$@"
fi
exec python3 -m pytest --import-mode=importlib "$@"
