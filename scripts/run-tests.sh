#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ -x .venv/bin/pytest ]]; then
  exec .venv/bin/pytest "$@"
fi
exec python3 -m pytest "$@"

