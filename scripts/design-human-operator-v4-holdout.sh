#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repo_dir/scripts/check-human-operator-v4-unlock.sh" holdout_design
