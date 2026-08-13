#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
./scripts/run-tests.sh
./scripts/replay-results.sh
./scripts/analyze-results.sh
./scripts/generate-figures.sh
./scripts/run-tests.sh
