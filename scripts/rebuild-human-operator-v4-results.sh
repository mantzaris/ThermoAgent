#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$repo_dir/scripts/run-human-operator-v4-tests.sh"
"$repo_dir/scripts/replay-human-operator-v4-results.sh"
"$repo_dir/scripts/analyze-human-operator-v4-results.sh"
"$repo_dir/scripts/generate-human-operator-v4-figures.sh"
"$repo_dir/scripts/validate-human-operator-v4-pdfs.sh"
"$repo_dir/scripts/build-human-operator-v4-report.sh"
