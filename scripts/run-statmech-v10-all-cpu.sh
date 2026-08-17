#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$repository_dir/scripts/run-statmech-v10-tests.sh"
"$repository_dir/scripts/run-statmech-v10-development.sh"
"$repository_dir/scripts/run-statmech-v10-freeze.sh"
"$repository_dir/scripts/run-statmech-v10-formal.sh"
"$repository_dir/scripts/run-statmech-v10-analysis.sh"
"$repository_dir/scripts/run-statmech-v10-paper.sh"
"$repository_dir/scripts/run-statmech-v10-export.sh"
