#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/run-human-operator-tests.sh"
"$script_dir/replay-human-operator-results.sh"
"$script_dir/analyze-human-operator-results.sh"
"$script_dir/generate-human-operator-figures.sh"
"$script_dir/validate-human-operator-pdfs.sh"
echo "ThermoHITL development no-go result rebuilt from retained ledgers."
