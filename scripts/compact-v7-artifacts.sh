#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS="${V7_RESULTS:-results/complexity_entropic_coordination_v7}"

exec "${PYTHON_BIN}" -m thermoagent.v7_compaction --results "${RESULTS}" "$@"
