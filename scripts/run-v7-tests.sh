#!/usr/bin/env bash
set -euo pipefail
REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTEST_BIN="${THERMO_V7_PYTEST:-${REPOSITORY}/.venv/bin/pytest}"
"${PYTEST_BIN}" -q "${REPOSITORY}/tests/test_v7_topology.py" "${REPOSITORY}/tests/test_v7_entropy.py" "${REPOSITORY}/tests/test_v7_environment.py"
