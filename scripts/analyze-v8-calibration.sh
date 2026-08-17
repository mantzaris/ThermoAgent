#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
stage="${1:?usage: analyze-v8-calibration.sh STAGE}"
exec "$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  analyze-calibration --stage "$stage"
