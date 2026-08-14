#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$repo_dir/scripts"
source "$script_dir/runpod-common.sh"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

"$python_bin" -m thermoagent capture-source-provenance \
  --root "$repo_dir" \
  --output "$repo_dir/results/human_operator_v3/reproducibility/execution_source.json" \
  >/dev/null

files=(
  results/human_operator_v3/calibration/thermodynamic_calibration_n10.json
  results/human_operator_v3/calibration/thermodynamic_calibration_n10_agent_periods.csv
  results/human_operator_v3/reproducibility/execution_source.json
  results/human_operator_v3/reproducibility/v3_real_llm_prelaunch_projection.json
)
for path in "${files[@]}"; do
  [[ -f "$repo_dir/$path" ]] || {
    echo "Missing v3 control artifact: $path" >&2
    exit 2
  }
done

ssh "${ssh_args[@]}" "$remote_host" mkdir -p \
  "$remote_dir/results/human_operator_v3/calibration" \
  "$remote_dir/results/human_operator_v3/reproducibility"
(
  cd "$repo_dir"
  rsync --archive --relative --no-owner --no-group --compress \
    --human-readable --itemize-changes --rsh="$rsync_shell_quoted" \
    "${files[@]}" "$remote_host:$remote_dir/"
)
echo "Synchronized only v3 controls; frozen v1/v2 and raw result namespaces were not touched."
