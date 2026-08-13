#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: $0 [bootstrap|resume]" >&2
  exit 2
fi
mode="${1:-bootstrap}"
[[ "$mode" == "bootstrap" || "$mode" == "resume" ]] || {
  echo "Mode must be bootstrap or resume." >&2
  exit 2
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$repo_dir/scripts"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

"$python_bin" -m thermoagent capture-source-provenance \
  --root "$repo_dir" \
  --output "$repo_dir/results/entropy_triggered_v2/reproducibility/execution_source.json" \
  >/dev/null

files=(
  results/reproducibility/macrostate_calibration.json
  results/entropy_triggered_v2/calibration/calibration_manifest.json
  results/entropy_triggered_v2/calibration/trigger_nominal_calibration.json
  results/entropy_triggered_v2/reproducibility/execution_source.json
)
if [[ "$mode" == "resume" ]]; then
  optional=(
    results/entropy_triggered_v2/validation
    results/entropy_triggered_v2/training
    results/entropy_triggered_v2/checkpoints
    results/entropy_triggered_v2/protocol
    results/entropy_triggered_v2/manifests
    results/entropy_triggered_v2/logs/setup
  )
  for path in "${optional[@]}"; do
    [[ -e "$repo_dir/$path" ]] && files+=("$path")
  done
fi
for path in "${files[@]}"; do
  [[ -e "$repo_dir/$path" ]] || {
    echo "Missing required v2 control artifact: $path" >&2
    exit 2
  }
done

ssh "${ssh_args[@]}" "$remote_host" mkdir -p \
  "$remote_dir/results/entropy_triggered_v2/reproducibility" \
  "$remote_dir/results/entropy_triggered_v2/calibration" \
  "$remote_dir/results/reproducibility"
(
  cd "$repo_dir"
  rsync --archive --relative --no-owner --no-group --compress \
    --human-readable --itemize-changes --rsh="$rsync_shell_quoted" \
    "${files[@]}" "$remote_host:$remote_dir/"
)
echo "Synchronized v2 $mode controls; raw results and frozen-v1 artifacts were not modified."
