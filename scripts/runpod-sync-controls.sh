#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"

files=(
  results/reproducibility/excluded_runs.json
  results/reproducibility/pilot_v5_interrupted_config.yaml
  results/manifests/pilot_v5_interrupted.json
  results/reproducibility/pilot_v6_interrupted_config.yaml
  results/manifests/pilot_v6_interrupted.json
  results/reproducibility/pilot_v7_interrupted_config.yaml
  results/manifests/pilot_v7_interrupted.json
)
for path in "${files[@]}"; do
  [[ -f "$repo_dir/$path" ]] || {
    echo "Missing required result-control artifact: $path" >&2
    exit 1
  }
done

ssh "${ssh_args[@]}" "$remote_host" mkdir -p \
  "$remote_dir/results/reproducibility" "$remote_dir/results/manifests"
(
  cd "$repo_dir"
  rsync --archive --relative --no-owner --no-group --compress \
    --human-readable --itemize-changes --rsh="$rsync_shell_quoted" \
    "${files[@]}" "$remote_host:$remote_dir/"
)
echo "Synchronized prospective result-control metadata (runtime data untouched)."
