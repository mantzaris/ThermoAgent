#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$repo_dir/results/entropy_triggered_v2"
echo "Active tmux jobs"
tmux list-sessions 2>/dev/null || echo "none"
echo
echo "Completed job statuses"
find "$root/manifests/job-status" -maxdepth 1 -type f -printf '%f ' \
  -exec sed -n '1p' {} \; 2>/dev/null || true
echo
echo "Outcome-sealed holdout health"
holdout_manifests=$(find "$root/manifests" -maxdepth 1 -type f \
  -name 'holdout_locked-*.json' 2>/dev/null | wc -l)
holdout_outputs=$(find "$root/raw/holdout_locked" -mindepth 1 -maxdepth 1 \
  -type d 2>/dev/null | wc -l)
printf 'manifests=%s published_episode_directories=%s expected=1296\n' \
  "$holdout_manifests" "$holdout_outputs"
echo "No partial outcome values are displayed by this command."
