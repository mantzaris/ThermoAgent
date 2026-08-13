#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Active tmux jobs"
tmux list-sessions 2>/dev/null || echo "none"
echo
echo "Completed job statuses"
find "$repo_dir/results/manifests/job-status" -maxdepth 1 -type f -printf '%f ' -exec sed -n '1p' {} \; 2>/dev/null || true
echo
echo "Recent job logs"
find "$repo_dir/results/logs/jobs" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | cut -d' ' -f2- | while read -r path; do
      echo "== $path =="
      tail -n 8 "$path"
    done
