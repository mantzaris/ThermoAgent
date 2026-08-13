#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 JOB_NAME COMMAND [ARG ...]" >&2
  exit 2
fi
job_name="$1"
shift
[[ "$job_name" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Job name may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="$repo_dir/results/entropy_triggered_v2/logs/jobs"
status_dir="$repo_dir/results/entropy_triggered_v2/manifests/job-status"
mkdir -p "$log_dir" "$status_dir"
log_path="$log_dir/$job_name.log"
status_path="$status_dir/$job_name.exit"

if tmux has-session -t "$job_name" 2>/dev/null; then
  echo "tmux job already active: $job_name" >&2
  exit 2
fi
if [[ -e "$status_path" ]]; then
  echo "Completed status already exists for $job_name; choose a new job name." >&2
  exit 2
fi

printf -v quoted_repo '%q' "$repo_dir"
printf -v quoted_log '%q' "$log_path"
printf -v quoted_status '%q' "$status_path"
printf -v quoted_command '%q ' "$@"
tmux new-session -d -s "$job_name" \
  "cd $quoted_repo && set -o pipefail; $quoted_command >$quoted_log 2>&1; job_code=\$?; printf '%s\\n' \"\$job_code\" >$quoted_status"
echo "Started $job_name; health log: $log_path; exit status: $status_path"
