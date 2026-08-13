#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"
local_runs="$repo_dir/runs"

mkdir -p "$local_runs"

if (( $# > 1 )); then
  echo "Usage: $0 [run-id]" >&2
  exit 2
fi

if (( $# == 1 )); then
  run_id="$1"
  if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Run ID may contain only letters, numbers, dot, underscore, and hyphen." >&2
    exit 2
  fi
  mkdir -p "$local_runs/$run_id"
  rsync --archive --no-owner --no-group --compress --human-readable --itemize-changes \
    --rsh="$rsync_shell_quoted" \
    "$remote_host:$remote_dir/runs/$run_id/" "$local_runs/$run_id/"
  echo "Fetched run $run_id into $local_runs/$run_id."
else
  rsync --archive --no-owner --no-group --compress --human-readable --itemize-changes \
    --rsh="$rsync_shell_quoted" \
    "$remote_host:$remote_dir/runs/" "$local_runs/"
  echo "Fetched all runs into $local_runs."
fi
