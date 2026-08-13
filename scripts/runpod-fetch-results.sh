#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"

mkdir -p "$repo_dir/results"
rsync --archive --no-owner --no-group --compress --human-readable --itemize-changes \
  --rsh="$rsync_shell_quoted" \
  "$remote_host:$remote_dir/results/" "$repo_dir/results/"
echo "Fetched research results into $repo_dir/results."
