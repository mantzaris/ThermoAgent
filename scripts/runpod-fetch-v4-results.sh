#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$repo_dir/scripts"
source "$script_dir/runpod-common.sh"
destination="$repo_dir/results/human_operator_v4"
mkdir -p "$destination"
rsync --archive --no-owner --no-group --compress --human-readable \
  --itemize-changes --rsh="$rsync_shell_quoted" \
  "$remote_host:$remote_dir/results/human_operator_v4/" "$destination/"
echo "Fetched only the additive human_operator_v4 namespace into $destination."
