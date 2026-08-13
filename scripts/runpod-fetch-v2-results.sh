#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$repo_dir/scripts"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"
destination="$repo_dir/results/entropy_triggered_v2"
mkdir -p "$destination"
rsync --archive --no-owner --no-group --compress --human-readable \
  --itemize-changes --rsh="$rsync_shell_quoted" \
  "$remote_host:$remote_dir/results/entropy_triggered_v2/" "$destination/"
echo "Fetched only the entropy-triggered v2 namespace into $destination."
