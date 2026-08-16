#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$repo_dir/scripts"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"

destination="$repo_dir/results/generalized_entropic_consensus_v6"
mkdir -p "$destination"
rsync --archive --no-owner --no-group --compress --human-readable \
  --itemize-changes --rsh="$rsync_shell_quoted" \
  "$remote_host:$remote_dir/results/generalized_entropic_consensus_v6/" \
  "$destination/"
echo "Fetched only the additive generalized_entropic_consensus_v6 namespace into $destination."
