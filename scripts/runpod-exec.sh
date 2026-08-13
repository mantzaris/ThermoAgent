#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"
printf -v quoted_remote_dir '%q' "$remote_dir"

if (( $# == 0 )); then
  ssh "${ssh_args[@]}" -t "$remote_host" "cd $quoted_remote_dir && exec bash -l"
  exit 0
fi

printf -v quoted_command '%q ' "$@"
ssh "${ssh_args[@]}" "$remote_host" "cd $quoted_remote_dir && $quoted_command"
