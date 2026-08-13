#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"
printf -v quoted_remote_dir '%q' "$remote_dir"

ssh "${ssh_args[@]}" "$remote_host" "cd $quoted_remote_dir && bash -s" <<'REMOTE'
set -euo pipefail

date --iso-8601=seconds
nvidia-smi \
  --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader

echo
echo "Active compute processes"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

echo
echo "Top local processes"
ps -eo pid,etime,pcpu,pmem,comm --sort=-pcpu | sed -n '1,12p'

echo
echo "Recent run artifacts"
if [[ -d runs ]]; then
  find runs -maxdepth 3 -type f -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' \
    | sort -nr \
    | cut -d' ' -f2- \
    | sed -n '1,20p'
else
  echo "No runs directory yet."
fi
REMOTE
