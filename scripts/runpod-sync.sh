#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-common.sh
source "$script_dir/runpod-common.sh"

command -v rsync >/dev/null || {
  echo "rsync is required on the local computer" >&2
  exit 1
}

ssh "${ssh_args[@]}" "$remote_host" mkdir -p "$remote_dir"

# This is intentionally a one-way, non-deleting deployment. Runtime artifacts
# and any local authentication material are never sent to the RunPod.
rsync \
  --archive \
  --no-owner \
  --no-group \
  --compress \
  --human-readable \
  --itemize-changes \
  --rsh="$rsync_shell_quoted" \
  --exclude='.git/' \
  --exclude='.codex/' \
  --exclude='.agents/' \
  --exclude='.ssh/' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='**/.env' \
  --exclude='**/.env.*' \
  --exclude='*.pem' \
  --exclude='*.key' \
  --exclude='id_rsa*' \
  --exclude='id_ed25519*' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='runs/' \
  --exclude='results/' \
  "$repo_dir/" "$remote_host:$remote_dir/"

echo "Synchronized $repo_dir to $remote_host:$remote_dir (no remote files deleted)."
