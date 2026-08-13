#!/usr/bin/env bash
# Shared non-secret SSH configuration. Source this file; do not execute it.

remote_host="${THERMO_REMOTE_HOST:-runpod-thermo}"
remote_port="${THERMO_REMOTE_PORT:-}"
remote_identity="${THERMO_REMOTE_IDENTITY:-}"
remote_known_hosts="${THERMO_REMOTE_KNOWN_HOSTS:-}"
remote_dir="${THERMO_REMOTE_DIR:-/workspace/ThermoAgent}"

ssh_args=()
if [[ -n "$remote_port" ]]; then
  [[ "$remote_port" =~ ^[0-9]+$ ]] || {
    echo "THERMO_REMOTE_PORT must be numeric" >&2
    exit 2
  }
  ssh_args+=( -p "$remote_port" )
fi
if [[ -n "$remote_identity" ]]; then
  [[ -f "$remote_identity" ]] || {
    echo "THERMO_REMOTE_IDENTITY does not name a readable file" >&2
    exit 2
  }
  ssh_args+=( -i "$remote_identity" )
fi
if [[ -n "$remote_known_hosts" ]]; then
  [[ -f "$remote_known_hosts" ]] || {
    echo "THERMO_REMOTE_KNOWN_HOSTS does not name a readable file" >&2
    exit 2
  }
  ssh_args+=( -o "UserKnownHostsFile=$remote_known_hosts" -o "StrictHostKeyChecking=yes" )
fi

rsync_shell=(ssh)
rsync_shell+=("${ssh_args[@]}")
printf -v rsync_shell_quoted '%q ' "${rsync_shell[@]}"
