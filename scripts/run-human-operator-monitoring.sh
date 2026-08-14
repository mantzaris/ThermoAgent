#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/human-operator-common.sh"
run_human_command analysis/monitoring.log monitoring
run_human_command analysis/gates_after_monitoring.log gates
