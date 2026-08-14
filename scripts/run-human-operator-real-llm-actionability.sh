#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/human-operator-common.sh"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
real_stage="${THERMO_HUMAN_REAL_STAGE:-development_real_llm_actionability}"
run_human_command "diagnostics/${real_stage}.log" real-llm-actionability \
  --stage "$real_stage" --seeds "${THERMO_HUMAN_REAL_SEEDS:-13101,13102}"
