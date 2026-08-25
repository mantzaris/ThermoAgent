#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "qwen" && "$1" != "granite" ) ]]; then
  echo "usage: $0 qwen|granite" >&2
  exit 2
fi
if [[ "${THERMO_V15_ENABLE_LLM:-0}" != "1" ]]; then
  echo "set THERMO_V15_ENABLE_LLM=1 to authorize frozen-model reconstruction" >&2
  exit 2
fi

EXPECTED_COMMIT="b309f0ab76cb24377de5872eebc811582af1f43f"
EXPECTED_SEMANTIC_SOURCE="f8d4fa546ba46a42cd4234dd8af6ad60309c231f2997e10d0d25830f6dddb2f2"
EXPECTED_LEGACY_SOURCE="ec9f26223a335558b2789ebd59ee3c3fa0f9e7d1b815fd9b09a1e1960af55e78"
FROZEN_ROOT="${THERMO_V15_FROZEN_ROOT:-/workspace/ThermoAgent-v15-frozen-b309f0ab}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/ThermoAgent/.venv/bin/python}"
export THERMO_V15_ARTIFACT_ROOT="${THERMO_V15_ARTIFACT_ROOT:-/workspace/ThermoAgent-v15-reconstruction-b309f0ab}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HOME="${HF_HOME:-/workspace/ThermoAgent-v15-model-cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0

[[ -x "$PYTHON_BIN" ]] || {
  echo "pinned V15 Python environment is unavailable: $PYTHON_BIN" >&2
  exit 2
}
[[ -d "$FROZEN_ROOT/.git" ]] || {
  echo "frozen reconstruction checkout is unavailable: $FROZEN_ROOT" >&2
  exit 2
}
[[ "$(git -C "$FROZEN_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "frozen reconstruction checkout is not at the committed V15 reference" >&2
  exit 2
}
[[ -z "$(git -C "$FROZEN_ROOT" status --porcelain --untracked-files=no)" ]] || {
  echo "frozen reconstruction checkout has tracked modifications" >&2
  exit 2
}

cd "$FROZEN_ROOT"
"$PYTHON_BIN" results/collective_agent_statmech_v15/reproducibility/verify_source_checksum.py >/dev/null

"$PYTHON_BIN" - "$1" "$EXPECTED_SEMANTIC_SOURCE" "$EXPECTED_LEGACY_SOURCE" <<'PY'
import copy
import json
import os
import sys
from pathlib import Path

from thermoagent.statmech_llm_v15 import experiment
from thermoagent.statmech_llm_v15.workflow import (
    artifact_root,
    atomic_json,
    execution_source_checksum,
    load_yaml,
    repository_root,
    sha256_file,
    utc_now,
)

model_key, expected_semantic, expected_legacy = sys.argv[1:]
repository = repository_root()
protocol_path = repository / "configs/statmech_v15/protocol_frozen.yaml"
protocol = load_yaml(protocol_path)
observed_semantic = execution_source_checksum(repository)
if observed_semantic != expected_semantic:
    raise RuntimeError(
        "clean reconstruction source differs from the audited semantic digest"
    )
if str(protocol["provenance"]["execution_source_sha256"]) != expected_legacy:
    raise RuntimeError("frozen protocol does not contain the audited legacy digest")
audit_path = (
    repository
    / "results/collective_agent_statmech_v15/reproducibility/verification_clean.json"
)
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("status") != "passed":
    raise RuntimeError("clean-source provenance audit did not pass")
if audit.get("semantic_source_sha256") != expected_semantic:
    raise RuntimeError("semantic-source audit mismatch")
if audit.get("reconstructed_legacy_execution_source_sha256") != expected_legacy:
    raise RuntimeError("legacy-source reconstruction mismatch")

# The historical enumerator accidentally included the digest of one ignored
# root-level .pyc file.  The verified clean semantic tree is identical in all
# scientific source.  Patch only the in-process checksum lookup so the frozen
# runner records its historical digest without fabricating or staging bytecode.
def audited_legacy_checksum(candidate: Path) -> str:
    if execution_source_checksum(candidate) != expected_semantic:
        raise RuntimeError("source changed after provenance verification")
    return expected_legacy

experiment.execution_source_checksum = audited_legacy_checksum
authorized_gpu_hours_text = os.environ.get("THERMO_V15_AUTHORIZED_GPU_HOURS")
authorized_gpu_hours = None
if authorized_gpu_hours_text is not None:
    authorized_gpu_hours = float(authorized_gpu_hours_text)
    frozen_gpu_hours = float(protocol["compute"]["hard_generation_gpu_hours"])
    if authorized_gpu_hours < frozen_gpu_hours:
        raise RuntimeError("authorized reconstruction ceiling cannot lower the freeze")
    if authorized_gpu_hours > 50.0:
        raise RuntimeError("reconstruction wrapper refuses an authorization above 50 hours")
    frozen_budget_check = experiment._assert_next_panel_within_compute_budget

    def authorized_budget_check(panel_root, panel, protocol_value):
        operational_protocol = copy.deepcopy(protocol_value)
        operational_protocol["compute"]["hard_generation_gpu_hours"] = authorized_gpu_hours
        return frozen_budget_check(panel_root, panel, operational_protocol)

    experiment._assert_next_panel_within_compute_budget = authorized_budget_check
atomic_json(
    {
        "recorded_at_utc": utc_now(),
        "commit": "b309f0ab76cb24377de5872eebc811582af1f43f",
        "protocol_sha256": sha256_file(protocol_path),
        "clean_semantic_source_sha256": observed_semantic,
        "audited_legacy_execution_source_sha256": expected_legacy,
        "compatibility_scope": (
            "in_process_checksum_lookup_and_authorized_budget_guard_only"
            if authorized_gpu_hours is not None
            else "in_process_checksum_lookup_only"
        ),
        "scientific_source_or_protocol_modified": False,
        "ignored_bytecode_recreated": False,
        "frozen_generation_gpu_hour_ceiling": float(
            protocol["compute"]["hard_generation_gpu_hours"]
        ),
        "authorized_reconstruction_gpu_hour_ceiling": authorized_gpu_hours,
        "operational_ceiling_override_scope": (
            "external reconstruction budget guard only"
            if authorized_gpu_hours is not None
            else None
        ),
        "decision_prompt_token_and_invalidity_ceilings_modified": False,
    },
    artifact_root() / "reproducibility/reconstruction_source_compatibility.json",
)
result = experiment.run_formal_model(repository, model_key)
print(json.dumps(result, indent=2, sort_keys=True))
PY
