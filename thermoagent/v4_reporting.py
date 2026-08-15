"""Build the honest, development-only ThermoHITL v4 result package.

This module never runs validation, training, or holdout.  It summarizes the
formal development artifacts and records the prospective Gate 3 stop.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd


APPLICATIONS = ("commercial", "humanitarian", "utility_restoration")
METHODS = (
    "autonomy_no_operator",
    "thermohitl_v4_rule",
    "fixed_communication",
    "no_communication",
)


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_pandas(path, pd.DataFrame(list(rows)))


def _write_pandas(path: Path, frame: pd.DataFrame) -> None:
    """Write LF CSVs across both legacy and current pandas versions."""

    try:
        frame.to_csv(path, index=False, lineterminator="\n")
    except TypeError:  # pandas < 1.5 used the underscored keyword.
        frame.to_csv(path, index=False, line_terminator="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summaries(root: Path) -> Dict[str, pd.DataFrame]:
    return {
        path.parent.name: pd.read_csv(path)
        for path in sorted((root / "development").glob("development_gate_*/episode_summary.csv"))
    }


def _experimental_design(root: Path, summaries: Mapping[str, pd.DataFrame]) -> None:
    rows = []
    for stage, frame in summaries.items():
        rows.append({
            "evidence_stage": "development",
            "matrix": stage,
            "planner": "deterministic independent agents",
            "applications": ";".join(sorted(frame.application.unique())),
            "regimes": ";".join(sorted(frame.regime.unique())),
            "information_conditions": ";".join(sorted(frame.information_condition.unique())),
            "methods": ";".join(sorted(frame.method.unique())),
            "environment_seeds": ";".join(map(str, sorted(frame.environment_seed.unique()))),
            "independent_environment_seeds": int(frame.environment_seed.nunique()),
            "episodes": len(frame),
            "failed_episodes": int((frame.status != "complete").sum()),
            "confirmatory": False,
        })
    rows.append({
        "evidence_stage": "development_real_qwen_qualification",
        "matrix": "real_qwen_qualification",
        "planner": "Qwen independent structured planner",
        "applications": ";".join(APPLICATIONS),
        "regimes": "qualification scenarios",
        "information_conditions": "private observations",
        "methods": "real_qwen_agent_actionability",
        "environment_seeds": "24901;24902",
        "independent_environment_seeds": 2,
        "episodes": 6,
        "failed_episodes": 0,
        "confirmatory": False,
    })
    _write_frame(root / "tables" / "experimental_design.csv", rows)


def _gate_table(root: Path, gate_status: Mapping[str, Any]) -> None:
    labels = {
        "gate_1_engineering_integrity": "Engineering integrity",
        "gate_2_agent_actionability": "Autonomous-agent actionability",
        "gate_3_coordination_necessity": "Coordination necessity",
        "gate_4_human_causal_usefulness": "Simulated-operator causal usefulness",
        "gate_5_thermodynamic_incremental_value": "Thermodynamic incremental value",
        "gate_6_trigger_feasibility": "Trigger feasibility",
        "gate_7_mechanism_specificity": "Mechanism specificity",
    }
    rows = []
    for identifier, value in gate_status["gates"].items():
        passed = bool(value["passed"])
        rows.append({
            "gate": identifier,
            "description": labels[identifier],
            "passed": passed,
            "evidence_stage": "development",
            "disposition": "qualified" if passed else "failed; stop before validation",
            "confirmatory": False,
        })
    _write_frame(root / "tables" / "gate_outcomes.csv", rows)


def _actionability_table(
    root: Path,
    summaries: Mapping[str, pd.DataFrame],
    qwen: Mapping[str, Any],
) -> None:
    frame = summaries["development_gate_human"]
    rows = []
    for application, group in frame.groupby("application", sort=True):
        attempts = int(group.structured_attempts.sum())
        accepted = int(group.material_actions_accepted.sum())
        rows.append({
            "evidence_type": "deterministic_mock_agent",
            "application": application,
            "episodes": len(group),
            "structured_attempts": attempts,
            "first_pass_validity": float(group.first_pass_valid.sum() / max(attempts, 1)),
            "validity_after_one_repair": float(group.valid_after_repair.sum() / max(attempts, 1)),
            "material_actions_accepted": accepted,
            "accepted_to_next_stage_rate": float(group.material_actions_next_stage.sum() / max(accepted, 1)),
            "accepted_to_service_rate": float(group.material_actions_reached_service.sum() / max(accepted, 1)),
            "confirmatory": False,
        })
    for application, value in qwen["applications"].items():
        rows.append({
            "evidence_type": "real_qwen_agent",
            "application": application,
            "episodes": int(value["episodes"]),
            "structured_attempts": int(value["episodes"]),
            "first_pass_validity": float(value["first_pass_validity"]),
            "validity_after_one_repair": float(value["validity_after_one_repair"]),
            "material_actions_accepted": int(value["episodes"]),
            "accepted_to_next_stage_rate": float(value["accepted_to_next_stage"]),
            "accepted_to_service_rate": float(value["accepted_to_service"]),
            "confirmatory": False,
        })
    _write_frame(root / "tables" / "actionability.csv", rows)


def _causal_chain_table(root: Path, summaries: Mapping[str, pd.DataFrame]) -> None:
    frame = summaries["development_gate_human"]
    rows = []
    for (application, method), group in frame.groupby(["application", "method"], sort=True):
        rows.append({
            "evidence_stage": "development",
            "application": application,
            "method": method,
            "episodes": len(group),
            "operator_requests": int(group.operator_requests.sum()),
            "operator_interventions": int(group.operator_interventions.sum()),
            "commitment_changes": int(group.commitment_changes.sum()),
            "accepted_material_actions": int(group.material_actions_accepted.sum()),
            "next_stage_actions": int(group.material_actions_next_stage.sum()),
            "demand_or_service_arrivals": int(group.material_actions_reached_service.sum()),
            "primary_outcome_changes": int(group.beneficial_interventions.sum() + group.harmful_interventions.sum()),
            "beneficial_interventions": int(group.beneficial_interventions.sum()),
            "harmful_interventions": int(group.harmful_interventions.sum()),
            "complete_causal_chains": int(group.complete_causal_chains.sum()),
            "confirmatory": False,
        })
    _write_frame(root / "tables" / "causal_chain_accounting.csv", rows)


def _trigger_table(root: Path, summaries: Mapping[str, pd.DataFrame]) -> None:
    frame = summaries["development_gate_trigger"]
    rows = []
    for keys, group in frame.groupby(
        ["application", "regime", "information_condition"], sort=True
    ):
        application, regime, information = keys
        requests = group.operator_requests.astype(float) > 0
        rows.append({
            "evidence_stage": "development",
            "application": application,
            "regime": regime,
            "information_condition": information,
            "independent_panels": len(group),
            "activation_rate": float(requests.mean()),
            "timely_activation_rate": float(group.timely_activation.astype(str).str.lower().eq("true").mean()),
            "pre_disruption_false_activation_rate": float(group.pre_disruption_false_activation.astype(float).mean()),
            "nominal_false_activation_rate": float(group.nominal_false_activation.astype(float).mean()),
            "mean_operator_requests": float(group.operator_requests.mean()),
            "mean_operator_interventions": float(group.operator_interventions.mean()),
            "mean_operator_minutes": float(group.operator_minutes.mean()),
            "mean_active_agent_epoch_fraction": float(group.communication_active_agent_epoch_fraction.mean()),
            "low_confidence_decisions": int(group.low_confidence_operator_decisions.sum()),
            "confirmatory": False,
        })
    _write_frame(root / "tables" / "trigger_timing_and_burden.csv", rows)


def _compute_table(
    root: Path,
    summaries: Mapping[str, pd.DataFrame],
    qwen: Mapping[str, Any],
) -> None:
    numeric = (
        "agent_messages", "agent_message_bytes", "thermodynamic_sketch_messages",
        "thermodynamic_sketch_bytes", "tool_calls", "llm_calls", "prompt_tokens",
        "generated_tokens", "llm_latency_seconds", "wall_clock_seconds",
        "operator_interventions", "operator_minutes", "operator_workload_auc",
    )
    rows = []
    for stage, frame in summaries.items():
        row: Dict[str, Any] = {
            "evidence_stage": "development_deterministic",
            "workflow": stage,
            "episodes": len(frame),
            "single_gpu_hours": 0.0,
            "estimated_gpu_cost_usd_at_0_34_per_hour": 0.0,
        }
        row.update({column: float(frame[column].sum()) for column in numeric})
        rows.append(row)
    load_seconds_from_log = 29.44
    measured_gpu_seconds = load_seconds_from_log + float(qwen["llm_latency_seconds"])
    rows.append({
        "evidence_stage": "development_real_qwen_qualification",
        "workflow": "real_qwen_qualification",
        "episodes": int(qwen["episodes"]),
        "agent_messages": 0,
        "agent_message_bytes": 0,
        "thermodynamic_sketch_messages": 0,
        "thermodynamic_sketch_bytes": 0,
        "tool_calls": int(qwen["episodes"]),
        "llm_calls": int(qwen["llm_calls"]),
        "prompt_tokens": int(qwen["prompt_tokens"]),
        "generated_tokens": int(qwen["generated_tokens"]),
        "llm_latency_seconds": float(qwen["llm_latency_seconds"]),
        "wall_clock_seconds": measured_gpu_seconds,
        "operator_interventions": 0,
        "operator_minutes": 0,
        "operator_workload_auc": 0,
        "single_gpu_hours": measured_gpu_seconds / 3600.0,
        "estimated_gpu_cost_usd_at_0_34_per_hour": measured_gpu_seconds / 3600.0 * 0.34,
    })
    _write_frame(root / "tables" / "compute_and_communication_accounting.csv", rows)
    totals = pd.DataFrame(rows)
    _write_json(root / "reproducibility" / "v4_actual_compute.json", {
        "evidence_stage": "development",
        "deterministic_episodes": int(sum(len(value) for value in summaries.values())),
        "real_qwen_episodes": int(qwen["episodes"]),
        "real_qwen_calls": int(qwen["llm_calls"]),
        "prompt_tokens": int(qwen["prompt_tokens"]),
        "generated_tokens": int(qwen["generated_tokens"]),
        "measured_generation_latency_seconds": float(qwen["llm_latency_seconds"]),
        "model_load_seconds_from_progress_log": load_seconds_from_log,
        "measured_single_gpu_hours": measured_gpu_seconds / 3600.0,
        "estimated_cost_usd_at_0_34_per_gpu_hour": measured_gpu_seconds / 3600.0 * 0.34,
        "deterministic_cpu_wall_seconds_summed_across_episodes": float(
            sum(value.wall_clock_seconds.sum() for value in summaries.values())
        ),
        "qualification_caveat": "GPU accounting covers the recorded 29.44-second model load and measured generation latency; setup checks used negligible GPU compute.",
        "validation_training_holdout_gpu_hours": 0.0,
    })


def _qwen_table(root: Path, qwen: Mapping[str, Any]) -> None:
    rows = []
    for case in qwen["cases"]:
        rows.append({
            "evidence_stage": "development_real_qwen_qualification",
            "application": case["application"],
            "environment_seed": case["environment_seed"],
            "agent_id": case["agent_id"],
            "role": case["role"],
            "tool": case["executed_tool"],
            "first_pass_valid": case["first_pass_valid"],
            "valid_after_one_repair": case["valid_after_one_repair"],
            "repair_attempted": case["repair_attempted"],
            "material_action_accepted": case["material_action_accepted"],
            "material_action_next_stage": case["material_action_next_stage"],
            "material_action_reached_service": case["material_action_reached_service"],
            "prompt_tokens": case["prompt_tokens"],
            "generated_tokens": case["generated_tokens"],
            "latency_seconds": case["latency_seconds"],
            "model_identifier": qwen["model_identifier"],
            "model_revision": qwen["model_revision"],
            "precision": qwen["precision"],
        })
    _write_frame(root / "tables" / "real_qwen_qualification.csv", rows)


def _failure_table(root: Path) -> None:
    rows = [
        {
            "stage": "implementation_pilot_v1_coordination",
            "count": 3,
            "type": "development implementation failure",
            "reason": "multi-scope agent incident mismatch (KeyError: communications_zone)",
            "included_in_formal_analysis": False,
            "disposition": "retained; mechanics corrected before formal development",
        },
        {
            "stage": "real_qwen_qualification_attempt",
            "count": 1,
            "type": "infrastructure/pre-execution",
            "reason": "interrupted provenance metadata command before episode output",
            "included_in_formal_analysis": False,
            "disposition": "retained in setup log; no outcome was generated",
        },
        {
            "stage": "real_qwen_qualification_attempt",
            "count": 1,
            "type": "infrastructure/pre-execution",
            "reason": "configured cache path did not contain the already cached model",
            "included_in_formal_analysis": False,
            "disposition": "retained in setup log; corrected cache path, no package change",
        },
        {
            "stage": "formal_development",
            "count": 0,
            "type": "episode failure",
            "reason": "none",
            "included_in_formal_analysis": True,
            "disposition": "all 1,584 deterministic episodes completed",
        },
    ]
    _write_frame(root / "tables" / "failed_runs.csv", rows)


def _hypothesis_table(root: Path) -> None:
    rows = [
        ("H1", "Communication materially useful in fragmented regimes", "unsupported at the frozen gate", "Utility restoration improved 4.43%, below the 5% target; Gate 3 failed."),
        ("H2", "Bounded simulated-operator intervention causally improves at least two applications", "supported in development", "Humanitarian and utility restoration passed; commercial did not."),
        ("H3", "KPI plus entropy/disagreement improves causal utility in humanitarian and utility", "supported in development", "Both cluster-bootstrap lower bounds exceeded zero at budget one."),
        ("H4", "Benefit increases with information fragmentation", "supported in development", "Fragmented gains were positive; globally public gains were zero."),
        ("H5", "Thermodynamic triage improves effort/safety without >2% loss degradation", "mixed development evidence", "No harmful selections; humanitarian loss degradation was 1.78%, utility improved 1.10%; effort superiority was not confirmatory."),
        ("H6", "Distributed estimates degrade predictably and support safe abstention", "mixed development evidence", "Partition error was measured, but no low-confidence operator decision occurred, so abstention was not exercised."),
        ("H7", "Complete causal chain is observed", "supported in development", "Complete chains: 2 commercial, 32 humanitarian, 34 utility restoration."),
        ("H8", "Commercial logistics is a KPI-sufficient boundary condition", "supported as development boundary evidence", "Commercial thermodynamic incremental utility was exactly zero."),
    ]
    _write_frame(root / "tables" / "hypothesis_outcomes.csv", [
        {"hypothesis": key, "short_statement": statement, "status": status,
         "evidence": evidence, "evidence_stage": "development", "confirmatory": False}
        for key, statement, status, evidence in rows
    ])


def _stage_dispositions(root: Path) -> None:
    common = {
        "status": "prospectively_not_run",
        "reason": "Required development Gate 3 (coordination necessity) failed.",
        "failed_gate": "gate_3_coordination_necessity",
        "failed_application": "utility_restoration",
        "observed_aggregate_relative_loss_reduction": 0.04431717249899478,
        "required_aggregate_relative_loss_reduction": 0.05,
        "outcomes_observed": False,
        "fabricated_rows": 0,
    }
    for stage in ("validation", "training", "holdout"):
        payload = {"stage": stage, **common}
        _write_json(root / stage / "NOT_RUN.json", payload)
        (root / stage / "README.md").write_text(
            "# %s disposition\n\n"
            "This stage was **prospectively not run**. Required development "
            "Gate 3 failed because fixed communication reduced utility-restoration "
            "loss by 4.43%%, below the frozen 5%% aggregate threshold. No outcomes, "
            "RL checkpoints, or placeholder numerical result rows exist here.\n" % stage.title(),
            encoding="utf-8",
        )
    _write_frame(root / "tables" / "stage_disposition.csv", [
        {"stage": stage, **common} for stage in ("validation", "training", "holdout")
    ])


def _protocol_provenance(repository: Path, root: Path) -> None:
    config = repository / "configs" / "human_operator_v4_development.yaml"
    note = repository / "notes" / "37_v4_prospective_protocol.md"
    manifests = sorted((root / "manifests").glob("development_gate_*/*.json"))
    manifest = _json(manifests[0]) if manifests else {}
    value = {
        "v3_scientific_snapshot": "3f844966930b1cfb5a43bdf3a4d3e744391d1018",
        "v2_scientific_snapshot": "c0aa6fe6c98cbce0cdd5e40a0f720a98f5facbe6",
        "protocol_config": str(config.relative_to(repository)),
        "protocol_config_sha256": _sha256(config),
        "protocol_note": str(note.relative_to(repository)),
        "protocol_note_sha256": _sha256(note),
        "development_source_checksum": manifest.get("source_checksum"),
        "development_execution_commit": manifest.get("git_commit"),
        "development_execution_dirty_tree": manifest.get("dirty_tree"),
        "development_manifest_count": len(manifests),
        "holdout_frozen": False,
        "reason_no_holdout_freeze": "Prospective Gate 3 stop before validation.",
    }
    _write_json(root / "protocol" / "frozen_development_protocol_checksums.json", value)
    (root / "protocol" / "README.md").write_text(
        "# Frozen v4 development protocol\n\n"
        "The authoritative machine-readable protocol is "
        "[`configs/human_operator_v4_development.yaml`](../../../configs/human_operator_v4_development.yaml); "
        "the rationale and hypotheses are in "
        "[`notes/37_v4_prospective_protocol.md`](../../../notes/37_v4_prospective_protocol.md). "
        "Checksums and the execution source checksum are recorded in "
        "`frozen_development_protocol_checksums.json`. Gate 3 stopped the study "
        "before validation, training, or holdout freeze.\n",
        encoding="utf-8",
    )


def _infer(path: Path) -> tuple[str, str, str, str]:
    value = path.as_posix()
    application = next((item for item in APPLICATIONS if item in value), "all_or_not_applicable")
    method = next((item for item in METHODS if item in value), "multiple_or_not_applicable")
    stage = path.parts[0] if path.parts else "root"
    suffix = "".join(path.suffixes).lower()
    artifact_type = {
        ".csv": "table_or_data", ".json": "structured_metadata", ".jsonl.gz": "compressed_event_ledger",
        ".pdf": "vector_figure", ".png": "figure_preview", ".svg": "vector_dashboard_export",
        ".md": "documentation", ".log": "log", ".xml": "test_report",
        ".tar.gz": "compressed_raw_archive",
    }.get(suffix, "artifact")
    description = path.stem.replace("_", " ").replace("-", " ")
    return artifact_type, stage, application, method + ": " + description


def build_index(root: Path) -> Dict[str, Any]:
    destination = root / "INDEX.csv"
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file() and value != destination):
        relative = path.relative_to(root)
        artifact_type, stage, application, method_description = _infer(relative)
        method, description = method_description.split(": ", 1)
        rows.append({
            "path": relative.as_posix(),
            "artifact_type": artifact_type,
            "stage": stage,
            "description": description,
            "application": application,
            "method": method,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "indexed_artifacts_excluding_self": len(rows),
        "index_path": str(destination),
        "self_exclusion": "INDEX.csv cannot contain its own stable checksum and is intentionally excluded",
    }


def build_v4_reporting(repository: Path) -> Dict[str, Any]:
    root = repository / "results" / "human_operator_v4"
    summaries = _summaries(root)
    gate_status = _json(root / "development" / "gate_status.json")
    qwen = _json(root / "development" / "real_qwen_qualification.json")
    _experimental_design(root, summaries)
    _gate_table(root, gate_status)
    _actionability_table(root, summaries, qwen)
    _causal_chain_table(root, summaries)
    _trigger_table(root, summaries)
    _compute_table(root, summaries, qwen)
    _qwen_table(root, qwen)
    _failure_table(root)
    _hypothesis_table(root)
    _stage_dispositions(root)
    _protocol_provenance(repository, root)
    # Existing cluster-aware outputs are promoted into the table namespace.
    for source, name in (
        (root / "statistics" / "coordination_paired_effects.csv", "coordination_paired_effects.csv"),
        (root / "statistics" / "human_causal_paired_effects.csv", "human_causal_paired_effects.csv"),
        (root / "statistics" / "conditional_permutation_test.csv", "mechanism_specificity.csv"),
    ):
        _write_pandas(root / "tables" / name, pd.read_csv(source))
    return {
        "gate_decision": gate_status["decision"],
        "tables_generated": len(list((root / "tables").glob("*.csv"))),
        **build_index(root),
    }
