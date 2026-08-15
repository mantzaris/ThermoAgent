"""Build the complete development-only ThermoHITL V5 result package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import yaml

from .v5_experiments import atomic_json, write_csv


APPLICATIONS = ("commercial", "humanitarian", "utility_restoration")


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_table(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_frame(destination, frame)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=False, lineterminator="\n")
    except TypeError:  # pandas < 1.5
        frame.to_csv(path, index=False, line_terminator="\n")


def _experimental_design(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for stage in (
        "pilot", "pilot_iteration_2", "pilot_iteration_3", "pilot_iteration_4",
        "development_primary", "development_primary_v2", "sketch_ablation",
    ):
        report = _json(root / "development" / stage / "run_report.json")
        rows.append({
            "stage": stage,
            "evidence_type": "deterministic independent-agent development",
            "eligible_for_inference": stage == "development_primary_v2" or stage == "sketch_ablation",
            "episodes": report["episodes"], "candidate_rows": report["candidate_rows"],
            "failed_runs": report["failures"], "simulated_operator": True,
            "confirmatory": False,
        })
    qwen = _json(root / "development" / "real_qwen_qualification.json")
    rows.append({
        "stage": "real_qwen_qualification", "evidence_type": "real Qwen independent-agent qualification",
        "eligible_for_inference": False, "episodes": qwen["episodes"], "candidate_rows": qwen["decision_epochs"],
        "failed_runs": 0, "simulated_operator": False, "confirmatory": False,
    })
    training = _json(root / "training" / "training_summary.json")
    rows.append({
        "stage": "multiseed_decentralized_rl", "evidence_type": "development-only RL policy training",
        "eligible_for_inference": False, "episodes": len(training["training_seeds"]) * 2,
        "candidate_rows": sum(value["evaluation_decision_epochs"] for value in pd.read_csv(root / "training" / "seed_manifest.csv").to_dict("records")),
        "failed_runs": training["failed_seeds"], "simulated_operator": False, "confirmatory": False,
    })
    rows.extend([
        {"stage": "validation", "evidence_type": "prospectively locked—not run", "eligible_for_inference": False,
         "episodes": 0, "candidate_rows": 0, "failed_runs": 0, "simulated_operator": True, "confirmatory": False},
        {"stage": "sealed_holdout", "evidence_type": "prospectively locked—not run", "eligible_for_inference": False,
         "episodes": 0, "candidate_rows": 0, "failed_runs": 0, "simulated_operator": True, "confirmatory": False},
    ])
    frame = pd.DataFrame(rows)
    _write_frame(root / "tables" / "experimental_design.csv", frame)
    return frame


def _actionability(root: Path) -> None:
    summary = pd.read_csv(root / "development" / "development_primary_v2" / "episode_summary.csv")
    candidates = pd.read_csv(root / "development" / "development_primary_v2" / "candidate_interventions.csv")
    qwen = _json(root / "development" / "real_qwen_qualification.json")
    rows = []
    for application, group in summary.groupby("application"):
        attempted = int(group.incidents.sum()); accepted = int(group.fixed_accepted_actions.sum())
        service = int(group.fixed_service_reaching_actions.sum())
        rows.append({
            "evidence_type": "deterministic_independent_agent", "application": application,
            "episodes": len(group), "decision_epochs": attempted, "first_pass_validity": 1.0,
            "validity_after_one_repair": 1.0, "material_acceptance": accepted / attempted,
            "accepted_to_next_stage": service / accepted, "service_reaching": service / attempted,
            "action_diversity": int(candidates[candidates.application == application].autonomous_action.nunique()),
            "private_evidence_action_divergence": None,
        })
    for application, value in qwen["applications"].items():
        rows.append({"evidence_type": "real_qwen_agent", "application": application, **value})
    write_csv(root / "tables" / "agent_actionability.csv", rows)


def _intervention_accounting(root: Path) -> None:
    candidates = pd.read_csv(root / "development" / "development_primary_v2" / "candidate_interventions.csv")
    rows = []
    for (application, condition), group in candidates.groupby(["application", "information_condition"]):
        rows.append({
            "application": application, "information_condition": condition,
            "counterfactual_candidates": len(group), "beneficial": int(group.beneficial.sum()),
            "neutral": int((group.causal_effect.abs() <= 1e-12).sum()), "harmful": int(group.harmful.sum()),
            "accepted_actions": int(group.accepted_action.sum()), "next_stage": int(group.reached_next_stage.sum()),
            "reached_service": int(group.reached_service.sum()), "changed_commitment": int(group.changed_commitment.sum()),
            "mean_effect": float(group.causal_effect.mean()), "minimum_effect": float(group.causal_effect.min()),
            "maximum_effect": float(group.causal_effect.max()),
        })
    write_csv(root / "tables" / "counterfactual_intervention_accounting.csv", rows)


def _compute(root: Path) -> None:
    qwen = _json(root / "development" / "real_qwen_qualification.json")
    training = _json(root / "training" / "training_summary.json")
    summary = pd.read_csv(root / "development" / "development_primary_v2" / "episode_summary.csv")
    sketch = pd.read_csv(root / "development" / "sketch_ablation" / "episode_summary.csv")
    manifest = pd.read_csv(root / "training" / "seed_manifest.csv")
    qwen_gpu_hours = float(qwen["wall_seconds_including_model_load"]) / 3600.0
    rows = [
        {
            "workflow": "formal_development_deterministic", "episodes": len(summary),
            "rl_decision_epochs": 0, "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0,
            "operational_messages": int(summary.operational_messages.sum()),
            "operational_bytes": int(summary.operational_bytes.sum()),
            "thermodynamic_sketch_messages": int(summary.sketch_messages.sum()),
            "thermodynamic_sketch_bytes": int(summary.sketch_bytes.sum()),
            "gpu_hours": 0.0, "estimated_gpu_cost_usd_at_0_34_per_hour": 0.0,
        },
        {
            "workflow": "sketch_ablation_deterministic", "episodes": len(sketch),
            "rl_decision_epochs": 0, "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0,
            "operational_messages": int(sketch.operational_messages.sum()),
            "operational_bytes": int(sketch.operational_bytes.sum()),
            "thermodynamic_sketch_messages": int(sketch.sketch_messages.sum()),
            "thermodynamic_sketch_bytes": int(sketch.sketch_bytes.sum()),
            "gpu_hours": 0.0, "estimated_gpu_cost_usd_at_0_34_per_hour": 0.0,
        },
        {
            "workflow": "real_qwen_qualification", "episodes": qwen["episodes"],
            "rl_decision_epochs": 0, "llm_calls": qwen["llm_calls"], "prompt_tokens": qwen["prompt_tokens"],
            "generated_tokens": qwen["generated_tokens"], "operational_messages": 0, "operational_bytes": 0,
            "thermodynamic_sketch_messages": 0, "thermodynamic_sketch_bytes": 0,
            "gpu_hours": qwen_gpu_hours, "estimated_gpu_cost_usd_at_0_34_per_hour": qwen_gpu_hours * 0.34,
        },
        {
            "workflow": "multiseed_rl_cpu", "episodes": len(manifest),
            "rl_decision_epochs": int(manifest.training_decision_epochs.sum() + manifest.evaluation_decision_epochs.sum()),
            "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0,
            "operational_messages": 0, "operational_bytes": 0,
            "thermodynamic_sketch_messages": 0, "thermodynamic_sketch_bytes": 0,
            "gpu_hours": 0.0, "estimated_gpu_cost_usd_at_0_34_per_hour": 0.0,
        },
    ]
    write_csv(root / "tables" / "compute_communication_and_token_accounting.csv", rows)
    atomic_json(root / "reproducibility" / "environment" / "compute_summary.json", {
        "additional_v5_single_gpu_hours": qwen_gpu_hours,
        "estimated_gpu_cost_usd_at_0_34_per_hour": qwen_gpu_hours * 0.34,
        "qwen_calls": qwen["llm_calls"], "prompt_tokens": qwen["prompt_tokens"],
        "generated_tokens": qwen["generated_tokens"],
        "rl_cpu_wall_seconds_sum": float(manifest.wall_seconds.sum()),
        "projected_cap_gpu_hours": 50.0, "projected_cost_cap_usd": 40.0,
    })


def _hypotheses(root: Path, gates: Mapping[str, Any]) -> None:
    statuses = [
        ("H1", "Coordination is consequential in fragmented primary applications", "supported in development", "gate_3_coordination_necessity"),
        ("H2", "Bounded simulated-operator intervention causally improves outcomes", "supported in development", "gate_4_human_causal_usefulness"),
        ("H3", "KPI plus entropy/disagreement beats KPI-only in both primary applications", "unsupported", "gate_5_thermodynamic_incremental_value"),
        ("H4", "Benefit increases under private fragmentation", "unsupported", "gate_7_mechanism_specificity"),
        ("H5", "Thermodynamic triage reduces harm without service degradation", "mixed; abstention helps but primary triage does not", "gate_8_safety_and_abstention"),
        ("H6", "Distributed estimates support useful bounded-cost monitoring", "mixed; compression passes, causal value fails", "gate_9_communication_cost_feasibility"),
        ("H7", "Complete causal chains occur", "supported for bounded-oracle development control", "gate_4_human_causal_usefulness"),
        ("H8", "Commercial is a KPI-sufficient boundary", "consistent but not confirmatory", "gate_5_thermodynamic_incremental_value"),
    ]
    rows = [{"hypothesis": h, "statement": text, "outcome": outcome, "supporting_gate": gate,
             "gate_passed": gates["gates"][gate]["passed"], "evidence_stage": "development"}
            for h, text, outcome, gate in statuses]
    write_csv(root / "tables" / "hypothesis_outcomes.csv", rows)


def _stage_disposition(root: Path, gates: Mapping[str, Any]) -> None:
    payload = {
        "development": {"status": "complete", "episodes": 840, "eligible": True},
        "multiseed_training": {"status": "complete_development_only", "policy_runs": 10, "eligible": True},
        "validation": {"status": "not_run_prospectively_locked", "episodes": 0,
                       "reason": "not all ten development gates passed"},
        "sealed_holdout": {"status": "not_run_prospectively_locked", "episodes": 0,
                           "reason": "validation was not unlocked"},
        "all_progression_gates_passed": gates["all_progression_gates_passed"],
    }
    atomic_json(root / "manifests" / "stage_disposition.json", payload)
    (root / "validation" / "NOT_RUN.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "validation" / "NOT_RUN.md").write_text(
        "# Validation not run\n\nValidation remained prospectively locked because required V5 development gates failed. No validation outcomes exist.\n",
        encoding="utf-8",
    )
    (root / "holdout" / "NOT_RUN.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "holdout" / "NOT_RUN.md").write_text(
        "# Sealed holdout not run\n\nThe sealed V5 holdout was never opened because validation was not unlocked. No holdout outcomes exist.\n",
        encoding="utf-8",
    )


def _infer(relative: Path) -> Tuple[str, str, str]:
    path = relative.as_posix()
    suffixes = "".join(relative.suffixes[-2:])
    artifact = {
        ".csv": "table_or_data", ".json": "structured_metadata", ".jsonl.gz": "compressed_event_ledger",
        ".pdf": "vector_figure", ".png": "figure_preview", ".svg": "dashboard_vector_export",
        ".md": "documentation", ".log": "log", ".pt": "small_rl_checkpoint",
    }.get(suffixes, {".csv": "table_or_data", ".json": "structured_metadata", ".pdf": "vector_figure",
                     ".png": "figure_preview", ".svg": "dashboard_vector_export", ".md": "documentation",
                     ".log": "log", ".pt": "small_rl_checkpoint"}.get(relative.suffix, "artifact"))
    stage = path.split("/", 1)[0] if "/" in path else "root"
    application = next((value for value in APPLICATIONS if value in path), "all")
    return artifact, stage, application


def build_index(root: Path) -> Dict[str, Any]:
    destination = root / "INDEX.csv"
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file() and value != destination):
        relative = path.relative_to(root)
        artifact, stage, application = _infer(relative)
        rows.append({
            "path": relative.as_posix(), "artifact_type": artifact, "stage": stage,
            "description": relative.stem.replace("_", " ").replace("-", " "),
            "application": application, "method": "all" if application == "all" else "mixed",
            "size_bytes": path.stat().st_size, "sha256": _sha256(path),
        })
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return {"indexed_artifacts_excluding_self": len(rows), "index_path": str(destination)}


def build_v5_reporting(repository: Path) -> Dict[str, Any]:
    root = repository / "results" / "human_operator_v5"
    (root / "tables").mkdir(parents=True, exist_ok=True)
    _experimental_design(root)
    _actionability(root)
    _intervention_accounting(root)
    _compute(root)
    gates = _json(root / "development" / "gate_status.json")
    _hypotheses(root, gates)
    _stage_disposition(root, gates)
    for name in (
        "primary_incremental_value.csv", "coordination_necessity.csv", "human_causal_usefulness.csv",
        "fragmentation_interaction.csv", "low_consensus_abstention.csv", "trigger_feasibility.csv",
        "sketch_communication_accounting.csv", "sketch_policy_comparisons.csv", "shortcut_and_support_diagnostics.csv",
    ):
        _copy_table(root / "statistics" / name, root / "tables" / name)
    _copy_table(root / "training" / "seed_manifest.csv", root / "tables" / "rl_training_seeds.csv")
    return {"gate_disposition": gates["disposition"], **build_index(root)}
