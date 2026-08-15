"""Evidence-bound analysis and indexing for the ThermoHITL v3 study.

The module intentionally supports a scientifically valid no-go result.  It
never synthesizes validation, training, or holdout observations when a
prospective development gate blocks those stages.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .events import sha256_file
from .human_experiments import _atomic_json, _write_dict_csv, source_checksum, utc_now
from .human_operator import OPERATOR_PROFILES, OperatorViewCondition


PRIMARY_BOOTSTRAP_SEED = 20260814
PRIMARY_BOOTSTRAP_REPLICATES = 10_000


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _episodes(results_root: Path, stage: str) -> Iterable[Dict[str, Any]]:
    for path in sorted((results_root / "raw" / stage).glob("*/episode.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def _summary(results_root: Path, stage: str) -> pd.DataFrame:
    path = results_root / stage / "episode_summary.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _paired_bootstrap(
    frame: pd.DataFrame,
    reference: str,
    treatment: str,
    comparison: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return rows
    for application in sorted(frame.application.unique()):
        selected = frame[frame.application == application]
        for regime in ("aggregate", "moderate", "correlated", "compound"):
            subset = selected if regime == "aggregate" else selected[
                selected.scenario.astype(str).str.contains("-" + regime + "-")
            ]
            pivot = subset.pivot_table(
                index=["environment_seed", "scenario"],
                columns="method",
                values="primary_outcome",
                aggfunc="first",
            )
            if reference not in pivot or treatment not in pivot:
                continue
            pivot = pivot.dropna(subset=[reference, treatment])
            if pivot.empty:
                continue
            differences = (
                pivot[treatment].to_numpy(dtype=float)
                - pivot[reference].to_numpy(dtype=float)
            )
            relative = differences / np.maximum(
                np.abs(pivot[reference].to_numpy(dtype=float)), 1e-9
            )
            rng = np.random.RandomState(
                PRIMARY_BOOTSTRAP_SEED
                + (0 if application == "commercial" else 10_000)
                + len(regime)
            )
            indices = rng.randint(
                0, len(differences),
                size=(PRIMARY_BOOTSTRAP_REPLICATES, len(differences)),
            )
            boot_diff = differences[indices].mean(axis=1)
            boot_relative = relative[indices].mean(axis=1)
            sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
            rows.append({
                "evidence_stage": "development",
                "comparison": comparison,
                "application": application,
                "regime": regime,
                "reference": reference,
                "treatment": treatment,
                "paired_episodes": len(differences),
                "reference_mean_loss": float(pivot[reference].mean()),
                "treatment_mean_loss": float(pivot[treatment].mean()),
                "mean_difference_treatment_minus_reference": float(differences.mean()),
                "difference_ci95_low": float(np.quantile(boot_diff, 0.025)),
                "difference_ci95_high": float(np.quantile(boot_diff, 0.975)),
                "mean_relative_difference": float(relative.mean()),
                "relative_ci95_low": float(np.quantile(boot_relative, 0.025)),
                "relative_ci95_high": float(np.quantile(boot_relative, 0.975)),
                "paired_win_rate": float(np.mean(differences < 0.0)),
                "standardized_effect": (
                    float(differences.mean() / sd) if sd > 1e-12 else 0.0
                ),
                "bootstrap_replicates": PRIMARY_BOOTSTRAP_REPLICATES,
                "bootstrap_seed": PRIMARY_BOOTSTRAP_SEED,
                "confirmatory": False,
            })
    return rows


def _actionability_rows(results_root: Path) -> List[Dict[str, Any]]:
    stages = (
        "development_trigger_candidate_n10_v4",
        "development_real_llm_actionability",
        "development_real_llm_actionability_retry1",
    )
    output: List[Dict[str, Any]] = []
    for stage in stages:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for episode in _episodes(results_root, stage):
            grouped.setdefault(episode["application"], []).append(episode)
        for application, episodes in sorted(grouped.items()):
            attempts = sum(row["actionability"]["structured_attempts"] for row in episodes)
            first = sum(row["actionability"]["first_pass_valid"] for row in episodes)
            final = sum(row["actionability"]["valid_after_one_repair"] for row in episodes)
            accepted = sum(row["actionability"]["material_actions_accepted"] for row in episodes)
            entered = sum(row["actionability"].get("material_actions_entered_transit", 0) for row in episodes)
            next_stage = sum(row["actionability"]["material_actions_next_stage"] for row in episodes)
            demand = sum(row["actionability"]["material_actions_reached_demand"] for row in episodes)
            output.append({
                "stage": stage,
                "planner": "Qwen2.5-7B-Instruct" if "real_llm" in stage else "deterministic mock",
                "application": application,
                "episodes": len(episodes),
                "structured_attempts": attempts,
                "first_pass_validity": first / max(attempts, 1),
                "valid_after_one_repair": final / max(attempts, 1),
                "material_actions_accepted": accepted,
                "accepted_to_transit": entered / max(accepted, 1),
                "accepted_to_next_stage": next_stage / max(accepted, 1),
                "accepted_to_demand": demand / max(accepted, 1),
            })
    return output


def _counterfactual_rows(results_root: Path) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for stage in (
        "development_gate_preliminary_v3_n10",
        "development_trigger_candidate_n10_v4",
    ):
        for episode in _episodes(results_root, stage):
            for probe in episode.get("counterfactuals", []):
                output.append({
                    "evidence_stage": "development",
                    "source_stage": stage,
                    "run_id": episode["run_id"],
                    "application": episode["application"],
                    "method": episode["method"],
                    "scenario": episode["scenario"],
                    "environment_seed": episode["environment_seed"],
                    **probe,
                })
    return output


def _counterfactual_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    output: List[Dict[str, Any]] = []
    for (application, method), group in frame.groupby(["application", "method"], sort=True):
        effect = group.intervention_effect.astype(float).to_numpy()
        output.append({
            "evidence_stage": "development",
            "application": application,
            "method": method,
            "paired_interventions": len(group),
            "beneficial": int(np.sum(effect > 1e-9)),
            "harmful": int(np.sum(effect < -1e-9)),
            "no_primary_effect": int(np.sum(np.abs(effect) <= 1e-9)),
            "mean_intervention_effect": float(np.mean(effect)),
            "median_intervention_effect": float(np.median(effect)),
            "commitment_or_agent_action_changed": int(group.agent_accepted.astype(bool).sum()),
            "accepted_material_action_changed": int(group.material_action_accepted.astype(bool).sum()),
            "material_reached_demand": int(group.material_reached_demand.astype(bool).sum()),
            "primary_outcome_changed": int(group.primary_outcome_changed.astype(bool).sum()),
            "complete_causal_chain": int((
                group.agent_accepted.astype(bool)
                & group.material_action_accepted.astype(bool)
                & group.material_reached_demand.astype(bool)
                & group.primary_outcome_changed.astype(bool)
            ).sum()),
            "confirmatory": False,
        })
    return output


def _compute_accounting(results_root: Path) -> Dict[str, Any]:
    manifests = []
    failures = []
    for path in sorted((results_root / "manifests").glob("*.json")):
        value = _read_json(path)
        if value.get("study") != "thermohitl_v3":
            continue
        if value.get("run_id"):
            manifests.append(value)
        if value.get("completion_status") not in (None, "complete"):
            failures.append({
                "manifest": str(path),
                "run_id": value.get("run_id", path.stem),
                "stage": value.get("stage", "development"),
                "status": value.get("completion_status", "recorded_abort"),
                "failure_reason": value.get("failure_reason", "see retained manifest"),
            })
    total_wall = sum(float(row.get("wall_clock_seconds", 0.0) or 0.0) for row in manifests)
    gpu_hours = sum(float(row.get("single_gpu_hours", 0.0) or 0.0) for row in manifests)
    manifest_gpu_hours = gpu_hours
    # Per-episode manifests start after planner construction. Two successful
    # qualification launches plus one collision-guarded launch initialized the
    # model. Conservatively account 80 seconds per observed initialization for
    # tokenizer/checkpoint setup and teardown; keep this estimate separate from
    # the exact episode-manifest sum.
    real_stages = {
        row.get("stage") for row in manifests
        if str(row.get("stage", "")).startswith("development_real_llm_actionability")
    }
    model_initializations = 3 if len(real_stages) >= 2 else (1 if real_stages else 0)
    estimated_model_overhead_seconds = 80.0 * model_initializations
    total_gpu_hours = manifest_gpu_hours + estimated_model_overhead_seconds / 3600.0
    record = {
        "created_at": utc_now(),
        "scope": "ThermoHITL v3 additive namespace only",
        "episode_manifests": len(manifests),
        "completed_episodes": sum(row.get("completion_status") == "complete" for row in manifests),
        "failed_episode_attempts": len(failures),
        "summed_episode_wall_clock_seconds": total_wall,
        "manifest_accounted_episode_gpu_hours": manifest_gpu_hours,
        "estimated_model_initialization_gpu_hours": estimated_model_overhead_seconds / 3600.0,
        "model_initializations_including_guarded_failed_launch": model_initializations,
        "additional_single_gpu_hours": total_gpu_hours,
        "llm_calls": sum(int(row.get("llm_calls", 0) or 0) for row in manifests),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in manifests),
        "generated_tokens": sum(int(row.get("generated_tokens", 0) or 0) for row in manifests),
        "estimated_gpu_cost_usd_at_0_34_per_hour": total_gpu_hours * 0.34,
        "cap_single_gpu_hours": 40.0,
        "validation_gpu_hours": 0.0,
        "training_gpu_hours": 0.0,
        "holdout_gpu_hours": 0.0,
        "reason_large_stages_not_run": "prospective Gate 5 failed across applications",
        "accounting_note": (
            "CPU/mock episode wall time is summed for reproducibility but is not GPU time. "
            "The only v3 GPU use is the explicitly identified real-LLM actionability stage. "
            "Episode GPU time is exact from manifests; model-initialization overhead is a labeled conservative estimate."
        ),
    }
    _write_dict_csv(results_root / "tables" / "failed_runs.csv", failures)
    return record


def _stage_stop_records(results_root: Path, gates: Mapping[str, Any]) -> None:
    record = {
        "created_at": utc_now(),
        "status": "not_run",
        "reason": "prospective development Gate 5 failed",
        "gate_decision": gates.get("decision"),
        "failed_or_incomplete_gates": gates.get("failed_or_incomplete_gates", []),
        "outcomes_opened": False,
        "episodes": 0,
        "scientific_interpretation": (
            "The stage is absent by design; absence is not missing data or an execution failure."
        ),
    }
    for directory, filename in (
        ("validation", "NOT_RUN.json"),
        ("training", "NOT_RUN.json"),
        ("holdout_locked", "NOT_RUN.json"),
        ("checkpoints", "NOT_TRAINED.json"),
    ):
        _atomic_json(results_root / directory / filename, record)


def _hypotheses(gates: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"hypothesis": "H1", "status": "not_tested", "evidence_stage": "none", "reason": "validation and holdout blocked before ThermoHITL-versus-KPI outcome comparison"},
        {"hypothesis": "H2", "status": "not_tested", "evidence_stage": "none", "reason": "always-on non-inferiority comparison not run"},
        {"hypothesis": "H3", "status": "not_tested", "evidence_stage": "none", "reason": "no confirmatory performance-effort frontier"},
        {"hypothesis": "H4", "status": "development_supported", "evidence_stage": "development", "reason": "Gate 6 passed, but no out-of-sample confirmation"},
        {"hypothesis": "H5", "status": "development_supported", "evidence_stage": "development", "reason": "paired branches contain complete causal chains in both applications"},
        {"hypothesis": "H6", "status": "unsupported_cross_application", "evidence_stage": "held-out development seeds", "reason": "commercial same-information incremental value failed; humanitarian passed"},
        {"hypothesis": "H7", "status": "not_tested", "evidence_stage": "none", "reason": "partition robustness was not promoted beyond development"},
        {"hypothesis": "H8", "status": "unsupported_or_not_tested", "evidence_stage": "development", "reason": "required information value did not replicate across applications"},
    ]


def _experimental_design() -> List[Dict[str, Any]]:
    return [
        {"stage": "antecedent diagnosis", "status": "complete", "unit": "v2 event ledger", "count": 952, "purpose": "diagnose action validation and material arrival"},
        {"stage": "v3 deterministic development", "status": "complete", "unit": "episode", "count": 809, "purpose": "mechanics, causal probes, and six prospective gates"},
        {"stage": "real-Qwen actionability", "status": "qualified if artifacts present", "unit": "episode", "count": "see actionability table", "purpose": "first-pass and one-repair structured validity"},
        {"stage": "validation", "status": "not run", "unit": "episode", "count": 0, "purpose": "blocked by Gate 5"},
        {"stage": "multi-seed training", "status": "not run", "unit": "independent RL seed", "count": 0, "purpose": "blocked by Gate 5"},
        {"stage": "locked holdout", "status": "not run", "unit": "episode", "count": 0, "purpose": "blocked by Gate 5"},
    ]


def build_index(results_root: Path) -> Path:
    rows: List[Dict[str, Any]] = []
    descriptions = {
        "README.md": "complete evidence-bound v3 result guide",
        "PAPER_SUMMARY.md": "paper-facing no-go summary",
        "PAPER_OUTLINE.md": "20–30 page paper outline and evidence map",
        "gate_status.json": "prospective six-gate decision",
        "causal_value_summary.json": "same-information causal information-value result",
    }
    for path in sorted(results_root.rglob("*")):
        if not path.is_file() or path.name == "INDEX.csv":
            continue
        relative = path.relative_to(results_root)
        parts = relative.parts
        stage = parts[0] if parts else "root"
        name_lower = path.name.lower()
        application = (
            "commercial" if "commercial" in name_lower
            else "humanitarian" if "humanitarian" in name_lower else "both_or_na"
        )
        method = "not_encoded"
        for candidate in (
            "thermohitl_rule", "local_kpi_trigger", "autonomous_no_human",
            "fixed_communication_no_human", "no_communication",
        ):
            if candidate in name_lower:
                method = candidate
                break
        rows.append({
            "artifact_path": str(Path("results/human_operator_v3") / relative),
            "artifact_type": path.suffix.lstrip(".") or "file",
            "stage": stage,
            "description": descriptions.get(path.name, path.stem.replace("_", " ")),
            "application": application,
            "method": method,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    output = results_root / "INDEX.csv"
    _write_dict_csv(output, rows)
    return output


def run(repository_root: Path, results_root: Path) -> Dict[str, Any]:
    """Regenerate all v3 development statistics and fail-closed stage records."""

    results_root.mkdir(parents=True, exist_ok=True)
    gates = _read_json(results_root / "development" / "gate_status.json")
    if gates.get("holdout_unlocked"):
        raise RuntimeError(
            "This development-only analyzer must not stand in for an eligible holdout analysis"
        )
    _stage_stop_records(results_root, gates)

    episode_frames: List[pd.DataFrame] = []
    for stage in (
        "development_gate_coordination_n10",
        "development_gate_preliminary_v3_n10",
        "development_trigger_candidate_n10_v4",
        "development_real_llm_actionability",
        "development_real_llm_actionability_retry1",
    ):
        frame = _summary(results_root, stage)
        if not frame.empty:
            frame.insert(0, "evidence_stage", stage)
            episode_frames.append(frame)
    combined = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    combined.to_csv(
        results_root / "processed" / "development_episodes.csv",
        index=False,
        lineterminator="\n",
    )

    effects = []
    effects.extend(_paired_bootstrap(
        _summary(results_root, "development_gate_coordination_n10"),
        "no_communication", "fixed_communication_no_human",
        "fixed communication minus no communication",
    ))
    effects.extend(_paired_bootstrap(
        _summary(results_root, "development_gate_preliminary_v3_n10"),
        "autonomous_no_human", "local_kpi_trigger",
        "bounded KPI-triggered human minus autonomous no human",
    ))
    _write_dict_csv(results_root / "statistics" / "development_paired_effects.csv", effects)
    _write_dict_csv(results_root / "tables" / "development_paired_effects.csv", effects)

    actionability = _actionability_rows(results_root)
    _write_dict_csv(results_root / "statistics" / "actionability_summary.csv", actionability)
    _write_dict_csv(results_root / "tables" / "actionability_diagnostics.csv", actionability)

    counterfactuals = _counterfactual_rows(results_root)
    _write_dict_csv(results_root / "counterfactuals" / "paired_intervention_effects.csv", counterfactuals)
    counter_summary = _counterfactual_summary(counterfactuals)
    _write_dict_csv(results_root / "statistics" / "counterfactual_summary.csv", counter_summary)
    _write_dict_csv(results_root / "tables" / "counterfactual_causal_chain.csv", counter_summary)

    monitoring_path = results_root / "monitoring" / "causal_incremental_value.csv"
    if monitoring_path.is_file():
        pd.read_csv(monitoring_path).to_csv(
            results_root / "tables" / "monitoring_incremental_value.csv",
            index=False,
            lineterminator="\n",
        )

    hypothesis_rows = _hypotheses(gates)
    _write_dict_csv(results_root / "tables" / "hypothesis_outcomes.csv", hypothesis_rows)
    _write_dict_csv(results_root / "tables" / "experimental_design.csv", _experimental_design())
    _write_dict_csv(results_root / "tables" / "operator_profiles.csv", [
        {**asdict(profile), "evidence_type": "simulated_operator"}
        for profile in OPERATOR_PROFILES.values()
    ])
    _write_dict_csv(results_root / "tables" / "operator_view_conditions.csv", [
        {
            "condition": condition.value,
            "execution_boundary": (
                "privileged evaluator-only upper bound"
                if condition == OperatorViewCondition.EVALUATOR_ORACLE
                else "schema-limited authorized view"
            ),
        }
        for condition in OperatorViewCondition
    ])
    if not combined.empty:
        communication_columns = [
            "total_communication_messages", "prompt_tokens", "generated_tokens",
            "llm_calls", "llm_latency_seconds", "operator_requests",
            "operator_interventions", "operator_minutes",
        ]
        communication = (
            combined.groupby(["application", "method"], as_index=False)[communication_columns]
            .mean(numeric_only=True)
        )
        communication.insert(0, "evidence_stage", "development")
        communication.to_csv(
            results_root / "tables" / "communication_budgets.csv",
            index=False,
            lineterminator="\n",
        )
    trigger = next((gate for gate in gates.get("gates", []) if gate.get("gate") == 6), {})
    _write_dict_csv(results_root / "tables" / "trigger_parameters.csv", [
        {**trigger.get("frozen_candidate_parameters", {}), "stage": "development candidate only", "promoted_to_validation": False}
    ])
    _write_dict_csv(results_root / "tables" / "rl_training_seeds.csv", [{
        "status": "not_run", "independent_training_seeds": 0,
        "reason": "prospective Gate 5 failed before training",
    }])
    _write_dict_csv(results_root / "tables" / "holdout_results.csv", [{
        "status": "not_run", "episodes": 0, "outcomes_opened": False,
        "reason": "prospective Gate 5 failed before holdout design/freeze",
    }])
    not_run = [{
        "status": "not_run",
        "observations": 0,
        "reason": "prospective Gate 5 failed before validation, training, and holdout",
    }]
    for filename in (
        "main_paired_comparisons.csv",
        "noninferiority_analysis.csv",
        "communication_reductions.csv",
        "pareto_operating_points.csv",
        "ablation_results.csv",
    ):
        _write_dict_csv(results_root / "tables" / filename, not_run)

    compute = _compute_accounting(results_root)
    _atomic_json(results_root / "reproducibility" / "compute_accounting.json", compute)
    _write_dict_csv(results_root / "tables" / "compute_accounting.csv", [compute])
    _atomic_json(results_root / "statistics" / "analysis_manifest.json", {
        "created_at": utc_now(),
        "analysis_scope": "development only; no validation or holdout inference",
        "source_checksum": source_checksum(repository_root),
        "bootstrap_seed": PRIMARY_BOOTSTRAP_SEED,
        "bootstrap_replicates": PRIMARY_BOOTSTRAP_REPLICATES,
        "gate_decision": gates.get("decision"),
        "artifacts": {
            "paired_effects": sha256_file(results_root / "statistics" / "development_paired_effects.csv"),
            "counterfactual_summary": sha256_file(results_root / "statistics" / "counterfactual_summary.csv"),
            "actionability": sha256_file(results_root / "statistics" / "actionability_summary.csv"),
        },
    })
    index = build_index(results_root)
    return {
        "status": "complete_development_no_go",
        "gate_decision": gates.get("decision"),
        "development_episode_rows": len(combined),
        "paired_effect_rows": len(effects),
        "counterfactual_rows": len(counterfactuals),
        "additional_single_gpu_hours": compute["additional_single_gpu_hours"],
        "index": str(index),
    }
