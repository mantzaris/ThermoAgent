"""Frozen V7 formal-development execution and panel-level dynamic analysis."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .events import sha256_file
from .v5_analysis import paired_bootstrap
from .v5_experiments import atomic_json, source_checksum, utc_now, write_csv
from .v7_experiments import aggregate_stage, run_episode
from .v7_formal_analysis import (
    FEATURE_BLOCKS, _inner_c, _pipeline, analyze_risk_stage,
    power_from_pilot, prepare_candidates,
)
from .v7_learned_controller import FittedRiskController
from .v7_policies import V7SelectiveController


REFERENCE_STAGE = "development_formal_reference"
DYNAMIC_STAGE = "development_formal_dynamic"
COMMUNICATION_STAGE = "development_formal_communication"


class _StageLock:
    def __init__(self, results_root: Path, stage: str) -> None:
        path = results_root / "logs" / (stage + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write("pid=%d\n" % os.getpid())
        self.handle.flush()

    def close(self) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()

    def __enter__(self) -> "_StageLock":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    return pd.read_csv(path).to_dict("records")


def _freeze(results_root: Path) -> Mapping[str, Any]:
    path = results_root / "protocol" / "freeze_manifest.json"
    if not path.exists():
        raise RuntimeError("V7 formal development is locked until protocol freeze")
    value = json.loads(path.read_text(encoding="utf-8"))
    protocol = results_root / value["protocol_path"]
    if sha256_file(protocol) != value["protocol_sha256"]:
        raise RuntimeError("frozen V7 protocol checksum mismatch")
    return value


def _status(
    results_root: Path, stage: str, completed: int, total: int,
    started: str, state: str,
) -> None:
    atomic_json(results_root / "logs" / "supervisor_status.json", {
        "stage": stage, "state": state, "completed": int(completed),
        "total": int(total), "started_at": started, "updated_at": utc_now(),
    })


def run_reference_development(
    repository: Path, results_root: Path,
) -> Dict[str, Any]:
    freeze = _freeze(results_root)
    manifest = _read_csv(results_root / "manifests" / "development_inputs.csv")
    started = utc_now()
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    with _StageLock(results_root, REFERENCE_STAGE):
        for index, row in enumerate(manifest, start=1):
            _status(results_root, REFERENCE_STAGE, index - 1, len(manifest), started, "running")
            try:
                output = run_episode(
                    str(row["application"]), str(row["complexity"]),
                    str(row["coupling"]), str(row["fragmentation"]),
                    str(row["network_disruption"]), str(row["topology_family"]),
                    int(row["environment_seed"]),
                    V7SelectiveController("always_act", 1.0),
                    str(row["information_condition"]),
                    str(row["reference_sketch_policy"]), results_root,
                    REFERENCE_STAGE, int(row["counterfactual_limit_per_epoch"]),
                    operational_communication_policy=str(row["operational_communication_policy"]),
                )
                completed.append(dict(output["summary"]))
            except Exception as error:
                failure = {
                    **row, "status": "failed", "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
                failures.append(failure)
            _status(results_root, REFERENCE_STAGE, index, len(manifest), started, "running")
        if failures:
            write_csv(results_root / "negative_results" / "formal_reference_failures.csv", failures)
        execution = aggregate_stage(results_root, REFERENCE_STAGE)
        risk = analyze_risk_stage(results_root, REFERENCE_STAGE, coverage=0.60)
        pilot_effects = pd.read_csv(
            results_root / "pilots_iteration3" / "analysis" / "paired_incremental_effects.csv"
        )
        power = power_from_pilot(pilot_effects, practical_effect=0.040)
        write_csv(results_root / "development" / "power_precision_plan.csv", power)
        report = {
            "stage": REFERENCE_STAGE, "completed": len(completed),
            "failed": len(failures), "execution": execution,
            "risk_analysis": risk, "power_plan": power,
            "freeze_manifest_sha256": sha256_file(
                results_root / "protocol" / "freeze_manifest.json"
            ),
            "source_commit": freeze["source_commit"],
        }
        atomic_json(results_root / "development" / "reference_summary.json", report)
        _status(results_root, REFERENCE_STAGE, len(manifest), len(manifest), started, "complete")
        return report


def _fit_outer_models(
    frame: pd.DataFrame, application: str,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], Any]]:
    subset = frame[frame.application == application].reset_index(drop=True)
    splitter = GroupKFold(n_splits=5)
    rows: List[Dict[str, Any]] = []
    models: Dict[Tuple[str, str], Any] = {}
    for fold, (train, test) in enumerate(
        splitter.split(subset, subset.harmful_label, subset.panel_id), start=1,
    ):
        train_frame = subset.iloc[train]
        test_frame = subset.iloc[test]
        for block in ("strongest_nonentropic", "generalized_entropic"):
            numeric = FEATURE_BLOCKS[block]
            c_value = _inner_c(train_frame, numeric, train_frame.panel_id.to_numpy())
            if train_frame.harmful_label.nunique() < 2:
                raise RuntimeError("formal outer training fold contains one harm class")
            model = _pipeline(numeric, c_value)
            model.fit(train_frame, train_frame.harmful_label)
            model_key = "%s-fold-%d" % (block, fold)
            models[(application, model_key)] = model
            for panel_id in sorted(test_frame.panel_id.unique()):
                rows.append({
                    "application": application, "fold": fold,
                    "feature_block": block, "model_key": model_key,
                    "panel_id": panel_id, "selected_c": c_value,
                    "training_panels": train_frame.panel_id.nunique(),
                    "test_panels": test_frame.panel_id.nunique(),
                    "panel_disjoint": set(train_frame.panel_id).isdisjoint(set(test_frame.panel_id)),
                    "environment_seed_disjoint": set(train_frame.environment_seed).isdisjoint(set(test_frame.environment_seed)),
                })
    return rows, models


def run_crossfit_dynamic_development(
    repository: Path, results_root: Path,
) -> Dict[str, Any]:
    _freeze(results_root)
    reference = prepare_candidates(pd.read_csv(
        results_root / REFERENCE_STAGE / "candidate_decisions.csv"
    ))
    manifest = pd.read_csv(results_root / "manifests" / "development_inputs.csv")
    audit_rows: List[Dict[str, Any]] = []
    models: Dict[Tuple[str, str], Any] = {}
    for application in ("humanitarian", "utility_restoration"):
        audit, fitted = _fit_outer_models(reference, application)
        audit_rows.extend(audit)
        models.update(fitted)
    write_csv(results_root / "development" / "dynamic_grouped_fold_audit.csv", audit_rows)
    model_root = results_root / "training" / "crossfit_risk_models"
    model_root.mkdir(parents=True, exist_ok=True)
    model_rows = []
    for (application, key), model in sorted(models.items()):
        path = model_root / ("%s-%s.joblib" % (application, key))
        joblib.dump(model, path, compress=3)
        model_rows.append({
            "application": application, "model_key": key,
            "path": str(path.relative_to(results_root)),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        })
    write_csv(results_root / "training" / "crossfit_risk_model_manifest.csv", model_rows)
    assignments = pd.DataFrame(audit_rows).drop_duplicates([
        "application", "feature_block", "panel_id",
    ])
    started = utc_now()
    completed = []
    failures = []
    total = len(assignments)
    with _StageLock(results_root, DYNAMIC_STAGE):
        for index, row in enumerate(assignments.to_dict("records"), start=1):
            _status(results_root, DYNAMIC_STAGE, index - 1, total, started, "running")
            panel = reference[reference.panel_id == row["panel_id"]].iloc[0]
            block = str(row["feature_block"])
            key = str(row["model_key"])
            controller = FittedRiskController(
                models[(str(row["application"]), key)],
                "crossfit_%s" % block, str(panel.topology_family),
                autonomous_coverage=0.60, operator_slots_per_epoch=1,
            )
            try:
                output = run_episode(
                    str(panel.application), str(panel.complexity),
                    str(panel.coupling), str(panel.fragmentation),
                    str(panel.network_disruption), str(panel.topology_family),
                    int(panel.environment_seed), controller,
                    str(panel.information_condition), "event_triggered",
                    results_root, DYNAMIC_STAGE, 2,
                    operational_communication_policy="agent_event_triggered",
                )
                completed.append(dict(output["summary"]))
            except Exception as error:
                failures.append({
                    **row, "status": "failed", "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                })
            _status(results_root, DYNAMIC_STAGE, index, total, started, "running")
        if failures:
            write_csv(results_root / "negative_results" / "formal_dynamic_failures.csv", failures)
        execution = aggregate_stage(results_root, DYNAMIC_STAGE)
        analysis = analyze_dynamic_development(results_root)
        report = {
            "stage": DYNAMIC_STAGE, "completed": len(completed),
            "failed": len(failures), "execution": execution,
            "analysis": analysis,
        }
        atomic_json(results_root / "development" / "dynamic_summary.json", report)
        _status(results_root, DYNAMIC_STAGE, total, total, started, "complete")
        return report


def _panel_bootstrap(values: Sequence[float], seed: int) -> Dict[str, float]:
    return paired_bootstrap(pd.Series(list(values), dtype=float), 10000, seed)


def _interaction(frame: pd.DataFrame, seed: int) -> Dict[str, Any]:
    coded = frame.copy()
    coded["coupling_value"] = coded.coupling.map({"low": 0.0, "medium": 0.5, "high": 1.0})
    coded["fragmentation_value"] = coded.fragmentation.map({"low": 0.0, "medium": 0.5, "high": 1.0})
    coded["size_value"] = coded.complexity.map({"small": 0.0, "medium": 0.5, "large": 1.0})
    coded["application_value"] = coded.application.eq("utility_restoration").astype(float)
    design = np.column_stack([
        np.ones(len(coded)), coded.coupling_value, coded.fragmentation_value,
        coded.size_value, coded.application_value,
        coded.coupling_value * coded.fragmentation_value,
    ])
    target = coded.harm_reduction.to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(10000):
        index = rng.randint(0, len(coded), len(coded))
        value, _, sampled_rank, _ = np.linalg.lstsq(design[index], target[index], rcond=None)
        if sampled_rank == design.shape[1]:
            boot.append(float(value[-1]))
    return {
        "panels": len(coded), "design_rank": int(rank),
        "coupling_fragmentation_interaction": float(coefficients[-1]),
        "ci95_low": float(np.quantile(boot, 0.025)) if boot else None,
        "ci95_high": float(np.quantile(boot, 0.975)) if boot else None,
        "bootstrap_replicates_retained": len(boot),
    }


def analyze_dynamic_development(results_root: Path) -> Dict[str, Any]:
    summaries = pd.read_csv(results_root / DYNAMIC_STAGE / "episode_summary.csv")
    decisions = pd.read_csv(results_root / DYNAMIC_STAGE / "candidate_decisions.csv")
    selected = decisions[
        decisions.accepted_physical_action.astype(bool)
        & decisions.delegation_action.eq("execute_autonomously")
    ].groupby("run_id").size().rename("accepted_autonomous_actions")
    eligible = decisions[
        ~decisions.proposed_operational_action.eq("no_operational_action")
    ].groupby("run_id").size().rename("eligible_proposals")
    summaries = summaries.merge(selected, on="run_id", how="left").merge(eligible, on="run_id", how="left")
    summaries[["accepted_autonomous_actions", "eligible_proposals"]] = summaries[["accepted_autonomous_actions", "eligible_proposals"]].fillna(0)
    summaries["actual_autonomous_coverage"] = summaries.accepted_autonomous_actions / summaries.eligible_proposals.clip(lower=1)
    summaries["autonomous_completed_actions"] = (
        summaries.autonomous_harmful_actions + summaries.autonomous_beneficial_actions
        + summaries.autonomous_neutral_actions
    )
    summaries["autonomous_harm_rate"] = (
        summaries.autonomous_harmful_actions
        / summaries.autonomous_completed_actions.clip(lower=1)
    )
    key = [
        "application", "complexity", "coupling", "fragmentation",
        "network_disruption", "topology_family", "information_condition",
        "environment_seed", "sketch_policy", "operational_communication_policy",
    ]
    baseline = summaries[summaries.controller.eq("crossfit_strongest_nonentropic")]
    entropic = summaries[summaries.controller.eq("crossfit_generalized_entropic")]
    paired = baseline.merge(entropic, on=key, suffixes=("_baseline", "_entropic"), validate="one_to_one")
    paired["harm_reduction"] = paired.autonomous_harm_rate_baseline - paired.autonomous_harm_rate_entropic
    paired["service_loss_difference"] = paired.service_loss_entropic - paired.service_loss_baseline
    paired["relative_service_degradation"] = paired.service_loss_difference / paired.service_loss_baseline.abs().clip(lower=1e-9)
    paired["causal_utility_gain"] = paired.net_causal_utility_entropic - paired.net_causal_utility_baseline
    paired["coverage_difference"] = paired.actual_autonomous_coverage_entropic - paired.actual_autonomous_coverage_baseline
    write_csv(results_root / "statistics" / "dynamic_paired_panel_effects.csv", paired.to_dict("records"))
    application_rows = []
    h2_passes = []
    for application, subset in paired.groupby("application", sort=True):
        high = subset[
            subset.coupling.eq("high") & subset.fragmentation.eq("high")
            & subset.information_condition.eq("private_fragmented")
            & subset.complexity.eq("medium")
        ]
        harm = _panel_bootstrap(high.harm_reduction, 787711)
        service = _panel_bootstrap(high.relative_service_degradation, 787712)
        utility = _panel_bootstrap(high.causal_utility_gain, 787713)
        row = {
            "application": application, "panels": len(high),
            "harm_reduction": harm["mean"], "harm_ci95_low": harm["ci_low"],
            "harm_ci95_high": harm["ci_high"],
            "relative_service_degradation": service["mean"],
            "service_ci95_low": service["ci_low"], "service_ci95_high": service["ci_high"],
            "causal_utility_gain": utility["mean"],
            "utility_ci95_low": utility["ci_low"], "utility_ci95_high": utility["ci_high"],
            "coverage_difference": float(high.coverage_difference.mean()),
            "mean_absolute_coverage_difference": float(
                high.coverage_difference.abs().mean()
            ),
        }
        row["h2_pass"] = bool(
            row["harm_reduction"] >= 0.040 and row["harm_ci95_low"] > 0.0
            and row["service_ci95_high"] <= 0.020
            and row["mean_absolute_coverage_difference"] <= 0.020
        )
        h2_passes.append(row["h2_pass"])
        application_rows.append(row)
    interaction = _interaction(paired, 787714)
    app_points = paired.groupby("application").harm_reduction.mean().to_dict()
    h1_pass = bool(
        interaction["coupling_fragmentation_interaction"] >= 0.020
        and interaction["ci95_low"] is not None and interaction["ci95_low"] > 0.0
        and all(float(value) > 0.0 for value in app_points.values())
    )
    write_csv(results_root / "statistics" / "high_complexity_dynamic_effects.csv", application_rows)
    write_csv(results_root / "statistics" / "complexity_interaction.csv", [interaction])
    report = {
        "independent_panels": len(paired), "H1_pass": h1_pass,
        "H2_pass": bool(h2_passes and all(h2_passes)),
        "interaction": interaction, "high_complexity": application_rows,
        "application_mean_harm_reduction": app_points,
        "matched_dynamic_execution": True,
    }
    atomic_json(results_root / "statistics" / "dynamic_primary_analysis.json", report)
    return report


def run_communication_development(
    repository: Path, results_root: Path,
) -> Dict[str, Any]:
    _freeze(results_root)
    manifest = pd.read_csv(results_root / "manifests" / "development_inputs.csv")
    subset = manifest[
        manifest.coupling.eq("high") & manifest.fragmentation.eq("high")
        & manifest.information_condition.eq("private_fragmented")
        & manifest.complexity.eq("medium")
    ].groupby("application", sort=True).head(6)
    policies = ("none", "periodic", "event_triggered", "always_on")
    total = len(subset) * len(policies)
    completed = []
    failures = []
    started = utc_now()
    with _StageLock(results_root, COMMUNICATION_STAGE):
        index = 0
        for row in subset.to_dict("records"):
            for sketch in policies:
                index += 1
                _status(results_root, COMMUNICATION_STAGE, index - 1, total, started, "running")
                try:
                    output = run_episode(
                        str(row["application"]), str(row["complexity"]),
                        str(row["coupling"]), str(row["fragmentation"]),
                        str(row["network_disruption"]), str(row["topology_family"]),
                        int(row["environment_seed"]),
                        V7SelectiveController("always_act", 1.0),
                        str(row["information_condition"]), sketch,
                        results_root, COMMUNICATION_STAGE, 2,
                        operational_communication_policy="agent_event_triggered",
                    )
                    completed.append(dict(output["summary"]))
                except Exception as error:
                    failures.append({
                        **row, "sketch_policy": sketch, "status": "failed",
                        "failure_type": type(error).__name__, "failure_reason": str(error),
                    })
                _status(results_root, COMMUNICATION_STAGE, index, total, started, "running")
        if failures:
            write_csv(results_root / "negative_results" / "formal_communication_failures.csv", failures)
        execution = aggregate_stage(results_root, COMMUNICATION_STAGE)
        analysis = analyze_communication_development(results_root)
        report = {"completed": len(completed), "failed": len(failures), "execution": execution, "analysis": analysis}
        atomic_json(results_root / "development" / "communication_summary.json", report)
        _status(results_root, COMMUNICATION_STAGE, total, total, started, "complete")
        return report


def analyze_communication_development(results_root: Path) -> Dict[str, Any]:
    frame = pd.read_csv(results_root / COMMUNICATION_STAGE / "episode_summary.csv")
    key = [
        "application", "complexity", "coupling", "fragmentation",
        "network_disruption", "topology_family", "information_condition",
        "environment_seed", "controller", "operational_communication_policy",
    ]
    event = frame[frame.sketch_policy.eq("event_triggered")]
    always = frame[frame.sketch_policy.eq("always_on")]
    paired = event.merge(always, on=key, suffixes=("_event", "_always"), validate="one_to_one")
    paired["message_reduction"] = 1.0 - paired.total_messages_event / paired.total_messages_always.clip(lower=1)
    paired["byte_reduction"] = 1.0 - paired.total_bytes_event / paired.total_bytes_always.clip(lower=1)
    paired["harm_rate_event"] = paired.harmful_actions_event / (
        paired.harmful_actions_event + paired.beneficial_actions_event + paired.neutral_actions_event
    ).clip(lower=1)
    paired["harm_rate_always"] = paired.harmful_actions_always / (
        paired.harmful_actions_always + paired.beneficial_actions_always + paired.neutral_actions_always
    ).clip(lower=1)
    paired["harm_degradation"] = paired.harm_rate_event - paired.harm_rate_always
    rows = []
    passes = []
    for application, subset in paired.groupby("application", sort=True):
        messages = _panel_bootstrap(subset.message_reduction, 787721)
        byte = _panel_bootstrap(subset.byte_reduction, 787722)
        harm = _panel_bootstrap(subset.harm_degradation, 787723)
        maximum_mae = float(subset.distributed_estimation_mae_event.max())
        row = {
            "application": application, "panels": len(subset),
            "message_reduction": messages["mean"], "message_ci95_low": messages["ci_low"],
            "byte_reduction": byte["mean"], "byte_ci95_low": byte["ci_low"],
            "harm_degradation": harm["mean"], "harm_ci95_high": harm["ci_high"],
            "maximum_event_estimation_mae": maximum_mae,
        }
        row["h3_pass"] = bool(
            row["message_reduction"] >= 0.20 and row["byte_reduction"] >= 0.20
            and row["harm_ci95_high"] <= 0.020 and maximum_mae <= 0.080
        )
        rows.append(row)
        passes.append(row["h3_pass"])
    write_csv(results_root / "statistics" / "communication_paired_panels.csv", paired.to_dict("records"))
    write_csv(results_root / "statistics" / "communication_primary_effects.csv", rows)
    report = {"H3_pass": bool(passes and all(passes)), "applications": rows}
    atomic_json(results_root / "statistics" / "communication_primary_analysis.json", report)
    return report


def evaluate_formal_development_gates(results_root: Path) -> Dict[str, Any]:
    dynamic = json.loads((results_root / "statistics" / "dynamic_primary_analysis.json").read_text(encoding="utf-8"))
    communication = json.loads((results_root / "statistics" / "communication_primary_analysis.json").read_text(encoding="utf-8"))
    replay_path = results_root / "reproducibility" / "replay" / "replay_summary.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {}
    failure_files = (
        results_root / "negative_results" / "formal_reference_failures.csv",
        results_root / "negative_results" / "formal_dynamic_failures.csv",
        results_root / "negative_results" / "formal_communication_failures.csv",
    )
    failure_count = 0
    for path in failure_files:
        if path.exists():
            failure_frame = pd.read_csv(path)
            if not (
                len(failure_frame) == 1
                and "status" in failure_frame
                and str(failure_frame.status.iloc[0]) == "no_rows"
            ):
                failure_count += len(failure_frame)
    engineering_integrity = bool(
        failure_count == 0
        and replay.get("status") == "pass"
        and int(replay.get("replay_mismatches", -1)) == 0
        and int(replay.get("privacy_failures", -1)) == 0
        and float(replay.get("maximum_conservation_residual", float("inf"))) <= 1e-9
    )
    formal_pass = bool(
        engineering_integrity and dynamic["H1_pass"] and dynamic["H2_pass"]
        and communication["H3_pass"]
    )
    report = {
        "formal_development_primary_pass": formal_pass,
        "engineering_integrity_pass": engineering_integrity,
        "formal_episode_failures": failure_count,
        "H1_pass": bool(dynamic["H1_pass"]), "H2_pass": bool(dynamic["H2_pass"]),
        "H3_pass": bool(communication["H3_pass"]),
        "RL_training_unlocked": formal_pass,
        "Qwen_qualification_unlocked": formal_pass,
        "validation_unlocked": False,
        "holdout_unlocked": False,
        "replay_status_at_evaluation": replay.get("status", "pending_replay_after_formal"),
        "disposition": (
            "primary formal-development mechanism passes; learned-agent qualification required next"
            if formal_pass else
            "formal-development no-go; training, Qwen, validation, and holdout remain locked"
        ),
    }
    atomic_json(results_root / "manifests" / "stage_disposition.json", report)
    return report
