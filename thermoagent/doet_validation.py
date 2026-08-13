"""Preregistered validation-only selection for the DOET operating point."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_rows(
    candidate: pd.DataFrame,
    fixed: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["application", "scenario_name", "seed", "n_agents"]
    metric_columns = [
        "primary_outcome",
        "total_communication_messages",
        "total_communication_bytes",
        "prompt_tokens",
        "generated_tokens",
        "llm_calls",
        "llm_latency_seconds",
        "quiet_mode_fraction",
        "communication_active_decision_epochs",
        "tool_proposals",
        "trigger_activations",
    ]
    left = candidate[keys + metric_columns].copy()
    right = fixed[keys + metric_columns].copy()
    paired = left.merge(right, on=keys, suffixes=("_doet", "_fixed"), validate="one_to_one")
    if len(paired) != len(candidate) or len(paired) != len(fixed):
        raise ValueError("every validation candidate requires an exact fixed pair")
    paired["relative_degradation"] = (
        paired["primary_outcome_doet"] - paired["primary_outcome_fixed"]
    ) / np.maximum(np.abs(paired["primary_outcome_fixed"]), 1e-9)
    for metric in (
        "total_communication_messages",
        "total_communication_bytes",
        "prompt_tokens",
        "generated_tokens",
        "llm_calls",
        "llm_latency_seconds",
    ):
        paired[metric + "_reduction"] = 1.0 - (
            paired[metric + "_doet"]
            / np.maximum(paired[metric + "_fixed"], 1e-9)
        )
    return paired


def _candidate_summary(variant: str, paired: pd.DataFrame) -> Dict[str, Any]:
    non_nominal = paired[paired["scenario_name"] != "nominal"]
    by_application = non_nominal.groupby("application")["relative_degradation"].mean()
    by_regime = non_nominal.groupby(["application", "scenario_name"])["relative_degradation"].mean()
    nominal = paired[paired["scenario_name"] == "nominal"]
    application_means = {
        str(key): float(value) for key, value in by_application.items()
    }
    regime_means = {
        "%s:%s" % key: float(value) for key, value in by_regime.items()
    }
    validation_eligible = (
        set(application_means) == {"commercial", "humanitarian"}
        and all(value <= 0.01 for value in application_means.values())
        and all(value <= 0.02 for value in regime_means.values())
    )
    return {
        "method_variant": variant,
        "paired_episodes": int(len(paired)),
        "non_nominal_pairs": int(len(non_nominal)),
        "commercial_mean_relative_degradation": application_means.get("commercial"),
        "humanitarian_mean_relative_degradation": application_means.get("humanitarian"),
        "worst_regime_mean_relative_degradation": float(max(regime_means.values())),
        "mean_relative_degradation": float(non_nominal["relative_degradation"].mean()),
        "mean_message_reduction": float(non_nominal["total_communication_messages_reduction"].mean()),
        "mean_byte_reduction": float(non_nominal["total_communication_bytes_reduction"].mean()),
        "mean_prompt_token_reduction": float(non_nominal["prompt_tokens_reduction"].mean()),
        "mean_generated_token_reduction": float(non_nominal["generated_tokens_reduction"].mean()),
        "mean_llm_call_reduction": float(non_nominal["llm_calls_reduction"].mean()),
        "mean_latency_reduction": float(non_nominal["llm_latency_seconds_reduction"].mean()),
        "nominal_false_active_fraction": float(
            (1.0 - nominal["quiet_mode_fraction_doet"]).mean()
        ),
        "mean_active_agent_step_fraction": float(
            (1.0 - paired["quiet_mode_fraction_doet"]).mean()
        ),
        "mean_active_decision_fraction": float(
            (
                paired["communication_active_decision_epochs_doet"]
                / np.maximum(paired["tool_proposals_doet"], 1.0)
            ).mean()
        ),
        "mean_trigger_activations": float(paired["trigger_activations_doet"].mean()),
        "validation_noninferiority_eligible": bool(validation_eligible),
        "application_degradation_json": json.dumps(application_means, sort_keys=True),
        "regime_degradation_json": json.dumps(regime_means, sort_keys=True),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run(results_root: Path) -> Dict[str, Any]:
    validation_dir = results_root / "validation"
    episodes_path = validation_dir / "episodes.csv"
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)
    frame = pd.read_csv(episodes_path)
    failed = frame[frame["status"] != "complete"]
    if len(failed):
        raise ValueError(
            "validation selection requires all planned runs; %d failed" % len(failed)
        )
    fixed = frame[frame["method"] == "fixed_always_on"].copy()
    candidates = frame[frame["method"] == "doet_rule"].copy()
    if fixed.empty or candidates.empty:
        raise ValueError("validation requires fixed_always_on and doet_rule")
    summaries: List[Dict[str, Any]] = []
    paired_frames: Dict[str, pd.DataFrame] = {}
    for variant in sorted(candidates["method_variant"].unique()):
        paired = _paired_rows(
            candidates[candidates["method_variant"] == variant], fixed
        )
        paired_frames[str(variant)] = paired
        summaries.append(_candidate_summary(str(variant), paired))

    eligible = [
        row for row in summaries
        if row["validation_noninferiority_eligible"]
    ]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (
                row["mean_message_reduction"],
                -row["nominal_false_active_fraction"],
                -row["worst_regime_mean_relative_degradation"],
                row["method_variant"],
            ),
            reverse=True,
        )[0]
        selection_status = "eligible operating point selected"
    else:
        selected = sorted(
            summaries,
            key=lambda row: (
                row["worst_regime_mean_relative_degradation"],
                row["mean_relative_degradation"],
                -row["mean_message_reduction"],
                row["method_variant"],
            ),
        )[0]
        selection_status = (
            "no candidate met validation eligibility; minimum worst degradation selected and failure retained"
        )

    selected_variant = str(selected["method_variant"])
    manifest_paths = sorted((results_root / "manifests").glob(
        "validation-*-doet_rule-*-v%s.json" % selected_variant
    ))
    if not manifest_paths:
        raise FileNotFoundError("no selected-candidate manifest found")
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    trigger = manifest["experiment_configuration"]["resolved_trigger"]
    active_fraction = float(selected["mean_active_agent_step_fraction"])
    selected_pairs = paired_frames[selected_variant]
    target_messages = float(
        selected_pairs["total_communication_messages_doet"].mean()
    )
    kpi_rows = frame[frame["method"] == "kpi_cusum_trigger"].copy()
    if kpi_rows.empty or kpi_rows["method_variant"].nunique() != 1:
        raise ValueError(
            "validation requires exactly one private-local-KPI trigger variant"
        )
    kpi_reference_messages = float(
        kpi_rows["total_communication_messages"].mean()
    )
    # Freeze a transparent validation-only rate adjustment. It changes only
    # the local KPI residual magnitude, never the true label or global state.
    # The relationship to resulting traffic is approximate, so achieved
    # mismatch remains a required holdout table.
    kpi_signal_scale_unclipped = (
        target_messages / max(kpi_reference_messages, 1e-9)
    )
    kpi_signal_scale = min(4.0, max(0.25, kpi_signal_scale_unclipped))
    kpi_manifest_paths = sorted((results_root / "manifests").glob(
        "validation-*-kpi_cusum_trigger-*.json"
    ))
    if not kpi_manifest_paths:
        raise FileNotFoundError("no KPI-trigger validation manifest found")
    kpi_manifest = json.loads(
        kpi_manifest_paths[0].read_text(encoding="utf-8")
    )
    resolved_kpi = kpi_manifest["experiment_configuration"][
        "resolved_trigger"
    ]
    kpi_parameters = dict(resolved_kpi["parameters"])
    kpi_parameters["signal_scale"] = kpi_signal_scale
    kpi_trigger = {
        "normalizers_path": resolved_kpi["normalizers_path"],
        "normalizers_key": resolved_kpi["normalizers_key"],
        "parameters": kpi_parameters,
    }
    fixed_messages_total = float(
        selected_pairs["total_communication_messages_fixed"].sum()
    )
    fixed_active_total = float(
        selected_pairs["communication_active_decision_epochs_fixed"].sum()
    )
    fixed_messages_per_active_decision = (
        fixed_messages_total / max(fixed_active_total, 1.0)
    )
    target_intensive_decisions = (
        target_messages / max(fixed_messages_per_active_decision, 1e-9)
    )
    experiment_config = manifest["experiment_configuration"]
    resolved_scenario = experiment_config["resolved_scenario_name"]
    validation_horizon = int(
        experiment_config["scenarios"][resolved_scenario]["horizon"]
    )
    activation_interval = 2
    mean_agents = float(selected_pairs["n_agents"].mean())
    activation_opportunities = mean_agents * math.ceil(
        validation_horizon / activation_interval
    )
    random_probability = min(
        1.0, max(0.0, target_intensive_decisions / activation_opportunities)
    )
    periodic_candidates = range(2, validation_horizon + 1)
    periodic_interval = min(
        periodic_candidates,
        key=lambda interval: (
            abs(
                mean_agents * math.ceil(validation_horizon / interval)
                - target_intensive_decisions
            ),
            interval,
        ),
    )
    predicted_periodic_decisions = mean_agents * math.ceil(
        validation_horizon / periodic_interval
    )
    predicted_periodic_messages = (
        predicted_periodic_decisions * fixed_messages_per_active_decision
    )
    predicted_random_messages = (
        activation_opportunities * random_probability
        * fixed_messages_per_active_decision
    )
    controls = {
        "source": "validation only",
        "selected_method_variant": selected_variant,
        "target_active_agent_step_fraction": active_fraction,
        "target_counted_messages_per_episode": target_messages,
        "fixed_messages_per_active_decision": fixed_messages_per_active_decision,
        "target_intensive_decisions_per_episode": target_intensive_decisions,
        "activation_opportunity_interval": activation_interval,
        "quiet_local_planning_interval": 8,
        "random_gate_probability": random_probability,
        "periodic_interval": periodic_interval,
        "predicted_random_messages_per_episode": predicted_random_messages,
        "predicted_periodic_messages_per_episode": predicted_periodic_messages,
        "kpi_reference_messages_per_episode": kpi_reference_messages,
        "kpi_signal_scale_unclipped": kpi_signal_scale_unclipped,
        "kpi_signal_scale": kpi_signal_scale,
        "kpi_trigger": kpi_trigger,
        "rule": (
            "validation DOET total counted messages (including sketches) are "
            "converted to intensive decisions using fixed-control messages per "
            "active decision; random probability and periodic cadence minimize "
            "the resulting expected message-count mismatch. Inactive controls "
            "retain quiet local planning every eight periods. The private-KPI "
            "CUSUM residual is multiplied by the clipped DOET/KPI validation "
            "message ratio; achieved KPI mismatch is reported because this is "
            "an approximate, trajectory-dependent rate match."
        ),
    }
    comparison_path = validation_dir / "trigger_candidate_comparison.csv"
    _write_csv(comparison_path, summaries)
    selected_pairs_path = validation_dir / "selected_trigger_pairs.csv"
    paired_frames[selected_variant].to_csv(selected_pairs_path, index=False)
    controls_path = validation_dir / "budget_matched_controls.json"
    controls_path.write_text(
        json.dumps(controls, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection = {
        "status": selection_status,
        "selected_method_variant": selected_variant,
        "selected_summary": selected,
        "selection_rule": (
            "eligible iff mean degradation <=1% in both applications and every "
            "application-regime mean <=2%; among eligible maximize counted message "
            "reduction, then minimize nominal false-active fraction; if none, minimize "
            "worst regime degradation and label validation failure"
        ),
        "candidate_count": len(summaries),
        "fixed_primary_benchmark": "fixed_always_on",
        "selected_trigger": trigger,
        "budget_matched_controls": controls,
        "generated_at": _utc_now(),
        "input_checksum": _sha256(episodes_path),
    }
    selection_path = validation_dir / "trigger_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol_dir = results_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    selected_trigger_path = protocol_dir / "selected_trigger.json"
    selected_trigger_path.write_text(
        json.dumps(
            {
                "status": "selected on validation; not yet holdout-frozen",
                "method_variant": selected_variant,
                "trigger": trigger,
                "budget_matched_controls": controls,
                "selection_checksum": _sha256(selection_path),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    readme = validation_dir / "README.md"
    readme.write_text(
        "# DOET validation selection\n\n"
        "Selected `%s`. %s\n\n"
        "The selection used the preregistered lexicographic rule in "
        "`trigger_selection.json`. The data are validation evidence, not the new "
        "locked holdout. Budget-matched control rates were derived here and must "
        "not be changed after holdout freeze.\n" % (selected_variant, selection_status),
        encoding="utf-8",
    )
    return {
        "status": selection_status,
        "selected_method_variant": selected_variant,
        "eligible_candidates": len(eligible),
        "candidate_count": len(summaries),
        "selection_path": str(selection_path),
        "selected_trigger_path": str(selected_trigger_path),
    }
