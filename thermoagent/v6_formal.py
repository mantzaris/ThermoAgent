"""Guarded one-shot validation and holdout execution for V6.

The module is intentionally unusable until the preceding gate report grants
permission.  Formal controllers are fitted only on frozen development rows;
validation and holdout rows never participate in fitting or threshold choice.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml

from .events import sha256_file
from .v5_analysis import paired_bootstrap
from .v5_experiments import atomic_json, write_csv
from .v6_analysis import (
    FEATURE_BLOCKS, FittedRiskController, fit_group_excluded_conformal,
    make_risk_pipeline, prepare_risk_frame,
)
from .v6_experiments import aggregate_stage, run_episode, utc_now


def _json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _models(
    results_root: Path,
) -> Tuple[Dict[Tuple[str, str, str], Any], str]:
    candidates = prepare_risk_frame(pd.read_csv(
        results_root / "development" / "formal_reference" / "candidate_decisions.csv"
    ))
    candidates = candidates[
        ~candidates.proposed_action.isin(
            ["verify", "request_peer_evidence", "defer", "no_action"]
        )
    ].reset_index(drop=True)
    report = _json(results_root / "development" / "risk_analysis" / "risk_analysis.json")
    baseline = str(report["selected_strongest_nonentropic_baseline"])
    folds = pd.read_csv(
        results_root / "development" / "risk_analysis" / "grouped_fold_audit.csv"
    )
    output: Dict[Tuple[str, str, str], Any] = {}
    manifest_rows: List[Dict[str, Any]] = []
    checkpoint_root = results_root / "protocol" / "frozen_risk_models"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for (application, condition), subset in candidates.groupby(
        ["application", "information_condition"], sort=True,
    ):
        labels = subset.harmful_label.to_numpy(dtype=int)
        for block in (baseline, "combined_generalized_entropic"):
            selected = folds[
                (folds.application == application)
                & (folds.information_condition == condition)
                & (folds.feature_block == block)
            ].selected_c
            if selected.empty:
                raise RuntimeError("missing frozen-development regularization selection")
            # Median fold choice is deterministic and was specified before any
            # formal outcome; validation is never used to choose C.
            c_value = float(np.median(selected.to_numpy(dtype=float)))
            if block == "conformal_risk_control":
                held_out = sorted(subset.split_family.unique())[-1]
                model, c_value = fit_group_excluded_conformal(subset, held_out)
            else:
                model = make_risk_pipeline(FEATURE_BLOCKS[block], c_value)
                model.fit(subset, labels)
            output[(str(application), str(condition), block)] = model
            path = checkpoint_root / ("%s-%s-%s.pkl.gz" % (application, condition, block))
            with path.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
                    pickle.dump(model, handle, protocol=4)
            manifest_rows.append({
                "application": application,
                "information_condition": condition,
                "feature_block": block,
                "selected_c": c_value,
                "development_rows": len(subset),
                "checkpoint": str(path.relative_to(results_root)),
                "sha256": sha256_file(path),
            })
    write_csv(results_root / "protocol" / "frozen_risk_model_manifest.csv", manifest_rows)
    return output, baseline


def _verify_seal(results_root: Path, stage: str) -> pd.DataFrame:
    if stage not in ("validation", "holdout"):
        raise ValueError("formal stage must be validation or holdout")
    freeze = _json(results_root / "protocol" / "freeze_manifest.json")
    key = "%s_manifest" % stage
    checksum_key = "%s_manifest_sha256" % stage
    path = results_root / freeze[key]
    if sha256_file(path) != freeze[checksum_key]:
        raise RuntimeError("sealed %s manifest checksum mismatch" % stage)
    return pd.read_csv(path)


def _assert_unlocked(results_root: Path, stage: str) -> None:
    if stage == "validation":
        gate = _json(results_root / "development" / "gate_status.json")
        unlocked = gate.get("validation_unlocked", False)
        reason = "all frozen development gates did not pass"
    else:
        gate = _json(results_root / "validation" / "gate_status.json")
        unlocked = gate.get("holdout_unlocked", False)
        reason = "validation gates did not pass"
    if not unlocked:
        raise RuntimeError("%s is prospectively locked: %s" % (stage, reason))


def run_formal_stage(repository: Path, results_root: Path, stage: str) -> Dict[str, Any]:
    _assert_unlocked(results_root, stage)
    rows = _verify_seal(results_root, stage)
    outcome_seal = results_root / "logs" / (stage + "_outcome_seal.json")
    if outcome_seal.exists() and _json(outcome_seal).get("status") == "complete":
        raise RuntimeError("%s already completed; one-shot policy forbids rerun" % stage)
    atomic_json(outcome_seal, {
        "stage": stage, "status": "running_outcomes_sealed",
        "started_at": utc_now(), "completed_panels": 0,
        "total_panels": int(len(rows)),
        "permitted_monitoring": ["process_health", "completion_count", "schema", "finite_values", "disk"],
    })
    models, baseline = _models(results_root)
    completed = 0
    for row in rows.itertuples(index=False):
        for block in (baseline, "combined_generalized_entropic"):
            controller = FittedRiskController(
                models[(row.application, row.information_condition, block)],
                str(row.regime), 0.50, 1,
            )
            run_episode(
                repository, results_root, stage,
                str(row.application), str(row.regime),
                str(row.information_condition), int(row.environment_seed),
                "%s_frozen" % block, 0.50, "event_triggered", 1,
                resume=True, controller_override=controller,
                extra_configuration={
                    "formal_stage": stage,
                    "frozen_development_model": True,
                    "outcome_sealed_during_execution": True,
                },
            )
        completed += 1
        if completed % 10 == 0 or completed == len(rows):
            atomic_json(outcome_seal, {
                "stage": stage,
                "status": "running_outcomes_sealed" if completed < len(rows) else "execution_complete_analysis_unopened",
                "started_at": _json(outcome_seal).get("started_at"),
                "completed_panels": completed, "total_panels": int(len(rows)),
            })
    aggregate = aggregate_stage(results_root, stage)
    analysis = analyze_formal_stage(results_root, stage, baseline)
    atomic_json(outcome_seal, {
        "stage": stage, "status": "complete", "completed_at": utc_now(),
        "completed_panels": completed, "total_panels": int(len(rows)),
        "analysis_opened_only_after_execution": True,
    })
    return {"execution": aggregate, "analysis": analysis}


def analyze_formal_stage(results_root: Path, stage: str, baseline: str) -> Dict[str, Any]:
    destination = results_root / stage
    summaries = pd.read_csv(destination / "episode_summary.csv")
    summaries["autonomous_harm_rate"] = (
        summaries.autonomous_harmful_actions
        / summaries.autonomous_completed_actions.clip(lower=1)
    )
    rows: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    sources: Dict[Tuple[str, str], pd.DataFrame] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        for condition in ("private_fragmented", "public_shared"):
            subset = summaries[
                (summaries.application == application)
                & (summaries.information_condition == condition)
            ]
            first = subset[subset.controller == "%s_frozen" % baseline]
            second = subset[subset.controller == "combined_generalized_entropic_frozen"]
            paired = first.merge(
                second,
                on=["application", "regime", "information_condition", "environment_seed"],
                suffixes=("_baseline", "_combined"), validate="one_to_one",
            )
            paired["harm_rate_reduction"] = paired.autonomous_harm_rate_baseline - paired.autonomous_harm_rate_combined
            paired["relative_service_loss_degradation"] = (
                paired.service_loss_combined - paired.service_loss_baseline
            ) / paired.service_loss_baseline.abs().clip(lower=1e-9)
            paired["utility_gain"] = paired.net_causal_utility_combined - paired.net_causal_utility_baseline
            sources[(application, condition)] = paired
            harm = paired_bootstrap(paired.harm_rate_reduction, 10000, 67601)
            service = paired_bootstrap(paired.relative_service_loss_degradation, 10000, 67602)
            utility = paired_bootstrap(paired.utility_gain, 10000, 67603)
            rows.append({
                "application": application, "information_condition": condition,
                "baseline": baseline, "panels": len(paired),
                "harm_rate_reduction": harm["mean"], "harm_ci95_low": harm["ci_low"], "harm_ci95_high": harm["ci_high"],
                "relative_service_loss_degradation": service["mean"], "service_ci95_low": service["ci_low"], "service_ci95_high": service["ci_high"],
                "net_causal_utility_gain": utility["mean"], "utility_ci95_low": utility["ci_low"], "utility_ci95_high": utility["ci_high"],
                "baseline_action_coverage": float((
                    first.autonomous_executions
                    / first.eligible_operational_proposals.clip(lower=1)
                ).mean()),
                "combined_action_coverage": float((
                    second.autonomous_executions
                    / second.eligible_operational_proposals.clip(lower=1)
                ).mean()),
                "combined_operator_minutes": float(second.operator_minutes.mean()),
            })
            for regime, group in paired.groupby("regime", sort=True):
                interval = paired_bootstrap(group.harm_rate_reduction, 10000, 67604)
                regime_rows.append({
                    "application": application, "information_condition": condition,
                    "regime": regime, "panels": len(group),
                    "harm_rate_reduction": interval["mean"],
                    "harm_ci95_low": interval["ci_low"], "harm_ci95_high": interval["ci_high"],
                })
    interactions = []
    for application in ("commercial", "humanitarian", "utility_restoration"):
        private = sources[(application, "private_fragmented")][["environment_seed", "regime", "harm_rate_reduction"]]
        public = sources[(application, "public_shared")][["environment_seed", "regime", "harm_rate_reduction"]]
        matched = private.merge(public, on=["environment_seed", "regime"], suffixes=("_private", "_public"), validate="one_to_one")
        difference = matched.harm_rate_reduction_private - matched.harm_rate_reduction_public
        interval = paired_bootstrap(difference, 10000, 67605)
        interactions.append({
            "application": application, "matched_panels": len(matched),
            "private_minus_public_harm_reduction": interval["mean"],
            "ci95_low": interval["ci_low"], "ci95_high": interval["ci_high"],
        })
    write_csv(destination / "paired_effects.csv", rows)
    write_csv(destination / "regime_effects.csv", regime_rows)
    write_csv(destination / "fragmentation_interaction.csv", interactions)
    timing_rows: List[Dict[str, Any]] = []
    for (application, condition, controller_name), group in summaries.groupby(
        ["application", "information_condition", "controller"], sort=True,
    ):
        disrupted = group[group.regime != "nominal"]
        nominal = group[group.regime == "nominal"]
        timing_rows.append({
            "application": application,
            "information_condition": condition,
            "controller": controller_name,
            "disrupted_panels": int(len(disrupted)),
            "post_disruption_activation_rate": float(
                (disrupted.post_disruption_escalations > 0).mean()
            ) if len(disrupted) else None,
            "timely_activation_rate_by_step_4": float(
                disrupted.timely_post_disruption_activation_by_step_4.mean()
            ) if len(disrupted) else None,
            "pre_disruption_false_activation_rate": float(
                (disrupted.pre_disruption_escalations > 0).mean()
            ) if len(disrupted) else None,
            "mean_escalations_per_disrupted_panel": float(
                disrupted.escalations.mean()
            ) if len(disrupted) else None,
            "nominal_panels": int(len(nominal)),
            "nominal_false_activation_rate": float(
                nominal.nominal_false_activation.mean()
            ) if len(nominal) else None,
            "mean_operator_minutes": float(group.operator_minutes.mean()),
            "mean_maximum_queue_length": float(group.maximum_queue_length.mean()),
        })
    write_csv(destination / "trigger_timing.csv", timing_rows)
    report = {
        "stage": stage, "baseline": baseline, "episodes": len(summaries),
        "paired_effects": rows, "fragmentation_interaction": interactions,
        "trigger_timing": timing_rows,
    }
    atomic_json(destination / "formal_analysis.json", report)
    return report


def evaluate_validation_gates(repository: Path, results_root: Path) -> Dict[str, Any]:
    config = yaml.safe_load((repository / "configs" / "generalized_entropic_consensus_v6.yaml").read_text(encoding="utf-8"))
    rows = pd.read_csv(results_root / "validation" / "paired_effects.csv")
    regimes = pd.read_csv(results_root / "validation" / "regime_effects.csv")
    interaction = pd.read_csv(results_root / "validation" / "fragmentation_interaction.csv")
    timing = pd.read_csv(results_root / "validation" / "trigger_timing.csv")
    checks: List[Dict[str, Any]] = []
    for application in ("humanitarian", "utility_restoration"):
        row = rows[(rows.application == application) & (rows.information_condition == "private_fragmented")].iloc[0]
        positive = int(((regimes.application == application) & (regimes.information_condition == "private_fragmented") & (regimes.harm_rate_reduction > 0)).sum())
        checks.extend([
            {"condition": application + "_harm_reduction", "observed": row.harm_rate_reduction, "passed": bool(row.harm_rate_reduction >= .03)},
            {"condition": application + "_harm_ci", "observed": row.harm_ci95_low, "passed": bool(row.harm_ci95_low > 0)},
            {"condition": application + "_service", "observed": row.service_ci95_high, "passed": bool(row.service_ci95_high <= .02)},
            {"condition": application + "_utility", "observed": row.utility_ci95_low, "passed": bool(row.utility_ci95_low >= -.02)},
            {"condition": application + "_positive_regimes", "observed": positive, "passed": positive >= 3},
        ])
        mechanism = interaction[interaction.application == application].iloc[0]
        checks.extend([
            {"condition": application + "_interaction", "observed": mechanism.private_minus_public_harm_reduction, "passed": bool(mechanism.private_minus_public_harm_reduction >= .02)},
            {"condition": application + "_interaction_ci", "observed": mechanism.ci95_low, "passed": bool(mechanism.ci95_low > 0)},
        ])
        timing_row = timing[
            (timing.application == application)
            & (timing.information_condition == "private_fragmented")
            & (timing.controller == "combined_generalized_entropic_frozen")
        ].iloc[0]
        checks.extend([
            {"condition": application + "_timely_activation", "observed": timing_row.timely_activation_rate_by_step_4, "passed": bool(timing_row.timely_activation_rate_by_step_4 >= .75)},
            {"condition": application + "_pre_disruption_false_activation", "observed": timing_row.pre_disruption_false_activation_rate, "passed": bool(timing_row.pre_disruption_false_activation_rate <= .10)},
            {"condition": application + "_nominal_false_activation", "observed": timing_row.nominal_false_activation_rate, "passed": bool(timing_row.nominal_false_activation_rate <= .10)},
            {"condition": application + "_escalation_burden", "observed": timing_row.mean_escalations_per_disrupted_panel, "passed": bool(timing_row.mean_escalations_per_disrupted_panel <= 3.5)},
        ])
    passed = all(value["passed"] for value in checks)
    report = {
        "stage": "validation", "checks": checks,
        "all_required_validation_gates_passed": passed,
        "holdout_unlocked": passed,
        "thresholds_unchanged": True,
    }
    atomic_json(results_root / "validation" / "gate_status.json", report)
    write_csv(results_root / "validation" / "gate_checks.csv", checks)
    return report
