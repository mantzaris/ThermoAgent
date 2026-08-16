"""Communication-cost and distributed-estimation analysis for V6 sketches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .v5_analysis import paired_bootstrap
from .v5_experiments import atomic_json, write_csv
from .v6_analysis import (
    FEATURE_BLOCKS, contexts_to_risk_frame, fit_group_excluded_model,
    prepare_risk_frame, select_at_coverage,
)


def analyze_sketch_stage(stage_root: Path, destination: Path) -> Dict[str, Any]:
    summaries = pd.read_csv(stage_root / "episode_summary.csv")
    consensus = pd.read_csv(stage_root / "distributed_consensus.csv")
    candidates = prepare_risk_frame(pd.read_csv(stage_root / "candidate_decisions.csv"))
    cost_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    for policy, subset in summaries.groupby("sketch_policy", sort=True):
        cost_rows.append({
            "sketch_policy": policy,
            "episodes": int(len(subset)),
            "sketch_messages_mean": float(subset.sketch_messages.mean()),
            "sketch_bytes_mean": float(subset.sketch_bytes.mean()),
            "operational_messages_mean": float(subset.operational_messages.mean()),
            "total_messages_mean": float(subset.total_messages.mean()),
            "total_bytes_mean": float(subset.total_bytes.mean()),
            "sketch_latency_seconds_mean": float(subset.sketch_latency.mean()),
        })
    for policy, subset in consensus.groupby("sketch_policy", sort=True):
        error_rows.append({
            "sketch_policy": policy,
            "estimates": int(len(subset)),
            "distributed_estimation_mae": float(subset.evaluator_distributed_error.mean()),
            "distributed_estimation_median": float(subset.evaluator_distributed_error.median()),
            "distributed_estimation_maximum": float(subset.evaluator_distributed_error.max()),
            "missing_agents_mean": float(subset.missing_agent_count.mean()),
        })
    keys = ["application", "regime", "information_condition", "environment_seed"]
    always = summaries[summaries.sketch_policy == "always_on"].set_index(keys)
    event = summaries[summaries.sketch_policy == "event_triggered"].set_index(keys)
    if not always.index.is_unique or not event.index.is_unique:
        raise ValueError("sketch comparison requires one row per matched panel")
    matched = always[["sketch_messages", "sketch_bytes", "total_messages", "total_bytes"]].join(
        event[["sketch_messages", "sketch_bytes", "total_messages", "total_bytes"]],
        lsuffix="_always", rsuffix="_event", how="inner",
    )
    for measure in ("sketch_messages", "sketch_bytes", "total_messages", "total_bytes"):
        matched[measure + "_reduction"] = (
            matched[measure + "_always"] - matched[measure + "_event"]
        ) / matched[measure + "_always"].clip(lower=1e-9)
    reduction_rows: List[Dict[str, Any]] = []
    for application, subset in matched.reset_index().groupby("application", sort=True):
        row: Dict[str, Any] = {"application": application, "matched_panels": int(len(subset))}
        for measure in ("sketch_messages", "sketch_bytes", "total_messages", "total_bytes"):
            interval = paired_bootstrap(subset[measure + "_reduction"], 10000, 66810)
            row[measure + "_reduction"] = interval["mean"]
            row[measure + "_ci95_low"] = interval["ci_low"]
            row[measure + "_ci95_high"] = interval["ci_high"]
        reduction_rows.append(row)

    candidates = candidates[~candidates.proposed_action.isin(
        ["verify", "request_peer_evidence", "defer", "no_action"]
    )].reset_index(drop=True)
    safety_rows: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian", "utility_restoration"):
        subset = candidates[
            (candidates.application == application)
            & (candidates.information_condition == "private_fragmented")
        ]
        event_rows = subset[subset.sketch_policy == "event_triggered"]
        always_rows = subset[subset.sketch_policy == "always_on"]
        if event_rows.empty or always_rows.empty:
            continue
        event_selections: List[pd.DataFrame] = []
        always_selections: List[pd.DataFrame] = []
        for split_family in sorted(event_rows.split_family.unique()):
            model, _ = fit_group_excluded_model(
                event_rows, "combined_generalized_entropic", split_family,
            )
            for source, target in (
                (event_rows, event_selections), (always_rows, always_selections),
            ):
                test = source[source.split_family == split_family].reset_index(drop=True)
                scores = model.predict_proba(test)[:, 1]
                target.append(select_at_coverage(test, scores, 0.50))
        event_selected = pd.concat(event_selections, ignore_index=True)
        always_selected = pd.concat(always_selections, ignore_index=True)
        paired = event_selected[["cluster_id", "harmful_action_rate", "mean_causal_utility"]].merge(
            always_selected[["cluster_id", "harmful_action_rate", "mean_causal_utility"]],
            on="cluster_id", suffixes=("_event", "_always"), validate="one_to_one",
        )
        paired["harm_rate_degradation"] = (
            paired.harmful_action_rate_event - paired.harmful_action_rate_always
        )
        paired["utility_degradation"] = (
            paired.mean_causal_utility_always - paired.mean_causal_utility_event
        )
        harm = paired_bootstrap(paired.harm_rate_degradation, 10000, 66811)
        utility = paired_bootstrap(paired.utility_degradation, 10000, 66812)
        safety_rows.append({
            "application": application,
            "matched_panels": int(len(paired)),
            "event_minus_always_harm_rate": harm["mean"],
            "harm_ci95_low": harm["ci_low"],
            "harm_ci95_high": harm["ci_high"],
            "always_minus_event_causal_utility": utility["mean"],
            "utility_ci95_low": utility["ci_low"],
            "utility_ci95_high": utility["ci_high"],
        })
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "sketch_costs.csv", cost_rows)
    write_csv(destination / "distributed_estimation_error.csv", error_rows)
    write_csv(destination / "event_vs_always_reductions.csv", reduction_rows)
    write_csv(destination / "event_vs_always_safety.csv", safety_rows)
    report = {
        "stage": "development_sketch_communication",
        "episodes": int(len(summaries)),
        "costs": cost_rows,
        "estimation": error_rows,
        "reductions": reduction_rows,
        "safety": safety_rows,
        "all_sketch_traffic_counted": True,
    }
    atomic_json(destination / "communication_analysis.json", report)
    return report
