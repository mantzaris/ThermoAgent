"""Post-development V5 abstention audit used to motivate V6.

This module is intentionally read-only with respect to the frozen V5 result
namespace.  It uses V5 cross-fitted scores and candidates, writes only to the
V6 namespace, and cannot alter the original V5 gate disposition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .v5_analysis import paired_bootstrap
from .v5_experiments import atomic_json, write_csv


PRIMARY_REGIMES = ("partition", "telemetry_integrity", "compound", "ood")
SCORE_BLOCK = "kpi_plus_entropy_disagreement"
CONSENSUS_THRESHOLD = 0.42
PANEL_BUDGET = 2
ESCALATION_MESSAGES = 2
ESCALATION_BYTES = 512


def _boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).astype(bool)


def _candidate_frame(v5_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(
        v5_root / "development" / "development_primary_v2" / "candidate_interventions.csv"
    )
    for column in (
        "beneficial", "harmful", "accepted_action", "reached_next_stage",
        "reached_service", "changed_commitment",
    ):
        candidates[column] = _boolean(candidates[column])
    predictions = pd.read_csv(v5_root / "statistics" / "candidate_crossfit_predictions.csv.gz")
    predictions = predictions[predictions["feature_block"] == SCORE_BLOCK]
    score_map = predictions.set_index("candidate_id")["predicted_value"]
    candidates["predicted_value"] = candidates["candidate_id"].map(score_map)
    if candidates["predicted_value"].isna().any():
        raise ValueError("V5 reanalysis could not map every candidate to its frozen score")
    episodes = pd.read_csv(
        v5_root / "development" / "development_primary_v2" / "episode_summary.csv"
    )
    return candidates, episodes


def _ranked_indices(group: pd.DataFrame) -> List[int]:
    ordered = group.sort_values(
        ["predicted_value", "incident_id", "action"],
        ascending=[False, True, True], kind="mergesort",
    )
    return [int(value) for value in ordered.index]


def _select_panel(
    group: pd.DataFrame,
    policy: str,
    score_threshold: float = 0.0,
) -> Dict[str, Any]:
    selected: List[int] = []
    selected_incidents = set()
    escalated: List[int] = []
    for position in _ranked_indices(group):
        row = group.loc[position]
        incident_id = str(row["incident_id"])
        if incident_id in selected_incidents or float(row["visible_severity"]) < 0.30:
            continue
        score = float(row["predicted_value"])
        confidence = float(row["consensus_confidence"])
        if policy == "original_safe":
            eligible = score > 0.0 and confidence >= CONSENSUS_THRESHOLD
        elif policy == "same_score_no_consensus":
            eligible = score > 0.0
        elif policy == "mandatory_intervention":
            eligible = True
        elif policy == "coverage_matched_no_consensus":
            eligible = score > float(score_threshold)
        elif policy == "operator_budget_matched_escalation":
            eligible = score > 0.0
        else:
            raise ValueError("unknown V5 audit policy: %s" % policy)
        if not eligible:
            continue
        selected.append(position)
        selected_incidents.add(incident_id)
        if policy == "operator_budget_matched_escalation" and confidence < CONSENSUS_THRESHOLD:
            escalated.append(position)
        if len(selected) >= PANEL_BUDGET:
            break

    eligible_incidents = int(
        group[group["visible_severity"] >= 0.30]["incident_id"].nunique()
    )
    selected_frame = group.loc[selected] if selected else group.iloc[0:0]
    first_per_incident = group.drop_duplicates("incident_id")
    baseline_loss = float(first_per_incident["loss_without"].sum())
    selected_effect = float(selected_frame["causal_effect"].sum()) if selected else 0.0
    harmful_count = int(selected_frame["harmful"].sum()) if selected else 0
    beneficial_count = int(selected_frame["beneficial"].sum()) if selected else 0
    neutral_count = len(selected) - harmful_count - beneficial_count
    # The V5 triage policies used bounded simulated-operator minutes for every
    # selected intervention.  The escalation variant retains the identical
    # total budget and reports which selected low-consensus cases specifically
    # entered the escalation queue.
    operator_minutes = float(selected_frame["operator_minutes"].sum()) if selected else 0.0
    return {
        "cluster_id": str(group["cluster_id"].iloc[0]),
        "application": str(group["application"].iloc[0]),
        "regime": str(group["regime"].iloc[0]),
        "information_condition": str(group["information_condition"].iloc[0]),
        "environment_seed": int(group["environment_seed"].iloc[0]),
        "topology_family": str(group["topology_family"].iloc[0]),
        "scenario_family": str(group["scenario_family"].iloc[0]),
        "policy": policy,
        "score_threshold": float(score_threshold),
        "eligible_incidents": eligible_incidents,
        "selected_count": len(selected),
        "action_coverage": float(len(selected) / max(eligible_incidents, 1)),
        "beneficial_count": beneficial_count,
        "neutral_count": neutral_count,
        "harmful_count": harmful_count,
        "harmful_action_rate": float(harmful_count / max(len(selected), 1)),
        "causal_utility": selected_effect,
        "baseline_service_loss": baseline_loss,
        "service_loss": baseline_loss - selected_effect,
        "operator_escalations": len(escalated),
        "operator_minutes": operator_minutes,
        "maximum_queue_length": len(escalated),
        "selected_candidates": ";".join(selected_frame["candidate_id"].astype(str)),
        "escalated_candidates": ";".join(group.loc[escalated, "candidate_id"].astype(str)),
    }


def _select(
    frame: pd.DataFrame,
    policy: str,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    rows = [
        _select_panel(group, policy, score_threshold)
        for _, group in frame.groupby("cluster_id", sort=True)
    ]
    return pd.DataFrame(rows)


def _coverage_threshold(frame: pd.DataFrame, target_count: int) -> Tuple[float, int]:
    positive = sorted(
        {float(value) for value in frame.loc[frame["predicted_value"] > 0.0, "predicted_value"]}
    )
    thresholds = [0.0] + positive
    cache: Dict[int, int] = {}

    def count_at(index: int) -> int:
        if index not in cache:
            selected = _select(
                frame, "coverage_matched_no_consensus", thresholds[index]
            )
            cache[index] = int(selected["selected_count"].sum())
        return cache[index]

    # Selected count is monotone non-increasing in the score threshold.  A
    # binary search avoids thousands of repeated panel rankings in this audit.
    low, high = 0, len(thresholds) - 1
    while low <= high:
        middle = (low + high) // 2
        if count_at(middle) > int(target_count):
            low = middle + 1
        else:
            high = middle - 1
    candidates = sorted({max(0, min(len(thresholds) - 1, value)) for value in (low - 1, low, low + 1)})
    best = (float("inf"), 0.0, 0)
    for index in candidates:
        count = count_at(index)
        candidate = (abs(count - int(target_count)), float(thresholds[index]), count)
        if candidate < best:
            best = candidate
    return float(best[1]), int(best[2])


def _summary_rows(panel_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for (application, policy), group in panel_rows.groupby(["application", "policy"], sort=True):
        summaries.append({
            "application": application,
            "policy": policy,
            "independent_panels": int(len(group)),
            "eligible_incidents": int(group["eligible_incidents"].sum()),
            "selected_actions": int(group["selected_count"].sum()),
            "action_coverage": float(group["selected_count"].sum() / max(group["eligible_incidents"].sum(), 1)),
            "beneficial_actions": int(group["beneficial_count"].sum()),
            "neutral_actions": int(group["neutral_count"].sum()),
            "harmful_actions": int(group["harmful_count"].sum()),
            "harmful_action_rate": float(group["harmful_count"].sum() / max(group["selected_count"].sum(), 1)),
            "mean_panel_causal_utility": float(group["causal_utility"].mean()),
            "mean_panel_service_loss": float(group["service_loss"].mean()),
            "operator_escalations": int(group["operator_escalations"].sum()),
            "mean_operator_minutes": float(group["operator_minutes"].mean()),
            "maximum_queue_length": int(group["maximum_queue_length"].max()),
            "mean_total_messages": float(group["total_messages"].mean()),
            "mean_total_bytes": float(group["total_bytes"].mean()),
        })
    return summaries


def _paired_intervals(panel_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    comparisons = (
        "same_score_no_consensus",
        "mandatory_intervention",
        "coverage_matched_no_consensus",
        "operator_budget_matched_escalation",
    )
    rows: List[Dict[str, Any]] = []
    for application in sorted(panel_rows["application"].unique()):
        app = panel_rows[panel_rows["application"] == application]
        safe = app[app["policy"] == "original_safe"].set_index("cluster_id")
        for comparator in comparisons:
            other = app[app["policy"] == comparator].set_index("cluster_id")
            common = safe.index.intersection(other.index)
            for metric in (
                "action_coverage", "harmful_action_rate", "causal_utility",
                "service_loss", "operator_minutes", "total_messages", "total_bytes",
            ):
                delta = safe.loc[common, metric].to_numpy(dtype=float) - other.loc[common, metric].to_numpy(dtype=float)
                interval = paired_bootstrap(delta, replicates=10000, seed=66051)
                rows.append({
                    "application": application,
                    "reference_policy": "original_safe",
                    "comparator_policy": comparator,
                    "metric": metric,
                    "difference_definition": "reference_minus_comparator",
                    "paired_mean_difference": interval["mean"],
                    "ci95_low": interval["ci_low"],
                    "ci95_high": interval["ci_high"],
                    "independent_panels": interval["n_clusters"],
                    "bootstrap_replicates": interval["bootstrap_replicates"],
                    "bootstrap_seed": interval["bootstrap_seed"],
                })
    return rows


def _qwen_effect_audit(v5_root: Path) -> List[Dict[str, Any]]:
    decisions = pd.read_csv(
        v5_root / "development" / "real_qwen_qualification" / "decision_epochs.csv"
    )
    rows: List[Dict[str, Any]] = []
    for application, group in decisions.groupby("application", sort=True):
        effect = group["causal_effect"].to_numpy(dtype=float)
        rows.append({
            "application": str(application),
            "episodes": int(group["run_id"].nunique()),
            "decision_epochs": int(len(group)),
            "beneficial_actions": int((effect > 1e-9).sum()),
            "neutral_actions": int((np.abs(effect) <= 1e-9).sum()),
            "harmful_actions": int((effect < -1e-9).sum()),
            "harmful_action_rate": float((effect < -1e-9).mean()),
            "mean_causal_effect": float(effect.mean()),
            "accepted_action_rate": float(_boolean(group["material_action_accepted"]).mean()),
            "accepted_metric_caveat": "Includes accepted verification/evidence actions; not exclusively material flow.",
            "service_reaching_rate": float(_boolean(group["reached_service"]).mean()),
        })
    return rows


def run_v5_abstention_reanalysis(repository: Path, output_root: Path) -> Dict[str, Any]:
    """Run the fair V5 post-development audit and write only V6 artifacts."""
    v5_root = repository / "results" / "human_operator_v5"
    candidates, episodes = _candidate_frame(v5_root)
    candidates = candidates[
        (candidates["information_condition"] == "private_fragmented")
        & (candidates["regime"].isin(PRIMARY_REGIMES))
    ].reset_index(drop=True)
    episode_costs = episodes[[
        "cluster_id", "sketch_messages", "sketch_bytes",
        "operational_messages", "operational_bytes",
    ]].drop_duplicates("cluster_id")

    panel_frames: List[pd.DataFrame] = []
    thresholds: List[Dict[str, Any]] = []
    for application in sorted(candidates["application"].unique()):
        app = candidates[candidates["application"] == application].reset_index(drop=True)
        safe = _select(app, "original_safe")
        target = int(safe["selected_count"].sum())
        threshold, achieved = _coverage_threshold(app, target)
        thresholds.append({
            "application": application,
            "calibration_evidence": "post-development V5 audit only",
            "target_safe_selected_actions": target,
            "selected_threshold": threshold,
            "achieved_selected_actions": achieved,
            "absolute_count_mismatch": abs(achieved - target),
        })
        panel_frames.extend([
            safe,
            _select(app, "same_score_no_consensus"),
            _select(app, "mandatory_intervention"),
            _select(app, "coverage_matched_no_consensus", threshold),
            _select(app, "operator_budget_matched_escalation"),
        ])

    panel_rows = pd.concat(panel_frames, ignore_index=True)
    panel_rows = panel_rows.merge(episode_costs, on="cluster_id", how="left", validate="many_to_one")
    panel_rows["escalation_messages"] = panel_rows["operator_escalations"] * ESCALATION_MESSAGES
    panel_rows["escalation_bytes"] = panel_rows["operator_escalations"] * ESCALATION_BYTES
    panel_rows["total_messages"] = (
        panel_rows["sketch_messages"] + panel_rows["operational_messages"]
        + panel_rows["escalation_messages"]
    )
    panel_rows["total_bytes"] = (
        panel_rows["sketch_bytes"] + panel_rows["operational_bytes"]
        + panel_rows["escalation_bytes"]
    )

    destination = output_root / "v5_reanalysis"
    destination.mkdir(parents=True, exist_ok=True)
    summaries = _summary_rows(panel_rows)
    intervals = _paired_intervals(panel_rows)
    qwen_audit = _qwen_effect_audit(v5_root)
    write_csv(destination / "abstention_policy_panel_results.csv", panel_rows.to_dict("records"))
    write_csv(destination / "abstention_policy_summary.csv", summaries)
    write_csv(destination / "paired_bootstrap_intervals.csv", intervals)
    write_csv(destination / "coverage_matching_calibration.csv", thresholds)
    write_csv(destination / "qwen_effect_audit.csv", qwen_audit)

    report = {
        "study": "V5 abstention post-development audit for V6 design",
        "evidence_status": "post-development descriptive reanalysis; cannot unlock or alter V5 gates",
        "source_v5_commit": "c895235d02dd05ccc9315621d818def9345a398c",
        "source_score_block": SCORE_BLOCK,
        "consensus_threshold": CONSENSUS_THRESHOLD,
        "panel_budget": PANEL_BUDGET,
        "applications": sorted(candidates["application"].unique().tolist()),
        "independent_panels": int(candidates["cluster_id"].nunique()),
        "candidate_rows": int(len(candidates)),
        "policies": [
            "original_safe", "same_score_no_consensus", "mandatory_intervention",
            "coverage_matched_no_consensus", "operator_budget_matched_escalation",
        ],
        "communication_accounting": {
            "base_sketch_and_operational_traffic": "read from each frozen V5 episode summary",
            "escalation_messages_per_case": ESCALATION_MESSAGES,
            "escalation_bytes_per_case": ESCALATION_BYTES,
            "note": "Escalation traffic is a transparent post-development accounting convention, not a rerun of V5 dynamics.",
        },
        "summary": summaries,
        "coverage_thresholds": thresholds,
        "qwen_effect_audit": qwen_audit,
    }
    atomic_json(destination / "abstention_reanalysis.json", report)
    return report
