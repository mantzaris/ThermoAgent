"""Panel-level pilot and formal analysis for V7 complexity interactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .v5_analysis import paired_bootstrap
from .v5_experiments import atomic_json, write_csv
from .v7_io import read_csv_artifact


RISK_COLUMNS = {
    "kpi_confidence": "risk_kpi_confidence",
    "predictive_uncertainty": "risk_predictive_uncertainty",
    "shannon_js": "risk_shannon_js",
    "generalized_tsallis_gini": "risk_generalized_tsallis_gini",
    "graph_disagreement": "risk_graph_disagreement",
    "combined_generalized_entropic": "risk_combined_generalized_entropic",
}
LEVEL_VALUE = {"low": 0.0, "medium": 0.5, "high": 1.0}
SIZE_VALUE = {"small": 0.0, "medium": 0.5, "large": 1.0}


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(
        average_precision_score(labels, scores)
    )


def _select_at_coverage(
    frame: pd.DataFrame, score_column: str, coverage: float,
) -> Dict[str, float]:
    ordered = frame.sort_values(
        [score_column, "agent_id", "target", "step"],
        ascending=[True, True, True, True], kind="mergesort",
    )
    count = int(round(float(coverage) * len(ordered)))
    count = max(1, min(len(ordered), count)) if len(ordered) else 0
    selected = ordered.iloc[:count]
    harm_rate = float(selected.counterfactual_harmful.mean()) if count else 0.0
    utility = float(selected.counterfactual_causal_utility.mean()) if count else 0.0
    return {
        "selected_actions": count,
        "harm_rate": harm_rate,
        "mean_causal_utility": utility,
        "beneficial_rate": float(selected.counterfactual_beneficial.mean()) if count else 0.0,
    }


def analyze_pilot(
    results_root: Path,
    stage: str = "pilots",
    coverage: float = 0.60,
) -> Dict[str, Any]:
    summaries = read_csv_artifact(results_root / stage / "episode_summary.csv")
    candidates = read_csv_artifact(results_root / stage / "candidate_decisions.csv")
    evaluated = candidates[
        candidates.counterfactual_evaluated.astype(bool)
        & ~candidates.proposed_operational_action.eq("no_operational_action")
    ].copy()
    if "counterfactual_action_accepted" in evaluated:
        evaluated = evaluated[
            evaluated.counterfactual_action_accepted.astype(bool)
        ].copy()
    else:
        # Retained pilot iteration 1 predates the explicit field. Under its
        # always-act collector, actual acceptance is the matched acceptance.
        evaluated = evaluated[evaluated.accepted_physical_action.astype(bool)].copy()
    evaluated["counterfactual_harmful"] = evaluated.counterfactual_harmful.astype(bool)
    evaluated["counterfactual_beneficial"] = evaluated.counterfactual_beneficial.astype(bool)
    metric_rows: List[Dict[str, Any]] = []
    for (application, complexity), subset in evaluated.groupby(
        ["application", "complexity"], sort=True,
    ):
        labels = subset.counterfactual_harmful.to_numpy(dtype=int)
        for method, column in RISK_COLUMNS.items():
            auc, ap = _safe_auc(labels, subset[column].to_numpy(dtype=float))
            metric_rows.append({
                "application": application,
                "complexity": complexity,
                "method": method,
                "rows": len(subset),
                "independent_panels": subset.run_id.nunique(),
                "harm_prevalence": float(labels.mean()) if len(labels) else 0.0,
                "roc_auc": auc,
                "average_precision": ap,
            })
    selection_rows: List[Dict[str, Any]] = []
    keys = [
        "run_id", "application", "complexity", "coupling", "fragmentation",
        "network_disruption", "topology_family", "environment_seed",
    ]
    for group_key, subset in evaluated.groupby(keys, sort=True):
        base = dict(zip(keys, group_key))
        for method, column in RISK_COLUMNS.items():
            selection_rows.append({
                **base, "method": method, "coverage": coverage,
                **_select_at_coverage(subset, column, coverage),
            })
    selection = pd.DataFrame(selection_rows)
    baseline = selection[selection.method == "kpi_confidence"]
    entropic = selection[selection.method == "combined_generalized_entropic"]
    paired = baseline.merge(
        entropic, on=keys + ["coverage"], suffixes=("_kpi", "_entropic"),
        validate="one_to_one",
    )
    paired["incremental_harm_reduction"] = (
        paired.harm_rate_kpi - paired.harm_rate_entropic
    )
    paired["incremental_utility"] = (
        paired.mean_causal_utility_entropic - paired.mean_causal_utility_kpi
    )
    paired["coupling_numeric"] = paired.coupling.map(LEVEL_VALUE)
    paired["fragmentation_numeric"] = paired.fragmentation.map(LEVEL_VALUE)
    paired["size_numeric"] = paired.complexity.map(SIZE_VALUE)
    paired["coupling_fragmentation"] = (
        paired.coupling_numeric * paired.fragmentation_numeric
    )
    interaction = {
        "coefficient": None,
        "intercept": None,
        "panels": len(paired),
    }
    if len(paired) >= 4:
        design = np.column_stack([
            np.ones(len(paired)), paired.coupling_numeric,
            paired.fragmentation_numeric, paired.size_numeric,
            paired.coupling_fragmentation,
        ])
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, paired.incremental_harm_reduction.to_numpy(dtype=float), rcond=None,
        )
        interaction = {
            "intercept": float(coefficients[0]),
            "coupling_coefficient": float(coefficients[1]),
            "fragmentation_coefficient": float(coefficients[2]),
            "size_coefficient": float(coefficients[3]),
            "coupling_fragmentation_interaction": float(coefficients[4]),
            "design_rank": int(rank),
            "panels": len(paired),
        }
    high = paired[
        (paired.coupling == "high") & (paired.fragmentation == "high")
    ]
    high_rows: List[Dict[str, Any]] = []
    for application, subset in high.groupby("application", sort=True):
        interval = paired_bootstrap(
            subset.incremental_harm_reduction, 10000, 770901,
        )
        utility = paired_bootstrap(
            subset.incremental_utility, 10000, 770902,
        )
        high_rows.append({
            "application": application,
            "panels": len(subset),
            "harm_rate_reduction": interval["mean"],
            "harm_ci95_low": interval["ci_low"],
            "harm_ci95_high": interval["ci_high"],
            "causal_utility_gain": utility["mean"],
            "utility_ci95_low": utility["ci_low"],
            "utility_ci95_high": utility["ci_high"],
        })
    communication_rows: List[Dict[str, Any]] = []
    communication_keys = [
        "application", "complexity", "coupling", "fragmentation",
        "network_disruption", "topology_family", "environment_seed",
        "information_condition", "controller",
    ]
    always = summaries[summaries.sketch_policy == "always_on"]
    event = summaries[summaries.sketch_policy == "event_triggered"]
    if not always.empty and not event.empty:
        matched = always.merge(
            event, on=communication_keys, suffixes=("_always", "_event"),
        )
        for application, subset in matched.groupby("application", sort=True):
            for measure in ("total_messages", "total_bytes", "sketch_messages", "sketch_bytes"):
                reduction = (
                    subset[measure + "_always"] - subset[measure + "_event"]
                ) / subset[measure + "_always"].clip(lower=1e-9)
                interval = paired_bootstrap(reduction, 10000, 770903)
                communication_rows.append({
                    "application": application, "measure": measure,
                    "matched_panels": len(subset), "relative_reduction": interval["mean"],
                    "ci95_low": interval["ci_low"], "ci95_high": interval["ci_high"],
                })
    variation_rows = []
    for application, subset in evaluated.groupby("application", sort=True):
        variation_rows.append({
            "application": application,
            "rows": len(subset),
            "independent_panels": subset.run_id.nunique(),
            "beneficial_actions": int(subset.counterfactual_beneficial.sum()),
            "harmful_actions": int(subset.counterfactual_harmful.sum()),
            "neutral_actions": int((~subset.counterfactual_beneficial & ~subset.counterfactual_harmful).sum()),
            "shannon_std": float(subset.shannon_local.std()),
            "js_disagreement_std": float(subset.js_disagreement.std()),
            "graph_disagreement_std": float(subset.graph_disagreement.std()),
            "distributed_contributors_mean": float(
                subset.distributed_contributor_count.mean()
            ) if "distributed_contributor_count" in subset else None,
            "kpi_severity_std": float(subset.severity.std()),
            "competitive_panels": int(subset.groupby("run_id").counterfactual_harmful.nunique().gt(1).sum()),
        })
    destination = results_root / stage / "analysis"
    write_csv(destination / "risk_ranking_metrics.csv", metric_rows)
    write_csv(destination / "matched_coverage_selections.csv", selection_rows)
    write_csv(destination / "paired_incremental_effects.csv", paired.to_dict("records"))
    write_csv(destination / "high_complexity_effects.csv", high_rows)
    write_csv(destination / "communication_reductions.csv", communication_rows)
    write_csv(destination / "mechanism_variation.csv", variation_rows)
    report = {
        "stage": stage,
        "evidence_status": "retained_feasibility_pilot",
        "episodes": len(summaries),
        "candidate_rows": len(candidates),
        "counterfactual_action_rows": len(evaluated),
        "independent_panels": summaries.run_id.nunique(),
        "interaction_model": interaction,
        "high_complexity": high_rows,
        "communication": communication_rows,
        "variation": variation_rows,
        "not_confirmatory": True,
    }
    atomic_json(destination / "pilot_analysis.json", report)
    return report
