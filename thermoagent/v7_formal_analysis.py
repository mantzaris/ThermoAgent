"""Leakage-resistant panel-level risk and complexity analysis for V7."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .v5_analysis import paired_bootstrap
from .v5_experiments import atomic_json, write_csv


BASE_NUMERIC = (
    "severity", "safety_risk", "resource_scarcity", "delay",
    "predictive_uncertainty", "action_probability", "action_value",
    "action_value_margin", "communication_reliability",
    "coupling_numeric", "fragmentation_numeric", "size_normalized",
)
ENTROPIC_NUMERIC = (
    "shannon_local", "pooled_uncertainty", "js_disagreement",
    "jt_disagreement_0_5", "jt_disagreement_1_5", "jt_disagreement_2",
    "jt_disagreement_3", "graph_disagreement", "consensus",
    "consensus_residual", "entropy_slope", "disagreement_slope",
)
CATEGORICAL = ("proposed_operational_action", "topology_family")
FEATURE_BLOCKS = {
    "strongest_nonentropic": BASE_NUMERIC,
    "shannon_js": BASE_NUMERIC + (
        "shannon_local", "pooled_uncertainty", "js_disagreement",
    ),
    "generalized_entropic": BASE_NUMERIC + ENTROPIC_NUMERIC,
}


def prepare_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output = output[
        output.counterfactual_evaluated.astype(bool)
        & output.counterfactual_action_accepted.astype(bool)
        & ~output.proposed_operational_action.eq("no_operational_action")
    ].reset_index(drop=True)
    output["harmful_label"] = output.counterfactual_harmful.astype(int)
    # Matched controller and communication variants of the same stochastic
    # environment are one independent panel.  Using run_id here would leak a
    # graph/scenario replica across folds because run_id includes the policy.
    panel_columns = (
        "application", "environment_seed", "topology_family", "complexity",
        "coupling", "fragmentation", "network_disruption",
        "information_condition",
    )
    output["panel_id"] = output.loc[:, panel_columns].astype(str).agg("|".join, axis=1)
    output["cluster_id"] = output["panel_id"]
    output["scenario_family"] = output.loc[:, (
        "application", "coupling", "fragmentation", "network_disruption",
        "information_condition",
    )].astype(str).agg("|".join, axis=1)
    return output


def _pipeline(numeric: Sequence[str], c_value: float) -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), list(numeric)),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
    ])
    return Pipeline([
        ("features", transformer),
        ("model", LogisticRegression(
            C=float(c_value), penalty="l2", solver="liblinear",
            max_iter=3000, random_state=77701,
        )),
    ])


def _inner_c(
    frame: pd.DataFrame, numeric: Sequence[str], groups: np.ndarray,
    c_values: Sequence[float] = (0.03, 0.10, 0.30, 1.0, 3.0),
) -> float:
    unique = np.unique(groups)
    if len(unique) < 3:
        return 0.30
    splitter = GroupKFold(n_splits=min(4, len(unique)))
    best = None
    for c_value in c_values:
        losses = []
        for train, test in splitter.split(frame, frame.harmful_label, groups):
            labels = frame.harmful_label.iloc[train]
            if labels.nunique() < 2:
                continue
            model = _pipeline(numeric, c_value)
            model.fit(frame.iloc[train], labels)
            probabilities = model.predict_proba(frame.iloc[test])[:, 1]
            target = frame.harmful_label.iloc[test].to_numpy(dtype=float)
            losses.append(float(np.mean(np.square(probabilities - target))))
        score = float(np.mean(losses)) if losses else float("inf")
        candidate = (score, float(c_value))
        if best is None or candidate < best:
            best = candidate
    return float(best[1] if best is not None else 0.30)


def crossfit_feature_block(
    frame: pd.DataFrame, block: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    numeric = FEATURE_BLOCKS[block]
    groups = frame.panel_id.to_numpy()
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("formal V7 cross-fitting requires at least five independent panels")
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    predictions = np.zeros(len(frame), dtype=float)
    audits: List[Dict[str, Any]] = []
    for fold, (train, test) in enumerate(
        splitter.split(frame, frame.harmful_label, groups), start=1,
    ):
        train_frame = frame.iloc[train]
        test_frame = frame.iloc[test]
        c_value = _inner_c(
            train_frame, numeric, train_frame.panel_id.to_numpy(),
        )
        if train_frame.harmful_label.nunique() < 2:
            predictions[test] = float(train_frame.harmful_label.mean())
        else:
            model = _pipeline(numeric, c_value)
            model.fit(train_frame, train_frame.harmful_label)
            predictions[test] = model.predict_proba(test_frame)[:, 1]
        audits.append({
            "feature_block": block, "fold": fold,
            "training_rows": len(train), "test_rows": len(test),
            "training_panels": train_frame.panel_id.nunique(),
            "test_panels": test_frame.panel_id.nunique(),
            "panel_disjoint": set(train_frame.panel_id).isdisjoint(set(test_frame.panel_id)),
            "environment_seed_disjoint": set(train_frame.environment_seed).isdisjoint(set(test_frame.environment_seed)),
            "graph_instance_disjoint": set(train_frame.panel_id).isdisjoint(set(test_frame.panel_id)),
            "scenario_instance_disjoint": set(train_frame.cluster_id).isdisjoint(set(test_frame.cluster_id)),
            "selected_c": c_value,
        })
    return predictions, audits


def _panel_selection(
    frame: pd.DataFrame, score: str, coverage: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for panel_id, subset in frame.groupby("panel_id", sort=True):
        ordered = subset.sort_values(
            [score, "step", "agent_id", "target"], kind="mergesort",
        )
        count = max(1, min(len(ordered), int(round(coverage * len(ordered)))))
        selected = ordered.iloc[:count]
        rows.append({
            "panel_id": panel_id,
            "application": selected.application.iloc[0],
            "complexity": selected.complexity.iloc[0],
            "coupling": selected.coupling.iloc[0],
            "fragmentation": selected.fragmentation.iloc[0],
            "network_disruption": selected.network_disruption.iloc[0],
            "topology_family": selected.topology_family.iloc[0],
            "environment_seed": int(selected.environment_seed.iloc[0]),
            "selected_actions": count,
            "action_coverage": float(count / len(ordered)),
            "harm_rate": float(selected.harmful_label.mean()),
            "causal_utility": float(selected.counterfactual_causal_utility.mean()),
            "service_loss_proxy": float(-selected.counterfactual_causal_utility.sum()),
        })
    return pd.DataFrame(rows)


def analyze_risk_stage(
    results_root: Path,
    stage: str,
    coverage: float = 0.60,
) -> Dict[str, Any]:
    frame = prepare_candidates(pd.read_csv(results_root / stage / "candidate_decisions.csv"))
    scored_frames: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    selection_frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    for application, subset in frame.groupby("application", sort=True):
        subset = subset.reset_index(drop=True)
        for block in FEATURE_BLOCKS:
            predictions, audits = crossfit_feature_block(subset, block)
            fold_rows.extend({"application": application, **value} for value in audits)
            score_column = "risk_%s" % block
            subset[score_column] = predictions
            auc = roc_auc_score(subset.harmful_label, predictions) if subset.harmful_label.nunique() > 1 else float("nan")
            ap = average_precision_score(subset.harmful_label, predictions) if subset.harmful_label.nunique() > 1 else float("nan")
            metric_rows.append({
                "application": application, "feature_block": block,
                "rows": len(subset), "independent_panels": subset.panel_id.nunique(),
                "harm_prevalence": float(subset.harmful_label.mean()),
                "roc_auc": float(auc), "average_precision": float(ap),
                "brier_score": float(brier_score_loss(subset.harmful_label, predictions)),
            })
            selection_frames[(application, block)] = _panel_selection(
                subset, score_column, coverage,
            )
        scored_frames.append(subset)
    paired_rows: List[Dict[str, Any]] = []
    high_rows: List[Dict[str, Any]] = []
    interactions: List[Dict[str, Any]] = []
    for application in sorted(frame.application.unique()):
        baseline = selection_frames[(application, "strongest_nonentropic")]
        entropic = selection_frames[(application, "generalized_entropic")]
        paired = baseline.merge(
            entropic, on=[
                "panel_id", "application", "complexity", "coupling",
                "fragmentation", "network_disruption", "topology_family",
                "environment_seed",
            ], suffixes=("_baseline", "_entropic"), validate="one_to_one",
        )
        paired["harm_reduction"] = paired.harm_rate_baseline - paired.harm_rate_entropic
        paired["utility_gain"] = paired.causal_utility_entropic - paired.causal_utility_baseline
        paired["service_loss_difference"] = paired.service_loss_proxy_entropic - paired.service_loss_proxy_baseline
        paired_rows.extend(paired.to_dict("records"))
        high = paired[(paired.coupling == "high") & (paired.fragmentation == "high")]
        harm_interval = paired_bootstrap(high.harm_reduction, 10000, 77711)
        service_interval = paired_bootstrap(high.service_loss_difference, 10000, 77712)
        high_rows.append({
            "application": application, "panels": len(high),
            "harm_reduction": harm_interval["mean"],
            "harm_ci95_low": harm_interval["ci_low"],
            "harm_ci95_high": harm_interval["ci_high"],
            "service_loss_difference": service_interval["mean"],
            "service_ci95_low": service_interval["ci_low"],
            "service_ci95_high": service_interval["ci_high"],
        })
        coded = paired.copy()
        coded["coupling_value"] = coded.coupling.map({"low": 0.0, "medium": 0.5, "high": 1.0})
        coded["fragmentation_value"] = coded.fragmentation.map({"low": 0.0, "medium": 0.5, "high": 1.0})
        coded["size_value"] = coded.complexity.map({"small": 0.0, "medium": 0.5, "large": 1.0})
        design = np.column_stack([
            np.ones(len(coded)), coded.coupling_value,
            coded.fragmentation_value, coded.size_value,
            coded.coupling_value * coded.fragmentation_value,
        ])
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, coded.harm_reduction, rcond=None,
        )
        rng = np.random.RandomState(77713)
        bootstrap = []
        for _ in range(10000):
            indices = rng.randint(0, len(coded), len(coded))
            sampled_design = design[indices]
            sampled_target = coded.harm_reduction.to_numpy()[indices]
            coefficient, _, sampled_rank, _ = np.linalg.lstsq(
                sampled_design, sampled_target, rcond=None,
            )
            if sampled_rank == design.shape[1]:
                bootstrap.append(float(coefficient[4]))
        interactions.append({
            "application": application, "panels": len(coded),
            "design_rank": int(rank),
            "coupling_fragmentation_interaction": float(coefficients[4]),
            "interaction_ci95_low": float(np.quantile(bootstrap, 0.025)) if bootstrap else None,
            "interaction_ci95_high": float(np.quantile(bootstrap, 0.975)) if bootstrap else None,
        })
    destination = results_root / stage / "risk_analysis"
    write_csv(destination / "crossfit_predictions.csv", pd.concat(scored_frames).to_dict("records"))
    write_csv(destination / "grouped_fold_audit.csv", fold_rows)
    write_csv(destination / "ranking_metrics.csv", metric_rows)
    write_csv(destination / "paired_panel_effects.csv", paired_rows)
    write_csv(destination / "high_complexity_effects.csv", high_rows)
    write_csv(destination / "complexity_interaction.csv", interactions)
    report = {
        "stage": stage, "coverage": coverage,
        "rows": len(frame), "independent_panels": frame.panel_id.nunique(),
        "feature_blocks": {key: list(value) for key, value in FEATURE_BLOCKS.items()},
        "ranking": metric_rows, "high_complexity": high_rows,
        "interactions": interactions,
        "panel_level_inference": True,
        "candidate_rows_not_treated_as_independent_replicates": True,
    }
    atomic_json(destination / "risk_analysis.json", report)
    return report


def power_from_pilot(
    pilot_effects: pd.DataFrame,
    practical_effect: float = 0.040,
    target_power: float = 0.80,
) -> List[Dict[str, Any]]:
    """Normal-approximation planning from panel SD; final CIs use bootstrap."""
    rows = []
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143 if target_power == 0.80 else 1.0364333894937898
    for application, subset in pilot_effects.groupby("application", sort=True):
        standard_deviation = float(subset.incremental_harm_reduction.std(ddof=1))
        required = int(np.ceil(
            ((z_alpha + z_power) * standard_deviation / practical_effect) ** 2
        )) if standard_deviation > 0 else 12
        rows.append({
            "application": application,
            "pilot_panels": len(subset),
            "pilot_effect_sd": standard_deviation,
            "practical_absolute_harm_reduction": practical_effect,
            "target_power": target_power,
            "normal_approximation_required_panels": max(12, required),
            "planning_only": True,
        })
    return rows
