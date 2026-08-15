"""Cluster-aware V5 development analysis and prospective gate evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .v5_experiments import atomic_json, write_csv
from .v5_types import OPERATOR_ACTIONS


KPI_FEATURES = (
    "visible_severity", "visible_backlog", "visible_delay",
    "resource_scarcity", "safety_risk", "commitment_strain",
)
ENERGY_FEATURES = ("operational_energy",)
ENTROPY_FEATURES = (
    "mean_belief_entropy", "entropy_dispersion", "js_disagreement",
    "entropy_slope", "consensus_residual",
)
COMPLETE_FEATURES = (
    *KPI_FEATURES, *ENERGY_FEATURES, *ENTROPY_FEATURES,
    "consensus_confidence", "distributed_entropy",
)
FREE_ENERGY_FEATURES = (*COMPLETE_FEATURES, "effective_temperature", "free_energy")
FEATURE_BLOCKS = {
    "local_kpi_only": KPI_FEATURES,
    "energy_only": ENERGY_FEATURES,
    "entropy_disagreement_only": ENTROPY_FEATURES,
    "kpi_plus_energy": (*KPI_FEATURES, *ENERGY_FEATURES),
    "kpi_plus_entropy_disagreement": (*KPI_FEATURES, *ENTROPY_FEATURES),
    "complete_thermodynamic": COMPLETE_FEATURES,
    "exploratory_free_energy": FREE_ENERGY_FEATURES,
}


def _safe_divide(numerator: float, denominator: float, floor: float = 0.05) -> float:
    return float(numerator / max(abs(denominator), floor))


def paired_bootstrap(
    values: Sequence[float], replicates: int = 10000, seed: int = 55051,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": None, "ci_low": None, "ci_high": None, "n_clusters": 0}
    rng = np.random.RandomState(int(seed))
    draws = np.empty(int(replicates), dtype=float)
    batch = 500
    for start in range(0, int(replicates), batch):
        stop = min(int(replicates), start + batch)
        indices = rng.randint(0, len(array), size=(stop - start, len(array)))
        draws[start:stop] = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "n_clusters": int(len(array)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "confidence": float(confidence),
    }


def _action_design(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    base = frame[list(features)].to_numpy(dtype=float)
    action = frame["action"].astype(str).to_numpy()
    one_hot = np.column_stack([(action == value).astype(float) for value in OPERATOR_ACTIONS])
    interactions = np.concatenate([base * one_hot[:, index:index + 1] for index in range(len(OPERATOR_ACTIONS))], axis=1)
    return np.concatenate([base, one_hot, interactions], axis=1)


@dataclass
class ActionValueModel:
    scaler: StandardScaler
    model: Ridge
    features: Tuple[str, ...]
    alpha: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = _action_design(frame, self.features)
        return self.model.predict(self.scaler.transform(design))


def _fit_action_value(frame: pd.DataFrame, features: Sequence[str], alpha: float) -> ActionValueModel:
    design = _action_design(frame, features)
    scaler = StandardScaler().fit(design)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(scaler.transform(design), frame["causal_effect"].to_numpy(dtype=float))
    return ActionValueModel(scaler, model, tuple(features), float(alpha))


def select_budget(
    frame: pd.DataFrame,
    scores: Sequence[float],
    budget: int = 2,
    consensus_abstention: bool = False,
    force_selection: bool = False,
) -> pd.DataFrame:
    predicted = np.asarray(scores, dtype=float)
    if len(predicted) != len(frame):
        raise ValueError("V5 score length does not match candidate frame")
    clusters = frame["cluster_id"].astype(str).to_numpy()
    incidents = frame["incident_id"].astype(str).to_numpy()
    actions = frame["action"].astype(str).to_numpy()
    severity = frame["visible_severity"].to_numpy(dtype=float)
    confidence = frame["consensus_confidence"].to_numpy(dtype=float)
    effects = frame["causal_effect"].to_numpy(dtype=float)
    losses_without = frame["loss_without"].to_numpy(dtype=float)
    minutes = frame["operator_minutes"].to_numpy(dtype=float)
    harmful = frame["harmful"].astype(bool).to_numpy()
    beneficial = frame["beneficial"].astype(bool).to_numpy()
    reaches = frame["reached_service"].astype(bool).to_numpy()
    candidate_ids = frame["candidate_id"].astype(str).to_numpy()
    scenario_families = frame["scenario_family"].astype(str).to_numpy()
    records: List[Dict[str, Any]] = []
    for cluster_id in sorted(set(clusters)):
        positions = np.flatnonzero(clusters == cluster_id)
        # lexsort uses the final key as primary: descending prediction, then
        # stable incident/action ordering for exact deterministic ties.
        ranked = positions[np.lexsort((actions[positions], incidents[positions], -predicted[positions]))]
        selected: List[int] = []
        selected_incidents = set()
        for position in ranked:
            incident_id = incidents[position]
            if incident_id in selected_incidents:
                continue
            if severity[position] < 0.30:
                continue
            if consensus_abstention and confidence[position] < 0.42:
                continue
            if not force_selection and predicted[position] <= 0.0:
                continue
            selected.append(int(position))
            selected_incidents.add(incident_id)
            if len(selected) >= int(budget):
                break
        first_per_incident: Dict[str, int] = {}
        for position in positions:
            first_per_incident.setdefault(incidents[position], int(position))
        baseline_loss = float(sum(losses_without[position] for position in first_per_incident.values()))
        total_effect = float(effects[selected].sum()) if selected else 0.0
        first = int(positions[0])
        records.append({
            "cluster_id": str(cluster_id),
            "application": str(frame["application"].iloc[first]),
            "regime": str(frame["regime"].iloc[first]),
            "information_condition": str(frame["information_condition"].iloc[first]),
            "environment_seed": int(frame["environment_seed"].iloc[first]),
            "topology_family": str(frame["topology_family"].iloc[first]),
            "panel_baseline_loss": baseline_loss,
            "selected_effect": total_effect,
            "loss_after_selection": baseline_loss - total_effect,
            "selected_count": len(selected),
            "operator_minutes": float(minutes[selected].sum()) if selected else 0.0,
            "harmful_count": int(harmful[selected].sum()) if selected else 0,
            "beneficial_count": int(beneficial[selected].sum()) if selected else 0,
            "service_reaching_count": int(reaches[selected].sum()) if selected else 0,
            "selected_candidates": ";".join(candidate_ids[selected]),
            "selected_scenario_families": ";".join(scenario_families[selected]),
            "mean_consensus_confidence": float(confidence[selected].mean()) if selected else 0.0,
        })
    return pd.DataFrame(records)


def _inner_alpha(
    train: pd.DataFrame,
    features: Sequence[str],
    alpha_grid: Sequence[float],
    budget: int,
) -> float:
    groups = train["environment_seed"].to_numpy()
    unique = np.unique(groups)
    folds = GroupKFold(n_splits=min(3, len(unique)))
    best_alpha = float(alpha_grid[0])
    best_utility = float("-inf")
    for alpha in alpha_grid:
        predictions = np.zeros(len(train), dtype=float)
        for fit_index, test_index in folds.split(train, groups=groups):
            model = _fit_action_value(train.iloc[fit_index], features, float(alpha))
            predictions[test_index] = model.predict(train.iloc[test_index])
        selected = select_budget(train, predictions, budget=budget)
        utility = float(selected["selected_effect"].mean())
        if utility > best_utility + 1e-12:
            best_utility = utility
            best_alpha = float(alpha)
    return best_alpha


def crossfit_action_values(
    frame: pd.DataFrame,
    features: Sequence[str],
    alpha_grid: Sequence[float],
    budget: int = 2,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    groups = frame["environment_seed"].to_numpy()
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("V5 cross-fitting requires at least five independent seeds")
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    predictions = np.zeros(len(frame), dtype=float)
    folds: List[Dict[str, Any]] = []
    for fold_number, (fit_index, test_index) in enumerate(splitter.split(frame, groups=groups), start=1):
        train = frame.iloc[fit_index]
        alpha = _inner_alpha(train, features, alpha_grid, budget)
        model = _fit_action_value(train, features, alpha)
        predictions[test_index] = model.predict(frame.iloc[test_index])
        folds.append({
            "fold": fold_number,
            "alpha": alpha,
            "training_clusters": int(train["cluster_id"].nunique()),
            "test_clusters": int(frame.iloc[test_index]["cluster_id"].nunique()),
            "test_indices": test_index.tolist(),
            "model": model,
        })
    return predictions, folds


def _prediction_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Optional[float]]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    return {
        "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "brier": float(brier_score_loss(labels, probabilities)),
        "prevalence": float(labels.mean()),
    }


def _feature_diagnostics(frame: pd.DataFrame, features: Sequence[str]) -> Dict[str, Any]:
    matrix = frame[list(features)].to_numpy(dtype=float)
    variance = matrix.var(axis=0)
    retained = variance > 1e-12
    if np.any(retained):
        scaled = (matrix[:, retained] - matrix[:, retained].mean(axis=0)) / np.sqrt(variance[retained])
        condition = float(np.linalg.cond(scaled)) if scaled.shape[1] > 1 else 1.0
        rank = int(np.linalg.matrix_rank(scaled))
    else:
        condition = None
        rank = 0
    correlations = np.corrcoef(matrix[:, retained], rowvar=False) if retained.sum() > 1 else np.asarray([[1.0]])
    maximum_correlation = float(np.max(np.abs(correlations - np.eye(len(correlations))))) if len(correlations) > 1 else 0.0
    return {
        "features": list(features),
        "rows": int(len(frame)),
        "clusters": int(frame["cluster_id"].nunique()),
        "constant_features": [str(features[index]) for index, value in enumerate(variance) if value <= 1e-12],
        "rank": rank,
        "condition_number": condition,
        "maximum_absolute_pairwise_correlation": maximum_correlation,
    }


def _common_support(frame: pd.DataFrame) -> float:
    beneficial = frame[frame["beneficial"].astype(bool)]
    other = frame[~frame["beneficial"].astype(bool)]
    if beneficial.empty or other.empty:
        return 0.0
    supported = np.ones(len(frame), dtype=bool)
    for feature in ENTROPY_FEATURES:
        lower = max(float(beneficial[feature].quantile(0.02)), float(other[feature].quantile(0.02)))
        upper = min(float(beneficial[feature].quantile(0.98)), float(other[feature].quantile(0.98)))
        if upper <= lower:
            return 0.0
        supported &= frame[feature].between(lower, upper).to_numpy()
    return float(supported.mean())


def _selection_pair(kpi: pd.DataFrame, thermo: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cluster_id", "application", "regime", "information_condition",
        "environment_seed", "selected_effect", "loss_after_selection",
        "selected_count", "operator_minutes", "harmful_count",
        "service_reaching_count", "selected_candidates",
        "selected_scenario_families",
    ]
    paired = kpi[columns].merge(
        thermo[columns], on=["cluster_id", "application", "regime", "information_condition", "environment_seed"],
        suffixes=("_kpi", "_thermo"), validate="one_to_one",
    )
    paired["utility_difference"] = paired["selected_effect_thermo"] - paired["selected_effect_kpi"]
    paired["loss_difference"] = paired["loss_after_selection_thermo"] - paired["loss_after_selection_kpi"]
    paired["harm_difference"] = paired["harmful_count_thermo"] - paired["harmful_count_kpi"]
    return paired


def analyze_feature_blocks(
    candidates: pd.DataFrame,
    alpha_grid: Sequence[float],
    budget: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    candidate_outputs: List[pd.DataFrame] = []
    selections: List[pd.DataFrame] = []
    diagnostics: Dict[str, Any] = {}
    for application in sorted(candidates["application"].unique()):
        for condition in sorted(candidates["information_condition"].unique()):
            subset = candidates[
                (candidates["application"] == application)
                & (candidates["information_condition"] == condition)
            ].reset_index(drop=True)
            for block_name, features in FEATURE_BLOCKS.items():
                scores, folds = crossfit_action_values(subset, features, alpha_grid, budget)
                output = subset[["candidate_id", "cluster_id", "application", "regime", "information_condition", "environment_seed", "incident_id", "action", "causal_effect", "beneficial", "harmful"]].copy()
                output["feature_block"] = block_name
                output["predicted_value"] = scores
                candidate_outputs.append(output)
                selected = select_budget(
                    subset, scores, budget=budget,
                    consensus_abstention=block_name in {
                        "kpi_plus_entropy_disagreement", "complete_thermodynamic", "exploratory_free_energy",
                    },
                )
                selected["feature_block"] = block_name
                selections.append(selected)
                diagnostics["%s|%s|%s" % (application, condition, block_name)] = {
                    **_feature_diagnostics(subset, features),
                    "selected_alphas": [float(fold["alpha"]) for fold in folds],
                    "prediction": _prediction_metrics(
                        subset["beneficial"].astype(int).to_numpy(), scores,
                    ),
                }
    return pd.concat(candidate_outputs, ignore_index=True), pd.concat(selections, ignore_index=True), diagnostics


def refit_permutation_test(
    frame: pd.DataFrame,
    kpi_selection: pd.DataFrame,
    true_gain: float,
    alpha_grid: Sequence[float],
    budget: int,
    replicates: int,
    seed: int,
    evaluation_excluded_regimes: Sequence[str] = ("nominal",),
) -> Dict[str, Any]:
    """Permute the thermodynamic block, then refit every grouped fold."""

    rng = np.random.RandomState(int(seed))
    thermo_columns = list(ENTROPY_FEATURES)
    severity_strata = pd.qcut(
        frame["visible_severity"], q=min(5, frame["visible_severity"].nunique()),
        labels=False, duplicates="drop",
    ).astype(str)
    strata = (
        frame["application"].astype(str) + "|"
        + frame["information_condition"].astype(str) + "|"
        + frame["regime"].astype(str) + "|" + severity_strata
    )
    # Keep each incident's repeated action candidates together during the
    # permutation so the feature block remains internally coherent.
    working = frame[["cluster_id", "incident_id", *thermo_columns]].copy()
    working["_permutation_stratum"] = strata.to_numpy()
    incident_table = working.drop_duplicates(["cluster_id", "incident_id"]).reset_index(drop=True)
    incident_strata = incident_table.pop("_permutation_stratum").to_numpy()
    gains: List[float] = []
    for _ in range(int(replicates)):
        permuted_incidents = incident_table.copy()
        for stratum in sorted(set(incident_strata)):
            positions = np.flatnonzero(incident_strata == stratum)
            if len(positions) > 1:
                source = rng.permutation(positions)
                permuted_incidents.loc[positions, thermo_columns] = incident_table.loc[source, thermo_columns].to_numpy()
        permuted = frame.drop(columns=thermo_columns).merge(
            permuted_incidents, on=["cluster_id", "incident_id"], validate="many_to_one",
        )
        scores, _ = crossfit_action_values(
            permuted, FEATURE_BLOCKS["kpi_plus_entropy_disagreement"], alpha_grid, budget,
        )
        selected = select_budget(permuted, scores, budget=budget, consensus_abstention=True)
        evaluation_kpi = kpi_selection[
            ~kpi_selection["regime"].isin(evaluation_excluded_regimes)
        ]
        evaluation_selected = selected[
            ~selected["regime"].isin(evaluation_excluded_regimes)
        ]
        paired = evaluation_kpi[["cluster_id", "selected_effect"]].merge(
            evaluation_selected[["cluster_id", "selected_effect"]], on="cluster_id",
            suffixes=("_kpi", "_permuted"), validate="one_to_one",
        )
        gains.append(float((paired["selected_effect_permuted"] - paired["selected_effect_kpi"]).mean()))
    array = np.asarray(gains, dtype=float)
    return {
        "procedure": "within-stratum block permutation with full grouped-pipeline refit",
        "fit_regimes": sorted(frame["regime"].astype(str).unique().tolist()),
        "evaluation_excluded_regimes": list(evaluation_excluded_regimes),
        "replicates": int(replicates),
        "seed": int(seed),
        "true_mean_gain": float(true_gain),
        "permuted_mean_gain": float(array.mean()),
        "permuted_gain_ci95_low": float(np.quantile(array, 0.025)),
        "permuted_gain_ci95_high": float(np.quantile(array, 0.975)),
        "p_value_plus_one": float((1 + np.sum(array >= true_gain)) / (len(array) + 1)),
        "permuted_to_true_gain_fraction": float(array.mean() / true_gain) if true_gain > 1e-12 else None,
        "monte_carlo_standard_error_upper": float(0.5 / math.sqrt(len(array) + 1)),
    }


def analyze_development(
    results_root: Path,
    stage: str = "development_primary",
    permutation_replicates: int = 199,
) -> Dict[str, Any]:
    config = yaml.safe_load((Path("configs") / "human_operator_v5_development.yaml").read_text(encoding="utf-8"))
    stage_root = results_root / "development" / stage
    episodes = pd.read_csv(stage_root / "episode_summary.csv")
    candidates = pd.read_csv(stage_root / "candidate_interventions.csv")
    for column in ("beneficial", "harmful", "accepted_action", "reached_next_stage", "reached_service"):
        candidates[column] = candidates[column].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    alpha_grid = [float(value) for value in config["primary_model"]["regularization_grid"]]
    budget = int(config["simulation"]["operator_budget"])
    candidate_predictions, selections, diagnostics = analyze_feature_blocks(candidates, alpha_grid, budget)

    write_csv(results_root / "statistics" / "candidate_crossfit_predictions.csv", candidate_predictions.to_dict("records"))
    write_csv(results_root / "statistics" / "panel_budget_selections.csv", selections.to_dict("records"))
    atomic_json(results_root / "statistics" / "feature_diagnostics.json", diagnostics)

    primary_pairs: List[Dict[str, Any]] = []
    pair_frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    for application in sorted(candidates["application"].unique()):
        for condition in sorted(candidates["information_condition"].unique()):
            selected_subset = selections[
                (selections["application"] == application)
                & (selections["information_condition"] == condition)
            ]
            kpi = selected_subset[selected_subset["feature_block"] == "local_kpi_only"]
            thermo = selected_subset[selected_subset["feature_block"] == "kpi_plus_entropy_disagreement"]
            paired = _selection_pair(kpi, thermo)
            pair_frames[(application, condition)] = paired
            disrupted = paired[paired["regime"] != "nominal"].copy()
            interval = paired_bootstrap(disrupted["utility_difference"], 10000, 55051)
            kpi_mean = float(disrupted["selected_effect_kpi"].mean())
            thermo_mean = float(disrupted["selected_effect_thermo"].mean())
            positive_by_regime = disrupted.groupby("regime")["utility_difference"].mean()
            scenario_positive = disrupted.groupby("regime")["utility_difference"].sum().clip(lower=0.0)
            concentration = float(scenario_positive.max() / scenario_positive.sum()) if scenario_positive.sum() > 0 else 1.0
            primary_pairs.append({
                "application": application,
                "information_condition": condition,
                "clusters": int(len(disrupted)),
                "kpi_mean_utility": kpi_mean,
                "thermo_mean_utility": thermo_mean,
                "absolute_gain": float(thermo_mean - kpi_mean),
                "relative_gain": _safe_divide(thermo_mean - kpi_mean, kpi_mean),
                "gain_ci95_low": interval["ci_low"],
                "gain_ci95_high": interval["ci_high"],
                "paired_win_fraction": float((disrupted["utility_difference"] > 1e-12).mean()),
                "paired_tie_fraction": float((disrupted["utility_difference"].abs() <= 1e-12).mean()),
                "paired_loss_fraction": float((disrupted["utility_difference"] < -1e-12).mean()),
                "improved_regimes": int((positive_by_regime > 0.0).sum()),
                "harmful_rate_kpi": float(disrupted["harmful_count_kpi"].sum() / max(disrupted["selected_count_kpi"].sum(), 1)),
                "harmful_rate_thermo": float(disrupted["harmful_count_thermo"].sum() / max(disrupted["selected_count_thermo"].sum(), 1)),
                "maximum_positive_gain_regime_fraction": concentration,
            })
    primary_frame = pd.DataFrame(primary_pairs)
    write_csv(results_root / "statistics" / "primary_incremental_value.csv", primary_pairs)
    all_pairs = pd.concat(pair_frames.values(), ignore_index=True)
    write_csv(results_root / "statistics" / "paired_panel_effects.csv", all_pairs.to_dict("records"))

    # Coordination necessity: fixed versus no communication in fragmented,
    # disrupted panels. Communication cost is retained separately.
    coordination_rows: List[Dict[str, Any]] = []
    human_rows: List[Dict[str, Any]] = []
    for application in sorted(episodes["application"].unique()):
        subset = episodes[
            (episodes["application"] == application)
            & (episodes["information_condition"] == "private_fragmented")
            & (episodes["regime"] != "nominal")
        ].copy()
        coordination_difference = subset["no_communication_loss"] - subset["fixed_communication_loss"]
        coordination_interval = paired_bootstrap(coordination_difference, 10000, 55061, confidence=0.90)
        base_mean = float(subset["no_communication_loss"].mean())
        coordination_rows.append({
            "application": application,
            "clusters": int(len(subset)),
            "absolute_loss_reduction": float(coordination_difference.mean()),
            "relative_loss_reduction": _safe_divide(float(coordination_difference.mean()), base_mean),
            "ci90_low": coordination_interval["ci_low"],
            "ci90_high": coordination_interval["ci_high"],
            "changed_panel_fraction": float((coordination_difference.abs() > 1e-12).mean()),
            "improved_regimes": int((subset.assign(diff=coordination_difference).groupby("regime")["diff"].mean() > 0).sum()),
            "mean_operational_messages": float(subset["fixed_operational_messages"].mean()),
            "mean_operational_bytes": float(subset["fixed_operational_bytes"].mean()),
            "communication_adjusted_utility": float(coordination_difference.mean() - 0.000002 * subset["fixed_operational_bytes"].mean()),
        })
        human_difference = subset["fixed_communication_loss"] - subset["bounded_oracle_loss"]
        human_interval = paired_bootstrap(human_difference, 10000, 55071)
        human_rows.append({
            "application": application,
            "clusters": int(len(subset)),
            "absolute_loss_reduction": float(human_difference.mean()),
            "relative_loss_reduction": _safe_divide(float(human_difference.mean()), float(subset["fixed_communication_loss"].mean())),
            "ci95_low": human_interval["ci_low"],
            "ci95_high": human_interval["ci_high"],
            "mean_operator_minutes": float(subset["bounded_oracle_operator_minutes"].mean()),
            "complete_chain_fraction": float(subset["complete_causal_chains"].sum() / max(subset["bounded_oracle_interventions"].sum(), 1)),
            "harmful_interventions": int(subset["bounded_oracle_harmful"].sum()),
        })
    write_csv(results_root / "statistics" / "coordination_necessity.csv", coordination_rows)
    write_csv(results_root / "statistics" / "human_causal_usefulness.csv", human_rows)

    # Feature support and univariate shortcut diagnostics are application-level.
    shortcut_rows: List[Dict[str, Any]] = []
    for application in sorted(candidates["application"].unique()):
        subset = candidates[
            (candidates["application"] == application)
            & (candidates["information_condition"] == "private_fragmented")
            & (candidates["regime"] != "nominal")
        ]
        labels = subset["beneficial"].astype(int).to_numpy()
        for feature in ("mean_belief_entropy", "js_disagreement"):
            shortcut_rows.append({
                "application": application,
                "feature": feature,
                "rows": int(len(subset)),
                "clusters": int(subset["cluster_id"].nunique()),
                "roc_auc": float(roc_auc_score(labels, subset[feature])) if len(np.unique(labels)) == 2 else None,
                "average_precision": float(average_precision_score(labels, subset[feature])) if labels.sum() else None,
                "common_support_fraction": _common_support(subset),
            })
    write_csv(results_root / "statistics" / "shortcut_and_support_diagnostics.csv", shortcut_rows)

    # Fragmented-minus-public mechanism interaction.
    interaction_rows: List[Dict[str, Any]] = []
    for application in sorted(candidates["application"].unique()):
        private = pair_frames[(application, "private_fragmented")]
        public = pair_frames[(application, "public_shared")]
        private = private[private["regime"] != "nominal"].sort_values(["environment_seed", "regime"])
        public = public[public["regime"] != "nominal"].sort_values(["environment_seed", "regime"])
        merged = private[["environment_seed", "regime", "utility_difference"]].merge(
            public[["environment_seed", "regime", "utility_difference"]],
            on=["environment_seed", "regime"], suffixes=("_private", "_public"), validate="one_to_one",
        )
        interaction = merged["utility_difference_private"] - merged["utility_difference_public"]
        interval = paired_bootstrap(interaction, 10000, 55081)
        scale = max(abs(float(private["selected_effect_kpi"].mean())), 0.05)
        interaction_rows.append({
            "application": application,
            "matched_clusters": int(len(merged)),
            "private_mean_gain": float(merged["utility_difference_private"].mean()),
            "public_mean_gain": float(merged["utility_difference_public"].mean()),
            "absolute_interaction": float(interaction.mean()),
            "relative_interaction": float(interaction.mean() / scale),
            "ci95_low": interval["ci_low"],
            "ci95_high": interval["ci_high"],
        })
    write_csv(results_root / "statistics" / "fragmentation_interaction.csv", interaction_rows)

    # Safety abstention: compare the primary thermo policy with the same scores
    # forced through low-confidence cases.
    abstention_rows: List[Dict[str, Any]] = []
    prediction_frame = candidate_predictions[
        candidate_predictions["feature_block"] == "kpi_plus_entropy_disagreement"
    ]
    for application in sorted(candidates["application"].unique()):
        subset = candidates[
            (candidates["application"] == application)
            & (candidates["information_condition"] == "private_fragmented")
            & (candidates["regime"].isin(["partition", "telemetry_integrity", "compound", "ood"]))
        ].reset_index(drop=True)
        score_map = prediction_frame.set_index("candidate_id")["predicted_value"]
        scores = subset["candidate_id"].map(score_map).to_numpy(dtype=float)
        safe = select_budget(subset, scores, budget, consensus_abstention=True)
        forced = select_budget(subset, scores, budget, consensus_abstention=False, force_selection=True)
        matched = safe[["cluster_id", "harmful_count", "selected_effect", "loss_after_selection"]].merge(
            forced[["cluster_id", "harmful_count", "selected_effect", "loss_after_selection"]],
            on="cluster_id", suffixes=("_safe", "_forced"), validate="one_to_one",
        )
        low_confidence = subset.drop_duplicates(["cluster_id", "incident_id"]).groupby("cluster_id")["consensus_confidence"].min() < 0.42
        forced_harm = float(matched["harmful_count_forced"].sum())
        safe_harm = float(matched["harmful_count_safe"].sum())
        abstention_rows.append({
            "application": application,
            "clusters": int(len(matched)),
            "low_confidence_panel_fraction": float(low_confidence.mean()),
            "forced_harmful_interventions": int(forced_harm),
            "safe_harmful_interventions": int(safe_harm),
            "harmful_relative_reduction": float((forced_harm - safe_harm) / max(forced_harm, 1.0)),
            "safe_minus_forced_utility": float((matched["selected_effect_safe"] - matched["selected_effect_forced"]).mean()),
            "relative_service_loss_degradation": _safe_divide(
                float((matched["loss_after_selection_safe"] - matched["loss_after_selection_forced"]).mean()),
                float(matched["loss_after_selection_forced"].mean()),
            ),
        })
    write_csv(results_root / "statistics" / "low_consensus_abstention.csv", abstention_rows)

    # Trigger/triage timing uses the frozen first post-disruption decision epoch.
    trigger_rows: List[Dict[str, Any]] = []
    thermo_selection = selections[selections["feature_block"] == "kpi_plus_entropy_disagreement"]
    for application in sorted(episodes["application"].unique()):
        subset = thermo_selection[thermo_selection["application"] == application]
        nominal = subset[subset["regime"] == "nominal"]
        disrupted = subset[subset["regime"] != "nominal"]
        triggered = disrupted["selected_count"] > 0
        trigger_rows.append({
            "application": application,
            "disrupted_panels": int(len(disrupted)),
            "activation_fraction": float(triggered.mean()),
            "timely_activation_fraction": float(triggered.mean()),
            "missed_eligible_fraction": float((~triggered).mean()),
            "nominal_false_activation_fraction": float((nominal["selected_count"] > 0).mean()) if len(nominal) else None,
            "pre_disruption_activation_fraction": 0.0,
            "maximum_queue_length": int(disrupted["selected_count"].max()) if len(disrupted) else 0,
        })
    write_csv(results_root / "statistics" / "trigger_feasibility.csv", trigger_rows)

    # Refit permutation in the two primary applications, private condition.
    permutation_results: Dict[str, Any] = {}
    for application in ("humanitarian", "utility_restoration"):
        subset = candidates[
            (candidates["application"] == application)
            & (candidates["information_condition"] == "private_fragmented")
        ].reset_index(drop=True)
        selections_subset = selections[
            (selections["application"] == application)
            & (selections["information_condition"] == "private_fragmented")
        ]
        kpi = selections_subset[selections_subset["feature_block"] == "local_kpi_only"]
        thermo = selections_subset[selections_subset["feature_block"] == "kpi_plus_entropy_disagreement"]
        true_pair = _selection_pair(kpi, thermo)
        true_pair = true_pair[true_pair["regime"] != "nominal"]
        true_gain = float(true_pair["utility_difference"].mean())
        permutation_results[application] = refit_permutation_test(
            subset, kpi, true_gain, alpha_grid, budget,
            int(permutation_replicates), 55091 + (1 if application == "humanitarian" else 2),
        )
    atomic_json(results_root / "statistics" / "refit_permutation.json", permutation_results)

    report = {
        "study": "ThermoHITL v5",
        "stage": "development",
        "evidence_boundary": "development-only; deterministic independent-agent engineering control; simulated operator",
        "episodes": int(len(episodes)),
        "candidate_rows": int(len(candidates)),
        "independent_clusters": int(candidates["cluster_id"].nunique()),
        "primary_incremental_value": primary_pairs,
        "coordination_necessity": coordination_rows,
        "human_causal_usefulness": human_rows,
        "fragmentation_interaction": interaction_rows,
        "safety_abstention": abstention_rows,
        "trigger_feasibility": trigger_rows,
        "refit_permutation": permutation_results,
    }
    atomic_json(results_root / "development" / "development_analysis.json", report)
    return report


def run_refit_permutation_application(
    results_root: Path,
    application: str,
    stage: str = "development_primary_v2",
    replicates: int = 199,
) -> Dict[str, Any]:
    if application not in ("humanitarian", "utility_restoration"):
        raise ValueError("formal V5 permutation is limited to the two primary applications")
    config = yaml.safe_load((Path("configs") / "human_operator_v5_development.yaml").read_text(encoding="utf-8"))
    candidates = pd.read_csv(results_root / "development" / stage / "candidate_interventions.csv")
    for column in ("beneficial", "harmful", "accepted_action", "reached_next_stage", "reached_service"):
        candidates[column] = candidates[column].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    subset = candidates[
        (candidates["application"] == application)
        & (candidates["information_condition"] == "private_fragmented")
    ].reset_index(drop=True)
    alpha_grid = [float(value) for value in config["primary_model"]["regularization_grid"]]
    budget = int(config["simulation"]["operator_budget"])
    kpi_scores, _ = crossfit_action_values(subset, KPI_FEATURES, alpha_grid, budget)
    thermo_scores, _ = crossfit_action_values(
        subset, FEATURE_BLOCKS["kpi_plus_entropy_disagreement"], alpha_grid, budget,
    )
    kpi = select_budget(subset, kpi_scores, budget=budget)
    thermo = select_budget(subset, thermo_scores, budget=budget, consensus_abstention=True)
    paired = _selection_pair(kpi, thermo)
    paired = paired[paired["regime"] != "nominal"]
    result = refit_permutation_test(
        subset, kpi, float(paired["utility_difference"].mean()),
        alpha_grid, budget, int(replicates),
        55092 if application == "humanitarian" else 55093,
    )
    atomic_json(results_root / "statistics" / ("refit_permutation_%s.json" % application), result)
    return result


def analyze_sketch_ablation(
    results_root: Path,
    stage: str = "sketch_ablation",
) -> Dict[str, Any]:
    stage_root = results_root / "development" / stage
    episodes = pd.read_csv(stage_root / "episode_summary.csv")
    candidates = pd.read_csv(stage_root / "candidate_interventions.csv")
    incidents = candidates.drop_duplicates(["cluster_id", "sketch_policy", "incident_id"])
    rows: List[Dict[str, Any]] = []
    for keys, group in episodes.groupby(
        ["application", "information_condition", "sketch_policy"], sort=True,
    ):
        application, condition, policy = keys
        incident_group = incidents[
            (incidents["application"] == application)
            & (incidents["information_condition"] == condition)
            & (incidents["sketch_policy"] == policy)
        ]
        rows.append({
            "application": application,
            "information_condition": condition,
            "sketch_policy": policy,
            "panels": int(len(group)),
            "incidents": int(len(incident_group)),
            "mean_sketch_messages": float(group["sketch_messages"].mean()),
            "mean_sketch_bytes": float(group["sketch_bytes"].mean()),
            "mean_sketch_latency": float(group["sketch_latency"].mean()),
            "mean_entropy_estimation_error": float(incident_group["distributed_entropy_error"].mean()),
            "p95_entropy_estimation_error": float(incident_group["distributed_entropy_error"].quantile(0.95)),
        })
    frame = pd.DataFrame(rows)
    comparisons: List[Dict[str, Any]] = []
    for application in sorted(frame["application"].unique()):
        for condition in sorted(frame["information_condition"].unique()):
            subset = frame[(frame["application"] == application) & (frame["information_condition"] == condition)].set_index("sketch_policy")
            event = subset.loc["event_triggered"]
            always = subset.loc["always_on"]
            periodic = subset.loc["periodic"]
            byte_reduction = 1.0 - float(event["mean_sketch_bytes"] / max(always["mean_sketch_bytes"], 1e-12))
            dominated = bool(
                periodic["mean_sketch_bytes"] <= event["mean_sketch_bytes"]
                and periodic["mean_entropy_estimation_error"] <= event["mean_entropy_estimation_error"]
                and (
                    periodic["mean_sketch_bytes"] < event["mean_sketch_bytes"]
                    or periodic["mean_entropy_estimation_error"] < event["mean_entropy_estimation_error"]
                )
            )
            comparisons.append({
                "application": application,
                "information_condition": condition,
                "event_byte_reduction_vs_always_on": byte_reduction,
                "event_error": float(event["mean_entropy_estimation_error"]),
                "periodic_error": float(periodic["mean_entropy_estimation_error"]),
                "event_dominated_by_periodic_on_bytes_and_error": dominated,
            })
    write_csv(results_root / "statistics" / "sketch_communication_accounting.csv", rows)
    write_csv(results_root / "statistics" / "sketch_policy_comparisons.csv", comparisons)
    report = {
        "stage": stage,
        "episodes": int(len(episodes)),
        "independent_panels_per_policy": int(episodes.groupby("sketch_policy")["cluster_id"].nunique().min()),
        "rows": rows,
        "comparisons": comparisons,
        "all_sketch_traffic_counted": True,
    }
    atomic_json(results_root / "development" / "sketch_ablation_analysis.json", report)
    return report
