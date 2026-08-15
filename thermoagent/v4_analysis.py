"""Cluster-aware prospective gate analysis for ThermoHITL v4.

The independent unit is an environment panel.  Candidate interventions and
time steps are never treated as independent inferential observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .v4_experiments import utc_now, write_csv


LOCAL_KPI_FEATURES = (
    "service_deficit", "backlog", "lateness", "resource_scarcity",
    "commitment_strain", "safety_stress", "disruption_risk",
    "actionability_flag",
)
ENERGY_FEATURES = ("operational_energy", "standardized_energy")
ENTROPY_DISAGREEMENT_FEATURES = (
    "distributed_entropy", "entropy_anomaly", "entropy_slope",
    "belief_disagreement", "consensus_confidence",
)
COMPLETE_THERMO_FEATURES = (
    *ENERGY_FEATURES,
    "belief_entropy", "alternative_entropy", "commitment_entropy",
    *ENTROPY_DISAGREEMENT_FEATURES,
    "entropy_acceleration",
)
FEATURE_BLOCKS: Dict[str, Tuple[str, ...]] = {
    "local_kpi_only": LOCAL_KPI_FEATURES,
    "energy_only": ENERGY_FEATURES,
    "entropy_disagreement_only": ENTROPY_DISAGREEMENT_FEATURES,
    "kpi_plus_energy": (*LOCAL_KPI_FEATURES, *ENERGY_FEATURES),
    "kpi_plus_entropy_disagreement": (
        *LOCAL_KPI_FEATURES, *ENTROPY_DISAGREEMENT_FEATURES,
    ),
    "complete_thermodynamic": (*LOCAL_KPI_FEATURES, *COMPLETE_THERMO_FEATURES),
    "exploratory_free_energy": (
        *LOCAL_KPI_FEATURES, *COMPLETE_THERMO_FEATURES,
        "free_energy_diagnostic", "free_energy_residual",
    ),
}
PRIMARY_KPI_BLOCK = "local_kpi_only"
PRIMARY_THERMO_BLOCK = "kpi_plus_entropy_disagreement"


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _summary(results_root: Path, stage: str) -> pd.DataFrame:
    path = results_root / "development" / stage / "episode_summary.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _candidates(results_root: Path, stage: str) -> pd.DataFrame:
    path = results_root / "development" / stage / "candidate_interventions.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    return frame


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    return float(average_precision_score(labels, scores)) if labels.sum() else None


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def _condition_number(frame: pd.DataFrame, features: Sequence[str]) -> Optional[float]:
    matrix = frame[list(features)].to_numpy(dtype=float)
    scale = matrix.std(axis=0)
    retained = scale > 1e-12
    if not np.any(retained):
        return None
    standardized = (matrix[:, retained] - matrix[:, retained].mean(axis=0)) / scale[retained]
    if standardized.shape[1] == 1:
        return 1.0
    value = float(np.linalg.cond(standardized))
    return value if np.isfinite(value) else None


def _budget_selection(
    frame: pd.DataFrame,
    score_column: str,
    budget: int,
) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["cluster_id", score_column, "incident_id"],
        ascending=[True, False, True],
    )
    selected = ranked.groupby("cluster_id", sort=True).head(int(budget)).copy()
    return selected.groupby("cluster_id", sort=True).agg(
        causal_utility=("causal_utility", "sum"),
        intervention_effect=("intervention_effect", "sum"),
        operator_minutes=("operator_minutes", "sum"),
        mean_branch_loss=("loss_with_intervention", "mean"),
        harmful_interventions=("harmful", "sum"),
        selected_interventions=("incident_id", "size"),
        selected_incidents=("incident_id", lambda values: ";".join(sorted(map(str, values)))),
        application=("application", "first"),
        regime=("regime", "first"),
        information_condition=("information_condition", "first"),
        environment_seed=("environment_seed", "first"),
    )


def _utility_for_scores(frame: pd.DataFrame, scores: np.ndarray, budget: int) -> float:
    temporary = frame.copy()
    temporary["_score"] = scores
    selected = _budget_selection(temporary, "_score", budget)
    return float(selected["causal_utility"].mean()) if len(selected) else float("-inf")


@dataclass
class FoldModel:
    indices: np.ndarray
    model: Optional[Pipeline]
    constant_score: float
    selected_c: Optional[float]


def _fit_model(
    train: pd.DataFrame,
    features: Sequence[str],
    c_value: float,
) -> Tuple[Optional[Pipeline], float]:
    labels = train["beneficial"].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        return None, float(labels.mean())
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value), class_weight="balanced", max_iter=2500,
            random_state=44041,
        ),
    )
    model.fit(train[list(features)].to_numpy(dtype=float), labels)
    return model, float(labels.mean())


def _select_regularization(
    train: pd.DataFrame,
    features: Sequence[str],
    c_grid: Sequence[float],
    budget: int,
) -> float:
    groups = train["environment_seed"].to_numpy()
    unique = np.unique(groups)
    if len(unique) < 3 or train["beneficial"].nunique() < 2:
        return float(c_grid[0])
    folds = GroupKFold(n_splits=min(3, len(unique)))
    best_c = float(c_grid[0])
    best_utility = float("-inf")
    for c_value in c_grid:
        scores = np.zeros(len(train), dtype=float)
        for inner_train, inner_test in folds.split(train, groups=groups):
            model, constant = _fit_model(train.iloc[inner_train], features, float(c_value))
            if model is None:
                scores[inner_test] = constant
            else:
                scores[inner_test] = model.predict_proba(
                    train.iloc[inner_test][list(features)].to_numpy(dtype=float)
                )[:, 1]
        utility = _utility_for_scores(train, scores, budget)
        if utility > best_utility + 1e-12:
            best_utility = utility
            best_c = float(c_value)
    return best_c


def crossfit_scores(
    frame: pd.DataFrame,
    features: Sequence[str],
    c_grid: Sequence[float],
    budget: int,
) -> Tuple[np.ndarray, List[FoldModel]]:
    groups = frame["environment_seed"].to_numpy()
    unique = np.unique(groups)
    if len(unique) < 4:
        raise ValueError("v4 cross-fitting requires at least four environment seeds")
    scores = np.zeros(len(frame), dtype=float)
    folds: List[FoldModel] = []
    splitter = GroupKFold(n_splits=min(4, len(unique)))
    for train_index, test_index in splitter.split(frame, groups=groups):
        train = frame.iloc[train_index]
        selected_c = _select_regularization(train, features, c_grid, budget)
        model, constant = _fit_model(train, features, selected_c)
        if model is None:
            scores[test_index] = constant
        else:
            scores[test_index] = model.predict_proba(
                frame.iloc[test_index][list(features)].to_numpy(dtype=float)
            )[:, 1]
        folds.append(FoldModel(test_index, model, constant, selected_c))
    return scores, folds


def _paired_bootstrap(
    differences: np.ndarray,
    replicates: int,
    seed: int,
) -> Dict[str, Any]:
    if not len(differences):
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, len(differences), size=(int(replicates), len(differences)))
    values = differences[indices].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def _permuted_score_distribution(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[FoldModel],
    kpi_selection: pd.DataFrame,
    true_gain: float,
    budget: int,
    replicates: int,
    seed: int,
) -> Dict[str, Any]:
    thermo_columns = [value for value in features if value not in LOCAL_KPI_FEATURES]
    thermo_positions = [list(features).index(value) for value in thermo_columns]
    severity = (
        0.55 * frame["service_deficit"]
        + 0.25 * frame["backlog"]
        + 0.20 * frame["safety_stress"]
    )
    strata = (severity * 20.0).round().astype(int).astype(str)
    grouping = frame["regime"].astype(str) + "|" + strata
    rng = np.random.RandomState(int(seed))
    gains = np.zeros(int(replicates), dtype=float)
    fold_arrays: List[Tuple[FoldModel, np.ndarray, List[np.ndarray]]] = []
    for fold in folds:
        matrix = frame.iloc[fold.indices][list(features)].to_numpy(dtype=float)
        fold_groups = grouping.iloc[fold.indices].to_numpy()
        positions = [
            np.flatnonzero(fold_groups == value)
            for value in sorted(set(fold_groups))
        ]
        fold_arrays.append((fold, matrix, positions))
    for replicate in range(int(replicates)):
        scores = np.zeros(len(frame), dtype=float)
        for fold, base_matrix, group_positions in fold_arrays:
            if fold.model is None:
                scores[fold.indices] = fold.constant_score
                continue
            matrix = base_matrix.copy()
            for positions in group_positions:
                if len(positions) <= 1:
                    continue
                permutation = rng.permutation(positions)
                matrix[np.ix_(positions, thermo_positions)] = base_matrix[
                    np.ix_(permutation, thermo_positions)
                ]
            scores[fold.indices] = fold.model.predict_proba(matrix)[:, 1]
        temporary = frame.copy()
        temporary["_permuted"] = scores
        selected = _budget_selection(temporary, "_permuted", budget)
        paired = selected.join(
            kpi_selection[["causal_utility"]], how="inner", rsuffix="_kpi"
        )
        gains[replicate] = float(
            (paired["causal_utility"] - paired["causal_utility_kpi"]).mean()
        )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_permuted_gain": float(gains.mean()),
        "median_permuted_gain": float(np.median(gains)),
        "permuted_gain_ci95_low": float(np.quantile(gains, 0.025)),
        "permuted_gain_ci95_high": float(np.quantile(gains, 0.975)),
        "probability_permuted_at_least_true": float(np.mean(gains >= true_gain - 1e-12)),
        "mean_gain_fraction_of_true": (
            float(gains.mean() / true_gain) if true_gain > 1e-12 else None
        ),
    }


def analyze_feature_blocks(
    candidates: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    budget = int(protocol["simulation"]["operator_budget"])
    c_grid = [float(value) for value in protocol["primary_model"]["regularization_grid"]]
    bootstrap = protocol["gates"]["thermodynamic_incremental_value"]
    permutation = protocol["gates"]["mechanism_specificity"]
    performance_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    primary_rows: List[Dict[str, Any]] = []
    permutation_rows: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    for application in sorted(candidates["application"].unique()):
        for information in sorted(candidates["information_condition"].unique()):
            frame = candidates[
                (candidates.application == application)
                & (candidates.information_condition == information)
            ].copy().reset_index(drop=True)
            block_scores: Dict[str, np.ndarray] = {}
            block_folds: Dict[str, List[FoldModel]] = {}
            block_selected: Dict[str, pd.DataFrame] = {}
            for block, features in FEATURE_BLOCKS.items():
                scores, folds = crossfit_scores(frame, features, c_grid, budget)
                score_column = "score_" + block
                frame[score_column] = scores
                block_scores[block] = scores
                block_folds[block] = folds
                selected = _budget_selection(frame, score_column, budget)
                block_selected[block] = selected
                labels = frame["beneficial"].to_numpy(dtype=int)
                performance_rows.append({
                    "evidence_stage": "development",
                    "application": application,
                    "information_condition": information,
                    "feature_block": block,
                    "candidate_rows": len(frame),
                    "independent_clusters": frame.cluster_id.nunique(),
                    "positive_prevalence": float(labels.mean()),
                    "average_precision": _safe_ap(labels, scores),
                    "roc_auc": _safe_auc(labels, scores),
                    "brier_score": float(brier_score_loss(labels, scores)),
                    "condition_number": _condition_number(frame, features),
                    "mean_budgeted_causal_utility": float(selected.causal_utility.mean()),
                    "mean_operator_minutes": float(selected.operator_minutes.mean()),
                    "harmful_intervention_rate": float(
                        selected.harmful_interventions.sum()
                        / max(selected.selected_interventions.sum(), 1)
                    ),
                    "selected_c_values": ";".join(
                        str(value.selected_c) for value in folds
                    ),
                    "confirmatory": False,
                })
                for cluster_id, row in selected.iterrows():
                    selection_rows.append({
                        "application": application,
                        "information_condition": information,
                        "feature_block": block,
                        "cluster_id": cluster_id,
                        **row.to_dict(),
                    })
            kpi = block_selected[PRIMARY_KPI_BLOCK]
            thermo = block_selected[PRIMARY_THERMO_BLOCK]
            paired = thermo.join(kpi, how="inner", lsuffix="_thermo", rsuffix="_kpi")
            differences = (
                paired["causal_utility_thermo"] - paired["causal_utility_kpi"]
            ).to_numpy(dtype=float)
            bootstrap_result = _paired_bootstrap(
                differences,
                int(bootstrap["bootstrap_replicates"]),
                int(bootstrap["bootstrap_seed"])
                + (0 if application == "commercial" else 100 if application == "humanitarian" else 200)
                + (0 if information == "private_fragmented" else 10),
            )
            kpi_mean = float(paired["causal_utility_kpi"].mean())
            thermo_mean = float(paired["causal_utility_thermo"].mean())
            relative_gain = (
                float((thermo_mean - kpi_mean) / abs(kpi_mean))
                if abs(kpi_mean) >= 1e-9 else None
            )
            harmful_kpi = float(
                paired.harmful_interventions_kpi.sum()
                / max(paired.selected_interventions_kpi.sum(), 1)
            )
            harmful_thermo = float(
                paired.harmful_interventions_thermo.sum()
                / max(paired.selected_interventions_thermo.sum(), 1)
            )
            mean_kpi_loss = float(paired.mean_branch_loss_kpi.mean())
            mean_thermo_loss = float(paired.mean_branch_loss_thermo.mean())
            loss_degradation = float(
                (mean_thermo_loss - mean_kpi_loss) / max(abs(mean_kpi_loss), 1e-9)
            )
            row = {
                "evidence_stage": "development",
                "application": application,
                "information_condition": information,
                "candidate_rows": len(frame),
                "independent_clusters": len(paired),
                "operator_budget": budget,
                "kpi_mean_causal_utility": kpi_mean,
                "thermodynamic_mean_causal_utility": thermo_mean,
                "paired_mean_utility_gain": bootstrap_result["mean"],
                "utility_gain_ci95_low": bootstrap_result["ci95_low"],
                "utility_gain_ci95_high": bootstrap_result["ci95_high"],
                "relative_utility_gain": relative_gain,
                "kpi_harmful_intervention_rate": harmful_kpi,
                "thermodynamic_harmful_intervention_rate": harmful_thermo,
                "harmful_rate_increase": harmful_thermo - harmful_kpi,
                "kpi_mean_branch_loss": mean_kpi_loss,
                "thermodynamic_mean_branch_loss": mean_thermo_loss,
                "relative_service_loss_degradation": loss_degradation,
                "bootstrap_replicates": bootstrap_result.get("bootstrap_replicates"),
                "confirmatory": False,
            }
            primary_rows.append(row)
            for cluster_id, values in paired.iterrows():
                cluster_rows.append({
                    "application": application,
                    "information_condition": information,
                    "cluster_id": cluster_id,
                    "regime": values["regime_thermo"],
                    "environment_seed": int(values["environment_seed_thermo"]),
                    "kpi_causal_utility": values["causal_utility_kpi"],
                    "thermodynamic_causal_utility": values["causal_utility_thermo"],
                    "paired_utility_gain": values["causal_utility_thermo"] - values["causal_utility_kpi"],
                    "kpi_selected_incidents": values["selected_incidents_kpi"],
                    "thermodynamic_selected_incidents": values["selected_incidents_thermo"],
                })
            if (
                information == "private_fragmented"
                and application in {"humanitarian", "utility_restoration"}
            ):
                permutation_result = _permuted_score_distribution(
                    frame,
                    FEATURE_BLOCKS[PRIMARY_THERMO_BLOCK],
                    block_folds[PRIMARY_THERMO_BLOCK],
                    kpi,
                    float(bootstrap_result["mean"] or 0.0),
                    budget,
                    int(permutation["within_kpi_stratum_permutation_replicates"]),
                    int(permutation["permutation_seed"])
                    + (100 if application == "humanitarian" else 200),
                )
                permutation_rows.append({
                    "application": application,
                    "information_condition": information,
                    "true_paired_utility_gain": bootstrap_result["mean"],
                    **permutation_result,
                })
    return {
        "performance_rows": performance_rows,
        "selection_rows": selection_rows,
        "primary_rows": primary_rows,
        "permutation_rows": permutation_rows,
        "cluster_rows": cluster_rows,
    }


def _paired_method_rows(
    frame: pd.DataFrame,
    reference: str,
    treatment: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for application in sorted(frame.application.unique()):
        selected = frame[frame.application == application]
        for regime in ["aggregate", *sorted(selected.regime.unique())]:
            subset = selected if regime == "aggregate" else selected[selected.regime == regime]
            pivot = subset.pivot_table(
                index=["environment_seed", "regime", "information_condition"],
                columns="method", values="primary_outcome", aggfunc="first",
            )
            if reference not in pivot or treatment not in pivot:
                continue
            pivot = pivot.dropna(subset=[reference, treatment])
            ref = pivot[reference].to_numpy(dtype=float)
            treated = pivot[treatment].to_numpy(dtype=float)
            differences = treated - ref
            relative_reduction = (ref - treated) / np.maximum(np.abs(ref), 1e-9)
            boot = _paired_bootstrap(differences, 10_000, 44100 + len(output))
            output.append({
                "application": application,
                "regime": regime,
                "reference": reference,
                "treatment": treatment,
                "independent_panels": len(pivot),
                "reference_mean_loss": float(ref.mean()),
                "treatment_mean_loss": float(treated.mean()),
                "mean_treatment_minus_reference": float(differences.mean()),
                "difference_ci95_low": boot["ci95_low"],
                "difference_ci95_high": boot["ci95_high"],
                "mean_relative_loss_reduction": float(relative_reduction.mean()),
                "paired_win_rate": float(np.mean(differences < -1e-12)),
                "evidence_stage": "development",
            })
    return output


def _gate3(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    threshold = protocol["gates"]["coordination_necessity"]
    applications: Dict[str, Any] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        app_rows = [row for row in rows if row["application"] == application]
        aggregate = next((row for row in app_rows if row["regime"] == "aggregate"), None)
        improved = [
            row["regime"] for row in app_rows
            if row["regime"] != "aggregate"
            and row["mean_relative_loss_reduction"]
            >= float(threshold["per_regime_relative_loss_reduction_minimum"])
        ]
        passed = bool(
            aggregate
            and aggregate["mean_relative_loss_reduction"]
            >= float(threshold["aggregate_relative_loss_reduction_minimum"])
            and len(improved)
            >= int(threshold["improved_fragmented_regimes_per_application_minimum"])
        )
        applications[application] = {
            "passed": passed,
            "aggregate_relative_loss_reduction": None if aggregate is None else aggregate["mean_relative_loss_reduction"],
            "improved_regimes": improved,
        }
    return {"passed": all(value["passed"] for value in applications.values()), "applications": applications}


def _gate4(
    rows: Sequence[Mapping[str, Any]],
    human_summary: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    threshold = protocol["gates"]["human_causal_usefulness"]
    applications: Dict[str, Any] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        app_rows = [row for row in rows if row["application"] == application]
        aggregate = next((row for row in app_rows if row["regime"] == "aggregate"), None)
        improved = [
            row["regime"] for row in app_rows
            if row["regime"] != "aggregate"
            and row["mean_relative_loss_reduction"]
            >= float(threshold["per_regime_relative_loss_reduction_minimum"])
        ]
        treatment = human_summary[
            (human_summary.application == application)
            & (human_summary.method == "thermohitl_v4_rule")
        ]
        chains = int(treatment.complete_causal_chains.sum()) if len(treatment) else 0
        passed = bool(
            aggregate
            and aggregate["mean_relative_loss_reduction"]
            >= float(threshold["relative_loss_reduction_minimum"])
            and len(improved) >= int(threshold["improved_regimes_per_application_minimum"])
            and chains >= int(threshold["complete_causal_chain_minimum"])
        )
        applications[application] = {
            "passed": passed,
            "aggregate_relative_loss_reduction": None if aggregate is None else aggregate["mean_relative_loss_reduction"],
            "improved_regimes": improved,
            "complete_causal_chains": chains,
        }
    pass_count = sum(value["passed"] for value in applications.values())
    return {
        "passed": pass_count >= int(threshold["required_application_count"]),
        "passing_application_count": pass_count,
        "applications": applications,
    }


def _gate5(primary_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    threshold = protocol["gates"]["thermodynamic_incremental_value"]
    applications: Dict[str, Any] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        row = next((
            value for value in primary_rows
            if value["application"] == application
            and value["information_condition"] == "private_fragmented"
        ), None)
        passed = bool(
            row
            and float(row["paired_mean_utility_gain"]) > 0.0
            and float(row["utility_gain_ci95_low"]) > float(threshold["cluster_bootstrap_lower_bound_minimum"])
            and float(row["relative_utility_gain"] or -1e9)
            >= float(threshold["relative_budgeted_utility_gain_minimum"])
            and float(row["harmful_rate_increase"])
            <= float(threshold["harmful_intervention_rate_increase_maximum"])
            and float(row["relative_service_loss_degradation"])
            <= float(threshold["service_loss_noninferiority_margin"])
        )
        applications[application] = {"passed": passed, "result": row}
    required = threshold["required_applications"]
    return {
        "passed": all(applications[value]["passed"] for value in required),
        "required_applications": required,
        "applications": applications,
    }


def _gate6(trigger_summary: pd.DataFrame, protocol: Mapping[str, Any]) -> Dict[str, Any]:
    threshold = protocol["gates"]["trigger_feasibility"]
    disrupted = trigger_summary[trigger_summary.regime != "nominal"]
    nominal = trigger_summary[trigger_summary.regime == "nominal"]
    by_regime: List[Dict[str, Any]] = []
    every_nonzero = True
    for (application, regime), group in disrupted.groupby(["application", "regime"], sort=True):
        activation = float(np.mean(group.operator_requests.astype(float) > 0))
        timely = float(group.timely_activation.astype(str).str.lower().eq("true").mean())
        every_nonzero = every_nonzero and activation > 0.0
        by_regime.append({
            "application": application, "regime": regime,
            "activation_rate": activation, "timely_activation_rate": timely,
            "episodes": len(group),
        })
    timely = float(disrupted.timely_activation.astype(str).str.lower().eq("true").mean()) if len(disrupted) else 0.0
    pre_false = float(disrupted.pre_disruption_false_activation.astype(str).str.lower().eq("true").mean()) if len(disrupted) else 1.0
    nominal_false = float(nominal.nominal_false_activation.astype(str).str.lower().eq("true").mean()) if len(nominal) else 1.0
    active_fraction = float(disrupted.communication_active_agent_epoch_fraction.mean()) if len(disrupted) else 1.0
    low_count = int(disrupted.low_confidence_operator_decisions.sum()) if len(disrupted) else 0
    safe_count = int(disrupted.safe_low_confidence_decisions.sum()) if len(disrupted) else 0
    safe_rate = safe_count / max(low_count, 1)
    passed = bool(
        every_nonzero
        and timely >= float(threshold["timely_activation_minimum"])
        and pre_false <= float(threshold["pre_disruption_false_activation_maximum"])
        and nominal_false <= float(threshold["nominal_false_activation_maximum"])
        and active_fraction <= float(threshold["maximum_active_epoch_fraction"])
        and (low_count == 0 or safe_rate >= 0.95)
    )
    return {
        "passed": passed,
        "nonzero_every_disrupted_application_regime": every_nonzero,
        "timely_activation_rate": timely,
        "pre_disruption_false_activation_rate": pre_false,
        "nominal_false_activation_rate": nominal_false,
        "communication_active_agent_epoch_fraction": active_fraction,
        "low_confidence_decisions": low_count,
        "safe_low_confidence_decision_rate": safe_rate,
        "by_regime": by_regime,
    }


def _gate7(
    primary_rows: Sequence[Mapping[str, Any]],
    permutation_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    threshold = protocol["gates"]["mechanism_specificity"]
    applications: Dict[str, Any] = {}
    for application in ("humanitarian", "utility_restoration"):
        fragmented = next(value for value in primary_rows if value["application"] == application and value["information_condition"] == "private_fragmented")
        public = next(value for value in primary_rows if value["application"] == application and value["information_condition"] == "globally_public")
        permutation = next(value for value in permutation_rows if value["application"] == application and value["information_condition"] == "private_fragmented")
        fragmented_gain = float(fragmented["relative_utility_gain"] or 0.0)
        public_gain = float(public["relative_utility_gain"] or 0.0)
        interaction = fragmented_gain - public_gain
        mean_fraction = permutation.get("mean_gain_fraction_of_true")
        passed = bool(
            interaction >= float(threshold["fragmented_minus_public_gain_minimum"])
            and mean_fraction is not None
            and float(mean_fraction) <= float(threshold["shuffled_gain_fraction_maximum"])
        )
        applications[application] = {
            "passed": passed,
            "fragmented_relative_gain": fragmented_gain,
            "public_relative_gain": public_gain,
            "fragmented_minus_public_gain": interaction,
            "permuted_mean_gain_fraction": mean_fraction,
            "permutation_probability_at_least_true": permutation["probability_permuted_at_least_true"],
        }
    return {"passed": all(value["passed"] for value in applications.values()), "applications": applications}


def analyze_v4_development(repository: Path) -> Dict[str, Any]:
    results_root = repository / "results" / "human_operator_v4"
    protocol = yaml.safe_load(
        (repository / "configs" / "human_operator_v4_development.yaml").read_text(encoding="utf-8")
    )
    coordination = _summary(results_root, "development_gate_coordination")
    human = _summary(results_root, "development_gate_human")
    trigger = _summary(results_root, "development_gate_trigger")
    candidates = _candidates(results_root, "development_gate_monitoring")
    if any(value.empty for value in (coordination, human, trigger, candidates)):
        raise ValueError("formal v4 development stages are incomplete")

    coordination_rows = _paired_method_rows(
        coordination, "no_communication", "fixed_communication"
    )
    human_rows = _paired_method_rows(
        human, "autonomy_no_operator", "thermohitl_v4_rule"
    )
    features = analyze_feature_blocks(candidates, protocol)
    gate3 = _gate3(coordination_rows, protocol)
    gate4 = _gate4(human_rows, human, protocol)
    gate5 = _gate5(features["primary_rows"], protocol)
    gate6 = _gate6(trigger, protocol)
    gate7 = _gate7(features["primary_rows"], features["permutation_rows"], protocol)

    replay = _json(results_root / "reproducibility" / "v4_replay_report.json")
    tests = _json(results_root / "reproducibility" / "test_report.json")
    qwen = _json(results_root / "development" / "real_qwen_qualification.json")
    gate1 = {
        "passed": bool(
            tests.get("failed", 1) == 0
            and replay.get("mismatches", 1) == 0
            and replay.get("maximum_conservation_residual", 1.0) <= 1e-9
        ),
        "tests": tests,
        "replay_summary": {
            key: replay.get(key) for key in (
                "episodes_replayed", "mismatches", "maximum_conservation_residual"
            )
        },
    }
    gate2 = {
        "passed": bool(qwen.get("passed", False)),
        "real_qwen_qualification": qwen,
        "deterministic_first_pass_validity": float(
            human.first_pass_valid.sum() / max(human.structured_attempts.sum(), 1)
        ),
        "deterministic_accepted_to_next_stage": float(
            human.material_actions_next_stage.sum() / max(human.material_actions_accepted.sum(), 1)
        ),
        "deterministic_accepted_to_service": float(
            human.material_actions_reached_service.sum() / max(human.material_actions_accepted.sum(), 1)
        ),
    }
    gates = {
        "gate_1_engineering_integrity": gate1,
        "gate_2_agent_actionability": gate2,
        "gate_3_coordination_necessity": gate3,
        "gate_4_human_causal_usefulness": gate4,
        "gate_5_thermodynamic_incremental_value": gate5,
        "gate_6_trigger_feasibility": gate6,
        "gate_7_mechanism_specificity": gate7,
    }
    passed = all(value["passed"] for value in gates.values())
    report = {
        "study": "ThermoHITL v4",
        "generated_at": utc_now(),
        "evidence_stage": "development",
        "all_required_gates_passed": passed,
        "decision": "unlock_validation_and_training" if passed else "stop_before_validation",
        "gates": gates,
        "prohibited_inference": "development evidence is not confirmatory and simulated operators are not human participants",
    }

    write_csv(results_root / "statistics" / "coordination_paired_effects.csv", coordination_rows)
    write_csv(results_root / "statistics" / "human_causal_paired_effects.csv", human_rows)
    write_csv(results_root / "statistics" / "feature_block_performance.csv", features["performance_rows"])
    write_csv(results_root / "statistics" / "budgeted_causal_utility.csv", features["primary_rows"])
    write_csv(results_root / "statistics" / "candidate_selection_by_cluster.csv", features["cluster_rows"])
    write_csv(results_root / "statistics" / "conditional_permutation_test.csv", features["permutation_rows"])
    write_csv(results_root / "tables" / "feature_block_comparison.csv", features["performance_rows"])
    write_csv(results_root / "tables" / "primary_incremental_value.csv", features["primary_rows"])
    _write_json(results_root / "development" / "gate_status.json", report)
    return report
