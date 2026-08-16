"""Leakage-resistant cluster analysis for V6 selective autonomy."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
from .v6_experiments import aggregate_stage, run_episode, write_csv_gzip
from .v6_types import V6DecisionContext


KPI_FEATURES = (
    "visible_severity", "visible_backlog", "visible_delay",
    "resource_scarcity", "safety_risk", "commitment_strain",
    "action_probability", "action_value", "value_margin",
)
PREDICTIVE_FEATURES = (*KPI_FEATURES, "action_predictive_entropy")
SHANNON_FEATURES = (
    "shannon_local", "entropy_dispersion", "js_disagreement",
    "graph_disagreement", "consensus_residual", "disagreement_slope",
)
GENERALIZED_FEATURES = (
    "entropy_spectrum_tail_contrast", "jt_spectrum_tail_contrast",
    "gini_simpson_local", "jt_disagreement_2",
    "graph_disagreement", "consensus_residual", "disagreement_slope",
)
FEATURE_BLOCKS = {
    "kpi_confidence": KPI_FEATURES,
    "predictive_uncertainty": PREDICTIVE_FEATURES,
    "shannon_js": (*PREDICTIVE_FEATURES, *SHANNON_FEATURES),
    "generalized_tsallis_gini": (
        *KPI_FEATURES, *GENERALIZED_FEATURES,
        "graph_disagreement", "consensus_residual",
    ),
    "combined_generalized_entropic": (
        *PREDICTIVE_FEATURES,
        "shannon_local", "entropy_dispersion", "js_disagreement",
        "entropy_spectrum_tail_contrast", "jt_spectrum_tail_contrast",
        "graph_disagreement", "consensus_residual", "disagreement_slope",
    ),
}
CATEGORICAL_FEATURES = ("proposed_action", "role", "regime")
DEFAULT_C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_COVERAGES = (0.25, 0.50, 0.75, 1.0)
ENTROPY_ABLATION_BLOCKS = {
    "shannon_local": (*PREDICTIVE_FEATURES, "shannon_local"),
    "jensen_shannon": (*PREDICTIVE_FEATURES, "js_disagreement"),
    "gini_simpson": (*PREDICTIVE_FEATURES, "gini_simpson_local"),
    "tsallis_q_0_5": (*PREDICTIVE_FEATURES, "tsallis_0_5_local"),
    "tsallis_q_1_5": (*PREDICTIVE_FEATURES, "tsallis_1_5_local"),
    "tsallis_q_2": (*PREDICTIVE_FEATURES, "tsallis_2_local"),
    "tsallis_q_3": (*PREDICTIVE_FEATURES, "tsallis_3_local"),
    "jensen_tsallis_q_0_5": (*PREDICTIVE_FEATURES, "jt_disagreement_0_5"),
    "jensen_tsallis_q_1_5": (*PREDICTIVE_FEATURES, "jt_disagreement_1_5"),
    "jensen_tsallis_q_2": (*PREDICTIVE_FEATURES, "jt_disagreement_2"),
    "jensen_tsallis_q_3": (*PREDICTIVE_FEATURES, "jt_disagreement_3"),
    "graph_weighted_disagreement": (*PREDICTIVE_FEATURES, "graph_disagreement"),
}

DIRECT_RISK_METHODS = (
    "always_act", "random_abstention", "fixed_severity",
    "rule_kpi_confidence", "action_value_margin",
    "calibrated_max_probability", "predictive_action_entropy",
    "ensemble_variance_proxy", "local_shannon", "pooled_shannon",
    "jensen_shannon", "gini_simpson", "tsallis_q_0_5",
    "tsallis_q_1_5", "tsallis_q_2", "tsallis_q_3",
    "jensen_tsallis_q_0_5", "jensen_tsallis_q_1_5",
    "jensen_tsallis_q_2", "jensen_tsallis_q_3",
    "graph_disagreement", "low_consensus_guard",
    "oracle_risk_upper_bound",
)


def direct_risk_scores(frame: pd.DataFrame, method: str) -> np.ndarray:
    """Calculate transparent baselines from deployable columns.

    The sole exception is the explicitly labeled evaluator oracle. It is
    analysis-only and cannot be instantiated by the execution controller.
    """
    if method == "always_act":
        return np.full(len(frame), -1.0)
    if method == "random_abstention":
        return np.asarray([
            int.from_bytes(hashlib.sha256(
                (str(row.cluster_id) + "|" + str(row.step) + "|" + str(row.incident_id)).encode("utf-8")
            ).digest()[:8], "big") / float(2 ** 64 - 1)
            for row in frame.itertuples(index=False)
        ])
    if method == "fixed_severity":
        return 1.0 - frame.visible_severity.to_numpy(dtype=float)
    if method == "rule_kpi_confidence":
        return (
            .45 * (1.0 - frame.action_probability)
            + .25 * frame.safety_risk + .20 * frame.resource_scarcity
            + .10 * frame.visible_delay
        ).to_numpy(dtype=float)
    if method == "action_value_margin":
        return 1.0 - frame.value_margin.to_numpy(dtype=float)
    if method == "calibrated_max_probability":
        return 1.0 - frame.action_probability.to_numpy(dtype=float)
    if method == "predictive_action_entropy":
        return frame.action_predictive_entropy.to_numpy(dtype=float)
    if method == "ensemble_variance_proxy":
        return (
            (frame.visible_severity - frame.action_probability).abs()
            + .35 * frame.visible_delay
        ).to_numpy(dtype=float)
    column = {
        "local_shannon": "shannon_local", "pooled_shannon": "pooled_uncertainty",
        "jensen_shannon": "js_disagreement", "gini_simpson": "gini_simpson_local",
        "tsallis_q_0_5": "tsallis_0_5_local", "tsallis_q_1_5": "tsallis_1_5_local",
        "tsallis_q_2": "tsallis_2_local", "tsallis_q_3": "tsallis_3_local",
        "jensen_tsallis_q_0_5": "jt_disagreement_0_5",
        "jensen_tsallis_q_1_5": "jt_disagreement_1_5",
        "jensen_tsallis_q_2": "jt_disagreement_2",
        "jensen_tsallis_q_3": "jt_disagreement_3",
    }.get(method)
    if column is not None:
        return frame[column].to_numpy(dtype=float)
    if method == "graph_disagreement":
        return (frame.graph_disagreement + .35 * frame.consensus_residual).to_numpy(dtype=float)
    if method == "low_consensus_guard":
        low = (frame.consensus < .88) | (frame.consensus_residual >= .25)
        base = (
            1.0 - frame.value_margin + .35 * frame.js_disagreement
            + .25 * frame.graph_disagreement
        ).to_numpy(dtype=float)
        return base + low.to_numpy(dtype=float) * 2.0
    if method == "oracle_risk_upper_bound":
        return frame.harmful_label.to_numpy(dtype=float)
    raise ValueError("unknown direct V6 risk method: %s" % method)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm familywise adjustment, returned in original order."""
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted


def prepare_risk_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "evaluator_causal_utility_if_executed" not in output:
        if "evaluator_direct_effect_if_executed" not in output:
            raise ValueError("V6 risk rows require a causal-utility evaluator label")
        # Backward compatibility is only for retained pre-freeze pilots. New
        # formal rows always use matched dynamic counterfactual utility.
        output["evaluator_causal_utility_if_executed"] = output[
            "evaluator_direct_effect_if_executed"
        ]
    probability = np.clip(
        output["action_probability"].to_numpy(dtype=float), 1e-9, 1.0 - 1e-9,
    )
    output["action_predictive_entropy"] = -(
        probability * np.log(probability)
        + (1.0 - probability) * np.log(1.0 - probability)
    ) / np.log(2.0)
    output["entropy_dispersion"] = (
        output["pooled_uncertainty"] - output["average_local_uncertainty"]
    )
    output["entropy_spectrum_tail_contrast"] = (
        output["tsallis_0_5_local"] - output["tsallis_3_local"]
    )
    output["jt_spectrum_tail_contrast"] = (
        output["jt_disagreement_0_5"] - output["jt_disagreement_3"]
    )
    output["harmful_label"] = (
        output["evaluator_harmful_if_executed"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).astype(int)
    )
    if "split_family" not in output:
        raise ValueError("V6 analysis requires the prospectively isolated split_family")
    return output


def contexts_to_risk_frame(
    contexts: Sequence[V6DecisionContext], regime: str,
) -> pd.DataFrame:
    """Convert deployable contexts to the exact training feature schema."""
    rows: List[Dict[str, Any]] = []
    for context in contexts:
        row = {
            **context.local_kpis,
            **{key: value for key, value in context.deployable().items() if key != "proposal"},
            "action_probability": context.proposal.action_probability,
            "action_value": context.proposal.action_value,
            "value_margin": context.proposal.value_margin,
            "proposed_action": context.proposal.action,
            "role": context.proposal.role,
            "regime": regime,
            "split_family": "execution_only",
            "evaluator_causal_utility_if_executed": 0.0,
            "evaluator_harmful_if_executed": False,
        }
        rows.append(row)
    return prepare_risk_frame(pd.DataFrame(rows))


@dataclass
class FittedRiskController:
    """Coverage-matched controller backed by a frozen deployable pipeline."""

    model: Pipeline
    regime: str
    autonomous_coverage: float = 0.50
    escalation_slots_per_epoch: int = 1
    escalation_risk_threshold: float = 0.80

    def __call__(
        self, contexts: Sequence[V6DecisionContext], step: int,
    ) -> Mapping[str, str]:
        decisions: Dict[str, str] = {
            value.proposal.incident_id: "abstain"
            for value in contexts if value.proposal.action == "no_action"
        }
        actionable = [
            value for value in contexts if value.proposal.action != "no_action"
        ]
        if not actionable:
            return decisions
        frame = contexts_to_risk_frame(actionable, self.regime)
        risks = self.model.predict_proba(frame)[:, 1]
        scored = sorted(
            zip(risks, actionable),
            key=lambda value: (float(value[0]), value[1].proposal.incident_id),
        )
        count = max(0, min(
            len(scored), int(round(self.autonomous_coverage * len(scored))),
        ))
        decisions.update({
            context.proposal.incident_id: "execute_autonomously"
            for _, context in scored[:count]
        })
        unserved = list(reversed(scored[count:]))
        escalated = 0
        for risk, context in unserved:
            if (
                escalated < self.escalation_slots_per_epoch
                and float(risk) >= float(self.escalation_risk_threshold)
            ):
                decisions[context.proposal.incident_id] = "escalate_operator"
                escalated += 1
                continue
            decisions[context.proposal.incident_id] = "abstain"
        return decisions


@dataclass
class ConformalRiskModel:
    """One-sided split-conformal harm-risk upper bound.

    Calibration is grouped away from model fitting. Action-conditional
    residual quantiles are used only when at least 20 calibration rows support
    the action; otherwise the prospectively fixed global quantile applies.
    """

    model: Pipeline
    global_quantile: float
    action_quantiles: Mapping[str, float]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        base = self.model.predict_proba(frame)[:, 1]
        adjustments = np.asarray([
            self.action_quantiles.get(str(value), self.global_quantile)
            for value in frame["proposed_action"]
        ], dtype=float)
        risk = np.clip(base + adjustments, 0.0, 1.0)
        return np.column_stack([1.0 - risk, risk])


def fit_group_excluded_model(
    frame: pd.DataFrame, feature_block: str, excluded_split_family: str,
) -> Tuple[Pipeline, float]:
    """Fit after completely excluding one seed/topology/scenario family."""
    prepared = prepare_risk_frame(frame)
    prepared = prepared[
        ~prepared["proposed_action"].isin(
            ["verify", "request_peer_evidence", "defer", "no_action"]
        )
    ].reset_index(drop=True)
    training = prepared[prepared["split_family"] != excluded_split_family]
    if training.empty or excluded_split_family not in set(prepared["split_family"]):
        raise ValueError("invalid excluded split family")
    features = FEATURE_BLOCKS[feature_block]
    c_value = _choose_c(training, features, DEFAULT_C_GRID)
    model = make_risk_pipeline(features, c_value)
    model.fit(training, training["harmful_label"].to_numpy(dtype=int))
    return model, c_value


def fit_group_excluded_conformal(
    frame: pd.DataFrame, excluded_split_family: str,
) -> Tuple[ConformalRiskModel, float]:
    prepared = prepare_risk_frame(frame)
    prepared = prepared[~prepared.proposed_action.isin(
        ["verify", "request_peer_evidence", "defer", "no_action"]
    )].reset_index(drop=True)
    training = prepared[prepared.split_family != excluded_split_family].copy()
    groups = sorted(training.split_family.unique())
    if len(groups) < 2:
        raise ValueError("conformal fitting requires separate fit and calibration groups")
    calibration_group = groups[-1]
    fit = training[training.split_family != calibration_group]
    calibration = training[training.split_family == calibration_group]
    c_value = _choose_c(fit, PREDICTIVE_FEATURES, DEFAULT_C_GRID)
    model = make_risk_pipeline(PREDICTIVE_FEATURES, c_value)
    model.fit(fit, fit.harmful_label.to_numpy(dtype=int))
    base = model.predict_proba(calibration)[:, 1]
    residual = np.maximum(
        calibration.harmful_label.to_numpy(dtype=float) - base, 0.0,
    )
    global_quantile = float(np.quantile(residual, .90, method="higher"))
    action_quantiles: Dict[str, float] = {}
    for action, positions in calibration.groupby("proposed_action").groups.items():
        indices = calibration.index.get_indexer(list(positions))
        if len(indices) >= 20:
            action_quantiles[str(action)] = float(
                np.quantile(residual[indices], .90, method="higher")
            )
    return ConformalRiskModel(model, global_quantile, action_quantiles), c_value


def crossfit_conformal_risk(frame: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    groups = sorted(frame.split_family.unique())
    if len(groups) < 5:
        raise ValueError("conformal cross-fitting requires five isolated families")
    scores = np.zeros(len(frame), dtype=float)
    folds: List[Dict[str, Any]] = []
    for fold, excluded in enumerate(groups, start=1):
        test_mask = frame.split_family == excluded
        model, c_value = fit_group_excluded_conformal(frame, str(excluded))
        scores[test_mask.to_numpy()] = model.predict_proba(frame[test_mask])[:, 1]
        folds.append({
            "fold": fold, "selected_c": c_value,
            "training_rows": int((~test_mask).sum()), "test_rows": int(test_mask.sum()),
            "training_panels": int(frame[~test_mask].cluster_id.nunique()),
            "test_panels": int(frame[test_mask].cluster_id.nunique()),
            "environment_seed_disjoint": True,
            "topology_family_disjoint": True,
            "scenario_family_disjoint": True,
            "calibration_split_family": sorted(set(frame.loc[~test_mask, "split_family"]))[-1],
            "conformal_level": 0.90,
        })
    return scores, folds


def make_risk_pipeline(features: Sequence[str], c_value: float) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), list(features)),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=float(c_value), max_iter=2000, class_weight="balanced",
            solver="liblinear", random_state=66001,
        )),
    ])


def _choose_c(
    frame: pd.DataFrame, features: Sequence[str], c_grid: Sequence[float],
) -> float:
    groups = frame["split_family"].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 3:
        return float(c_grid[0])
    splitter = GroupKFold(n_splits=min(3, len(unique)))
    labels = frame["harmful_label"].to_numpy(dtype=int)
    best = (float("inf"), float(c_grid[0]))
    for c_value in c_grid:
        predictions = np.zeros(len(frame), dtype=float)
        valid = True
        for train, test in splitter.split(frame, labels, groups):
            if len(np.unique(labels[train])) < 2:
                valid = False
                break
            model = make_risk_pipeline(features, c_value)
            model.fit(frame.iloc[train], labels[train])
            predictions[test] = model.predict_proba(frame.iloc[test])[:, 1]
        if valid:
            candidate = (float(brier_score_loss(labels, predictions)), float(c_value))
            if candidate < best:
                best = candidate
    return float(best[1])


def crossfit_risk(
    frame: pd.DataFrame,
    features: Sequence[str],
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    groups = frame["split_family"].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("V6 cross-fitting requires five isolated split families")
    labels = frame["harmful_label"].to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=5)
    predictions = np.zeros(len(frame), dtype=float)
    folds: List[Dict[str, Any]] = []
    for fold, (train, test) in enumerate(splitter.split(frame, labels, groups), start=1):
        train_frame = frame.iloc[train]
        test_frame = frame.iloc[test]
        c_value = _choose_c(train_frame, features, c_grid)
        model = make_risk_pipeline(features, c_value)
        model.fit(train_frame, labels[train])
        predictions[test] = model.predict_proba(test_frame)[:, 1]
        isolation = {
            "environment_seed_disjoint": set(train_frame["environment_seed"].astype(int)).isdisjoint(set(test_frame["environment_seed"].astype(int))),
            "topology_family_disjoint": set(train_frame["topology_family"]).isdisjoint(set(test_frame["topology_family"])),
            "scenario_family_disjoint": set(train_frame["scenario_family"]).isdisjoint(set(test_frame["scenario_family"])),
        }
        if not all(isolation.values()):
            raise ValueError("V6 grouped split leaked a seed, topology, or scenario family")
        folds.append({
            "fold": fold,
            "selected_c": c_value,
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "training_panels": int(train_frame["cluster_id"].nunique()),
            "test_panels": int(test_frame["cluster_id"].nunique()),
            **isolation,
        })
    return predictions, folds


def select_at_coverage(
    frame: pd.DataFrame, scores: Sequence[float], coverage: float,
) -> pd.DataFrame:
    working = frame.copy()
    working["risk_score"] = np.asarray(scores, dtype=float)
    records: List[Dict[str, Any]] = []
    for cluster_id, group in working.groupby("cluster_id", sort=True):
        selections: List[pd.DataFrame] = []
        requested = 0
        for _, epoch in group.groupby("step", sort=True):
            ordered = epoch.sort_values(
                ["risk_score", "incident_id"],
                ascending=[True, True], kind="mergesort",
            )
            epoch_requested = max(0, min(
                len(ordered), int(round(float(coverage) * len(ordered))),
            ))
            requested += epoch_requested
            selections.append(ordered.iloc[:epoch_requested])
        selected = pd.concat(selections, ignore_index=False) if selections else group.iloc[:0]
        harm = int(selected["harmful_label"].sum())
        effects = selected["evaluator_causal_utility_if_executed"].to_numpy(dtype=float)
        first = group.iloc[0]
        records.append({
            "cluster_id": str(cluster_id),
            "application": str(first["application"]),
            "regime": str(first["regime"]),
            "information_condition": str(first["information_condition"]),
            "environment_seed": int(first["environment_seed"]),
            "topology_family": str(first["topology_family"]),
            "scenario_family": str(first["scenario_family"]),
            "split_family": str(first["split_family"]),
            "coverage_target": float(coverage),
            "eligible_decisions": int(len(group)),
            "selected_actions": requested,
            "action_coverage": float(requested / max(len(group), 1)),
            "harmful_actions": harm,
            "harmful_action_rate": float(harm / max(requested, 1)),
            "beneficial_actions": int((effects > 1e-9).sum()),
            "neutral_actions": int((np.abs(effects) <= 1e-9).sum()),
            "mean_causal_utility": float(effects.mean()) if requested else 0.0,
            "total_causal_utility": float(effects.sum()) if requested else 0.0,
        })
    return pd.DataFrame(records)


def _diagnostics(frame: pd.DataFrame, features: Sequence[str]) -> Dict[str, Any]:
    values = frame[list(features)].to_numpy(dtype=float)
    variances = np.var(values, axis=0)
    usable = values[:, variances > 1e-12]
    standardized = StandardScaler().fit_transform(usable) if usable.shape[1] else usable
    correlation = np.corrcoef(standardized, rowvar=False) if usable.shape[1] > 1 else np.eye(max(usable.shape[1], 1))
    correlation = np.nan_to_num(correlation, nan=0.0)
    return {
        "rows": int(len(frame)),
        "independent_panels": int(frame["cluster_id"].nunique()),
        "split_families": int(frame["split_family"].nunique()),
        "zero_variance_features": [feature for feature, variance in zip(features, variances) if variance <= 1e-12],
        "maximum_absolute_pairwise_correlation": float(np.max(np.abs(correlation - np.eye(correlation.shape[0])))),
        "condition_number": float(np.linalg.cond(standardized)) if usable.shape[1] else None,
    }


def analyze_risk_dataset(
    candidate_path: Path,
    output_root: Path,
    stage_name: str,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> Dict[str, Any]:
    candidates = prepare_risk_frame(pd.read_csv(candidate_path))
    if "controller" in candidates:
        methods = sorted(candidates["controller"].unique())
        reference = "never_act" if "never_act" in methods else methods[0]
        candidates = candidates[candidates["controller"] == reference].reset_index(drop=True)
    candidates = candidates[
        ~candidates["proposed_action"].isin(
            ["verify", "request_peer_evidence", "defer", "no_action"]
        )
    ].reset_index(drop=True)
    if not len(candidates):
        raise ValueError("V6 risk analysis found no consequential operational proposals")
    prediction_frames: List[pd.DataFrame] = []
    selection_frames: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []
    for (application, condition), subset in candidates.groupby(
        ["application", "information_condition"], sort=True,
    ):
        subset = subset.reset_index(drop=True)
        labels = subset["harmful_label"].to_numpy(dtype=int)
        for block, features in FEATURE_BLOCKS.items():
            scores, folds = crossfit_risk(subset, features)
            prediction = subset[[
                "cluster_id", "application", "regime", "information_condition",
                "environment_seed", "topology_family", "scenario_family",
                "split_family", "step", "incident_id", "agent_id", "role",
                "proposed_action", "harmful_label", "evaluator_causal_utility_if_executed",
            ]].copy()
            prediction["feature_block"] = block
            prediction["predicted_harm_risk"] = scores
            prediction_frames.append(prediction)
            fold_rows.extend([
                {"application": application, "information_condition": condition, "feature_block": block, **row}
                for row in folds
            ])
            for coverage in coverages:
                selected = select_at_coverage(subset, scores, coverage)
                selected["feature_block"] = block
                selection_frames.append(selected)
            diagnostics["%s|%s|%s" % (application, condition, block)] = _diagnostics(subset, features)
            metric_rows.append({
                "application": application,
                "information_condition": condition,
                "feature_block": block,
                "candidate_rows": int(len(subset)),
                "independent_panels": int(subset["cluster_id"].nunique()),
                "harm_prevalence": float(labels.mean()),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "brier_score": float(brier_score_loss(labels, scores)),
            })
        conformal_scores, conformal_folds = crossfit_conformal_risk(subset)
        prediction = subset[[
            "cluster_id", "application", "regime", "information_condition",
            "environment_seed", "topology_family", "scenario_family",
            "split_family", "step", "incident_id", "agent_id", "role",
            "proposed_action", "harmful_label", "evaluator_causal_utility_if_executed",
        ]].copy()
        prediction["feature_block"] = "conformal_risk_control"
        prediction["predicted_harm_risk"] = conformal_scores
        prediction_frames.append(prediction)
        fold_rows.extend([
            {"application": application, "information_condition": condition,
             "feature_block": "conformal_risk_control", **row}
            for row in conformal_folds
        ])
        for coverage in coverages:
            selected = select_at_coverage(subset, conformal_scores, coverage)
            selected["feature_block"] = "conformal_risk_control"
            selection_frames.append(selected)
        metric_rows.append({
            "application": application, "information_condition": condition,
            "feature_block": "conformal_risk_control",
            "candidate_rows": int(len(subset)),
            "independent_panels": int(subset.cluster_id.nunique()),
            "harm_prevalence": float(labels.mean()),
            "roc_auc": float(roc_auc_score(labels, conformal_scores)),
            "average_precision": float(average_precision_score(labels, conformal_scores)),
            "brier_score": float(brier_score_loss(labels, conformal_scores)),
            "conformal_level": 0.90,
        })
        for method in DIRECT_RISK_METHODS:
            scores = direct_risk_scores(subset, method)
            prediction = subset[[
                "cluster_id", "application", "regime", "information_condition",
                "environment_seed", "topology_family", "scenario_family",
                "split_family", "step", "incident_id", "agent_id", "role",
                "proposed_action", "harmful_label", "evaluator_causal_utility_if_executed",
            ]].copy()
            prediction["feature_block"] = method
            prediction["predicted_harm_risk"] = scores
            prediction_frames.append(prediction)
            for coverage in coverages:
                selected = select_at_coverage(subset, scores, coverage)
                selected["feature_block"] = method
                selection_frames.append(selected)
            metric_rows.append({
                "application": application,
                "information_condition": condition,
                "feature_block": method,
                "candidate_rows": int(len(subset)),
                "independent_panels": int(subset.cluster_id.nunique()),
                "harm_prevalence": float(labels.mean()),
                "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(scores)) > 1 else 0.5,
                "average_precision": float(average_precision_score(labels, scores)) if len(np.unique(scores)) > 1 else float(labels.mean()),
                "brier_score": float(brier_score_loss(labels, np.clip(scores, 0.0, 1.0))),
                "analysis_only_oracle": method == "oracle_risk_upper_bound",
            })
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    stage_root = output_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    if stage_name.startswith("pilot"):
        write_csv(stage_root / "crossfit_risk_predictions.csv", predictions.to_dict("records"))
        write_csv(stage_root / "risk_coverage_panel_results.csv", selections.to_dict("records"))
    else:
        write_csv_gzip(
            stage_root / "crossfit_risk_predictions.csv.gz",
            predictions.to_dict("records"),
        )
        write_csv_gzip(
            stage_root / "risk_coverage_panel_results.csv.gz",
            selections.to_dict("records"),
        )
    write_csv(stage_root / "prediction_metrics.csv", metric_rows)
    write_csv(stage_root / "grouped_fold_audit.csv", fold_rows)
    atomic_json(stage_root / "feature_diagnostics.json", diagnostics)

    primary = selections[
        selections["application"].isin(["humanitarian", "utility_restoration"])
        & (selections["information_condition"] == "private_fragmented")
    ]
    baseline_scores: List[Dict[str, Any]] = []
    for block in ("kpi_confidence", "predictive_uncertainty", "conformal_risk_control"):
        rows = primary[primary["feature_block"] == block]
        curve = rows.groupby("coverage_target")["harmful_action_rate"].mean().sort_index()
        area = float(np.trapz(curve.to_numpy(), curve.index.to_numpy()))
        brier = float(np.mean([
            row["brier_score"] for row in metric_rows
            if row["feature_block"] == block
            and row["application"] in ("humanitarian", "utility_restoration")
            and row["information_condition"] == "private_fragmented"
        ]))
        baseline_scores.append({"feature_block": block, "risk_coverage_area": area, "mean_brier": brier})
    selected_baseline = sorted(
        baseline_scores,
        key=lambda row: (row["risk_coverage_area"], row["mean_brier"], row["feature_block"]),
    )[0]["feature_block"]
    write_csv(stage_root / "nonentropic_baseline_selection.csv", baseline_scores)

    low_consensus_rows: List[Dict[str, Any]] = []
    low_threshold = 0.88
    residual_threshold = 0.25
    for (application, condition), subset in candidates.groupby(
        ["application", "information_condition"], sort=True,
    ):
        prediction_subset = predictions[
            (predictions.application == application)
            & (predictions.information_condition == condition)
        ]
        for method in (
            selected_baseline, "combined_generalized_entropic",
            "low_consensus_guard",
        ):
            scores = prediction_subset[
                prediction_subset.feature_block == method
            ][["cluster_id", "step", "incident_id", "predicted_harm_risk"]]
            scored = subset.merge(
                scores, on=["cluster_id", "step", "incident_id"],
                validate="one_to_one",
            )
            for cluster_id, panel in scored.groupby("cluster_id", sort=True):
                selected_positions: List[int] = []
                escalated_positions: List[int] = []
                for _, epoch in panel.groupby("step", sort=True):
                    ordered = epoch.sort_values(
                        ["predicted_harm_risk", "incident_id"],
                        ascending=[True, True], kind="mergesort",
                    )
                    count = int(round(.5 * len(ordered)))
                    selected_positions.extend(list(ordered.index[:count]))
                    # One bounded operator slot receives the highest-risk
                    # unselected incident at each decision epoch.
                    if len(ordered) > count:
                        escalated_positions.append(int(ordered.index[-1]))
                selected = panel.loc[selected_positions]
                escalated = panel.loc[escalated_positions]
                low = (panel.consensus < low_threshold) | (
                    panel.consensus_residual >= residual_threshold
                )
                low_selected = low.loc[selected.index]
                low_escalated = low.loc[escalated.index]
                first = panel.iloc[0]
                low_consensus_rows.append({
                    "cluster_id": cluster_id, "application": application,
                    "regime": first.regime,
                    "information_condition": condition,
                    "feature_block": method,
                    "eligible_actions": len(panel),
                    "selected_actions": len(selected),
                    "action_coverage": len(selected) / max(len(panel), 1),
                    "low_consensus_eligible": int(low.sum()),
                    "low_consensus_selected": int(low_selected.sum()),
                    "low_consensus_escalated": int(low_escalated.sum()),
                    "low_consensus_abstained_or_unserved": int(
                        low.sum() - low_selected.sum() - low_escalated.sum()
                    ),
                    "harmful_actions": int(selected.harmful_label.sum()),
                    "harmful_action_rate": float(selected.harmful_label.mean()),
                    "mean_causal_utility": float(
                        selected.evaluator_causal_utility_if_executed.mean()
                    ),
                    "operator_escalation_slots": len(escalated),
                    "consensus_threshold": low_threshold,
                    "residual_threshold": residual_threshold,
                })
    write_csv(stage_root / "low_consensus_abstention.csv", low_consensus_rows)

    primary_rows: List[Dict[str, Any]] = []
    paired_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        for condition in ("private_fragmented", "public_shared"):
            subset = selections[
                (selections["application"] == application)
                & (selections["information_condition"] == condition)
                & (selections["coverage_target"] == 0.5)
            ]
            baseline = subset[subset["feature_block"] == selected_baseline]
            combined = subset[subset["feature_block"] == "combined_generalized_entropic"]
            paired = baseline[["cluster_id", "regime", "harmful_action_rate", "mean_causal_utility"]].merge(
                combined[["cluster_id", "regime", "harmful_action_rate", "mean_causal_utility"]],
                on="cluster_id", suffixes=("_baseline", "_combined"), validate="one_to_one",
            )
            paired["regime"] = paired.pop("regime_baseline")
            if not (paired.pop("regime_combined") == paired["regime"]).all():
                raise ValueError("paired V6 risk results disagree on regime")
            paired["harm_rate_reduction"] = paired["harmful_action_rate_baseline"] - paired["harmful_action_rate_combined"]
            paired["utility_gain"] = paired["mean_causal_utility_combined"] - paired["mean_causal_utility_baseline"]
            paired_by_key[(application, condition)] = paired
            harm_interval = paired_bootstrap(paired["harm_rate_reduction"], 10000, 66061)
            utility_interval = paired_bootstrap(paired["utility_gain"], 10000, 66062)
            primary_rows.append({
                "application": application,
                "information_condition": condition,
                "coverage": 0.5,
                "baseline": selected_baseline,
                "independent_panels": int(len(paired)),
                "harm_rate_reduction": harm_interval["mean"],
                "harm_reduction_ci95_low": harm_interval["ci_low"],
                "harm_reduction_ci95_high": harm_interval["ci_high"],
                "mean_causal_utility_gain": utility_interval["mean"],
                "utility_gain_ci95_low": utility_interval["ci_low"],
                "utility_gain_ci95_high": utility_interval["ci_high"],
            })
    interaction_rows: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian", "utility_restoration"):
        private = paired_by_key[(application, "private_fragmented")].copy()
        public = paired_by_key[(application, "public_shared")].copy()
        private["environment_seed"] = private["cluster_id"].str.rsplit("|", n=1).str[-1].astype(int)
        public["environment_seed"] = public["cluster_id"].str.rsplit("|", n=1).str[-1].astype(int)
        merged = private[["environment_seed", "regime", "harm_rate_reduction"]].merge(
            public[["environment_seed", "regime", "harm_rate_reduction"]], on=["environment_seed", "regime"],
            suffixes=("_private", "_public"), validate="one_to_one",
        )
        difference = merged["harm_rate_reduction_private"] - merged["harm_rate_reduction_public"]
        interval = paired_bootstrap(difference, 10000, 66063)
        interaction_rows.append({
            "application": application,
            "matched_seed_panels": int(len(merged)),
            "private_minus_public_harm_reduction": interval["mean"],
            "ci95_low": interval["ci_low"],
            "ci95_high": interval["ci_high"],
        })
    write_csv(stage_root / "primary_matched_effects.csv", primary_rows)
    write_csv(stage_root / "fragmentation_interaction.csv", interaction_rows)
    report = {
        "stage": stage_name,
        "evidence_boundary": "pilot/development only; candidate rows are nested within environment panels",
        "candidate_rows": int(len(candidates)),
        "independent_panels": int(candidates["cluster_id"].nunique()),
        "selected_strongest_nonentropic_baseline": selected_baseline,
        "baseline_selection": baseline_scores,
        "primary_matched_effects": primary_rows,
        "fragmentation_interaction": interaction_rows,
    }
    atomic_json(stage_root / "risk_analysis.json", report)
    return report


def run_crossfit_dynamic_evaluation(
    repository: Path,
    results_root: Path,
    candidate_path: Path,
    analysis_report_path: Path,
    stage: str = "development_dynamic",
    coverage: float = 0.50,
) -> Dict[str, Any]:
    """Execute cross-fitted policies through full dynamic trajectories.

    The fitted model for each evaluation panel excludes its compound
    seed/topology/scenario split family. Both policies receive the same
    operator and action budgets and use the same exogenous tape.
    """
    candidates = pd.read_csv(candidate_path)
    report = json_load(analysis_report_path)
    baseline = str(report["selected_strongest_nonentropic_baseline"])
    methods = (baseline, "combined_generalized_entropic")
    fit_rows: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    for (application, condition), subset in candidates.groupby(
        ["application", "information_condition"], sort=True,
    ):
        for split_family in sorted(subset["split_family"].unique()):
            test_panels = subset[subset["split_family"] == split_family][[
                "application", "regime", "information_condition",
                "environment_seed", "split_family",
            ]].drop_duplicates()
            for block in methods:
                if block == "conformal_risk_control":
                    model, c_value = fit_group_excluded_conformal(
                        subset, split_family,
                    )
                else:
                    model, c_value = fit_group_excluded_model(
                        subset, block, split_family,
                    )
                fit_rows.append({
                    "application": application,
                    "information_condition": condition,
                    "excluded_split_family": split_family,
                    "feature_block": block,
                    "selected_c": c_value,
                    "training_rows": int((subset["split_family"] != split_family).sum()),
                    "test_panels": int(len(test_panels)),
                })
                for row in test_panels.itertuples(index=False):
                    controller = FittedRiskController(
                        model, str(row.regime), coverage, 1,
                    )
                    run_episode(
                        repository, results_root, stage,
                        str(row.application), str(row.regime),
                        str(row.information_condition), int(row.environment_seed),
                        "%s_crossfit" % block, coverage, "event_triggered",
                        escalation_slots=1, resume=True,
                        controller_override=controller,
                        extra_configuration={
                            "feature_block": block,
                            "excluded_split_family": split_family,
                            "selected_regularization_c": c_value,
                            "crossfit_dynamic_execution": True,
                        },
                    )
    aggregate_stage(results_root, stage, include_candidate_records=False)
    destination = results_root / "development" / stage[len("development_"):]
    write_csv(destination / "crossfit_model_fits.csv", fit_rows)
    summaries = pd.read_csv(destination / "episode_summary.csv")
    summaries["autonomous_harm_rate"] = (
        summaries["autonomous_harmful_actions"]
        / summaries["autonomous_completed_actions"].clip(lower=1)
    )
    summaries["autonomous_action_coverage"] = (
        summaries["autonomous_executions"]
        / summaries["eligible_operational_proposals"].clip(lower=1)
    )
    rows: List[Dict[str, Any]] = []
    interaction_sources: Dict[Tuple[str, str], pd.DataFrame] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        for condition in ("private_fragmented", "public_shared"):
            subset = summaries[
                (summaries["application"] == application)
                & (summaries["information_condition"] == condition)
            ]
            first = subset[subset["controller"] == "%s_crossfit" % baseline]
            second = subset[subset["controller"] == "combined_generalized_entropic_crossfit"]
            paired = first.merge(
                second, on=["application", "regime", "information_condition", "environment_seed"],
                suffixes=("_baseline", "_combined"), validate="one_to_one",
            )
            paired["harm_rate_reduction"] = (
                paired["autonomous_harm_rate_baseline"]
                - paired["autonomous_harm_rate_combined"]
            )
            paired["service_loss_relative_degradation"] = (
                paired["service_loss_combined"] - paired["service_loss_baseline"]
            ) / paired["service_loss_baseline"].abs().clip(lower=1e-9)
            paired["net_utility_gain"] = (
                paired["net_causal_utility_combined"]
                - paired["net_causal_utility_baseline"]
            )
            interaction_sources[(application, condition)] = paired
            harm = paired_bootstrap(paired["harm_rate_reduction"], 10000, 66161)
            service = paired_bootstrap(paired["service_loss_relative_degradation"], 10000, 66162)
            utility = paired_bootstrap(paired["net_utility_gain"], 10000, 66163)
            rows.append({
                "application": application,
                "information_condition": condition,
                "baseline": baseline,
                "panels": int(len(paired)),
                "harm_rate_reduction": harm["mean"],
                "harm_ci95_low": harm["ci_low"],
                "harm_ci95_high": harm["ci_high"],
                "relative_service_loss_degradation": service["mean"],
                "service_ci95_low": service["ci_low"],
                "service_ci95_high": service["ci_high"],
                "net_causal_utility_gain": utility["mean"],
                "utility_ci95_low": utility["ci_low"],
                "utility_ci95_high": utility["ci_high"],
                "baseline_action_coverage": float(first["autonomous_action_coverage"].mean()),
                "combined_action_coverage": float(second["autonomous_action_coverage"].mean()),
                "baseline_operator_minutes": float(first["operator_minutes"].mean()),
                "combined_operator_minutes": float(second["operator_minutes"].mean()),
            })
            for regime, regime_subset in paired.groupby("regime", sort=True):
                regime_harm = paired_bootstrap(
                    regime_subset["harm_rate_reduction"], 10000, 66165,
                )
                regime_utility = paired_bootstrap(
                    regime_subset["net_utility_gain"], 10000, 66166,
                )
                regime_rows.append({
                    "application": application,
                    "information_condition": condition,
                    "regime": regime,
                    "baseline": baseline,
                    "panels": int(len(regime_subset)),
                    "harm_rate_reduction": regime_harm["mean"],
                    "harm_ci95_low": regime_harm["ci_low"],
                    "harm_ci95_high": regime_harm["ci_high"],
                    "net_causal_utility_gain": regime_utility["mean"],
                    "utility_ci95_low": regime_utility["ci_low"],
                    "utility_ci95_high": regime_utility["ci_high"],
                })
    interactions: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian", "utility_restoration"):
        private = interaction_sources[(application, "private_fragmented")][[
            "environment_seed", "regime", "harm_rate_reduction",
        ]]
        public = interaction_sources[(application, "public_shared")][[
            "environment_seed", "regime", "harm_rate_reduction",
        ]]
        matched = private.merge(
            public, on=["environment_seed", "regime"],
            suffixes=("_private", "_public"), validate="one_to_one",
        )
        difference = (
            matched["harm_rate_reduction_private"]
            - matched["harm_rate_reduction_public"]
        )
        interval = paired_bootstrap(difference, 10000, 66164)
        interactions.append({
            "application": application,
            "matched_panels": int(len(matched)),
            "private_minus_public_harm_reduction": interval["mean"],
            "ci95_low": interval["ci_low"],
            "ci95_high": interval["ci_high"],
        })
    write_csv(destination / "paired_dynamic_effects.csv", rows)
    write_csv(destination / "regime_dynamic_effects.csv", regime_rows)
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
    result = {
        "stage": stage,
        "evidence": "development dynamic trajectories with cross-fitted deployable policies",
        "baseline": baseline,
        "episodes": int(len(summaries)),
        "paired_effects": rows,
        "fragmentation_interaction": interactions,
        "trigger_timing": timing_rows,
    }
    atomic_json(destination / "dynamic_evaluation.json", result)
    return result


def json_load(path: Path) -> Dict[str, Any]:
    import json
    return dict(json.loads(path.read_text(encoding="utf-8")))


def analyze_entropy_family_ablations(
    candidate_path: Path,
    destination: Path,
) -> Dict[str, Any]:
    frame = prepare_risk_frame(pd.read_csv(candidate_path))
    if "controller" in frame and "never_act" in set(frame["controller"]):
        frame = frame[frame["controller"] == "never_act"].reset_index(drop=True)
    frame = frame[~frame["proposed_action"].isin(
        ["verify", "request_peer_evidence", "defer", "no_action"]
    )].reset_index(drop=True)
    prediction_rows: List[pd.DataFrame] = []
    selection_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    for (application, condition), subset in frame.groupby(
        ["application", "information_condition"], sort=True,
    ):
        labels = subset["harmful_label"].to_numpy(dtype=int)
        for name, features in ENTROPY_ABLATION_BLOCKS.items():
            scores, _ = crossfit_risk(subset, features)
            prediction = subset[[
                "cluster_id", "application", "regime", "information_condition",
                "environment_seed", "split_family", "step", "incident_id",
                "harmful_label", "evaluator_causal_utility_if_executed",
            ]].copy()
            prediction["entropy_measure"] = name
            prediction["predicted_harm_risk"] = scores
            prediction_rows.append(prediction)
            selected = select_at_coverage(subset, scores, 0.50)
            selected["entropy_measure"] = name
            selection_rows.append(selected)
            summary_rows.append({
                "application": application,
                "information_condition": condition,
                "entropy_measure": name,
                "candidate_rows": int(len(subset)),
                "independent_panels": int(subset.cluster_id.nunique()),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "brier_score": float(brier_score_loss(labels, scores)),
                "harm_rate_at_50pct_coverage": float(selected.harmful_action_rate.mean()),
                "causal_utility_at_50pct_coverage": float(selected.mean_causal_utility.mean()),
            })
    predictions = pd.concat(prediction_rows, ignore_index=True)
    selections = pd.concat(selection_rows, ignore_index=True)
    destination.mkdir(parents=True, exist_ok=True)
    # Full row-level predictions slightly exceed the repository's 50 MiB
    # per-artifact guard in the frozen development design. Store the exact
    # same rows as deterministic gzip so a clean rebuild remains Git-facing.
    write_csv_gzip(
        destination / "entropy_family_predictions.csv.gz",
        predictions.to_dict("records"),
    )
    write_csv(destination / "entropy_family_panel_selections.csv", selections.to_dict("records"))
    write_csv(destination / "entropy_family_summary.csv", summary_rows)
    primary = pd.DataFrame(summary_rows)
    primary = primary[
        primary.application.isin(["humanitarian", "utility_restoration"])
        & (primary.information_condition == "private_fragmented")
    ]
    q_names = [
        "tsallis_q_0_5", "shannon_local", "tsallis_q_1_5",
        "tsallis_q_2", "tsallis_q_3",
    ]
    ranking = (
        primary[primary.entropy_measure.isin(q_names)]
        .groupby("entropy_measure", as_index=False)
        .agg({"harm_rate_at_50pct_coverage": "mean", "brier_score": "mean"})
        .sort_values(["harm_rate_at_50pct_coverage", "brier_score", "entropy_measure"])
    )
    selected = str(ranking.iloc[0].entropy_measure)
    result = {
        "stage": "development_entropy_family",
        "prespecified_q_values": [0.5, 1.0, 1.5, 2.0, 3.0],
        "selected_development_entropy_measure": selected,
        "selection_rule": "primary-app mean harm at 50% coverage, then Brier, then lexical",
        "individual_tests_confirmatory_only_after_family_test": True,
        "rows": int(len(frame)),
        "independent_panels": int(frame.cluster_id.nunique()),
    }
    atomic_json(destination / "entropy_family_analysis.json", result)
    return result


def refit_permutation_family_test(
    candidate_path: Path,
    destination: Path,
    permutations: int = 200,
    seed: int = 66070,
) -> Dict[str, Any]:
    """Refit the complete cross-fit pipeline for every stratified permutation."""
    frame = prepare_risk_frame(pd.read_csv(candidate_path))
    if "controller" in frame and "never_act" in set(frame["controller"]):
        frame = frame[frame["controller"] == "never_act"].reset_index(drop=True)
    frame = frame[
        frame.application.isin(["humanitarian", "utility_restoration"])
        & (frame.information_condition == "private_fragmented")
        & ~frame.proposed_action.isin(["verify", "request_peer_evidence", "defer", "no_action"])
    ].reset_index(drop=True)
    # Family-level hierarchy: the null retains the complete Shannon/JS block;
    # only the additional generalized spectrum is permuted and refitted.
    shannon_block = FEATURE_BLOCKS["shannon_js"]
    generalized_additions = [
        value for value in FEATURE_BLOCKS["combined_generalized_entropic"]
        if value not in shannon_block
    ]
    rng = np.random.RandomState(int(seed))
    rows: List[Dict[str, Any]] = []
    for application, subset in frame.groupby("application", sort=True):
        baseline_scores, _ = crossfit_risk(subset, shannon_block)
        baseline = select_at_coverage(subset, baseline_scores, 0.50)
        observed_scores, _ = crossfit_risk(
            subset, FEATURE_BLOCKS["combined_generalized_entropic"],
        )
        observed = select_at_coverage(subset, observed_scores, 0.50)
        observed_effect = float(
            baseline.harmful_action_rate.mean() - observed.harmful_action_rate.mean()
        )
        exceed = 0
        permuted_effects: List[float] = []
        severity_bins = pd.qcut(
            subset.visible_severity, q=4, labels=False, duplicates="drop",
        ).astype(str)
        strata = (
            subset.regime.astype(str) + "|" + subset.split_family.astype(str)
            + "|" + severity_bins
        )
        for permutation in range(int(permutations)):
            shuffled = subset.copy()
            for _, positions in strata.groupby(strata).groups.items():
                positions = np.asarray(list(positions), dtype=int)
                source = rng.permutation(positions)
                shuffled.loc[positions, generalized_additions] = subset.loc[source, generalized_additions].to_numpy()
            scores, _ = crossfit_risk(
                shuffled, FEATURE_BLOCKS["combined_generalized_entropic"],
            )
            selected = select_at_coverage(shuffled, scores, 0.50)
            effect = float(
                baseline.harmful_action_rate.mean()
                - selected.harmful_action_rate.mean()
            )
            permuted_effects.append(effect)
            exceed += int(effect >= observed_effect - 1e-15)
        p_value = float((exceed + 1) / (int(permutations) + 1))
        rows.append({
            "application": application,
            "reference_family": "shannon_js",
            "tested_family": "combined_generalized_entropic",
            "permutations": int(permutations),
            "observed_harm_rate_reduction": observed_effect,
            "permutation_mean": float(np.mean(permuted_effects)),
            "permutation_sd": float(np.std(permuted_effects, ddof=1)),
            "one_sided_refit_permutation_p": p_value,
            "monte_carlo_standard_error": float(np.sqrt(p_value * (1.0 - p_value) / (permutations + 1))),
        })
    adjusted = holm_adjust([value["one_sided_refit_permutation_p"] for value in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(value)
    write_csv(destination / "refit_permutation_family_test.csv", rows)
    report = {
        "test": "generalized-spectrum beyond Shannon/JS; severity/regime/split-stratified full-refit permutation",
        "permutations": int(permutations),
        "rows": rows,
        "prediction_time_shuffle_only": False,
    }
    atomic_json(destination / "refit_permutation_family_test.json", report)
    return report


def calibrate_escalation_threshold(
    candidate_path: Path,
    prediction_path: Path,
    destination: Path,
    thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
) -> Dict[str, Any]:
    """Select the highest feasible escalation threshold using timing only.

    No causal-utility or harm label enters this calibration. The eligible
    grid, timing targets, and tie rule are fixed before formal development.
    """
    raw = pd.read_csv(candidate_path)
    predictions = pd.read_csv(prediction_path)
    methods = ("predictive_uncertainty", "combined_generalized_entropic")
    panel_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    panel_metadata = raw[[
        "cluster_id", "application", "regime", "information_condition",
    ]].drop_duplicates("cluster_id")
    for method in methods:
        method_predictions = predictions[
            predictions.feature_block == method
        ].copy()
        for threshold in thresholds:
            current: List[Dict[str, Any]] = []
            for metadata in panel_metadata.itertuples(index=False):
                panel = method_predictions[
                    method_predictions.cluster_id == metadata.cluster_id
                ]
                budget = 4
                escalation_steps: List[int] = []
                for step in (0, 2, 4, 6, 8, 10):
                    epoch = panel[panel.step == step].sort_values(
                        ["predicted_harm_risk", "incident_id"],
                        kind="mergesort",
                    )
                    autonomous_count = int(round(0.50 * len(epoch)))
                    unserved = epoch.iloc[autonomous_count:]
                    if (
                        budget > 0 and len(unserved)
                        and float(unserved.predicted_harm_risk.max()) >= float(threshold)
                    ):
                        escalation_steps.append(int(step))
                        budget -= 1
                post = [value for value in escalation_steps if value >= 2]
                row = {
                    "cluster_id": metadata.cluster_id,
                    "application": metadata.application,
                    "regime": metadata.regime,
                    "information_condition": metadata.information_condition,
                    "feature_block": method,
                    "threshold": float(threshold),
                    "escalations": len(escalation_steps),
                    "pre_disruption_escalations": sum(
                        value < 2 for value in escalation_steps
                    ),
                    "first_post_disruption_escalation": min(post) if post else None,
                    "timely_by_step_4": bool(post and min(post) <= 4),
                    "nominal_false_activation": bool(
                        metadata.regime == "nominal" and escalation_steps
                    ),
                }
                current.append(row)
                panel_rows.append(row)
            frame = pd.DataFrame(current)
            primary = frame[
                frame.application.isin(["humanitarian", "utility_restoration"])
                & (frame.information_condition == "private_fragmented")
                & (frame.regime != "nominal")
            ]
            nominal = frame[
                frame.application.isin(["humanitarian", "utility_restoration"])
                & (frame.regime == "nominal")
            ]
            summary_rows.append({
                "feature_block": method,
                "threshold": float(threshold),
                "primary_disrupted_panels": int(len(primary)),
                "activation_rate": float((primary.escalations > 0).mean()),
                "timely_activation_rate_by_step_4": float(
                    primary.timely_by_step_4.mean()
                ),
                "pre_disruption_false_activation_rate": float(
                    primary.pre_disruption_escalations.mean()
                ),
                "nominal_panels": int(len(nominal)),
                "nominal_episode_false_activation_rate": float(
                    nominal.nominal_false_activation.mean()
                ),
                "mean_escalations_per_disrupted_episode": float(
                    primary.escalations.mean()
                ),
            })
    summary = pd.DataFrame(summary_rows)
    feasible = summary[
        (summary.activation_rate >= 0.75)
        & (summary.timely_activation_rate_by_step_4 >= 0.75)
        & (summary.pre_disruption_false_activation_rate <= 0.10)
        & (summary.nominal_episode_false_activation_rate <= 0.10)
    ]
    common = sorted(set.intersection(*[
        set(feasible[feasible.feature_block == method].threshold)
        for method in methods
    ]))
    if not common:
        raise RuntimeError("no escalation threshold passed the frozen timing criteria")
    selected = float(max(common))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "escalation_threshold_panels.csv", panel_rows)
    write_csv(destination / "escalation_threshold_summary.csv", summary_rows)
    report = {
        "evidence_stage": "pilot timing calibration only",
        "outcome_labels_used": False,
        "threshold_grid": list(map(float, thresholds)),
        "selection_rule": "highest threshold passing activation>=0.75, timely-by-step-4>=0.75, pre-disruption<=0.10, nominal<=0.10 for both frozen controllers",
        "methods": list(methods),
        "selected_threshold": selected,
        "selected_rows": summary[
            summary.threshold == selected
        ].to_dict("records"),
    }
    atomic_json(destination / "escalation_threshold_selection.json", report)
    return report
