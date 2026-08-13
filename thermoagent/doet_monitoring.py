"""Validate distributed operational entropy against conventional KPI detectors.

The frozen v1 main study is evaluated out of environment seed.  The already
seen v1 holdout is an external diagnostic set only.  One scripted-independent
trajectory per scenario is used so paired method copies are not treated as
independent monitoring observations.
"""

from __future__ import annotations

import argparse
import gzip
import io
import hashlib
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .doet_analysis import (
    SEVERE_SERVICE_LOSS_PERSISTENCE,
    SEVERE_SERVICE_LOSS_THRESHOLD,
    _severe_service_collapse_step,
)


KPI_FEATURES = [
    "backlog_ratio",
    "unmet_need_ratio",
    "service_loss",
    "maximum_impairment",
    "ordinary_messages_this_step",
    "kpi_rolling_mean",
    "kpi_rolling_variance",
    "kpi_ewma",
]
ENTROPY_FEATURES = [
    "distributed_entropy_mean",
    "entropy_absolute_deviation",
    "entropy_change",
    "entropy_standardized_absolute_residual",
    "consensus_rmse",
]
RESTRICTED_LOCAL_KPI_FEATURES = [
    "local_backlog_pressure",
    "local_inventory_capacity_ratio",
    "local_impairment",
    "local_delay",
    "local_service_shortfall",
    "local_commitment_strain",
    "local_communication_reliability",
]


def _analysis_environment() -> Dict[str, str]:
    packages = (
        "numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
    )
    values = {"python": platform.python_version()}
    for package in packages:
        try:
            values[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:
            values[package.replace("-", "_")] = "not-installed"
    return values


DIRECT_DETECTORS = {
    "backlog_threshold": "backlog_ratio",
    "service_level_threshold": "service_loss",
    "unmet_need_threshold": "unmet_need_ratio",
    "capacity_loss_indicator": "maximum_impairment",
    "communication_volume_threshold": "ordinary_messages_this_step",
    "rolling_mean": "kpi_rolling_mean",
    "rolling_variance": "kpi_rolling_variance",
    "EWMA": "kpi_ewma",
    "operational_entropy_high": "distributed_entropy_mean",
    "operational_entropy_low": "negative_distributed_entropy",
    "operational_entropy_absolute_deviation": "entropy_absolute_deviation",
    "operational_entropy_change": "absolute_entropy_change",
    "operational_entropy_standardized_residual": "entropy_standardized_absolute_residual",
    "exact_entropy_evaluator_only": "exact_entropy",
}
STATEFUL_DETECTORS = {
    "KPI_CUSUM": "kpi_cusum",
    "Page_Hinkley": "page_hinkley",
    "entropy_CUSUM": "entropy_cusum",
}
MODEL_DETECTORS = {
    "multivariate_KPI_logistic",
    "multivariate_KPI_plus_entropy_logistic",
}


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def communication_group(scenario: str) -> str:
    prefix = str(scenario).split("-", 1)[0]
    return {
        "reliable": "connected",
        "intermittent": "degraded",
        "partition": "partition",
    }.get(prefix, prefix)


def disruption_group(scenario_name: str) -> str:
    value = str(scenario_name).lower()
    if "nominal" in value:
        return "nominal"
    if "compound" in value:
        return "compound"
    if "correlated" in value:
        return "correlated"
    return "isolated"


def raw_episode_directory(results_root: Path, stage: str, run_id: str) -> Path:
    directory = results_root / "raw" / stage / run_id
    if not directory.is_dir():
        raise FileNotFoundError("missing frozen raw episode: %s" % directory)
    return directory


def impairment_by_step(event_path: Path) -> Dict[int, Dict[str, float]]:
    observations: Dict[int, Dict[str, float]] = {}
    with gzip.open(event_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("kind") != "observation_delivery":
                continue
            observation = event["payload"]["observation"]
            recipient = str(event["payload"]["recipient"])
            observations.setdefault(int(event["step"]), {})[recipient] = float(
                observation["impairment"]
            )
    return observations


def local_agent_records(event_path: Path) -> List[Dict[str, Any]]:
    """Recover private local KPIs and each agent's final-round gossip estimate."""

    observations: Dict[Tuple[int, str], Dict[str, Any]] = {}
    sketches: Dict[Tuple[int, str], Dict[str, Any]] = {}
    roles: Dict[str, str] = {}
    with gzip.open(event_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("kind") == "topology_snapshot":
                roles = {
                    str(agent): str(record["role"])
                    for agent, record in event["payload"]["agents"].items()
                }
            elif event.get("kind") == "observation_delivery":
                key = (int(event["step"]), str(event["payload"]["recipient"]))
                observations[key] = dict(event["payload"]["observation"])
            elif event.get("kind") == "macrostate_sketch":
                key = (int(event["step"]), str(event["actor"]))
                prior = sketches.get(key)
                if prior is None or int(event["payload"]["round"]) >= int(prior["round"]):
                    sketches[key] = dict(event["payload"])
    rows: List[Dict[str, Any]] = []
    for key, observation in sorted(observations.items()):
        sketch = sketches.get(key)
        if sketch is None:
            raise ValueError("missing distributed sketch for local observation %s" % (key,))
        distribution = np.clip(np.asarray(sketch["distribution"], dtype=float), 1e-12, None)
        distribution /= distribution.sum()
        entropy = float(-np.sum(distribution * np.log(distribution)) / np.log(len(distribution)))
        forecast = max(float(observation["local_forecast"]), 1.0)
        capacity = max(float(observation["capacity"]), 1.0)
        rows.append({
            "step": key[0],
            "agent_id": key[1],
            "role": roles.get(key[1], "unknown"),
            "local_backlog_pressure": min(float(observation["backlog"]) / forecast, 2.0) / 2.0,
            "local_inventory_capacity_ratio": min(float(observation["inventory"]) / capacity, 2.0) / 2.0,
            "local_impairment": float(observation["impairment"]),
            "local_delay": float(observation["delay"]),
            "local_service_shortfall": float(observation["service_shortfall"]),
            "local_commitment_strain": float(observation["commitment_strain"]),
            "local_communication_reliability": float(observation["communication_reliability"]),
            "local_distributed_entropy": entropy,
        })
    return rows


def prepare_frame(results_root: Path) -> pd.DataFrame:
    source = results_root / "processed" / "time_series.csv"
    frame = pd.read_csv(source)
    valid = frame.get("analysis_valid", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    frame = frame[
        frame["stage"].isin(["main", "holdout"])
        & (frame["method"] == "scripted_independent")
        & valid
        & (frame["completion_status"] == "complete")
    ].copy()
    if frame.empty:
        raise ValueError("no eligible frozen v1 monitoring trajectories")
    augmented: List[pd.DataFrame] = []
    for run_id, group in frame.groupby("run_id", sort=True):
        group = group.sort_values("step").copy()
        stage = str(group.iloc[0]["stage"])
        observations = impairment_by_step(
            raw_episode_directory(results_root, stage, str(run_id)) / "events.jsonl.gz"
        )
        group["maximum_impairment"] = [
            max(observations.get(int(step), {"none": 0.0}).values())
            for step in group["step"]
        ]
        group["mean_impairment"] = [
            float(np.mean(list(observations.get(int(step), {"none": 0.0}).values())))
            for step in group["step"]
        ]
        group["impairment_ranked_agents"] = [
            ";".join(
                agent
                for agent, _ in sorted(
                    observations.get(int(step), {}).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            for step in group["step"]
        ]
        demand = group["cumulative_demand"].astype(float).clip(lower=1e-9)
        group["backlog_ratio"] = group["backlog"].astype(float) / demand
        group["unmet_need_ratio"] = group["weighted_backlog"].astype(float) / demand
        messages = group["messages"].astype(float)
        group["ordinary_messages_this_step"] = messages.diff().fillna(messages).clip(lower=0.0)
        pressure_column = "backlog_ratio" if str(group.iloc[0]["application"]) == "commercial" else "unmet_need_ratio"
        group["kpi_pressure"] = np.maximum.reduce([
            group[pressure_column].astype(float).to_numpy(),
            group["service_loss"].astype(float).to_numpy(),
            group["maximum_impairment"].astype(float).to_numpy(),
        ])
        group["kpi_rolling_mean"] = group["kpi_pressure"].rolling(3, min_periods=1).mean()
        group["kpi_rolling_variance"] = group["kpi_pressure"].rolling(4, min_periods=2).var().fillna(0.0)
        group["kpi_ewma"] = group["kpi_pressure"].ewm(alpha=0.30, adjust=False).mean()
        entropy = group["distributed_entropy_mean"].astype(float)
        group["entropy_change"] = entropy.diff().fillna(0.0)
        group["absolute_entropy_change"] = group["entropy_change"].abs()
        group["negative_distributed_entropy"] = -entropy
        losses = group["service_loss"].astype(float).to_numpy()
        group["future_service_loss_3"] = [
            float(np.mean(losses[index + 1 : min(len(losses), index + 4)]))
            if index + 1 < len(losses)
            else float(losses[index])
            for index in range(len(losses))
        ]
        group["communication_group"] = communication_group(str(group.iloc[0]["scenario"]))
        group["disruption_group"] = disruption_group(str(group.iloc[0]["scenario_name"]))
        augmented.append(group)
    output = pd.concat(augmented, ignore_index=True)
    output["disruption_label"] = output["disruption_active"].astype(bool).astype(int)
    return output


def prepare_local_frame(results_root: Path, global_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the restricted-information agent/time-point diagnostic panel."""

    rows: List[Dict[str, Any]] = []
    for run_id, group in global_frame.groupby("run_id", sort=True):
        metadata = group.sort_values("step").set_index("step")
        first = group.iloc[0]
        event_path = raw_episode_directory(
            results_root, str(first["stage"]), str(run_id)
        ) / "events.jsonl.gz"
        records = local_agent_records(event_path)
        for record in records:
            step = int(record["step"])
            global_row = metadata.loc[step]
            record.update({
                "run_id": run_id,
                "stage": str(first["stage"]),
                "application": str(first["application"]),
                "scenario_name": str(first["scenario_name"]),
                "seed": int(first["seed"]),
                "communication_group": str(first["communication_group"]),
                "disruption_group": str(first["disruption_group"]),
                "disruption_label": int(global_row["disruption_label"]),
                "future_service_loss_3": float(global_row["future_service_loss_3"]),
            })
            rows.append(record)
    output = pd.DataFrame(rows).sort_values(["run_id", "agent_id", "step"])
    output["local_entropy_change"] = (
        output.groupby(["run_id", "agent_id"])["local_distributed_entropy"].diff().fillna(0.0)
    )
    return output.reset_index(drop=True)


def calibration_parameters(training: pd.DataFrame) -> Dict[str, float]:
    nominal = training[training["disruption_group"] == "nominal"]
    if nominal.empty:
        nominal = training[training["disruption_label"] == 0]
    entropy = nominal["distributed_entropy_mean"].astype(float)
    kpi = nominal["kpi_pressure"].astype(float)
    return {
        "entropy_center": float(entropy.mean()),
        "entropy_median": float(entropy.median()),
        "entropy_scale": max(float(entropy.std(ddof=0)), 1e-6),
        "kpi_center": float(kpi.mean()),
        "kpi_scale": max(float(kpi.std(ddof=0)), 1e-6),
    }


def stateful(values: np.ndarray, kind: str) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    if kind == "cusum":
        level = 0.0
        for index, value in enumerate(values):
            level = max(0.0, 0.95 * level + float(value) - 0.50)
            output[index] = level
        return output
    if kind == "page_hinkley":
        running_mean = 0.0
        cumulative = 0.0
        minimum = 0.0
        for index, value in enumerate(values, start=1):
            running_mean += (float(value) - running_mean) / index
            cumulative += float(value) - running_mean - 0.05
            minimum = min(minimum, cumulative)
            output[index - 1] = cumulative - minimum
        return output
    raise ValueError("unknown stateful detector")


def detector_columns(frame: pd.DataFrame, parameters: Mapping[str, float]) -> pd.DataFrame:
    output: List[pd.DataFrame] = []
    for _, group in frame.groupby("run_id", sort=False):
        group = group.sort_values("step").copy()
        entropy = group["distributed_entropy_mean"].astype(float)
        entropy_z = (entropy - parameters["entropy_center"]) / parameters["entropy_scale"]
        group["entropy_absolute_deviation"] = (entropy - parameters["entropy_median"]).abs()
        group["entropy_standardized_absolute_residual"] = entropy_z.abs()
        kpi_z = (group["kpi_pressure"].astype(float) - parameters["kpi_center"]) / parameters["kpi_scale"]
        group["kpi_cusum"] = stateful(kpi_z.to_numpy(), "cusum")
        group["page_hinkley"] = stateful(kpi_z.to_numpy(), "page_hinkley")
        group["entropy_cusum"] = stateful(entropy_z.abs().to_numpy(), "cusum")
        output.append(group)
    return pd.concat(output, ignore_index=True)


def numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    matrix = frame[list(columns)].astype(float).replace([np.inf, -np.inf], np.nan)
    return matrix.fillna(0.0).to_numpy()


def score_split(training: pd.DataFrame, evaluation: pd.DataFrame, split: str) -> pd.DataFrame:
    parameters = calibration_parameters(training)
    training = detector_columns(training, parameters)
    evaluation = detector_columns(evaluation, parameters)
    nominal = training[training["disruption_group"] == "nominal"]
    if nominal.empty:
        nominal = training[training["disruption_label"] == 0]
    records: List[pd.DataFrame] = []
    for detector, column in {**DIRECT_DETECTORS, **STATEFUL_DETECTORS}.items():
        threshold = float(nominal[column].astype(float).quantile(0.95))
        piece = evaluation[[
            "run_id", "stage", "application", "scenario", "scenario_name",
            "seed", "step", "communication_group", "disruption_group",
            "disruption_label", "service_loss", "future_service_loss_3",
            "disruption_sources", "surprisal_ranked_agents",
            "impairment_ranked_agents",
        ]].copy()
        piece["detector"] = detector
        piece["score"] = evaluation[column].astype(float).to_numpy()
        piece["threshold"] = threshold
        piece["activated"] = piece["score"] > threshold
        piece["split"] = split
        records.append(piece)

    labels = training["disruption_label"].astype(int).to_numpy()
    for detector, features in (
        ("multivariate_KPI_logistic", KPI_FEATURES),
        ("multivariate_KPI_plus_entropy_logistic", KPI_FEATURES + ENTROPY_FEATURES),
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=20260813,
                solver="liblinear",
            ),
        )
        model.fit(numeric_matrix(training, features), labels)
        train_scores = model.predict_proba(numeric_matrix(training, features))[:, 1]
        nominal_scores = train_scores[nominal.index.to_numpy()]
        threshold = float(np.quantile(nominal_scores, 0.95))
        scores = model.predict_proba(numeric_matrix(evaluation, features))[:, 1]
        piece = evaluation[[
            "run_id", "stage", "application", "scenario", "scenario_name",
            "seed", "step", "communication_group", "disruption_group",
            "disruption_label", "service_loss", "future_service_loss_3",
            "disruption_sources", "surprisal_ranked_agents",
            "impairment_ranked_agents",
        ]].copy()
        piece["detector"] = detector
        piece["score"] = scores
        piece["threshold"] = threshold
        piece["activated"] = scores > threshold
        piece["split"] = split
        records.append(piece)
    return pd.concat(records, ignore_index=True)


def cross_validated_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output: List[pd.DataFrame] = []
    for application in ("commercial", "humanitarian"):
        main = frame[(frame["stage"] == "main") & (frame["application"] == application)]
        holdout = frame[(frame["stage"] == "holdout") & (frame["application"] == application)]
        for seed in sorted(main["seed"].unique()):
            training = main[main["seed"] != seed].copy()
            evaluation = main[main["seed"] == seed].copy()
            output.append(score_split(training, evaluation, "main_oof_seed_%s" % seed))
        output.append(score_split(main.copy(), holdout.copy(), "original_holdout_external_diagnostic"))
    scores = pd.concat(output, ignore_index=True)
    scores["evaluation_stage"] = np.where(
        scores["stage"] == "main", "original_main_oof", "original_holdout_diagnostic"
    )
    return scores


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores)) if labels.sum() else float("nan")


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def metric_row(group: pd.DataFrame, stratum_type: str, stratum_value: str) -> Dict[str, Any]:
    labels = group["disruption_label"].astype(int).to_numpy()
    scores = group["score"].astype(float).to_numpy()
    predictions = group["activated"].astype(bool).to_numpy()
    prevalence = float(labels.mean()) if len(labels) else float("nan")
    ap = safe_ap(labels, scores)
    true_positive = int(np.sum(predictions & (labels == 1)))
    false_positive = int(np.sum(predictions & (labels == 0)))
    return {
        "evaluation_stage": str(group.iloc[0]["evaluation_stage"]),
        "application": str(group.iloc[0]["application"]),
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "detector": str(group.iloc[0]["detector"]),
        "n_episodes": int(group["run_id"].nunique()),
        "n_timepoints": len(group),
        "positive_timepoints": int(labels.sum()),
        "positive_class_prevalence": prevalence,
        "average_precision": ap,
        "average_precision_lift_over_prevalence": ap / prevalence if prevalence > 0 and np.isfinite(ap) else float("nan"),
        "roc_auc": safe_auc(labels, scores),
        "precision_at_nominal_95pct_threshold": true_positive / max(true_positive + false_positive, 1),
        "recall_at_nominal_95pct_threshold": true_positive / max(int(labels.sum()), 1),
        "false_alarm_rate": float(np.mean(predictions[labels == 0])) if np.any(labels == 0) else float("nan"),
        "mean_training_derived_threshold": float(group["threshold"].mean()),
    }


def monitoring_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (_, _, detector), group in scores.groupby(["evaluation_stage", "application", "detector"]):
        rows.append(metric_row(group, "overall", "all"))
        for value, subset in group.groupby("communication_group"):
            rows.append(metric_row(subset, "communication", str(value)))
        for value, subset in group.groupby("disruption_group"):
            rows.append(metric_row(subset, "disruption", str(value)))
        for (communication, disruption), subset in group.groupby(["communication_group", "disruption_group"]):
            rows.append(metric_row(subset, "communication_x_disruption", "%s:%s" % (communication, disruption)))
    return pd.DataFrame(rows).sort_values(
        ["evaluation_stage", "application", "stratum_type", "stratum_value", "detector"]
    )


def detection_lead_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (stage, application, detector, run_id), group in scores.groupby(
        ["evaluation_stage", "application", "detector", "run_id"]
    ):
        group = group.sort_values("step")
        active = group[group["disruption_label"] == 1]
        onset = int(active["step"].min()) if not active.empty else None
        pre = group if onset is None else group[group["step"] < onset]
        collapse_step = _severe_service_collapse_step(group, onset)
        post_detection = (
            group[(group["step"] >= onset) & group["activated"].astype(bool)]
            if onset is not None
            else group.iloc[0:0]
        )
        detection = int(post_detection["step"].min()) if not post_detection.empty else None
        false_activations = int(pre["activated"].astype(bool).sum())
        lead = collapse_step - detection if collapse_step is not None and detection is not None else None
        rows.append({
            "row_type": "episode",
            "evaluation_stage": stage,
            "application": application,
            "detector": detector,
            "run_id": run_id,
            "scenario_name": str(group.iloc[0]["scenario_name"]),
            "communication_group": str(group.iloc[0]["communication_group"]),
            "disruption_group": str(group.iloc[0]["disruption_group"]),
            "disruption_step": onset,
            "detection_step": detection,
            "detection_delay": detection - onset if detection is not None and onset is not None else None,
            "visible_collapse_step": collapse_step,
            "severe_service_loss_threshold": SEVERE_SERVICE_LOSS_THRESHOLD,
            "severe_service_loss_persistence": SEVERE_SERVICE_LOSS_PERSISTENCE,
            "detection_lead_before_collapse": lead,
            "detected_strictly_before_collapse": bool(
                lead is not None and lead > 0
            ),
            "false_activations_before_disruption_or_during_nominal": false_activations,
            "pre_disruption_or_nominal_timepoints": len(pre),
        })
    episode = pd.DataFrame(rows)
    summaries: List[Dict[str, Any]] = []
    for (stage, application, detector), group in episode.groupby(
        ["evaluation_stage", "application", "detector"]
    ):
        disrupted = group[group["disruption_step"].notna()]
        detected = disrupted["detection_step"].notna()
        visible = disrupted["visible_collapse_step"].notna()
        before = disrupted.loc[
            visible, "detected_strictly_before_collapse"
        ].astype(bool)
        summaries.append({
            "row_type": "summary",
            "evaluation_stage": stage,
            "application": application,
            "detector": detector,
            "run_id": "__summary__",
            "scenario_name": "all",
            "communication_group": "all",
            "disruption_group": "all",
            "disruption_step": None,
            "detection_step": None,
            "detection_delay": float(pd.to_numeric(disrupted.loc[detected, "detection_delay"]).mean()) if detected.any() else None,
            "visible_collapse_step": None,
            "severe_service_loss_threshold": SEVERE_SERVICE_LOSS_THRESHOLD,
            "severe_service_loss_persistence": SEVERE_SERVICE_LOSS_PERSISTENCE,
            "detection_lead_before_collapse": float(pd.to_numeric(disrupted.loc[detected & visible, "detection_lead_before_collapse"]).mean()) if (detected & visible).any() else None,
            "detected_strictly_before_collapse": float(before.mean()) if len(before) else None,
            "false_activations_before_disruption_or_during_nominal": int(group["false_activations_before_disruption_or_during_nominal"].sum()),
            "pre_disruption_or_nominal_timepoints": int(group["pre_disruption_or_nominal_timepoints"].sum()),
            "n_episodes": len(group),
            "n_disrupted_episodes": len(disrupted),
            "detection_rate": float(detected.mean()) if len(disrupted) else None,
            "false_activation_rate": float(group["false_activations_before_disruption_or_during_nominal"].sum() / max(group["pre_disruption_or_nominal_timepoints"].sum(), 1)),
        })
    return pd.concat([episode, pd.DataFrame(summaries)], ignore_index=True)


def localization_table(scores: pd.DataFrame) -> pd.DataFrame:
    # Detector scores repeat the source fields. Keep one row per original episode.
    base = scores[scores["detector"] == "operational_entropy_absolute_deviation"].copy()
    rows: List[Dict[str, Any]] = []
    for (stage, application, run_id), group in base.groupby(["evaluation_stage", "application", "run_id"]):
        active = group[group["disruption_label"] == 1].sort_values("step")
        sources = {item for item in str(group.iloc[0]["disruption_sources"]).split(";") if item and item != "nan"}
        if active.empty or not sources:
            continue
        onset = active.iloc[0]
        for signal, column in (
            ("local_surprisal", "surprisal_ranked_agents"),
            ("ordinary_impairment", "impairment_ranked_agents"),
        ):
            ranking = [item for item in str(onset[column]).split(";") if item]
            rows.append({
                "row_type": "episode",
                "evaluation_stage": stage,
                "application": application,
                "run_id": run_id,
                "scenario_name": str(onset["scenario_name"]),
                "communication_group": str(onset["communication_group"]),
                "disruption_group": str(onset["disruption_group"]),
                "signal": signal,
                "true_sources": ";".join(sorted(sources)),
                "top1_prediction": ranking[0] if ranking else None,
                "top1_correct": bool(ranking and ranking[0] in sources),
                "top3_correct": bool(set(ranking[:3]) & sources),
            })
    episode = pd.DataFrame(rows)
    summaries: List[Dict[str, Any]] = []
    for keys, group in episode.groupby(["evaluation_stage", "application", "signal"]):
        summaries.append({
            "row_type": "summary",
            "evaluation_stage": keys[0],
            "application": keys[1],
            "run_id": "__summary__",
            "scenario_name": "all",
            "communication_group": "all",
            "disruption_group": "all",
            "signal": keys[2],
            "true_sources": "multiple",
            "top1_prediction": None,
            "top1_correct": float(group["top1_correct"].mean()),
            "top3_correct": float(group["top3_correct"].mean()),
            "n_episodes": len(group),
        })
    return pd.concat([episode, pd.DataFrame(summaries)], ignore_index=True)


def regression_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for application in ("commercial", "humanitarian"):
        main = frame[(frame["stage"] == "main") & (frame["application"] == application)].copy()
        holdout = frame[(frame["stage"] == "holdout") & (frame["application"] == application)].copy()
        splits = []
        for seed in sorted(main["seed"].unique()):
            splits.append((main[main["seed"] != seed], main[main["seed"] == seed], "original_main_oof"))
        splits.append((main, holdout, "original_holdout_diagnostic"))
        for raw_training, raw_evaluation, stage in splits:
            parameters = calibration_parameters(raw_training)
            training = detector_columns(raw_training, parameters)
            evaluation = detector_columns(raw_evaluation, parameters)
            for model_name, features in (
                ("ordinary_KPI_regression", KPI_FEATURES),
                ("ordinary_KPI_plus_entropy_regression", KPI_FEATURES + ENTROPY_FEATURES),
            ):
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(numeric_matrix(training, features), training["future_service_loss_3"].astype(float))
                prediction = model.predict(numeric_matrix(evaluation, features))
                piece = evaluation[["run_id", "seed", "step", "application"]].copy()
                piece["evaluation_stage"] = stage
                piece["model"] = model_name
                piece["observed_future_service_loss_3"] = evaluation["future_service_loss_3"].astype(float).to_numpy()
                piece["predicted_future_service_loss_3"] = prediction
                rows.append(piece)
    return pd.concat(rows, ignore_index=True)


def restricted_information_table(local: pd.DataFrame) -> pd.DataFrame:
    """Test entropy as a compressed shared statistic when only local KPIs exist."""

    classification_records: List[pd.DataFrame] = []
    regression_records: List[pd.DataFrame] = []
    for application in ("commercial", "humanitarian"):
        main = local[(local["stage"] == "main") & (local["application"] == application)].copy()
        holdout = local[(local["stage"] == "holdout") & (local["application"] == application)].copy()
        splits = []
        for seed in sorted(main["seed"].unique()):
            splits.append((main[main["seed"] != seed], main[main["seed"] == seed], "original_main_oof"))
        splits.append((main, holdout, "original_holdout_diagnostic"))
        for raw_training, raw_evaluation, stage in splits:
            nominal = raw_training[raw_training["disruption_group"] == "nominal"]
            if nominal.empty:
                nominal = raw_training[raw_training["disruption_label"] == 0]
            center = float(nominal["local_distributed_entropy"].mean())
            scale = max(float(nominal["local_distributed_entropy"].std(ddof=0)), 1e-6)
            training = raw_training.copy()
            evaluation = raw_evaluation.copy()
            for subset in (training, evaluation):
                subset["local_entropy_absolute_residual"] = (
                    (subset["local_distributed_entropy"].astype(float) - center) / scale
                ).abs()
                subset["absolute_local_entropy_change"] = subset["local_entropy_change"].astype(float).abs()
            entropy_features = [
                "local_distributed_entropy",
                "local_entropy_absolute_residual",
                "absolute_local_entropy_change",
            ]
            labels = training["disruption_label"].astype(int).to_numpy()
            for model_name, features in (
                ("restricted_local_KPI_logistic", RESTRICTED_LOCAL_KPI_FEATURES),
                (
                    "restricted_local_KPI_plus_distributed_entropy_logistic",
                    RESTRICTED_LOCAL_KPI_FEATURES + entropy_features,
                ),
            ):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=20260813,
                        solver="liblinear",
                    ),
                )
                model.fit(numeric_matrix(training, features), labels)
                scores = model.predict_proba(numeric_matrix(evaluation, features))[:, 1]
                piece = evaluation[["run_id", "agent_id", "seed", "step", "application"]].copy()
                piece["evaluation_stage"] = stage
                piece["model"] = model_name
                piece["label"] = evaluation["disruption_label"].astype(int).to_numpy()
                piece["score"] = scores
                classification_records.append(piece)
            for model_name, features in (
                ("restricted_local_KPI_regression", RESTRICTED_LOCAL_KPI_FEATURES),
                (
                    "restricted_local_KPI_plus_distributed_entropy_regression",
                    RESTRICTED_LOCAL_KPI_FEATURES + entropy_features,
                ),
            ):
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(
                    numeric_matrix(training, features),
                    training["future_service_loss_3"].astype(float),
                )
                prediction = model.predict(numeric_matrix(evaluation, features))
                piece = evaluation[["run_id", "agent_id", "seed", "step", "application"]].copy()
                piece["evaluation_stage"] = stage
                piece["model"] = model_name
                piece["observed"] = evaluation["future_service_loss_3"].astype(float).to_numpy()
                piece["predicted"] = prediction
                regression_records.append(piece)

    classification = pd.concat(classification_records, ignore_index=True)
    regression = pd.concat(regression_records, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for (stage, application, model), group in classification.groupby(["evaluation_stage", "application", "model"]):
        labels = group["label"].astype(int).to_numpy()
        values = group["score"].astype(float).to_numpy()
        rows.append({
            "analysis": "restricted_local_disruption_classification",
            "evaluation_stage": stage,
            "application": application,
            "model": model,
            "n_episodes": group["run_id"].nunique(),
            "n_timepoints": len(group),
            "average_precision": safe_ap(labels, values),
            "roc_auc": safe_auc(labels, values),
            "brier_score": float(brier_score_loss(labels, values)),
            "rmse": None,
            "r_squared": None,
            "spearman": None,
        })
    for (stage, application, model), group in regression.groupby(["evaluation_stage", "application", "model"]):
        observed = group["observed"].astype(float).to_numpy()
        predicted = group["predicted"].astype(float).to_numpy()
        residual = observed - predicted
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        correlation = stats.spearmanr(observed, predicted).correlation
        rows.append({
            "analysis": "restricted_local_future_service_loss_regression",
            "evaluation_stage": stage,
            "application": application,
            "model": model,
            "n_episodes": group["run_id"].nunique(),
            "n_timepoints": len(group),
            "average_precision": None,
            "roc_auc": None,
            "brier_score": None,
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r_squared": float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else None,
            "spearman": float(correlation) if np.isfinite(correlation) else None,
        })
    summary = pd.DataFrame(rows)
    differences: List[Dict[str, Any]] = []
    for (analysis, stage, application), group in summary.groupby(["analysis", "evaluation_stage", "application"]):
        ordinary = group[~group["model"].str.contains("plus_distributed_entropy")].iloc[0]
        entropy = group[group["model"].str.contains("plus_distributed_entropy")].iloc[0]
        classification_analysis = "classification" in analysis
        differences.append({
            "analysis": analysis + "_incremental_entropy",
            "evaluation_stage": stage,
            "application": application,
            "model": "restricted_local_KPI_plus_entropy_minus_local_KPI",
            "n_episodes": int(entropy["n_episodes"]),
            "n_timepoints": int(entropy["n_timepoints"]),
            "average_precision": float(entropy["average_precision"] - ordinary["average_precision"]) if classification_analysis else None,
            "roc_auc": float(entropy["roc_auc"] - ordinary["roc_auc"]) if classification_analysis else None,
            "brier_score": float(entropy["brier_score"] - ordinary["brier_score"]) if classification_analysis else None,
            "rmse": float(entropy["rmse"] - ordinary["rmse"]) if not classification_analysis else None,
            "r_squared": float(entropy["r_squared"] - ordinary["r_squared"]) if not classification_analysis else None,
            "spearman": float(entropy["spearman"] - ordinary["spearman"]) if not classification_analysis else None,
        })
    return pd.concat([summary, pd.DataFrame(differences)], ignore_index=True)


def incremental_value_table(scores: pd.DataFrame, regression: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    detection = scores[scores["detector"].isin(MODEL_DETECTORS)]
    for (stage, application, detector), group in detection.groupby(["evaluation_stage", "application", "detector"]):
        labels = group["disruption_label"].astype(int).to_numpy()
        values = group["score"].astype(float).to_numpy()
        rows.append({
            "analysis": "disruption_classification",
            "evaluation_stage": stage,
            "application": application,
            "model": detector,
            "n_episodes": group["run_id"].nunique(),
            "n_timepoints": len(group),
            "average_precision": safe_ap(labels, values),
            "roc_auc": safe_auc(labels, values),
            "brier_score": float(brier_score_loss(labels, values)),
            "rmse": None,
            "r_squared": None,
            "spearman": None,
        })
    for (stage, application, model), group in regression.groupby(["evaluation_stage", "application", "model"]):
        observed = group["observed_future_service_loss_3"].astype(float).to_numpy()
        predicted = group["predicted_future_service_loss_3"].astype(float).to_numpy()
        residual = observed - predicted
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        correlation = stats.spearmanr(observed, predicted).correlation
        rows.append({
            "analysis": "future_service_loss_regression",
            "evaluation_stage": stage,
            "application": application,
            "model": model,
            "n_episodes": group["run_id"].nunique(),
            "n_timepoints": len(group),
            "average_precision": None,
            "roc_auc": None,
            "brier_score": None,
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r_squared": float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else None,
            "spearman": float(correlation) if np.isfinite(correlation) else None,
        })
        try:
            bins = pd.qcut(predicted, 5, labels=False, duplicates="drop")
        except ValueError:
            bins = np.zeros(len(predicted), dtype=int)
        for bin_index in sorted(pd.Series(bins).dropna().unique()):
            selected = np.asarray(bins) == bin_index
            rows.append({
                "analysis": "future_service_loss_calibration_bin",
                "evaluation_stage": stage,
                "application": application,
                "model": model,
                "n_episodes": group.loc[selected, "run_id"].nunique(),
                "n_timepoints": int(selected.sum()),
                "average_precision": None,
                "roc_auc": None,
                "brier_score": None,
                "rmse": float(np.sqrt(np.mean(residual[selected] ** 2))),
                "r_squared": None,
                "spearman": None,
                "calibration_bin": int(bin_index),
                "mean_prediction": float(predicted[selected].mean()),
                "mean_observed": float(observed[selected].mean()),
            })
    output = pd.DataFrame(rows)
    summary = output[output["analysis"].isin(["disruption_classification", "future_service_loss_regression"])].copy()
    difference_rows: List[Dict[str, Any]] = []
    for (analysis, stage, application), group in summary.groupby(["analysis", "evaluation_stage", "application"]):
        if len(group) != 2:
            continue
        ordinary = group[~group["model"].str.contains("plus_entropy")].iloc[0]
        entropy = group[group["model"].str.contains("plus_entropy")].iloc[0]
        difference_rows.append({
            "analysis": analysis + "_incremental_entropy",
            "evaluation_stage": stage,
            "application": application,
            "model": "KPI_plus_entropy_minus_KPI",
            "n_episodes": int(entropy["n_episodes"]),
            "n_timepoints": int(entropy["n_timepoints"]),
            "average_precision": float(entropy["average_precision"] - ordinary["average_precision"]) if analysis == "disruption_classification" else None,
            "roc_auc": float(entropy["roc_auc"] - ordinary["roc_auc"]) if analysis == "disruption_classification" else None,
            "brier_score": float(entropy["brier_score"] - ordinary["brier_score"]) if analysis == "disruption_classification" else None,
            "rmse": float(entropy["rmse"] - ordinary["rmse"]) if analysis == "future_service_loss_regression" else None,
            "r_squared": float(entropy["r_squared"] - ordinary["r_squared"]) if analysis == "future_service_loss_regression" else None,
            "spearman": float(entropy["spearman"] - ordinary["spearman"]) if analysis == "future_service_loss_regression" else None,
        })
    return pd.concat([output, pd.DataFrame(difference_rows)], ignore_index=True)


def write_figures(monitoring: pd.DataFrame, incremental: pd.DataFrame, output_root: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    pdf_dir = output_root / "figures" / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        "operational_entropy_absolute_deviation",
        "operational_entropy_high",
        "backlog_threshold",
        "service_level_threshold",
        "capacity_loss_indicator",
        "KPI_CUSUM",
        "Page_Hinkley",
        "multivariate_KPI_logistic",
        "multivariate_KPI_plus_entropy_logistic",
    ]
    labels = {
        "operational_entropy_absolute_deviation": "Entropy |deviation|",
        "operational_entropy_high": "Entropy high",
        "backlog_threshold": "Backlog",
        "service_level_threshold": "Service loss",
        "capacity_loss_indicator": "Capacity loss",
        "KPI_CUSUM": "KPI CUSUM",
        "Page_Hinkley": "Page–Hinkley",
        "multivariate_KPI_logistic": "KPI logistic",
        "multivariate_KPI_plus_entropy_logistic": "KPI + entropy",
    }
    colors = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#009E73", "#F0E442", "#CC79A7", "#666666", "#000000"]
    overall = monitoring[
        (monitoring["stratum_type"] == "overall") & monitoring["detector"].isin(selected)
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for column, stage in enumerate(("original_main_oof", "original_holdout_diagnostic")):
        for row_index, application in enumerate(("commercial", "humanitarian")):
            ax = axes[row_index, column]
            group = overall[(overall["evaluation_stage"] == stage) & (overall["application"] == application)].set_index("detector")
            values = [group.loc[item, "average_precision"] if item in group.index else np.nan for item in selected]
            ax.bar(np.arange(len(selected)), values, color=colors, edgecolor="black", linewidth=0.3)
            prevalence = float(group["positive_class_prevalence"].iloc[0]) if len(group) else np.nan
            ax.axhline(prevalence, color="black", linestyle="--", linewidth=0.8)
            ax.text(
                0.02,
                min(prevalence + 0.025, 0.96),
                "positive prevalence",
                transform=ax.get_yaxis_transform(),
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            )
            ax.set_xticks(np.arange(len(selected)))
            ax.set_xticklabels([labels[item] for item in selected], rotation=36, ha="right")
            ax.set_ylim(0, 1.03)
            ax.set_ylabel("Average precision")
            ax.set_title("%s — %s" % (application.title(), "main OOF" if column == 0 else "seen holdout diagnostic"))
    figure.suptitle("Disruption monitoring: distributed entropy versus ordinary KPI baselines", fontsize=12)
    comparison = pdf_dir / "monitoring_baseline_comparison.pdf"
    figure.savefig(
        comparison,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)

    differences = incremental[incremental["analysis"].str.endswith("incremental_entropy")].copy()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    panels = [
        ("disruption_classification_incremental_entropy", "average_precision", "A. Full evaluator KPIs: disruption AP"),
        ("restricted_local_disruption_classification_incremental_entropy", "average_precision", "B. Private local KPIs: disruption AP"),
        ("future_service_loss_regression_incremental_entropy", "r_squared", "C. Full evaluator KPIs: future-loss $R^2$"),
        ("restricted_local_future_service_loss_regression_incremental_entropy", "r_squared", "D. Private local KPIs: future-loss $R^2$"),
    ]
    for ax, (analysis, metric, title) in zip(axes.ravel(), panels):
        group = differences[differences["analysis"] == analysis]
        order = {("original_main_oof", "commercial"): 0, ("original_main_oof", "humanitarian"): 1, ("original_holdout_diagnostic", "commercial"): 2, ("original_holdout_diagnostic", "humanitarian"): 3}
        group = group.assign(
            plot_order=[order[(row.evaluation_stage, row.application)] for row in group.itertuples()]
        ).sort_values("plot_order")
        positions = np.arange(len(group))
        bar_colors = ["#0072B2" if app == "commercial" else "#D55E00" for app in group["application"]]
        values = group[metric].astype(float).to_numpy()
        values[np.abs(values) < 1e-12] = 0.0
        hatches = ["" if stage == "original_main_oof" else "//" for stage in group["evaluation_stage"]]
        bars = ax.bar(positions, values, color=bar_colors, edgecolor="black", linewidth=0.3)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            ["%s\n%s" % ("Commercial" if row.application == "commercial" else "Humanitarian", "main OOF" if row.evaluation_stage == "original_main_oof" else "seen holdout") for row in group.itertuples()],
            rotation=14,
            ha="right",
        )
        ax.set_ylabel("KPI + entropy minus KPI")
        ax.set_title(title)
        if np.allclose(values, 0.0):
            ax.set_ylim(-0.01, 0.01)
            ax.text(0.5, 0.62, "No AP increment", transform=ax.transAxes, ha="center")
    figure.suptitle("Incremental value of distributed operational entropy", fontsize=12)
    incremental_pdf = pdf_dir / "entropy_incremental_value.pdf"
    figure.savefig(
        incremental_pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    return [comparison, incremental_pdf]


def write_readme(output_root: Path, monitoring: pd.DataFrame, incremental: pd.DataFrame) -> None:
    overall = monitoring[monitoring["stratum_type"] == "overall"]
    lines = [
        "# Monitoring validation against ordinary logistics indicators",
        "",
        "This retrospective gate uses exactly one frozen `scripted_independent` trajectory per application/scenario/seed, avoiding pseudo-replication of the same exogenous panel across methods. Original-main estimates are leave-one-environment-seed-out. Models and nominal 95th-percentile thresholds are refit without the held seed. The already seen original holdout is external diagnostic evidence only.",
        "",
        "All detector outputs use distributed entropy for deployable entropy rows. `exact_entropy_evaluator_only` is explicitly non-deployable. Entropy sketches are not free and this diagnostic does not make a communication-efficiency claim.",
        "",
        "## Overall results",
        "",
    ]
    display = [
        "operational_entropy_high",
        "operational_entropy_low",
        "operational_entropy_absolute_deviation",
        "KPI_CUSUM",
        "Page_Hinkley",
        "multivariate_KPI_logistic",
        "multivariate_KPI_plus_entropy_logistic",
    ]
    for stage in ("original_main_oof", "original_holdout_diagnostic"):
        lines.append("### %s" % ("Original main, out of seed" if stage == "original_main_oof" else "Seen original holdout, diagnostic only"))
        lines.append("")
        lines.append("| Application | Detector | Prevalence | AP | AP/prevalence | ROC AUC | Recall | False alarm |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        subset = overall[(overall["evaluation_stage"] == stage) & overall["detector"].isin(display)]
        for row in subset.sort_values(["application", "detector"]).itertuples():
            lines.append(
                "| %s | `%s` | %.3f | %.3f | %.2f | %.3f | %.3f | %.3f |"
                % (
                    row.application,
                    row.detector,
                    row.positive_class_prevalence,
                    row.average_precision,
                    row.average_precision_lift_over_prevalence,
                    row.roc_auc,
                    row.recall_at_nominal_95pct_threshold,
                    row.false_alarm_rate,
                )
            )
        lines.append("")
    delta = incremental[incremental["analysis"].str.endswith("incremental_entropy")]
    lines.extend([
        "## Incremental-value interpretation",
        "",
        "The definitive values are in `incremental_value.csv`. Positive AP/AUC/$R^2$ differences favor adding entropy; negative RMSE and Brier differences favor it. The ordinary model already includes current service loss, backlog/unmet-need pressure, impairment, communication volume, rolling moments, and EWMA, making this a stringent incremental test.",
        "",
        "The full evaluator-KPI classifier is already perfect on both stages, so entropy adds exactly zero AP or ROC AUC and cannot claim independent disruption-classification value there. The high-direction entropy ranking is perfect on the seen holdout, but its main-derived nominal threshold has zero recall; absolute-deviation calibration reverses and false-alarms on every holdout-negative timepoint. This is a threshold-transfer warning, not positive confirmation.",
        "",
    ])
    for row in delta.sort_values(["analysis", "evaluation_stage", "application"]).itertuples():
        if "classification" in row.analysis:
            lines.append("- %s, %s: adding entropy changes AP by `%+.4f`, ROC AUC by `%+.4f`, and Brier score by `%+.4f`." % (row.application, row.evaluation_stage, row.average_precision, row.roc_auc, row.brier_score))
        else:
            lines.append("- %s, %s: adding entropy changes future-loss RMSE by `%+.4f`, $R^2$ by `%+.4f`, and Spearman correlation by `%+.4f`." % (row.application, row.evaluation_stage, row.rmse, row.r_squared, row.spearman))
    lines.extend([
        "",
        "Rows prefixed `restricted_local_` compare each independent agent's private local KPI vector with the same vector plus that agent's final-round distributed entropy estimate. They test entropy as a privacy-preserving compressed system statistic after the full evaluator-KPI model has already shown no classification increment.",
        "",
        "Results are reported separately by connected, degraded, and partitioned communication and by isolated, correlated, compound, and nominal regimes in `monitoring_baselines.csv`. Episode-level detection timing is in `detection_lead_time.csv`; entropy-surprisal and ordinary-impairment localization are in `localization.csv`.",
        "The timing table uses the same pre-holdout evaluability correction as H4: sustained severe collapse is the third consecutive post-disruption period with normalized service loss at least 0.90, and same-period detections receive no lead-time credit. The earlier warm-up-sensitive rule and its 12-episode development audit remain in `../protocol/h4_evaluability_audit.json`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "./scripts/run-monitoring-validation-v2.sh",
        "```",
    ])
    (output_root / "monitoring" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(results_root: Path, output_root: Path) -> Dict[str, Any]:
    frame = prepare_frame(results_root)
    local = prepare_local_frame(results_root, frame)
    scores = cross_validated_scores(frame)
    monitoring = monitoring_table(scores)
    detections = detection_lead_table(scores)
    localization = localization_table(scores)
    regression = regression_predictions(frame)
    incremental = incremental_value_table(scores, regression)
    incremental = pd.concat(
        [incremental, restricted_information_table(local)], ignore_index=True
    )
    directory = output_root / "monitoring"
    directory.mkdir(parents=True, exist_ok=True)
    monitoring.to_csv(directory / "monitoring_baselines.csv", index=False)
    incremental.to_csv(directory / "incremental_value.csv", index=False)
    detections.to_csv(directory / "detection_lead_time.csv", index=False)
    localization.to_csv(directory / "localization.csv", index=False)
    scored_path = directory / "scored_timepoints.csv.gz"
    with scored_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline=""
            ) as handle:
                scores.to_csv(handle, index=False)
    pdfs = write_figures(monitoring, incremental, output_root)
    write_readme(output_root, monitoring, incremental)
    outputs = [
        directory / "monitoring_baselines.csv",
        directory / "incremental_value.csv",
        directory / "detection_lead_time.csv",
        directory / "localization.csv",
        directory / "scored_timepoints.csv.gz",
        directory / "README.md",
        *pdfs,
    ]
    manifest = {
        "status": "complete",
        "analysis_environment": _analysis_environment(),
        "scope": "frozen v1 main out-of-seed and seen holdout diagnostic monitoring validation",
        "trajectory_method": "scripted_independent only",
        "main_environment_seeds": sorted(int(value) for value in frame.loc[frame["stage"] == "main", "seed"].unique()),
        "holdout_environment_seeds": sorted(int(value) for value in frame.loc[frame["stage"] == "holdout", "seed"].unique()),
        "episodes": int(frame["run_id"].nunique()),
        "timepoints": len(frame),
        "restricted_local_agent_timepoints": len(local),
        "source_time_series_sha256": sha256_file(results_root / "processed" / "time_series.csv"),
        "outputs": {str(path.relative_to(output_root)): sha256_file(path) for path in outputs},
    }
    (directory / "monitoring_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-root", type=Path, default=Path("results/entropy_triggered_v2"))
    args = parser.parse_args(argv)
    print(json.dumps(run(args.results_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
