"""Episode-level aggregation and preregistered paired statistical analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .events import EventLedger, sha256_file
from .mechanics import ENERGY_WEIGHT_SENSITIVITY


PRIMARY_COMPARATORS = [
    "learned_no_entropy",
    "autonomous_fixed_comm",
    "scripted_independent",
    "centralized_lookahead",
    "centralized_llm",
]


def _analysis_valid_mask(frame: pd.DataFrame) -> pd.Series:
    if "analysis_valid" not in frame:
        return pd.Series(True, index=frame.index)
    return frame["analysis_valid"].fillna(True).astype(bool)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def collect_results(results_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_rows: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    seen_run_ids = set()
    exclusion_rules: List[Dict[str, str]] = []
    exclusion_path = results_root / "reproducibility" / "excluded_runs.json"
    if exclusion_path.exists():
        exclusion_rules = list(
            json.loads(exclusion_path.read_text(encoding="utf-8")).get("rules", [])
        )

    def exclusion_reason(run_id: str) -> Optional[str]:
        return next((
            str(rule["reason"]) for rule in exclusion_rules
            if str(rule.get("run_id_contains", ""))
            and str(rule["run_id_contains"]) in run_id
        ), None)

    for episode_path in sorted((results_root / "raw").glob("*/*/episode.json")):
        value = json.loads(episode_path.read_text(encoding="utf-8"))
        seen_run_ids.add(value["run_id"])
        stage = episode_path.parents[1].name
        disruption_sources: List[str] = []
        event_paths = list(episode_path.parent.glob("events.jsonl*"))
        if event_paths:
            try:
                events = EventLedger.read_jsonl(event_paths[0]).events
                disruptions = [event for event in events if event.kind == "disruption"]
                if disruptions:
                    disruption_sources = list(disruptions[0].payload.get("affected", []))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                disruption_sources = []
        excluded_because = exclusion_reason(value["run_id"])
        metadata = {
            "run_id": value["run_id"], "stage": stage, "application": value["application"],
            "method": value["method"], "scenario": value["scenario"], "seed": value["seed"],
            "completion_status": value["completion_status"], "wall_clock_seconds": value["wall_clock_seconds"],
            "disruption_sources": ";".join(disruption_sources),
            "analysis_valid": excluded_because is None,
            "exclusion_reason": excluded_because,
        }
        summary_candidates = list((results_root / stage).glob("episodes.csv"))
        scenario_name = None
        n_agents = None
        if summary_candidates:
            summary = pd.read_csv(summary_candidates[0])
            matched = summary[summary["run_id"] == value["run_id"]]
            if not matched.empty:
                scenario_name = matched.iloc[0].get("scenario_name")
                n_agents = matched.iloc[0].get("n_agents")
        metadata["scenario_name"] = scenario_name or value["scenario"]
        metadata["n_agents"] = n_agents
        episode_rows.append({**metadata, **value["metrics"], **value["planner_metrics"]})
        # Observable agentic behavior spans planner-level rates and system
        # outcomes such as agreement quality and commitment breaches.
        agent_rows.append({**metadata, **value["metrics"], **value["agent_metrics"]})
        for row in value["time_series"]:
            time_rows.append({**metadata, **row})
    # Failed/timed-out rows do not have a valid episode ledger. Retain them as
    # primary experimental units from each stage completion table rather than
    # silently dropping them during aggregation.
    for stage in ("smoke", "pilot", "main", "ablations", "holdout"):
        summary_path = results_root / stage / "episodes.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            run_id = str(row.get("run_id", ""))
            if not run_id or run_id in seen_run_ids:
                continue
            status = str(row.get("status", row.get("completion_status", "failed")))
            episode_rows.append({
                "run_id": run_id,
                "stage": stage,
                "application": row.get("application"),
                "method": row.get("method"),
                "scenario": row.get("scenario"),
                "scenario_name": row.get("scenario_name"),
                "seed": row.get("seed"),
                "n_agents": row.get("n_agents"),
                "completion_status": status,
                "failure_reason": row.get("error"),
                "wall_clock_seconds": row.get("wall_clock_seconds", 0.0),
                "analysis_valid": exclusion_reason(run_id) is None,
                "exclusion_reason": exclusion_reason(run_id),
            })
    episodes = pd.DataFrame(episode_rows)
    time_series = pd.DataFrame(time_rows)
    agent_metrics = pd.DataFrame(agent_rows)
    return episodes, time_series, agent_metrics


def paired_bootstrap(improvements: np.ndarray, seed: int = 20260811, draws: int = 10000) -> Tuple[float, float]:
    values = np.asarray(improvements, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = values[rng.randint(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def hierarchical_paired_bootstrap(
    paired: pd.DataFrame,
    improvement_column: str,
    cluster_column: str = "seed",
    seed: int = 20260811,
    draws: int = 10000,
) -> Tuple[float, float]:
    """Cluster bootstrap repeated scenario cells at the environment-seed level."""

    if paired.empty:
        return float("nan"), float("nan")
    clusters = list(paired[cluster_column].drop_duplicates())
    rng = np.random.RandomState(seed)
    bootstrap_means = np.empty(draws, dtype=float)
    groups = {cluster: paired[paired[cluster_column] == cluster][improvement_column].to_numpy(float) for cluster in clusters}
    for draw in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        bootstrap_means[draw] = np.mean(np.concatenate([groups[cluster] for cluster in sampled]))
    return float(np.quantile(bootstrap_means, 0.025)), float(np.quantile(bootstrap_means, 0.975))


def sign_flip_pvalue(values: np.ndarray, seed: int = 20260811, draws: int = 20000) -> float:
    """Two-sided paired randomization test with deterministic Monte Carlo."""

    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(values.mean()))
    if values.size <= 16:
        masks = np.arange(2 ** values.size, dtype=np.uint64)[:, None]
        bits = ((masks >> np.arange(values.size, dtype=np.uint64)) & 1).astype(float)
        signs = bits * 2.0 - 1.0
        null = np.abs((signs * values).mean(axis=1))
    else:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1.0, 1.0], size=(draws, values.size))
        null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (len(null) + 1))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def primary_statistics(episodes: pd.DataFrame, stage: str = "main") -> pd.DataFrame:
    frame = episodes[
        (episodes["stage"] == stage)
        & (episodes["completion_status"] == "complete")
        & _analysis_valid_mask(episodes)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for application in sorted(frame["application"].dropna().unique()):
        app = frame[frame["application"] == application]
        thermo = app[app["method"] == "thermoagent"]
        for comparator in PRIMARY_COMPARATORS:
            control = app[app["method"] == comparator]
            keys = ["scenario_name", "seed", "n_agents"]
            paired = thermo.merge(control, on=keys, suffixes=("_thermo", "_control"))
            if paired.empty:
                continue
            # Lower primary outcome is better; positive improvement favors ThermoAgent.
            improvement = paired["primary_outcome_control"].to_numpy(float) - paired["primary_outcome_thermo"].to_numpy(float)
            paired = paired.assign(improvement=improvement)
            ci_low, ci_high = hierarchical_paired_bootstrap(paired, "improvement")
            cluster_improvement = paired.groupby("seed")["improvement"].mean().to_numpy(float)
            sd = float(np.std(cluster_improvement, ddof=1)) if cluster_improvement.size > 1 else float("nan")
            if cluster_improvement.size > 1 and np.any(np.abs(cluster_improvement) > 0):
                p_value = float(stats.ttest_1samp(cluster_improvement, 0.0).pvalue)
            else:
                p_value = 1.0
            randomization_p = sign_flip_pvalue(cluster_improvement)
            rows.append({
                "application": application,
                "treatment": "thermoagent",
                "comparator": comparator,
                "n_pairs": int(improvement.size),
                "n_environment_seed_clusters": int(cluster_improvement.size),
                "mean_treatment": float(paired["primary_outcome_thermo"].mean()),
                "mean_comparator": float(paired["primary_outcome_control"].mean()),
                "mean_improvement": float(improvement.mean()),
                "relative_improvement_percent": float(100.0 * improvement.mean() / max(abs(paired["primary_outcome_control"].mean()), 1e-9)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_win_rate": float(np.mean(improvement > 0)),
                "probability_superiority": float(np.mean(improvement > 0) + 0.5 * np.mean(improvement == 0)),
                "standardized_effect_dz": float(improvement.mean() / sd) if sd > 0 else 0.0,
                "p_value_paired_t": p_value,
                "p_value_sign_flip": randomization_p,
            })
    output = pd.DataFrame(rows)
    if not output.empty:
        adjusted: List[float] = [1.0] * len(output)
        for application in output["application"].unique():
            indices = output.index[output["application"] == application].tolist()
            values = output.loc[indices, "p_value_sign_flip"].tolist()
            for index, value in zip(indices, holm_adjust(values)):
                adjusted[int(index)] = value
        output["p_value_holm"] = adjusted
    return output


def scenario_paired_statistics(episodes: pd.DataFrame, stage: str = "main") -> pd.DataFrame:
    frame = episodes[
        (episodes["stage"] == stage)
        & (episodes["completion_status"] == "complete")
        & _analysis_valid_mask(episodes)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for application in sorted(frame["application"].dropna().unique()):
        app = frame[frame["application"] == application]
        thermo = app[app["method"] == "thermoagent"]
        for comparator in PRIMARY_COMPARATORS:
            control = app[app["method"] == comparator]
            paired = thermo.merge(control, on=["scenario_name", "seed", "n_agents"], suffixes=("_thermo", "_control"))
            for scenario_name, group in paired.groupby("scenario_name"):
                improvement = group["primary_outcome_control"].to_numpy(float) - group["primary_outcome_thermo"].to_numpy(float)
                low, high = paired_bootstrap(improvement)
                rows.append({
                    "application": application,
                    "scenario_name": scenario_name,
                    "treatment": "thermoagent",
                    "comparator": comparator,
                    "n_pairs": len(improvement),
                    "mean_improvement": float(np.mean(improvement)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_win_rate": float(np.mean(improvement > 0)),
                    "p_value_sign_flip": sign_flip_pvalue(improvement),
                })
    return pd.DataFrame(rows)


def all_method_paired_statistics(episodes: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Parameter-matched comparisons against ThermoAgent within one stage."""

    frame = episodes[
        (episodes["stage"] == stage)
        & (episodes["completion_status"] == "complete")
        & _analysis_valid_mask(episodes)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for application in sorted(frame["application"].dropna().unique()):
        app = frame[frame["application"] == application]
        thermo = app[app["method"] == "thermoagent"]
        for comparator in sorted(set(app["method"]) - {"thermoagent"}):
            control = app[app["method"] == comparator]
            paired = thermo.merge(
                control,
                on=["scenario_name", "seed", "n_agents"],
                suffixes=("_thermo", "_control"),
            )
            if paired.empty:
                continue
            improvement = (
                paired["primary_outcome_control"].to_numpy(float)
                - paired["primary_outcome_thermo"].to_numpy(float)
            )
            low, high = paired_bootstrap(improvement)
            rows.append({
                "stage": stage,
                "application": application,
                "treatment": "thermoagent",
                "comparator": comparator,
                "n_pairs": len(improvement),
                "mean_improvement": float(np.mean(improvement)),
                "ci95_low": low,
                "ci95_high": high,
                "paired_win_rate": float(np.mean(improvement > 0)),
                "probability_superiority": float(
                    np.mean(improvement > 0) + 0.5 * np.mean(improvement == 0)
                ),
                "p_value_sign_flip": sign_flip_pvalue(improvement),
            })
    output = pd.DataFrame(rows)
    if not output.empty:
        adjusted = np.ones(len(output), dtype=float)
        for application in output["application"].unique():
            indices = output.index[output["application"] == application].tolist()
            adjusted[indices] = holm_adjust(
                output.loc[indices, "p_value_sign_flip"].tolist()
            )
        output["p_value_holm"] = adjusted
    return output


def method_summary(episodes: pd.DataFrame, stage: str = "main") -> pd.DataFrame:
    frame = episodes[
        (episodes["stage"] == stage)
        & (episodes["completion_status"] == "complete")
        & _analysis_valid_mask(episodes)
    ]
    metrics = [
        "primary_outcome", "fulfillment_rate", "fairness", "messages",
        "monitor_sketch_messages", "total_communication_messages",
        "total_communication_bytes", "generated_tokens", "wall_clock_seconds",
    ]
    rows = []
    for keys, group in frame.groupby(["application", "scenario_name", "method"], dropna=False):
        row = {"application": keys[0], "scenario_name": keys[1], "method": keys[2], "n": len(group)}
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].astype(float).to_numpy()
            ci_low, ci_high = paired_bootstrap(values, seed=20260812, draws=5000)
            row.update({metric + "_mean": float(np.mean(values)), metric + "_ci95_low": ci_low, metric + "_ci95_high": ci_high})
        rows.append(row)
    return pd.DataFrame(rows)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order].astype(int)
    positives = labels.sum()
    if positives == 0:
        return float("nan")
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, labels.size + 1)
    return float(np.sum(precision * labels) / positives)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum()))


def _valid_final_pilot_mask(frame: pd.DataFrame) -> pd.Series:
    """Select only the post-audit, paired final pilot cells."""

    names = frame["scenario_name"].astype(str)
    return (
        (frame["stage"] == "pilot")
        & names.str.startswith("paired_")
        & _analysis_valid_mask(frame)
    )


def failure_aware_paired_statistics(
    episodes: pd.DataFrame,
    stage: str = "main",
) -> pd.DataFrame:
    """Pair every planned row and treat asymmetric failure as a ranked loss.

    Numeric effects remain complete-case because unlike the commercial bounded
    service-loss AUC, humanitarian weighted unmet need has no common finite
    ceiling. The failure-aware win rate prevents a method from benefiting when
    its failed or timed-out episodes have no numeric outcome.
    """

    frame = episodes[
        (episodes["stage"] == stage) & _analysis_valid_mask(episodes)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for application in sorted(frame["application"].dropna().unique()):
        app = frame[frame["application"] == application]
        thermo = app[app["method"] == "thermoagent"]
        for comparator in PRIMARY_COMPARATORS:
            control = app[app["method"] == comparator]
            paired = thermo.merge(
                control,
                on=["scenario_name", "seed", "n_agents"],
                suffixes=("_thermo", "_control"),
            )
            if paired.empty:
                continue
            thermo_complete = paired["completion_status_thermo"] == "complete"
            control_complete = paired["completion_status_control"] == "complete"
            both_complete = thermo_complete & control_complete
            scores = np.where(
                thermo_complete & ~control_complete,
                1.0,
                np.where(
                    ~thermo_complete & control_complete,
                    0.0,
                    np.where(
                        ~thermo_complete & ~control_complete,
                        0.5,
                        np.where(
                            paired["primary_outcome_thermo"].astype(float)
                            < paired["primary_outcome_control"].astype(float),
                            1.0,
                            np.where(
                                paired["primary_outcome_thermo"].astype(float)
                                > paired["primary_outcome_control"].astype(float),
                                0.0,
                                0.5,
                            ),
                        ),
                    ),
                ),
            )
            complete_improvement = (
                paired.loc[both_complete, "primary_outcome_control"].astype(float)
                - paired.loc[both_complete, "primary_outcome_thermo"].astype(float)
            ).to_numpy()
            low, high = paired_bootstrap(complete_improvement)
            rows.append({
                "stage": stage,
                "application": application,
                "treatment": "thermoagent",
                "comparator": comparator,
                "n_planned_pairs": len(paired),
                "n_both_complete": int(both_complete.sum()),
                "thermoagent_only_failures": int((~thermo_complete & control_complete).sum()),
                "comparator_only_failures": int((thermo_complete & ~control_complete).sum()),
                "both_failed": int((~thermo_complete & ~control_complete).sum()),
                "failure_aware_win_rate": float(np.mean(scores)),
                "complete_case_mean_improvement": (
                    float(np.mean(complete_improvement))
                    if complete_improvement.size else None
                ),
                "complete_case_ci95_low": low,
                "complete_case_ci95_high": high,
            })
    return pd.DataFrame(rows)


def monitoring_statistics(
    time_series: pd.DataFrame,
    results_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if time_series.empty:
        return (pd.DataFrame(),) * 6
    pilot = time_series[time_series["stage"] == "pilot"]
    calibration = pilot[
        (pilot["scenario_name"] == "paired_nominal_v8")
        & (pilot["method"] == "scripted_independent")
    ]
    if calibration.empty:
        calibration = pilot[pilot["scenario_name"] == "nominal"]
    if calibration.empty:
        calibration = time_series[~time_series["disruption_active"].astype(bool)]
    global_frame = time_series[
        time_series["stage"].isin(["main", "holdout"])
        & _analysis_valid_mask(time_series)
    ].copy()
    if global_frame.empty:
        # Pilot-only analysis remains useful before main, while excluding
        # obsolete mechanism-diagnostic and interrupted qualification cells.
        final_pilot = time_series[_valid_final_pilot_mask(time_series)]
        global_frame = final_pilot.copy() if not final_pilot.empty else pilot.copy()
    nominal_free_median = float(calibration["exact_free_energy"].median())
    global_frame["absolute_free_energy_deviation"] = (
        global_frame["exact_free_energy"].astype(float) - nominal_free_median
    ).abs()
    calibration = calibration.copy()
    calibration["absolute_free_energy_deviation"] = (
        calibration["exact_free_energy"].astype(float) - nominal_free_median
    ).abs()
    labels = global_frame["disruption_active"].astype(int).to_numpy()
    signal_names = [
        "exact_free_energy", "absolute_free_energy_deviation", "exact_entropy",
        "exact_energy", "distributed_free_energy_mean", "interaction_entropy",
    ]
    signal_names.extend(sorted(
        column for column in global_frame.columns
        if column.startswith("exact_energy_sensitivity_")
        or column.startswith("exact_free_energy_sensitivity_")
    ))
    aggregate_rows: List[Dict[str, Any]] = []
    thresholds: Dict[str, float] = {}
    for signal in signal_names:
        if signal not in global_frame or signal not in calibration:
            continue
        threshold = float(calibration[signal].quantile(0.95))
        thresholds[signal] = threshold
        scores = global_frame[signal].astype(float).to_numpy()
        predicted = scores > threshold
        true_positive = int(np.sum(predicted & (labels == 1)))
        aggregate_rows.append({
            "signal": signal,
            "direction": "high",
            "threshold_nominal_95pct": threshold,
            "average_precision": _average_precision(labels, scores),
            "roc_auc": _roc_auc(labels, scores),
            "precision_at_threshold": true_positive / max(int(np.sum(predicted)), 1),
            "recall_at_threshold": true_positive / max(int(np.sum(labels == 1)), 1),
            "false_alarm_rate": float(np.mean(predicted[labels == 0])) if np.any(labels == 0) else float("nan"),
            "n_timepoints": len(global_frame),
            "nominal_reference_median": nominal_free_median if "free_energy" in signal else None,
        })
    aggregate = pd.DataFrame(aggregate_rows)
    episode_rows: List[Dict[str, Any]] = []
    for run_id, group in global_frame.groupby("run_id"):
        group = group.sort_values("step")
        active = group[group["disruption_active"].astype(bool)]
        if active.empty:
            continue
        disruption_step = int(active["step"].min())
        pre = group[group["step"] < disruption_step]["service_loss"]
        baseline = float(pre.mean()) if not pre.empty else float(group.iloc[0]["service_loss"])
        collapse = group[(group["step"] >= disruption_step) & (group["service_loss"] > baseline + 0.10)]
        collapse_step = int(collapse["step"].min()) if not collapse.empty else None
        for signal, threshold in thresholds.items():
            detected = group[(group["step"] >= disruption_step) & (group[signal] > threshold)]
            detection_step = int(detected["step"].min()) if not detected.empty else None
            lead = (collapse_step - detection_step) if detection_step is not None and collapse_step is not None else None
            episode_rows.append({
                "run_id": run_id,
                "application": group.iloc[0]["application"],
                "method": group.iloc[0]["method"],
                "scenario_name": group.iloc[0]["scenario_name"],
                "signal": signal,
                "disruption_step": disruption_step,
                "detection_step": detection_step,
                "detection_delay": None if detection_step is None else detection_step - disruption_step,
                "visible_collapse_step": collapse_step,
                "detection_lead_before_collapse": lead,
                "detected_before_collapse": bool(lead is not None and lead >= 0),
            })

    prediction_rows: List[Dict[str, Any]] = []
    future_rows: List[Dict[str, Any]] = []
    for run_id, group in global_frame.groupby("run_id"):
        group = group.sort_values("step").copy()
        losses = group["service_loss"].astype(float).to_numpy()
        future_loss = np.asarray([
            np.mean(losses[index + 1 : min(len(losses), index + 4)])
            if index + 1 < len(losses) else losses[index]
            for index in range(len(losses))
        ])
        group["future_service_loss_3_period"] = future_loss
        future_rows.extend(group.to_dict("records"))
        for signal in signal_names:
            if signal not in group or len(group) < 3:
                continue
            correlation = stats.spearmanr(group[signal].astype(float), future_loss).correlation
            prediction_rows.append({
                "run_id": run_id,
                "application": group.iloc[0]["application"],
                "method": group.iloc[0]["method"],
                "scenario_name": group.iloc[0]["scenario_name"],
                "signal": signal,
                "spearman_future_service_loss": float(correlation) if np.isfinite(correlation) else None,
            })
    predictive = pd.DataFrame(prediction_rows)

    calibration_rows: List[Dict[str, Any]] = []
    future_frame = pd.DataFrame(future_rows)
    if not future_frame.empty:
        for application, app in future_frame.groupby("application"):
            try:
                bins = pd.qcut(app["exact_free_energy"], q=5, labels=False, duplicates="drop")
            except ValueError:
                continue
            for bin_index, group in app.assign(calibration_bin=bins).groupby("calibration_bin"):
                calibration_rows.append({
                    "application": application,
                    "free_energy_quantile_bin": int(bin_index),
                    "mean_free_energy": float(group["exact_free_energy"].mean()),
                    "mean_future_service_loss_3_period": float(group["future_service_loss_3_period"].mean()),
                    "n_timepoints": len(group),
                })

    convergence_rows: List[Dict[str, Any]] = []
    if not global_frame.empty:
        global_frame["entropy_estimation_absolute_error"] = (
            global_frame["distributed_entropy_mean"] - global_frame["exact_entropy"]
        ).abs()
        global_frame["free_energy_estimation_absolute_error"] = (
            global_frame["distributed_free_energy_mean"] - global_frame["exact_free_energy"]
        ).abs()
        for communication, group in global_frame.groupby(global_frame["scenario"].astype(str).str.split("-").str[0]):
            convergence_rows.append({
                "communication_regime": communication,
                "n_timepoints": len(group),
                "mean_consensus_rmse": float(group["consensus_rmse"].mean()),
                "mean_entropy_absolute_error": float(group["entropy_estimation_absolute_error"].mean()),
                "mean_free_energy_absolute_error": float(group["free_energy_estimation_absolute_error"].mean()),
                "rmse_entropy_error_spearman": float(stats.spearmanr(group["consensus_rmse"], group["entropy_estimation_absolute_error"]).correlation),
                "rmse_free_energy_error_spearman": float(stats.spearmanr(group["consensus_rmse"], group["free_energy_estimation_absolute_error"]).correlation),
            })

    localization_rows: List[Dict[str, Any]] = []
    for run_id, group in global_frame.groupby("run_id"):
        sources = {source for source in str(group.iloc[0].get("disruption_sources", "")).split(";") if source}
        disrupted = group[group["disruption_active"].astype(bool)].sort_values("step")
        if not sources or disrupted.empty:
            continue
        first_disrupted = disrupted.iloc[0]
        predicted_source = str(first_disrupted["max_surprisal_agent"])
        ranking = [
            agent_id for agent_id in str(
                first_disrupted.get("surprisal_ranked_agents", predicted_source)
            ).split(";") if agent_id
        ]
        localization_rows.append({
            "run_id": run_id,
            "application": group.iloc[0]["application"],
            "scenario_name": group.iloc[0]["scenario_name"],
            "predicted_source": predicted_source,
            "true_sources": ";".join(sorted(sources)),
            "top1_localization_correct": predicted_source in sources,
            "top3_localization_correct": bool(set(ranking[:3]) & sources),
        })
    return (
        aggregate,
        pd.DataFrame(episode_rows),
        predictive,
        pd.DataFrame(calibration_rows),
        pd.DataFrame(convergence_rows),
        pd.DataFrame(localization_rows),
    )


def estimator_comparison_statistics(time_series: pd.DataFrame) -> pd.DataFrame:
    """Compare exact, distributed, delayed, noisy, and absent estimators."""

    frame = time_series[
        time_series["stage"].isin(["main", "holdout"])
        & _analysis_valid_mask(time_series)
    ].copy()
    if frame.empty:
        frame = time_series[_valid_final_pilot_mask(time_series)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["communication_regime"] = frame["scenario"].astype(str).str.split("-").str[0]
    estimators = {
        "exact_evaluator_only": ("exact_entropy", "exact_free_energy"),
        "agent_local_distributed": (
            "distributed_entropy_mean", "distributed_free_energy_mean"
        ),
        "one_period_delayed": ("delayed_entropy_mean", "delayed_free_energy_mean"),
        "noisy_distributed_sigma_0.01": (
            "noisy_entropy_mean", "noisy_free_energy_mean"
        ),
    }
    rows: List[Dict[str, Any]] = []
    for (application, communication), group in frame.groupby(
        ["application", "communication_regime"]
    ):
        for name, (entropy_column, free_column) in estimators.items():
            if entropy_column not in group or free_column not in group:
                continue
            rows.append({
                "application": application,
                "communication_regime": communication,
                "estimator": name,
                "n_timepoints": len(group),
                "entropy_mae_vs_exact": float(
                    np.mean(np.abs(group[entropy_column] - group["exact_entropy"]))
                ),
                "free_energy_mae_vs_exact": float(
                    np.mean(np.abs(group[free_column] - group["exact_free_energy"]))
                ),
            })
        rows.append({
            "application": application,
            "communication_regime": communication,
            "estimator": "no_entropy_estimate",
            "n_timepoints": len(group),
            "entropy_mae_vs_exact": float(np.mean(np.abs(group["exact_entropy"]))),
            "free_energy_mae_vs_exact": float(
                np.mean(np.abs(group["exact_free_energy"]))
            ),
        })
    return pd.DataFrame(rows)


def detection_episode_summary(detections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monitoring timing at the complete-episode unit."""

    columns = [
        "application", "method", "signal", "n_disrupted_episodes",
        "detected_episodes", "detection_rate", "mean_detection_delay_detected",
        "median_detection_delay_detected", "visible_collapse_episodes",
        "detected_before_visible_collapse",
        "proportion_before_visible_collapse",
    ]
    if detections.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for keys, group in detections.groupby(["application", "method", "signal"]):
        detected = group["detection_step"].notna()
        delays = pd.to_numeric(
            group.loc[detected, "detection_delay"], errors="coerce"
        ).dropna()
        visible = group["visible_collapse_step"].notna()
        before = (
            group.loc[visible, "detected_before_collapse"]
            .fillna(False)
            .astype(bool)
        )
        rows.append({
            "application": keys[0],
            "method": keys[1],
            "signal": keys[2],
            "n_disrupted_episodes": len(group),
            "detected_episodes": int(detected.sum()),
            "detection_rate": float(detected.mean()),
            "mean_detection_delay_detected": (
                float(delays.mean()) if len(delays) else None
            ),
            "median_detection_delay_detected": (
                float(delays.median()) if len(delays) else None
            ),
            "visible_collapse_episodes": int(visible.sum()),
            "detected_before_visible_collapse": int(before.sum()),
            "proportion_before_visible_collapse": (
                float(before.mean()) if len(before) else None
            ),
        })
    return pd.DataFrame(rows, columns=columns)


def localization_episode_summary(localization: pd.DataFrame) -> pd.DataFrame:
    """Report top-k disrupted-source localization at the episode unit."""

    columns = [
        "application", "scenario_name", "n_disrupted_episodes",
        "top1_localization_accuracy", "top3_localization_accuracy",
    ]
    if localization.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for keys, group in localization.groupby(["application", "scenario_name"]):
        rows.append({
            "application": keys[0],
            "scenario_name": keys[1],
            "n_disrupted_episodes": len(group),
            "top1_localization_accuracy": float(
                group["top1_localization_correct"].astype(bool).mean()
            ),
            "top3_localization_accuracy": float(
                group["top3_localization_correct"].astype(bool).mean()
            ),
        })
    return pd.DataFrame(rows, columns=columns)


def interaction_regime_statistics(time_series: pd.DataFrame) -> pd.DataFrame:
    """Summarize joint operational/interaction entropy regimes."""

    pilot = time_series[
        (time_series["stage"] == "pilot")
        & (time_series["scenario_name"] == "paired_nominal_v8")
        & (time_series["method"] == "scripted_independent")
    ]
    if pilot.empty:
        pilot = time_series[~time_series["disruption_active"].astype(bool)]
    evaluation = time_series[
        time_series["stage"].isin(["main", "holdout"])
        & _analysis_valid_mask(time_series)
    ].copy()
    if evaluation.empty:
        evaluation = time_series[_valid_final_pilot_mask(time_series)].copy()
    if pilot.empty or evaluation.empty:
        return pd.DataFrame()
    operational_threshold = float(pilot["exact_entropy"].quantile(0.75))
    interaction_threshold = float(pilot["interaction_entropy"].quantile(0.75))
    evaluation["operational_regime"] = np.where(
        evaluation["exact_entropy"] > operational_threshold, "high", "low"
    )
    evaluation["interaction_regime"] = np.where(
        evaluation["interaction_entropy"] > interaction_threshold,
        "broad",
        "concentrated",
    )
    rows: List[Dict[str, Any]] = []
    for keys, group in evaluation.groupby(
        ["application", "operational_regime", "interaction_regime"]
    ):
        rows.append({
            "application": keys[0],
            "operational_entropy_regime": keys[1],
            "interaction_entropy_regime": keys[2],
            "n_timepoints": len(group),
            "mean_service_loss": float(group["service_loss"].mean()),
            "mean_free_energy_gap": float(group["exact_free_energy"].mean()),
            "operational_threshold_nominal_75pct": operational_threshold,
            "interaction_threshold_nominal_75pct": interaction_threshold,
        })
    return pd.DataFrame(rows)


def energy_weight_sensitivity_statistics(time_series: pd.DataFrame) -> pd.DataFrame:
    """Summarize pre-main fixed energy-weight alternatives without refitting."""

    if time_series.empty or "disruption_active" not in time_series:
        return pd.DataFrame()
    frame = time_series[
        time_series["stage"].isin(["main", "holdout"])
        & _analysis_valid_mask(time_series)
    ].copy()
    if frame.empty:
        frame = time_series[_valid_final_pilot_mask(time_series)].copy()
    if frame.empty:
        return pd.DataFrame()
    future_frames: List[pd.DataFrame] = []
    for _, group in frame.groupby("run_id"):
        group = group.sort_values("step").copy()
        losses = group["service_loss"].astype(float).to_numpy()
        group["future_service_loss_3_period"] = [
            float(np.mean(losses[index + 1 : min(len(losses), index + 4)]))
            if index + 1 < len(losses) else float(losses[index])
            for index in range(len(losses))
        ]
        future_frames.append(group)
    frame = pd.concat(future_frames, ignore_index=True)
    variants: Dict[str, Tuple[float, float, float, float]] = {
        "primary": (0.35, 0.20, 0.30, 0.15),
        **ENERGY_WEIGHT_SENSITIVITY,
    }
    rows: List[Dict[str, Any]] = []
    for application, app in frame.groupby("application"):
        labels = app["disruption_active"].astype(int).to_numpy()
        for variant, weights in variants.items():
            for construct in ("energy", "free_energy"):
                signal = (
                    "exact_" + construct if variant == "primary"
                    else "exact_%s_sensitivity_%s" % (construct, variant)
                )
                if signal not in app:
                    continue
                values = app[signal].astype(float)
                nominal = values[labels == 0]
                disrupted = values[labels == 1]
                current_correlation = stats.spearmanr(values, app["service_loss"].astype(float)).correlation
                future_correlation = stats.spearmanr(
                    values, app["future_service_loss_3_period"].astype(float)
                ).correlation
                rows.append({
                    "application": application,
                    "construct": construct,
                    "weight_variant": variant,
                    "weight_backlog": weights[0],
                    "weight_delay": weights[1],
                    "weight_shortfall": weights[2],
                    "weight_commitment": weights[3],
                    "nominal_mean": float(nominal.mean()) if len(nominal) else None,
                    "disrupted_mean": float(disrupted.mean()) if len(disrupted) else None,
                    "disrupted_minus_nominal": (
                        float(disrupted.mean() - nominal.mean())
                        if len(nominal) and len(disrupted) else None
                    ),
                    "average_precision_disruption": _average_precision(labels, values.to_numpy()),
                    "roc_auc_disruption": _roc_auc(labels, values.to_numpy()),
                    "spearman_current_service_loss": (
                        float(current_correlation) if np.isfinite(current_correlation) else None
                    ),
                    "spearman_future_service_loss": (
                        float(future_correlation) if np.isfinite(future_correlation) else None
                    ),
                    "n_timepoints": len(app),
                })
    return pd.DataFrame(rows)


def write_analysis(results_root: Path) -> Dict[str, Any]:
    processed = results_root / "processed"
    statistics = results_root / "statistics"
    tables = results_root / "tables"
    for directory in (processed, statistics, tables):
        directory.mkdir(parents=True, exist_ok=True)
    episodes, time_series, agent_metrics = collect_results(results_root)
    episodes.to_csv(processed / "episodes.csv", index=False)
    time_series.to_csv(processed / "time_series.csv", index=False)
    agent_metrics.to_csv(processed / "agent_metrics.csv", index=False)
    primary = primary_statistics(episodes)
    main_all_methods = all_method_paired_statistics(episodes, stage="main")
    holdout_primary = primary_statistics(episodes, stage="holdout")
    ablation_primary = all_method_paired_statistics(episodes, stage="ablations")
    failure_aware_main = failure_aware_paired_statistics(episodes, stage="main")
    failure_aware_holdout = failure_aware_paired_statistics(episodes, stage="holdout")
    scenario_primary = scenario_paired_statistics(episodes)
    pilot_scenario = scenario_paired_statistics(episodes, stage="pilot")
    summary = method_summary(episodes)
    monitoring, detections, predictive, calibration_bins, convergence, localization = monitoring_statistics(time_series, results_root)
    detection_summary = detection_episode_summary(detections)
    localization_summary = localization_episode_summary(localization)
    energy_sensitivity = energy_weight_sensitivity_statistics(time_series)
    estimator_comparison = estimator_comparison_statistics(time_series)
    interaction_regimes = interaction_regime_statistics(time_series)
    primary.to_csv(statistics / "primary_paired_comparisons.csv", index=False)
    main_all_methods.to_csv(
        statistics / "main_all_method_paired_comparisons.csv", index=False
    )
    holdout_primary.to_csv(statistics / "holdout_paired_comparisons.csv", index=False)
    ablation_primary.to_csv(statistics / "ablation_paired_comparisons.csv", index=False)
    failure_aware_main.to_csv(statistics / "main_failure_aware_comparisons.csv", index=False)
    failure_aware_holdout.to_csv(statistics / "holdout_failure_aware_comparisons.csv", index=False)
    scenario_primary.to_csv(statistics / "scenario_paired_comparisons.csv", index=False)
    pilot_scenario.to_csv(statistics / "pilot_scenario_paired_diagnostics.csv", index=False)
    summary.to_csv(tables / "method_summary.csv", index=False)
    monitoring.to_csv(statistics / "monitoring_summary.csv", index=False)
    detections.to_csv(statistics / "detection_episodes.csv", index=False)
    detection_summary.to_csv(
        statistics / "detection_episode_summary.csv", index=False
    )
    predictive.to_csv(statistics / "monitoring_predictive_value.csv", index=False)
    calibration_bins.to_csv(statistics / "free_energy_calibration.csv", index=False)
    convergence.to_csv(statistics / "distributed_convergence.csv", index=False)
    localization.to_csv(statistics / "source_localization.csv", index=False)
    localization_summary.to_csv(
        statistics / "source_localization_summary.csv", index=False
    )
    energy_sensitivity.to_csv(statistics / "energy_weight_sensitivity.csv", index=False)
    estimator_comparison.to_csv(statistics / "estimator_comparison.csv", index=False)
    interaction_regimes.to_csv(statistics / "interaction_entropy_regimes.csv", index=False)
    failures = episodes[episodes["completion_status"] != "complete"].copy()
    failures.to_csv(statistics / "failed_episodes.csv", index=False)
    excluded = episodes[~_analysis_valid_mask(episodes)].copy()
    excluded.to_csv(statistics / "excluded_episodes.csv", index=False)
    completion = (
        episodes.groupby(["stage", "application", "method"], dropna=False)
        .agg(
            episodes=("run_id", "size"),
            complete=("completion_status", lambda values: int(np.sum(values == "complete"))),
            failed_or_timed_out=("completion_status", lambda values: int(np.sum(values != "complete"))),
        )
        .reset_index()
    ) if not episodes.empty else pd.DataFrame()
    completion.to_csv(statistics / "completion_by_method.csv", index=False)
    compute = {
        "episode_wall_clock_hours": float(episodes.get("wall_clock_seconds", pd.Series(dtype=float)).sum() / 3600.0),
        "llm_calls": int(episodes.get("llm_calls", pd.Series(dtype=float)).fillna(0).sum()),
        "prompt_tokens": int(episodes.get("prompt_tokens", pd.Series(dtype=float)).fillna(0).sum()),
        "generated_tokens": int(episodes.get("generated_tokens", pd.Series(dtype=float)).fillna(0).sum()),
        "episodes": int(len(episodes)),
        "failed_episodes": int(np.sum(episodes.get("completion_status", pd.Series(dtype=str)) != "complete")) if len(episodes) else 0,
        "excluded_complete_episodes": int(
            np.sum(~_analysis_valid_mask(episodes)) if len(episodes) else 0
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Episode wall time excludes one-time model loading unless an external sweep log adds it; GPU-hour accounting is finalized from job manifests.",
    }
    (statistics / "compute_accounting.json").write_text(json.dumps(compute, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return compute


def build_index(results_root: Path) -> int:
    index_path = results_root / "INDEX.csv"
    rows: List[Dict[str, Any]] = []
    for path in sorted(p for p in results_root.rglob("*") if p.is_file() and p != index_path):
        relative = path.relative_to(results_root)
        parts = relative.parts
        stage = next((name for name in ("smoke", "pilot", "main", "ablations", "holdout") if name in parts), "cross-stage")
        artifact_type = path.suffix.lstrip(".") or "file"
        application = "both"
        method = "multiple"
        scenario = "multiple"
        seed = "multiple"
        if path.name == "episode.json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                application, method, scenario, seed = value["application"], value["method"], value["scenario"], value["seed"]
            except (KeyError, json.JSONDecodeError):
                pass
        rows.append({
            "artifact_path": str(relative),
            "artifact_type": artifact_type,
            "experiment_stage": stage,
            "application": application,
            "method": method,
            "scenario": scenario,
            "seed": seed,
            "short_description": "ThermoAgent generated artifact: %s" % path.name,
            "generated_timestamp": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "generating_command": "see results/README.md reproduction commands",
            "checksum": sha256_file(path),
        })
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else [
                "artifact_path", "artifact_type", "experiment_stage",
                "application", "method", "scenario", "seed",
                "short_description", "generated_timestamp",
                "generating_command", "checksum",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
