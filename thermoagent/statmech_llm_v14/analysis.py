"""Frozen cluster-level analysis for the V14 memory and quench study."""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from thermoagent.statmech_llm_v12.estimators import (
    block_time_reversal_kl,
    paired_cluster_bootstrap,
    shannon_entropy,
    time_shuffle_floor,
)

from .experiment import formal_panel_design, graph_for_panel
from .observables import (
    belief_action_lag,
    binder_cumulant,
    conditional_entropy_rate,
    conditional_memory_depths,
    dependence_bias_audit,
    integrated_correlation_time,
    irreversibility_sensitivity,
    macrostate_code,
    mean_reported_uncertainty,
    pairwise_information_summary,
    phase_path_length,
    plugin_entropy,
    recovery_time,
    signed_polygon_area,
    standardized_nominal_distance,
    standardized_nominal_fit,
    total_correlation,
)
from .workflow import (
    artifact_root,
    atomic_csv,
    atomic_json,
    execution_source_checksum,
    load_yaml,
    sha256_file,
    tree_digest,
    utc_now,
)


PRIMARY_Z_FEATURES = (
    "belief_magnetization",
    "action_magnetization",
    "belief_action_overlap",
    "reference_energy_per_agent",
    "energy_variance",
    "configuration_entropy",
    "entropy_rate",
    "total_correlation",
    "pairwise_mutual_information",
    "edge_mutual_information",
    "belief_susceptibility",
    "spatial_belief_correlation",
    "belief_disagreement",
)

OBSERVABLE_FAMILIES = {
    "order": ("belief_magnetization", "action_magnetization", "belief_action_overlap"),
    "uncertainty": ("mean_confidence", "mean_individual_uncertainty", "belief_disagreement"),
    "energy": ("reference_energy_per_agent", "energy_variance"),
    "entropy": (
        "configuration_entropy",
        "entropy_rate",
        "total_correlation",
        "pairwise_mutual_information",
        "edge_mutual_information",
    ),
    "response": ("belief_susceptibility", "binder_cumulant", "spatial_belief_correlation"),
    "temporal": ("integrated_correlation_time", "path_reversal_divergence", "belief_action_lag"),
}

INFORMATION_AUDIT_METRICS = (
    "total_correlation_raw",
    "total_correlation_null_mean",
    "total_correlation_bias_adjusted",
    "total_correlation_normalized_raw",
    "total_correlation_normalized_adjusted",
    "pairwise_mutual_information_raw",
    "pairwise_mutual_information_null_mean",
    "pairwise_mutual_information_bias_adjusted",
    "edge_mutual_information_raw",
    "edge_mutual_information_null_mean",
    "edge_mutual_information_bias_adjusted",
)


def _bits(value: object) -> np.ndarray:
    return np.asarray([int(item) for item in str(value).split(";")], dtype=int)


def _floats(value: object) -> np.ndarray:
    return np.asarray([float(item) for item in str(value).split(";")], dtype=float)


def _analysis_seed(token: str) -> int:
    return int(14300000 + sum((index + 1) * ord(value) for index, value in enumerate(token)) % 600000)


def _macro_codes(frame: pd.DataFrame, widths: Mapping[str, object]) -> np.ndarray:
    tuples = [
        macrostate_code(
            {
                key: float(getattr(row, key))
                for key in (
                    "belief_magnetization",
                    "action_magnetization",
                    "belief_action_overlap",
                    "reference_energy_per_agent",
                    "belief_disagreement",
                )
            },
            widths,  # type: ignore[arg-type]
        )
        for row in frame.itertuples()
    ]
    mapping = {value: index for index, value in enumerate(sorted(set(tuples)))}
    return np.asarray([mapping[value] for value in tuples], dtype=int)


def _mean_marginal_entropy(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=int)
    return float(
        np.mean(
            [
                shannon_entropy([np.sum(values[:, column] < 0), np.sum(values[:, column] > 0)])
                for column in range(values.shape[1])
            ]
        )
    )


def rolling_macrostates(
    frame: pd.DataFrame,
    panel: Mapping[str, object],
    protocol: Mapping[str, object],
    window_sweeps: int,
    information_null_replicates: int = 0,
    graph_override=None,
) -> List[Dict[str, object]]:
    """Aggregate online-compatible observables once per sweep."""

    n = int(panel["n_agents"])
    window_updates = int(window_sweeps) * n
    widths = protocol["analysis"]["entropy_coarse_graining"]  # type: ignore[index]
    all_codes = _macro_codes(frame, widths)
    beliefs = np.vstack([_bits(value) for value in frame["beliefs"]])
    actions = np.vstack([_bits(value) for value in frame["actions"]])
    output: List[Dict[str, object]] = []
    graph = graph_override if graph_override is not None else graph_for_panel(panel)
    for end in range(n - 1, len(frame), n):
        start = max(0, end - window_updates + 1)
        window = frame.iloc[start : end + 1]
        b = beliefs[start : end + 1]
        a = actions[start : end + 1]
        codes = all_codes[start : end + 1]
        energy = window["reference_energy_per_agent"].to_numpy(float)
        magnetization = window["belief_magnetization"].to_numpy(float)
        confidence = np.concatenate([_floats(value) for value in window["confidences"]])
        information = pairwise_information_summary(b, graph.adjacency)
        block = int(protocol["analysis"]["irreversibility"]["primary_block_length"])  # type: ignore[index]
        pseudocount = float(protocol["analysis"]["irreversibility"]["primary_pseudocount"])  # type: ignore[index]
        raw_irreversibility = block_time_reversal_kl(codes, block, pseudocount) if len(codes) > 2 * block else 0.0
        row = frame.iloc[end]
        joint = np.concatenate([b, a], axis=1)
        if int(information_null_replicates) > 0:
            dependence = dependence_bias_audit(
                joint,
                graph.adjacency,
                int(information_null_replicates),
                _analysis_seed(
                    "%s:w%d:s%d" % (panel["panel_id"], int(window_sweeps), int(end // n + 1))
                ),
            )
        else:
            raw_total = total_correlation(joint) if len(window) > 1 else 0.0
            dependence = {
                "total_correlation_raw": raw_total,
                "total_correlation_null_mean": float("nan"),
                "total_correlation_bias_adjusted": float("nan"),
                "total_correlation_normalized_raw": float("nan"),
                "total_correlation_normalized_adjusted": float("nan"),
                "pairwise_mutual_information_raw": information[
                    "mean_pairwise_belief_mutual_information"
                ],
                "pairwise_mutual_information_null_mean": float("nan"),
                "pairwise_mutual_information_bias_adjusted": float("nan"),
                "edge_mutual_information_raw": information[
                    "mean_edge_belief_mutual_information"
                ],
                "edge_mutual_information_null_mean": float("nan"),
                "edge_mutual_information_bias_adjusted": float("nan"),
                "marginal_entropy_sum": float("nan"),
                "null_replicates": 0.0,
            }
        output.append(
            {
                "cluster_id": panel["cluster_id"],
                "panel_id": panel["panel_id"],
                "disruption": panel["disruption"],
                "n_agents": n,
                "topology": panel["topology"],
                "sweep": int(end // n + 1),
                "phase": str(row["phase"]),
                "window_sweeps": int(window_sweeps),
                "belief_magnetization": float(row["belief_magnetization"]),
                "action_magnetization": float(row["action_magnetization"]),
                "belief_action_overlap": float(row["belief_action_overlap"]),
                "belief_disagreement": float(row["belief_disagreement"]),
                "reference_energy_per_agent": float(row["reference_energy_per_agent"]),
                "energy_variance": float(n * np.var(energy, ddof=1)) if len(energy) > 1 else 0.0,
                "configuration_entropy": plugin_entropy(codes.tolist()),
                "entropy_rate": conditional_entropy_rate(codes, 1, 0.5),
                "mean_individual_entropy": _mean_marginal_entropy(joint),
                "mean_individual_uncertainty": mean_reported_uncertainty(confidence),
                # The frozen V14 coordinate is retained unchanged.  The
                # versioned audit columns below quantify its finite-sample
                # floor without altering that primary coordinate.
                "total_correlation": total_correlation(joint) if len(window) > 1 else 0.0,
                **dependence,
                **information,
                "pairwise_mutual_information": information["mean_pairwise_belief_mutual_information"],
                "edge_mutual_information": information["mean_edge_belief_mutual_information"],
                "belief_susceptibility": float(n * np.var(magnetization, ddof=1)) if len(magnetization) > 1 else 0.0,
                "binder_cumulant": binder_cumulant(magnetization),
                "spatial_belief_correlation": float(window["spatial_belief_correlation"].mean()),
                "integrated_correlation_time": integrated_correlation_time(magnetization),
                "path_reversal_divergence": float(raw_irreversibility),
                "belief_action_lag": belief_action_lag(b, a, 1),
                "mean_confidence": float(np.mean(confidence)),
                "messages_delivered": int(window["messages_delivered"].sum()),
                "message_current": float(window["messages_delivered"].sum() / max(len(window), 1)),
                "wire_bytes": int(window["wire_bytes"].sum()),
                "corrupted_messages": int(window["message_corrupted"].sum()),
            }
        )
    return output


def panel_summary(
    frame: pd.DataFrame,
    macro: pd.DataFrame,
    panel: Mapping[str, object],
    protocol: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    n = int(panel["n_agents"])
    widths = protocol["analysis"]["entropy_coarse_graining"]  # type: ignore[index]
    codes = _macro_codes(frame, widths)
    irr = protocol["analysis"]["irreversibility"]  # type: ignore[index]
    sensitivity = irreversibility_sensitivity(
        codes,
        irr["block_lengths"],  # type: ignore[index]
        irr["pseudocounts"],  # type: ignore[index]
        int(irr["time_shuffle_replicates_per_panel"]),  # type: ignore[index]
        _analysis_seed(str(panel["panel_id"])),
    )
    primary = next(
        item
        for item in sensitivity
        if int(item["block_length"]) == int(irr["primary_block_length"])  # type: ignore[index]
        and np.isclose(float(item["pseudocount"]), float(irr["primary_pseudocount"]))  # type: ignore[index]
    )
    depths = conditional_memory_depths(codes, protocol["analysis"]["memory_depths"])  # type: ignore[index]
    beliefs = np.vstack([_bits(value) for value in frame["beliefs"]])
    actions = np.vstack([_bits(value) for value in frame["actions"]])
    base: Dict[str, object] = {
        "cluster_id": panel["cluster_id"],
        "panel_id": panel["panel_id"],
        "disruption": panel["disruption"],
        "n_agents": n,
        "sweeps": int(panel["sweeps"]),
        "attempted_updates": int(len(frame)),
        "valid_after_repair_fraction": float(frame["valid_after_repair"].mean()),
        "mean_abs_belief_magnetization": float(frame["belief_magnetization"].abs().mean()),
        "mean_abs_action_magnetization": float(frame["action_magnetization"].abs().mean()),
        "mean_belief_action_overlap": float(frame["belief_action_overlap"].mean()),
        "mean_disagreement": float(frame["belief_disagreement"].mean()),
        "configuration_entropy": plugin_entropy(codes.tolist()),
        "entropy_rate_nats_per_update": conditional_entropy_rate(codes, 1, 0.5),
        "total_correlation": total_correlation(np.concatenate([beliefs, actions], axis=1)),
        "mean_reference_energy_per_agent": float(frame["reference_energy_per_agent"].mean()),
        "energy_fluctuation_N_var_e": float(n * frame["reference_energy_per_agent"].var(ddof=1)),
        "belief_susceptibility": float(n * frame["belief_magnetization"].var(ddof=1)),
        "belief_integrated_correlation_time_updates": integrated_correlation_time(frame["belief_magnetization"]),
        "belief_action_lag": belief_action_lag(beliefs, actions, 1),
        "raw_block_divergence_nats_per_update": primary["raw_block_divergence_nats_per_update"],
        "shuffle_floor_nats_per_update": primary["shuffle_floor_nats_per_update"],
        "adjusted_pathwise_irreversibility_nats_per_update": primary["adjusted_irreversibility_nats_per_update"],
        "message_opportunities": int(frame["message_opportunities"].sum()),
        "messages_delivered": int(frame["messages_delivered"].sum()),
        "wire_bytes": int(frame["wire_bytes"].sum()),
        "privacy_mutations": int(frame["unrelated_peer_private_mutations"].sum()),
        "prompt_tokens": int(frame["prompt_tokens"].sum()),
        "generated_tokens": int(frame["generated_tokens"].sum()),
        "latency_seconds": float(frame["latency_seconds"].sum()),
    }
    for depth, value in depths.items():
        base["conditional_memory_depth_%d" % depth] = value
    sensitivity_rows = [{"cluster_id": panel["cluster_id"], "panel_id": panel["panel_id"], "disruption": panel["disruption"], **item} for item in sensitivity]
    depth_rows = [
        {
            "cluster_id": panel["cluster_id"],
            "panel_id": panel["panel_id"],
            "disruption": panel["disruption"],
            "memory_depth": depth,
            "conditional_mutual_information_nats": value,
        }
        for depth, value in depths.items()
    ]
    return base, sensitivity_rows, depth_rows


def _impute_training(
    training: np.ndarray, testing: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(training, dtype=float).copy()
    test = np.asarray(testing, dtype=float).copy()
    medians = np.nanmedian(train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    for column in range(train.shape[1]):
        train[~np.isfinite(train[:, column]), column] = medians[column]
        test[~np.isfinite(test[:, column]), column] = medians[column]
    return train, test, medians


def _nominal_training_selection(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    nominal = frame[frame["disruption"] == "nominal"]
    if window == "all_nominal":
        return nominal
    baseline = nominal[nominal["phase"] == "baseline"]
    if window == "baseline_15":
        return baseline
    count = int(window.rsplit("_", 1)[-1])
    return baseline.groupby("panel_id", group_keys=False).tail(count)


def nominal_distance_analysis(
    macro: pd.DataFrame,
    protocol: Mapping[str, object],
    *,
    return_thresholds: bool = False,
    include_single_observable_ablations: bool = True,
):
    """Fit each nominal manifold without using the held-out cluster.

    The optional third return value is the explicit cluster-specific map of
    training-nominal 95th-percentile thresholds.  It is consumed by recovery
    analysis; a held-out panel baseline is never used to fit its own threshold.
    The default two-value return is retained for backwards compatibility with
    the historical analysis API.
    """

    settings = protocol["analysis"]["nominal_distance"]  # type: ignore[index]
    primary_estimator = str(settings["primary_estimator"])  # type: ignore[index]
    primary_ridge = float(settings["primary_ridge_fraction"])  # type: ignore[index]
    output = macro.copy()
    output["macrostate_distance"] = np.nan
    output["training_nominal_threshold_95"] = np.nan
    robustness: List[Dict[str, object]] = []
    thresholds: Dict[str, float] = {}
    variants: List[Tuple[str, float, str, Tuple[str, ...], str, str, str]] = []
    for estimator in settings["estimators"]:  # type: ignore[index]
        ridges = settings["ridge_fractions"] if estimator in ("shrinkage", "robust") else [0.0]  # type: ignore[index]
        for ridge in ridges:
            for nominal_window in protocol["analysis"]["nominal_fit_windows"]:  # type: ignore[index]
                variants.append(
                    (
                        str(estimator),
                        float(ridge),
                        str(nominal_window),
                        PRIMARY_Z_FEATURES,
                        "all",
                        "none",
                        "none",
                    )
                )
    for family, values in OBSERVABLE_FAMILIES.items():
        retained = tuple(value for value in PRIMARY_Z_FEATURES if value not in values)
        variants.append(
            (
                primary_estimator,
                primary_ridge,
                "all_nominal",
                retained,
                "without_family_%s" % family,
                str(family),
                "none",
            )
        )
    if include_single_observable_ablations:
        for feature in PRIMARY_Z_FEATURES:
            retained = tuple(value for value in PRIMARY_Z_FEATURES if value != feature)
            variants.append(
                (
                    primary_estimator,
                    primary_ridge,
                    "all_nominal",
                    retained,
                    "without_observable_%s" % feature,
                    "none",
                    str(feature),
                )
            )
    for estimator, ridge, nominal_window, features, ablation, deleted_family, deleted_feature in variants:
        for cluster in sorted(output["cluster_id"].unique()):
            training_pool = _nominal_training_selection(output[output["cluster_id"] != cluster], nominal_window)
            test = output[output["cluster_id"] == cluster]
            training_clusters = sorted(str(value) for value in training_pool["cluster_id"].unique())
            if str(cluster) in training_clusters:
                raise AssertionError("held-out cluster entered nominal fitting")
            train_values, test_values, _ = _impute_training(
                training_pool[list(features)].to_numpy(float), test[list(features)].to_numpy(float)
            )
            try:
                fitted = standardized_nominal_fit(train_values, estimator, ridge)
                distances = standardized_nominal_distance(test_values, fitted)
                train_distances = standardized_nominal_distance(train_values, fitted)
                failed = ""
            except Exception as error:  # robust covariance is an explicitly optional sensitivity
                distances = np.full(len(test), np.nan)
                train_distances = np.full(len(training_pool), np.nan)
                failed = type(error).__name__
            primary = (
                estimator == primary_estimator
                and np.isclose(ridge, primary_ridge)
                and nominal_window == "all_nominal"
                and ablation == "all"
            )
            if primary:
                output.loc[test.index, "macrostate_distance"] = distances
            threshold = float(np.nanquantile(train_distances, 0.95)) if np.any(np.isfinite(train_distances)) else float("nan")
            if primary:
                thresholds[str(cluster)] = threshold
                output.loc[test.index, "training_nominal_threshold_95"] = threshold
            for disruption in sorted(test["disruption"].unique()):
                indices = np.flatnonzero(test["disruption"].to_numpy() == disruption)
                selected = distances[indices]
                robustness.append(
                    {
                        "cluster_id": cluster,
                        "disruption": disruption,
                        "estimator": estimator,
                        "ridge_fraction": ridge,
                        "nominal_fit_window": nominal_window,
                        "rolling_window_sweeps": int(output["window_sweeps"].iloc[0]),
                        "ablation": ablation,
                        "deleted_family": deleted_family,
                        "deleted_observable": deleted_feature,
                        "feature_count": len(features),
                        "training_clusters": json.dumps(training_clusters),
                        "held_out_cluster_excluded": True,
                        "maximum_distance": float(np.nanmax(selected)) if np.any(np.isfinite(selected)) else float("nan"),
                        "mean_distance": float(np.nanmean(selected)) if np.any(np.isfinite(selected)) else float("nan"),
                        "nominal_threshold_95": threshold,
                        "failure": failed,
                    }
                )
    if output["macrostate_distance"].isna().any():
        raise RuntimeError("primary nominal distance contains missing values")
    if output["training_nominal_threshold_95"].isna().any() or set(thresholds) != {
        str(value) for value in output["cluster_id"].unique()
    }:
        raise RuntimeError("primary training-nominal thresholds are incomplete")
    if return_thresholds:
        return output, pd.DataFrame(robustness), thresholds
    return output, pd.DataFrame(robustness)


def disruption_summaries(
    macro: pd.DataFrame,
    protocol: Mapping[str, object],
    training_nominal_thresholds: Mapping[str, float],
) -> pd.DataFrame:
    final_window = int(protocol["analysis"]["recovery"]["final_window_sweeps"])  # type: ignore[index]
    rows: List[Dict[str, object]] = []
    for panel_id, group in macro.groupby("panel_id"):
        group = group.sort_values("sweep")
        baseline = group[group["phase"] == "baseline"]
        disruption = group[group["phase"] == "disruption"]
        recovery = group[group["phase"] == "recovery"]
        cluster = str(group["cluster_id"].iloc[0])
        if cluster not in training_nominal_thresholds:
            raise KeyError("missing training-nominal threshold for held-out cluster %s" % cluster)
        threshold = float(training_nominal_thresholds[cluster])
        if not np.isfinite(threshold):
            raise ValueError("nonfinite training-nominal threshold for cluster %s" % cluster)
        energy_baseline = float(baseline["reference_energy_per_agent"].mean())
        entropy_baseline = float(baseline["configuration_entropy"].mean())
        early_peak = float(recovery["macrostate_distance"].max())
        final_mean = float(recovery["macrostate_distance"].tail(final_window).mean())
        early_fixed = float(recovery["macrostate_distance"].head(final_window).mean())
        rows.append(
            {
                "cluster_id": group["cluster_id"].iloc[0],
                "panel_id": panel_id,
                "disruption": group["disruption"].iloc[0],
                "baseline_mean_distance": float(baseline["macrostate_distance"].mean()),
                "maximum_disruption_distance": float(disruption["macrostate_distance"].max()),
                "maximum_post_quench_distance": float(pd.concat([disruption, recovery])["macrostate_distance"].max()),
                "integrated_disruption_distance": float(disruption["macrostate_distance"].sum()),
                "counter_quench_peak_distance": early_peak,
                "final_five_sweep_mean_distance": final_mean,
                "recovery_drop_estimand": early_peak - final_mean,
                "fixed_early_five_sweep_mean_distance": early_fixed,
                "fixed_early_minus_late_recovery_distance": early_fixed - final_mean,
                "recovery_time_sweeps": recovery_time(
                    recovery["macrostate_distance"], threshold, int(protocol["analysis"]["recovery"]["consecutive_sweeps_within_threshold"])  # type: ignore[index]
                ),
                "time_outside_baseline_sweeps": int(np.sum(group["macrostate_distance"] > threshold)),
                "entropy_overshoot": float(disruption["configuration_entropy"].max() - entropy_baseline),
                "energy_overshoot": float(np.max(np.abs(disruption["reference_energy_per_agent"] - energy_baseline))),
                "macrostate_path_length": phase_path_length(group[list(PRIMARY_Z_FEATURES)].to_numpy(float)),
                "energy_entropy_signed_loop_area": signed_polygon_area(
                    group["reference_energy_per_agent"], group["configuration_entropy"]
                ),
                "belief_action_signed_loop_area": signed_polygon_area(
                    group["belief_magnetization"], group["action_magnetization"]
                ),
                "baseline_threshold_95": threshold,
                "threshold_source": "leave_one_cluster_out_training_nominal",
            }
        )
    return pd.DataFrame(rows)


def information_estimator_contrasts(
    macrostates: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare field and nominal disruption-period dependence by cluster."""

    rows: List[Dict[str, object]] = []
    for window, window_group in macrostates.groupby("window_sweeps"):
        for cluster, cluster_group in window_group.groupby("cluster_id"):
            for metric in INFORMATION_AUDIT_METRICS:
                field = cluster_group[
                    (cluster_group["disruption"] == "field_reversal")
                    & (cluster_group["phase"] == "disruption")
                ][metric].to_numpy(float)
                nominal = cluster_group[
                    (cluster_group["disruption"] == "nominal")
                    & (cluster_group["phase"] == "disruption")
                ][metric].to_numpy(float)
                if field.size == 0 or nominal.size == 0:
                    raise RuntimeError("information contrast lacks a matched disruption period")
                rows.append(
                    {
                        "window_sweeps": int(window),
                        "cluster_id": str(cluster),
                        "metric": metric,
                        "field_reversal_mean": float(np.mean(field)),
                        "nominal_mean": float(np.mean(nominal)),
                        "field_minus_nominal": float(np.mean(field) - np.mean(nominal)),
                    }
                )
    cluster_rows = pd.DataFrame(rows)
    summaries: List[Dict[str, object]] = []
    for (window, metric), group in cluster_rows.groupby(["window_sweeps", "metric"]):
        wrapped = {
            str(row.cluster_id): [float(row.field_minus_nominal)] for row in group.itertuples()
        }
        effect = paired_cluster_bootstrap(
            wrapped,
            10000,
            _analysis_seed("information:%s:%s" % (window, metric)),
        )
        summaries.append(
            {
                "window_sweeps": int(window),
                "metric": str(metric),
                **effect,
                "independent_unit": "graph_environment_cluster",
            }
        )
    return cluster_rows, pd.DataFrame(summaries)


def phase_feature_table(macro: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    available = sorted(set(sum((list(values) for values in OBSERVABLE_FAMILIES.values()), [])))
    for panel_id, group in macro.groupby("panel_id"):
        baseline = group[group["phase"] == "baseline"]
        disruption = group[group["phase"] == "disruption"]
        recovery = group[group["phase"] == "recovery"]
        row: Dict[str, object] = {
            "panel_id": panel_id,
            "cluster_id": str(group["cluster_id"].iloc[0]),
            "label": str(group["disruption"].iloc[0]),
        }
        for variable in available:
            row["delta_%s" % variable] = float(disruption[variable].mean() - baseline[variable].mean())
            row["recovery_%s" % variable] = float(recovery[variable].mean() - baseline[variable].mean())
            row["peak_%s" % variable] = float(np.max(np.abs(disruption[variable] - baseline[variable].mean())))
        rows.append(row)
    return pd.DataFrame(rows)


def _representation_columns(protocol: Mapping[str, object], representation: str) -> List[str]:
    variables = protocol["analysis"]["representations"][representation]  # type: ignore[index]
    return [prefix + str(variable) for variable in variables for prefix in ("delta_", "recovery_", "peak_")]


def representation_cv(
    features: pd.DataFrame,
    protocol: Mapping[str, object],
    *,
    collect_coefficients: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: List[Dict[str, object]] = []
    folds: List[Dict[str, object]] = []
    coefficients: List[Dict[str, object]] = []
    labels = sorted(features["label"].unique())
    settings = protocol["analysis"]["representation_model"]  # type: ignore[index]
    for representation in protocol["analysis"]["representations"]:  # type: ignore[index]
        columns = _representation_columns(protocol, str(representation))
        for cluster in sorted(features["cluster_id"].unique()):
            train = features[features["cluster_id"] != cluster]
            test = features[features["cluster_id"] == cluster]
            train_values, test_values, _ = _impute_training(
                train[columns].to_numpy(float), test[columns].to_numpy(float)
            )
            scaler = StandardScaler().fit(train_values)
            model = LogisticRegression(
                C=float(settings["regularization_C"]),  # type: ignore[index]
                max_iter=4000,
                random_state=1414,
                solver="lbfgs",
            ).fit(scaler.transform(train_values), train["label"].astype(str))
            probability = model.predict_proba(scaler.transform(test_values))
            predicted = model.classes_[np.argmax(probability, axis=1)]
            truth = test["label"].astype(str).to_numpy()
            brier = float(
                np.mean(
                    [
                        np.sum((probability[index] - np.asarray([value == truth[index] for value in model.classes_], dtype=float)) ** 2)
                        for index in range(len(test))
                    ]
                )
            )
            fold_predictions = []
            for index, (_, item) in enumerate(test.iterrows()):
                truth_index = list(model.classes_).index(str(item["label"]))
                record = {
                    "representation": representation,
                    "held_out_cluster": cluster,
                    "panel_id": item["panel_id"],
                    "truth": item["label"],
                    "prediction": predicted[index],
                    "correct": int(predicted[index] == item["label"]),
                    "truth_probability": float(probability[index, truth_index]),
                }
                predictions.append(record)
                fold_predictions.append(record)
            folds.append(
                {
                    "representation": representation,
                    "held_out_cluster": cluster,
                    "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                    "macro_recall": float(recall_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
                    "multiclass_log_loss": float(log_loss(truth, probability, labels=list(model.classes_))),
                    "multiclass_brier": brier,
                    "test_panels": int(len(test)),
                }
            )
            if collect_coefficients:
                for class_index, label in enumerate(model.classes_):
                    for column, coefficient in zip(columns, model.coef_[class_index]):
                        coefficients.append(
                            {
                                "representation": representation,
                                "held_out_cluster": cluster,
                                "class": label,
                                "feature": column,
                                "standardized_coefficient": float(coefficient),
                            }
                        )
    return pd.DataFrame(predictions), pd.DataFrame(folds), pd.DataFrame(coefficients)


def _one_cluster_preserving_permutation(
    replicate: int,
    features: pd.DataFrame,
    protocol: Mapping[str, object],
    seed: int,
) -> Dict[str, float]:
    permuted = features.copy()
    rng = np.random.default_rng(int(seed) + 104729 * int(replicate))
    for cluster in sorted(permuted["cluster_id"].unique()):
        selected = permuted["cluster_id"] == cluster
        original = permuted.loc[selected, "label"].astype(str).to_numpy()
        shuffled = rng.permutation(original)
        if sorted(original.tolist()) != sorted(shuffled.tolist()):
            raise AssertionError("cluster-preserving permutation changed a panel set")
        permuted.loc[selected, "label"] = shuffled
    _, folds, _ = representation_cv(permuted, protocol, collect_coefficients=False)
    scores = folds.groupby("representation")["balanced_accuracy"].mean().to_dict()
    full = float(scores["full_statmech"])
    order = float(scores["order_only"])
    simple = float(scores["simple_uncertainty"])
    return {
        "replicate": int(replicate),
        "full_statmech_balanced_accuracy": full,
        "order_only_balanced_accuracy": order,
        "simple_uncertainty_balanced_accuracy": simple,
        "full_minus_order_only_balanced_accuracy": full - order,
        "full_minus_simple_uncertainty_balanced_accuracy": full - simple,
    }


def cluster_preserving_permutation_analysis(
    features: pd.DataFrame,
    protocol: Mapping[str, object],
    *,
    replicates: int,
    seed: int,
    workers: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the complete LOCO classifier pipeline under cluster permutations."""

    count = int(replicates)
    if count < 1:
        raise ValueError("permutation analysis requires at least one replicate")
    observed_predictions, observed_folds, _ = representation_cv(
        features, protocol, collect_coefficients=False
    )
    del observed_predictions
    observed_scores = observed_folds.groupby("representation")["balanced_accuracy"].mean().to_dict()
    observed = {
        "full_statmech_balanced_accuracy": float(observed_scores["full_statmech"]),
        "order_only_balanced_accuracy": float(observed_scores["order_only"]),
        "simple_uncertainty_balanced_accuracy": float(observed_scores["simple_uncertainty"]),
    }
    observed["full_minus_order_only_balanced_accuracy"] = (
        observed["full_statmech_balanced_accuracy"] - observed["order_only_balanced_accuracy"]
    )
    observed["full_minus_simple_uncertainty_balanced_accuracy"] = (
        observed["full_statmech_balanced_accuracy"]
        - observed["simple_uncertainty_balanced_accuracy"]
    )
    jobs = max(1, int(workers))
    rows = Parallel(n_jobs=jobs, prefer="processes", batch_size=10)(
        delayed(_one_cluster_preserving_permutation)(index, features, protocol, int(seed))
        for index in range(count)
    )
    null = pd.DataFrame(rows).sort_values("replicate").reset_index(drop=True)
    summaries: List[Dict[str, object]] = []
    for metric, observed_value in observed.items():
        values = null[metric].to_numpy(float)
        summaries.append(
            {
                "metric": metric,
                "observed": float(observed_value),
                "null_mean": float(np.mean(values)),
                "null_standard_deviation": float(np.std(values, ddof=1)),
                "null_q025": float(np.quantile(values, 0.025)),
                "null_q975": float(np.quantile(values, 0.975)),
                "upper_tail_empirical_p": float(
                    (1.0 + np.sum(values >= float(observed_value) - 1e-15)) / (count + 1.0)
                ),
                "replicates": count,
                "seed": int(seed),
                "permutation_unit": "condition_labels_within_graph_environment_cluster",
                "pipeline_refit_per_permutation": True,
            }
        )
    return null, pd.DataFrame(summaries)


def _bootstrap_effect(values: Mapping[str, Sequence[float]], seed: int) -> Dict[str, float]:
    if not values:
        raise ValueError("effect requires independent clusters")
    return paired_cluster_bootstrap(values, 10000, seed)


def exact_sign_flip_p(values: Sequence[float], direction: int = 1) -> float:
    data = np.asarray(values, dtype=float) * int(direction)
    if data.size > 20:
        raise ValueError("exact sign-flip enumeration is intentionally bounded")
    observed = float(np.mean(data))
    null = [float(np.mean(data * np.asarray(signs))) for signs in itertools.product((-1.0, 1.0), repeat=data.size)]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def primary_hypotheses(
    disruption: pd.DataFrame,
    folds: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cluster_values: Dict[str, Dict[str, float]] = {"H2": {}, "H3": {}, "H4": {}}
    for cluster, group in disruption.groupby("cluster_id"):
        field = group[group["disruption"] == "field_reversal"].iloc[0]
        nominal = group[group["disruption"] == "nominal"].iloc[0]
        cluster_values["H2"][str(cluster)] = float(field["maximum_post_quench_distance"] - nominal["maximum_post_quench_distance"])
        cluster_values["H3"][str(cluster)] = float(field["recovery_drop_estimand"])
    for cluster, group in folds.groupby("held_out_cluster"):
        full = float(group[group["representation"] == "full_statmech"]["balanced_accuracy"].iloc[0])
        order = float(group[group["representation"] == "order_only"]["balanced_accuracy"].iloc[0])
        cluster_values["H4"][str(cluster)] = full - order
    rows: List[Dict[str, object]] = []
    raw_p: List[float] = []
    metadata = {
        "H2": ("field_reversal_minus_nominal_maximum_post_quench_distance", "regularized_distance_units"),
        "H3": ("early_recovery_peak_minus_final_five_sweep_mean_distance", "regularized_distance_units"),
        "H4": ("full_minus_order_only_balanced_accuracy", "proportion"),
    }
    for index, hypothesis in enumerate(("H2", "H3", "H4")):
        values = cluster_values[hypothesis]
        wrapped = {cluster: [value] for cluster, value in values.items()}
        effect = _bootstrap_effect(wrapped, 14400000 + index)
        pvalue = exact_sign_flip_p(list(values.values()), 1)
        raw_p.append(pvalue)
        rows.append(
            {
                "hypothesis": hypothesis,
                "estimand": metadata[hypothesis][0],
                "unit": metadata[hypothesis][1],
                **effect,
                "exact_one_sided_sign_flip_p": pvalue,
                "cluster_values": json.dumps(values, sort_keys=True),
            }
        )
    adjusted = holm_adjust(raw_p)
    dispositions: Dict[str, object] = {}
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
        hypothesis = str(row["hypothesis"])
        numerical_criterion = bool(value < 0.05 and float(row["estimate"]) > 0.0)
        valid_directional_test = hypothesis != "H3"
        inferential_support = bool(numerical_criterion and valid_directional_test)
        row["frozen_numerical_criterion_met"] = numerical_criterion
        row["valid_directional_test"] = valid_directional_test
        row["inferential_support"] = inferential_support
        row["trajectory_evidence_consistent_with_recovery"] = bool(
            hypothesis == "H3" and float(row["estimate"]) > 0.0
        )
        dispositions[hypothesis] = {
            # Kept as an explicit false compatibility guard so no downstream
            # report can silently resurrect the historical H3 disposition.
            "supported": inferential_support,
            "frozen_numerical_criterion_met": numerical_criterion,
            "valid_directional_test": valid_directional_test,
            "inferential_support": inferential_support,
            "trajectory_evidence_consistent_with_recovery": bool(
                hypothesis == "H3" and float(row["estimate"]) > 0.0
            ),
            "estimate": row["estimate"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "exact_one_sided_sign_flip_p": row["exact_one_sided_sign_flip_p"],
            "holm_adjusted_p": value,
        }
    return pd.DataFrame(rows), dispositions


def memory_discovery_replication(repository: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    v12 = json.loads(
        (repository / "results/llm_agent_statmech_v12/statistics/primary_results.json").read_text(encoding="utf-8")
    )["memory_effects"]["adjusted_block_kl_nats_per_update"]
    v13_table = pd.read_csv(repository / "results/collective_agent_statmech_v13/tables/hypothesis_effects.csv")
    v13 = v13_table[(v13_table["hypothesis"] == "H3")].iloc[0]
    rows = [
        {
            "study": "V12_discovery",
            "role": "discovery",
            "estimate": float(v12["estimate"]),
            "ci_low": float(v12["ci_low"]),
            "ci_high": float(v12["ci_high"]),
            "independent_clusters": int(v12["independent_clusters"]),
        },
        {
            "study": "V13_replication",
            "role": "prospective_replication",
            "estimate": float(v13["estimate"]),
            "ci_low": float(v13["ci_low"]),
            "ci_high": float(v13["ci_high"]),
            "independent_clusters": int(v13["independent_clusters"]),
        },
    ]
    for row in rows:
        row["approximate_standard_error"] = (row["ci_high"] - row["ci_low"]) / (2.0 * 1.96)
    weights = np.asarray([1.0 / row["approximate_standard_error"] ** 2 for row in rows])
    effects = np.asarray([row["estimate"] for row in rows])
    combined = float(np.sum(weights * effects) / np.sum(weights))
    standard_error = float(math.sqrt(1.0 / np.sum(weights)))
    rows.append(
        {
            "study": "descriptive_fixed_effect_synthesis",
            "role": "descriptive_not_prospectively_pooled",
            "estimate": combined,
            "ci_low": combined - 1.96 * standard_error,
            "ci_high": combined + 1.96 * standard_error,
            "independent_clusters": int(sum(row["independent_clusters"] for row in rows)),
            "approximate_standard_error": standard_error,
        }
    )
    return pd.DataFrame(rows), {
        "v12_discovery_estimate": rows[0]["estimate"],
        "v13_replication_estimate": rows[1]["estimate"],
        "descriptive_fixed_effect_estimate": combined,
        "descriptive_fixed_effect_ci": [combined - 1.96 * standard_error, combined + 1.96 * standard_error],
        "warning": "V12 discovery and V13 replication are reported separately; synthesis is descriptive.",
    }


def _legacy_memory_sensitivity(repository: Path, protocol: Mapping[str, object]) -> pd.DataFrame:
    roots = [
        Path(os.environ.get("THERMO_V13_ARTIFACT_ROOT", "/workspace/ThermoAgent-v13-artifacts")),
        Path(os.environ.get("THERMO_V12_ARTIFACT_ROOT", "/workspace/ThermoAgent-v12-artifacts")),
    ]
    rows: List[Dict[str, object]] = []
    for study, root in zip(("V13_replication", "V12_discovery"), roots):
        candidates = sorted((root / "formal/panels").glob("*memory*.csv")) if (root / "formal/panels").exists() else []
        for path in candidates:
            frame = pd.read_csv(path)
            widths = protocol["analysis"]["entropy_coarse_graining"]  # type: ignore[index]
            try:
                codes = _macro_codes(frame, widths)
            except Exception:
                continue
            for item in irreversibility_sensitivity(codes, [2, 3, 4], [0.5], 200, _analysis_seed(path.name)):
                rows.append(
                    {
                        "study": study,
                        "panel_id": path.stem,
                        "regime": "persistent_memory" if "persistent" in path.stem else "markovized",
                        **item,
                        **{"conditional_memory_depth_%d" % key: value for key, value in conditional_memory_depths(codes, [1, 2, 3]).items()},
                    }
                )
    if not rows:
        return pd.DataFrame(
            [{"study": "external_raw_unavailable", "panel_id": "not_computed", "regime": "unknown", "block_length": 3, "pseudocount": 0.5, "raw_block_divergence_nats_per_update": float("nan"), "shuffle_floor_nats_per_update": float("nan"), "adjusted_irreversibility_nats_per_update": float("nan")}]
        )
    return pd.DataFrame(rows)


def network_snapshot_tables(
    root: Path,
    protocol: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    for panel in formal_panel_design(protocol):
        if panel["cluster_id"] != "V14Q_g0":
            continue
        frame = pd.read_csv(root / "panels" / (str(panel["panel_id"]) + ".csv"))
        graph = graph_for_panel(panel)
        for phase in ("baseline", "disruption", "recovery"):
            phase_frame = frame[frame["phase"] == phase]
            selected = phase_frame.iloc[-1]
            beliefs = _bits(selected["beliefs"])
            actions = _bits(selected["actions"])
            confidence = _floats(selected["confidences"])
            memories = _bits(selected["memory_states"])
            for node in range(int(panel["n_agents"])):
                nodes.append(
                    {
                        "panel_id": panel["panel_id"],
                        "disruption": panel["disruption"],
                        "phase": phase,
                        "node": node,
                        "community": 0 if node < int(panel["n_agents"]) // 2 else 1,
                        "belief": int(beliefs[node]),
                        "action": int(actions[node]),
                        "confidence": float(confidence[node]),
                        "uncertainty": float(-confidence[node] * math.log(max(confidence[node], 1e-12)) - (1.0 - confidence[node]) * math.log(max(1.0 - confidence[node], 1e-12))),
                        "memory": int(memories[node]),
                    }
                )
            counts: Dict[Tuple[int, int], int] = {}
            for item in phase_frame.itertuples():
                if int(item.recipient) >= 0:
                    key = (int(item.scheduled_agent), int(item.recipient))
                    counts[key] = counts.get(key, 0) + 1
            for source, target in zip(*np.nonzero(graph.adjacency)):
                cross = (int(source) < graph.n_agents // 2) != (int(target) < graph.n_agents // 2)
                active = not (panel["disruption"] == "network_partition" and phase == "disruption" and cross)
                edges.append(
                    {
                        "panel_id": panel["panel_id"],
                        "disruption": panel["disruption"],
                        "phase": phase,
                        "source": int(source),
                        "target": int(target),
                        "active": int(active),
                        "message_count": int(counts.get((int(source), int(target)), 0)),
                        "cross_community": int(cross),
                    }
                )
    return pd.DataFrame(nodes), pd.DataFrame(edges)


def _archive_historical_derived_results(result: Path, paper: Path) -> Dict[str, object]:
    """Preserve the committed V14 derived disposition before correction."""

    archive = result / "corrections/v14_scientific_audit_v1_1/historical_v1_0"
    archive.mkdir(parents=True, exist_ok=True)
    candidates = {
        "primary_results.json": result / "statistics/primary_results.json",
        "hypothesis_effects.csv": result / "tables/hypothesis_effects.csv",
        "quench_recovery.csv": result / "tables/quench_recovery.csv",
        "README.md": result / "README.md",
        "CLAIMS_MATRIX.md": result / "CLAIMS_MATRIX.md",
        "results_macros.tex": paper / "results_macros.tex",
    }
    rows: List[Dict[str, object]] = []
    for name, source in candidates.items():
        destination = archive / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
        if destination.exists():
            rows.append(
                {
                    "file": name,
                    "bytes": int(destination.stat().st_size),
                    "sha256": sha256_file(destination),
                }
            )
    atomic_json(
        {
            "archive_role": "immutable copy of committed V14 derived reporting before audit correction",
            "historical_commit": "103e4c4598ecc26a98c37a8d03ee3663f9be1070",
            "files": rows,
        },
        archive / "manifest.json",
    )
    return {"path": str(archive), "files": rows}


def analyze_formal(repository: Path) -> Dict[str, object]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    repository = Path(repository).resolve()
    protocol_path = repository / "configs/statmech_v14/protocol_frozen.yaml"
    audit_path = repository / "configs/statmech_v14/scientific_audit_v1.1.yaml"
    protocol = load_yaml(protocol_path)
    audit = load_yaml(audit_path)
    root = artifact_root() / "formal"
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion["status"] != "complete":
        raise RuntimeError("V14 formal execution is incomplete")
    historical_formal_digest = tree_digest(root)
    primary_window = int(protocol["analysis"]["primary_window_sweeps"])  # type: ignore[index]
    windows = sorted(
        {
            primary_window,
            *[int(value) for value in protocol["analysis"]["alternative_window_sweeps"]],  # type: ignore[index]
        }
    )
    information_null_replicates = int(
        audit["information_bias_audit"]["null_replicates_per_window"]  # type: ignore[index]
    )
    macro_rows_by_window: Dict[int, List[Dict[str, object]]] = {window: [] for window in windows}
    panel_rows: List[Dict[str, object]] = []
    irr_rows: List[Dict[str, object]] = []
    memory_depth_rows: List[Dict[str, object]] = []
    for panel in formal_panel_design(protocol):
        path = root / "panels" / (str(panel["panel_id"]) + ".csv")
        frame = pd.read_csv(path)
        if len(frame) != int(panel["n_agents"]) * int(panel["sweeps"]):
            raise RuntimeError("formal trajectory row mismatch: %s" % path.name)
        panel_macros: Dict[int, pd.DataFrame] = {}
        for window in windows:
            rolling = rolling_macrostates(
                frame,
                panel,
                protocol,
                int(window),
                information_null_replicates=information_null_replicates,
            )
            macro_rows_by_window[window].extend(rolling)
            panel_macros[window] = pd.DataFrame(rolling)
        summary, sensitivity, depths = panel_summary(
            frame, panel_macros[primary_window], panel, protocol
        )
        panel_rows.append(summary)
        irr_rows.extend(sensitivity)
        memory_depth_rows.extend(depths)

    distance_frames: List[pd.DataFrame] = []
    robustness_frames: List[pd.DataFrame] = []
    primary_macro: pd.DataFrame | None = None
    primary_thresholds: Dict[str, float] = {}
    for window in windows:
        window_macro = pd.DataFrame(macro_rows_by_window[window])
        corrected, robustness, thresholds = nominal_distance_analysis(
            window_macro,
            protocol,
            return_thresholds=True,
            include_single_observable_ablations=True,
        )
        distance_frames.append(corrected)
        robustness_frames.append(robustness)
        if window == primary_window:
            primary_macro = corrected
            primary_thresholds = thresholds
    if primary_macro is None:
        raise AssertionError("primary rolling representation was not generated")
    all_windows = pd.concat(distance_frames, ignore_index=True)
    robustness = pd.concat(robustness_frames, ignore_index=True)
    alternative_windows = all_windows[all_windows["window_sweeps"] != primary_window].copy()
    disruption = disruption_summaries(primary_macro, protocol, primary_thresholds)
    phase_features = phase_feature_table(primary_macro)
    predictions, folds, coefficients = representation_cv(phase_features, protocol)
    permutation_workers = int(os.environ.get("THERMO_V14_PERMUTATION_WORKERS", "1"))
    permutation_null, permutation_summary = cluster_preserving_permutation_analysis(
        phase_features,
        protocol,
        replicates=int(audit["permutation_analysis"]["replicates"]),  # type: ignore[index]
        seed=int(audit["permutation_analysis"]["seed"]),  # type: ignore[index]
        workers=permutation_workers,
    )
    information_clusters, information_summary = information_estimator_contrasts(all_windows)
    hypotheses, dispositions = primary_hypotheses(disruption, folds)
    memory, memory_summary = memory_discovery_replication(repository)
    legacy_sensitivity = _legacy_memory_sensitivity(repository, protocol)
    nodes, edges = network_snapshot_tables(root, protocol)
    result = repository / "results/collective_agent_statmech_v14"
    tables = result / "tables"
    statistics = result / "statistics"
    source = result / "figures/source_data"
    for directory in (
        tables,
        statistics,
        source,
        result / "reproducibility",
        result / "logs",
        result / "corrections/v14_scientific_audit_v1_1",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    historical_archive = _archive_historical_derived_results(
        result, repository / "paper/jstat_v14"
    )
    information_columns = [
        "cluster_id",
        "panel_id",
        "disruption",
        "sweep",
        "phase",
        "window_sweeps",
        *INFORMATION_AUDIT_METRICS,
        "marginal_entropy_sum",
        "null_replicates",
    ]
    outputs = {
        "panel_statistics.csv": pd.DataFrame(panel_rows),
        "macrostate_trajectories.csv": primary_macro,
        "macrostate_trajectories_alternative_windows.csv": alternative_windows,
        "macrostate_trajectories_all_windows.csv": all_windows,
        "quench_recovery.csv": disruption,
        "macrostate_distance_robustness.csv": robustness,
        "representation_features.csv": phase_features,
        "representation_predictions.csv": predictions,
        "representation_cv.csv": folds,
        "representation_coefficients.csv": coefficients,
        "representation_permutation_null.csv": permutation_null,
        "representation_permutation_summary.csv": permutation_summary,
        "information_estimator_audit.csv": all_windows[information_columns],
        "information_estimator_cluster_contrasts.csv": information_clusters,
        "information_estimator_contrast_summary.csv": information_summary,
        "hypothesis_effects.csv": hypotheses,
        "irreversibility_sensitivity.csv": pd.DataFrame(irr_rows),
        "conditional_memory_depth.csv": pd.DataFrame(memory_depth_rows),
        "memory_discovery_replication.csv": memory,
        "legacy_memory_block_sensitivity.csv": legacy_sensitivity,
        "network_snapshot_nodes.csv": nodes,
        "network_snapshot_edges.csv": edges,
    }
    for name, frame in outputs.items():
        atomic_csv(frame.to_dict("records"), tables / name)
    field_recovery = disruption[disruption["disruption"] == "field_reversal"]
    correction: Dict[str, object] = {
        "audit": audit["audit"],
        "generated_at": utc_now(),
        "historical_protocol_sha256": sha256_file(protocol_path),
        "historical_execution_source_sha256": protocol["provenance"]["execution_source_sha256"],  # type: ignore[index]
        "audit_execution_source_sha256": execution_source_checksum(repository),
        "historical_formal_tree_before_analysis": historical_formal_digest,
        "historical_archive": historical_archive,
        "raw_outcomes_changed": False,
        "threshold_correction": {
            "source": "leave_one_cluster_out_training_nominal",
            "thresholds_by_held_out_cluster": primary_thresholds,
            "field_reversal_recovery_times_sweeps": {
                str(row.cluster_id): float(row.recovery_time_sweeps)
                for row in field_recovery.itertuples()
            },
        },
        "H3": dispositions["H3"],
        "historical_holm_values_retained": {
            str(row.hypothesis): {
                "raw": float(row.exact_one_sided_sign_flip_p),
                "holm": float(row.holm_adjusted_p),
            }
            for row in hypotheses.itertuples()
        },
        "permutation_analysis": permutation_summary.to_dict("records"),
        "sensitivity_completion": {
            "rolling_windows": windows,
            "single_observable_deletions": list(PRIMARY_Z_FEATURES),
            "robustness_rows": int(len(robustness)),
            "information_null_replicates_per_window": information_null_replicates,
        },
    }
    correction_path = result / "corrections/v14_scientific_audit_v1_1/correction_record.json"
    atomic_json(correction, correction_path)
    primary: Dict[str, object] = {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "audit_version": audit["audit"],
        "audit_config_sha256": sha256_file(audit_path),
        "audit_execution_source_sha256": correction["audit_execution_source_sha256"],
        "formal_trajectories": int(len(panel_rows)),
        "independent_clusters": int(pd.DataFrame(panel_rows)["cluster_id"].nunique()),
        "macrostate_rows": int(len(primary_macro)),
        "all_rolling_window_rows": int(len(all_windows)),
        "confirmatory_dispositions": dispositions,
        "memory_discovery_replication": memory_summary,
        "privacy_mutations": int(pd.DataFrame(panel_rows)["privacy_mutations"].sum()),
        "nonfinite_primary_macrostate_values": int(
            np.sum(~np.isfinite(primary_macro[list(PRIMARY_Z_FEATURES)].to_numpy(float)))
        ),
        "formal_accounting": completion,
        "permutation_analysis": permutation_summary.to_dict("records"),
        "analysis_wall_seconds": time.perf_counter() - started,
        "analysis_cpu_seconds": time.process_time() - cpu_started,
    }
    atomic_json(primary, statistics / "primary_results.json")
    atomic_json(
        {
            "generated_at": utc_now(),
            "external_v14_root": str(artifact_root()),
            "historical_formal_tree": historical_formal_digest,
            "tables": {name: {"rows": int(len(frame))} for name, frame in outputs.items()},
        },
        artifact_root() / "analysis" / "aggregate_manifest_v1_1.json",
    )
    return primary


__all__ = [
    "INFORMATION_AUDIT_METRICS",
    "OBSERVABLE_FAMILIES",
    "PRIMARY_Z_FEATURES",
    "analyze_formal",
    "cluster_preserving_permutation_analysis",
    "disruption_summaries",
    "exact_sign_flip_p",
    "holm_adjust",
    "information_estimator_contrasts",
    "nominal_distance_analysis",
    "panel_summary",
    "phase_feature_table",
    "representation_cv",
    "rolling_macrostates",
]
