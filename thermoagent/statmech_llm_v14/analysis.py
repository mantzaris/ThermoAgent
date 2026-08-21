"""Frozen cluster-level analysis for the V14 memory and quench study."""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.preprocessing import StandardScaler

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
from .workflow import artifact_root, atomic_csv, atomic_json, load_yaml, sha256_file, utc_now


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
) -> List[Dict[str, object]]:
    """Aggregate online-compatible observables once per sweep."""

    n = int(panel["n_agents"])
    window_updates = int(window_sweeps) * n
    widths = protocol["analysis"]["entropy_coarse_graining"]  # type: ignore[index]
    all_codes = _macro_codes(frame, widths)
    beliefs = np.vstack([_bits(value) for value in frame["beliefs"]])
    actions = np.vstack([_bits(value) for value in frame["actions"]])
    output: List[Dict[str, object]] = []
    graph = graph_for_panel(panel)
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
                "total_correlation": total_correlation(joint) if len(window) > 1 else 0.0,
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    settings = protocol["analysis"]["nominal_distance"]  # type: ignore[index]
    primary_estimator = str(settings["primary_estimator"])  # type: ignore[index]
    primary_ridge = float(settings["primary_ridge_fraction"])  # type: ignore[index]
    output = macro.copy()
    output["macrostate_distance"] = np.nan
    robustness: List[Dict[str, object]] = []
    variants: List[Tuple[str, float, str, Tuple[str, ...], str]] = []
    for estimator in settings["estimators"]:  # type: ignore[index]
        ridges = settings["ridge_fractions"] if estimator in ("shrinkage", "robust") else [0.0]  # type: ignore[index]
        for ridge in ridges:
            for nominal_window in protocol["analysis"]["nominal_fit_windows"]:  # type: ignore[index]
                for rolling_window in [protocol["analysis"]["primary_window_sweeps"]]:  # type: ignore[index]
                    variants.append((str(estimator), float(ridge), str(nominal_window), PRIMARY_Z_FEATURES, "all"))
    for family, values in OBSERVABLE_FAMILIES.items():
        retained = tuple(value for value in PRIMARY_Z_FEATURES if value not in values)
        variants.append((primary_estimator, primary_ridge, "all_nominal", retained, "without_%s" % family))
    for estimator, ridge, nominal_window, features, ablation in variants:
        for cluster in sorted(output["cluster_id"].unique()):
            training_pool = _nominal_training_selection(output[output["cluster_id"] != cluster], nominal_window)
            test = output[output["cluster_id"] == cluster]
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
                        "feature_count": len(features),
                        "maximum_distance": float(np.nanmax(selected)) if np.any(np.isfinite(selected)) else float("nan"),
                        "mean_distance": float(np.nanmean(selected)) if np.any(np.isfinite(selected)) else float("nan"),
                        "nominal_threshold_95": threshold,
                        "failure": failed,
                    }
                )
    if output["macrostate_distance"].isna().any():
        raise RuntimeError("primary nominal distance contains missing values")
    return output, pd.DataFrame(robustness)


def disruption_summaries(
    macro: pd.DataFrame,
    protocol: Mapping[str, object],
) -> pd.DataFrame:
    final_window = int(protocol["analysis"]["recovery"]["final_window_sweeps"])  # type: ignore[index]
    rows: List[Dict[str, object]] = []
    for panel_id, group in macro.groupby("panel_id"):
        group = group.sort_values("sweep")
        baseline = group[group["phase"] == "baseline"]
        disruption = group[group["phase"] == "disruption"]
        recovery = group[group["phase"] == "recovery"]
        threshold = float(np.quantile(baseline["macrostate_distance"], 0.95))
        energy_baseline = float(baseline["reference_energy_per_agent"].mean())
        entropy_baseline = float(baseline["configuration_entropy"].mean())
        early_peak = float(recovery["macrostate_distance"].max())
        final_mean = float(recovery["macrostate_distance"].tail(final_window).mean())
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
            }
        )
    return pd.DataFrame(rows)


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
        supported = bool(value < 0.05 and float(row["estimate"]) > 0.0)
        dispositions[str(row["hypothesis"])] = {
            "supported": supported,
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


def analyze_formal(repository: Path) -> Dict[str, object]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    repository = Path(repository).resolve()
    protocol_path = repository / "configs/statmech_v14/protocol_frozen.yaml"
    protocol = load_yaml(protocol_path)
    root = artifact_root() / "formal"
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion["status"] != "complete":
        raise RuntimeError("V14 formal execution is incomplete")
    primary_window = int(protocol["analysis"]["primary_window_sweeps"])  # type: ignore[index]
    macro_rows: List[Dict[str, object]] = []
    panel_rows: List[Dict[str, object]] = []
    irr_rows: List[Dict[str, object]] = []
    memory_depth_rows: List[Dict[str, object]] = []
    alternative_windows: List[Dict[str, object]] = []
    for panel in formal_panel_design(protocol):
        path = root / "panels" / (str(panel["panel_id"]) + ".csv")
        frame = pd.read_csv(path)
        if len(frame) != int(panel["n_agents"]) * int(panel["sweeps"]):
            raise RuntimeError("formal trajectory row mismatch: %s" % path.name)
        rolling = rolling_macrostates(frame, panel, protocol, primary_window)
        macro = pd.DataFrame(rolling)
        macro_rows.extend(rolling)
        summary, sensitivity, depths = panel_summary(frame, macro, panel, protocol)
        panel_rows.append(summary)
        irr_rows.extend(sensitivity)
        memory_depth_rows.extend(depths)
        for window in protocol["analysis"]["alternative_window_sweeps"]:  # type: ignore[index]
            alternative_windows.extend(rolling_macrostates(frame, panel, protocol, int(window)))
    macro = pd.DataFrame(macro_rows)
    macro, robustness = nominal_distance_analysis(macro, protocol)
    disruption = disruption_summaries(macro, protocol)
    phase_features = phase_feature_table(macro)
    predictions, folds, coefficients = representation_cv(phase_features, protocol)
    hypotheses, dispositions = primary_hypotheses(disruption, folds)
    memory, memory_summary = memory_discovery_replication(repository)
    legacy_sensitivity = _legacy_memory_sensitivity(repository, protocol)
    nodes, edges = network_snapshot_tables(root, protocol)
    result = repository / "results/collective_agent_statmech_v14"
    tables = result / "tables"
    statistics = result / "statistics"
    source = result / "figures/source_data"
    for directory in (tables, statistics, source, result / "reproducibility", result / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "panel_statistics.csv": pd.DataFrame(panel_rows),
        "macrostate_trajectories.csv": macro,
        "macrostate_trajectories_alternative_windows.csv": pd.DataFrame(alternative_windows),
        "quench_recovery.csv": disruption,
        "macrostate_distance_robustness.csv": robustness,
        "representation_features.csv": phase_features,
        "representation_predictions.csv": predictions,
        "representation_cv.csv": folds,
        "representation_coefficients.csv": coefficients,
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
    primary: Dict[str, object] = {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "formal_trajectories": int(len(panel_rows)),
        "independent_clusters": int(pd.DataFrame(panel_rows)["cluster_id"].nunique()),
        "macrostate_rows": int(len(macro)),
        "confirmatory_dispositions": dispositions,
        "memory_discovery_replication": memory_summary,
        "privacy_mutations": int(pd.DataFrame(panel_rows)["privacy_mutations"].sum()),
        "nonfinite_primary_macrostate_values": int(np.sum(~np.isfinite(macro[list(PRIMARY_Z_FEATURES)].to_numpy(float)))),
        "formal_accounting": completion,
        "analysis_wall_seconds": time.perf_counter() - started,
        "analysis_cpu_seconds": time.process_time() - cpu_started,
    }
    atomic_json(primary, statistics / "primary_results.json")
    atomic_json(
        {
            "generated_at": utc_now(),
            "external_v14_root": str(artifact_root()),
            "tables": {name: {"rows": int(len(frame))} for name, frame in outputs.items()},
        },
        artifact_root() / "analysis" / "aggregate_manifest.json",
    )
    return primary


__all__ = [
    "OBSERVABLE_FAMILIES",
    "PRIMARY_Z_FEATURES",
    "analyze_formal",
    "disruption_summaries",
    "exact_sign_flip_p",
    "holm_adjust",
    "nominal_distance_analysis",
    "panel_summary",
    "phase_feature_table",
    "representation_cv",
    "rolling_macrostates",
]
