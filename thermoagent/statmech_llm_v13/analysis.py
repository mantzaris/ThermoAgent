"""Frozen trajectory-level statistical analysis for V13."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thermoagent.statmech_llm_v12.estimators import (
    block_time_reversal_kl,
    conditional_mutual_information_history,
    paired_cluster_bootstrap,
    shannon_entropy,
    time_shuffle_floor,
)

from .experiment import formal_panel_design, graph_for_panel
from .observables import (
    conditional_entropy_rate,
    integrated_correlation_time,
    macrostate_code,
    mahalanobis_distance,
    plugin_entropy,
    regularized_mahalanobis_fit,
    total_correlation,
)
from .surrogate import fit_kinetic_surrogate, simulate_surrogate_grid
from .workflow import artifact_root, atomic_csv, atomic_json, load_yaml, sha256_file, utc_now


Z_FEATURES = (
    "belief_magnetization",
    "action_magnetization",
    "belief_action_overlap",
    "configuration_entropy",
    "entropy_rate",
    "total_correlation",
    "reference_energy_per_agent",
    "energy_variance",
    "belief_susceptibility",
    "spatial_belief_correlation",
    "belief_disagreement",
)


def _bits(value: object) -> np.ndarray:
    return np.asarray([int(item) for item in str(value).split(";")], dtype=int)


def _floats(value: object) -> np.ndarray:
    return np.asarray([float(item) for item in str(value).split(";")], dtype=float)


def _macro_codes(frame: pd.DataFrame, widths: Mapping[str, object]) -> np.ndarray:
    tuples = [
        macrostate_code(
            {key: float(getattr(row, key)) for key in (
                "belief_magnetization",
                "action_magnetization",
                "belief_action_overlap",
                "reference_energy_per_agent",
                "belief_disagreement",
            )},
            widths,  # type: ignore[arg-type]
        )
        for row in frame.itertuples()
    ]
    mapping = {value: index for index, value in enumerate(sorted(set(tuples)))}
    return np.asarray([mapping[value] for value in tuples], dtype=int)


def _marginal_entropy(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=int)
    entropies = [shannon_entropy([np.sum(values[:, column] < 0), np.sum(values[:, column] > 0)]) for column in range(values.shape[1])]
    return float(np.mean(entropies))


def _rolling_macrostates(
    frame: pd.DataFrame, panel: Mapping[str, object], protocol: Mapping[str, object]
) -> List[Dict[str, object]]:
    n = int(panel["n_agents"])
    window_updates = int(protocol["analysis"]["primary_window_sweeps"]) * n  # type: ignore[index]
    widths = protocol["analysis"]["entropy_coarse_graining"]  # type: ignore[index]
    codes = _macro_codes(frame, widths)
    beliefs = np.vstack([_bits(value) for value in frame["beliefs"]])
    actions = np.vstack([_bits(value) for value in frame["actions"]])
    energy = frame["reference_energy_per_agent"].to_numpy(float)
    output: List[Dict[str, object]] = []
    for end in range(n - 1, len(frame), n):
        start = max(0, end - window_updates + 1)
        window = frame.iloc[start : end + 1]
        b = beliefs[start : end + 1]
        a = actions[start : end + 1]
        m = window["belief_magnetization"].to_numpy(float)
        e = energy[start : end + 1]
        state = codes[start : end + 1]
        last = frame.iloc[end]
        joint = np.concatenate([b, a], axis=1)
        output.append(
            {
                "family": panel["family"],
                "subset": panel["subset"],
                "cluster_id": panel["cluster_id"],
                "panel_id": panel["panel_id"],
                "n_agents": n,
                "topology": panel["topology"],
                "regime": panel["regime"],
                "disruption": panel["disruption"],
                "coupling_strength": float(panel["coupling_strength"]),
                "sampling_temperature": float(panel["sampling_temperature"]),
                "initial_condition": panel["initial_condition"],
                "sweep": int(end // n + 1),
                "phase": str(last["phase"]),
                "belief_magnetization": float(last["belief_magnetization"]),
                "action_magnetization": float(last["action_magnetization"]),
                "belief_action_overlap": float(last["belief_action_overlap"]),
                "belief_disagreement": float(last["belief_disagreement"]),
                "configuration_entropy": plugin_entropy(state.tolist()),
                "entropy_rate": conditional_entropy_rate(state, 1, 0.5),
                "single_agent_marginal_entropy": _marginal_entropy(joint),
                "total_correlation": total_correlation(joint) if len(window) >= 2 else 0.0,
                "reference_energy_per_agent": float(last["reference_energy_per_agent"]),
                "energy_variance": float(n * np.var(e, ddof=1)) if e.size >= 2 else 0.0,
                "belief_susceptibility": float(n * np.var(m, ddof=1)) if m.size >= 2 else 0.0,
                "spatial_belief_correlation": float(window["spatial_belief_correlation"].mean()),
                "mean_confidence": float(np.mean(np.concatenate([_floats(value) for value in window["confidences"]]))),
                "messages_delivered": int(window["messages_delivered"].sum()),
                "wire_bytes": int(window["wire_bytes"].sum()),
                "message_current": float(window["messages_delivered"].sum() / max(len(window), 1)),
                "workload_mean": float(np.mean(np.concatenate([_floats(value) for value in window["workloads"]]))),
                "workload_variance": float(np.var(np.concatenate([_floats(value) for value in window["workloads"]]))),
                "local_configuration_entropy": float(window["local_configuration_entropy"].mean()),
                "cross_community_deliveries": int(window["cross_community_delivery"].sum()),
                "corrupted_messages": int(window["message_corrupted"].sum()),
            }
        )
    return output


def _relaxation_time_sweeps(frame: pd.DataFrame, n_agents: int, threshold: float = 0.5) -> float:
    values = frame.iloc[n_agents - 1 :: n_agents]["belief_magnetization"].abs().to_numpy(float)
    for index in range(max(values.size - 1, 0)):
        if values[index] <= threshold and values[index + 1] <= threshold:
            return float(index + 1)
    return float(values.size)


def _trajectory_loop_area(macro: pd.DataFrame) -> float:
    if len(macro) < 3:
        return 0.0
    x = macro["reference_energy_per_agent"].to_numpy(float)
    y = macro["configuration_entropy"].to_numpy(float)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _panel_summary(
    frame: pd.DataFrame,
    macro: pd.DataFrame,
    panel: Mapping[str, object],
    protocol: Mapping[str, object],
) -> Dict[str, object]:
    n = int(panel["n_agents"])
    burn = int(panel["burn_in_sweeps"]) * n
    retained = frame.iloc[burn:].copy()
    if len(retained) < 8:
        raise RuntimeError("panel has too few retained updates")
    widths = protocol["analysis"]["entropy_coarse_graining"]  # type: ignore[index]
    codes = _macro_codes(retained, widths)
    raw_kl = block_time_reversal_kl(codes, int(protocol["analysis"]["primary_block_length"]), float(protocol["analysis"]["primary_pseudocount"]))  # type: ignore[index]
    floor = time_shuffle_floor(
        codes,
        int(protocol["analysis"]["primary_block_length"]),  # type: ignore[index]
        float(protocol["analysis"]["primary_pseudocount"]),  # type: ignore[index]
        int(protocol["analysis"]["time_shuffle_replicates_per_panel"]),  # type: ignore[index]
        _seed_for_analysis(str(panel["panel_id"])),
    )
    beliefs = np.vstack([_bits(value) for value in retained["beliefs"]])
    actions = np.vstack([_bits(value) for value in retained["actions"]])
    belief_m = retained["belief_magnetization"].to_numpy(float)
    action_m = retained["action_magnetization"].to_numpy(float)
    energy = retained["reference_energy_per_agent"].to_numpy(float)
    confidence = np.concatenate([_floats(value) for value in retained["confidences"]])
    history_cmi = conditional_mutual_information_history(codes, 1, 0.1) if len(codes) > 4 else float("nan")
    return {
        "family": panel["family"],
        "subset": panel["subset"],
        "cluster_id": panel["cluster_id"],
        "panel_id": panel["panel_id"],
        "n_agents": n,
        "topology": panel["topology"],
        "regime": panel["regime"],
        "disruption": panel["disruption"],
        "coupling_strength": float(panel["coupling_strength"]),
        "sampling_temperature": float(panel["sampling_temperature"]),
        "initial_condition": panel["initial_condition"],
        "sweeps": int(panel["sweeps"]),
        "attempted_updates": int(len(frame)),
        "retained_updates": int(len(retained)),
        "valid_fraction": float(retained["valid_after_repair"].mean()),
        "mean_belief_magnetization": float(np.mean(belief_m)),
        "mean_abs_belief_magnetization": float(np.mean(np.abs(belief_m))),
        "mean_action_magnetization": float(np.mean(action_m)),
        "mean_abs_action_magnetization": float(np.mean(np.abs(action_m))),
        "mean_belief_action_overlap": float(retained["belief_action_overlap"].mean()),
        "mean_belief_disagreement": float(retained["belief_disagreement"].mean()),
        "belief_susceptibility": float(n * np.var(belief_m, ddof=1)),
        "action_susceptibility": float(n * np.var(action_m, ddof=1)),
        "belief_integrated_autocorrelation_time_updates": integrated_correlation_time(belief_m),
        "action_integrated_autocorrelation_time_updates": integrated_correlation_time(action_m),
        "relaxation_time_sweeps": _relaxation_time_sweeps(frame, n),
        "configuration_entropy": plugin_entropy(codes.tolist()),
        "entropy_rate_nats_per_update": conditional_entropy_rate(codes, 1, 0.5),
        "single_agent_marginal_entropy": _marginal_entropy(np.concatenate([beliefs, actions], axis=1)),
        "total_correlation": total_correlation(np.concatenate([beliefs, actions], axis=1)),
        "mean_reference_energy_per_agent": float(np.mean(energy)),
        "energy_fluctuation_N_var_e": float(n * np.var(energy, ddof=1)),
        "mean_spatial_belief_correlation": float(retained["spatial_belief_correlation"].mean()),
        "raw_block_kl_nats_per_update": raw_kl,
        "time_shuffle_floor_nats_per_update": floor["mean"],
        "adjusted_block_irreversibility_nats_per_update": raw_kl - floor["mean"],
        "history_order_1_conditional_mutual_information": history_cmi,
        "energy_entropy_loop_area": _trajectory_loop_area(macro),
        "belief_transition_minus_plus": int(np.sum((retained["belief_before"] == -1) & (retained["belief_after"] == 1))),
        "belief_transition_plus_minus": int(np.sum((retained["belief_before"] == 1) & (retained["belief_after"] == -1))),
        "mean_confidence": float(np.mean(confidence)),
        "message_opportunities": int(retained["message_opportunities"].sum()),
        "messages_delivered": int(retained["messages_delivered"].sum()),
        "wire_bytes": int(retained["wire_bytes"].sum()),
        "prompt_tokens": int(retained["prompt_tokens"].sum()),
        "generated_tokens": int(retained["generated_tokens"].sum()),
        "latency_seconds": float(retained["latency_seconds"].sum()),
        "privacy_mutations": int(retained["unrelated_peer_private_mutations"].sum()),
    }


def _seed_for_analysis(token: str) -> int:
    return int(13400000 + sum((index + 1) * ord(value) for index, value in enumerate(token)) % 500000)


def _paired_differences(
    panel: pd.DataFrame,
    metric: str,
    factor: str,
    low: object,
    high: object,
    subset: str,
) -> Dict[str, List[float]]:
    selected = panel[panel["subset"] == subset]
    output: Dict[str, List[float]] = {}
    for cluster, group in selected.groupby("cluster_id"):
        low_values = group[group[factor] == low][metric].to_numpy(float)
        high_values = group[group[factor] == high][metric].to_numpy(float)
        if low_values.size and high_values.size:
            output[str(cluster)] = [float(np.mean(high_values) - np.mean(low_values))]
    return output


def _sign_flip_p(values: Sequence[float], direction: int, seed: int, replicates: int = 10000) -> float:
    array = np.asarray(values, dtype=float) * int(direction)
    observed = float(np.mean(array))
    rng = np.random.default_rng(int(seed))
    null = np.asarray([np.mean(array * rng.choice((-1.0, 1.0), size=array.size)) for _ in range(int(replicates))])
    return float((1.0 + np.sum(null >= observed)) / (null.size + 1.0))


def _holm(pvalues: Sequence[float]) -> List[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (values.size - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _effect(values: Mapping[str, Sequence[float]], seed: int, direction: int = 1) -> Dict[str, float]:
    result = paired_cluster_bootstrap(values, 10000, int(seed))
    cluster_values = [float(np.mean(item)) for item in values.values()]
    result["one_sided_sign_flip_p"] = _sign_flip_p(cluster_values, direction, seed + 101)
    return result


def _primary_effects(panel: pd.DataFrame) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    hypothesis_p: Dict[str, float] = {}
    metrics = (
        "mean_abs_belief_magnetization",
        "belief_susceptibility",
        "belief_integrated_autocorrelation_time_updates",
    )
    h1_ps = []
    h2_ps = []
    for index, metric in enumerate(metrics):
        coupling = _paired_differences(panel, metric, "coupling_strength", 0.35, 0.80, "modular_primary")
        effect = _effect(coupling, 13500000 + index, 1)
        h1_ps.append(effect["one_sided_sign_flip_p"])
        rows.append({"hypothesis": "H1", "contrast": "coupling_0.80_minus_0.35", "metric": metric, **effect})
        noise = _paired_differences(panel, metric, "sampling_temperature", 0.50, 0.85, "modular_primary")
        effect = _effect(noise, 13510000 + index, -1)
        h2_ps.append(effect["one_sided_sign_flip_p"])
        rows.append({"hypothesis": "H2", "contrast": "noise_0.85_minus_0.50", "metric": metric, **effect})
    hypothesis_p["H1"] = max(h1_ps)
    hypothesis_p["H2"] = max(h2_ps)
    memory = panel[panel["subset"] == "memory_confirmation"]
    memory_values: Dict[str, List[float]] = {}
    metric = "adjusted_block_irreversibility_nats_per_update"
    for cluster, group in memory.groupby("cluster_id"):
        markov = group[group["regime"] == "markovized"][metric].to_numpy(float)
        persistent = group[group["regime"] == "persistent_memory"][metric].to_numpy(float)
        if markov.size == 1 and persistent.size == 1:
            memory_values[str(cluster)] = [float(persistent[0] - markov[0])]
    memory_effect = _effect(memory_values, 13520000, 1)
    hypothesis_p["H3"] = memory_effect["one_sided_sign_flip_p"]
    rows.append({"hypothesis": "H3", "contrast": "persistent_minus_markovized", "metric": metric, **memory_effect})
    adjusted = _holm([hypothesis_p[key] for key in ("H1", "H2", "H3")])
    dispositions = {}
    for key, value in zip(("H1", "H2", "H3"), adjusted):
        endpoints = [row for row in rows if row["hypothesis"] == key]
        if key == "H1":
            interval_pass = all(float(row["ci_low"]) > 0.0 for row in endpoints)
        elif key == "H2":
            interval_pass = all(float(row["ci_high"]) < 0.0 for row in endpoints)
        else:
            interval_pass = float(endpoints[0]["ci_low"]) > 0.0
        dispositions[key] = {
            "intersection_union_raw_p": hypothesis_p[key],
            "holm_adjusted_p": value,
            "all_directional_intervals_pass": interval_pass,
            "supported": bool(value < 0.05 and interval_pass),
        }
        for row in endpoints:
            row["hypothesis_raw_p"] = hypothesis_p[key]
            row["hypothesis_holm_p"] = value
    return rows, dispositions


def _nominal_distances(macro: pd.DataFrame, ridge: float) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    output = macro.copy()
    output["macrostate_distance"] = np.nan
    summaries: List[Dict[str, object]] = []
    c = output[output["family"] == "C_disruption_recovery"]
    for cluster in sorted(c["cluster_id"].unique()):
        training = c[(c["cluster_id"] != cluster) & (c["disruption"] == "nominal")]
        if len(training) < len(Z_FEATURES) + 2:
            raise RuntimeError("insufficient leave-cluster-out nominal manifold data")
        center, precision = regularized_mahalanobis_fit(training[list(Z_FEATURES)].to_numpy(float), ridge)
        selection = (output["family"] == "C_disruption_recovery") & (output["cluster_id"] == cluster)
        output.loc[selection, "macrostate_distance"] = mahalanobis_distance(
            output.loc[selection, list(Z_FEATURES)].to_numpy(float), center, precision
        )
        threshold = float(np.quantile(mahalanobis_distance(training[list(Z_FEATURES)].to_numpy(float), center, precision), 0.95))
        for panel_id, group in output[selection].groupby("panel_id"):
            disruption = group[group["phase"] == "disruption"]
            recovery = group[group["phase"] == "recovery"]
            values = recovery["macrostate_distance"].to_numpy(float)
            recovery_time = float(len(values))
            for index in range(max(len(values) - 1, 0)):
                if values[index] <= threshold and values[index + 1] <= threshold:
                    recovery_time = float(index + 1)
                    break
            summaries.append(
                {
                    "cluster_id": cluster,
                    "panel_id": panel_id,
                    "disruption": str(group["disruption"].iloc[0]),
                    "nominal_threshold_95": threshold,
                    "baseline_mean_distance": float(group[group["phase"] == "baseline"]["macrostate_distance"].mean()),
                    "maximum_disruption_distance": float(disruption["macrostate_distance"].max()),
                    "mean_disruption_distance": float(disruption["macrostate_distance"].mean()),
                    "recovery_time_sweeps": recovery_time,
                    "final_residual_distance": float(recovery["macrostate_distance"].iloc[-1]),
                    "time_outside_nominal_sweeps": int(np.sum(group["macrostate_distance"] > threshold)),
                    "entropy_overshoot": float(disruption["configuration_entropy"].max() - group[group["phase"] == "baseline"]["configuration_entropy"].mean()),
                    "energy_overshoot": float(np.max(np.abs(disruption["reference_energy_per_agent"] - group[group["phase"] == "baseline"]["reference_energy_per_agent"].mean()))),
                }
            )
    return output, summaries


def _phase_features(macro: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    c = macro[macro["family"] == "C_disruption_recovery"]
    for panel_id, group in c.groupby("panel_id"):
        baseline = group[group["phase"] == "baseline"]
        disrupted = group[group["phase"] == "disruption"]
        recovery = group[group["phase"] == "recovery"]
        row: Dict[str, object] = {
            "panel_id": panel_id,
            "cluster_id": str(group["cluster_id"].iloc[0]),
            "label": str(group["disruption"].iloc[0]),
        }
        variables = set(Z_FEATURES) | {"mean_confidence", "messages_delivered", "message_current", "workload_variance", "macrostate_distance"}
        for variable in sorted(variables):
            row["delta_%s" % variable] = float(disrupted[variable].mean() - baseline[variable].mean())
            row["recovery_%s" % variable] = float(recovery[variable].mean() - baseline[variable].mean())
            row["max_%s" % variable] = float(disrupted[variable].max())
        rows.append(row)
    return pd.DataFrame(rows)


REPRESENTATIONS = {
    "simple": ["belief_magnetization", "action_magnetization", "mean_confidence", "messages_delivered"],
    "order_only": ["belief_magnetization", "action_magnetization", "belief_action_overlap", "belief_disagreement"],
    "full_statmech": list(Z_FEATURES) + ["macrostate_distance"],
}


def _representation_cv(features: pd.DataFrame) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    predictions: List[Dict[str, object]] = []
    folds: List[Dict[str, object]] = []
    labels = sorted(features["label"].unique())
    for representation, variables in REPRESENTATIONS.items():
        columns = ["delta_%s" % value for value in variables] + ["recovery_%s" % value for value in variables]
        for cluster in sorted(features["cluster_id"].unique()):
            train = features[features["cluster_id"] != cluster]
            test = features[features["cluster_id"] == cluster]
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, multi_class="auto", random_state=1313))
            model.fit(train[columns].to_numpy(float), train["label"].astype(str))
            predicted = model.predict(test[columns].to_numpy(float))
            probabilities = model.predict_proba(test[columns].to_numpy(float))
            class_order = list(model.classes_)
            correct = []
            for index, (_, item) in enumerate(test.iterrows()):
                truth = str(item["label"])
                choice = str(predicted[index])
                correct.append(int(truth == choice))
                predictions.append(
                    {
                        "representation": representation,
                        "held_out_cluster": cluster,
                        "panel_id": item["panel_id"],
                        "truth": truth,
                        "prediction": choice,
                        "correct": int(truth == choice),
                        "truth_probability": float(probabilities[index, class_order.index(truth)]),
                    }
                )
            folds.append(
                {
                    "representation": representation,
                    "held_out_cluster": cluster,
                    "accuracy": float(np.mean(correct)),
                    "multiclass_log_loss": float(-np.mean([math.log(max(predictions[-len(test) + index]["truth_probability"], 1e-12)) for index in range(len(test))])),
                    "test_panels": int(len(test)),
                    "chance_accuracy": 1.0 / len(labels),
                }
            )
    return predictions, folds


def _disruption_hypotheses(
    distance_summary: pd.DataFrame, representation_folds: pd.DataFrame
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    effects: List[Dict[str, object]] = []
    h4: Dict[str, List[float]] = {}
    for cluster, group in distance_summary.groupby("cluster_id"):
        nominal = float(group[group["disruption"] == "nominal"]["maximum_disruption_distance"].iloc[0])
        disrupted = group[group["disruption"] != "nominal"]["maximum_disruption_distance"].to_numpy(float)
        h4[str(cluster)] = [float(np.mean(disrupted) - nominal)]
    h4_effect = _effect(h4, 13600000, 1)
    effects.append({"hypothesis": "H4", "metric": "maximum_macrostate_departure_disrupted_minus_nominal", **h4_effect})
    full = representation_folds[representation_folds["representation"] == "full_statmech"]
    h5 = {str(row.held_out_cluster): [float(row.accuracy - row.chance_accuracy)] for row in full.itertuples()}
    h5_effect = _effect(h5, 13610000, 1)
    effects.append({"hypothesis": "H5", "metric": "full_representation_accuracy_minus_chance", **h5_effect})
    reduced = representation_folds[representation_folds["representation"] != "full_statmech"]
    h6: Dict[str, List[float]] = {}
    for cluster in sorted(full["held_out_cluster"].unique()):
        full_value = float(full[full["held_out_cluster"] == cluster]["accuracy"].iloc[0])
        reduced_value = float(reduced[reduced["held_out_cluster"] == cluster]["accuracy"].max())
        h6[str(cluster)] = [full_value - reduced_value]
    h6_effect = _effect(h6, 13620000, 1)
    effects.append({"hypothesis": "H6", "metric": "full_accuracy_minus_strongest_reduced", **h6_effect})
    dispositions = {
        row["hypothesis"]: {
            "supported": bool(float(row["ci_low"]) > 0.0),
            "criterion": "paired cluster-bootstrap lower 95% bound above zero",
        }
        for row in effects
    }
    return effects, dispositions


def _surrogate_aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    frame = pd.DataFrame(rows)
    keys = ["n_agents", "topology", "coupling_strength", "sampling_temperature"]
    metrics = ["mean_abs_belief_magnetization", "mean_abs_action_magnetization", "belief_susceptibility", "belief_correlation_time_sweeps", "mean_field_belief", "mean_field_action", "local_belief_stability_index"]
    output: List[Dict[str, object]] = []
    for values, group in frame.groupby(keys, sort=True):
        row: Dict[str, object] = dict(zip(keys, values))
        row["independent_seeds"] = int(len(group))
        for metric in metrics:
            data = group[metric].to_numpy(float)
            row[metric + "_mean"] = float(np.mean(data))
            row[metric + "_sd"] = float(np.std(data, ddof=1))
            row[metric + "_q025"] = float(np.quantile(data, 0.025))
            row[metric + "_q975"] = float(np.quantile(data, 0.975))
        output.append(row)
    return output


def _v12_discovery(repository: Path) -> List[Dict[str, object]]:
    source = json.loads((repository / "results/llm_agent_statmech_v12/statistics/primary_results.json").read_text(encoding="utf-8"))
    rows = []
    for key, value in source["collective_factor_effects"].items():
        factor, metric = key.split(":", 1)
        if metric in ("mean_abs_belief_magnetization", "belief_susceptibility", "belief_integrated_autocorrelation_time_updates"):
            rows.append({"study": "V12_discovery", "factor": factor, "metric": metric, **value})
    for metric, value in source["memory_effects"].items():
        if metric == "adjusted_block_kl_nats_per_update":
            rows.append({"study": "V12_discovery", "factor": "persistent_memory", "metric": metric, **value})
    return rows


def _network_snapshot_tables(
    root: Path, protocol: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Select the frozen cluster-0 before/during/after snapshots."""

    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    panels = [
        panel for panel in formal_panel_design(protocol)
        if panel["family"] == "C_disruption_recovery" and panel["cluster_id"] == "C_quench_n16_g0"
    ]
    for panel in panels:
        frame = pd.read_csv(root / "panels" / (str(panel["panel_id"]) + ".csv"))
        graph = graph_for_panel(panel)
        for phase in ("baseline", "disruption", "recovery"):
            phase_frame = frame[frame["phase"] == phase]
            selected = phase_frame.iloc[-1]
            beliefs = _bits(selected["beliefs"])
            actions = _bits(selected["actions"])
            confidence = _floats(selected["confidences"])
            for node in range(int(panel["n_agents"])):
                probability = float(np.clip(confidence[node], 1e-8, 1.0 - 1e-8))
                uncertainty = float(-probability * np.log(probability) - (1.0 - probability) * np.log(1.0 - probability))
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
                        "confidence_uncertainty": uncertainty,
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
    return nodes, edges


def analyze_formal(repository: Path) -> Dict[str, object]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    repository = Path(repository).resolve()
    protocol_path = repository / "configs/statmech_v13/protocol_frozen_v1.2.yaml"
    protocol = load_yaml(protocol_path)
    root = artifact_root() / "formal"
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion["status"] != "complete":
        raise RuntimeError("formal execution is incomplete")
    panel_rows: List[Dict[str, object]] = []
    macro_rows: List[Dict[str, object]] = []
    for panel in formal_panel_design(protocol):
        frame = pd.read_csv(root / "panels" / (str(panel["panel_id"]) + ".csv"))
        rolling = _rolling_macrostates(frame, panel, protocol)
        macro = pd.DataFrame(rolling)
        macro_rows.extend(rolling)
        panel_rows.append(_panel_summary(frame, macro, panel, protocol))
    panel_frame = pd.DataFrame(panel_rows)
    macro_frame = pd.DataFrame(macro_rows)
    macro_frame, distance_rows = _nominal_distances(
        macro_frame, float(protocol["analysis"]["nominal_distance"]["ridge_fraction"])  # type: ignore[index]
    )
    distance_frame = pd.DataFrame(distance_rows)
    phase_features = _phase_features(macro_frame)
    predictions, folds = _representation_cv(phase_features)
    fold_frame = pd.DataFrame(folds)
    primary_rows, primary_dispositions = _primary_effects(panel_frame)
    disruption_rows, disruption_dispositions = _disruption_hypotheses(distance_frame, fold_frame)
    micro = pd.read_csv(root / "microscopic_response.csv")
    parameters = fit_kinetic_surrogate(micro)
    surrogate_seed_rows = simulate_surrogate_grid(parameters, protocol["surrogate"])  # type: ignore[arg-type,index]
    surrogate_rows = _surrogate_aggregate(surrogate_seed_rows)
    surrogate_frame = pd.DataFrame(surrogate_rows)
    # H7 is a directional explanatory check, not a superiority claim.
    direct_modular = panel_frame[panel_frame["subset"] == "modular_primary"]
    direct_coupling = float(np.mean([value[0] for value in _paired_differences(direct_modular, "mean_abs_belief_magnetization", "coupling_strength", 0.35, 0.80, "modular_primary").values()]))
    direct_noise = float(np.mean([value[0] for value in _paired_differences(direct_modular, "mean_abs_belief_magnetization", "sampling_temperature", 0.50, 0.85, "modular_primary").values()]))
    surrogate_anchor = surrogate_frame[(surrogate_frame["topology"] == "modular") & surrogate_frame["coupling_strength"].isin([0.35, 0.80]) & surrogate_frame["sampling_temperature"].isin([0.50, 0.85])]
    surrogate_coupling = float(surrogate_anchor.groupby("coupling_strength")["mean_abs_belief_magnetization_mean"].mean().diff().iloc[-1])
    surrogate_noise = float(surrogate_anchor.groupby("sampling_temperature")["mean_abs_belief_magnetization_mean"].mean().diff().iloc[-1])
    h7 = {
        "direct_coupling_effect": direct_coupling,
        "surrogate_coupling_effect": surrogate_coupling,
        "direct_noise_effect": direct_noise,
        "surrogate_noise_effect": surrogate_noise,
        "coupling_direction_captured": bool(np.sign(direct_coupling) == np.sign(surrogate_coupling)),
        "noise_direction_captured": bool(np.sign(direct_noise) == np.sign(surrogate_noise)),
        "supported": bool(np.sign(direct_coupling) == np.sign(surrogate_coupling) and np.sign(direct_noise) == np.sign(surrogate_noise)),
    }
    result_dir = repository / "results/collective_agent_statmech_v13"
    tables = result_dir / "tables"
    statistics = result_dir / "statistics"
    source = result_dir / "figures/source_data"
    tables.mkdir(parents=True, exist_ok=True)
    statistics.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    atomic_csv(panel_rows, tables / "panel_statistics.csv")
    atomic_csv(macro_frame.to_dict("records"), tables / "macrostate_trajectories.csv")
    atomic_csv(distance_rows, tables / "disruption_recovery.csv")
    atomic_csv(phase_features.to_dict("records"), tables / "representation_features.csv")
    atomic_csv(predictions, tables / "representation_predictions.csv")
    atomic_csv(folds, tables / "representation_cv.csv")
    atomic_csv(primary_rows + disruption_rows, tables / "hypothesis_effects.csv")
    atomic_csv(_v12_discovery(repository), tables / "v12_discovery_effects.csv")
    atomic_csv(surrogate_rows, tables / "surrogate_phase_map.csv")
    snapshot_nodes, snapshot_edges = _network_snapshot_tables(root, protocol)
    atomic_csv(snapshot_nodes, tables / "network_snapshot_nodes.csv")
    atomic_csv(snapshot_edges, tables / "network_snapshot_edges.csv")
    micro_compact = micro[
        [
            "cell_id", "replicate", "private_field", "neighbor_field", "current_belief",
            "current_action", "coupling_strength", "sampling_temperature", "belief_after",
            "action_after", "belief_switched", "action_switched", "latent_plus_label",
            "display_order", "valid_after_repair",
        ]
    ]
    atomic_csv(micro_compact.to_dict("records"), tables / "microscopic_response.csv")
    atomic_json(parameters, statistics / "fitted_kinetic_surrogate.json")
    primary: Dict[str, object] = {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "panel_count": int(len(panel_frame)),
        "macrostate_rows": int(len(macro_frame)),
        "primary_confirmatory": primary_dispositions,
        "disruption_hypotheses": disruption_dispositions,
        "H7_surrogate": h7,
        "privacy_mutations": int(panel_frame["privacy_mutations"].sum()),
        "formal_accounting": completion,
        "analysis_wall_seconds": time.perf_counter() - started,
        "analysis_cpu_seconds": time.process_time() - cpu_started,
    }
    atomic_json(primary, statistics / "primary_results.json")
    atomic_json({"rows": surrogate_seed_rows}, artifact_root() / "analysis" / "surrogate_seed_results.json")
    return primary
