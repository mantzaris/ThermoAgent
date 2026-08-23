"""Frozen cluster-level analysis for V15 cross-model memory and quench trajectories."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from thermoagent.statmech_llm_v12.estimators import paired_cluster_bootstrap
from thermoagent.statmech_llm_v14.analysis import (
    _macro_codes,
    exact_sign_flip_p,
    holm_adjust,
    rolling_macrostates,
)
from thermoagent.statmech_llm_v14.observables import (
    conditional_memory_depths,
    irreversibility_sensitivity,
    phase_path_length,
    recovery_time,
    signed_polygon_area,
    standardized_nominal_distance,
    standardized_nominal_fit,
)

from .experiment import formal_panel_design, graph_for_panel
from .workflow import (
    artifact_root,
    atomic_csv,
    atomic_json,
    load_yaml,
    sha256_file,
    tree_digest,
    utc_now,
)


V15_Z_FEATURES = (
    "belief_magnetization",
    "action_magnetization",
    "belief_action_overlap",
    "reference_energy_per_agent",
    "energy_variance",
    "configuration_entropy",
    "entropy_rate",
    "total_correlation_bias_adjusted",
    "pairwise_mutual_information_bias_adjusted",
    "edge_mutual_information_bias_adjusted",
    "belief_susceptibility",
    "spatial_belief_correlation",
    "belief_disagreement",
)


def _analysis_seed(token: str) -> int:
    return int(15150000 + sum((index + 1) * ord(value) for index, value in enumerate(token)) % 900000)


def _impute_training(training: np.ndarray, testing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train = np.asarray(training, dtype=float).copy()
    test = np.asarray(testing, dtype=float).copy()
    medians = np.nanmedian(train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    for column in range(train.shape[1]):
        train[~np.isfinite(train[:, column]), column] = medians[column]
        test[~np.isfinite(test[:, column]), column] = medians[column]
    return train, test


def _panel_analysis(
    panel: Mapping[str, object],
    protocol: Mapping[str, object],
    panel_root: str,
) -> Dict[str, object]:
    path = Path(panel_root) / (str(panel["panel_id"]) + ".csv")
    frame = pd.read_csv(path)
    expected = int(panel["n_agents"]) * int(panel["sweeps"])
    if len(frame) != expected:
        raise RuntimeError("V15 formal trajectory row mismatch: %s" % path.name)
    graph = graph_for_panel(panel)
    windows = sorted(
        {
            int(protocol["analysis"]["primary_window_sweeps"]),  # type: ignore[index]
            *[int(value) for value in protocol["analysis"]["alternative_window_sweeps"]],  # type: ignore[index]
        }
    )
    null_replicates = int(protocol["analysis"]["information_null"]["replicates_per_window"])  # type: ignore[index]
    macro_rows: List[Dict[str, object]] = []
    for window in windows:
        rows = rolling_macrostates(
            frame,
            panel,
            protocol,
            window,
            information_null_replicates=null_replicates,
            graph_override=graph,
        )
        for row in rows:
            row.update(
                {
                    "model_key": panel["model_key"],
                    "condition": panel["condition"],
                    "memory_mode": panel["memory_mode"],
                }
            )
        macro_rows.extend(rows)
    codes = _macro_codes(frame, protocol["analysis"]["entropy_coarse_graining"])  # type: ignore[index]
    settings = protocol["analysis"]["irreversibility"]  # type: ignore[index]
    sensitivity = irreversibility_sensitivity(
        codes,
        settings["sensitivity_block_lengths"],  # type: ignore[index]
        settings["sensitivity_pseudocounts"],  # type: ignore[index]
        int(settings["time_shuffle_replicates_per_panel"]),  # type: ignore[index]
        # Matched arms in one graph/model cluster use the same shuffle tape so
        # the finite-sample floor contributes less Monte Carlo noise to paired
        # memory contrasts.  The observed state sequences remain arm-specific.
        _analysis_seed("%s:%s" % (panel["model_key"], panel["cluster_id"])),
    )
    primary = next(
        row
        for row in sensitivity
        if int(row["block_length"]) == int(settings["primary_block_length"])  # type: ignore[index]
        and np.isclose(float(row["pseudocount"]), float(settings["primary_pseudocount"]))  # type: ignore[index]
    )
    depths = conditional_memory_depths(codes, (1, 2, 3))
    panel_summary: Dict[str, object] = {
        "model_key": panel["model_key"],
        "model_id": panel["model_id"],
        "model_revision": panel["model_revision"],
        "cluster_id": panel["cluster_id"],
        "panel_id": panel["panel_id"],
        "condition": panel["condition"],
        "disruption": panel["disruption"],
        "memory_mode": panel["memory_mode"],
        "attempted_updates": len(frame),
        "valid_after_repair_fraction": float(frame["valid_after_repair"].mean()),
        "belief_minus_to_plus": int(
            np.sum((frame["belief_before"] < 0) & (frame["belief_after"] > 0))
        ),
        "belief_plus_to_minus": int(
            np.sum((frame["belief_before"] > 0) & (frame["belief_after"] < 0))
        ),
        "latent_plus_occupancy": float((frame["belief_after"] > 0).mean()),
        "raw_block_divergence_nats_per_update": primary[
            "raw_block_divergence_nats_per_update"
        ],
        "shuffle_floor_nats_per_update": primary["shuffle_floor_nats_per_update"],
        "adjusted_pathwise_irreversibility_nats_per_update": primary[
            "adjusted_irreversibility_nats_per_update"
        ],
        "prompt_tokens": int(frame["prompt_tokens"].sum()),
        "generated_tokens": int(frame["generated_tokens"].sum()),
        "model_calls": int(frame["model_calls"].sum()),
        "latency_seconds": float(frame["latency_seconds"].sum()),
        "mean_prompt_tokens": float(frame["prompt_tokens"].mean()),
        "mean_prompt_memory_entries": float(frame["prompt_memory_entry_count"].mean()),
        "mean_prompt_memory_characters": float(frame["prompt_memory_characters"].mean()),
        "message_opportunities": int(frame["message_opportunities"].sum()),
        "messages_delivered": int(frame["messages_delivered"].sum()),
        "wire_bytes": int(frame["wire_bytes"].sum()),
        "privacy_mutations": int(frame["unrelated_peer_private_mutations"].sum()),
    }
    for depth, value in depths.items():
        panel_summary["conditional_memory_depth_%d" % depth] = float(value)
    sensitivity_rows = [
        {
            "model_key": panel["model_key"],
            "cluster_id": panel["cluster_id"],
            "panel_id": panel["panel_id"],
            "condition": panel["condition"],
            **row,
        }
        for row in sensitivity
    ]
    return {
        "macro_rows": macro_rows,
        "panel_summary": panel_summary,
        "irreversibility_rows": sensitivity_rows,
    }


def fit_nominal_distances(
    macrostates: pd.DataFrame,
    protocol: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    output = macrostates.copy()
    output["macrostate_distance"] = np.nan
    output["training_nominal_threshold_95"] = np.nan
    diagnostics: List[Dict[str, object]] = []
    thresholds: Dict[str, float] = {}
    ridge = float(protocol["analysis"]["nominal_ridge_fraction"])  # type: ignore[index]
    quantile = float(protocol["analysis"]["nominal_threshold_quantile"])  # type: ignore[index]
    for (model, window), model_window in output.groupby(["model_key", "window_sweeps"]):
        for cluster in sorted(model_window["cluster_id"].unique()):
            training = model_window[
                (model_window["condition"] == "nominal_markovized")
                & (model_window["cluster_id"] != cluster)
            ]
            testing = model_window[model_window["cluster_id"] == cluster]
            training_clusters = sorted(str(value) for value in training["cluster_id"].unique())
            if str(cluster) in training_clusters:
                raise AssertionError("held-out V15 cluster entered nominal fitting")
            train_values, test_values = _impute_training(
                training[list(V15_Z_FEATURES)].to_numpy(float),
                testing[list(V15_Z_FEATURES)].to_numpy(float),
            )
            fitted = standardized_nominal_fit(train_values, "shrinkage", ridge)
            train_distances = standardized_nominal_distance(train_values, fitted)
            test_distances = standardized_nominal_distance(test_values, fitted)
            threshold = float(np.quantile(train_distances, quantile))
            output.loc[testing.index, "macrostate_distance"] = test_distances
            output.loc[testing.index, "training_nominal_threshold_95"] = threshold
            key = "%s:%s:w%d" % (model, cluster, int(window))
            thresholds[key] = threshold
            for condition in sorted(testing["condition"].unique()):
                mask = testing["condition"].to_numpy() == condition
                selected = test_distances[np.flatnonzero(mask)]
                diagnostics.append(
                    {
                        "model_key": model,
                        "window_sweeps": int(window),
                        "held_out_cluster": cluster,
                        "condition": condition,
                        "training_clusters": json.dumps(training_clusters),
                        "held_out_cluster_excluded": True,
                        "training_nominal_threshold_95": threshold,
                        "maximum_distance": float(np.max(selected)),
                        "mean_distance": float(np.mean(selected)),
                    }
                )
    if output["macrostate_distance"].isna().any():
        raise RuntimeError("V15 nominal distance contains missing values")
    return output, pd.DataFrame(diagnostics), thresholds


def quench_summaries(
    primary_macro: pd.DataFrame,
    protocol: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    early_start, early_end = [int(value) for value in protocol["analysis"]["recovery"]["early_sweeps"]]  # type: ignore[index]
    late_start, late_end = [int(value) for value in protocol["analysis"]["recovery"]["late_sweeps"]]  # type: ignore[index]
    consecutive = int(protocol["analysis"]["recovery"]["consecutive_sweeps_within_threshold"])  # type: ignore[index]
    rows: List[Dict[str, object]] = []
    for panel_id, group in primary_macro.groupby("panel_id"):
        group = group.sort_values("sweep")
        baseline = group[group["phase"] == "baseline"]
        disruption = group[group["phase"] == "disruption"]
        recovery = group[group["phase"] == "recovery"]
        model = str(group["model_key"].iloc[0])
        cluster = str(group["cluster_id"].iloc[0])
        key = "%s:%s:w%d" % (
            model,
            cluster,
            int(protocol["analysis"]["primary_window_sweeps"]),  # type: ignore[index]
        )
        threshold = float(thresholds[key])
        early = group[(group["sweep"] >= early_start) & (group["sweep"] <= early_end)]
        late = group[(group["sweep"] >= late_start) & (group["sweep"] <= late_end)]
        if len(early) != early_end - early_start + 1 or len(late) != late_end - late_start + 1:
            raise RuntimeError("fixed recovery windows are incomplete")
        rows.append(
            {
                "model_key": model,
                "cluster_id": cluster,
                "panel_id": panel_id,
                "condition": group["condition"].iloc[0],
                "memory_mode": group["memory_mode"].iloc[0],
                "baseline_mean_distance": float(baseline["macrostate_distance"].mean()),
                "maximum_disruption_distance": float(disruption["macrostate_distance"].max()),
                "maximum_post_quench_distance": float(
                    pd.concat([disruption, recovery])["macrostate_distance"].max()
                ),
                "early_recovery_mean_distance_sweeps_31_35": float(
                    early["macrostate_distance"].mean()
                ),
                "late_recovery_mean_distance_sweeps_41_45": float(
                    late["macrostate_distance"].mean()
                ),
                "fixed_early_minus_late_recovery_distance": float(
                    early["macrostate_distance"].mean() - late["macrostate_distance"].mean()
                ),
                "recovery_time_sweeps": recovery_time(
                    recovery["macrostate_distance"], threshold, consecutive
                ),
                "final_five_sweep_mean_distance": float(
                    recovery["macrostate_distance"].tail(5).mean()
                ),
                "macrostate_path_length": phase_path_length(
                    group[list(V15_Z_FEATURES)].to_numpy(float)
                ),
                "energy_entropy_signed_loop_area": signed_polygon_area(
                    group["reference_energy_per_agent"], group["configuration_entropy"]
                ),
                "belief_action_signed_loop_area": signed_polygon_area(
                    group["belief_magnetization"], group["action_magnetization"]
                ),
                "training_nominal_threshold_95": threshold,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(values: Mapping[str, float], seed: int) -> Dict[str, float]:
    wrapped = {key: [float(value)] for key, value in values.items()}
    return paired_cluster_bootstrap(wrapped, 10000, int(seed))


def primary_hypotheses(
    panels: pd.DataFrame,
    quench: pd.DataFrame,
    protocol: Mapping[str, object],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    values: Dict[str, Dict[str, float]] = {key: {} for key in ("H1", "H2", "H3", "H4")}
    for cluster, group in quench[quench["model_key"] == "granite"].groupby("cluster_id"):
        field = group[group["condition"] == "field_markovized"].iloc[0]
        nominal = group[group["condition"] == "nominal_markovized"].iloc[0]
        values["H1"][str(cluster)] = float(
            field["maximum_post_quench_distance"] - nominal["maximum_post_quench_distance"]
        )
    for (model, cluster), group in panels.groupby(["model_key", "cluster_id"]):
        metric = "adjusted_pathwise_irreversibility_nats_per_update"
        markov = group[group["condition"] == "field_markovized"].iloc[0]
        persistent = group[group["condition"] == "field_persistent"].iloc[0]
        scrambled = group[group["condition"] == "field_scrambled"].iloc[0]
        unit = "%s:%s" % (model, cluster)
        values["H2"][unit] = float(persistent[metric] - markov[metric])
        values["H3"][unit] = float(persistent[metric] - scrambled[metric])
    for (model, cluster), group in quench.groupby(["model_key", "cluster_id"]):
        field = group[group["condition"] == "field_markovized"].iloc[0]
        values["H4"]["%s:%s" % (model, cluster)] = float(
            field["fixed_early_minus_late_recovery_distance"]
        )
    metadata = {
        "H1": ("Granite field minus nominal maximum post-quench distance", "distance_units"),
        "H2": ("persistent minus Markovized adjusted pathwise irreversibility", "nats_per_attempted_update"),
        "H3": ("persistent minus scrambled adjusted pathwise irreversibility", "nats_per_attempted_update"),
        "H4": ("recovery sweeps 31-35 minus sweeps 41-45 distance", "distance_units"),
    }
    rows: List[Dict[str, object]] = []
    raw_p: Dict[str, float] = {}
    for index, hypothesis in enumerate(("H1", "H2", "H3", "H4")):
        effect = _bootstrap(values[hypothesis], 15160000 + 101 * index)
        pvalue = exact_sign_flip_p(list(values[hypothesis].values()), 1)
        raw_p[hypothesis] = pvalue
        rows.append(
            {
                "hypothesis": hypothesis,
                "estimand": metadata[hypothesis][0],
                "unit": metadata[hypothesis][1],
                **effect,
                "exact_one_sided_sign_flip_p": pvalue,
                "cluster_values": json.dumps(values[hypothesis], sort_keys=True),
                "independent_clusters": len(values[hypothesis]),
            }
        )
    adjusted_secondary = holm_adjust([raw_p[key] for key in ("H2", "H3", "H4")])
    for row in rows:
        hypothesis = str(row["hypothesis"])
        if hypothesis == "H1":
            row["multiplicity_adjusted_p"] = raw_p[hypothesis]
            row["allocated_alpha"] = float(protocol["multiplicity"]["H1_alpha"])  # type: ignore[index]
        else:
            index = ("H2", "H3", "H4").index(hypothesis)
            row["multiplicity_adjusted_p"] = adjusted_secondary[index]
            row["allocated_alpha"] = float(protocol["multiplicity"]["H2_H3_H4_family_alpha"])  # type: ignore[index]
        row["supported"] = bool(
            float(row["estimate"]) > 0.0
            and float(row["multiplicity_adjusted_p"]) < float(row["allocated_alpha"])
        )
    disposition = {
        str(row["hypothesis"]): {
            key: row[key]
            for key in (
                "estimate",
                "ci_low",
                "ci_high",
                "exact_one_sided_sign_flip_p",
                "multiplicity_adjusted_p",
                "allocated_alpha",
                "supported",
                "independent_clusters",
            )
        }
        for row in rows
    }
    return pd.DataFrame(rows), disposition


def memory_prompt_balance(panels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (model, cluster), group in panels.groupby(["model_key", "cluster_id"]):
        persistent = group[group["condition"] == "field_persistent"].iloc[0]
        scrambled = group[group["condition"] == "field_scrambled"].iloc[0]
        rows.append(
            {
                "model_key": model,
                "cluster_id": cluster,
                "persistent_mean_prompt_tokens": float(persistent["mean_prompt_tokens"]),
                "scrambled_mean_prompt_tokens": float(scrambled["mean_prompt_tokens"]),
                "persistent_minus_scrambled_mean_prompt_tokens": float(
                    persistent["mean_prompt_tokens"] - scrambled["mean_prompt_tokens"]
                ),
                "persistent_mean_memory_entries": float(
                    persistent["mean_prompt_memory_entries"]
                ),
                "scrambled_mean_memory_entries": float(scrambled["mean_prompt_memory_entries"]),
            }
        )
    return pd.DataFrame(rows)


def analyze_formal(repository: Path) -> Dict[str, object]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    repository = Path(repository).resolve()
    protocol_path = repository / "configs/statmech_v15/protocol_frozen.yaml"
    protocol = load_yaml(protocol_path)
    completion_path = artifact_root() / "formal/completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["status"] != "complete":
        raise RuntimeError("V15 formal execution is incomplete")
    panels = formal_panel_design(protocol)
    workers = max(1, int(os.environ.get("THERMO_V15_ANALYSIS_WORKERS", "1")))
    analyzed = Parallel(n_jobs=workers, prefer="processes")(
        delayed(_panel_analysis)(panel, protocol, str(artifact_root() / "formal/panels"))
        for panel in panels
    )
    macro = pd.DataFrame(
        [row for result in analyzed for row in result["macro_rows"]]
    )
    panel_table = pd.DataFrame([result["panel_summary"] for result in analyzed])
    irreversibility = pd.DataFrame(
        [row for result in analyzed for row in result["irreversibility_rows"]]
    )
    macro, nominal_diagnostics, thresholds = fit_nominal_distances(macro, protocol)
    primary_window = int(protocol["analysis"]["primary_window_sweeps"])  # type: ignore[index]
    primary_macro = macro[macro["window_sweeps"] == primary_window].copy()
    quench = quench_summaries(primary_macro, protocol, thresholds)
    effects, dispositions = primary_hypotheses(panel_table, quench, protocol)
    prompt_balance = memory_prompt_balance(panel_table)
    result = repository / "results/collective_agent_statmech_v15"
    tables = result / "tables"
    statistics = result / "statistics"
    for directory in (tables, statistics, result / "figures/source_data", result / "reproducibility", result / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "panel_statistics.csv": panel_table,
        "macrostate_trajectories.csv": primary_macro,
        "macrostate_trajectories_all_windows.csv": macro,
        "nominal_distance_diagnostics.csv": nominal_diagnostics,
        "quench_recovery.csv": quench,
        "hypothesis_effects.csv": effects,
        "irreversibility_sensitivity.csv": irreversibility,
        "memory_prompt_balance.csv": prompt_balance,
    }
    for name, frame in outputs.items():
        atomic_csv(frame, tables / name)
    primary = {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "formal_completion": completion,
        "formal_trajectories": len(panel_table),
        "independent_clusters_per_model": int(protocol["network"]["clusters_per_model"]),  # type: ignore[index]
        "model_keys": sorted(panel_table["model_key"].unique()),
        "confirmatory_dispositions": dispositions,
        "privacy_mutations": int(panel_table["privacy_mutations"].sum()),
        "nonfinite_primary_features": int(
            np.sum(~np.isfinite(primary_macro[list(V15_Z_FEATURES)].to_numpy(float)))
        ),
        "analysis_wall_seconds": float(time.perf_counter() - started),
        "analysis_cpu_seconds": float(time.process_time() - cpu_started),
        "external_formal_tree": tree_digest(artifact_root() / "formal"),
        "external_raw_tree": tree_digest(artifact_root() / "raw/formal"),
    }
    atomic_json(primary, statistics / "primary_results.json")
    atomic_json(
        {
            "generated_at": utc_now(),
            "tables": {name: {"rows": len(frame)} for name, frame in outputs.items()},
        },
        artifact_root() / "analysis/aggregate_manifest.json",
    )
    return primary


__all__ = [
    "V15_Z_FEATURES",
    "analyze_formal",
    "fit_nominal_distances",
    "memory_prompt_balance",
    "primary_hypotheses",
    "quench_summaries",
]
