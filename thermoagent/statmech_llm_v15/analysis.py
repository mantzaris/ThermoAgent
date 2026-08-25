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

from thermoagent.statmech_llm_v12.estimators import (
    block_time_reversal_kl,
    paired_cluster_bootstrap,
)
from thermoagent.statmech_llm_v12.core import LatentMapping, MEMORY_STATES
from thermoagent.statmech_llm_v14.analysis import (
    _macro_codes,
    exact_sign_flip_p,
    holm_adjust,
    rolling_macrostates,
)
from thermoagent.statmech_llm_v14.observables import (
    conditional_memory_depths,
    phase_path_length,
    recovery_time,
    signed_polygon_area,
    standardized_nominal_distance,
    standardized_nominal_fit,
)

from .experiment import formal_panel_design, graph_for_panel
from .collective_observables import phase_collective_observables
from .simulation import memory_control_tape
from .workflow import (
    artifact_root,
    atomic_csv,
    atomic_json,
    load_yaml,
    execution_source_checksum,
    sha256_file,
    sha256_json,
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


def irreversibility_sensitivity_with_floor_uncertainty(
    codes: Sequence[int],
    block_lengths: Sequence[int],
    pseudocounts: Sequence[float],
    shuffle_replicates: int,
    seed: int,
) -> List[Dict[str, float]]:
    """Reproduce the frozen shuffle floor and retain its sampling audit.

    The permutation sequence and floor mean exactly match the V14/V15 frozen
    implementation.  Additional columns distinguish the empirical null spread
    from Monte Carlo uncertainty of the null mean.  They are a delayed
    reporting audit and do not modify the primary adjusted statistic.
    """

    values = np.asarray(codes, dtype=int)
    replicates = int(shuffle_replicates)
    if values.ndim != 1 or values.size < 4 or replicates < 2:
        raise ValueError("irreversibility audit needs one path and at least two shuffles")
    output: List[Dict[str, float]] = []
    for block in block_lengths:
        for pseudocount in pseudocounts:
            block_value = int(block)
            pseudocount_value = float(pseudocount)
            raw = block_time_reversal_kl(values, block_value, pseudocount_value)
            rng = np.random.default_rng(
                int(seed + 101 * block_value + round(100 * pseudocount_value))
            )
            samples = np.asarray(
                [
                    block_time_reversal_kl(
                        values[rng.permutation(values.size)],
                        block_value,
                        pseudocount_value,
                    )
                    for _ in range(replicates)
                ],
                dtype=float,
            )
            floor_mean = float(np.mean(samples))
            floor_sd = float(np.std(samples, ddof=1))
            floor_se = float(floor_sd / np.sqrt(samples.size))
            output.append(
                {
                    "block_length": block_value,
                    "pseudocount": pseudocount_value,
                    "raw_block_divergence_nats_per_update": float(raw),
                    "shuffle_floor_nats_per_update": floor_mean,
                    "shuffle_floor_median_nats_per_update": float(
                        np.median(samples)
                    ),
                    "shuffle_floor_null_ci_low": float(np.quantile(samples, 0.025)),
                    "shuffle_floor_null_ci_high": float(np.quantile(samples, 0.975)),
                    "shuffle_floor_null_sd": floor_sd,
                    "shuffle_floor_monte_carlo_se": floor_se,
                    "shuffle_floor_mean_mc_ci_low": float(
                        floor_mean - 1.96 * floor_se
                    ),
                    "shuffle_floor_mean_mc_ci_high": float(
                        floor_mean + 1.96 * floor_se
                    ),
                    "shuffle_replicates": float(samples.size),
                    "adjusted_irreversibility_nats_per_update": float(
                        raw - floor_mean
                    ),
                    "analysis_role": "post_reconstruction_floor_uncertainty_audit",
                }
            )
    return output


def _cluster_bootstrap_summary(
    values: Sequence[float], seed: int, replicates: int
) -> Dict[str, float]:
    """Summarize complete-trajectory values without resampling lower units."""

    observed = np.asarray(values, dtype=float)
    finite = observed[np.isfinite(observed)]
    if finite.size == 0:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "independent_clusters": 0.0,
            "clusters_total": float(observed.size),
            "undefined_clusters": float(observed.size),
            "bootstrap_replicates": float(replicates),
        }
    rng = np.random.default_rng(int(seed))
    # Every group contains only complete graph/environment trajectory values
    # (six per model in the formal design).  Drawing all cluster indices in one
    # array is exactly the same nonparametric bootstrap while avoiding millions
    # of Python-level RNG calls across the source-data summaries.
    indices = rng.integers(
        0,
        finite.size,
        size=(int(replicates), finite.size),
        endpoint=False,
    )
    means = finite[indices].mean(axis=1)
    return {
        "estimate": float(np.mean(finite)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "independent_clusters": float(finite.size),
        "clusters_total": float(observed.size),
        "undefined_clusters": float(observed.size - finite.size),
        "bootstrap_replicates": float(replicates),
    }


def _pooled_binder_bootstrap_summary(
    second_moments: Sequence[float],
    fourth_moments: Sequence[float],
    weights: Sequence[float],
    denominator_epsilon: float,
    seed: int,
    replicates: int,
) -> Dict[str, float]:
    """Pool cluster moments and bootstrap complete clusters.

    This is a pooling-rule sensitivity, not the primary cluster-first Binder
    summary.  Every bootstrap draw resamples complete trajectory clusters and
    then combines their moment numerators using their recorded update counts.
    """

    second = np.asarray(second_moments, dtype=float)
    fourth = np.asarray(fourth_moments, dtype=float)
    cluster_weights = np.asarray(weights, dtype=float)
    if not (second.shape == fourth.shape == cluster_weights.shape):
        raise ValueError("Binder moment and weight arrays must align")
    finite = (
        np.isfinite(second)
        & np.isfinite(fourth)
        & np.isfinite(cluster_weights)
        & (cluster_weights > 0.0)
    )
    second = second[finite]
    fourth = fourth[finite]
    cluster_weights = cluster_weights[finite]

    def pooled_value(
        selected_second: np.ndarray,
        selected_fourth: np.ndarray,
        selected_weights: np.ndarray,
    ) -> np.ndarray:
        totals = selected_weights.sum(axis=-1)
        second_value = (selected_second * selected_weights).sum(axis=-1) / totals
        fourth_value = (selected_fourth * selected_weights).sum(axis=-1) / totals
        with np.errstate(divide="ignore", invalid="ignore"):
            values = 1.0 - fourth_value / (3.0 * second_value ** 2)
        return np.where(second_value > float(denominator_epsilon), values, np.nan)

    if second.size:
        estimate = float(pooled_value(second, fourth, cluster_weights))
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(
            0,
            second.size,
            size=(int(replicates), second.size),
            endpoint=False,
        )
        samples = pooled_value(
            second[indices], fourth[indices], cluster_weights[indices]
        )
        finite_samples = samples[np.isfinite(samples)]
    else:
        estimate = float("nan")
        finite_samples = np.asarray([], dtype=float)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(finite_samples, 0.025))
        if finite_samples.size
        else float("nan"),
        "ci_high": float(np.quantile(finite_samples, 0.975))
        if finite_samples.size
        else float("nan"),
        "independent_clusters": float(second.size),
        "clusters_total": float(len(finite)),
        "undefined_clusters": float(len(finite) - second.size),
        "bootstrap_replicates": float(replicates),
        "valid_bootstrap_replicates": float(finite_samples.size),
    }


def _strict_cluster_values(values: Mapping[str, float]) -> str:
    """Serialize undefined descriptive values as JSON null, never NaN."""

    payload = {
        str(key): (float(value) if np.isfinite(float(value)) else None)
        for key, value in values.items()
    }
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def _collective_extension_panel(
    panel: Mapping[str, object],
    panel_root: str,
    extension: Mapping[str, object],
) -> Dict[str, List[Dict[str, object]]]:
    """Calculate secondary observables for one frozen V15 trajectory."""

    frame = pd.read_csv(Path(panel_root) / (str(panel["panel_id"]) + ".csv"))
    autocorrelation = extension["autocorrelation"]  # type: ignore[index]
    binder = extension["binder_cumulant"]  # type: ignore[index]
    calculated = phase_collective_observables(
        frame,
        graph_for_panel(panel).adjacency,
        int(panel["n_agents"]),
        int(autocorrelation["primary_lag_truncation_sweeps"]),  # type: ignore[index]
        autocorrelation["sensitivity_lag_truncation_sweeps"],  # type: ignore[index]
        float(binder["denominator_epsilon"]),  # type: ignore[index]
    )
    metadata = {
        "model_key": panel["model_key"],
        "cluster_id": panel["cluster_id"],
        "panel_id": panel["panel_id"],
        "condition": panel["condition"],
        "memory_mode": panel["memory_mode"],
        "n_agents": panel["n_agents"],
    }
    for rows in calculated.values():
        for row in rows:
            row.update(metadata)
    return calculated


def _summarize_collective_extension(
    frames: Mapping[str, pd.DataFrame], extension: Mapping[str, object]
) -> Dict[str, pd.DataFrame]:
    """Create model/condition summaries from complete trajectory estimates."""

    replicates = int(extension["uncertainty"]["replicates"])  # type: ignore[index]
    seed = int(extension["uncertainty"]["seed"])  # type: ignore[index]
    outputs: Dict[str, pd.DataFrame] = dict(frames)

    matrix = frames["connected_correlation_matrices"]
    outputs["connected_correlation_matrix_means"] = (
        matrix.groupby(
            ["model_key", "condition", "phase", "agent_i", "agent_j", "community_i", "community_j"],
            as_index=False,
        )
        .agg(
            connected_correlation=("connected_correlation", "mean"),
            between_cluster_sd=("connected_correlation", "std"),
            independent_clusters=("cluster_id", "nunique"),
        )
    )

    summary_specs = {
        "connected_correlation_profiles": (
            ["model_key", "condition", "phase", "graph_distance"],
            "connected_correlation",
            "connected_correlation_profile_summary",
        ),
        "autocorrelation_curves": (
            ["model_key", "condition", "phase", "lag_updates", "lag_sweeps"],
            "autocorrelation",
            "autocorrelation_curve_summary",
        ),
        "integrated_autocorrelation": (
            [
                "model_key",
                "condition",
                "phase",
                "lag_truncation_sweeps",
                "lag_truncation_updates",
                "is_primary",
            ],
            "integrated_autocorrelation_time_updates",
            "integrated_autocorrelation_summary",
        ),
        "binder_cumulants": (
            ["model_key", "condition", "phase"],
            "binder_cumulant",
            "binder_cumulant_summary",
        ),
        "magnetization_distributions": (
            ["model_key", "condition", "phase", "belief_magnetization"],
            "probability",
            "magnetization_distribution_summary",
        ),
    }
    for source_name, (groups, metric, destination_name) in summary_specs.items():
        rows: List[Dict[str, object]] = []
        for group_key, group in frames[source_name].groupby(groups, dropna=False):
            keys = group_key if isinstance(group_key, tuple) else (group_key,)
            label = ":".join(str(value) for value in keys)
            summary = _cluster_bootstrap_summary(
                group[metric].to_numpy(float),
                seed + _analysis_seed("extension:" + label),
                replicates,
            )
            rows.append({**dict(zip(groups, keys)), "metric": metric, **summary})
        outputs[destination_name] = pd.DataFrame(rows)

    binder_windows = frames["binder_cumulant_sensitivity"]
    binder_pooling_rows: List[Dict[str, object]] = []
    binder_groups = ["model_key", "condition", "phase", "temporal_window"]
    binder_epsilon = float(
        extension["binder_cumulant"]["denominator_epsilon"]  # type: ignore[index]
    )
    for group_key, group in binder_windows.groupby(binder_groups, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        label = ":".join(str(value) for value in keys)
        cluster_summary = _cluster_bootstrap_summary(
            group["binder_cumulant"].to_numpy(float),
            seed + _analysis_seed("extension:binder-cluster-mean:" + label),
            replicates,
        )
        binder_pooling_rows.append(
            {
                **dict(zip(binder_groups, keys)),
                "pooling_rule": "mean_of_cluster_cumulants",
                **cluster_summary,
            }
        )

        pooled_summary = _pooled_binder_bootstrap_summary(
            group["magnetization_second_moment"].to_numpy(float),
            group["magnetization_fourth_moment"].to_numpy(float),
            group["window_updates"].to_numpy(float),
            binder_epsilon,
            seed + _analysis_seed("extension:binder-pooled-moments:" + label),
            replicates,
        )
        binder_pooling_rows.append(
            {
                **dict(zip(binder_groups, keys)),
                "pooling_rule": "pooled_moments_across_clusters",
                **pooled_summary,
            }
        )
    outputs["binder_cumulant_pooling_sensitivity"] = pd.DataFrame(
        binder_pooling_rows
    )

    contrast_rows: List[Dict[str, object]] = []
    profiles = frames["connected_correlation_profiles"]
    persistence = frames["integrated_autocorrelation"]
    binders = frames["binder_cumulants"]
    for model in ("qwen", "granite"):
        model_profiles = profiles[
            (profiles["model_key"] == model)
            & (profiles["condition"] == "field_persistent")
            & np.isclose(profiles["graph_distance"], 1.0)
        ]
        for phase in ("disruption", "recovery"):
            effects = {}
            for cluster, group in model_profiles.groupby("cluster_id"):
                lookup = group.set_index("phase")["connected_correlation"]
                effects[str(cluster)] = float(lookup[phase] - lookup["baseline"])
            summary = _cluster_bootstrap_summary(
                list(effects.values()),
                seed + _analysis_seed("extension:%s:correlation:%s" % (model, phase)),
                replicates,
            )
            contrast_rows.append(
                {
                    "model_key": model,
                    "observable": "connected_graph_correlation",
                    "contrast": "%s_minus_baseline_at_graph_distance_1" % phase,
                    "unit": "dimensionless_connected_correlation",
                    **summary,
                    "cluster_values": _strict_cluster_values(effects),
                    "role": "secondary_descriptive_extension",
                }
            )

        model_persistence = persistence[
            (persistence["model_key"] == model)
            & (persistence["phase"] == "recovery")
            & persistence["is_primary"].astype(bool)
        ]
        effects = {}
        for cluster, group in model_persistence.groupby("cluster_id"):
            lookup = group.set_index("condition")[
                "integrated_autocorrelation_time_updates"
            ]
            effects[str(cluster)] = float(
                lookup["field_persistent"] - lookup["field_markovized"]
            )
        summary = _cluster_bootstrap_summary(
            list(effects.values()),
            seed + _analysis_seed("extension:%s:persistence" % model),
            replicates,
        )
        contrast_rows.append(
            {
                "model_key": model,
                "observable": "truncated_integrated_autocorrelation_time",
                "contrast": "persistent_minus_markovized_during_recovery",
                "unit": "attempted_updates",
                **summary,
                "cluster_values": _strict_cluster_values(effects),
                "role": "secondary_descriptive_extension",
            }
        )

        model_binders = binders[
            (binders["model_key"] == model)
            & (binders["condition"] == "field_markovized")
        ]
        for phase in ("disruption", "recovery"):
            effects = {}
            for cluster, group in model_binders.groupby("cluster_id"):
                lookup = group.set_index("phase")["binder_cumulant"]
                effects[str(cluster)] = float(lookup[phase] - lookup["baseline"])
            summary = _cluster_bootstrap_summary(
                list(effects.values()),
                seed + _analysis_seed("extension:%s:binder:%s" % (model, phase)),
                replicates,
            )
            contrast_rows.append(
                {
                    "model_key": model,
                    "observable": "binder_cumulant",
                    "contrast": "%s_minus_baseline_field_markovized" % phase,
                    "unit": "dimensionless",
                    **summary,
                    "cluster_values": _strict_cluster_values(effects),
                    "role": "secondary_descriptive_extension",
                }
            )
    outputs["collective_extension_contrasts"] = pd.DataFrame(contrast_rows)
    return outputs


def analyze_collective_extension(
    panels: Sequence[Mapping[str, object]],
    panel_root: Path,
    extension: Mapping[str, object],
    workers: int,
) -> Dict[str, pd.DataFrame]:
    """Analyze all panels and retain compact trajectory-level aggregates."""

    analyzed = Parallel(n_jobs=max(1, int(workers)), prefer="processes")(
        delayed(_collective_extension_panel)(panel, str(panel_root), extension)
        for panel in panels
    )
    names = (
        "correlation_profiles",
        "correlation_matrices",
        "autocorrelation_curves",
        "integrated_autocorrelation",
        "binder_cumulants",
        "binder_cumulant_sensitivity",
        "magnetization_distributions",
    )
    frames = {
        {
            "correlation_profiles": "connected_correlation_profiles",
            "correlation_matrices": "connected_correlation_matrices",
        }.get(name, name): pd.DataFrame(
            [row for panel_result in analyzed for row in panel_result[name]]
        )
        for name in names
    }
    return _summarize_collective_extension(frames, extension)


def _impute_training(training: np.ndarray, testing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train = np.asarray(training, dtype=float).copy()
    test = np.asarray(testing, dtype=float).copy()
    medians = np.nanmedian(train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    for column in range(train.shape[1]):
        train[~np.isfinite(train[:, column]), column] = medians[column]
        test[~np.isfinite(test[:, column]), column] = medians[column]
    return train, test


def _memory_entry_fields(entry: str) -> Dict[str, str]:
    fields = {}
    for token in str(entry).split(" "):
        if "=" not in token:
            raise ValueError("memory entry does not use the frozen key=value format")
        key, value = token.split("=", 1)
        fields[key] = value
    if set(fields) != {"t", "belief", "action", "memory"}:
        raise ValueError("memory entry fields differ from the frozen format")
    int(fields["t"])
    if fields["belief"] not in {"amber", "cobalt"} or fields["action"] not in {
        "amber",
        "cobalt",
    }:
        raise ValueError("memory entry contains an unknown state label")
    if fields["memory"] not in MEMORY_STATES:
        raise ValueError("memory entry contains an unknown bounded memory state")
    return fields


def memory_control_panel_audit(
    frame: pd.DataFrame, panel: Mapping[str, object]
) -> Dict[str, object]:
    """Reconstruct every displayed history entry from compact trajectory data."""

    condition = str(panel["condition"])
    mode = str(panel["memory_mode"])
    mapping = LatentMapping.balanced(int(panel["panel_seed"]) + 17011)
    if str(frame["latent_plus_label"].iloc[0]) != mapping.plus_label:
        raise RuntimeError("panel latent mapping differs from frozen reconstruction")
    reconstructed: List[Tuple[str, ...]] = []
    if mode == "scrambled_memory":
        tape = memory_control_tape(
            int(panel["n_agents"]),
            len(frame),
            int(panel["panel_seed"]),
            int(panel["control_seed"]),
            mapping,
        )
        reconstructed = [tuple(str(value) for value in row["entries"]) for row in tape]
    elif mode == "persistent_memory":
        histories: Dict[int, List[str]] = {
            agent: [] for agent in range(int(panel["n_agents"]))
        }
        for row in frame.sort_values("update").itertuples():
            agent = int(row.scheduled_agent)
            reconstructed.append(tuple(histories[agent][-3:]))
            if int(row.valid_after_repair):
                histories[agent].append(
                    "t=%d belief=%s action=%s memory=%s"
                    % (
                        int(row.update),
                        mapping.label(int(row.belief_after)),
                        mapping.label(int(row.action_after)),
                        str(row.memory_after),
                    )
                )
                histories[agent] = histories[agent][-3:]
    elif mode == "markovized":
        reconstructed = [tuple() for _ in range(len(frame))]
    else:
        raise ValueError("unknown V15 memory mode")

    ordered = frame.sort_values("update").reset_index(drop=True)
    if len(reconstructed) != len(ordered):
        raise AssertionError("memory reconstruction length mismatch")
    digest_matches = 0
    count_matches = 0
    character_matches = 0
    future_violations = 0
    labels = {"belief": {"amber": 0, "cobalt": 0}, "action": {"amber": 0, "cobalt": 0}}
    memory_counts = {state: 0 for state in MEMORY_STATES}
    entry_characters = []
    total_entries = 0
    for index, entries in enumerate(reconstructed):
        row = ordered.iloc[index]
        digest = sha256_json(list(entries))
        digest_matches += int(digest == str(row["memory_control_sha256"]))
        count_matches += int(len(entries) == int(row["prompt_memory_entry_count"]))
        characters = int(sum(len(value) for value in entries))
        character_matches += int(characters == int(row["prompt_memory_characters"]))
        entry_characters.append(characters)
        for entry in entries:
            fields = _memory_entry_fields(entry)
            future_violations += int(int(fields["t"]) >= int(row["update"]))
            labels["belief"][fields["belief"]] += 1
            labels["action"][fields["action"]] += 1
            memory_counts[fields["memory"]] += 1
            total_entries += 1
    prompt_tokens = ordered["prompt_tokens"].to_numpy(float)
    def fraction(count: int) -> float:
        return float(count / total_entries) if total_entries else float("nan")
    return {
        "model_key": panel["model_key"],
        "cluster_id": panel["cluster_id"],
        "panel_id": panel["panel_id"],
        "condition": condition,
        "memory_mode": mode,
        "updates": len(ordered),
        "total_displayed_memory_entries": total_entries,
        "mean_entries_per_prompt": float(total_entries / max(len(ordered), 1)),
        "mean_memory_characters_per_prompt": float(np.mean(entry_characters)),
        "sd_memory_characters_per_prompt": float(np.std(entry_characters, ddof=1)),
        "memory_characters_q05": float(np.quantile(entry_characters, 0.05)),
        "memory_characters_q50": float(np.quantile(entry_characters, 0.50)),
        "memory_characters_q95": float(np.quantile(entry_characters, 0.95)),
        "mean_prompt_tokens": float(np.mean(prompt_tokens)),
        "sd_prompt_tokens": float(np.std(prompt_tokens, ddof=1)),
        "prompt_token_q05": float(np.quantile(prompt_tokens, 0.05)),
        "prompt_token_q50": float(np.quantile(prompt_tokens, 0.50)),
        "prompt_token_q95": float(np.quantile(prompt_tokens, 0.95)),
        "belief_amber_fraction_in_memory": fraction(labels["belief"]["amber"]),
        "belief_cobalt_fraction_in_memory": fraction(labels["belief"]["cobalt"]),
        "action_amber_fraction_in_memory": fraction(labels["action"]["amber"]),
        "action_cobalt_fraction_in_memory": fraction(labels["action"]["cobalt"]),
        **{
            "memory_%s_fraction" % state: fraction(count)
            for state, count in memory_counts.items()
        },
        "digest_matches": digest_matches,
        "entry_count_matches": count_matches,
        "character_count_matches": character_matches,
        "future_information_violations": future_violations,
        "all_entries_reconstructed": bool(
            digest_matches == len(ordered)
            and count_matches == len(ordered)
            and character_matches == len(ordered)
        ),
        "donor_agent_state_used_by_construction": False,
        "peer_private_state_used_by_construction": False,
    }


def memory_control_balance_audit(panel_audit: pd.DataFrame) -> pd.DataFrame:
    """Pair persistent and synthetic histories within model and cluster."""

    metrics = (
        "mean_entries_per_prompt",
        "mean_memory_characters_per_prompt",
        "memory_characters_q05",
        "memory_characters_q50",
        "memory_characters_q95",
        "mean_prompt_tokens",
        "sd_prompt_tokens",
        "prompt_token_q05",
        "prompt_token_q50",
        "prompt_token_q95",
        "belief_amber_fraction_in_memory",
        "belief_cobalt_fraction_in_memory",
        "action_amber_fraction_in_memory",
        "action_cobalt_fraction_in_memory",
    ) + tuple("memory_%s_fraction" % state for state in MEMORY_STATES)
    rows = []
    for (model, cluster), group in panel_audit.groupby(["model_key", "cluster_id"]):
        lookup = group.set_index("condition")
        persistent = lookup.loc["field_persistent"]
        scrambled = lookup.loc["field_scrambled"]
        row: Dict[str, object] = {"model_key": model, "cluster_id": cluster}
        for metric in metrics:
            row["persistent_%s" % metric] = persistent[metric]
            row["scrambled_%s" % metric] = scrambled[metric]
            row["persistent_minus_scrambled_%s" % metric] = float(
                persistent[metric] - scrambled[metric]
            )
        row["both_controls_fully_reconstructed"] = bool(
            persistent["all_entries_reconstructed"]
            and scrambled["all_entries_reconstructed"]
        )
        row["future_information_violations"] = int(
            persistent["future_information_violations"]
            + scrambled["future_information_violations"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


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
    sensitivity = irreversibility_sensitivity_with_floor_uncertainty(
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
        "shuffle_floor_monte_carlo_se": primary[
            "shuffle_floor_monte_carlo_se"
        ],
        "shuffle_floor_mean_mc_ci_low": primary[
            "shuffle_floor_mean_mc_ci_low"
        ],
        "shuffle_floor_mean_mc_ci_high": primary[
            "shuffle_floor_mean_mc_ci_high"
        ],
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
        "memory_control_audit": memory_control_panel_audit(frame, panel),
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


def cluster_seed_audit(panels: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Verify that matching occurs within, and seed independence across, clusters."""

    frame = pd.DataFrame(panels)
    rows: List[Dict[str, object]] = []
    expected_conditions = {
        "nominal_markovized",
        "field_markovized",
        "field_persistent",
        "field_scrambled",
    }
    for (model, cluster), group in frame.groupby(["model_key", "cluster_id"]):
        unique_seeds = {
            field: sorted(int(value) for value in group[field].unique())
            for field in ("panel_seed", "graph_seed", "control_seed")
        }
        rows.append(
            {
                "model_key": model,
                "cluster_id": cluster,
                "panel_seed": unique_seeds["panel_seed"][0],
                "graph_seed": unique_seeds["graph_seed"][0],
                "control_seed": unique_seeds["control_seed"][0],
                "conditions": json.dumps(sorted(str(value) for value in group["condition"])),
                "matched_seed_within_cluster": all(
                    len(values) == 1 for values in unique_seeds.values()
                ),
                "complete_four_arm_panel": set(group["condition"]) == expected_conditions,
            }
        )
    output = pd.DataFrame(rows).sort_values(["model_key", "cluster_id"]).reset_index(
        drop=True
    )
    for field in ("panel_seed", "graph_seed", "control_seed"):
        output[field + "_globally_unique"] = ~output[field].duplicated(keep=False)
    output["model_seed_namespaces_disjoint"] = True
    for field in ("panel_seed", "graph_seed", "control_seed"):
        qwen = set(output.loc[output["model_key"] == "qwen", field])
        granite = set(output.loc[output["model_key"] == "granite", field])
        if qwen.intersection(granite):
            output["model_seed_namespaces_disjoint"] = False
    if not bool(
        output[
            [
                "matched_seed_within_cluster",
                "complete_four_arm_panel",
                "panel_seed_globally_unique",
                "graph_seed_globally_unique",
                "control_seed_globally_unique",
                "model_seed_namespaces_disjoint",
            ]
        ].to_numpy(bool).all()
    ):
        raise RuntimeError("V15 graph/environment cluster seed audit failed")
    return output


def raw_generation_accounting_audit(
    panel_root: Path,
    raw_root: Path,
    allow_synthetic_missing_records: bool = False,
) -> pd.DataFrame:
    """Reconcile content-addressed model records with completed transitions.

    A panel is written atomically only after all of its decisions complete. If
    infrastructure interrupts a panel, its already persisted call records are
    scientifically unused but still consumed compute. This audit keeps those
    orphaned attempts separate from the records referenced by completed panel
    rows. No raw prompt or completion content enters the repository-facing
    aggregate.
    """

    panel_root = Path(panel_root)
    raw_root = Path(raw_root)
    references: Dict[str, str] = {}
    duplicate_references = 0
    for path in sorted(panel_root.glob("*.csv")):
        frame = pd.read_csv(path, usecols=["model_key", "raw_artifact_sha256"])
        for row in frame.itertuples(index=False):
            digest = str(row.raw_artifact_sha256)
            if not digest or digest.lower() == "nan":
                continue
            if len(digest) != 64:
                raise RuntimeError("completed transition has an invalid raw-record digest")
            if digest in references:
                duplicate_references += 1
            references[digest] = str(row.model_key)

    record_rows: List[Dict[str, object]] = []
    observed_digests = set()
    for model in ("qwen", "granite"):
        for path in sorted((raw_root / model).glob("call_*.json")):
            digest = sha256_file(path)
            if digest in observed_digests:
                raise RuntimeError("duplicate content-addressed raw generation record")
            observed_digests.add(digest)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload_model = str(payload.get("model_key", ""))
            if payload_model != model:
                raise RuntimeError("raw generation record is stored under the wrong model")
            if not path.stem.endswith(digest[:12]):
                raise RuntimeError("raw generation filename does not match its content digest")
            if digest in references and references[digest] != model:
                raise RuntimeError("panel reference and raw generation model disagree")
            record_rows.append(
                {
                    "model_key": model,
                    "raw_artifact_sha256": digest,
                    "accounting_scope": (
                        "retained_panel" if digest in references else "orphan_interrupted_attempt"
                    ),
                    "decision_requests": 1,
                    "model_calls": int(payload["model_calls"]),
                    "prompt_tokens": int(payload["prompt_tokens"]),
                    "generated_tokens": int(payload["generated_tokens"]),
                    "latency_seconds": float(payload["latency_seconds"]),
                    "first_pass_valid": int(bool(payload["first_pass_valid"])),
                    "repair_attempted": int(bool(payload["repaired"]) or int(payload["model_calls"]) > 1),
                    "valid_after_repair": int(bool(payload["valid"])),
                    "raw_record_bytes": int(path.stat().st_size),
                }
            )

    if not record_rows and (not references or allow_synthetic_missing_records):
        return pd.DataFrame(
            [
                {
                    "model_key": "synthetic",
                    "accounting_scope": "not_applicable",
                    "status": "not_applicable_no_external_records",
                    "record_count": 0,
                    "decision_requests": 0,
                    "model_calls": 0,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "latency_seconds": 0.0,
                    "first_pass_valid": 0,
                    "repair_attempted": 0,
                    "valid_after_repair": 0,
                    "raw_record_bytes": 0,
                    "referenced_decisions_expected": int(len(references)),
                    "missing_referenced_records": 0,
                    "duplicate_reference_count": 0,
                    "raw_record_count_total": 0,
                    "orphan_record_count_total": 0,
                }
            ]
        )

    missing = sorted(set(references).difference(observed_digests))
    if missing:
        raise RuntimeError("completed transitions reference missing raw generation records")
    if duplicate_references:
        raise RuntimeError("multiple completed transitions reference one generation record")

    records = pd.DataFrame(record_rows)
    rows: List[Dict[str, object]] = []
    orphan_total = int(
        np.sum(records["accounting_scope"] == "orphan_interrupted_attempt")
    )
    for model in ("qwen", "granite"):
        for scope in ("retained_panel", "orphan_interrupted_attempt"):
            selected = records[
                (records["model_key"] == model)
                & (records["accounting_scope"] == scope)
            ]
            rows.append(
                {
                    "model_key": model,
                    "accounting_scope": scope,
                    "status": "passed",
                    "record_count": int(len(selected)),
                    **{
                        column: (
                            float(selected[column].sum())
                            if column == "latency_seconds"
                            else int(selected[column].sum())
                        )
                        for column in (
                            "decision_requests",
                            "model_calls",
                            "prompt_tokens",
                            "generated_tokens",
                            "latency_seconds",
                            "first_pass_valid",
                            "repair_attempted",
                            "valid_after_repair",
                            "raw_record_bytes",
                        )
                    },
                    "referenced_decisions_expected": int(len(references)),
                    "missing_referenced_records": 0,
                    "duplicate_reference_count": 0,
                    "raw_record_count_total": int(len(records)),
                    "orphan_record_count_total": orphan_total,
                }
            )
    return pd.DataFrame(rows)


def model_stratified_sensitivity(
    panels: pd.DataFrame, quench: pd.DataFrame
) -> pd.DataFrame:
    """Decompose frozen pooled contrasts without redefining confirmation.

    Qwen and Granite use nonoverlapping graph/environment seed namespaces.
    This table nevertheless reports each six-cluster model stratum separately
    so a pooled result cannot be misread as within-family replication.
    """

    rows: List[Dict[str, object]] = []
    metric = "adjusted_pathwise_irreversibility_nats_per_update"
    definitions = {
        "field_peak_minus_nominal": (
            "maximum_post_quench_distance",
            "field_markovized",
            "nominal_markovized",
            "distance_units",
        ),
        "persistent_minus_markovized": (
            metric,
            "field_persistent",
            "field_markovized",
            "nats_per_attempted_update",
        ),
        "persistent_minus_scrambled": (
            metric,
            "field_persistent",
            "field_scrambled",
            "nats_per_attempted_update",
        ),
    }
    for model in ("qwen", "granite"):
        for contrast, (column, left_condition, right_condition, unit) in definitions.items():
            source = quench if column == "maximum_post_quench_distance" else panels
            effects: Dict[str, float] = {}
            for cluster, group in source[source["model_key"] == model].groupby("cluster_id"):
                lookup = group.set_index("condition")[column]
                effects[str(cluster)] = float(lookup[left_condition] - lookup[right_condition])
            interval = _bootstrap(effects, _analysis_seed("stratum:%s:%s" % (model, contrast)))
            rows.append(
                {
                    "model_key": model,
                    "contrast": contrast,
                    "unit": unit,
                    **interval,
                    "exact_one_sided_sign_flip_p": exact_sign_flip_p(list(effects.values()), 1),
                    "positive_clusters": int(sum(value > 0.0 for value in effects.values())),
                    "cluster_values": json.dumps(effects, sort_keys=True),
                    "role": "descriptive_model_stratified_sensitivity",
                    "confirmatory_disposition_assigned": False,
                }
            )
        recovery_effects = {
            str(row.cluster_id): float(row.fixed_early_minus_late_recovery_distance)
            for row in quench[
                (quench["model_key"] == model)
                & (quench["condition"] == "field_markovized")
            ].itertuples()
        }
        interval = _bootstrap(
            recovery_effects, _analysis_seed("stratum:%s:fixed_recovery" % model)
        )
        rows.append(
            {
                "model_key": model,
                "contrast": "fixed_early_minus_late_recovery_distance",
                "unit": "distance_units",
                **interval,
                "exact_one_sided_sign_flip_p": exact_sign_flip_p(
                    list(recovery_effects.values()), 1
                ),
                "positive_clusters": int(
                    sum(value > 0.0 for value in recovery_effects.values())
                ),
                "cluster_values": json.dumps(recovery_effects, sort_keys=True),
                "role": "descriptive_model_stratified_sensitivity",
                "confirmatory_disposition_assigned": False,
            }
        )
    return pd.DataFrame(rows)


def analyze_formal(
    repository: Path,
    result_root_override: Path | None = None,
    allow_synthetic_raw_records: bool = False,
) -> Dict[str, object]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    repository = Path(repository).resolve()
    protocol_path = repository / "configs/statmech_v15/protocol_frozen.yaml"
    protocol = load_yaml(protocol_path)
    extension_path = repository / "configs/statmech_v15/collective_extension.yaml"
    extension = load_yaml(extension_path)
    completion_path = artifact_root() / "formal/completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["status"] != "complete":
        raise RuntimeError("V15 formal execution is incomplete")
    panels = formal_panel_design(protocol)
    seed_audit = cluster_seed_audit(panels)
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
    memory_control_audit = pd.DataFrame(
        [result["memory_control_audit"] for result in analyzed]
    )
    memory_control_balance = memory_control_balance_audit(memory_control_audit)
    macro, nominal_diagnostics, thresholds = fit_nominal_distances(macro, protocol)
    primary_window = int(protocol["analysis"]["primary_window_sweeps"])  # type: ignore[index]
    primary_macro = macro[macro["window_sweeps"] == primary_window].copy()
    quench = quench_summaries(primary_macro, protocol, thresholds)
    effects, dispositions = primary_hypotheses(panel_table, quench, protocol)
    prompt_balance = memory_prompt_balance(panel_table)
    stratified = model_stratified_sensitivity(panel_table, quench)
    collective_extension = analyze_collective_extension(
        panels,
        artifact_root() / "formal/panels",
        extension,  # type: ignore[arg-type]
        workers,
    )
    raw_accounting = raw_generation_accounting_audit(
        artifact_root() / "formal/panels",
        artifact_root() / "raw/formal",
        allow_synthetic_missing_records=bool(allow_synthetic_raw_records),
    )
    if set(raw_accounting["status"]) == {"passed"}:
        retained = raw_accounting[
            raw_accounting["accounting_scope"] == "retained_panel"
        ]
        expected_accounting = {
            "decision_requests": int(completion["observed_decision_rows"]),
            "model_calls": int(completion["model_calls"]),
            "prompt_tokens": int(completion["prompt_tokens"]),
            "generated_tokens": int(completion["generated_tokens"]),
        }
        observed_accounting = {
            key: int(retained[key].sum()) for key in expected_accounting
        }
        if observed_accounting != expected_accounting:
            raise RuntimeError("retained raw generation accounting disagrees with completion")
        if not np.isclose(
            float(retained["latency_seconds"].sum()),
            3600.0 * float(completion["generation_gpu_hours"]),
            rtol=1.0e-12,
            atol=1.0e-9,
        ):
            raise RuntimeError("retained raw latency disagrees with completion")
    result = (
        Path(result_root_override).resolve()
        if result_root_override is not None
        else repository / "results/collective_agent_statmech_v15"
    )
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
        "memory_control_panel_audit.csv": memory_control_audit,
        "memory_control_balance_audit.csv": memory_control_balance,
        "hypothesis_model_stratified.csv": stratified,
        "cluster_seed_audit.csv": seed_audit,
        "raw_generation_accounting.csv": raw_accounting,
    }
    for name, frame in collective_extension.items():
        outputs[name + ".csv"] = frame
    for name, frame in outputs.items():
        atomic_csv(frame, tables / name)
    primary = {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "collective_extension_sha256": sha256_file(extension_path),
        "analysis_source_sha256": execution_source_checksum(repository),
        "formal_completion": completion,
        "formal_trajectories": len(panel_table),
        "independent_clusters_per_model": int(protocol["network"]["clusters_per_model"]),  # type: ignore[index]
        "model_keys": sorted(panel_table["model_key"].unique()),
        "cluster_seed_audit_passed": True,
        "confirmatory_dispositions": dispositions,
        "privacy_mutations": int(panel_table["privacy_mutations"].sum()),
        "memory_control_audit": {
            "panels_fully_reconstructed": int(
                memory_control_audit["all_entries_reconstructed"].sum()
            ),
            "panels_total": int(len(memory_control_audit)),
            "future_information_violations": int(
                memory_control_audit["future_information_violations"].sum()
            ),
            "donor_agent_state_used": bool(
                memory_control_audit["donor_agent_state_used_by_construction"].any()
            ),
            "peer_private_state_used": bool(
                memory_control_audit["peer_private_state_used_by_construction"].any()
            ),
        },
        "raw_generation_accounting_audit": {
            "status": str(raw_accounting["status"].iloc[0]),
            "referenced_records": int(
                raw_accounting.loc[
                    raw_accounting["accounting_scope"] == "retained_panel",
                    "record_count",
                ].sum()
            ),
            "orphan_interrupted_records": int(
                raw_accounting.loc[
                    raw_accounting["accounting_scope"]
                    == "orphan_interrupted_attempt",
                    "record_count",
                ].sum()
            ),
            "missing_referenced_records": int(
                raw_accounting["missing_referenced_records"].max()
            ),
        },
        "nonfinite_primary_features": int(
            np.sum(~np.isfinite(primary_macro[list(V15_Z_FEATURES)].to_numpy(float)))
        ),
        "analysis_wall_seconds": float(time.perf_counter() - started),
        "analysis_cpu_seconds": float(time.process_time() - cpu_started),
        "external_formal_tree": tree_digest(artifact_root() / "formal"),
        "external_raw_tree": tree_digest(artifact_root() / "raw/formal"),
        "collective_extension": {
            "version": extension["version"],  # type: ignore[index]
            "status": extension["status"],  # type: ignore[index]
            "frozen_protocol_modified": bool(extension["frozen_protocol_modified"]),  # type: ignore[index]
            "output_tables": {
                name: len(frame) for name, frame in collective_extension.items()
            },
        },
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
    "analyze_collective_extension",
    "cluster_seed_audit",
    "fit_nominal_distances",
    "memory_control_balance_audit",
    "memory_control_panel_audit",
    "memory_prompt_balance",
    "model_stratified_sensitivity",
    "primary_hypotheses",
    "quench_summaries",
    "raw_generation_accounting_audit",
]
