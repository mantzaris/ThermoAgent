"""Panel-level V12 statistical-mechanics analysis and compact aggregation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .estimators import (
    autocorrelation,
    binary_mutual_information,
    binary_transfer_entropy,
    block_time_reversal_kl,
    conditional_mutual_information_history,
    gini_simpson,
    graph_distance_correlations,
    history_log_likelihood,
    hysteresis_area,
    information_permutation_floor,
    integrated_autocorrelation_time,
    jensen_shannon_divergence,
    markov_entropy_production,
    miller_madow_entropy,
    occupied_transition_counts,
    paired_cluster_bootstrap,
    probability_currents,
    row_stochastic,
    shannon_entropy,
    stationary_distribution,
    susceptibility,
    time_shuffle_floor,
    transition_pair_irreversibility,
    tsallis_entropy,
)
from .experiment import _graph_for_panel, formal_panel_design
from .graphs import build_delivery_graph, matched_opportunity_schedule, select_recipient
from .workflow import artifact_root, atomic_csv, atomic_json, load_yaml, sha256_file, utc_now


def _bits(value: str) -> np.ndarray:
    return np.asarray([int(item) for item in str(value).split(";")], dtype=int)


def _float_bits(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in str(value).split(";")], dtype=float)


def _bin(value: float, width: float) -> int:
    return int(np.floor(float(value) / float(width) + 0.5))


def _macrostate_sequence(frame: pd.DataFrame, widths: Mapping[str, object]) -> Tuple[np.ndarray, Dict[Tuple[int, ...], int]]:
    first = frame.iloc[0]
    tuples = [
        (
            _bin(first["belief_magnetization_before"], float(widths["belief_magnetization"])),
            _bin(first["action_magnetization_before"], float(widths["action_magnetization"])),
            _bin(first["belief_action_overlap_before"], float(widths["belief_action_overlap"])),
            _bin(first["reference_energy_per_agent_before"], float(widths["reference_energy_per_agent"])),
        )
    ] + [
        (
            _bin(row.belief_magnetization, float(widths["belief_magnetization"])),
            _bin(row.action_magnetization, float(widths["action_magnetization"])),
            _bin(row.belief_action_overlap, float(widths["belief_action_overlap"])),
            _bin(row.reference_energy_per_agent, float(widths["reference_energy_per_agent"])),
        )
        for row in frame.itertuples()
    ]
    mapping = {value: index for index, value in enumerate(sorted(set(tuples)))}
    return np.asarray([mapping[value] for value in tuples], dtype=int), mapping


def _configuration_entropy(states: Sequence[int]) -> Tuple[float, float, float, float, int]:
    _, counts = np.unique(np.asarray(states, dtype=int), return_counts=True)
    probability = counts / counts.sum()
    return (
        shannon_entropy(probability),
        miller_madow_entropy(counts),
        tsallis_entropy(probability, 0.5),
        tsallis_entropy(probability, 2.0),
        int(counts.size),
    )


def _early_late_js(states: Sequence[int]) -> float:
    values = np.asarray(states, dtype=int)
    split = values.size // 2
    alphabet = sorted(int(value) for value in np.unique(values))
    mapping = {value: index for index, value in enumerate(alphabet)}
    first = np.zeros(len(alphabet), dtype=float)
    second = np.zeros(len(alphabet), dtype=float)
    for value in values[:split]:
        first[mapping[int(value)]] += 1.0
    for value in values[split:]:
        second[mapping[int(value)]] += 1.0
    return jensen_shannon_divergence(first, second)


def _layer_path(frame: pd.DataFrame, column: str, n_agents: int) -> np.ndarray:
    before_column = "beliefs_before_vector" if column == "beliefs" else "actions_before_vector"
    vectors = np.vstack([_bits(frame.iloc[0][before_column])] + [_bits(value) for value in frame[column]])
    if int(n_agents) <= 4:
        weights = (1 << np.arange(int(n_agents), dtype=np.int64))[None, :]
        return np.sum((vectors > 0).astype(np.int64) * weights, axis=1).astype(int)
    # A fixed-width magnetization projection is the predeclared scalable layer
    # macrostate; it is not represented as a full microscopic Markov state.
    return np.rint(4.0 * np.mean(vectors, axis=1)).astype(int) + 4


def _augmented_state_path(
    frame: pd.DataFrame, n_agents: int, settings: Mapping[str, object]
) -> Tuple[np.ndarray, str]:
    snapshots: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    first = frame.iloc[0]
    snapshots.append(
        (
            _bits(first["beliefs_before_vector"]),
            _bits(first["actions_before_vector"]),
            _float_bits(first["confidences_before_vector"]),
            _bits(first["commitments_before_vector"]),
            _bits(first["memory_states_before_vector"]),
            _bits(first["workloads_before_vector"]),
        )
    )
    for row in frame.itertuples():
        snapshots.append(
            (
                _bits(row.beliefs),
                _bits(row.actions),
                _float_bits(row.confidences),
                _bits(row.commitments),
                _bits(row.memory_states),
                _bits(row.workloads),
            )
        )
    tuples: List[Tuple[int, ...]] = []
    if int(n_agents) <= 4:
        width = float(settings["confidence_bin_width"])
        for beliefs, actions, confidence, commitments, memories, workloads in snapshots:
            tuples.append(
                tuple(beliefs.tolist())
                + tuple(actions.tolist())
                + tuple(np.rint(confidence / width).astype(int).tolist())
                + tuple(commitments.tolist())
                + tuple(memories.tolist())
                + tuple(workloads.tolist())
            )
        representation = "full_categorical_plus_binned_confidence_workload"
    else:
        confidence_width = float(settings["confidence_bin_width"])
        fraction_width = float(settings["categorical_fraction_bin_width"])
        workload_width = float(settings["workload_mean_bin_width"])
        for beliefs, actions, confidence, commitments, memories, workloads in snapshots:
            tuple_value = [
                int(np.rint(4.0 * np.mean(beliefs))),
                int(np.rint(4.0 * np.mean(actions))),
                int(np.rint(4.0 * np.mean(beliefs * actions))),
                int(np.rint(np.mean(confidence) / confidence_width)),
                int(np.rint(np.mean(workloads) / workload_width)),
            ]
            tuple_value.extend(
                int(np.rint(np.mean(commitments == code) / fraction_width)) for code in range(5)
            )
            tuple_value.extend(
                int(np.rint(np.mean(memories == code) / fraction_width)) for code in range(5)
            )
            tuples.append(tuple(tuple_value))
        representation = "predeclared_augmented_macrostate"
    mapping = {value: index for index, value in enumerate(sorted(set(tuples)))}
    return np.asarray([mapping[value] for value in tuples], dtype=int), representation


def _projected_layer_irreversibility(
    path: Sequence[int], pseudocount: float, shuffle_replicates: int, seed: int
) -> Dict[str, float]:
    values = np.asarray(path, dtype=int)
    counts, _ = occupied_transition_counts(values)
    kernel = row_stochastic(counts, pseudocount)
    raw_block = block_time_reversal_kl(values, 3, pseudocount)
    floor = time_shuffle_floor(values, 3, pseudocount, shuffle_replicates, seed)
    return {
        "markov_epr": markov_entropy_production(stationary_distribution(kernel), kernel),
        "block_kl": raw_block,
        "shuffle_floor": floor["mean"],
        "adjusted_block_kl": raw_block - floor["mean"],
    }


def _panel_statistics(
    frame: pd.DataFrame,
    protocol: Mapping[str, object],
    panel_definition: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    burn = int(panel_definition["burn_in_sweeps"] * panel_definition["n_agents"])
    retained = frame.iloc[burn:].copy()
    if retained.shape[0] < 8:
        raise RuntimeError("panel has too few retained attempted updates")
    state_path = np.concatenate(
        [np.asarray([int(retained.iloc[0]["state_before"])]), retained["state_after"].to_numpy(int)]
    )
    widths = protocol["analysis"]["macrostate_bin_widths"]  # type: ignore[index]
    macro_path, macro_mapping = _macrostate_sequence(retained, widths)
    analysis_path = state_path if int(panel_definition["n_agents"]) <= 4 else macro_path
    representation = "belief_action_microstate" if int(panel_definition["n_agents"]) <= 4 else "predeclared_macrostate"
    counts, remap = occupied_transition_counts(analysis_path)
    pseudocount = float(protocol["analysis"]["primary_pseudocount"])  # type: ignore[index]
    kernel = row_stochastic(counts, pseudocount)
    stationary = stationary_distribution(kernel)
    current = probability_currents(stationary, kernel)
    epr = markov_entropy_production(stationary, kernel)
    pair_irreversibility = transition_pair_irreversibility(counts, pseudocount)
    block_values = {
        int(length): block_time_reversal_kl(analysis_path, int(length), pseudocount)
        for length in protocol["analysis"]["block_lengths"]  # type: ignore[index]
        if len(analysis_path) >= int(length)
    }
    primary_length = int(protocol["analysis"]["primary_block_length"])  # type: ignore[index]
    floor = time_shuffle_floor(
        analysis_path,
        primary_length,
        pseudocount,
        int(protocol["analysis"]["time_shuffle_replicates_per_panel"]),  # type: ignore[index]
        int(protocol["analysis"]["analysis_seed"]) + sum(ord(value) for value in str(panel_definition["panel_id"])),  # type: ignore[index]
    )
    augmented_path, augmented_representation = _augmented_state_path(
        retained,
        int(panel_definition["n_agents"]),
        protocol["analysis"]["augmented_state_coarse_graining"],  # type: ignore[index]
    )
    augmented_block = block_time_reversal_kl(augmented_path, primary_length, pseudocount)
    augmented_floor = time_shuffle_floor(
        augmented_path,
        primary_length,
        pseudocount,
        int(protocol["analysis"]["time_shuffle_replicates_per_panel"]),  # type: ignore[index]
        int(protocol["analysis"]["analysis_seed"]) + 2000 + sum(ord(value) for value in str(panel_definition["panel_id"])),  # type: ignore[index]
    )
    config_entropy, config_mm, config_t05, config_t2, occupied = _configuration_entropy(state_path)
    macro_entropy, macro_mm, macro_t05, macro_t2, occupied_macro = _configuration_entropy(macro_path)
    beliefs = np.vstack([_bits(value) for value in retained["beliefs"]])
    actions = np.vstack([_bits(value) for value in retained["actions"]])
    graph = _graph_for_panel(panel_definition)
    graph_diagnostics = graph.diagnostics()
    belief_correlations = graph_distance_correlations(beliefs, graph.adjacency)
    action_correlations = graph_distance_correlations(actions, graph.adjacency)
    edge_te: List[float] = []
    edge_mi: List[float] = []
    edge_te_floor: List[float] = []
    edge_mi_floor: List[float] = []
    information_replicates = int(protocol["analysis"]["information_permutation_replicates_per_panel"])  # type: ignore[index]
    information_seed = (
        int(protocol["analysis"]["analysis_seed"])
        + 5000
        + sum(ord(value) for value in str(panel_definition["panel_id"]))
    )  # type: ignore[index]
    for source, target in zip(*np.nonzero(graph.adjacency)):
        if source >= target:
            continue
        edge_mi.append(binary_mutual_information(beliefs[:, source], beliefs[:, target], 0.5))
        edge_te.append(binary_transfer_entropy(beliefs[:, source], beliefs[:, target], 0.5))
        edge_te.append(binary_transfer_entropy(beliefs[:, target], beliefs[:, source], 0.5))
        forward_floor = information_permutation_floor(
            beliefs[:, source], beliefs[:, target], 0.5, information_replicates, information_seed + 101 * source + target
        )
        reverse_floor = information_permutation_floor(
            beliefs[:, target], beliefs[:, source], 0.5, information_replicates, information_seed + 101 * target + source
        )
        edge_mi_floor.append(forward_floor["mutual_information_mean"])
        edge_te_floor.extend([forward_floor["transfer_entropy_mean"], reverse_floor["transfer_entropy_mean"]])
    markov = {}
    for history in protocol["analysis"]["history_orders"]:  # type: ignore[index]
        order = int(history)
        markov["history_%d_log_likelihood" % order] = (
            history_log_likelihood(analysis_path, order, 0.5) if len(analysis_path) >= 4 * order + 4 else float("nan")
        )
        if order <= 2 and len(analysis_path) > order + 2:
            markov["history_%d_conditional_mutual_information" % order] = conditional_mutual_information_history(
                analysis_path, order, 0.1
            )
        markov["augmented_history_%d_log_likelihood" % order] = (
            history_log_likelihood(augmented_path, order, 0.5)
            if len(augmented_path) >= 4 * order + 4
            else float("nan")
        )
        if order <= 2 and len(augmented_path) > order + 2:
            markov["augmented_history_%d_conditional_mutual_information" % order] = (
                conditional_mutual_information_history(augmented_path, order, 0.1)
            )
    sensitivity = {}
    for value in protocol["analysis"]["pseudocounts"]:  # type: ignore[index]
        pc = float(value)
        sensitivity["pair_irreversibility_pc_%s" % str(value).replace(".", "_")] = transition_pair_irreversibility(counts, pc)
        sensitivity["markov_epr_pc_%s" % str(value).replace(".", "_")] = markov_entropy_production(
            stationary_distribution(row_stochastic(counts, pc)), row_stochastic(counts, pc)
        )
    top_currents: List[Dict[str, object]] = []
    inverse = {mapped: original for original, mapped in remap.items()}
    indices = np.dstack(np.unravel_index(np.argsort(np.abs(current).ravel())[::-1], current.shape))[0]
    seen = 0
    for source, target in indices:
        if source >= target or abs(current[source, target]) <= 0.0:
            continue
        top_currents.append(
            {
                "panel_id": panel_definition["panel_id"],
                "rank": seen + 1,
                "source_state": int(inverse[int(source)]),
                "target_state": int(inverse[int(target)]),
                "current": float(current[source, target]),
                "state_representation": representation,
            }
        )
        seen += 1
        if seen == 8:
            break
    belief_m = retained["belief_magnetization"].to_numpy(float)
    action_m = retained["action_magnetization"].to_numpy(float)
    maximum_lag = min(20, max(1, len(belief_m) // 4))
    layer_shuffle = min(100, int(protocol["analysis"]["time_shuffle_replicates_per_panel"]))  # type: ignore[index]
    belief_layer = _projected_layer_irreversibility(
        _layer_path(retained, "beliefs", int(panel_definition["n_agents"])), pseudocount, layer_shuffle, information_seed + 7101
    )
    action_layer = _projected_layer_irreversibility(
        _layer_path(retained, "actions", int(panel_definition["n_agents"])), pseudocount, layer_shuffle, information_seed + 9101
    )
    row: Dict[str, object] = {
        "family": panel_definition["family"],
        "cluster_id": panel_definition["cluster_id"],
        "panel_id": panel_definition["panel_id"],
        "n_agents": int(panel_definition["n_agents"]),
        "topology": panel_definition["topology"],
        "alpha": float(panel_definition["alpha"]),
        "orientation": panel_definition["orientation"],
        "regime": panel_definition["regime"],
        "control": panel_definition["control"],
        "sampling_temperature": float(panel_definition["sampling_temperature"]),
        "coupling_strength": float(panel_definition["coupling_strength"]),
        "initial_condition": panel_definition["initial_condition"],
        "attempted_updates": int(frame.shape[0]),
        "retained_updates": int(retained.shape[0]),
        "valid_fraction": float(retained["valid_after_repair"].mean()),
        "state_representation": representation,
        "augmented_state_representation": augmented_representation,
        "augmented_occupied_states": int(np.unique(augmented_path).size),
        "occupied_states": occupied,
        "occupied_macrostates": occupied_macro,
        "transition_pairs_observed": int(np.count_nonzero(counts)),
        "configuration_shannon_entropy": config_entropy,
        "configuration_miller_madow_entropy": config_mm,
        "configuration_tsallis_q0_5": config_t05,
        "configuration_tsallis_q2": config_t2,
        "configuration_gini_simpson": gini_simpson(np.unique(state_path, return_counts=True)[1]),
        "macrostate_shannon_entropy": macro_entropy,
        "macrostate_miller_madow_entropy": macro_mm,
        "macrostate_tsallis_q0_5": macro_t05,
        "macrostate_tsallis_q2": macro_t2,
        "markov_epr_nats_per_update": epr,
        "markov_epr_nats_per_sweep": epr * int(panel_definition["n_agents"]),
        "transition_pair_irreversibility_nats_per_update": pair_irreversibility,
        "block_kl_nats_per_update": block_values[primary_length],
        "time_shuffle_floor": floor["mean"],
        "time_shuffle_floor_95": floor["ci_high"],
        "adjusted_block_kl_nats_per_update": block_values[primary_length] - floor["mean"],
        "augmented_block_kl_nats_per_update": augmented_block,
        "augmented_time_shuffle_floor": augmented_floor["mean"],
        "augmented_adjusted_block_kl_nats_per_update": augmented_block - augmented_floor["mean"],
        "belief_layer_markov_epr_nats_per_update": belief_layer["markov_epr"],
        "belief_layer_adjusted_block_kl_nats_per_update": belief_layer["adjusted_block_kl"],
        "action_layer_markov_epr_nats_per_update": action_layer["markov_epr"],
        "action_layer_adjusted_block_kl_nats_per_update": action_layer["adjusted_block_kl"],
        "mean_belief_magnetization": float(np.mean(belief_m)),
        "mean_abs_belief_magnetization": float(np.mean(np.abs(belief_m))),
        "mean_action_magnetization": float(np.mean(action_m)),
        "mean_abs_action_magnetization": float(np.mean(np.abs(action_m))),
        "mean_belief_action_overlap": float(retained["belief_action_overlap"].mean()),
        "mean_belief_disagreement": float(retained["belief_disagreement"].mean()),
        "mean_action_disagreement": float(retained["action_disagreement"].mean()),
        "belief_susceptibility": susceptibility(belief_m, int(panel_definition["n_agents"])),
        "action_susceptibility": susceptibility(action_m, int(panel_definition["n_agents"])),
        "mean_reference_energy_per_agent": float(retained["reference_energy_per_agent"].mean()),
        "belief_integrated_autocorrelation_time_updates": integrated_autocorrelation_time(belief_m, maximum_lag),
        "action_integrated_autocorrelation_time_updates": integrated_autocorrelation_time(action_m, maximum_lag),
        "neighbor_belief_mutual_information": float(np.mean(edge_mi)) if edge_mi else 0.0,
        "neighbor_belief_mutual_information_permutation_floor": float(np.mean(edge_mi_floor)) if edge_mi_floor else 0.0,
        "neighbor_belief_mutual_information_bias_corrected": (
            float(np.mean(edge_mi) - np.mean(edge_mi_floor)) if edge_mi else 0.0
        ),
        "directed_edge_transfer_entropy": float(np.mean(edge_te)) if edge_te else 0.0,
        "directed_edge_transfer_entropy_permutation_floor": float(np.mean(edge_te_floor)) if edge_te_floor else 0.0,
        "directed_edge_transfer_entropy_bias_corrected": (
            float(np.mean(edge_te) - np.mean(edge_te_floor)) if edge_te else 0.0
        ),
        "message_opportunities": int(retained["message_opportunities"].sum()),
        "messages_transmitted": int(retained["messages_transmitted"].sum()),
        "messages_delivered": int(retained["messages_delivered"].sum()),
        "messages_dropped": int(retained["messages_dropped"].sum()),
        "wire_bytes": int(retained["wire_bytes"].sum()),
        "prompt_tokens": int(retained["prompt_tokens"].sum()),
        "generated_tokens": int(retained["generated_tokens"].sum()),
        "latency_seconds": float(retained["latency_seconds"].sum()),
        "privacy_mutations": int(retained["unrelated_peer_private_mutations"].sum()),
        "mean_total_workload": float(retained["total_workload"].mean()),
        "current_l1": float(np.sum(np.abs(current)) / 2.0),
        "maximum_absolute_current": float(np.max(np.abs(current))),
        "early_late_state_js": _early_late_js(analysis_path),
        "early_late_belief_magnetization_difference": float(
            np.mean(belief_m[len(belief_m) // 2 :]) - np.mean(belief_m[: len(belief_m) // 2])
        ),
        "belief_correlation_distance_1": belief_correlations.get(1, float("nan")),
        "belief_correlation_distance_2": belief_correlations.get(2, float("nan")),
        "action_correlation_distance_1": action_correlations.get(1, float("nan")),
        **graph_diagnostics,
        **markov,
        **sensitivity,
    }
    for length, value in block_values.items():
        row["block_kl_length_%d" % length] = value
    return row, top_currents


def _logistic_fit(design: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)

    def objective(beta: np.ndarray) -> float:
        linear = np.clip(x.dot(beta), -35.0, 35.0)
        return float(np.sum(np.logaddexp(0.0, linear) - y * linear) + 1e-4 * np.dot(beta[1:], beta[1:]))

    def gradient(beta: np.ndarray) -> np.ndarray:
        linear = np.clip(x.dot(beta), -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        penalty = np.r_[0.0, 2e-4 * beta[1:]]
        return x.T.dot(probability - y) + penalty

    result = minimize(objective, np.zeros(x.shape[1]), jac=gradient, method="BFGS")
    beta = np.asarray(result.x, dtype=float)
    return beta, objective(beta)


def _micro_models(frame: pd.DataFrame, bootstrap_replicates: int, seed: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    valid = frame[frame["valid_after_repair"] == 1].copy()
    target = (valid["belief_after"].to_numpy(int) == 1).astype(float)
    private = valid["private_field"].to_numpy(float)
    neighbor = valid["neighbor_field"].to_numpy(float) * valid["coupling_strength"].to_numpy(float)
    current_belief = valid["current_belief"].to_numpy(float)
    current_action = valid["current_action"].to_numpy(float)
    temperature = valid["sampling_temperature"].to_numpy(float)
    memory = (valid["regime"].astype(str) == "persistent_memory").to_numpy(float)
    amber_first = valid["amber_first"].to_numpy(float)
    latent_amber = (valid["latent_plus_label"].astype(str) == "amber").to_numpy(float)
    paraphrase = valid["paraphrase"].to_numpy(float)
    designs = {
        "kinetic_logistic": np.column_stack(
            [
                np.ones(len(valid)), private, neighbor, current_belief, current_action,
                temperature, memory, amber_first, latent_amber, paraphrase,
            ]
        ),
        "nonlinear_additive": np.column_stack(
            [
                np.ones(len(valid)),
                private,
                private ** 2,
                neighbor,
                neighbor ** 2,
                current_belief,
                current_action,
                temperature,
                temperature ** 2,
                memory,
                amber_first,
                latent_amber,
                paraphrase,
            ]
        ),
        "persistence_interaction": np.column_stack(
            [
                np.ones(len(valid)),
                private,
                neighbor,
                current_belief,
                current_action,
                private * current_belief,
                neighbor * current_belief,
                current_belief * current_action,
                temperature,
                memory,
                amber_first,
                latent_amber,
                paraphrase,
            ]
        ),
    }
    held = np.asarray([sum(ord(char) for char in value) % 4 == 0 for value in valid["information_state_id"]], dtype=bool)
    rows: List[Dict[str, object]] = []
    for name, design in designs.items():
        beta, training_loss = _logistic_fit(design[~held], target[~held])
        probability = 1.0 / (1.0 + np.exp(-np.clip(design[held].dot(beta), -35.0, 35.0)))
        held_loss = float(-np.mean(target[held] * np.log(probability) + (1.0 - target[held]) * np.log(1.0 - probability)))
        rows.append(
            {
                "model": name,
                "response_layer": "belief",
                "training_rows": int(np.sum(~held)),
                "heldout_rows": int(np.sum(held)),
                "training_penalized_log_loss_sum": training_loss,
                "heldout_log_loss": held_loss,
                "field_coefficient": float(beta[1]),
                "neighbor_coefficient": float(beta[3] if name == "nonlinear_additive" else beta[2]),
                "belief_persistence_coefficient": float(beta[5] if name == "nonlinear_additive" else beta[3]),
                "action_persistence_coefficient": float(beta[6] if name == "nonlinear_additive" else beta[4]),
                "option_order_coefficient": float(beta[-3]),
                "latent_mapping_coefficient": float(beta[-2]),
                "prompt_paraphrase_coefficient": float(beta[-1]),
            }
        )
    action_target = (valid["action_after"].to_numpy(int) == 1).astype(float)
    next_belief = valid["belief_after"].to_numpy(float)
    action_designs = {
        "kinetic_logistic": np.column_stack(
            [
                np.ones(len(valid)), private, neighbor, current_action, next_belief,
                temperature, memory, amber_first, latent_amber, paraphrase,
            ]
        ),
        "nonlinear_additive": np.column_stack(
            [
                np.ones(len(valid)), private, private ** 2, neighbor, neighbor ** 2,
                current_action, next_belief, temperature, temperature ** 2,
                memory, amber_first, latent_amber, paraphrase,
            ]
        ),
        "persistence_interaction": np.column_stack(
            [
                np.ones(len(valid)), private, neighbor, current_action, next_belief,
                private * next_belief, neighbor * current_action,
                next_belief * current_action, temperature, memory,
                amber_first, latent_amber, paraphrase,
            ]
        ),
    }
    for name, design in action_designs.items():
        beta, training_loss = _logistic_fit(design[~held], action_target[~held])
        probability = 1.0 / (1.0 + np.exp(-np.clip(design[held].dot(beta), -35.0, 35.0)))
        held_loss = float(
            -np.mean(
                action_target[held] * np.log(probability)
                + (1.0 - action_target[held]) * np.log(1.0 - probability)
            )
        )
        rows.append(
            {
                "model": name,
                "response_layer": "action",
                "training_rows": int(np.sum(~held)),
                "heldout_rows": int(np.sum(held)),
                "training_penalized_log_loss_sum": training_loss,
                "heldout_log_loss": held_loss,
                "field_coefficient": float(beta[1]),
                "neighbor_coefficient": float(beta[3] if name == "nonlinear_additive" else beta[2]),
                "belief_persistence_coefficient": float(
                    beta[6] if name == "nonlinear_additive" else beta[4]
                ),
                "action_persistence_coefficient": float(
                    beta[5] if name == "nonlinear_additive" else beta[3]
                ),
                "option_order_coefficient": float(beta[-3]),
                "latent_mapping_coefficient": float(beta[-2]),
                "prompt_paraphrase_coefficient": float(beta[-1]),
            }
        )
    # H1 is a paired information-state contrast, not token-level inference.
    effects: Dict[str, List[float]] = {}
    group_columns = [
        "private_field",
        "current_belief",
        "current_action",
        "coupling_strength",
        "sampling_temperature",
        "regime",
        "replicate",
    ]
    for keys, group in valid.groupby(group_columns, sort=True):
        means = group.groupby("neighbor_field")["belief_after"].mean()
        if -1 not in means.index or 1 not in means.index:
            continue
        cluster = "|".join(str(item) for item in keys[:-1])
        effects.setdefault(cluster, []).append(float((means.loc[1] - means.loc[-1]) / 2.0))
    h1 = paired_cluster_bootstrap(effects, int(bootstrap_replicates), int(seed))
    h1["estimand"] = "latent_plus_choice_change_per_unit_neighbor_field"
    h1["valid_rows"] = int(len(valid))
    h1["latent_plus_occupancy"] = float(np.mean(target))
    h1["belief_minus_to_plus"] = int(np.sum((valid["current_belief"] == -1) & (valid["belief_after"] == 1)))
    h1["belief_plus_to_minus"] = int(np.sum((valid["current_belief"] == 1) & (valid["belief_after"] == -1)))
    return rows, h1


def _agent_statistics(
    frame: pd.DataFrame, panel_definition: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    burn = int(panel_definition["burn_in_sweeps"] * panel_definition["n_agents"])
    retained = frame.iloc[burn:].copy()
    retained["belief_switched"] = retained["belief_before"] != retained["belief_after"]
    retained["action_switched"] = retained["action_before"] != retained["action_after"]
    rows: List[Dict[str, object]] = []
    for agent, group in retained.groupby("scheduled_agent", sort=True):
        rows.append(
            {
                "family": panel_definition["family"],
                "cluster_id": panel_definition["cluster_id"],
                "panel_id": panel_definition["panel_id"],
                "scheduled_agent": int(agent),
                "attempted_updates": int(len(group)),
                "valid_fraction": float(group["valid_after_repair"].mean()),
                "belief_switch_rate": float(group["belief_switched"].mean()),
                "action_switch_rate": float(group["action_switched"].mean()),
                "mean_confidence": float(group["confidence_after"].mean()),
                "mean_neighbor_field": float(group["neighbor_field"].mean()),
                "message_delivery_rate": float(group["messages_delivered"].mean()),
                "mean_workload_change": float(group["workload_change"].mean()),
            }
        )
    belief_rates = np.asarray([row["belief_switch_rate"] for row in rows], dtype=float)
    action_rates = np.asarray([row["action_switch_rate"] for row in rows], dtype=float)
    return rows, {
        "agent_belief_switch_rate_sd": float(np.std(belief_rates, ddof=1)) if len(rows) > 1 else 0.0,
        "agent_action_switch_rate_sd": float(np.std(action_rates, ddof=1)) if len(rows) > 1 else 0.0,
    }


def _primary_effects(panel: pd.DataFrame, protocol: Mapping[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    primary = panel[panel["family"].isin(["small_network", "collective_network"])].copy()
    alpha_max = float(protocol["analysis"]["primary_strong_alpha"])  # type: ignore[index]
    bootstrap_n = int(protocol["analysis"]["cluster_bootstrap_replicates"])  # type: ignore[index]
    seed = int(protocol["analysis"]["analysis_seed"])  # type: ignore[index]
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, object] = {}
    metrics = [
        "adjusted_block_kl_nats_per_update",
        "markov_epr_nats_per_update",
        "mean_abs_belief_magnetization",
        "belief_susceptibility",
        "belief_integrated_autocorrelation_time_updates",
    ]
    for family in ("small_network", "collective_network"):
        subset = primary[primary["family"] == family]
        for metric in metrics:
            effects: Dict[str, List[float]] = {}
            for cluster, group in subset.groupby("cluster_id", sort=True):
                baseline = group[np.isclose(group["alpha"], 0.0)]
                strong = group[np.isclose(group["alpha"], alpha_max)]
                if baseline.empty or strong.empty:
                    continue
                difference = float(strong[metric].mean() - baseline[metric].mean())
                effects[str(cluster)] = [difference]
                rows.append(
                    {
                        "family": family,
                        "cluster_id": cluster,
                        "metric": metric,
                        "alpha": alpha_max,
                        "paired_difference": difference,
                    }
                )
            summary = paired_cluster_bootstrap(effects, bootstrap_n, seed + len(summaries))
            summaries[family + ":" + metric] = summary
    return rows, summaries


def _quadratic_models(panel: pd.DataFrame, protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    primary = panel[panel["family"].isin(["small_network", "collective_network"])].copy()
    weak = [float(value) for value in protocol["analysis"]["weak_alpha_range"]]  # type: ignore[index]
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(int(protocol["analysis"]["analysis_seed"]) + 4301)  # type: ignore[index]
    bootstrap_n = int(protocol["analysis"]["cluster_bootstrap_replicates"])  # type: ignore[index]
    for family in ("small_network", "collective_network"):
        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for cluster, group in primary[primary["family"] == family].groupby("cluster_id", sort=True):
            averaged = group[group["alpha"].isin(weak)].groupby("alpha", as_index=False)[
                "adjusted_block_kl_nats_per_update"
            ].mean()
            if len(averaged) != len(weak):
                continue
            baseline = float(averaged.loc[np.isclose(averaged["alpha"], 0.0), "adjusted_block_kl_nats_per_update"].iloc[0])
            curves[str(cluster)] = (
                averaged["alpha"].to_numpy(float),
                averaged["adjusted_block_kl_nats_per_update"].to_numpy(float) - baseline,
            )
        forms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "linear": lambda alpha: alpha[:, None],
            "quadratic": lambda alpha: (alpha ** 2)[:, None],
            "linear_plus_quadratic": lambda alpha: np.column_stack([alpha, alpha ** 2]),
        }
        for name, function in forms.items():
            all_x = np.concatenate([value[0] for value in curves.values()])
            all_y = np.concatenate([value[1] for value in curves.values()])
            design = function(all_x)
            beta = np.linalg.lstsq(design, all_y, rcond=None)[0]
            held_errors: List[float] = []
            for held, (x_test, y_test) in curves.items():
                training = [value for key, value in curves.items() if key != held]
                x_train = np.concatenate([value[0] for value in training])
                y_train = np.concatenate([value[1] for value in training])
                fit = np.linalg.lstsq(function(x_train), y_train, rcond=None)[0]
                held_errors.extend((y_test - function(x_test).dot(fit)).tolist())
            residual = all_y - design.dot(beta)
            rss = float(np.dot(residual, residual))
            count = int(len(all_y))
            parameters = int(len(beta))
            aic = float(count * np.log(max(rss / max(count, 1), 1e-15)) + 2 * parameters)
            aicc = float(aic + (2 * parameters * (parameters + 1)) / max(count - parameters - 1, 1))
            keys = sorted(curves)
            bootstrap_beta = np.zeros((bootstrap_n, parameters), dtype=float)
            for index in range(bootstrap_n):
                sampled = [curves[keys[item]] for item in rng.integers(0, len(keys), len(keys))]
                x_sample = np.concatenate([value[0] for value in sampled])
                y_sample = np.concatenate([value[1] for value in sampled])
                bootstrap_beta[index] = np.linalg.lstsq(function(x_sample), y_sample, rcond=None)[0]
            linear_index = 0 if name != "quadratic" else None
            quadratic_index = parameters - 1 if name != "linear" else None
            rows.append(
                {
                    "family": family,
                    "model": name,
                    "independent_clusters": len(curves),
                    "linear_coefficient": float(beta[0]) if name != "quadratic" else 0.0,
                    "quadratic_coefficient": float(beta[-1]) if name != "linear" else 0.0,
                    "linear_ci_low": 0.0 if linear_index is None else float(np.quantile(bootstrap_beta[:, linear_index], 0.025)),
                    "linear_ci_high": 0.0 if linear_index is None else float(np.quantile(bootstrap_beta[:, linear_index], 0.975)),
                    "quadratic_ci_low": 0.0 if quadratic_index is None else float(np.quantile(bootstrap_beta[:, quadratic_index], 0.025)),
                    "quadratic_ci_high": 0.0 if quadratic_index is None else float(np.quantile(bootstrap_beta[:, quadratic_index], 0.975)),
                    "leave_cluster_out_rmse": float(np.sqrt(np.mean(np.square(held_errors)))),
                    "full_fit_rmse": float(np.sqrt(np.mean(np.square(residual)))),
                    "aicc_descriptive": aicc,
                }
            )
    return rows


def _sign_flip_pvalue(values: Sequence[float], replicates: int, seed: int) -> float:
    observed = abs(float(np.mean(values)))
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        null[index] = abs(float(np.mean(array * rng.choice([-1.0, 1.0], size=array.size))))
    return float((1.0 + np.sum(null >= observed)) / (1.0 + len(null)))


def _dose_response_table(
    panel: pd.DataFrame, protocol: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    primary = panel[panel["family"].isin(["small_network", "collective_network"])].copy()
    records: List[Dict[str, object]] = []
    summary: Dict[str, object] = {}
    bootstrap_n = int(protocol["analysis"]["cluster_bootstrap_replicates"])  # type: ignore[index]
    seed = int(protocol["analysis"]["analysis_seed"]) + 8100  # type: ignore[index]
    for family in ("small_network", "collective_network"):
        slopes: Dict[str, List[float]] = {}
        monotone: List[float] = []
        for cluster, group in primary[primary["family"] == family].groupby("cluster_id", sort=True):
            averaged = group.groupby("alpha", as_index=False)["adjusted_block_kl_nats_per_update"].mean().sort_values("alpha")
            baseline = float(averaged.iloc[0]["adjusted_block_kl_nats_per_update"])
            averaged["excess_irreversibility"] = averaged["adjusted_block_kl_nats_per_update"] - baseline
            slope = float(
                np.linalg.lstsq(
                    averaged["alpha"].to_numpy(float)[:, None],
                    averaged["excess_irreversibility"].to_numpy(float),
                    rcond=None,
                )[0][0]
            )
            slopes[str(cluster)] = [slope]
            increments = np.diff(averaged["excess_irreversibility"].to_numpy(float))
            monotone.append(float(np.all(increments >= 0.0)))
            for row in averaged.itertuples():
                records.append(
                    {
                        "family": family,
                        "cluster_id": cluster,
                        "alpha": float(row.alpha),
                        "excess_irreversibility": float(row.excess_irreversibility),
                        "cluster_slope": slope,
                    }
                )
        result = paired_cluster_bootstrap(slopes, bootstrap_n, seed + len(summary))
        result["monotone_cluster_fraction"] = float(np.mean(monotone))
        result["sign_flip_pvalue"] = _sign_flip_pvalue(
            [value[0] for value in slopes.values()], bootstrap_n, seed + 101 + len(summary)
        )
        summary[family] = result
    return records, summary


def _orientation_replication(
    panel: pd.DataFrame, protocol: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    strong = float(protocol["analysis"]["primary_strong_alpha"])  # type: ignore[index]
    bootstrap_n = int(protocol["analysis"]["cluster_bootstrap_replicates"])  # type: ignore[index]
    seed = int(protocol["analysis"]["analysis_seed"]) + 9100  # type: ignore[index]
    subset = panel[panel["family"].isin(["small_network", "collective_network"])]
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, object] = {}
    for family in ("small_network", "collective_network"):
        for n_agents in sorted(subset[subset["family"] == family]["n_agents"].unique()):
            selected = subset[(subset["family"] == family) & (subset["n_agents"] == n_agents)]
            for orientation in ("forward", "transpose"):
                effects: Dict[str, List[float]] = {}
                for cluster, group in selected.groupby("cluster_id", sort=True):
                    baseline = group[np.isclose(group["alpha"], 0.0)]
                    directed = group[np.isclose(group["alpha"], strong) & (group["orientation"] == orientation)]
                    if baseline.empty or directed.empty:
                        continue
                    value = float(
                        directed["adjusted_block_kl_nats_per_update"].mean()
                        - baseline["adjusted_block_kl_nats_per_update"].mean()
                    )
                    effects[str(cluster)] = [value]
                    rows.append(
                        {
                            "family": family,
                            "n_agents": int(n_agents),
                            "orientation": orientation,
                            "cluster_id": cluster,
                            "paired_excess_irreversibility": value,
                        }
                    )
                key = "%s:N%d:%s" % (family, int(n_agents), orientation)
                summaries[key] = paired_cluster_bootstrap(effects, bootstrap_n, seed + len(summaries))
    return rows, summaries


def _factor_effects(
    panel: pd.DataFrame, protocol: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    collective = panel[panel["family"] == "collective_network"].copy()
    bootstrap_n = int(protocol["analysis"]["cluster_bootstrap_replicates"])  # type: ignore[index]
    seed = int(protocol["analysis"]["analysis_seed"]) + 10100  # type: ignore[index]
    metrics = [
        "mean_abs_belief_magnetization",
        "belief_susceptibility",
        "belief_integrated_autocorrelation_time_updates",
        "neighbor_belief_mutual_information_bias_corrected",
    ]
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, object] = {}
    tests: List[Tuple[str, float]] = []
    for factor in ("coupling_strength", "sampling_temperature"):
        levels = sorted(collective[factor].unique())
        other = "sampling_temperature" if factor == "coupling_strength" else "coupling_strength"
        for metric in metrics:
            effects: Dict[str, List[float]] = {}
            grouping = ["n_agents", "topology", other, "replicate", "alpha"]
            for keys, group in collective.groupby(grouping, sort=True):
                low = group[np.isclose(group[factor], levels[0])]
                high = group[np.isclose(group[factor], levels[-1])]
                if low.empty or high.empty:
                    continue
                cluster = "|".join(str(value) for value in keys)
                value = float(high[metric].mean() - low[metric].mean())
                effects[cluster] = [value]
                rows.append({"factor": factor, "metric": metric, "matched_cell": cluster, "high_minus_low": value})
            result = paired_cluster_bootstrap(effects, bootstrap_n, seed + len(summaries))
            raw_p = _sign_flip_pvalue([value[0] for value in effects.values()], bootstrap_n, seed + 300 + len(summaries))
            result["raw_sign_flip_pvalue"] = raw_p
            key = factor + ":" + metric
            summaries[key] = result
            tests.append((key, raw_p))
    adjusted = _holm([value for _, value in tests])
    for (key, _), value in zip(tests, adjusted):
        summaries[key]["holm_adjusted_pvalue"] = value  # type: ignore[index]
    return rows, summaries


def _control_effects(
    panel: pd.DataFrame, protocol: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    controls = panel[panel["family"] == "controls"].copy()
    bootstrap_n = int(protocol["analysis"]["cluster_bootstrap_replicates"])  # type: ignore[index]
    seed = int(protocol["analysis"]["analysis_seed"]) + 11100  # type: ignore[index]
    metrics = [
        "adjusted_block_kl_nats_per_update",
        "mean_abs_belief_magnetization",
        "neighbor_belief_mutual_information_bias_corrected",
        "directed_edge_transfer_entropy_bias_corrected",
    ]
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, object] = {}
    tests: List[Tuple[str, float]] = []
    arms = sorted(value for value in controls["control"].unique() if value != "unaltered")
    for arm in arms:
        for metric in metrics:
            effects: Dict[str, List[float]] = {}
            for cluster, group in controls.groupby("cluster_id", sort=True):
                reference = group[group["control"] == "unaltered"]
                comparison = group[group["control"] == arm]
                if reference.empty or comparison.empty:
                    continue
                value = float(comparison[metric].mean() - reference[metric].mean())
                effects[str(cluster)] = [value]
                rows.append({"control": arm, "metric": metric, "cluster_id": cluster, "control_minus_unaltered": value})
            result = paired_cluster_bootstrap(effects, bootstrap_n, seed + len(summaries))
            pvalue = _sign_flip_pvalue([item[0] for item in effects.values()], bootstrap_n, seed + 300 + len(summaries))
            result["raw_sign_flip_pvalue"] = pvalue
            key = arm + ":" + metric
            summaries[key] = result
            tests.append((key, pvalue))
    adjusted = _holm([value for _, value in tests])
    for (key, _), value in zip(tests, adjusted):
        summaries[key]["holm_adjusted_pvalue"] = value  # type: ignore[index]
    return rows, summaries


def _memory_effects(
    panel: pd.DataFrame, protocol: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    memory = panel[panel["family"] == "persistent_memory"].copy()
    bootstrap_n = int(protocol["analysis"]["cluster_bootstrap_replicates"])  # type: ignore[index]
    seed = int(protocol["analysis"]["analysis_seed"]) + 12100  # type: ignore[index]
    metrics = [
        "adjusted_block_kl_nats_per_update",
        "history_1_conditional_mutual_information",
        "belief_integrated_autocorrelation_time_updates",
        "mean_abs_belief_magnetization",
    ]
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, object] = {}
    for metric in metrics:
        effects: Dict[str, List[float]] = {}
        grouping = ["cluster_id", "alpha", "orientation"]
        for keys, group in memory.groupby(grouping, sort=True):
            markov = group[group["regime"] == "markovized"]
            persistent = group[group["regime"] == "persistent_memory"]
            if markov.empty or persistent.empty:
                continue
            value = float(persistent[metric].mean() - markov[metric].mean())
            cluster = "%s|%.2f|%s" % (keys[0], float(keys[1]), keys[2])
            effects[cluster] = [value]
            rows.append({"metric": metric, "matched_arm": cluster, "persistent_minus_markovized": value})
        summaries[metric] = paired_cluster_bootstrap(effects, bootstrap_n, seed + len(summaries))
    return rows, summaries


def _holm(values: Sequence[float]) -> List[float]:
    raw = np.asarray(values, dtype=float)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _hysteresis_table(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted((root / "hysteresis").glob("*.csv")):
        frame = pd.read_csv(path)
        segment = frame.groupby("field_segment", as_index=False).agg(
            external_field=("external_field", "first"),
            belief_magnetization=("belief_magnetization", "mean"),
            action_magnetization=("action_magnetization", "mean"),
        )
        first = frame.iloc[0]
        rows.append(
            {
                "panel_id": first["panel_id"],
                "cluster_id": first["cluster_id"],
                "alpha": float(first["alpha"]),
                "orientation": first["orientation"],
                "belief_hysteresis_area": hysteresis_area(
                    segment["external_field"], segment["belief_magnetization"]
                ),
                "action_hysteresis_area": hysteresis_area(
                    segment["external_field"], segment["action_magnetization"]
                ),
            }
        )
    return rows


def _fit_surrogate_parameters(frame: pd.DataFrame) -> Dict[str, object]:
    """Fit the prospectively declared kinetic-Ising surrogate.

    This comparator is fitted only to valid isolated-agent response rows.  It
    is not used to generate or filter any LLM outcome.
    """

    valid = frame[frame["valid_after_repair"] == 1].copy()
    belief_design = np.column_stack(
        [
            np.ones(len(valid)),
            valid["private_field"].to_numpy(float),
            valid["neighbor_field"].to_numpy(float) * valid["coupling_strength"].to_numpy(float),
            valid["current_belief"].to_numpy(float),
            valid["current_action"].to_numpy(float),
        ]
    )
    belief_target = (valid["belief_after"].to_numpy(int) == 1).astype(float)
    action_design = np.column_stack(
        [
            np.ones(len(valid)),
            valid["belief_after"].to_numpy(float),
            valid["current_action"].to_numpy(float),
        ]
    )
    action_target = (valid["action_after"].to_numpy(int) == 1).astype(float)
    belief_beta, belief_loss = _logistic_fit(belief_design, belief_target)
    action_beta, action_loss = _logistic_fit(action_design, action_target)
    return {
        "belief_feature_order": [
            "intercept",
            "private_field",
            "weighted_neighbor_field",
            "previous_belief",
            "previous_action",
        ],
        "belief_coefficients": belief_beta.tolist(),
        "belief_penalized_log_loss_sum": float(belief_loss),
        "action_feature_order": ["intercept", "updated_belief", "previous_action"],
        "action_coefficients": action_beta.tolist(),
        "action_penalized_log_loss_sum": float(action_loss),
        "valid_fit_rows": int(len(valid)),
    }


def _surrogate_macrostate(beliefs: np.ndarray, actions: np.ndarray) -> Tuple[int, int, int]:
    return (
        int(np.rint(4.0 * float(np.mean(beliefs)))),
        int(np.rint(4.0 * float(np.mean(actions)))),
        int(np.rint(4.0 * float(np.mean(beliefs * actions)))),
    )


def _simulate_fitted_surrogate(
    micro: pd.DataFrame,
    protocol: Mapping[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    settings = protocol["surrogates"]["fitted_kinetic_ising"]  # type: ignore[index]
    parameters = _fit_surrogate_parameters(micro)
    belief_beta = np.asarray(parameters["belief_coefficients"], dtype=float)
    action_beta = np.asarray(parameters["action_coefficients"], dtype=float)
    levels = [float(value) for value in protocol["network"]["nonreciprocity_levels"]]  # type: ignore[index]
    sweeps = int(settings["trajectory_sweeps"])
    burn_sweeps = int(settings["burn_in_sweeps"])
    seeds_per_cell = int(settings["independent_seeds_per_cell"])
    rows: List[Dict[str, object]] = []

    def draw(linear: float, rng: np.random.Generator) -> int:
        probability = 1.0 / (1.0 + np.exp(-float(np.clip(linear, -35.0, 35.0))))
        return 1 if float(rng.random()) < probability else -1

    for n_agents in [int(value) for value in settings["agent_counts"]]:  # type: ignore[index]
        for topology in [str(value) for value in settings["topologies"]]:  # type: ignore[index]
            if topology == "modular" and n_agents < 8:
                continue
            for alpha in levels:
                orientations = (False,) if np.isclose(alpha, 0.0) else (False, True)
                for reverse in orientations:
                    for replicate in range(seeds_per_cell):
                        # The stochastic schedule and uniforms are common
                        # across alpha and orientation within a matched cell.
                        seed = 12850000 + 100000 * n_agents + 1000 * replicate + (500000 if topology == "modular" else 0)
                        graph = build_delivery_graph(n_agents, topology, seed + 17, seed + 37, alpha, reverse)
                        rng = np.random.default_rng(seed)
                        fields = np.asarray([1 if index % 2 == 0 else -1 for index in range(n_agents)], dtype=int)
                        rng.shuffle(fields)
                        beliefs = np.asarray([1 if index % 2 == 0 else -1 for index in range(n_agents)], dtype=int)
                        actions = -beliefs.copy()
                        rng.shuffle(beliefs)
                        rng.shuffle(actions)
                        inboxes: List[List[int]] = [[] for _ in range(n_agents)]
                        scheduled, uniforms = matched_opportunity_schedule(n_agents, n_agents * sweeps, seed + 101)
                        states: List[Tuple[int, int, int]] = [_surrogate_macrostate(beliefs, actions)]
                        for update, agent in enumerate(scheduled):
                            index = int(agent)
                            neighbor = float(np.mean(inboxes[index])) if inboxes[index] else 0.0
                            inboxes[index].clear()
                            belief_linear = float(
                                belief_beta.dot(
                                    [1.0, fields[index], 0.70 * neighbor, beliefs[index], actions[index]]
                                )
                            )
                            beliefs[index] = draw(belief_linear, rng)
                            action_linear = float(action_beta.dot([1.0, beliefs[index], actions[index]]))
                            actions[index] = draw(action_linear, rng)
                            recipient = select_recipient(graph.weights, index, float(uniforms[update]))
                            inboxes[recipient].append(int(beliefs[index]))
                            states.append(_surrogate_macrostate(beliefs, actions))
                        retained = states[burn_sweeps * n_agents :]
                        vocabulary = {value: idx for idx, value in enumerate(sorted(set(retained)))}
                        encoded = np.asarray([vocabulary[value] for value in retained], dtype=int)
                        raw = block_time_reversal_kl(encoded, 3, 0.5)
                        floor = time_shuffle_floor(encoded, 3, 0.5, 100, seed + 701)
                        rows.append(
                            {
                                "n_agents": n_agents,
                                "topology": topology,
                                "alpha": alpha,
                                "orientation": "transpose" if reverse else ("reciprocal" if np.isclose(alpha, 0.0) else "forward"),
                                "replicate": replicate,
                                "trajectory_sweeps": sweeps,
                                "burn_in_sweeps": burn_sweeps,
                                "occupied_macrostates": len(vocabulary),
                                "raw_block_kl": raw,
                                "shuffle_floor": floor["mean"],
                                "irreversibility": raw - floor["mean"],
                            }
                        )
    return rows, parameters


def analyze_formal(repository: Path) -> Dict[str, object]:
    analysis_started = time.perf_counter()
    analysis_cpu_started = time.process_time()
    repository = Path(repository).resolve()
    protocol_path = repository / "configs/statmech_llm/discovery/protocol.yaml"
    if not protocol_path.exists():
        raise RuntimeError("cannot analyze before V12 protocol freeze")
    protocol = load_yaml(protocol_path)
    root = artifact_root() / "formal"
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError("formal V12 execution is incomplete")
    design = {str(row["panel_id"]): row for row in formal_panel_design(protocol)}
    panel_rows: List[Dict[str, object]] = []
    current_rows: List[Dict[str, object]] = []
    agent_rows: List[Dict[str, object]] = []
    for path in sorted((root / "panels").glob("*.csv")):
        frame = pd.read_csv(path)
        panel_id = str(frame.iloc[0]["panel_id"])
        row, currents = _panel_statistics(frame, protocol, design[panel_id])
        agents, heterogeneity = _agent_statistics(frame, design[panel_id])
        row.update(heterogeneity)
        panel_rows.append(row)
        current_rows.extend(currents)
        agent_rows.extend(agents)
    panel = pd.DataFrame(panel_rows)
    micro = pd.read_csv(root / "microscopic_response.csv")
    micro_models, h1 = _micro_models(
        micro,
        int(protocol["analysis"]["cluster_bootstrap_replicates"]),  # type: ignore[index]
        int(protocol["analysis"]["analysis_seed"]),  # type: ignore[index]
    )
    effect_rows, effects = _primary_effects(panel, protocol)
    dose_rows, dose_summary = _dose_response_table(panel, protocol)
    orientation_rows, orientation_summary = _orientation_replication(panel, protocol)
    factor_rows, factor_summary = _factor_effects(panel, protocol)
    control_effect_rows, control_effect_summary = _control_effects(panel, protocol)
    memory_effect_rows, memory_effect_summary = _memory_effects(panel, protocol)
    quadratic = _quadratic_models(panel, protocol)
    surrogate, surrogate_parameters = _simulate_fitted_surrogate(micro, protocol)
    hysteresis = _hysteresis_table(root)
    controls = panel[panel["family"] == "controls"].to_dict(orient="records")
    memory = panel[panel["family"] == "persistent_memory"].to_dict(orient="records")
    relaxation = panel[panel["family"] == "relaxation"].to_dict(orient="records")
    output = artifact_root() / "analysis"
    atomic_csv(panel_rows, output / "panel_statistics.csv")
    atomic_csv(micro_models, output / "microscopic_models.csv")
    atomic_csv(effect_rows, output / "cluster_effects.csv")
    atomic_csv(dose_rows, output / "nonreciprocity_dose_response.csv")
    atomic_csv(orientation_rows, output / "orientation_replication.csv")
    atomic_csv(factor_rows, output / "collective_factor_effects.csv")
    atomic_csv(control_effect_rows, output / "control_effects.csv")
    atomic_csv(memory_effect_rows, output / "memory_effects.csv")
    atomic_csv(quadratic, output / "quadratic_models.csv")
    atomic_csv(current_rows, output / "probability_currents.csv")
    atomic_csv(agent_rows, output / "agent_statistics.csv")
    atomic_csv(hysteresis, output / "hysteresis.csv")
    atomic_csv(controls, output / "controls.csv")
    atomic_csv(memory, output / "memory.csv")
    atomic_csv(relaxation, output / "relaxation.csv")
    atomic_csv(surrogate, output / "fitted_surrogate.csv")
    atomic_json(surrogate_parameters, output / "fitted_surrogate_parameters.json")
    primary = {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "H1_individual_neighbor_response": h1,
        "paired_effects": effects,
        "nonreciprocity_dose_response": dose_summary,
        "orientation_replication": orientation_summary,
        "collective_factor_effects": factor_summary,
        "control_effects": control_effect_summary,
        "memory_effects": memory_effect_summary,
        "quadratic_models": quadratic,
        "fitted_surrogate_panels": int(len(surrogate)),
        "panel_count": int(len(panel)),
        "small_panel_count": int(np.sum(panel["family"] == "small_network")),
        "collective_panel_count": int(np.sum(panel["family"] == "collective_network")),
        "privacy_mutations": int(panel["privacy_mutations"].sum()),
        "total_attempted_updates": int(panel["attempted_updates"].sum() + len(micro)),
        "total_messages": int(panel["messages_transmitted"].sum()),
        "total_wire_bytes": int(panel["wire_bytes"].sum()),
        "formal_completion": completion,
        "analysis_wall_seconds": float(time.perf_counter() - analysis_started),
        "analysis_cpu_seconds": float(time.process_time() - analysis_cpu_started),
    }
    atomic_json(primary, output / "primary_results.json")
    return primary
