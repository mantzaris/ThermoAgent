"""Finite-state and pathwise estimators with explicit V12 units and limits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


Array = np.ndarray


def shannon_entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError("probabilities must be a nonnegative vector with positive mass")
    values = values / values.sum()
    active = values > 0.0
    return float(-np.sum(values[active] * np.log(values[active])))


def tsallis_entropy(probabilities: Sequence[float], q: float) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or np.any(values < 0.0) or values.sum() <= 0.0 or float(q) <= 0.0:
        raise ValueError("invalid Tsallis inputs")
    values = values / values.sum()
    if np.isclose(float(q), 1.0, atol=1e-8):
        return shannon_entropy(values)
    return float((1.0 - np.sum(values ** float(q))) / (float(q) - 1.0))


def gini_simpson(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values / values.sum()
    return float(1.0 - np.sum(values * values))


def miller_madow_entropy(counts: Sequence[float]) -> float:
    """First-order finite-sample correction to plug-in Shannon entropy."""

    values = np.asarray(counts, dtype=float)
    values = values[values > 0.0]
    if values.size == 0 or values.sum() <= 0.0:
        raise ValueError("entropy counts must contain positive mass")
    return float(shannon_entropy(values) + (values.size - 1.0) / (2.0 * values.sum()))


def jensen_shannon_divergence(first: Sequence[float], second: Sequence[float]) -> float:
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    if p.shape != q.shape or p.ndim != 1 or np.any(p < 0.0) or np.any(q < 0.0):
        raise ValueError("Jensen-Shannon inputs must be aligned nonnegative vectors")
    if p.sum() <= 0.0 or q.sum() <= 0.0:
        raise ValueError("Jensen-Shannon inputs need positive mass")
    p = p / p.sum()
    q = q / q.sum()
    mixture = 0.5 * (p + q)
    active_p = p > 0.0
    active_q = q > 0.0
    return float(
        0.5 * np.sum(p[active_p] * np.log(p[active_p] / mixture[active_p]))
        + 0.5 * np.sum(q[active_q] * np.log(q[active_q] / mixture[active_q]))
    )


def transition_counts(states: Sequence[int], state_count: Optional[int] = None) -> Array:
    values = np.asarray(states, dtype=int)
    if values.ndim != 1 or values.size < 2 or np.any(values < 0):
        raise ValueError("at least two nonnegative states are required")
    count = int(np.max(values) + 1 if state_count is None else state_count)
    if np.any(values >= count):
        raise ValueError("state exceeds declared state count")
    output = np.zeros((count, count), dtype=float)
    np.add.at(output, (values[:-1], values[1:]), 1.0)
    return output


def occupied_transition_counts(states: Sequence[int]) -> Tuple[Array, Dict[int, int]]:
    values = np.asarray(states, dtype=int)
    unique = sorted(int(value) for value in np.unique(values))
    remap = {value: index for index, value in enumerate(unique)}
    encoded = np.asarray([remap[int(value)] for value in values], dtype=int)
    return transition_counts(encoded, len(unique)), remap


def row_stochastic(counts: Array, pseudocount: float = 0.0, support: Optional[Array] = None) -> Array:
    raw = np.asarray(counts, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or np.any(raw < 0.0):
        raise ValueError("counts must be square and nonnegative")
    if float(pseudocount) < 0.0:
        raise ValueError("pseudocount must be nonnegative")
    if support is None:
        active = (raw + raw.T) > 0.0
        np.fill_diagonal(active, True)
    else:
        active = np.asarray(support, dtype=bool)
        if active.shape != raw.shape or not np.array_equal(active, active.T):
            raise ValueError("support must be symmetric")
    smoothed = raw + float(pseudocount) * active
    totals = smoothed.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("every retained source state needs outgoing support")
    return smoothed / totals


def stationary_distribution(kernel: Array) -> Array:
    matrix = np.asarray(kernel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.allclose(
        matrix.sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError("kernel must be row stochastic")
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vector = np.real(eigenvectors[:, index])
    if vector.sum() < 0.0:
        vector *= -1.0
    vector = np.maximum(vector, 0.0)
    if vector.sum() <= 0.0:
        raise ValueError("stationary eigenvector is invalid")
    return vector / vector.sum()


def probability_currents(stationary: Array, kernel: Array) -> Array:
    probability = np.asarray(stationary, dtype=float)
    matrix = np.asarray(kernel, dtype=float)
    flow = probability[:, None] * matrix
    return flow - flow.T


def markov_entropy_production(stationary: Array, kernel: Array, epsilon: float = 1e-15) -> float:
    """Discrete-time Schnakenberg rate in nats per attempted update."""

    probability = np.asarray(stationary, dtype=float)
    matrix = np.asarray(kernel, dtype=float)
    flow = probability[:, None] * matrix
    reverse = flow.T
    active = (flow > float(epsilon)) & (reverse > float(epsilon))
    current = flow - reverse
    result = 0.5 * np.sum(current[active] * np.log(flow[active] / reverse[active]))
    return float(max(result, 0.0))


def transition_pair_irreversibility(counts: Array, pseudocount: float = 0.5) -> float:
    """Empirical pair KL in nats per observed transition opportunity."""

    raw = np.asarray(counts, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or np.any(raw < 0.0):
        raise ValueError("invalid transition counts")
    support = (raw + raw.T) > 0.0
    np.fill_diagonal(support, np.diag(raw) > 0.0)
    adjusted = raw + float(pseudocount) * support
    total = float(adjusted.sum())
    if total <= 0.0:
        raise ValueError("empty transition table")
    joint = adjusted / total
    reverse = joint.T
    active = support & (joint > 0.0) & (reverse > 0.0)
    return float(max(np.sum(joint[active] * np.log(joint[active] / reverse[active])), 0.0))


def block_time_reversal_kl(states: Sequence[int], block_length: int, pseudocount: float = 0.5) -> float:
    """Forward/reversed word KL, normalized to nats per transition."""

    values = tuple(int(value) for value in states)
    length = int(block_length)
    if length < 2 or len(values) < length:
        raise ValueError("invalid block length")
    counts: Dict[Tuple[int, ...], float] = {}
    for start in range(len(values) - length + 1):
        word = values[start : start + length]
        counts[word] = counts.get(word, 0.0) + 1.0
    support = set(counts)
    support.update(tuple(reversed(word)) for word in tuple(support))
    adjusted = {word: counts.get(word, 0.0) + float(pseudocount) for word in support}
    total = float(sum(adjusted.values()))
    divergence = 0.0
    for word, count in adjusted.items():
        probability = count / total
        divergence += probability * math.log(count / adjusted[tuple(reversed(word))])
    return float(max(divergence, 0.0) / float(length - 1))


def time_shuffle_floor(
    states: Sequence[int],
    block_length: int,
    pseudocount: float,
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    values = np.asarray(states, dtype=int)
    rng = np.random.default_rng(int(seed))
    samples = np.asarray(
        [
            block_time_reversal_kl(values[rng.permutation(values.size)], block_length, pseudocount)
            for _ in range(int(replicates))
        ],
        dtype=float,
    )
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "replicates": float(samples.size),
    }


def conditional_mutual_information_history(states: Sequence[int], lag: int, pseudocount: float = 0.1) -> float:
    """I(X[t+1]; X[t-lag] | X[t-lag+1:t]) for a discrete path."""

    values = np.asarray(states, dtype=int)
    order = int(lag)
    if order < 1 or values.size <= order + 1:
        raise ValueError("trajectory too short for requested history")
    triples: Dict[Tuple[int, Tuple[int, ...], int], float] = {}
    for time in range(order, values.size - 1):
        old = int(values[time - order])
        context = tuple(int(value) for value in values[time - order + 1 : time + 1])
        new = int(values[time + 1])
        triples[(old, context, new)] = triples.get((old, context, new), 0.0) + 1.0
    old_context: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    new_context: Dict[Tuple[Tuple[int, ...], int], float] = {}
    context_count: Dict[Tuple[int, ...], float] = {}
    for (old, context, new), count in triples.items():
        old_context[(old, context)] = old_context.get((old, context), 0.0) + count
        new_context[(context, new)] = new_context.get((context, new), 0.0) + count
        context_count[context] = context_count.get(context, 0.0) + count
    total = float(sum(triples.values()))
    result = 0.0
    for (old, context, new), count in triples.items():
        numerator = (count + pseudocount) * (context_count[context] + pseudocount)
        denominator = (old_context[(old, context)] + pseudocount) * (new_context[(context, new)] + pseudocount)
        result += count / total * math.log(numerator / denominator)
    return float(max(result, 0.0))


def history_log_likelihood(states: Sequence[int], order: int, pseudocount: float = 0.5) -> float:
    """Chronological held-out log likelihood for Markov orders one to three."""

    values = np.asarray(states, dtype=int)
    history = int(order)
    if history < 1 or history > 3 or values.size < 4 * history + 4:
        raise ValueError("invalid history order or trajectory length")
    split = max(2 * history + 1, int(0.7 * values.size))
    counts: Dict[Tuple[int, ...], Dict[int, float]] = {}
    alphabet = sorted(int(value) for value in np.unique(values))
    for time in range(history, split):
        context = tuple(int(value) for value in values[time - history : time])
        destination = int(values[time])
        counts.setdefault(context, {})[destination] = counts.setdefault(context, {}).get(destination, 0.0) + 1.0
    log_likelihood = 0.0
    observations = 0
    for time in range(split, values.size):
        context = tuple(int(value) for value in values[time - history : time])
        destination = int(values[time])
        row = counts.get(context, {})
        denominator = sum(row.values()) + float(pseudocount) * len(alphabet)
        probability = (row.get(destination, 0.0) + float(pseudocount)) / denominator
        log_likelihood += math.log(probability)
        observations += 1
    return float(log_likelihood / max(observations, 1))


def autocorrelation(values: Sequence[float], maximum_lag: int) -> Array:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size < 3:
        raise ValueError("autocorrelation requires a one-dimensional series")
    centered = series - np.mean(series)
    variance = float(np.dot(centered, centered) / series.size)
    output = np.zeros(min(int(maximum_lag), series.size - 1) + 1, dtype=float)
    output[0] = 1.0
    if variance <= 1e-15:
        return output
    for lag in range(1, output.size):
        output[lag] = float(np.dot(centered[:-lag], centered[lag:]) / ((series.size - lag) * variance))
    return output


def integrated_autocorrelation_time(values: Sequence[float], maximum_lag: int) -> float:
    correlations = autocorrelation(values, maximum_lag)
    positive: List[float] = []
    for value in correlations[1:]:
        if value <= 0.0:
            break
        positive.append(float(value))
    return float(1.0 + 2.0 * sum(positive))


def network_observables(
    beliefs: Sequence[int],
    actions: Sequence[int],
    adjacency: Array,
    symmetric_weights: Array,
    private_fields: Sequence[int],
    j_b: float,
    j_a: float,
    coupling_k: float,
) -> Dict[str, float]:
    b = np.asarray(beliefs, dtype=float)
    a = np.asarray(actions, dtype=float)
    support = np.asarray(adjacency, dtype=bool)
    ws = np.asarray(symmetric_weights, dtype=float)
    h = np.asarray(private_fields, dtype=float)
    if b.shape != a.shape or ws.shape != (b.size, b.size):
        raise ValueError("observable shapes do not align")
    edges = np.transpose(np.triu(support, 1).nonzero())
    belief_disagreement = float(np.mean([b[i] != b[j] for i, j in edges])) if edges.size else 0.0
    action_disagreement = float(np.mean([a[i] != a[j] for i, j in edges])) if edges.size else 0.0
    energy = (
        -0.5 * float(j_b) * float(b.dot(ws).dot(b))
        - 0.5 * float(j_a) * float(a.dot(ws).dot(a))
        - float(coupling_k) * float(a.dot(b))
        - float(h.dot(b))
    )
    return {
        "belief_magnetization": float(np.mean(b)),
        "action_magnetization": float(np.mean(a)),
        "belief_action_overlap": float(np.mean(b * a)),
        "belief_disagreement": belief_disagreement,
        "action_disagreement": action_disagreement,
        "reference_energy": float(energy),
        "reference_energy_per_agent": float(energy / b.size),
    }


def graph_distance_correlations(states: Array, adjacency: Array) -> Dict[int, float]:
    """Time-averaged spin correlation by graph distance."""

    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != np.asarray(adjacency).shape[0]:
        raise ValueError("state trajectory and graph do not align")
    graph = nx.from_numpy_array(np.asarray(adjacency, dtype=int))
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    by_distance: Dict[int, List[float]] = {}
    centered = values - values.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    for i in range(values.shape[1]):
        for j in range(i + 1, values.shape[1]):
            distance = int(distances[i][j])
            denominator = float(scale[i] * scale[j])
            correlation = 0.0 if denominator <= 1e-12 else float(np.mean(centered[:, i] * centered[:, j]) / denominator)
            by_distance.setdefault(distance, []).append(correlation)
    return {distance: float(np.mean(items)) for distance, items in sorted(by_distance.items())}


def binary_mutual_information(x: Sequence[int], y: Sequence[int], pseudocount: float = 0.5) -> float:
    first = (np.asarray(x, dtype=int) > 0).astype(int)
    second = (np.asarray(y, dtype=int) > 0).astype(int)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("binary mutual information inputs must align")
    joint = np.full((2, 2), float(pseudocount), dtype=float)
    np.add.at(joint, (first, second), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    return float(np.sum(joint * np.log(joint / (px[:, None] * py[None, :]))))


def binary_transfer_entropy(source: Sequence[int], target: Sequence[int], pseudocount: float = 0.5) -> float:
    """I(source[t]; target[t+1] | target[t]) for binary trajectories."""

    x = (np.asarray(source, dtype=int) > 0).astype(int)
    y = (np.asarray(target, dtype=int) > 0).astype(int)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("transfer entropy inputs must align")
    table = np.full((2, 2, 2), float(pseudocount), dtype=float)
    np.add.at(table, (x[:-1], y[:-1], y[1:]), 1.0)
    p = table / table.sum()
    p_xy = p.sum(axis=2)
    p_yz = p.sum(axis=0)
    p_y = p.sum(axis=(0, 2))
    result = 0.0
    for xi in range(2):
        for yi in range(2):
            for zi in range(2):
                result += p[xi, yi, zi] * math.log(
                    (p[xi, yi, zi] * p_y[yi]) / (p_xy[xi, yi] * p_yz[yi, zi])
                )
    return float(max(result, 0.0))


def information_permutation_floor(
    source: Sequence[int],
    target: Sequence[int],
    pseudocount: float,
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    """Permutation bias floors for pairwise MI and lag-one transfer entropy."""

    x = np.asarray(source, dtype=int)
    y = np.asarray(target, dtype=int)
    if x.shape != y.shape or x.size < 4:
        raise ValueError("permutation inputs must be aligned")
    rng = np.random.default_rng(int(seed))
    mi = np.empty(int(replicates), dtype=float)
    te = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        permuted = x[rng.permutation(x.size)]
        mi[index] = binary_mutual_information(permuted, y, pseudocount)
        te[index] = binary_transfer_entropy(permuted, y, pseudocount)
    return {
        "mutual_information_mean": float(np.mean(mi)),
        "mutual_information_95": float(np.quantile(mi, 0.95)),
        "transfer_entropy_mean": float(np.mean(te)),
        "transfer_entropy_95": float(np.quantile(te, 0.95)),
        "replicates": float(replicates),
    }


def susceptibility(magnetization: Sequence[float], n_agents: int) -> float:
    return float(int(n_agents) * np.var(np.asarray(magnetization, dtype=float), ddof=1))


def hysteresis_area(fields: Sequence[float], magnetization: Sequence[float]) -> float:
    x = np.asarray(fields, dtype=float)
    y = np.asarray(magnetization, dtype=float)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("hysteresis curve arrays must align")
    return float(abs(np.trapz(y, x)))


def paired_cluster_bootstrap(
    values: Mapping[str, Sequence[float]], replicates: int, seed: int
) -> Dict[str, float]:
    if not values:
        raise ValueError("at least one independent cluster is required")
    cluster_means = np.asarray([np.mean(np.asarray(values[key], dtype=float)) for key in sorted(values)], dtype=float)
    if np.any(~np.isfinite(cluster_means)):
        raise ValueError("nonfinite cluster estimate")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=float)
    for index in range(samples.size):
        chosen = rng.integers(0, cluster_means.size, cluster_means.size)
        samples[index] = float(np.mean(cluster_means[chosen]))
    return {
        "estimate": float(np.mean(cluster_means)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "independent_clusters": float(cluster_means.size),
        "bootstrap_replicates": float(samples.size),
    }


def known_three_state_cycle(clockwise: float, counterclockwise: float, stay: float) -> Array:
    if min(clockwise, counterclockwise, stay) < 0.0 or not np.isclose(clockwise + counterclockwise + stay, 1.0):
        raise ValueError("cycle probabilities must be nonnegative and sum to one")
    kernel = np.zeros((3, 3), dtype=float)
    for state in range(3):
        kernel[state, state] = stay
        kernel[state, (state + 1) % 3] = clockwise
        kernel[state, (state - 1) % 3] = counterclockwise
    return kernel


def exact_cycle_entropy_production(clockwise: float, counterclockwise: float) -> float:
    if clockwise <= 0.0 or counterclockwise <= 0.0:
        raise ValueError("directed cycle rates must be positive")
    return float((clockwise - counterclockwise) * math.log(clockwise / counterclockwise))


def stationary_chain_sample(kernel: Array, transitions: int, seed: int) -> Array:
    matrix = np.asarray(kernel, dtype=float)
    stationary = stationary_distribution(matrix)
    rng = np.random.default_rng(int(seed))
    state = int(rng.choice(matrix.shape[0], p=stationary))
    path = np.empty(int(transitions) + 1, dtype=int)
    path[0] = state
    for index in range(int(transitions)):
        state = int(rng.choice(matrix.shape[0], p=matrix[state]))
        path[index + 1] = state
    return path
