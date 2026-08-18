"""Panel-level inference and finite-state irreversibility utilities for V11."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def safe_logit(probability: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(values / (1.0 - values))


def paired_cluster_bootstrap(
    effects: Mapping[str, Sequence[float]],
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    """Bootstrap independent clusters while preserving observations within each cluster."""

    if not effects:
        raise ValueError("at least one independent cluster is required")
    keys = sorted(effects)
    cluster_means = np.asarray([np.mean(np.asarray(effects[key], dtype=float)) for key in keys])
    if not np.all(np.isfinite(cluster_means)):
        raise ValueError("nonfinite cluster effect")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        selected = rng.integers(0, len(keys), len(keys))
        draws[index] = float(np.mean(cluster_means[selected]))
    return {
        "estimate": float(np.mean(cluster_means)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_probability_positive": float(np.mean(draws > 0.0)),
        "independent_clusters": float(len(keys)),
        "bootstrap_replicates": float(replicates),
    }


def calibration_summary(reported_probability: Sequence[float], binary_choice: Sequence[int], bins: int = 8) -> Dict[str, float]:
    probability = np.asarray(reported_probability, dtype=float)
    target = np.asarray(binary_choice, dtype=float)
    if probability.shape != target.shape or probability.size == 0:
        raise ValueError("calibration arrays must be nonempty and aligned")
    if np.any((probability < 0.0) | (probability > 1.0)) or np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("invalid probability or binary target")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    index = np.minimum(np.digitize(probability, edges[1:-1]), int(bins) - 1)
    ece = 0.0
    occupied = 0
    for bin_index in range(int(bins)):
        selected = index == bin_index
        if not np.any(selected):
            continue
        occupied += 1
        ece += float(np.mean(selected)) * abs(float(np.mean(probability[selected]) - np.mean(target[selected])))
    return {
        "count": float(probability.size),
        "brier": float(np.mean((probability - target) ** 2)),
        "expected_calibration_error": float(ece),
        "mean_reported_probability": float(np.mean(probability)),
        "empirical_right_frequency": float(np.mean(target)),
        "occupied_bins": float(occupied),
    }


def fit_reliability_response(reliability: Sequence[float], signed_logit_change: Sequence[float]) -> Dict[str, float]:
    r = np.asarray(reliability, dtype=float)
    change = np.asarray(signed_logit_change, dtype=float)
    if r.shape != change.shape or r.size < 2:
        raise ValueError("reliability response needs aligned observations")
    normative = np.log(r / (1.0 - r))
    design = np.column_stack([np.ones(r.size), normative])
    coefficients = np.linalg.lstsq(design, change, rcond=None)[0]
    fitted = design.dot(coefficients)
    return {
        "intercept": float(coefficients[0]),
        "normative_llr_slope": float(coefficients[1]),
        "r_squared": float(1.0 - np.sum((change - fitted) ** 2) / max(np.sum((change - np.mean(change)) ** 2), 1e-12)),
    }


def normalize_transition_counts(counts: np.ndarray, pseudocount: float = 0.0) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or np.any(values < 0.0):
        raise ValueError("transition counts must be a square nonnegative matrix")
    values = values + float(pseudocount)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("every source state must have at least one transition")
    return values / totals


def stationary_distribution(kernel: np.ndarray) -> np.ndarray:
    matrix = np.asarray(kernel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel must be square")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("kernel rows must sum to one")
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    selected = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vector = np.real(eigenvectors[:, selected])
    if np.sum(vector) < 0.0:
        vector = -vector
    vector = np.maximum(vector, 0.0)
    if vector.sum() <= 0.0:
        raise ValueError("stationary eigenvector is invalid")
    return vector / vector.sum()


def entropy_production_per_update(stationary: np.ndarray, kernel: np.ndarray, epsilon: float = 1e-15) -> float:
    """Stationary discrete-time entropy production per attempted update."""

    probability = np.asarray(stationary, dtype=float)
    matrix = np.asarray(kernel, dtype=float)
    flow = probability[:, None] * matrix
    reverse = flow.T
    mask = (flow > epsilon) & (reverse > epsilon)
    current = flow - reverse
    value = 0.5 * np.sum(current[mask] * np.log(flow[mask] / reverse[mask]))
    return float(max(value, 0.0))


def trajectory_transition_counts(states: Sequence[int], state_count: int) -> np.ndarray:
    values = np.asarray(states, dtype=int)
    if values.size < 2 or np.any(values < 0) or np.any(values >= int(state_count)):
        raise ValueError("invalid trajectory states")
    counts = np.zeros((int(state_count), int(state_count)), dtype=float)
    np.add.at(counts, (values[:-1], values[1:]), 1.0)
    return counts


def block_time_reversal_kl(states: Sequence[int], block_length: int, pseudocount: float = 0.5) -> float:
    """KL between forward and reversed empirical word distributions."""

    values = tuple(int(value) for value in states)
    length = int(block_length)
    if length < 2 or len(values) < length:
        raise ValueError("trajectory is too short for the requested block")
    forward: Dict[Tuple[int, ...], int] = {}
    for start in range(len(values) - length + 1):
        word = values[start : start + length]
        forward[word] = forward.get(word, 0) + 1
    support = sorted(set(forward) | {tuple(reversed(word)) for word in forward})
    total = sum(forward.values()) + float(pseudocount) * len(support)
    divergence = 0.0
    for word in support:
        p = (forward.get(word, 0) + float(pseudocount)) / total
        q = (forward.get(tuple(reversed(word)), 0) + float(pseudocount)) / total
        divergence += p * math.log(p / q)
    return float(max(divergence, 0.0))


def conditional_mutual_information_history(states: Sequence[int], history: int, pseudocount: float = 0.1) -> float:
    """I(X[t+1]; X[t-history] | intervening history) for discrete states."""

    values = np.asarray(states, dtype=int)
    lag = int(history)
    if lag < 1 or values.size <= lag + 1:
        raise ValueError("trajectory is too short")
    joint: Dict[Tuple[int, Tuple[int, ...], int], float] = {}
    context_counts: Dict[Tuple[int, ...], float] = {}
    old_context: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    new_context: Dict[Tuple[Tuple[int, ...], int], float] = {}
    for t in range(lag, values.size - 1):
        old = int(values[t - lag])
        context = tuple(int(item) for item in values[t - lag + 1 : t + 1])
        new = int(values[t + 1])
        joint[(old, context, new)] = joint.get((old, context, new), 0.0) + 1.0
        context_counts[context] = context_counts.get(context, 0.0) + 1.0
        old_context[(old, context)] = old_context.get((old, context), 0.0) + 1.0
        new_context[(context, new)] = new_context.get((context, new), 0.0) + 1.0
    total = float(sum(joint.values()))
    if total <= 0.0:
        return 0.0
    value = 0.0
    for (old, context, new), count in joint.items():
        numerator = (count + pseudocount) * (context_counts[context] + pseudocount)
        denominator = (old_context[(old, context)] + pseudocount) * (new_context[(context, new)] + pseudocount)
        value += (count / total) * math.log(numerator / denominator)
    return float(max(value, 0.0))


def wire_bytes(payloads: Iterable[bytes]) -> int:
    return int(sum(len(payload) for payload in payloads))
