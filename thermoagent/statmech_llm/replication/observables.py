"""Predeclared V13 collective observables and reduced-state utilities."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np

from thermoagent.statmech_llm.discovery.estimators import (
    integrated_autocorrelation_time,
    shannon_entropy,
)


def _binary(values: Sequence[int]) -> np.ndarray:
    output = np.asarray(values, dtype=int)
    if output.ndim != 1 or output.size == 0 or np.any(~np.isin(output, (-1, 1))):
        raise ValueError("expected a nonempty binary spin vector")
    return output


def reference_energy(
    beliefs: Sequence[int],
    actions: Sequence[int],
    symmetric_weights: np.ndarray,
    private_fields: Sequence[int],
    task_fields: Sequence[float] | None = None,
    j_b: float = 1.0,
    j_a: float = 0.65,
    coupling_k: float = 0.8,
) -> float:
    b = _binary(beliefs).astype(float)
    a = _binary(actions).astype(float)
    w = np.asarray(symmetric_weights, dtype=float)
    h = np.asarray(private_fields, dtype=float)
    g = np.zeros_like(h) if task_fields is None else np.asarray(task_fields, dtype=float)
    if w.shape != (b.size, b.size) or h.shape != b.shape or g.shape != b.shape:
        raise ValueError("reference-energy inputs do not align")
    return float(
        -0.5 * float(j_b) * b.dot(w).dot(b)
        -0.5 * float(j_a) * a.dot(w).dot(a)
        -float(coupling_k) * a.dot(b)
        -h.dot(b)
        -g.dot(a)
    )


def local_configuration_entropy(
    beliefs: Sequence[int], actions: Sequence[int], adjacency: np.ndarray
) -> float:
    """Entropy of node-local (belief, action, neighbor-majority) words."""

    b = _binary(beliefs)
    a = _binary(actions)
    support = np.asarray(adjacency, dtype=bool)
    if support.shape != (b.size, b.size):
        raise ValueError("adjacency does not align")
    words = []
    for node in range(b.size):
        neighbors = b[support[node]]
        total = int(neighbors.sum()) if neighbors.size else 0
        majority = 1 if total > 0 else (-1 if total < 0 else 0)
        words.append((int(b[node]), int(a[node]), majority))
    _, counts = np.unique(np.asarray(words, dtype=int), axis=0, return_counts=True)
    return shannon_entropy(counts)


def disagreement_density(states: Sequence[int], adjacency: np.ndarray) -> float:
    values = _binary(states)
    support = np.asarray(adjacency, dtype=bool)
    edges = np.transpose(np.triu(support, 1).nonzero())
    return float(np.mean([values[i] != values[j] for i, j in edges])) if edges.size else 0.0


def spatial_correlation(states: Sequence[int], adjacency: np.ndarray) -> float:
    values = _binary(states).astype(float)
    support = np.asarray(adjacency, dtype=bool)
    edges = np.transpose(np.triu(support, 1).nonzero())
    return float(np.mean([values[i] * values[j] for i, j in edges])) if edges.size else 0.0


def instantaneous_state(
    beliefs: Sequence[int],
    actions: Sequence[int],
    adjacency: np.ndarray,
    symmetric_weights: np.ndarray,
    private_fields: Sequence[int],
    j_b: float = 1.0,
    j_a: float = 0.65,
    coupling_k: float = 0.8,
) -> Dict[str, float]:
    b = _binary(beliefs)
    a = _binary(actions)
    energy = reference_energy(b, a, symmetric_weights, private_fields, j_b=j_b, j_a=j_a, coupling_k=coupling_k)
    return {
        "belief_magnetization": float(np.mean(b)),
        "action_magnetization": float(np.mean(a)),
        "belief_action_overlap": float(np.mean(b * a)),
        "belief_disagreement": disagreement_density(b, adjacency),
        "action_disagreement": disagreement_density(a, adjacency),
        "local_configuration_entropy": local_configuration_entropy(b, a, adjacency),
        "reference_energy": energy,
        "reference_energy_per_agent": float(energy / b.size),
        "spatial_belief_correlation": spatial_correlation(b, adjacency),
    }


def encode_binary_configuration(beliefs: Sequence[int], actions: Sequence[int]) -> int:
    b = _binary(beliefs)
    a = _binary(actions)
    value = 0
    for index in range(b.size):
        value |= int(b[index] > 0) << index
        value |= int(a[index] > 0) << (b.size + index)
    return int(value)


def plugin_entropy(values: Sequence[object]) -> float:
    if len(values) == 0:
        return 0.0
    counts: Dict[object, int] = {}
    for value in values:
        key = tuple(value) if isinstance(value, (list, tuple, np.ndarray)) else value
        counts[key] = counts.get(key, 0) + 1
    return shannon_entropy(list(counts.values()))


def conditional_entropy_rate(states: Sequence[int], order: int = 1, pseudocount: float = 0.5) -> float:
    values = np.asarray(states, dtype=int)
    history = int(order)
    if history < 1 or values.size <= history:
        return 0.0
    alphabet = sorted(int(item) for item in np.unique(values))
    contexts: Dict[Tuple[int, ...], Dict[int, float]] = {}
    for index in range(history, values.size):
        context = tuple(int(item) for item in values[index - history : index])
        destination = int(values[index])
        contexts.setdefault(context, {})[destination] = contexts.setdefault(context, {}).get(destination, 0.0) + 1.0
    total = float(values.size - history)
    entropy = 0.0
    for row in contexts.values():
        observations = float(sum(row.values()))
        probabilities = np.asarray([row.get(item, 0.0) + pseudocount for item in alphabet], dtype=float)
        entropy += observations / total * shannon_entropy(probabilities)
    return float(entropy)


def total_correlation(configurations: np.ndarray) -> float:
    """Multi-information of binary columns over a trajectory window."""

    values = np.asarray(configurations, dtype=int)
    if values.ndim != 2 or values.shape[0] < 2 or np.any(~np.isin(values, (-1, 1))):
        raise ValueError("total correlation needs a binary time-by-variable matrix")
    marginal = sum(shannon_entropy([np.sum(values[:, index] < 0), np.sum(values[:, index] > 0)]) for index in range(values.shape[1]))
    packed = [tuple(int(item) for item in row) for row in values]
    joint = plugin_entropy(packed)
    return float(max(marginal - joint, 0.0))


def macrostate_code(row: Mapping[str, float], widths: Mapping[str, float]) -> Tuple[int, ...]:
    keys = (
        "belief_magnetization",
        "action_magnetization",
        "belief_action_overlap",
        "reference_energy_per_agent",
        "belief_disagreement",
    )
    width_keys = (
        "magnetization_width",
        "magnetization_width",
        "overlap_width",
        "energy_per_agent_width",
        "disagreement_width",
    )
    return tuple(int(np.floor(float(row[key]) / float(widths[wkey]) + 0.5)) for key, wkey in zip(keys, width_keys))


def rolling_state_vectors(
    beliefs: np.ndarray,
    actions: np.ndarray,
    instantaneous: Sequence[Mapping[str, float]],
    widths: Mapping[str, float],
    window: int,
) -> list[Dict[str, float]]:
    b = np.asarray(beliefs, dtype=int)
    a = np.asarray(actions, dtype=int)
    if b.shape != a.shape or b.ndim != 2 or len(instantaneous) != b.shape[0]:
        raise ValueError("rolling inputs do not align")
    output: list[Dict[str, float]] = []
    codes = [macrostate_code(row, widths) for row in instantaneous]
    vocabulary = {value: index for index, value in enumerate(sorted(set(codes)))}
    encoded = np.asarray([vocabulary[value] for value in codes], dtype=int)
    for index, row in enumerate(instantaneous):
        start = max(0, index - int(window) + 1)
        b_window = b[start : index + 1]
        a_window = a[start : index + 1]
        state_window = encoded[start : index + 1]
        magnetization = b_window.mean(axis=1)
        energy = np.asarray([float(instantaneous[j]["reference_energy_per_agent"]) for j in range(start, index + 1)])
        if b_window.shape[0] >= 2:
            multi = total_correlation(np.concatenate([b_window, a_window], axis=1))
            susceptibility = float(b.shape[1] * np.var(magnetization, ddof=1))
            energy_variance = float(b.shape[1] * np.var(energy, ddof=1))
        else:
            multi = susceptibility = energy_variance = 0.0
        item = dict(row)
        item.update(
            {
                "configuration_entropy": plugin_entropy(state_window.tolist()),
                "entropy_rate": conditional_entropy_rate(state_window, 1, 0.5),
                "total_correlation": multi,
                "energy_variance": energy_variance,
                "belief_susceptibility": susceptibility,
            }
        )
        output.append(item)
    return output


def integrated_correlation_time(values: Sequence[float]) -> float:
    series = np.asarray(values, dtype=float)
    if series.size < 4:
        return 1.0
    return integrated_autocorrelation_time(series, min(30, max(1, series.size // 4)))


def regularized_mahalanobis_fit(values: np.ndarray, ridge_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("nominal manifold needs multiple observations")
    center = matrix.mean(axis=0)
    covariance = np.cov(matrix, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]])
    scale = float(np.trace(covariance) / covariance.shape[0])
    regularized = (1.0 - float(ridge_fraction)) * covariance + float(ridge_fraction) * max(scale, 1e-8) * np.eye(covariance.shape[0])
    return center, np.linalg.pinv(regularized)


def mahalanobis_distance(values: np.ndarray, center: np.ndarray, precision: np.ndarray) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(values, dtype=float))
    delta = matrix - np.asarray(center, dtype=float)
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0.0))
