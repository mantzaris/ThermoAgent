"""Distributed macrostate estimation over time-varying communication graphs."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np


def metropolis_matrix(agent_ids: Sequence[str], edges: Iterable[Tuple[str, str]]) -> np.ndarray:
    ids = list(agent_ids)
    index = {agent_id: i for i, agent_id in enumerate(ids)}
    neighbors: Dict[str, Set[str]] = {agent_id: set() for agent_id in ids}
    for left, right in edges:
        if left in index and right in index and left != right:
            neighbors[left].add(right)
            neighbors[right].add(left)
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    for left in ids:
        i = index[left]
        for right in neighbors[left]:
            j = index[right]
            matrix[i, j] = 1.0 / (1.0 + max(len(neighbors[left]), len(neighbors[right])))
        matrix[i, i] = 1.0 - matrix[i].sum()
    return matrix


def gossip_distributions(
    sketches: Mapping[str, Sequence[float]], edges_by_round: Sequence[Iterable[Tuple[str, str]]]
) -> Dict[str, np.ndarray]:
    estimates, _ = gossip_distributions_with_trace(sketches, edges_by_round)
    return estimates


def gossip_distributions_with_trace(
    sketches: Mapping[str, Sequence[float]],
    edges_by_round: Sequence[Iterable[Tuple[str, str]]],
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
    """Run Metropolis gossip and retain each agent's link-local round state."""

    ids = sorted(sketches)
    values = np.asarray([sketches[agent_id] for agent_id in ids], dtype=float)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("sketches must be equally sized nonnegative vectors")
    trace: List[Dict[str, np.ndarray]] = []
    for edges in edges_by_round:
        matrix = metropolis_matrix(ids, edges)
        values = matrix.dot(values)
        normalized_round = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
        trace.append({
            agent_id: normalized_round[index].copy()
            for index, agent_id in enumerate(ids)
        })
    normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    return (
        {agent_id: normalized[i].copy() for i, agent_id in enumerate(ids)},
        trace,
    )


def local_consensus_residuals(
    estimates: Mapping[str, Sequence[float]],
    edges: Iterable[Tuple[str, str]],
) -> Dict[str, float]:
    """Locally observable disagreement with current communication neighbors.

    This deliberately does not compare against evaluator-only global
    occupancy. An isolated agent can observe that it has no neighbors and gets
    a maximal uncertainty marker; a disconnected component cannot know that a
    different component disagrees with it.
    """

    ids = sorted(estimates)
    neighbors: Dict[str, Set[str]] = {agent_id: set() for agent_id in ids}
    for left, right in edges:
        if left in neighbors and right in neighbors and left != right:
            neighbors[left].add(right)
            neighbors[right].add(left)
    residuals: Dict[str, float] = {}
    for agent_id in ids:
        if not neighbors[agent_id]:
            residuals[agent_id] = 1.0
            continue
        own = np.asarray(estimates[agent_id], dtype=float)
        errors = [
            np.mean((own - np.asarray(estimates[neighbor], dtype=float)) ** 2)
            for neighbor in sorted(neighbors[agent_id])
        ]
        residuals[agent_id] = float(np.sqrt(np.mean(errors)))
    return residuals


def one_hot_sketch(state: int, k: int = 27, alpha: float = 0.1, population_size: int = 1) -> np.ndarray:
    if population_size < 1:
        raise ValueError("population_size must be positive")
    # Divide the pseudocount across the known population so the consensus
    # average equals the evaluator's once-smoothed occupancy distribution.
    sketch = np.full(k, alpha / population_size, dtype=float)
    sketch[int(state)] += 1.0
    return sketch / sketch.sum()


def consensus_rmse(estimates: Mapping[str, Sequence[float]], exact: Sequence[float]) -> float:
    exact_arr = np.asarray(exact, dtype=float)
    errors = [np.mean((np.asarray(value) - exact_arr) ** 2) for value in estimates.values()]
    return float(np.sqrt(np.mean(errors))) if errors else 0.0
