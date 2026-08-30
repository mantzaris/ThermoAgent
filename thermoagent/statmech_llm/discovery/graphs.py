"""Matched reciprocal and nonreciprocal delivery networks for V12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class DeliveryGraph:
    topology: str
    adjacency: np.ndarray
    symmetric: np.ndarray
    circulation: np.ndarray
    weights: np.ndarray
    alpha: float
    orientation_seed: int

    def validate(self) -> None:
        n = self.adjacency.shape[0]
        if self.adjacency.shape != (n, n) or self.weights.shape != (n, n):
            raise ValueError("delivery matrices must be square and aligned")
        if not np.array_equal(self.adjacency, self.adjacency.T) or np.any(np.diag(self.adjacency)):
            raise ValueError("base support must be a simple undirected graph")
        if not np.allclose(self.symmetric, self.symmetric.T, atol=1e-12):
            raise ValueError("reciprocal component must be symmetric")
        if not np.allclose(self.circulation, -self.circulation.T, atol=1e-12):
            raise ValueError("directed component must be antisymmetric")
        if not np.allclose(self.circulation.sum(axis=1), 0.0, atol=1e-10):
            raise ValueError("circulation must preserve every node's weighted degree")
        if np.any(self.weights < -1e-12) or np.any((self.weights > 0.0) != (self.adjacency > 0)):
            raise ValueError("weights must be nonnegative on unchanged support")
        if not np.allclose(self.weights.sum(axis=1), self.symmetric.sum(axis=1), atol=1e-10):
            raise ValueError("weighted out-degree changed with nonreciprocity")

    @property
    def n_agents(self) -> int:
        return int(self.weights.shape[0])

    def reverse(self) -> "DeliveryGraph":
        result = DeliveryGraph(
            self.topology,
            self.adjacency.copy(),
            self.symmetric.copy(),
            -self.circulation.copy(),
            self.weights.T.copy(),
            self.alpha,
            self.orientation_seed,
        )
        result.validate()
        return result

    def diagnostics(self) -> Dict[str, float]:
        numerator = float(np.linalg.norm(self.weights - self.weights.T, ord="fro"))
        denominator = float(np.linalg.norm(self.weights + self.weights.T, ord="fro")) + 1e-12
        support = self.adjacency > 0
        reciprocal = np.minimum(self.weights, self.weights.T)[support].sum()
        total = self.weights[support].sum()
        graph = nx.from_numpy_array(self.adjacency)
        # The stored circulation is the unit perturbation generator.  Report
        # the applied circulation, which must vanish in the reciprocal arm.
        circulation = float(abs(self.alpha) * np.sum(np.abs(self.circulation)) / 2.0)
        return {
            "n_agents": float(self.n_agents),
            "edge_count": float(graph.number_of_edges()),
            "mean_degree": float(np.mean([degree for _, degree in graph.degree()])),
            "density": float(nx.density(graph)),
            "clustering": float(nx.average_clustering(graph)),
            "components": float(nx.number_connected_components(graph)),
            "asymmetry_frobenius": numerator / denominator,
            "edge_reciprocity_weighted": float(reciprocal / max(total, 1e-12)),
            "circulation_l1": circulation,
            "mean_weighted_out_degree": float(np.mean(self.weights.sum(axis=1))),
            "mean_weighted_in_degree": float(np.mean(self.weights.sum(axis=0))),
        }


def _ring(n_agents: int) -> np.ndarray:
    matrix = np.zeros((n_agents, n_agents), dtype=int)
    for i in range(n_agents):
        matrix[i, (i + 1) % n_agents] = 1
        matrix[(i + 1) % n_agents, i] = 1
    return matrix


def _modular_regular(n_agents: int, seed: int) -> np.ndarray:
    """Connected fixed-degree modular graph, avoiding V10's increasing degree."""

    if n_agents < 8 or n_agents % 2:
        raise ValueError("modular topology needs an even N of at least eight")
    half = n_agents // 2
    matrix = np.zeros((n_agents, n_agents), dtype=int)
    # Each community starts as a degree-two ring.
    for offset in (0, half):
        for local in range(half):
            i = offset + local
            j = offset + ((local + 1) % half)
            matrix[i, j] = matrix[j, i] = 1
    # A perfect cross-community matching gives degree three and no bridges.
    rng = np.random.default_rng(int(seed))
    matching = rng.permutation(half)
    for left, right_local in enumerate(matching):
        right = half + int(right_local)
        matrix[left, right] = matrix[right, left] = 1
    return matrix


def base_adjacency(n_agents: int, topology: str, seed: int) -> np.ndarray:
    n = int(n_agents)
    if n < 3:
        raise ValueError("at least three agents are required")
    if topology == "ring":
        matrix = _ring(n)
    elif topology == "modular":
        matrix = _modular_regular(n, int(seed))
    else:
        raise ValueError("topology must be ring or modular")
    graph = nx.from_numpy_array(matrix)
    if not nx.is_connected(graph):
        raise AssertionError("generated delivery graph is disconnected")
    return matrix


def _cycle_circulation(adjacency: np.ndarray, seed: int) -> np.ndarray:
    graph = nx.from_numpy_array(np.asarray(adjacency, dtype=int))
    cycles: List[List[int]] = [list(cycle) for cycle in nx.cycle_basis(graph)]
    if not cycles:
        raise ValueError("nonreciprocity requires at least one supported cycle")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(cycles)
    flow = np.zeros_like(adjacency, dtype=float)
    for cycle_index, cycle in enumerate(cycles):
        direction = 1.0 if (cycle_index + int(seed)) % 2 == 0 else -1.0
        amplitude = direction * (0.6 + 0.4 * float(rng.random()))
        for index, source in enumerate(cycle):
            target = cycle[(index + 1) % len(cycle)]
            flow[source, target] += amplitude
            flow[target, source] -= amplitude
    maximum = float(np.max(np.abs(flow)))
    if maximum <= 0.0:
        raise AssertionError("orientation construction produced zero circulation")
    # Unit maximum and zero row sum; each oriented cycle is divergence-free.
    flow /= maximum
    if not np.allclose(flow.sum(axis=1), 0.0, atol=1e-12):
        raise AssertionError("cycle construction violated flow conservation")
    return flow


def build_delivery_graph(
    n_agents: int,
    topology: str,
    graph_seed: int,
    orientation_seed: int,
    alpha: float,
    reverse_orientation: bool = False,
) -> DeliveryGraph:
    if not 0.0 <= float(alpha) <= 0.8:
        raise ValueError("V12 alpha must be in [0,0.8]")
    adjacency = base_adjacency(int(n_agents), topology, int(graph_seed))
    degrees = adjacency.sum(axis=1).astype(float)
    if not np.allclose(degrees, degrees[0]):
        raise AssertionError("V12 primary graphs must have controlled regular degree")
    symmetric = adjacency.astype(float) / float(degrees[0])
    circulation = _cycle_circulation(adjacency, int(orientation_seed)) / float(degrees[0])
    if reverse_orientation:
        circulation *= -1.0
    weights = symmetric + float(alpha) * circulation
    result = DeliveryGraph(
        topology=topology,
        adjacency=adjacency,
        symmetric=symmetric,
        circulation=circulation,
        weights=weights,
        alpha=float(alpha),
        orientation_seed=int(orientation_seed),
    )
    result.validate()
    return result


def select_recipient(weights: np.ndarray, sender: int, uniform: float) -> int:
    probabilities = np.asarray(weights[int(sender)], dtype=float)
    if probabilities.sum() <= 0.0:
        raise ValueError("sender has no delivery opportunity")
    probabilities = probabilities / probabilities.sum()
    cumulative = np.cumsum(probabilities)
    index = min(int(np.searchsorted(cumulative, float(uniform), side="right")), probabilities.size - 1)
    if probabilities[index] <= 0.0:
        positive = np.flatnonzero(probabilities > 0.0)
        index = int(positive[-1])
    return int(index)


def matched_opportunity_schedule(n_agents: int, updates: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random-sequential agents plus fixed uniforms, common across graph arms."""

    rng = np.random.default_rng(int(seed))
    scheduled: List[int] = []
    while len(scheduled) < int(updates):
        scheduled.extend(int(value) for value in rng.permutation(int(n_agents)))
    return np.asarray(scheduled[: int(updates)], dtype=int), rng.random(int(updates))
