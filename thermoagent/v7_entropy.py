"""Distributed generalized-information measures for coupled V7 networks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .v6_entropy import (
    entropy_spectrum, generalized_disagreement, gini_simpson_impurity,
    graph_weighted_disagreement, reliability_weights, shannon_entropy,
    temporal_information_state, tsallis_entropy, weighted_pooled_belief,
)


@dataclass(frozen=True)
class V7EntropySketch:
    sender: str
    focal_asset: str
    belief_distribution: Tuple[float, ...]
    telemetry_confidence: float
    sent_step: int
    encoded_bytes: int
    hop_count: int = 0


def encoded_sketch_bytes(state_count: int) -> int:
    """Bounded wire accounting: header + fp16 belief + confidence + age."""
    return int(20 + 2 * int(state_count) + 4)


def weighted_information_state(
    beliefs: Sequence[Sequence[float]],
    confidences: Sequence[float],
    message_ages: Sequence[float],
    q: float = 1.0,
) -> Dict[str, object]:
    if not beliefs:
        raise ValueError("distributed state requires at least one belief")
    if not (len(beliefs) == len(confidences) == len(message_ages)):
        raise ValueError("belief, confidence, and age lengths must match")
    raw_weights = [
        max(1e-6, float(confidence)) * math.exp(-0.20 * max(float(age), 0.0))
        for confidence, age in zip(confidences, message_ages)
    ]
    weights = reliability_weights(raw_weights)
    pooled = weighted_pooled_belief(beliefs, weights)
    local = np.asarray([tsallis_entropy(value, q) for value in beliefs], dtype=float)
    average = float(np.dot(weights, local))
    pooled_entropy = tsallis_entropy(pooled, q)
    disagreement = generalized_disagreement(beliefs, weights, q)
    return {
        "weights": tuple(float(value) for value in weights),
        "pooled_belief": tuple(float(value) for value in pooled),
        "average_local_uncertainty": average,
        "pooled_uncertainty": pooled_entropy,
        "generalized_disagreement": disagreement,
        "consensus": 1.0 - disagreement,
        "gini_simpson_pooled": gini_simpson_impurity(pooled),
        "entropy_spectrum_pooled": entropy_spectrum(pooled),
    }


def graph_disagreement_from_active_edges(
    beliefs: Mapping[str, Sequence[float]],
    communication_graph: nx.Graph,
    q: float = 1.0,
) -> float:
    edges = []
    for first_node, second_node, data in communication_graph.edges(data=True):
        first = str(communication_graph.nodes[first_node].get("agent_id", first_node))
        second = str(communication_graph.nodes[second_node].get("agent_id", second_node))
        if first not in beliefs or second not in beliefs:
            continue
        if not bool(data.get("available", True)):
            continue
        latency = max(float(data.get("latency", 1.0)), 1.0)
        weight = (
            float(data.get("reliability", 1.0))
            * float(data.get("trust", 1.0)) / latency
        )
        edges.append((first, second, weight))
    return graph_weighted_disagreement(beliefs, edges, q=q) if edges else 0.0


def spectral_graph_entropy(graph: nx.Graph) -> float:
    """Normalized von-Neumann-style entropy of the graph Laplacian spectrum."""
    simple = nx.Graph(graph)
    if simple.number_of_nodes() < 2 or simple.number_of_edges() == 0:
        return 0.0
    # Construct the dense Laplacian directly.  V7 graphs contain at most 80
    # agents, and this avoids a fragile NetworkX/SciPy sparse-array version
    # coupling while remaining numerically exact for the study scale.
    nodes = list(simple.nodes())
    adjacency = nx.to_numpy_array(simple, nodelist=nodes, dtype=float, weight=None)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 1e-12] / total
    if len(probabilities) <= 1:
        return 0.0
    return float(
        -np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities))
    )


def allocation_entropy(values: Sequence[float]) -> float:
    masses = np.maximum(np.asarray(values, dtype=float), 0.0)
    if len(masses) < 2 or float(masses.sum()) <= 1e-12:
        return 0.0
    return shannon_entropy(masses)


def economic_gini(values: Sequence[float]) -> float:
    """Economic Gini over nonnegative allocations, distinct from impurity."""
    masses = np.sort(np.maximum(np.asarray(values, dtype=float), 0.0))
    if not len(masses) or float(masses.sum()) <= 1e-12:
        return 0.0
    count = len(masses)
    ranks = np.arange(1, count + 1, dtype=float)
    return float(
        np.sum((2.0 * ranks - count - 1.0) * masses)
        / (count * float(masses.sum()))
    )
