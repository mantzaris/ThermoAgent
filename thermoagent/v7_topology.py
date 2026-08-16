"""Structurally distinct communication and service graph generators for V7."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .v7_types import V7TopologyDiagnostics


HUMANITARIAN_TOPOLOGIES = ("random_geometric", "small_world", "modular")
UTILITY_TOPOLOGIES = ("grid", "scale_free", "modular")


def _connected(graph: nx.Graph, rng: np.random.RandomState) -> nx.Graph:
    """Connect components with the minimum number of explicit bridge edges."""
    graph = nx.Graph(graph)
    components = [sorted(value) for value in nx.connected_components(graph)]
    while len(components) > 1:
        first = components.pop(0)
        second = components.pop(0)
        left = first[int(rng.randint(0, len(first)))]
        right = second[int(rng.randint(0, len(second)))]
        graph.add_edge(left, right, structural_bridge=True)
        components = [sorted(value) for value in nx.connected_components(graph)]
    return graph


def generate_graph(family: str, node_count: int, seed: int) -> nx.Graph:
    """Generate a topology whose structure, not just label or weights, varies."""
    if node_count < 4:
        raise ValueError("V7 graph requires at least four nodes")
    rng = np.random.RandomState(int(seed))
    if family == "ring":
        graph = nx.cycle_graph(node_count)
    elif family == "chain":
        graph = nx.path_graph(node_count)
    elif family == "grid":
        width = int(math.ceil(math.sqrt(node_count)))
        raw = nx.grid_2d_graph(width, width)
        selected = list(raw.nodes())[:node_count]
        graph = nx.convert_node_labels_to_integers(raw.subgraph(selected).copy())
        graph = _connected(graph, rng)
    elif family == "small_world":
        neighbors = min(6, node_count - 1)
        if neighbors % 2:
            neighbors -= 1
        graph = nx.watts_strogatz_graph(node_count, max(2, neighbors), 0.22, seed=int(seed))
        graph = _connected(graph, rng)
    elif family == "scale_free":
        attachment = max(1, min(3, node_count // 8))
        graph = nx.barabasi_albert_graph(node_count, attachment, seed=int(seed))
    elif family == "random_geometric":
        radius = min(0.75, max(0.28, 1.55 / math.sqrt(node_count)))
        graph = nx.random_geometric_graph(node_count, radius, seed=int(seed))
        graph = _connected(graph, rng)
    elif family == "modular":
        communities = 3 if node_count < 24 else 4
        sizes = [node_count // communities] * communities
        for index in range(node_count % communities):
            sizes[index] += 1
        probabilities = np.full((communities, communities), 0.025, dtype=float)
        np.fill_diagonal(probabilities, 0.42)
        graph = nx.stochastic_block_model(
            sizes, probabilities.tolist(), seed=int(seed), selfloops=False,
        )
        graph = _connected(graph, rng)
    else:
        raise ValueError("unknown topology family: %s" % family)
    graph = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    for node in graph.nodes:
        graph.nodes[node]["community"] = int(node * 4 // max(node_count, 1))
    for first, second in graph.edges:
        graph.edges[first, second]["reliability"] = float(rng.uniform(0.82, 0.995))
        graph.edges[first, second]["latency"] = int(rng.randint(1, 4))
        graph.edges[first, second]["trust"] = float(rng.uniform(0.70, 1.0))
        graph.edges[first, second]["available"] = True
    return graph


def graph6_digest(graph: nx.Graph) -> str:
    canonical = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    payload = nx.to_graph6_bytes(canonical, header=False)
    return hashlib.sha256(payload).hexdigest()


def structural_fingerprint(graph: nx.Graph) -> Tuple[object, ...]:
    degree_sequence = tuple(sorted((degree for _, degree in graph.degree()), reverse=True))
    triangles = tuple(sorted(nx.triangles(graph).values(), reverse=True))
    cycles = tuple(sorted(len(value) for value in nx.cycle_basis(graph)))
    return (
        graph.number_of_nodes(), graph.number_of_edges(), degree_sequence,
        triangles, cycles, len(list(nx.connected_components(graph))),
    )


def topology_diagnostics(graph: nx.Graph, family: str) -> V7TopologyDiagnostics:
    simple = nx.Graph(graph)
    count = simple.number_of_nodes()
    edge_count = simple.number_of_edges()
    components = list(nx.connected_components(simple))
    largest_nodes = max(components, key=len) if components else set()
    largest = simple.subgraph(largest_nodes).copy()
    degrees = np.asarray([degree for _, degree in simple.degree()], dtype=float)
    path = None
    diameter = None
    if len(largest) > 1:
        path = float(nx.average_shortest_path_length(largest))
        diameter = int(nx.diameter(largest))
    modularity = None
    if edge_count:
        communities = list(nx.algorithms.community.greedy_modularity_communities(simple))
        if communities:
            modularity = float(nx.algorithms.community.modularity(simple, communities))
    algebraic = None
    if nx.is_connected(simple) and count > 2:
        try:
            algebraic = float(nx.algebraic_connectivity(simple, method="tracemin_pcg"))
        except Exception:
            algebraic = None
    assortativity = None
    if edge_count > 1 and float(np.var(degrees)) > 1e-12:
        value = float(nx.degree_assortativity_coefficient(simple))
        if np.isfinite(value):
            assortativity = value
    reliability = [
        float(data.get("reliability", 1.0))
        for _, _, data in simple.edges(data=True)
    ]
    return V7TopologyDiagnostics(
        family=family,
        node_count=count,
        edge_count=edge_count,
        density=float(nx.density(simple)),
        mean_degree=float(degrees.mean()) if len(degrees) else 0.0,
        degree_variance=float(degrees.var()) if len(degrees) else 0.0,
        connected_components=len(components),
        giant_component_fraction=float(len(largest_nodes) / max(count, 1)),
        average_shortest_path=path,
        clustering_coefficient=float(nx.average_clustering(simple)) if count else 0.0,
        modularity=modularity,
        algebraic_connectivity=algebraic,
        diameter=diameter,
        assortativity=assortativity,
        edge_reliability_mean=float(np.mean(reliability)) if reliability else 0.0,
        graph6_sha256=graph6_digest(simple),
    )


def topology_families_are_distinct(graphs: Mapping[str, nx.Graph]) -> bool:
    names = sorted(graphs)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            if nx.is_isomorphic(graphs[first], graphs[second]):
                return False
    return True


def apply_partition(
    graph: nx.Graph, severity: float, rng: np.random.RandomState,
) -> List[Tuple[int, int]]:
    """Disable actual edges; callers must honor ``available`` on delivery."""
    candidates = list(graph.edges())
    rng.shuffle(candidates)
    count = int(round(float(severity) * len(candidates)))
    disabled: List[Tuple[int, int]] = []
    for first, second in candidates[:count]:
        graph.edges[first, second]["available"] = False
        disabled.append((int(first), int(second)))
    return disabled


def restore_edges(graph: nx.Graph, edges: Iterable[Tuple[int, int]]) -> None:
    for first, second in edges:
        if graph.has_edge(first, second):
            graph.edges[first, second]["available"] = True
