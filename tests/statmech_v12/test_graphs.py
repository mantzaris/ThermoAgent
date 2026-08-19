import numpy as np

from thermoagent.statmech_llm_v12.graphs import (
    base_adjacency,
    build_delivery_graph,
    matched_opportunity_schedule,
    select_recipient,
)


def test_reciprocal_reference_and_nonreciprocity_are_matched():
    reciprocal = build_delivery_graph(8, "ring", 3, 7, 0.0)
    directed = build_delivery_graph(8, "ring", 3, 7, 0.8)
    reverse = directed.reverse()
    assert np.array_equal(reciprocal.adjacency, directed.adjacency)
    assert np.allclose(reciprocal.weights, reciprocal.weights.T)
    assert np.allclose(directed.weights.T, reverse.weights)
    assert np.allclose(reciprocal.weights.sum(axis=1), directed.weights.sum(axis=1))
    assert np.allclose(directed.weights.sum(axis=0), directed.weights.sum(axis=1))
    assert directed.diagnostics()["asymmetry_frobenius"] > 0.0
    assert reciprocal.diagnostics()["asymmetry_frobenius"] == 0.0
    assert reciprocal.diagnostics()["circulation_l1"] == 0.0
    assert directed.diagnostics()["circulation_l1"] > 0.0


def test_asymmetry_is_monotone_without_density_change():
    values = [build_delivery_graph(16, "modular", 13, 19, alpha) for alpha in (0.0, 0.2, 0.5, 0.8)]
    asymmetry = [item.diagnostics()["asymmetry_frobenius"] for item in values]
    assert asymmetry == sorted(asymmetry)
    assert all(item.diagnostics()["edge_count"] == values[0].diagnostics()["edge_count"] for item in values)


def test_topologies_are_structurally_distinct_and_modular_degree_is_fixed():
    ring = base_adjacency(16, "ring", 1)
    modular = base_adjacency(16, "modular", 1)
    assert not np.array_equal(ring, modular)
    assert np.all(ring.sum(axis=1) == 2)
    assert np.all(modular.sum(axis=1) == 3)


def test_random_sequential_schedule_attempts_every_agent_once_per_sweep():
    scheduled, uniforms = matched_opportunity_schedule(8, 24, 99)
    assert uniforms.shape == (24,)
    for start in range(0, 24, 8):
        assert sorted(scheduled[start : start + 8].tolist()) == list(range(8))


def test_recipient_selection_uses_only_supported_edges():
    graph = build_delivery_graph(8, "ring", 1, 2, 0.8)
    for sender in range(8):
        for uniform in (0.0, 0.25, 0.75, 0.999999):
            recipient = select_recipient(graph.weights, sender, uniform)
            assert graph.adjacency[sender, recipient] == 1
