import networkx as nx

from thermoagent.v7_topology import (
    generate_graph, structural_fingerprint, topology_diagnostics,
    topology_families_are_distinct,
)


def test_v7_topology_families_are_structurally_distinct():
    graphs = {
        family: generate_graph(family, 18, 77100 + index)
        for index, family in enumerate(
            ("ring", "grid", "small_world", "scale_free", "modular")
        )
    }
    assert topology_families_are_distinct(graphs)
    assert len({structural_fingerprint(graph) for graph in graphs.values()}) == len(graphs)


def test_v7_topology_diagnostics_are_finite_and_data_derived():
    graph = generate_graph("modular", 28, 77109)
    diagnostics = topology_diagnostics(graph, "modular")
    assert diagnostics.node_count == 28
    assert diagnostics.edge_count == graph.number_of_edges()
    assert 0.0 < diagnostics.density < 1.0
    assert diagnostics.connected_components == 1
    assert diagnostics.giant_component_fraction == 1.0
    assert len(diagnostics.graph6_sha256) == 64
