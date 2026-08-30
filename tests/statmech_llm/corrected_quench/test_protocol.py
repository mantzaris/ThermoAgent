from pathlib import Path

import numpy as np

from thermoagent.statmech_llm.corrected_quench.experiment import expected_decisions, formal_panel_design, graph_for_panel, panel_seed
from thermoagent.statmech_llm.corrected_quench.workflow import execution_source_checksum, load_yaml


ROOT = Path(__file__).resolve().parents[3]
def test_formal_panel_count_and_decision_accounting_are_frozen():
    protocol = load_yaml(ROOT / "configs/statmech_llm/corrected_quench/protocol_template.yaml")
    design = formal_panel_design(protocol)
    assert len(design) == 24
    assert expected_decisions(protocol) == 17280
    assert expected_decisions(protocol) == protocol["compute"]["expected_formal_decisions"]
    assert {panel["disruption"] for panel in design} == {
        "nominal", "field_reversal", "network_partition", "message_corruption"
    }


def test_matched_arms_share_seed_graph_and_random_tape_basis():
    protocol = load_yaml(ROOT / "configs/statmech_llm/corrected_quench/protocol_template.yaml")
    group = [panel for panel in formal_panel_design(protocol) if panel["cluster_id"] == "V14Q_g0"]
    assert len(group) == 4 and len({panel_seed(panel) for panel in group}) == 1
    graphs = [graph_for_panel(panel) for panel in group]
    assert all(np.array_equal(graphs[0].weights, graph.weights) for graph in graphs[1:])


def test_six_clusters_allow_three_hypothesis_holm_resolution():
    protocol = load_yaml(ROOT / "configs/statmech_llm/corrected_quench/protocol_template.yaml")
    clusters = int(protocol["network"]["graph_environment_clusters"])
    assert clusters == 6
    assert 3.0 * 2.0 ** (-clusters) < 0.05


def test_execution_source_checksum_is_deterministic():
    assert execution_source_checksum(ROOT) == execution_source_checksum(ROOT)
    assert len(execution_source_checksum(ROOT)) == 64


def test_corrected_quench_uses_semantic_namespace():
    assert (ROOT / "thermoagent/statmech_llm/corrected_quench").is_dir()
    assert (ROOT / "configs/statmech_llm/corrected_quench/protocol.yaml").is_file()
