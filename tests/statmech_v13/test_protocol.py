import subprocess
from pathlib import Path

import numpy as np

from thermoagent.statmech_llm_v12.provider import FunctionalProvider
from thermoagent.statmech_llm_v13.experiment import (
    expected_decisions,
    formal_panel_design,
    graph_for_panel,
    microscopic_response_rows,
    panel_seed,
)
from thermoagent.statmech_llm_v13.workflow import execution_source_checksum, load_yaml


ROOT = Path(__file__).resolve().parents[2]
PARENT = "457f6d635b60292623c8d97aa3b0c60d8d0aac4e"


def test_formal_design_count_and_declared_work_packages():
    protocol = load_yaml(ROOT / "configs/statmech_v13/protocol_template.yaml")
    design = formal_panel_design(protocol)
    assert len(design) == 72
    assert expected_decisions(protocol) == 32672
    assert sum(int(row["n_agents"]) * int(row["sweeps"]) for row in design) == 32384
    assert {row["family"] for row in design} == {
        "A_order_fluctuation", "A_ordered_relaxation", "B_memory_quench", "C_disruption_recovery"
    }


def test_matched_arms_share_seed_graph_and_counterfactual_tape_basis():
    protocol = load_yaml(ROOT / "configs/statmech_v13/protocol_template.yaml")
    design = formal_panel_design(protocol)
    group = [row for row in design if row["cluster_id"] == "C_quench_n16_g0"]
    assert len(group) == 4
    assert len({panel_seed(row) for row in group}) == 1
    graphs = [graph_for_panel(row) for row in group]
    assert all(np.array_equal(graphs[0].weights, graph.weights) for graph in graphs[1:])


def test_memory_sign_flip_resolution_can_survive_three_hypothesis_holm_correction():
    protocol = load_yaml(ROOT / "configs/statmech_v13/protocol_template.yaml")
    clusters = int(protocol["work_package_b"]["graph_environment_clusters"])
    assert clusters == 6
    assert 3.0 * (2.0 ** (-clusters)) < 0.05


def test_total_prompt_ceiling_counts_the_pilot_before_formal_raw_records():
    protocol = load_yaml(ROOT / "configs/statmech_v13/protocol_template.yaml")
    compute = protocol["compute"]
    assert compute["maximum_prompt_tokens"] == 18000000
    assert compute["maximum_formal_raw_prompt_tokens"] + 104455 == compute["maximum_prompt_tokens"]


def test_execution_checksum_is_deterministic():
    assert execution_source_checksum(ROOT) == execution_source_checksum(ROOT)
    assert len(execution_source_checksum(ROOT)) == 64


def test_microscopic_replicates_share_cells_and_counterbalance_labels():
    protocol = load_yaml(ROOT / "configs/statmech_v13/protocol_template.yaml")

    def choose_amber(prompt, seed):
        del prompt, seed
        return {
            "belief_choice": "amber",
            "action_choice": "amber",
            "confidence": 0.5,
            "commitment_status": "provisional",
            "memory_state": "stable",
            "outgoing_signal": "amber",
            "tool_action": "execute_selected",
            "reason_code": "neighbor_messages",
        }

    rows = microscopic_response_rows(FunctionalProvider(choose_amber), protocol)
    assert len(rows) == 288
    cell_counts = {}
    for row in rows:
        cell_counts.setdefault(row["cell_id"], []).append(row)
    assert len(cell_counts) == 144
    assert all(len(group) == 2 for group in cell_counts.values())
    assert all({item["latent_plus_label"] for item in group} == {"amber", "cobalt"} for group in cell_counts.values())


def test_frozen_v1_through_v12_paths_have_no_worktree_changes():
    output = subprocess.check_output(["git", "diff", "--name-only", PARENT, "--"], cwd=ROOT, text=True)
    changed = [line for line in output.splitlines() if line]
    forbidden = [
        path for path in changed
        if any(token in path for token in ("statmech_v12", "llm_agent_statmech_v12", "jstat_v12", "notes/v12_"))
    ]
    assert forbidden == []
