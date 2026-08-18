from pathlib import Path

import numpy as np

from thermoagent.statmech_llm_v11.core import EvidenceGroundedDecision, ProviderResult
from thermoagent.statmech_llm_v11.baselines import ScriptedBayesianProvider
from thermoagent.statmech_llm_v11.network import (
    DecentralizedEvidenceNetwork,
    StepTape,
    choose_recipient,
    make_network_agents,
    oriented_edge_signs,
    undirected_skeleton,
)


class FixedProvider:
    def __init__(self, probability):
        self.probability = probability

    def decide(self, prompt, seed):
        payload = {
            "probability_right": self.probability,
            "belief_choice": "right" if self.probability >= 0.5 else "left",
            "action_choice": "select_right" if self.probability >= 0.5 else "select_left",
            "commitment_status": "provisional",
            "outgoing_evidence_action": "abstain",
            "reason_code": "combined_evidence",
            "explanation": "fixed test output",
        }
        return ProviderResult(payload, True, False, 1, 1, 0.01, "x")


class InvalidProvider:
    def decide(self, prompt, seed):
        raise ValueError("invalid after bounded repair")


def test_topology_and_orientation_are_structural():
    ring = undirected_skeleton(8, "ring", 1)
    modular = undirected_skeleton(8, "modular", 1)
    assert np.all(ring == ring.T)
    assert not np.array_equal(ring, modular)
    orientation = oriented_edge_signs(ring, 4)
    assert np.array_equal(orientation, -orientation.T)


def test_nonreciprocity_changes_recipient_without_changing_support():
    graph = undirected_skeleton(6, "ring", 1)
    orientation = oriented_edge_signs(graph, 2)
    values_0 = [choose_recipient(0, graph, orientation, 0.0, u) for u in np.linspace(0.01, 0.99, 100)]
    values_a = [choose_recipient(0, graph, orientation, 0.8, u) for u in np.linspace(0.01, 0.99, 100)]
    assert set(values_0) == set(values_a) == set(np.flatnonzero(graph[0]))
    assert values_0 != values_a


def test_scheduler_does_not_choose_action_and_peer_state_is_isolated():
    graph = undirected_skeleton(4, "ring", 1)
    orientation = oriented_edge_signs(graph, 3)
    agents = make_network_agents(4, "route_viability", "right", 0.75, 5)
    network = DecentralizedEvidenceNetwork(agents, graph, orientation, "right")
    before = network.private_fingerprints()
    row = network.offered_step(
        FixedProvider(0.9), StepTape(0, 0.1, 0.3, 0.2, 9, False, 0), 0.0, 0.75, "route_viability", 0
    )
    assert row["action_choice"] == "select_right"
    assert row["causal_service_change"] < 0
    after = network.private_fingerprints()
    assert before[1:] == after[1:]
    assert row["unrelated_peer_private_mutations"] == 0


def test_invalid_provider_does_not_mutate_or_receive_substitute_action():
    graph = undirected_skeleton(4, "ring", 1)
    orientation = oriented_edge_signs(graph, 3)
    network = DecentralizedEvidenceNetwork(
        make_network_agents(4, "route_viability", "right", 0.75, 5), graph, orientation, "right"
    )
    before = network.private_fingerprints()
    decision_state_before = (
        network.agents[0].probability_right,
        network.agents[0].action,
        network.agents[0].commitment,
        network.agents[0].memory,
    )
    try:
        network.offered_step(
            InvalidProvider(), StepTape(0, 0.1, 0.3, 0.2, 9, False, 0), 0.0, 0.75, "route_viability", 0
        )
    except ValueError:
        pass
    assert network.private_fingerprints()[1:] == before[1:]
    assert (
        network.agents[0].probability_right,
        network.agents[0].action,
        network.agents[0].commitment,
        network.agents[0].memory,
    ) == decision_state_before


def test_matched_tape_is_deterministic():
    from thermoagent.statmech_llm_v11.network import generate_step_tape

    assert generate_step_tape(4, 20, 12) == generate_step_tape(4, 20, 12)


def test_delivered_packet_can_change_later_agent_action_and_service():
    graph = undirected_skeleton(4, "ring", 1)
    orientation = oriented_edge_signs(graph, 2)
    agents = make_network_agents(4, "route_viability", "right", 0.75, 6)
    with_message = DecentralizedEvidenceNetwork(agents, graph, orientation, "right")
    without_message = DecentralizedEvidenceNetwork(agents, graph, orientation, "right")
    sender_turn = StepTape(0, 0.1, 0.9, 0.01, 10, False, 0)
    sent = with_message.offered_step(
        ScriptedBayesianProvider(), sender_turn, 0.0, 0.85, "route_viability", 0
    )
    without_message.offered_step(
        FixedProvider(0.9), sender_turn, 0.0, 0.85, "route_viability", 0, control="no_message"
    )
    recipient = int(sent["recipient"])
    assert with_message.agents[recipient].inbox_size == 1
    recipient_turn = StepTape(recipient, 0.99, 0.1, 0.3, 11, False, 0)
    treated = with_message.offered_step(
        ScriptedBayesianProvider(), recipient_turn, 0.0, 0.55, "route_viability", 1
    )
    control = without_message.offered_step(
        ScriptedBayesianProvider(), recipient_turn, 0.0, 0.55, "route_viability", 1, control="no_message"
    )
    assert treated["inbox_count_before"] == 1
    assert control["inbox_count_before"] == 0
    assert (treated["probability_right"], treated["action_choice"], treated["service_after"]) != (
        control["probability_right"], control["action_choice"], control["service_after"]
    )


def test_formal_design_preserves_matched_alpha_panels():
    from thermoagent.statmech_llm_v11.formal import formal_panel_design

    settings = {
        "agent_counts": [4],
        "topologies": ["ring"],
        "orientation_seeds": [1],
        "environment_seeds": [2],
        "nonreciprocity_levels": [0.0, 0.5],
        "trajectory_turns": 8,
        "control_design": {
            "n_agents": 4,
            "topology": "ring",
            "alpha": 0.5,
            "turns": 4,
            "orientation_seeds": [1],
            "environment_seeds": [2],
            "controls": ["unaltered", "no_message"],
        },
    }
    design = formal_panel_design(settings)
    primary = [row for row in design if row["panel_family"] == "primary"]
    assert len(primary) == 4  # two applications, two matched alpha arms
    assert len({row["matched_cluster"] for row in primary}) == 2


def test_frozen_formal_template_has_declared_call_count_and_distinct_graphs():
    import yaml

    from thermoagent.statmech_llm_v11.formal import expected_formal_decisions

    root = Path(__file__).resolve().parents[2]
    settings = yaml.safe_load((root / "configs/statmech_v11/formal_template.yaml").read_text())["formal"]
    assert expected_formal_decisions(settings) == 20992
    for size in settings["agent_counts"]:
        assert not np.array_equal(undirected_skeleton(size, "ring", 4), undirected_skeleton(size, "modular", 4))
