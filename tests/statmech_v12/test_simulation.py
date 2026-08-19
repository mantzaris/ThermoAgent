import json

from thermoagent.statmech_llm_v12.core import LatentMapping
from thermoagent.statmech_llm_v12.graphs import build_delivery_graph
from thermoagent.statmech_llm_v12.provider import FunctionalProvider, InvalidStructuredDecision
from thermoagent.statmech_llm_v12.simulation import DecentralizedStatmechNetwork, generate_update_tape, make_agents, run_trajectory


def _visible_choice(prompt, seed):
    del seed
    view = json.loads(prompt.split("LOCAL_UPDATE=", 1)[1])["authorized_local_state"]
    label = "amber" if "amber" in str(view["private_observation"]) else "cobalt"
    if "balanced" in str(view["private_observation"]):
        label = str(view["current_belief"])
    return {
        "belief_choice": label,
        "action_choice": label,
        "confidence": 0.75,
        "commitment_status": "provisional",
        "memory_state": "stable",
        "outgoing_signal": label,
        "tool_action": "execute_selected",
        "reason_code": "private_observation",
    }


def test_scheduler_applies_provider_choice_without_substitution_and_preserves_privacy():
    provider = FunctionalProvider(_visible_choice)
    graph = build_delivery_graph(4, "ring", 1, 2, 0.5)
    rows = run_trajectory(
        provider,
        graph,
        100,
        2,
        "markovized",
        0.7,
        0.8,
        "disordered",
        mapping_override=LatentMapping("amber", ("amber", "cobalt")),
    )
    assert len(rows) == 8
    assert all(row["valid_after_repair"] == 1 for row in rows)
    assert sum(row["unrelated_peer_private_mutations"] for row in rows) == 0
    assert all(row["messages_transmitted"] == row["messages_delivered"] == 1 for row in rows)
    assert all(row["wire_bytes"] > 0 for row in rows)
    assert any(row["workload_change"] != 0 for row in rows)


def test_no_message_control_keeps_opportunities_but_blocks_wire_delivery():
    rows = run_trajectory(
        FunctionalProvider(_visible_choice),
        build_delivery_graph(4, "ring", 1, 2, 0.8),
        101,
        1,
        "markovized",
        0.7,
        0.8,
        "disordered",
        control="no_message",
    )
    assert sum(row["message_opportunities"] for row in rows) == 4
    assert sum(row["messages_transmitted"] for row in rows) == 0
    assert sum(row["messages_dropped"] for row in rows) == 4
    assert sum(row["wire_bytes"] for row in rows) == 0


def test_update_tape_is_seed_deterministic():
    assert generate_update_tape(4, 12, 7) == generate_update_tape(4, 12, 7)


class _InvalidProvider:
    def decide(self, prompt, seed, sampling_temperature=None):
        del prompt, seed, sampling_temperature
        raise InvalidStructuredDecision("deliberate invalid test response")


def test_invalid_provider_decision_is_not_replaced_by_scheduler():
    graph = build_delivery_graph(4, "ring", 1, 2, 0.0)
    mapping = LatentMapping("amber", ("amber", "cobalt"))
    network = DecentralizedStatmechNetwork(make_agents(4, 22, "disordered"), graph, mapping, "markovized", 0.7)
    before = [(agent.belief, agent.action) for agent in network.agents]
    row = network.offered_update(_InvalidProvider(), generate_update_tape(4, 1, 11)[0], 0, 0.7)
    after = [(agent.belief, agent.action) for agent in network.agents]
    assert before == after
    assert row["valid_after_repair"] == 0
    assert row["messages_transmitted"] == row["messages_delivered"] == 0


def _neighbor_choice(prompt, seed):
    del seed
    view = json.loads(prompt.split("LOCAL_UPDATE=", 1)[1])["authorized_local_state"]
    messages = view["delivered_neighbor_packets"]
    label = messages[-1]["signal"] if messages and messages[-1]["signal"] != "uncertain" else view["current_belief"]
    return {
        "belief_choice": label,
        "action_choice": label,
        "confidence": 0.7,
        "commitment_status": "revised",
        "memory_state": "stable",
        "outgoing_signal": label,
        "tool_action": "execute_selected",
        "reason_code": "neighbor_messages" if messages else "persistence",
    }


def test_delivered_messages_can_causally_change_later_agent_state():
    graph = build_delivery_graph(4, "ring", 5, 8, 0.5)
    altered = run_trajectory(FunctionalProvider(_neighbor_choice), graph, 515, 3, "markovized", 0.7, 0.8, "disordered")
    blocked = run_trajectory(
        FunctionalProvider(_neighbor_choice), graph, 515, 3, "markovized", 0.7, 0.8, "disordered", control="no_message"
    )
    assert any(a["state_after"] != b["state_after"] for a, b in zip(altered, blocked))
    assert sum(row["inbox_packets"] for row in altered) > 0
