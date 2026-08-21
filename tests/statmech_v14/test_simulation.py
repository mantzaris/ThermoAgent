import numpy as np

from thermoagent.statmech_llm_v12.provider import FunctionalProvider, KineticIsingProvider
from thermoagent.statmech_llm_v12.simulation import generate_update_tape
from thermoagent.statmech_llm_v14.simulation import (
    build_reciprocal_graph,
    make_v13_agents,
    partition_delivery_graph,
    run_v14_trajectory,
)


def _fixed(label="amber"):
    def decide(prompt, seed):
        del prompt, seed
        return {
            "belief_choice": label,
            "action_choice": label,
            "confidence": 0.75,
            "commitment_status": "provisional",
            "memory_state": "stable",
            "outgoing_signal": label,
            "tool_action": "execute_selected",
            "reason_code": "neighbor_messages",
        }

    return FunctionalProvider(decide)


def test_private_state_memory_inbox_and_outbox_are_isolated():
    agents = make_v13_agents(8, 1414, "disordered")
    peer = agents[1].private_fingerprint()
    agents[0]._memory_history.append("local-only mutation")
    assert agents[1].memory == ()
    assert agents[1].private_fingerprint() == peer
    assert agents[0]._inbox is not agents[1]._inbox
    assert agents[0]._outbox is not agents[1]._outbox


def test_scheduler_never_substitutes_agent_decision():
    graph = build_reciprocal_graph(8, "modular", 1415)
    left = run_v14_trajectory(_fixed("amber"), graph, 1416, 1, 0.8, 0.5, "nominal", [1, 1, 1])
    right = run_v14_trajectory(_fixed("cobalt"), graph, 1416, 1, 0.8, 0.5, "nominal", [1, 1, 1])
    assert left[0]["belief_after"] != right[0]["belief_after"]
    assert [row["scheduled_agent"] for row in left] == [row["scheduled_agent"] for row in right]
    assert sum(row["unrelated_peer_private_mutations"] for row in left + right) == 0


def test_random_sequential_sweep_visits_each_agent_once():
    tape = generate_update_tape(8, 24, 1417)
    for sweep in range(3):
        assert sorted(item.scheduled_agent for item in tape[sweep * 8 : (sweep + 1) * 8]) == list(range(8))


def test_quench_and_restoration_timing():
    graph = build_reciprocal_graph(8, "modular", 1418)
    rows = run_v14_trajectory(
        KineticIsingProvider(), graph, 1419, 3, 0.8, 0.5, "field_reversal", [1, 1, 1]
    )
    assert [row["phase"] for row in rows[:8]] == ["baseline"] * 8
    assert all(row["field_reversed"] == 1 for row in rows[8:16])
    assert all(row["field_reversed"] == 0 for row in rows[16:])
    for row in rows:
        base = np.asarray([int(value) for value in row["base_field_vector"].split(";")])
        active = np.asarray([int(value) for value in row["active_field_vector"].split(";")])
        assert np.array_equal(active, -base) if row["phase"] == "disruption" else np.array_equal(active, base)


def test_partition_reconnects_and_corruption_rate_is_exactly_half_of_senders():
    graph = build_reciprocal_graph(8, "modular", 1420)
    partitioned = partition_delivery_graph(graph)
    assert np.count_nonzero(partitioned.adjacency[:4, 4:]) == 0
    partition = run_v14_trajectory(
        KineticIsingProvider(), graph, 1421, 3, 0.8, 0.5, "network_partition", [1, 1, 1]
    )
    assert sum(row["cross_community_delivery"] for row in partition if row["phase"] == "disruption") == 0
    assert all(row["partition_active"] == 0 for row in partition if row["phase"] == "recovery")
    corrupt = run_v14_trajectory(
        KineticIsingProvider(), graph, 1421, 3, 0.8, 0.5, "message_corruption", [1, 1, 1]
    )
    disrupted = [row for row in corrupt if row["phase"] == "disruption"]
    assert sum(row["message_corrupted"] for row in disrupted) == 4
    assert sum(row["messages_delivered"] for row in disrupted) == 8


def test_matched_conditions_have_identical_opportunities_and_byte_schema():
    graph = build_reciprocal_graph(8, "modular", 1422)
    nominal = run_v14_trajectory(KineticIsingProvider(), graph, 1423, 3, 0.8, 0.5, "nominal", [1, 1, 1])
    field = run_v14_trajectory(KineticIsingProvider(), graph, 1423, 3, 0.8, 0.5, "field_reversal", [1, 1, 1])
    assert [row["scheduled_agent"] for row in nominal] == [row["scheduled_agent"] for row in field]
    assert sum(row["message_opportunities"] for row in nominal) == sum(row["message_opportunities"] for row in field) == 24
    assert sum(row["wire_bytes"] for row in nominal) == sum(row["wire_bytes"] for row in field)


def test_seeded_regeneration_is_exact_for_scripted_provider():
    graph = build_reciprocal_graph(8, "modular", 1424)
    first = run_v14_trajectory(KineticIsingProvider(), graph, 1425, 2, 0.8, 0.5, "nominal", [1, 1, 1])
    second = run_v14_trajectory(KineticIsingProvider(), graph, 1425, 2, 0.8, 0.5, "nominal", [1, 1, 1])
    assert first == second

