import json

import numpy as np

from thermoagent.statmech_llm_v12.provider import FunctionalProvider, KineticIsingProvider
from thermoagent.statmech_llm_v13.simulation import (
    build_reciprocal_graph,
    generate_update_tape,
    make_v13_agents,
    partition_delivery_graph,
    run_v13_trajectory,
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


def test_agents_own_separate_private_memory_inboxes_and_outboxes():
    agents = make_v13_agents(8, 13, "disordered")
    fingerprints = [agent.private_fingerprint() for agent in agents]
    agents[0]._memory_history.append("private mutation")
    assert agents[1].memory == ()
    assert agents[1].private_fingerprint() == fingerprints[1]
    assert agents[0].private_fingerprint() != fingerprints[0]
    assert agents[0]._inbox is not agents[1]._inbox
    assert agents[0]._outbox is not agents[1]._outbox


def test_scheduler_records_provider_choice_without_substitution():
    graph = build_reciprocal_graph(8, "modular", 19)
    amber = run_v13_trajectory(_fixed("amber"), graph, 31, 1, "markovized", 0.8, 0.5, "disordered")
    cobalt = run_v13_trajectory(_fixed("cobalt"), graph, 31, 1, "markovized", 0.8, 0.5, "disordered")
    assert amber[0]["belief_after"] != cobalt[0]["belief_after"]
    assert amber[0]["scheduled_agent"] == cobalt[0]["scheduled_agent"]
    assert all(row["unrelated_peer_private_mutations"] == 0 for row in amber + cobalt)


def test_random_permutation_update_semantics_visit_every_agent_per_sweep():
    tape = generate_update_tape(8, 24, 99)
    for sweep in range(3):
        scheduled = [item.scheduled_agent for item in tape[sweep * 8 : (sweep + 1) * 8]]
        assert sorted(scheduled) == list(range(8))


def test_partition_removes_cross_community_delivery_and_recovers():
    graph = build_reciprocal_graph(8, "modular", 23)
    partitioned = partition_delivery_graph(graph)
    assert np.count_nonzero(partitioned.adjacency[:4, 4:]) == 0
    rows = run_v13_trajectory(KineticIsingProvider(), graph, 41, 3, "markovized", 0.8, 0.5, "disordered", "network_partition", [1, 1, 1])
    disrupted = [row for row in rows if row["phase"] == "disruption"]
    recovered = [row for row in rows if row["phase"] == "recovery"]
    assert sum(row["cross_community_delivery"] for row in disrupted) == 0
    assert all(row["active_edge_count"] < graph.adjacency.sum() // 2 for row in disrupted)
    assert all(row["partition_active"] == 0 for row in recovered)
    assert all(row["active_edge_count"] == graph.adjacency.sum() // 2 for row in recovered)


def test_field_reversal_and_exact_half_sender_corruption_restore():
    graph = build_reciprocal_graph(8, "modular", 29)
    field = run_v13_trajectory(KineticIsingProvider(), graph, 43, 3, "markovized", 0.8, 0.5, "disordered", "field_reversal", [1, 1, 1])
    for row in field:
        base = np.asarray([int(value) for value in row["base_field_vector"].split(";")])
        active = np.asarray([int(value) for value in row["active_field_vector"].split(";")])
        assert np.array_equal(active, -base) if row["phase"] == "disruption" else np.array_equal(active, base)
    corrupted = run_v13_trajectory(KineticIsingProvider(), graph, 43, 3, "markovized", 0.8, 0.5, "disordered", "message_corruption", [1, 1, 1])
    disruption = [row for row in corrupted if row["phase"] == "disruption"]
    assert sum(row["message_corrupted"] for row in disruption) == 4
    assert sum(row["messages_delivered"] for row in disruption) == 8
    assert all(row["message_corrupted"] == 0 for row in corrupted if row["phase"] != "disruption")


def test_matched_conditions_share_schedule_and_message_opportunities():
    graph = build_reciprocal_graph(8, "modular", 37)
    nominal = run_v13_trajectory(KineticIsingProvider(), graph, 47, 3, "markovized", 0.8, 0.5, "disordered", "nominal", [1, 1, 1])
    partition = run_v13_trajectory(KineticIsingProvider(), graph, 47, 3, "markovized", 0.8, 0.5, "disordered", "network_partition", [1, 1, 1])
    assert [row["scheduled_agent"] for row in nominal] == [row["scheduled_agent"] for row in partition]
    assert sum(row["message_opportunities"] for row in nominal) == sum(row["message_opportunities"] for row in partition) == 24
    assert sum(row["wire_bytes"] for row in nominal) == sum(row["wire_bytes"] for row in partition)
