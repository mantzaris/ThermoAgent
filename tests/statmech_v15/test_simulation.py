import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v12.core import LatentMapping
from thermoagent.statmech_llm_v12.provider import FunctionalProvider, KineticIsingProvider
from thermoagent.statmech_llm_v13.simulation import build_reciprocal_graph
from thermoagent.statmech_llm_v15.experiment import formal_panel_design
from thermoagent.statmech_llm_v15.analysis import (
    memory_control_balance_audit,
    memory_control_panel_audit,
)
from thermoagent.statmech_llm_v15.simulation import (
    V15MemoryNetwork,
    memory_control_tape,
    run_v15_trajectory,
)
from thermoagent.statmech_llm_v15.workflow import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def _fixed_choice(label):
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


def test_formal_design_has_six_matched_clusters_per_model_and_new_seeds():
    protocol = load_yaml(ROOT / "configs/statmech_v15/protocol_template.yaml")
    panels = formal_panel_design(protocol)
    assert len(panels) == 48
    assert {row["model_key"] for row in panels} == {"qwen", "granite"}
    for model in ("qwen", "granite"):
        selected = [row for row in panels if row["model_key"] == model]
        assert len({row["cluster_id"] for row in selected}) == 6
        assert all(len([row for row in selected if row["cluster_id"] == cluster]) == 4 for cluster in {row["cluster_id"] for row in selected})
    assert min(int(row["panel_seed"]) for row in panels) >= 15151000


def test_scheduler_does_not_substitute_decision_and_private_state_isolated():
    graph = build_reciprocal_graph(8, "modular", 15001)
    kwargs = dict(
        graph=graph,
        panel_seed=15002,
        sweeps=3,
        condition="field_markovized",
        coupling_strength=0.8,
        sampling_temperature=0.5,
        periods_sweeps=[1, 1, 1],
    )
    amber = run_v15_trajectory(_fixed_choice("amber"), **kwargs)
    cobalt = run_v15_trajectory(_fixed_choice("cobalt"), **kwargs)
    assert [row["scheduled_agent"] for row in amber] == [row["scheduled_agent"] for row in cobalt]
    assert amber[0]["belief_after"] != cobalt[0]["belief_after"]
    assert sum(row["unrelated_peer_private_mutations"] for row in amber + cobalt) == 0
    assert all(row["messages_delivered"] == row["valid_after_repair"] for row in amber + cobalt)


def test_field_quench_and_restoration_use_identical_update_tapes():
    graph = build_reciprocal_graph(8, "modular", 15003)
    provider = KineticIsingProvider()
    nominal = run_v15_trajectory(provider, graph, 15004, 3, "nominal_markovized", 0.8, 0.5, [1, 1, 1])
    field = run_v15_trajectory(KineticIsingProvider(), graph, 15004, 3, "field_markovized", 0.8, 0.5, [1, 1, 1])
    assert [row["scheduled_agent"] for row in nominal] == [row["scheduled_agent"] for row in field]
    assert [row["phase"] for row in field] == ["baseline"] * 8 + ["disruption"] * 8 + ["recovery"] * 8
    for row in field:
        base = np.asarray([int(value) for value in row["base_field_vector"].split(";")])
        active = np.asarray([int(value) for value in row["active_field_vector"].split(";")])
        expected = -base if row["phase"] == "disruption" else base
        assert np.array_equal(active, expected)
    assert sum(row["message_opportunities"] for row in nominal) == sum(row["message_opportunities"] for row in field) == 24
    assert sum(row["wire_bytes"] for row in nominal) == sum(row["wire_bytes"] for row in field)


def test_scrambled_history_is_own_agent_past_only_and_deterministic():
    mapping = LatentMapping.balanced(15005)
    first = memory_control_tape(8, 40, 15006, 15007, mapping)
    second = memory_control_tape(8, 40, 15006, 15007, mapping)
    assert first == second
    for row in first:
        assert len(row["entries"]) <= 3
        for entry in row["entries"]:
            time_value = int(str(entry).split(" ", 1)[0].split("=", 1)[1])
            assert time_value < int(row["update"])


def test_scrambled_and_persistent_have_same_memory_section_shape():
    graph = build_reciprocal_graph(8, "modular", 15008)
    persistent = run_v15_trajectory(
        KineticIsingProvider(), graph, 15009, 5, "field_persistent", 0.8, 0.5, [1, 1, 3], control_seed=15010
    )
    scrambled = run_v15_trajectory(
        KineticIsingProvider(), graph, 15009, 5, "field_scrambled", 0.8, 0.5, [1, 1, 3], control_seed=15010
    )
    assert [row["scheduled_agent"] for row in persistent] == [row["scheduled_agent"] for row in scrambled]
    assert [row["prompt_memory_entry_count"] for row in persistent] == [row["prompt_memory_entry_count"] for row in scrambled]
    assert max(row["prompt_memory_entry_count"] for row in scrambled) <= 3
    assert all(len(row["memory_control_sha256"]) == 64 for row in scrambled)


def test_agent_memory_inbox_and_outbox_are_separate_objects():
    from thermoagent.statmech_llm_v13.simulation import make_v13_agents

    agents = make_v13_agents(8, 15011, "disordered")
    graph = build_reciprocal_graph(8, "modular", 15012)
    network = V15MemoryNetwork(agents, graph, LatentMapping.balanced(15013), "persistent_memory", 0.8, 15014)
    peer = network.agents[1].private_fingerprint()
    network.agents[0]._memory_history.append("local-only")
    assert network.agents[1].private_fingerprint() == peer
    assert network.agents[0]._inbox is not network.agents[1]._inbox
    assert network.agents[0]._outbox is not network.agents[1]._outbox


def test_memory_control_audit_reconstructs_histories_without_future_state():
    graph = build_reciprocal_graph(8, "modular", 15015)
    panel_seed = 15016
    control_seed = 15017
    audits = []
    for condition, mode in (
        ("field_persistent", "persistent_memory"),
        ("field_scrambled", "scrambled_memory"),
    ):
        rows = run_v15_trajectory(
            KineticIsingProvider(),
            graph,
            panel_seed,
            5,
            condition,
            0.8,
            0.5,
            [1, 1, 3],
            control_seed=control_seed,
        )
        panel = {
            "model_key": "synthetic",
            "cluster_id": "synthetic_c0",
            "panel_id": condition,
            "condition": condition,
            "memory_mode": mode,
            "panel_seed": panel_seed,
            "control_seed": control_seed,
            "n_agents": 8,
        }
        audit = memory_control_panel_audit(pd.DataFrame(rows), panel)
        assert audit["all_entries_reconstructed"]
        assert audit["future_information_violations"] == 0
        assert not audit["donor_agent_state_used_by_construction"]
        assert not audit["peer_private_state_used_by_construction"]
        audits.append(audit)
    balance = memory_control_balance_audit(pd.DataFrame(audits))
    assert len(balance) == 1
    assert bool(balance["both_controls_fully_reconstructed"].iloc[0])
    assert int(balance["future_information_violations"].iloc[0]) == 0


def test_trajectory_state_is_fresh_and_invariant_to_prior_arm_execution():
    graph = build_reciprocal_graph(8, "modular", 15018)
    arguments = dict(
        graph=graph,
        panel_seed=15019,
        sweeps=3,
        condition="field_markovized",
        coupling_strength=0.8,
        sampling_temperature=0.5,
        periods_sweeps=[1, 1, 1],
        control_seed=15020,
    )
    first = pd.DataFrame(run_v15_trajectory(KineticIsingProvider(), **arguments))
    shared_provider = KineticIsingProvider()
    run_v15_trajectory(
        shared_provider,
        graph,
        15019,
        3,
        "field_persistent",
        0.8,
        0.5,
        [1, 1, 1],
        control_seed=15020,
    )
    second = pd.DataFrame(run_v15_trajectory(shared_provider, **arguments))
    pd.testing.assert_frame_equal(first, second, check_exact=True)
