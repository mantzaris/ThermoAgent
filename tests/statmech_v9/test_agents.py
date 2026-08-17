import copy

import numpy as np

from thermoagent.statmech.agents import DecentralizedAgentSystem, make_independent_agents
from thermoagent.statmech.model import ModelParameters, topology_adjacency


def make_system(seed=7):
    adjacency = topology_adjacency(6, "ring", seed)
    return DecentralizedAgentSystem(make_independent_agents(6, seed), adjacency, adjacency, ModelParameters())


def test_agents_have_separate_memories_and_message_queues():
    system = make_system()
    first = system.private_agent_for_test(0)
    second = system.private_agent_for_test(1)
    first.revise_commitment("task", 1)
    system.broadcast(0)
    assert ("task", 1) in first.local_view().commitments
    assert ("task", 1) not in second.local_view().commitments
    assert second.local_view().delivered_messages
    assert not first.local_view().delivered_messages


def test_private_observation_counterfactual_changes_only_one_policy_input():
    baseline = make_system()
    counterfactual = copy.deepcopy(baseline)
    before = baseline.authorized_snapshot()
    counterfactual.private_agent_for_test(2).set_private_observation(5.0)
    after = counterfactual.authorized_snapshot()
    for identifier in range(6):
        if identifier == 2:
            assert before[identifier].private_field != after[identifier].private_field
        else:
            assert before[identifier] == after[identifier]
    baseline_decision = baseline.step(2, "belief", 0.9)
    counterfactual_decision = counterfactual.step(2, "belief", 0.9)
    assert baseline_decision != counterfactual_decision


def test_network_partition_blocks_delivery():
    system = make_system()
    system.communication[0, :] = 0.0
    system.communication[:, 0] = 0.0
    assert system.broadcast(0) == 0
    assert not system.message_ledger


def test_scheduler_does_not_change_unselected_agent_private_state():
    system = make_system()
    before = system.authorized_snapshot()
    system.step(3, "action", 0.2)
    after = system.authorized_snapshot()
    for identifier in range(6):
        if identifier != 3:
            assert before[identifier].own_belief == after[identifier].own_belief
            assert before[identifier].own_action == after[identifier].own_action
            assert before[identifier].own_memory == after[identifier].own_memory
