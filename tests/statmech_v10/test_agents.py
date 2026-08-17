import copy
import json

import numpy as np
import pytest

from thermoagent.statmech_llm.agents import (
    DecentralizedLLMNetwork,
    DeliveredMessage,
    FunctionalProvider,
    StructuredDecision,
    build_prompt,
    deserialize_delivered_message,
    make_agents,
    serialize_delivered_message,
)
from thermoagent.statmech_llm.applications import DefensiveUtilityMapping, HumanitarianCoordinationMapping


def payload(side="plan_right", signal="none", tool="commit_plan_right"):
    return {
        "belief_choice": side,
        "belief_confidence": 0.76,
        "action_choice": side,
        "commitment_status": "revise",
        "outgoing_signal": signal,
        "outgoing_message": "local evidence favors this plan",
        "tool_action": tool,
        "reason_code": "private_evidence",
    }


def test_strict_structured_schema_rejects_missing_extra_and_invalid_fields():
    StructuredDecision.from_mapping(payload())
    missing = payload()
    missing.pop("reason_code")
    with pytest.raises(ValueError):
        StructuredDecision.from_mapping(missing)
    extra = payload()
    extra["global_truth"] = 1
    with pytest.raises(ValueError):
        StructuredDecision.from_mapping(extra)
    invalid = payload()
    invalid["belief_choice"] = "A"
    with pytest.raises(ValueError):
        StructuredDecision.from_mapping(invalid)


def test_prompt_contains_only_authorized_agent_view():
    agents = make_agents(3, 4)
    network = DecentralizedLLMNetwork(agents, np.ones((3, 3)) - np.eye(3))
    view = agents[0].authorized_view(network.recipients(0), 0, "controlled_micro_update_belief")
    prompt = build_prompt(view, ("plan_left", "plan_right"), 0)
    assert "global truth" in prompt.lower()
    assert "private_observation" in prompt
    assert "influence_weight is an authorized reliability coefficient" in prompt
    assert agents[1].private_observation not in prompt
    assert agents[2].private_observation not in prompt


def test_counterfactual_private_evidence_can_change_only_target_decision_state():
    adjacency = np.ones((3, 3)) - np.eye(3)
    base_agents = make_agents(3, 5)
    positive = DecentralizedLLMNetwork(copy.deepcopy(base_agents), adjacency)
    negative = DecentralizedLLMNetwork(copy.deepcopy(base_agents), adjacency)
    positive.private_agent_for_test(0).set_private_observation("private sign POSITIVE")
    negative.private_agent_for_test(0).set_private_observation("private sign NEGATIVE")

    def respond(prompt, seed):
        return payload("plan_right" if "POSITIVE" in prompt else "plan_left")

    before_positive = positive.fingerprints()
    before_negative = negative.fingerprints()
    positive.offered_update(0, FunctionalProvider(respond), 1, "belief", ("plan_left", "plan_right"), 0)
    negative.offered_update(0, FunctionalProvider(respond), 1, "belief", ("plan_left", "plan_right"), 0)
    assert positive.private_agent_for_test(0)._belief == 1
    assert negative.private_agent_for_test(0)._belief == -1
    for peer in (1, 2):
        assert before_positive[peer] == positive.fingerprints()[peer]
        assert before_negative[peer] == negative.fingerprints()[peer]


def test_scheduler_offers_update_but_does_not_substitute_provider_action():
    network = DecentralizedLLMNetwork(make_agents(2, 6), np.ones((2, 2)) - np.eye(2))
    decision = network.offered_update(
        0,
        FunctionalProvider(lambda prompt, seed: payload("plan_left")),
        3,
        "action",
        ("plan_right", "plan_left"),
        1,
    )
    assert decision.action_spin == -1
    assert network.private_agent_for_test(0)._action == -1


def test_directed_delivery_obeys_edges_and_inboxes_remain_separate():
    # A[recipient, sender]: agent 0 may send to 1 but not 2.
    adjacency = np.zeros((3, 3))
    adjacency[1, 0] = 1.0
    network = DecentralizedLLMNetwork(make_agents(3, 7), adjacency)
    network.offered_update(
        0,
        FunctionalProvider(lambda prompt, seed: payload(signal="support_right")),
        9,
        None,
        ("plan_left", "plan_right"),
        0,
    )
    assert len(network.private_agent_for_test(1)._inbox) == 1
    assert len(network.private_agent_for_test(2)._inbox) == 0


def test_controlled_micro_update_retains_unscheduled_variable():
    network = DecentralizedLLMNetwork(make_agents(2, 8), np.zeros((2, 2)))
    agent = network.private_agent_for_test(0)
    prior_action = agent._action
    network.offered_update(
        0,
        FunctionalProvider(lambda prompt, seed: payload("plan_right" if prior_action < 0 else "plan_left")),
        10,
        "belief",
        ("plan_left", "plan_right"),
        0,
    )
    assert agent._action == prior_action


def test_typed_actions_have_consequential_application_transitions():
    humanitarian = HumanitarianCoordinationMapping()
    utility = DefensiveUtilityMapping()
    decision = StructuredDecision.from_mapping(payload("plan_right", tool="commit_plan_right"))
    humanitarian_effect = humanitarian.apply(decision)
    utility_effect = utility.apply(decision)
    assert humanitarian_effect["causal_service_change"] < 0.0
    assert utility_effect["causal_service_change"] != 0.0


def test_message_wire_round_trip_and_byte_accounting():
    message = DeliveredMessage(1, 2, 7, "conflict", "evidence differs", 1.35)
    encoded = serialize_delivered_message(message)
    decoded = deserialize_delivered_message(encoded)
    assert decoded.sender == message.sender
    assert decoded.recipient == message.recipient
    assert decoded.time_step == message.time_step
    assert decoded.outgoing_signal == message.outgoing_signal
    assert decoded.outgoing_message == message.outgoing_message
    assert np.isclose(decoded.influence_weight, message.influence_weight, atol=1e-6)


def test_message_opportunity_and_wire_bytes_match_across_reciprocity_conditions():
    base = np.ones((3, 3)) - np.eye(3)
    directed = base.copy()
    directed[0, 1], directed[1, 0] = 1.5, 0.5
    networks = [
        DecentralizedLLMNetwork(make_agents(3, 90), matrix)
        for matrix in (base, directed)
    ]
    for network in networks:
        network.offered_update(
            0,
            FunctionalProvider(lambda prompt, seed: payload(signal="support_right")),
            1,
            None,
            ("plan_left", "plan_right"),
            0,
        )
    assert len(networks[0].message_ledger) == len(networks[1].message_ledger) == 2
    assert networks[0].message_wire_bytes == networks[1].message_wire_bytes
