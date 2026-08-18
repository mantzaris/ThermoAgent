import json
from pathlib import Path

import numpy as np
import pytest

from thermoagent.statmech_llm_v11.core import (
    EvidenceGroundedDecision,
    EvidencePacket,
    IndependentEvidenceAgent,
    bayesian_probability_right,
    build_evidence_prompt,
    deserialize_evidence_packet,
    generate_private_evidence,
    serialize_evidence_packet,
)


def packet(observation="right", reliability=0.75, source="source"):
    return EvidencePacket(source, observation, reliability, 0, 0, 1.0, "route_viability", "local observation")


def decision(probability=0.8, action="select_right"):
    return EvidenceGroundedDecision(
        probability_right=probability,
        belief_choice="right" if probability >= 0.5 else "left",
        action_choice=action,
        commitment_status="provisional",
        outgoing_evidence_action="abstain",
        reason_code="combined_evidence",
        explanation="bounded",
    )


def test_wire_round_trip_and_exact_length_is_deterministic():
    payload = serialize_evidence_packet(packet())
    assert payload == serialize_evidence_packet(packet())
    assert deserialize_evidence_packet(payload) == packet()
    assert len(payload) > len("source")
    with pytest.raises(ValueError):
        deserialize_evidence_packet(payload[:-1])
    corrupted = bytearray(payload)
    corrupted[0] = 99
    with pytest.raises(ValueError):
        deserialize_evidence_packet(bytes(corrupted))


def test_bayesian_reference_uses_reliability_as_likelihood():
    assert np.isclose(bayesian_probability_right([packet("right", 0.75)]), 0.75)
    assert np.isclose(bayesian_probability_right([packet("left", 0.75)]), 0.25)
    assert np.isclose(bayesian_probability_right([packet("right", 0.75), packet("left", 0.75)]), 0.5)
    assert bayesian_probability_right([packet("right", 0.85)]) > bayesian_probability_right([packet("right", 0.65)])


def test_staleness_contracts_evidence_toward_no_information():
    fresh = EvidencePacket("peer", "right", 0.85, 0, 0, 1.0, "route_viability")
    stale = EvidencePacket("peer", "right", 0.85, 0, 4, 0.25, "route_viability")
    assert 0.5 < bayesian_probability_right([stale]) < bayesian_probability_right([fresh])


def test_generated_signal_frequency_matches_probability():
    rng = np.random.default_rng(17)
    values = [generate_private_evidence("right", 0.75, rng, "a", "route_viability").observation for _ in range(20000)]
    assert abs(np.mean(np.asarray(values) == "right") - 0.75) < 0.015


def test_probability_bounds_and_frozen_binary_threshold():
    decision(0.5).validate()
    with pytest.raises(ValueError):
        EvidenceGroundedDecision(
            0.7, "left", "defer", "uncommitted", "abstain", "insufficient_evidence", "bad"
        ).validate()
    with pytest.raises(ValueError):
        decision(1.2).validate()


def test_qualification_prompt_does_not_expose_prior_action_commitment_or_memory():
    agent = IndependentEvidenceAgent(3, "coordinator", packet())
    agent.apply_decision(decision(), 0)
    prompt = build_evidence_prompt(agent, "qualification_unanchored", ("left", "right"), 0, 1)
    envelope = json.loads(prompt.split("CONTROLLED_TASK=", 1)[1])
    view = envelope["authorized_local_view"]
    assert "previous_probability_right" not in view
    assert "previous_action" not in view
    assert "previous_commitment" not in view
    assert "bounded_memory" not in view
    assert "typed tool" in prompt
    assert "latent_state" not in prompt
    assert "counterfactual_outcome" not in prompt


def test_private_state_and_memory_are_isolated():
    first = IndependentEvidenceAgent(1, "one", packet(source="one"))
    second = IndependentEvidenceAgent(2, "two", packet("left", source="two"))
    second_before = second.private_fingerprint()
    first.receive(packet(source="peer"))
    first.apply_decision(decision(), 1)
    assert second.private_fingerprint() == second_before
    assert first.memory != second.memory


def test_agent_selected_message_tool_sends_only_immutable_private_evidence():
    private = packet(source="private_agent")
    agent = IndependentEvidenceAgent(1, "one", private)
    value = EvidenceGroundedDecision(
        0.2, "left", "select_left", "provisional", "send_private_evidence", "private_evidence", "x"
    )
    assert agent.apply_decision(value, 0)
    assert agent.private_evidence == private
