import copy

import pytest

from thermoagent.statmech_llm_v12.core import (
    AgentDecision,
    IndependentStatmechAgent,
    LatentMapping,
    SignalPacket,
    build_agent_prompt,
    decode_microstate,
    deserialize_signal_packet,
    encode_microstate,
    serialize_signal_packet,
)


def decision(label="amber"):
    return AgentDecision(
        belief_choice=label,
        action_choice=label,
        confidence=0.7,
        commitment_status="provisional",
        memory_state="stable",
        outgoing_signal=label,
        tool_action="execute_selected",
        reason_code="private_observation",
    )


def test_latent_mapping_round_trip_and_counterbalancing():
    amber = LatentMapping("amber", ("amber", "cobalt"))
    cobalt = LatentMapping("cobalt", ("cobalt", "amber"))
    for mapping in (amber, cobalt):
        assert mapping.spin(mapping.label(-1)) == -1
        assert mapping.spin(mapping.label(1)) == 1
    assert amber.spin("amber") == -cobalt.spin("amber")


def test_signal_packet_real_wire_round_trip_and_corruption_detection():
    packet = SignalPacket(7, 31, -1, 1, 0, 0.625, 2, 3)
    encoded = serialize_signal_packet(packet)
    assert encoded == serialize_signal_packet(packet)
    restored = deserialize_signal_packet(encoded)
    assert restored.sender_id == packet.sender_id
    assert restored.belief_spin == -1
    assert restored.action_spin == 1
    assert restored.signal_spin == 0
    assert restored.confidence == pytest.approx(packet.confidence, abs=1 / 65535)
    corrupted = encoded[:-1] + bytes([encoded[-1] ^ 1])
    with pytest.raises(ValueError, match="checksum"):
        deserialize_signal_packet(corrupted)


def test_agent_private_memory_inbox_and_peer_copy_are_isolated():
    first = IndependentStatmechAgent(0, "observer", 1, -1, 1)
    second = IndependentStatmechAgent(1, "operator", -1, 1, -1)
    peer_before = second.private_fingerprint()
    first.receive(SignalPacket(1, 0, 1, -1, 1, 0.8, 1, 0))
    first.apply_decision(decision("amber"), LatentMapping("amber", ("amber", "cobalt")), 1, True)
    assert second.private_fingerprint() == peer_before
    clone = first.clone()
    clone.receive(SignalPacket(2, 1, -1, -1, -1, 0.6, 0, 4))
    assert clone.inbox != first.inbox
    assert clone.memory == first.memory


def test_markovized_prompt_is_state_complete_without_hidden_global_state():
    agent = IndependentStatmechAgent(0, "observer", 1, -1, 1)
    mapping = LatentMapping("cobalt", ("amber", "cobalt"))
    prompt = build_agent_prompt(agent, mapping, 4, "markovized", 0.7, 0)
    assert "current_belief" in prompt
    assert "current_action" in prompt
    assert "current_local_workload" in prompt
    assert "bounded_private_memory" not in prompt
    assert "hidden truth" in prompt
    assert "global_network_state" not in prompt
    assert '"private_field_level"' not in prompt


def test_microstate_serialization_is_bijective_for_primary_projection():
    agents = [
        IndependentStatmechAgent(0, "a", 1, -1, 1),
        IndependentStatmechAgent(1, "b", -1, 1, -1),
        IndependentStatmechAgent(2, "c", 0, 1, 1),
    ]
    encoded = encode_microstate(agents)
    beliefs, actions = decode_microstate(encoded, 3)
    assert beliefs.tolist() == [-1, 1, 1]
    assert actions.tolist() == [1, -1, 1]
    with pytest.raises(ValueError):
        decode_microstate(2 ** 6, 3)


def test_decision_validation_separates_action_and_tool():
    bad = copy.copy(decision("amber"))
    object.__setattr__(bad, "tool_action", "unknown_tool")
    with pytest.raises(ValueError, match="typed tool"):
        bad.validate()
