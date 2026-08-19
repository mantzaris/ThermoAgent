"""Random-sequential decentralized LLM-agent network dynamics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .core import (
    AgentDecision,
    COMMITMENTS,
    IndependentStatmechAgent,
    LatentMapping,
    MEMORY_STATES,
    SignalPacket,
    StructuredProvider,
    build_agent_prompt,
    encode_microstate,
    serialize_signal_packet,
)
from .estimators import network_observables
from .graphs import DeliveryGraph, matched_opportunity_schedule, select_recipient
from .provider import InvalidStructuredDecision


CONTROLS = (
    "unaltered",
    "no_message",
    "content_permutation",
    "temporal_permutation",
    "sender_identity_permutation",
    "placebo",
)


@dataclass(frozen=True)
class UpdateTape:
    scheduled_agent: int
    recipient_uniform: float
    inference_seed: int
    reverse_display_order: bool
    paraphrase: int


def generate_update_tape(n_agents: int, updates: int, seed: int) -> List[UpdateTape]:
    scheduled, uniforms = matched_opportunity_schedule(int(n_agents), int(updates), int(seed))
    rng = np.random.default_rng(int(seed) + 7919)
    return [
        UpdateTape(
            scheduled_agent=int(scheduled[index]),
            recipient_uniform=float(uniforms[index]),
            inference_seed=(int(seed) * 100000 + index) % (2 ** 32 - 1),
            reverse_display_order=bool(rng.integers(0, 2)),
            paraphrase=int(rng.integers(0, 2)),
        )
        for index in range(int(updates))
    ]


def make_agents(n_agents: int, seed: int, initial_condition: str) -> List[IndependentStatmechAgent]:
    n = int(n_agents)
    if initial_condition not in ("ordered", "disordered"):
        raise ValueError("initial condition must be ordered or disordered")
    rng = np.random.default_rng(int(seed))
    fields = np.asarray([1 if index % 2 == 0 else -1 for index in range(n)], dtype=int)
    if n % 2:
        fields[-1] = 0
    rng.shuffle(fields)
    if initial_condition == "ordered":
        sign = 1 if int(seed) % 2 == 0 else -1
        beliefs = np.full(n, sign, dtype=int)
        actions = np.full(n, sign, dtype=int)
    else:
        beliefs = np.asarray([1 if index % 2 == 0 else -1 for index in range(n)], dtype=int)
        actions = -beliefs.copy()
        rng.shuffle(beliefs)
        rng.shuffle(actions)
    roles = ("local_observer", "coordination_agent", "task_operator", "safety_monitor")
    return [
        IndependentStatmechAgent(
            identifier=index,
            role=roles[index % len(roles)],
            private_field=int(fields[index]),
            _belief=int(beliefs[index]),
            _action=int(actions[index]),
        )
        for index in range(n)
    ]


def _transform_packet(
    packet: SignalPacket,
    control: str,
    history: Sequence[SignalPacket],
    uniform: float,
    n_agents: int,
) -> SignalPacket:
    if control not in CONTROLS:
        raise ValueError("unknown communication control")
    if control == "unaltered" or control == "no_message":
        return packet
    if control == "placebo":
        return SignalPacket(
            packet.sender_id,
            packet.time_step,
            packet.belief_spin,
            packet.action_spin,
            0,
            0.5,
            0,
            4,
        )
    if control == "sender_identity_permutation":
        return SignalPacket(
            (packet.sender_id + 1) % int(n_agents),
            packet.time_step,
            packet.belief_spin,
            packet.action_spin,
            packet.signal_spin,
            packet.confidence,
            packet.commitment_code,
            packet.memory_code,
        )
    if not history:
        return SignalPacket(
            packet.sender_id,
            packet.time_step,
            packet.belief_spin,
            packet.action_spin,
            0,
            0.5,
            0,
            4,
        )
    if control == "content_permutation":
        index = min(int(float(uniform) * len(history)), len(history) - 1)
        old = history[index]
        return SignalPacket(
            packet.sender_id,
            packet.time_step,
            old.belief_spin,
            old.action_spin,
            old.signal_spin,
            old.confidence,
            old.commitment_code,
            old.memory_code,
        )
    # Temporal permutation keeps an old timestamp and content, exposing age.
    old = history[0]
    return SignalPacket(
        packet.sender_id,
        old.time_step,
        old.belief_spin,
        old.action_spin,
        old.signal_spin,
        old.confidence,
        old.commitment_code,
        old.memory_code,
    )


class DecentralizedStatmechNetwork:
    """Environment scheduler plus isolated agent-owned local decisions."""

    def __init__(
        self,
        agents: Sequence[IndependentStatmechAgent],
        graph: DeliveryGraph,
        mapping: LatentMapping,
        regime: str,
        coupling_strength: float,
        control: str = "unaltered",
        j_b: float = 1.0,
        j_a: float = 0.65,
        belief_action_k: float = 0.8,
    ) -> None:
        self.agents = [agent.clone() for agent in agents]
        self.graph = graph
        self.mapping = mapping
        self.regime = str(regime)
        self.coupling_strength = float(coupling_strength)
        self.control = str(control)
        self.j_b = float(j_b)
        self.j_a = float(j_a)
        self.belief_action_k = float(belief_action_k)
        self.packet_history: List[SignalPacket] = []
        if len(self.agents) != self.graph.n_agents:
            raise ValueError("agent and graph sizes differ")
        if self.regime not in ("markovized", "persistent_memory") or self.control not in CONTROLS:
            raise ValueError("invalid regime or control")
        self.graph.validate()
        self.mapping.validate()

    def private_fingerprints(self) -> Tuple[str, ...]:
        return tuple(agent.private_fingerprint() for agent in self.agents)

    def state_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([agent.belief for agent in self.agents], dtype=int),
            np.asarray([agent.action for agent in self.agents], dtype=int),
        )

    def offered_update(
        self,
        provider: StructuredProvider,
        tape: UpdateTape,
        update_index: int,
        sampling_temperature: float,
    ) -> Dict[str, object]:
        scheduled = int(tape.scheduled_agent)
        agent = self.agents[scheduled]
        before_fingerprints = self.private_fingerprints()
        beliefs_before, actions_before = self.state_vectors()
        confidences_before = [item.confidence for item in self.agents]
        commitments_before = [COMMITMENTS.index(item.commitment) for item in self.agents]
        memories_before = [MEMORY_STATES.index(item.memory_state) for item in self.agents]
        workloads_before = [item.workload for item in self.agents]
        state_before = encode_microstate(self.agents)
        observables_before = network_observables(
            beliefs_before,
            actions_before,
            self.graph.adjacency,
            self.graph.symmetric,
            [item.private_field for item in self.agents],
            self.j_b,
            self.j_a,
            self.belief_action_k,
        )
        inbox_before = tuple(agent.inbox)
        neighbor_field = float(np.mean([packet.signal_spin for packet in inbox_before])) if inbox_before else 0.0
        order = tuple(reversed(self.mapping.display_order)) if tape.reverse_display_order else self.mapping.display_order
        prompt_mapping = LatentMapping(self.mapping.plus_label, tuple(order))
        prompt = build_agent_prompt(
            agent,
            prompt_mapping,
            int(update_index),
            self.regime,
            self.coupling_strength,
            int(tape.paraphrase),
        )
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        result = None
        decision: Optional[AgentDecision] = None
        invalid = False
        try:
            result = provider.decide(prompt, int(tape.inference_seed), float(sampling_temperature))
            decision = AgentDecision.from_mapping(result.payload)
        except InvalidStructuredDecision as error:
            invalid = True
            result = error.result
        recipient: Optional[int] = None
        transmitted = 0
        delivered = 0
        dropped = 0
        wire_bytes = 0
        workload_change = 0
        peer_mutations = 0
        belief_before = agent.belief
        action_before = agent.action
        confidence_before = agent.confidence
        commitment_before = agent.commitment
        memory_before = agent.memory_state
        if decision is not None:
            packet = agent.apply_decision(
                decision,
                self.mapping,
                int(update_index),
                persistent_memory=self.regime == "persistent_memory",
            )
            workload_change = agent.apply_tool_consequence(decision.tool_action)
            transformed = _transform_packet(
                packet, self.control, self.packet_history, tape.recipient_uniform, len(self.agents)
            )
            self.packet_history.append(packet)
            if self.control == "no_message":
                dropped = 1
            else:
                recipient = select_recipient(self.graph.weights, scheduled, tape.recipient_uniform)
                encoded = serialize_signal_packet(transformed)
                transmitted = 1
                wire_bytes = len(encoded)
                self.agents[recipient].receive(transformed)
                delivered = 1
        # An offered update consumes the currently delivered inbox even when a
        # decision remains invalid; no replacement action is selected.
        agent.clear_inbox()
        beliefs_after, actions_after = self.state_vectors()
        state_after = encode_microstate(self.agents)
        after_fingerprints = self.private_fingerprints()
        protected = {scheduled}
        if recipient is not None:
            protected.add(recipient)
        peer_mutations = int(
            sum(before_fingerprints[index] != after_fingerprints[index] for index in range(len(self.agents)) if index not in protected)
        )
        observables = network_observables(
            beliefs_after,
            actions_after,
            self.graph.adjacency,
            self.graph.symmetric,
            [item.private_field for item in self.agents],
            self.j_b,
            self.j_a,
            self.belief_action_k,
        )
        return {
            "update": int(update_index),
            "sweep": float((int(update_index) + 1) / len(self.agents)),
            "scheduled_agent": scheduled,
            "recipient": -1 if recipient is None else int(recipient),
            "state_before": int(state_before),
            "state_after": int(state_after),
            "belief_before": int(belief_before),
            "belief_after": int(agent.belief),
            "action_before": int(action_before),
            "action_after": int(agent.action),
            "confidence_before": float(confidence_before),
            "confidence_after": float(agent.confidence),
            "commitment_before": commitment_before,
            "commitment_after": agent.commitment,
            "memory_before": memory_before,
            "memory_after": agent.memory_state,
            "private_field": int(agent.private_field),
            "neighbor_field": neighbor_field,
            "inbox_packets": int(len(inbox_before)),
            "beliefs": ";".join(str(int(value)) for value in beliefs_after),
            "actions": ";".join(str(int(value)) for value in actions_after),
            "beliefs_before_vector": ";".join(str(int(value)) for value in beliefs_before),
            "actions_before_vector": ";".join(str(int(value)) for value in actions_before),
            "confidences": ";".join("%.6f" % item.confidence for item in self.agents),
            "confidences_before_vector": ";".join("%.6f" % value for value in confidences_before),
            "commitments": ";".join(str(COMMITMENTS.index(item.commitment)) for item in self.agents),
            "commitments_before_vector": ";".join(str(value) for value in commitments_before),
            "memory_states": ";".join(str(MEMORY_STATES.index(item.memory_state)) for item in self.agents),
            "memory_states_before_vector": ";".join(str(value) for value in memories_before),
            "workloads": ";".join(str(item.workload) for item in self.agents),
            "workloads_before_vector": ";".join(str(value) for value in workloads_before),
            "belief_magnetization_before": observables_before["belief_magnetization"],
            "action_magnetization_before": observables_before["action_magnetization"],
            "belief_action_overlap_before": observables_before["belief_action_overlap"],
            "reference_energy_per_agent_before": observables_before["reference_energy_per_agent"],
            "workload_change": int(workload_change),
            "total_workload": int(sum(item.workload for item in self.agents)),
            "message_opportunities": 1,
            "messages_transmitted": transmitted,
            "messages_delivered": delivered,
            "messages_dropped": dropped,
            "wire_bytes": int(wire_bytes),
            "prompt_sha256": prompt_sha,
            "raw_artifact_sha256": "" if result is None else result.raw_artifact_sha256,
            "valid_after_repair": int(not invalid),
            "first_pass_valid": int(result.first_pass_valid) if result is not None else 0,
            "repaired": int(result.repaired) if result is not None else 0,
            "repair_attempted": int(not result.first_pass_valid) if result is not None else 0,
            "model_calls": int(1 + (not result.first_pass_valid)) if result is not None else 0,
            "prompt_tokens": int(result.prompt_tokens) if result is not None else 0,
            "generated_tokens": int(result.generated_tokens) if result is not None else 0,
            "latency_seconds": float(result.latency_seconds) if result is not None else 0.0,
            "unrelated_peer_private_mutations": peer_mutations,
            **observables,
        }


def run_trajectory(
    provider: StructuredProvider,
    graph: DeliveryGraph,
    panel_seed: int,
    sweeps: int,
    regime: str,
    coupling_strength: float,
    sampling_temperature: float,
    initial_condition: str,
    control: str = "unaltered",
    metadata: Optional[Mapping[str, object]] = None,
    mapping_override: Optional[LatentMapping] = None,
) -> List[Dict[str, object]]:
    agents = make_agents(graph.n_agents, int(panel_seed), initial_condition)
    mapping = mapping_override or LatentMapping.balanced(int(panel_seed) + 17011)
    network = DecentralizedStatmechNetwork(
        agents,
        graph,
        mapping,
        regime,
        coupling_strength,
        control=control,
    )
    updates = int(sweeps) * graph.n_agents
    tape = generate_update_tape(graph.n_agents, updates, int(panel_seed) + 29009)
    prefix = dict(metadata or {})
    prefix.update(
        {
            "n_agents": graph.n_agents,
            "topology": graph.topology,
            "alpha": graph.alpha,
            "orientation_seed": graph.orientation_seed,
            "regime": regime,
            "coupling_strength": float(coupling_strength),
            "sampling_temperature": float(sampling_temperature),
            "initial_condition": initial_condition,
            "control": control,
            "latent_plus_label": mapping.plus_label,
        }
    )
    rows: List[Dict[str, object]] = []
    for update_index, item in enumerate(tape):
        row = network.offered_update(provider, item, update_index, sampling_temperature)
        row.update(prefix)
        rows.append(row)
    return rows
