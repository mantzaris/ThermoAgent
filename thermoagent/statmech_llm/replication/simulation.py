"""V13 random-sequential LLM-agent dynamics with frozen controlled quenches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from thermoagent.statmech_llm.discovery.core import (
    AgentDecision,
    IndependentStatmechAgent,
    LatentMapping,
    SignalPacket,
    StructuredProvider,
)
from thermoagent.statmech_llm.discovery.graphs import DeliveryGraph, build_delivery_graph
from thermoagent.statmech_llm.discovery.simulation import (
    DecentralizedStatmechNetwork,
    UpdateTape,
    generate_update_tape,
    make_agents,
)

from .observables import instantaneous_state


DISRUPTIONS = ("nominal", "field_reversal", "network_partition", "message_corruption")


@dataclass
class ReplicationAgent(IndependentStatmechAgent):
    """Agent-owned corruption hook used only to model channel corruption.

    The agent's stored decision and outbox remain its actual choice. The
    returned transport packet is corrupted by the channel after that choice.
    Exactly half the even-N senders are preassigned to the corrupted channel.
    """

    _channel_corruption_active: bool = False
    _corruptible_sender: bool = False

    def apply_decision(
        self,
        decision: AgentDecision,
        mapping: LatentMapping,
        time_step: int,
        persistent_memory: bool,
    ) -> SignalPacket:
        packet = super().apply_decision(decision, mapping, time_step, persistent_memory)
        if not (self._channel_corruption_active and self._corruptible_sender):
            return packet
        return SignalPacket(
            sender_id=packet.sender_id,
            time_step=packet.time_step,
            belief_spin=-packet.belief_spin,
            action_spin=-packet.action_spin,
            signal_spin=0 if packet.signal_spin == 0 else -packet.signal_spin,
            confidence=packet.confidence,
            commitment_code=packet.commitment_code,
            memory_code=packet.memory_code,
        )


def make_replication_agents(n_agents: int, seed: int, initial_condition: str) -> List[ReplicationAgent]:
    base = make_agents(int(n_agents), int(seed), str(initial_condition))
    return [
        ReplicationAgent(
            identifier=agent.identifier,
            role=agent.role,
            private_field=agent.private_field,
            _belief=agent.belief,
            _action=agent.action,
            _confidence=agent.confidence,
            _commitment=agent.commitment,
            _memory_state=agent.memory_state,
            _workload=agent.workload,
            _corruptible_sender=(agent.identifier % 2 == 0),
        )
        for agent in base
    ]


def partition_delivery_graph(graph: DeliveryGraph) -> DeliveryGraph:
    """Remove every cross-half edge and renormalize within communities."""

    n = graph.n_agents
    if n < 8 or n % 2:
        raise ValueError("partition quench requires an even modular graph")
    adjacency = np.asarray(graph.adjacency, dtype=int).copy()
    half = n // 2
    adjacency[:half, half:] = 0
    adjacency[half:, :half] = 0
    degree = adjacency.sum(axis=1).astype(float)
    if np.any(degree <= 0.0):
        raise ValueError("partition created an isolated agent")
    weights = adjacency / degree[:, None]
    partitioned = DeliveryGraph(
        topology=graph.topology + "_partitioned",
        adjacency=adjacency,
        symmetric=weights,
        circulation=np.zeros_like(weights),
        weights=weights,
        alpha=0.0,
        orientation_seed=graph.orientation_seed,
    )
    partitioned.validate()
    return partitioned


def phase_for_update(update: int, n_agents: int, periods_sweeps: Sequence[int] | None) -> str:
    if periods_sweeps is None:
        return "nominal"
    periods = [int(value) * int(n_agents) for value in periods_sweeps]
    if len(periods) != 3 or min(periods) <= 0:
        raise ValueError("periods must define positive baseline, disruption, recovery sweeps")
    if int(update) < periods[0]:
        return "baseline"
    if int(update) < periods[0] + periods[1]:
        return "disruption"
    return "recovery"


def build_reciprocal_graph(n_agents: int, topology: str, graph_seed: int) -> DeliveryGraph:
    return build_delivery_graph(int(n_agents), str(topology), int(graph_seed), int(graph_seed) + 31, 0.0, False)


def _bits(value: str) -> np.ndarray:
    return np.asarray([int(item) for item in str(value).split(";")], dtype=int)


def run_replication_trajectory(
    provider: StructuredProvider,
    graph: DeliveryGraph,
    panel_seed: int,
    sweeps: int,
    regime: str,
    coupling_strength: float,
    sampling_temperature: float,
    initial_condition: str,
    disruption: str = "nominal",
    periods_sweeps: Sequence[int] | None = None,
    metadata: Optional[Mapping[str, object]] = None,
    mapping_override: Optional[LatentMapping] = None,
) -> List[Dict[str, object]]:
    if disruption not in DISRUPTIONS:
        raise ValueError("unknown V13 disruption")
    if disruption == "network_partition" and graph.topology != "modular":
        raise ValueError("network partition is defined only for modular graphs")
    if not np.isclose(graph.alpha, 0.0):
        raise ValueError("V13 primary trajectories use reciprocal base delivery")
    agents = make_replication_agents(graph.n_agents, int(panel_seed), initial_condition)
    base_fields = np.asarray([agent.private_field for agent in agents], dtype=int)
    mapping = mapping_override or LatentMapping.balanced(int(panel_seed) + 17011)
    network = DecentralizedStatmechNetwork(
        agents,
        graph,
        mapping,
        regime,
        float(coupling_strength),
        control="unaltered",
    )
    partitioned = partition_delivery_graph(graph) if graph.topology == "modular" else None
    updates = int(sweeps) * graph.n_agents
    tape = generate_update_tape(graph.n_agents, updates, int(panel_seed) + 29009)
    prefix = dict(metadata or {})
    prefix.update(
        {
            "n_agents": graph.n_agents,
            "topology": graph.topology,
            "alpha": 0.0,
            "regime": regime,
            "coupling_strength": float(coupling_strength),
            "sampling_temperature": float(sampling_temperature),
            "initial_condition": initial_condition,
            "disruption": disruption,
            "latent_plus_label": mapping.plus_label,
        }
    )
    rows: List[Dict[str, object]] = []
    for update_index, tape_item in enumerate(tape):
        phase = phase_for_update(update_index, graph.n_agents, periods_sweeps)
        active = phase == "disruption"
        field_reversed = disruption == "field_reversal" and active
        partition_active = disruption == "network_partition" and active
        corruption_active = disruption == "message_corruption" and active
        active_fields = -base_fields if field_reversed else base_fields
        for index, agent in enumerate(network.agents):
            agent.private_field = int(active_fields[index])
            if isinstance(agent, ReplicationAgent):
                agent._channel_corruption_active = bool(corruption_active)
        network.graph = partitioned if partition_active and partitioned is not None else graph
        row = network.offered_update(provider, tape_item, update_index, sampling_temperature)
        beliefs_before = _bits(row["beliefs_before_vector"])
        actions_before = _bits(row["actions_before_vector"])
        beliefs_after = _bits(row["beliefs"])
        actions_after = _bits(row["actions"])
        before = instantaneous_state(
            beliefs_before, actions_before, graph.adjacency, graph.symmetric, active_fields
        )
        after = instantaneous_state(
            beliefs_after, actions_after, graph.adjacency, graph.symmetric, active_fields
        )
        # The reference layer is fixed across a communication partition. The
        # active delivery graph is recorded separately.
        for key, value in before.items():
            row[key + "_before"] = value
        row.update(after)
        sender = int(row["scheduled_agent"])
        recipient = int(row["recipient"])
        cross_community = int(
            recipient >= 0
            and ((sender < graph.n_agents // 2) != (recipient < graph.n_agents // 2))
        )
        row.update(
            {
                "phase": phase,
                "field_reversed": int(field_reversed),
                "partition_active": int(partition_active),
                "message_corrupted": int(
                    corruption_active
                    and sender % 2 == 0
                    and int(row["messages_transmitted"]) == 1
                ),
                "cross_community_delivery": cross_community,
                "active_edge_count": int(np.sum(network.graph.adjacency) // 2),
                "base_field_vector": ";".join(str(int(value)) for value in base_fields),
                "active_field_vector": ";".join(str(int(value)) for value in active_fields),
            }
        )
        row.update(prefix)
        rows.append(row)
    # Explicit restoration is tested and recorded even though the trajectory
    # object is not reused after return.
    network.graph = graph
    for index, agent in enumerate(network.agents):
        agent.private_field = int(base_fields[index])
        if isinstance(agent, ReplicationAgent):
            agent._channel_corruption_active = False
    if not np.array_equal(np.asarray([agent.private_field for agent in network.agents]), base_fields):
        raise AssertionError("private fields did not restore")
    if network.graph is not graph:
        raise AssertionError("delivery graph did not restore")
    return rows
