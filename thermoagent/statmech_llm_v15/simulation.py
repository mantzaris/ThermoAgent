"""Matched field-quench trajectories with genuine and placebo memory."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from thermoagent.statmech_llm_v12.core import LatentMapping, MEMORY_STATES, StructuredProvider
from thermoagent.statmech_llm_v12.graphs import DeliveryGraph
from thermoagent.statmech_llm_v12.simulation import DecentralizedStatmechNetwork, UpdateTape, generate_update_tape
from thermoagent.statmech_llm_v13.observables import instantaneous_state
from thermoagent.statmech_llm_v13.simulation import build_reciprocal_graph, make_v13_agents, phase_for_update


MEMORY_MODES = ("markovized", "persistent_memory", "scrambled_memory")
CONDITIONS = (
    "nominal_markovized",
    "field_markovized",
    "field_persistent",
    "field_scrambled",
)


def _bits(value: str) -> np.ndarray:
    return np.asarray([int(item) for item in str(value).split(";")], dtype=int)


def condition_specification(condition: str) -> Tuple[str, str]:
    if condition == "nominal_markovized":
        return "nominal", "markovized"
    if condition == "field_markovized":
        return "field_reversal", "markovized"
    if condition == "field_persistent":
        return "field_reversal", "persistent_memory"
    if condition == "field_scrambled":
        return "field_reversal", "scrambled_memory"
    raise ValueError("unknown V15 condition")


class V15MemoryNetwork(DecentralizedStatmechNetwork):
    """Use a matched prompt-format placebo without retaining genuine history."""

    def __init__(
        self,
        agents,
        graph: DeliveryGraph,
        mapping: LatentMapping,
        memory_mode: str,
        coupling_strength: float,
        control_seed: int,
    ) -> None:
        if memory_mode not in MEMORY_MODES:
            raise ValueError("invalid V15 memory mode")
        parent_regime = "markovized" if memory_mode == "markovized" else "persistent_memory"
        super().__init__(agents, graph, mapping, parent_regime, coupling_strength, control="unaltered")
        self.memory_mode = str(memory_mode)
        self.control_seed = int(control_seed)
        self._opportunity_times: List[List[int]] = [[] for _ in self.agents]

    def scrambled_entries(self, agent_id: int) -> Tuple[str, ...]:
        """Return deterministic own-agent placebo entries from past turn times."""

        entries: List[str] = []
        for past_time in self._opportunity_times[int(agent_id)][-3:]:
            digest = hashlib.sha256(
                ("%d:%d:%d" % (self.control_seed, int(agent_id), int(past_time))).encode("ascii")
            ).digest()
            belief_spin = 1 if digest[0] % 2 else -1
            action_spin = 1 if digest[1] % 2 else -1
            memory = MEMORY_STATES[int(digest[2]) % len(MEMORY_STATES)]
            entries.append(
                "t=%d belief=%s action=%s memory=%s"
                % (
                    int(past_time),
                    self.mapping.label(belief_spin),
                    self.mapping.label(action_spin),
                    memory,
                )
            )
        return tuple(entries)

    def offered_update(
        self,
        provider: StructuredProvider,
        tape: UpdateTape,
        update_index: int,
        sampling_temperature: float,
    ) -> Dict[str, object]:
        scheduled = int(tape.scheduled_agent)
        agent = self.agents[scheduled]
        entries: Tuple[str, ...] = tuple(agent.memory)
        if self.memory_mode == "scrambled_memory":
            entries = self.scrambled_entries(scheduled)
            agent._memory_history[:] = list(entries)
        row = super().offered_update(provider, tape, update_index, sampling_temperature)
        if self.memory_mode == "scrambled_memory":
            # The model-selected categorical memory state remains.  Only the
            # persistent free-history list is placebo-controlled and cleared.
            agent._memory_history.clear()
        self._opportunity_times[scheduled].append(int(update_index))
        row.update(
            {
                "memory_mode": self.memory_mode,
                "prompt_memory_entry_count": int(len(entries)),
                "prompt_memory_characters": int(sum(len(value) for value in entries)),
                "memory_control_sha256": hashlib.sha256(
                    json.dumps(list(entries), separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
        return row


def memory_control_tape(
    n_agents: int,
    updates: int,
    panel_seed: int,
    control_seed: int,
    mapping: LatentMapping,
) -> List[Dict[str, object]]:
    """Materialize the prospective placebo tape without any trajectory state."""

    agents = make_v13_agents(int(n_agents), int(panel_seed), "disordered")
    graph = build_reciprocal_graph(int(n_agents), "modular", int(panel_seed) + 7001)
    network = V15MemoryNetwork(
        agents,
        graph,
        mapping,
        "scrambled_memory",
        0.8,
        int(control_seed),
    )
    tape = generate_update_tape(int(n_agents), int(updates), int(panel_seed) + 29009)
    rows: List[Dict[str, object]] = []
    for update_index, item in enumerate(tape):
        agent_id = int(item.scheduled_agent)
        entries = network.scrambled_entries(agent_id)
        rows.append(
            {
                "update": int(update_index),
                "agent_id": agent_id,
                "entries": list(entries),
            }
        )
        network._opportunity_times[agent_id].append(int(update_index))
    return rows


def run_v15_trajectory(
    provider: StructuredProvider,
    graph: DeliveryGraph,
    panel_seed: int,
    sweeps: int,
    condition: str,
    coupling_strength: float,
    sampling_temperature: float,
    periods_sweeps: Sequence[int],
    metadata: Optional[Mapping[str, object]] = None,
    mapping_override: Optional[LatentMapping] = None,
    control_seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    disruption, memory_mode = condition_specification(condition)
    if graph.topology != "modular" or not np.isclose(graph.alpha, 0.0):
        raise ValueError("V15 formal design requires a reciprocal modular graph")
    agents = make_v13_agents(graph.n_agents, int(panel_seed), "disordered")
    base_fields = np.asarray([agent.private_field for agent in agents], dtype=int)
    mapping = mapping_override or LatentMapping.balanced(int(panel_seed) + 17011)
    network = V15MemoryNetwork(
        agents,
        graph,
        mapping,
        memory_mode,
        float(coupling_strength),
        int(control_seed if control_seed is not None else panel_seed + 51001),
    )
    updates = int(sweeps) * graph.n_agents
    tape = generate_update_tape(graph.n_agents, updates, int(panel_seed) + 29009)
    prefix = dict(metadata or {})
    prefix.update(
        {
            "n_agents": graph.n_agents,
            "topology": graph.topology,
            "alpha": 0.0,
            "regime": memory_mode,
            "condition": condition,
            "disruption": disruption,
            "coupling_strength": float(coupling_strength),
            "sampling_temperature": float(sampling_temperature),
            "initial_condition": "disordered",
            "latent_plus_label": mapping.plus_label,
        }
    )
    rows: List[Dict[str, object]] = []
    for update_index, tape_item in enumerate(tape):
        phase = phase_for_update(update_index, graph.n_agents, periods_sweeps)
        field_reversed = disruption == "field_reversal" and phase == "disruption"
        active_fields = -base_fields if field_reversed else base_fields
        for index, agent in enumerate(network.agents):
            agent.private_field = int(active_fields[index])
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
        for key, value in before.items():
            row[key + "_before"] = value
        row.update(after)
        row.update(
            {
                "phase": phase,
                "field_reversed": int(field_reversed),
                "message_corrupted": 0,
                "base_field_vector": ";".join(str(int(value)) for value in base_fields),
                "active_field_vector": ";".join(str(int(value)) for value in active_fields),
            }
        )
        row.update(prefix)
        rows.append(row)
    for index, agent in enumerate(network.agents):
        agent.private_field = int(base_fields[index])
    if not np.array_equal(
        np.asarray([agent.private_field for agent in network.agents]), base_fields
    ):
        raise AssertionError("private fields did not restore")
    return rows


__all__ = [
    "CONDITIONS",
    "MEMORY_MODES",
    "V15MemoryNetwork",
    "condition_specification",
    "memory_control_tape",
    "run_v15_trajectory",
]
