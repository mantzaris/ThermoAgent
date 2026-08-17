"""Independent autonomous agents for the V9 statistical model.

The mathematical equilibrium limit can be simulated directly from a microstate,
but this module makes the agent boundary executable: policies receive local
state plus explicitly delivered messages, never a system object or peer vault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .model import ModelParameters


@dataclass(frozen=True)
class LocalMessage:
    sender: int
    recipient: int
    time_step: int
    belief: int
    action: int


@dataclass(frozen=True)
class LocalView:
    own_belief: int
    own_action: int
    own_memory: float
    own_workload: float
    private_field: float
    task_field: float
    commitments: Tuple[Tuple[str, int], ...]
    delivered_messages: Tuple[LocalMessage, ...]


@dataclass
class IndependentAgent:
    """Persistent private state and a local stochastic decision rule."""

    identifier: int
    role: str
    _belief: int
    _action: int
    _memory: float
    _workload: float
    _private_field: float
    _task_field: float
    _commitments: Dict[str, int] = field(default_factory=dict)
    _inbox: List[LocalMessage] = field(default_factory=list)
    _outbox: List[LocalMessage] = field(default_factory=list)

    def local_view(self) -> LocalView:
        return LocalView(
            own_belief=self._belief,
            own_action=self._action,
            own_memory=self._memory,
            own_workload=self._workload,
            private_field=self._private_field,
            task_field=self._task_field,
            commitments=tuple(sorted(self._commitments.items())),
            delivered_messages=tuple(self._inbox),
        )

    def receive(self, message: LocalMessage) -> None:
        if message.recipient != self.identifier:
            raise ValueError("message recipient mismatch")
        self._inbox.append(message)

    def clear_inbox_before(self, time_step: int) -> None:
        self._inbox[:] = [message for message in self._inbox if message.time_step >= time_step]

    def set_private_observation(self, private_field: float) -> None:
        self._private_field = float(private_field)

    def set_task_field(self, task_field: float) -> None:
        self._task_field = float(task_field)

    def revise_commitment(self, name: str, value: int) -> None:
        self._commitments[str(name)] = int(value)

    def decide(
        self,
        layer: str,
        parameters: ModelParameters,
        communication_weights: Mapping[int, float],
        dependency_weights: Mapping[int, float],
        uniform_draw: float,
    ) -> int:
        """Sample from a local logit using only the authorized local view."""

        latest: Dict[int, LocalMessage] = {}
        for message in self._inbox:
            if message.sender not in latest or message.time_step > latest[message.sender].time_step:
                latest[message.sender] = message
        if layer == "belief":
            neighbor_term = sum(
                communication_weights.get(sender, 0.0) * message.belief
                for sender, message in latest.items()
            )
            field_value = (
                parameters.belief_coupling * neighbor_term
                + parameters.belief_action_coupling * self._action
                + self._private_field
                + parameters.memory_coupling * self._memory
            )
        elif layer == "action":
            neighbor_term = sum(
                dependency_weights.get(sender, 0.0) * message.action
                for sender, message in latest.items()
            )
            field_value = (
                parameters.action_coupling * neighbor_term
                + parameters.belief_action_coupling * self._belief
                + self._task_field
                + self._workload
            )
        else:
            raise ValueError("unknown decision layer")
        probability_plus = 1.0 / (1.0 + np.exp(-np.clip(2.0 * field_value / parameters.temperature, -700, 700)))
        return 1 if float(uniform_draw) < probability_plus else -1

    def apply_decision(self, layer: str, value: int, memory_rate: float) -> None:
        if value not in (-1, 1):
            raise ValueError("agent decisions are binary")
        if layer == "belief":
            self._belief = value
            self._memory = (1.0 - memory_rate) * self._memory + memory_rate * value
        elif layer == "action":
            self._action = value
        else:
            raise ValueError("unknown decision layer")

    def prepare_messages(self, neighbors: List[int], time_step: int) -> List[LocalMessage]:
        messages = [
            LocalMessage(self.identifier, recipient, time_step, self._belief, self._action)
            for recipient in neighbors
        ]
        self._outbox.extend(messages)
        return messages


class DecentralizedAgentSystem:
    """Schedules independent agents and explicitly routes their messages."""

    def __init__(
        self,
        agents: List[IndependentAgent],
        communication: np.ndarray,
        dependency: np.ndarray,
        parameters: ModelParameters,
    ) -> None:
        self._agents = {agent.identifier: agent for agent in agents}
        if sorted(self._agents) != list(range(len(agents))):
            raise ValueError("agent identifiers must be contiguous")
        self.communication = np.asarray(communication, dtype=float).copy()
        self.dependency = np.asarray(dependency, dtype=float).copy()
        if self.communication.shape != (len(agents), len(agents)) or self.dependency.shape != self.communication.shape:
            raise ValueError("network shape does not match agents")
        self.parameters = parameters
        self.time_step = 0
        self.message_ledger: List[LocalMessage] = []

    @property
    def n_agents(self) -> int:
        return len(self._agents)

    def authorized_snapshot(self) -> Dict[int, LocalView]:
        return {identifier: agent.local_view() for identifier, agent in self._agents.items()}

    def private_agent_for_test(self, identifier: int) -> IndependentAgent:
        """Explicit test-only access; policies never receive this reference."""

        return self._agents[identifier]

    def broadcast(self, sender: int) -> int:
        neighbors = list(np.flatnonzero(self.communication[sender] != 0.0).astype(int))
        messages = self._agents[sender].prepare_messages(neighbors, self.time_step)
        for message in messages:
            self._agents[message.recipient].receive(message)
            self.message_ledger.append(message)
        return len(messages)

    def step(self, agent_index: int, layer: str, uniform_draw: float) -> int:
        agent = self._agents[int(agent_index)]
        communication_weights = {
            int(index): float(weight)
            for index, weight in enumerate(self.communication[agent_index])
            if weight != 0.0
        }
        dependency_weights = {
            int(index): float(weight)
            for index, weight in enumerate(self.dependency[agent_index])
            if weight != 0.0
        }
        decision = agent.decide(
            layer,
            self.parameters,
            communication_weights,
            dependency_weights,
            uniform_draw,
        )
        agent.apply_decision(layer, decision, self.parameters.memory_rate)
        self.broadcast(agent_index)
        self.time_step += 1
        return decision


def make_independent_agents(n_agents: int, seed: int) -> List[IndependentAgent]:
    rng = np.random.default_rng(seed)
    return [
        IndependentAgent(
            identifier=index,
            role="belief_action_agent",
            _belief=int(rng.choice([-1, 1])),
            _action=int(rng.choice([-1, 1])),
            _memory=0.0,
            _workload=float(rng.uniform(0.0, 0.2)),
            _private_field=float(rng.normal(0.0, 0.2)),
            _task_field=float(rng.normal(0.0, 0.2)),
        )
        for index in range(n_agents)
    ]
