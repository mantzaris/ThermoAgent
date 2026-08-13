"""Independent autonomous-agent state and decision-loop scaffolding."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .events import EventLedger
from .types import (
    Commitment,
    EntropySummary,
    Identity,
    MemoryRecord,
    Message,
    PrivateObservation,
    UtilityWeights,
)


class PrivacyViolation(PermissionError):
    pass


class PrivateStateVault:
    """Capability boundary used by agents and exercised by privacy tests."""

    def __init__(self, owner: str) -> None:
        self._owner = owner
        self._observation: Optional[PrivateObservation] = None
        self._working_memory: Dict[str, Any] = {}
        self._episodic_memory: Deque[MemoryRecord] = deque(maxlen=64)

    def _authorize(self, requester: str) -> None:
        if requester != self._owner:
            raise PrivacyViolation("%s cannot inspect %s private state" % (requester, self._owner))

    def set_observation(self, requester: str, observation: PrivateObservation) -> None:
        self._authorize(requester)
        self._observation = deepcopy(observation)

    def observation(self, requester: str) -> PrivateObservation:
        self._authorize(requester)
        if self._observation is None:
            raise RuntimeError("private observation has not been delivered")
        return deepcopy(self._observation)

    def working_memory(self, requester: str) -> Dict[str, Any]:
        self._authorize(requester)
        return deepcopy(self._working_memory)

    def update_working(self, requester: str, values: Mapping[str, Any]) -> None:
        self._authorize(requester)
        self._working_memory.update(deepcopy(dict(values)))

    def add_episode(self, requester: str, record: MemoryRecord) -> None:
        self._authorize(requester)
        self._episodic_memory.append(deepcopy(record))

    def retrieve(self, requester: str, limit: int = 5) -> List[MemoryRecord]:
        self._authorize(requester)
        ranked = sorted(self._episodic_memory, key=lambda r: (r.importance, r.step), reverse=True)
        return deepcopy(ranked[:limit])


class AutonomousAgent:
    """One persistent agent with no references to another agent's vault."""

    def __init__(
        self,
        identity: Identity,
        utility: UtilityWeights,
        risk_tolerance: float,
        rng_seed: int,
    ) -> None:
        self.identity = identity
        self.utility = deepcopy(utility)
        self.risk_tolerance = float(risk_tolerance)
        self.vault = PrivateStateVault(identity.agent_id)
        self.beliefs: Dict[str, Dict[str, float]] = {}
        self.commitments: Dict[str, Commitment] = {}
        # Private commitment authority also includes a local view of temporary
        # coalition contracts.  It is updated only by this agent's validated
        # tool results or explicitly delivered proposals.
        self.coalition_ledger: Dict[str, Dict[str, Any]] = {}
        self.partner_trust: Dict[str, float] = {}
        self.inbox: Deque[Message] = deque()
        self.outbox: Deque[Message] = deque()
        self.entropy = EntropySummary()
        self.policy_state = np.zeros(8, dtype=float)
        self.communication_budget = 12
        self.last_plan_summary = "No prior plan."
        self.last_tool_ok = True
        self.rng = np.random.RandomState(rng_seed)

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    def deliver_observation(self, observation: PrivateObservation, ledger: EventLedger) -> None:
        self.vault.set_observation(self.agent_id, observation)
        ledger.append(
            observation.step,
            "observation_delivery",
            "simulator",
            {"recipient": self.agent_id, "observation": asdict(observation)},
            private_to=self.agent_id,
        )

    def deliver_message(self, message: Message) -> None:
        if message.recipient not in (self.agent_id, "broadcast"):
            raise PrivacyViolation("message delivered to wrong recipient")
        self.inbox.append(deepcopy(message))

    def retrieve_context(
        self,
        step: int,
        ledger: EventLedger,
        include_episodic_memory: bool = True,
    ) -> Dict[str, Any]:
        observation = self.vault.observation(self.agent_id)
        memories = self.vault.retrieve(self.agent_id, 4) if include_episodic_memory else []
        ledger.append(
            step,
            "memory_retrieval",
            self.agent_id,
            {
                "count": len(memories),
                "kinds": [m.kind for m in memories],
                "episodic_memory_enabled": include_episodic_memory,
            },
            private_to=self.agent_id,
        )
        return {
            "identity": asdict(self.identity),
            "utility": asdict(self.utility),
            "risk_tolerance": self.risk_tolerance,
            "observation": asdict(observation),
            "beliefs": deepcopy(self.beliefs),
            "working_memory": self.vault.working_memory(self.agent_id),
            "memories": [asdict(m) for m in memories],
            "commitments": [asdict(c) for c in self.commitments.values()],
            "coalitions": deepcopy(self.coalition_ledger),
            "partner_trust": deepcopy(self.partner_trust),
            "messages": [asdict(m) for m in list(self.inbox)[-8:]],
            "entropy": asdict(self.entropy),
            "communication_budget": self.communication_budget,
            "last_plan_summary": self.last_plan_summary,
            "last_tool_ok": self.last_tool_ok,
        }

    def update_beliefs(self) -> None:
        for message in list(self.inbox)[-8:]:
            sender = message.sender
            self.partner_trust.setdefault(sender, 0.5)
            if message.kind in ("delivery_verified", "commitment_honored"):
                self.partner_trust[sender] = min(1.0, self.partner_trust[sender] + 0.1)
            if message.kind in ("commitment_breach", "invalid_claim"):
                self.partner_trust[sender] = max(0.0, self.partner_trust[sender] - 0.2)
            if message.kind in ("need", "summary", "offer", "counteroffer"):
                self.beliefs.setdefault(sender, {})[message.kind] = float(message.sent_step)

    def observation_vector(self, role_index: int, include_entropy: bool = True) -> np.ndarray:
        obs = self.vault.observation(self.agent_id)
        pending = sum(1 for c in self.commitments.values() if c.status == "proposed")
        accepted = sum(1 for c in self.commitments.values() if c.status == "accepted")
        trust = float(np.mean(list(self.partner_trust.values()))) if self.partner_trust else 0.5
        entropy = self.entropy if include_entropy else EntropySummary()
        values = [
            min(obs.backlog / max(obs.local_forecast, 1.0), 2.0) / 2.0,
            min(obs.inventory / max(obs.capacity, 1.0), 2.0) / 2.0,
            obs.impairment,
            obs.delay,
            obs.service_shortfall,
            obs.commitment_strain,
            obs.communication_reliability,
            min(obs.private_cost / 5.0, 1.0),
            self.utility.service / 2.0,
            self.utility.cost / 2.0,
            self.utility.fairness / 2.0,
            self.utility.disclosure / 2.0,
            min(pending / 4.0, 1.0),
            min(accepted / 4.0, 1.0),
            trust,
            min(self.communication_budget / 12.0, 1.0),
            entropy.local_surprisal / 8.0,
            entropy.local_entropy,
            min(entropy.local_free_energy / 2.0, 1.0),
            max(-1.0, min(1.0, entropy.delta_free_energy)),
            entropy.interaction_entropy,
            min(entropy.consensus_error * 20.0, 1.0),
            role_index / 10.0,
            float(not self.last_tool_ok),
        ]
        return np.asarray(values, dtype=np.float32)

    def reflect(self, step: int, plan_summary: str, tool_ok: bool, result_code: str) -> None:
        revised = not tool_ok and self.last_tool_ok
        self.last_plan_summary = plan_summary
        self.last_tool_ok = bool(tool_ok)
        self.vault.update_working(self.agent_id, {"last_result": result_code, "needs_revision": not tool_ok})
        self.vault.add_episode(
            self.agent_id,
            MemoryRecord(
                step=step,
                kind="failure" if not tool_ok else "outcome",
                summary="%s -> %s" % (plan_summary[:100], result_code),
                importance=0.9 if not tool_ok else 0.4,
            ),
        )
