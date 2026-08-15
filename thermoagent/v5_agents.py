"""Independent V5 agents with capability-checked private state."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .agents import PrivacyViolation
from .events import EventLedger
from .types import MemoryRecord, Message
from .v5_types import (
    INCIDENT_MODES, PRIMARY_ACTION_FOR_MODE, SECONDARY_ACTION_FOR_MODE,
    V5Commitment, V5Identity, V5PrivateObservation,
)


class V5PrivateVault:
    def __init__(self, owner: str) -> None:
        self._owner = str(owner)
        self._observation: Optional[V5PrivateObservation] = None
        self._working: Dict[str, Any] = {}
        self._memory: Deque[MemoryRecord] = deque(maxlen=128)

    def _check(self, requester: str) -> None:
        if requester != self._owner:
            raise PrivacyViolation("%s cannot inspect %s V5 private state" % (requester, self._owner))

    def set_observation(self, requester: str, observation: V5PrivateObservation) -> None:
        self._check(requester)
        self._observation = deepcopy(observation)

    def observation(self, requester: str) -> V5PrivateObservation:
        self._check(requester)
        if self._observation is None:
            raise RuntimeError("private observation has not been delivered")
        return deepcopy(self._observation)

    def memory(self, requester: str) -> List[MemoryRecord]:
        self._check(requester)
        return deepcopy(list(self._memory))

    def remember(self, requester: str, record: MemoryRecord) -> None:
        self._check(requester)
        self._memory.append(deepcopy(record))

    def working(self, requester: str) -> Dict[str, Any]:
        self._check(requester)
        return deepcopy(self._working)

    def update(self, requester: str, values: Mapping[str, Any]) -> None:
        self._check(requester)
        self._working.update(deepcopy(dict(values)))


@dataclass(frozen=True)
class V5Utility:
    service_weight: float
    safety_weight: float
    cost_weight: float
    disclosure_cost: float
    risk_tolerance: float


class IndependentV5Agent:
    """Persistent agent; it never holds references to peer vaults."""

    def __init__(self, identity: V5Identity, utility: V5Utility, seed: int) -> None:
        self.identity = identity
        self.utility = deepcopy(utility)
        self.vault = V5PrivateVault(identity.agent_id)
        self.inbox: Deque[Message] = deque()
        self.outbox: Deque[Message] = deque()
        self.commitments: Dict[str, V5Commitment] = {}
        self.private_beliefs: Dict[str, Tuple[float, ...]] = {}
        self.rng = np.random.RandomState(int(seed))

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    def deliver(self, observation: V5PrivateObservation, ledger: EventLedger) -> None:
        self.vault.set_observation(self.agent_id, observation)
        evidence = np.maximum(np.asarray(observation.private_evidence, dtype=float), 1e-9)
        evidence /= evidence.sum()
        self.private_beliefs[observation.incident_id] = tuple(float(value) for value in evidence)
        ledger.append(
            observation.step, "observation_delivery", "simulator",
            {"recipient": self.agent_id, "observation": asdict(observation), "v5": True},
            private_to=self.agent_id,
        )
        ledger.append(
            observation.step, "belief_update", self.agent_id,
            {"incident_id": observation.incident_id, "belief_distribution": list(evidence), "v5": True},
            private_to=self.agent_id,
        )

    def receive(self, message: Message) -> None:
        if message.recipient not in (self.agent_id, "broadcast"):
            raise PrivacyViolation("message delivered across the V5 authority boundary")
        self.inbox.append(deepcopy(message))

    def context(self) -> Dict[str, Any]:
        observation = self.vault.observation(self.agent_id)
        return {
            "identity": asdict(self.identity),
            "utility": asdict(self.utility),
            "private_observation": asdict(observation),
            "private_memory": [asdict(item) for item in self.vault.memory(self.agent_id)],
            "private_beliefs": deepcopy(self.private_beliefs),
            "inbox": [asdict(item) for item in self.inbox],
            "commitments": [asdict(item) for item in self.commitments.values()],
        }

    def preferred_action(self, belief: Optional[np.ndarray] = None) -> str:
        observation = self.vault.observation(self.agent_id)
        distribution = np.asarray(
            belief if belief is not None else self.private_beliefs[observation.incident_id],
            dtype=float,
        )
        mode = INCIDENT_MODES[int(np.argmax(distribution))]
        confidence = float(distribution.max())
        if observation.telemetry_confidence < 0.35 and confidence < 0.55:
            return "request_peer_evidence"
        return PRIMARY_ACTION_FOR_MODE[mode]

    def evaluate_commitment(self, commitment: V5Commitment) -> str:
        observation = self.vault.observation(self.agent_id)
        value = (
            self.utility.service_weight * observation.visible_severity
            + self.utility.safety_weight * observation.safety_risk
            - self.utility.cost_weight * observation.private_cost
        )
        burden = commitment.quantity / max(observation.private_inventory, 0.25)
        if value >= 0.55 * burden:
            return "accept"
        if value >= 0.33 * burden and commitment.revision < 2:
            return "counter"
        return "reject"

    def apply_commitment(self, commitment: V5Commitment, decision: str) -> V5Commitment:
        if decision not in ("accept", "counter", "reject"):
            raise ValueError("invalid commitment decision")
        updated = deepcopy(commitment)
        if decision == "counter":
            updated.quantity *= 0.75
            updated.revision += 1
            updated.status = "countered"
        else:
            updated.status = "accepted" if decision == "accept" else "rejected"
        self.commitments[updated.commitment_id] = deepcopy(updated)
        self.vault.remember(
            self.agent_id,
            MemoryRecord(
                step=self.vault.observation(self.agent_id).step,
                kind="commitment",
                summary="%s %s" % (decision, commitment.commitment_id),
                importance=0.8,
            ),
        )
        return updated
