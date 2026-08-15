"""Independent v4 agents with private observations and separate authority."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .agents import PrivacyViolation
from .events import EventLedger
from .types import MemoryRecord, Message, PlanOutput, ToolResult
from .v4_types import V4Commitment, V4Identity, V4PrivateObservation, bounded_probability


class V4PrivateStateVault:
    """A capability boundary for v4 observations, memory, and hidden beliefs."""

    def __init__(self, owner: str) -> None:
        self._owner = str(owner)
        self._observation: Optional[V4PrivateObservation] = None
        self._working: Dict[str, Any] = {}
        self._episodes: Deque[MemoryRecord] = deque(maxlen=96)

    def _authorize(self, requester: str) -> None:
        if requester != self._owner:
            raise PrivacyViolation("%s cannot inspect %s v4 private state" % (requester, self._owner))

    def set_observation(self, requester: str, observation: V4PrivateObservation) -> None:
        self._authorize(requester)
        self._observation = deepcopy(observation)

    def observation(self, requester: str) -> V4PrivateObservation:
        self._authorize(requester)
        if self._observation is None:
            raise RuntimeError("v4 private observation has not been delivered")
        return deepcopy(self._observation)

    def working_memory(self, requester: str) -> Dict[str, Any]:
        self._authorize(requester)
        return deepcopy(self._working)

    def update_working(self, requester: str, values: Mapping[str, Any]) -> None:
        self._authorize(requester)
        self._working.update(deepcopy(dict(values)))

    def remember(self, requester: str, record: MemoryRecord) -> None:
        self._authorize(requester)
        self._episodes.append(deepcopy(record))

    def retrieve(self, requester: str, limit: int = 5) -> List[MemoryRecord]:
        self._authorize(requester)
        ranked = sorted(self._episodes, key=lambda item: (item.importance, item.step), reverse=True)
        return deepcopy(ranked[:limit])


@dataclass(frozen=True)
class V4AgentUtility:
    service_weight: float
    cost_weight: float
    safety_weight: float
    disclosure_cost: float
    priority_weight: float
    reservation_value: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class IndependentV4Agent:
    """Persistent independent organization; no references to peer vaults."""

    def __init__(
        self,
        identity: V4Identity,
        utility: V4AgentUtility,
        rng_seed: int,
    ) -> None:
        self.identity = identity
        self.utility = deepcopy(utility)
        self.vault = V4PrivateStateVault(identity.agent_id)
        self.inbox: Deque[Message] = deque()
        self.outbox: Deque[Message] = deque()
        self.commitments: Dict[str, V4Commitment] = {}
        self.partner_reliability: Dict[str, float] = {}
        self.private_beliefs: Dict[str, List[float]] = {}
        self.authority: Dict[str, Any] = {"autonomy_level": 0, "directives": []}
        self.last_plan = "continue authorized local operation"
        self.last_result = "not_run"
        self.rng = np.random.RandomState(int(rng_seed))

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    def deliver_observation(self, observation: V4PrivateObservation, ledger: EventLedger) -> None:
        self.vault.set_observation(self.agent_id, observation)
        operational = bounded_probability(observation.belief_operational)
        unreliable = bounded_probability(observation.belief_telemetry_unreliable)
        physical = max(1e-6, 1.0 - operational - 0.5 * unreliable)
        values = np.asarray([operational, physical, max(1e-6, unreliable)], dtype=float)
        values /= values.sum()
        self.private_beliefs[observation.incident_id] = values.tolist()
        ledger.append(
            observation.step,
            "observation_delivery",
            "simulator",
            {"recipient": self.agent_id, "observation": asdict(observation)},
            private_to=self.agent_id,
        )
        ledger.append(
            observation.step,
            "belief_update",
            self.agent_id,
            {
                "incident_id": observation.incident_id,
                "belief_distribution": values.tolist(),
                "source": "private_observation_only",
            },
            private_to=self.agent_id,
        )

    def receive_message(self, message: Message) -> None:
        if message.recipient not in (self.agent_id, "broadcast"):
            raise PrivacyViolation("v4 message delivered to wrong agent")
        self.inbox.append(deepcopy(message))
        self.partner_reliability.setdefault(message.sender, 0.5)

    def context(self, ledger: EventLedger, include_memory: bool = True) -> Dict[str, Any]:
        observation = self.vault.observation(self.agent_id)
        memories = self.vault.retrieve(self.agent_id, 5) if include_memory else []
        ledger.append(
            observation.step,
            "memory_retrieval",
            self.agent_id,
            {"count": len(memories), "v4": True},
            private_to=self.agent_id,
        )
        return {
            "identity": asdict(self.identity),
            "utility": self.utility.as_dict(),
            "observation": asdict(observation),
            "private_beliefs": deepcopy(self.private_beliefs),
            "working_memory": self.vault.working_memory(self.agent_id),
            "episodic_memory": [asdict(record) for record in memories],
            "messages": [asdict(message) for message in list(self.inbox)[-8:]],
            "commitments": [asdict(value) for value in self.commitments.values()],
            "partner_reliability": deepcopy(self.partner_reliability),
            "authority": deepcopy(self.authority),
            "last_plan": self.last_plan,
            "last_result": self.last_result,
        }

    def coarse_sketch(self) -> Dict[str, Any]:
        """Bounded summary; costs, inventory, and raw telemetry stay private."""

        observation = self.vault.observation(self.agent_id)
        distribution = self.private_beliefs[observation.incident_id]
        severity = max(observation.local_service_deficit, observation.local_backlog)
        return {
            "agent_id": self.agent_id,
            "role": self.identity.role,
            "incident_id": observation.incident_id,
            "severity_band": "high" if severity >= 0.55 else "nominal" if severity >= 0.20 else "low",
            "belief_distribution": [round(float(value), 4) for value in distribution],
            "telemetry_confidence_band": (
                "low" if observation.telemetry_confidence < 0.45
                else "nominal" if observation.telemetry_confidence < 0.75
                else "high"
            ),
            "communication_reliability_band": (
                "low" if observation.communication_reliability < 0.45
                else "nominal" if observation.communication_reliability < 0.80
                else "high"
            ),
        }

    def evaluate_commitment(self, commitment: V4Commitment) -> str:
        """Private utility can produce accept, reject, or counter decisions."""

        observation = self.vault.observation(self.agent_id)
        value = (
            self.utility.service_weight * observation.local_service_deficit
            + self.utility.safety_weight * observation.local_safety_stress
            + self.utility.priority_weight * observation.private_priority
            - self.utility.cost_weight * observation.private_cost
        )
        burden = commitment.quantity / max(1.0, 10.0 * observation.observed_resource_available)
        if value >= burden:
            return "accept"
        if commitment.negotiation_round < 2 and value >= 0.65 * burden:
            return "counter"
        return "reject"

    def apply_commitment_decision(self, commitment: V4Commitment, decision: str) -> V4Commitment:
        if decision not in ("accept", "reject", "counter"):
            raise ValueError("invalid commitment decision")
        updated = deepcopy(commitment)
        if decision == "counter":
            updated.status = "countered"
            updated.quantity = max(0.1, 0.75 * updated.quantity)
            updated.negotiation_round += 1
        else:
            updated.status = "accepted" if decision == "accept" else "rejected"
        self.commitments[updated.commitment_id] = deepcopy(updated)
        return updated

    def choose_local_plan(self) -> PlanOutput:
        """Transparent deterministic planner used only for development mechanics."""

        observation = self.vault.observation(self.agent_id)
        role = self.identity.role
        incident = observation.incident_id
        if observation.telemetry_confidence < 0.45 and role in {
            "distribution_zone", "substation", "microgrid", "critical_load",
        }:
            return PlanOutput(
                "Request bounded verification before committing a scarce restoration resource.",
                "request_telemetry_verification",
                {"asset_id": incident, "reason": "conflicting or low-confidence local evidence"},
                "Private telemetry confidence is below the prospective abstention boundary.",
                0.82,
            )
        if role == "crew_dispatch":
            return PlanOutput(
                "Dispatch one locally available crew to the incident.",
                "dispatch_field_crew",
                {"crew_id": "crew_1", "target_zone": incident, "skill": "electrical"},
                "Local service deficit is actionable under current authority.",
                0.76,
            )
        if role == "parts_depot":
            return PlanOutput(
                "Allocate one validated abstract spare component.",
                "allocate_spare_component",
                {"component": "switch_module", "quantity": 1, "target_zone": incident},
                "The local parts record shows an available unit.",
                0.74,
            )
        if role == "mobile_generation":
            return PlanOutput(
                "Route one mobile generator to protect critical service.",
                "route_mobile_generator",
                {"generator_id": "generator_1", "target_zone": incident},
                "Critical-load service deficit exceeds the local response threshold.",
                0.71,
            )
        if role == "critical_load":
            return PlanOutput(
                "Request critical-load prioritization while retaining local autonomy.",
                "prioritize_critical_load",
                {"load_id": self.agent_id, "priority": "critical", "duration": 4},
                "Private service priority is high.",
                0.79,
            )
        return PlanOutput(
            "Continue local operation and exchange a bounded consensus sketch.",
            "request_cross_agent_consensus",
            {"topic": incident, "participants": []},
            "No stronger locally authorized action is justified.",
            0.61,
        )

    def reflect(self, step: int, plan: PlanOutput, result: ToolResult) -> None:
        self.last_plan = plan.plan_summary
        self.last_result = result.code
        self.vault.update_working(
            self.agent_id,
            {"last_tool": plan.tool, "last_result": result.code, "needs_revision": not result.ok},
        )
        self.vault.remember(
            self.agent_id,
            MemoryRecord(
                step=step,
                kind="outcome" if result.ok else "failure",
                summary="%s -> %s" % (plan.tool, result.code),
                importance=0.45 if result.ok else 0.90,
            ),
        )
