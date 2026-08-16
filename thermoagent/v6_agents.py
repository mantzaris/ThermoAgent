"""Genuinely separate V6 agent state and typed authority boundaries."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .agents import PrivacyViolation
from .events import EventLedger
from .types import MemoryRecord, Message
from .v6_types import (
    INCIDENT_MODES, OPERATIONAL_ACTIONS, PRIMARY_ACTION_FOR_MODE,
    SECONDARY_ACTION_FOR_MODE, V6ActionProposal, V6Commitment, V6Identity,
    V6PrivateObservation, V6ToolCall, V6Utility,
)


ROLE_ACTIONS: Dict[str, Tuple[str, ...]] = {
    "supplier": ("verify", "authorize_emergency_resource", "deploy_repair_capacity", "revise_commitment", "request_peer_evidence", "defer", "no_action"),
    "carrier": ("verify", "reroute_or_reconfigure", "deploy_repair_capacity", "revise_commitment", "request_peer_evidence", "defer", "no_action"),
    "warehouse": ("verify", "authorize_emergency_resource", "reroute_or_reconfigure", "deploy_repair_capacity", "revise_commitment", "defer", "no_action"),
    "retailer": ("verify", "request_peer_evidence", "authorize_emergency_resource", "revise_commitment", "defer", "no_action"),
    "ngo": ("verify", "authorize_emergency_resource", "reroute_or_reconfigure", "request_peer_evidence", "revise_commitment", "defer", "no_action"),
    "regional_hub": ("verify", "authorize_emergency_resource", "reroute_or_reconfigure", "deploy_repair_capacity", "request_peer_evidence", "defer", "no_action"),
    "clinic": ("verify", "request_peer_evidence", "authorize_emergency_resource", "revise_commitment", "defer", "no_action"),
    "distribution_node": ("verify", "reroute_or_reconfigure", "deploy_repair_capacity", "isolate_or_quarantine", "request_peer_evidence", "defer", "no_action"),
    "field_crew": ("verify", "deploy_repair_capacity", "reroute_or_reconfigure", "isolate_or_quarantine", "request_peer_evidence", "defer", "no_action"),
    "communications": ("verify", "request_peer_evidence", "reroute_or_reconfigure", "isolate_or_quarantine", "defer", "no_action"),
    "cyber_defense": ("verify", "request_peer_evidence", "isolate_or_quarantine", "reroute_or_reconfigure", "defer", "no_action"),
    "resource_allocation": ("verify", "authorize_emergency_resource", "deploy_repair_capacity", "revise_commitment", "request_peer_evidence", "defer", "no_action"),
    "critical_load": ("verify", "request_peer_evidence", "authorize_emergency_resource", "revise_commitment", "defer", "no_action"),
}


class V6PrivateVault:
    def __init__(self, owner: str) -> None:
        self._owner = str(owner)
        self._observations: Dict[str, V6PrivateObservation] = {}
        self._memory: Deque[MemoryRecord] = deque(maxlen=256)
        self._working: Dict[str, Any] = {}

    def _authorize(self, requester: str) -> None:
        if str(requester) != self._owner:
            raise PrivacyViolation("%s cannot inspect %s V6 private state" % (requester, self._owner))

    def set_observation(self, requester: str, observation: V6PrivateObservation) -> None:
        self._authorize(requester)
        self._observations[observation.incident_id] = deepcopy(observation)

    def observation(self, requester: str, incident_id: str) -> V6PrivateObservation:
        self._authorize(requester)
        if incident_id not in self._observations:
            raise RuntimeError("private observation has not been delivered")
        return deepcopy(self._observations[incident_id])

    def memory(self, requester: str) -> List[MemoryRecord]:
        self._authorize(requester)
        return deepcopy(list(self._memory))

    def remember(self, requester: str, record: MemoryRecord) -> None:
        self._authorize(requester)
        self._memory.append(deepcopy(record))

    def working(self, requester: str) -> Dict[str, Any]:
        self._authorize(requester)
        return deepcopy(self._working)

    def update(self, requester: str, values: Mapping[str, Any]) -> None:
        self._authorize(requester)
        self._working.update(deepcopy(dict(values)))


class V6ToolRegistry:
    PHYSICAL_ACTIONS = (
        "authorize_emergency_resource", "reroute_or_reconfigure",
        "deploy_repair_capacity", "isolate_or_quarantine", "revise_commitment",
    )

    def allowed_actions(self, role: str) -> Tuple[str, ...]:
        if role not in ROLE_ACTIONS:
            raise ValueError("unknown V6 role: %s" % role)
        return ROLE_ACTIONS[role]

    def action_mask(self, role: str) -> np.ndarray:
        allowed = set(self.allowed_actions(role))
        return np.asarray([action in allowed for action in OPERATIONAL_ACTIONS], dtype=bool)

    def validate(
        self, identity: V6Identity, call: V6ToolCall,
    ) -> Tuple[bool, str, Optional[V6ToolCall]]:
        if call.action not in OPERATIONAL_ACTIONS or call.action not in self.allowed_actions(identity.role):
            return False, "action_not_permitted", None
        if call.incident_id not in identity.incident_scope:
            return False, "incident_outside_private_scope", None
        if not np.isfinite(call.quantity) or not 0.0 <= float(call.quantity) <= 2.0:
            return False, "quantity_out_of_bounds", None
        if len(call.reason_code) < 1 or len(call.reason_code) > 96:
            return False, "reason_out_of_bounds", None
        if call.target_agent is not None and len(call.target_agent) > 96:
            return False, "target_out_of_bounds", None
        return True, "validated", deepcopy(call)


class IndependentV6Agent:
    """Persistent decentralized agent with no reference to peer private vaults."""

    def __init__(self, identity: V6Identity, utility: V6Utility, seed: int) -> None:
        self.identity = deepcopy(identity)
        self.utility = deepcopy(utility)
        self.vault = V6PrivateVault(identity.agent_id)
        self.inbox: Deque[Message] = deque()
        self.outbox: Deque[Message] = deque()
        self.commitments: Dict[str, V6Commitment] = {}
        self.private_beliefs: Dict[str, Tuple[float, ...]] = {}
        self.rng = np.random.RandomState(int(seed))

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    def deliver(self, observation: V6PrivateObservation, ledger: EventLedger) -> None:
        self.vault.set_observation(self.agent_id, observation)
        evidence = np.maximum(np.asarray(observation.private_evidence, dtype=float), 1e-12)
        evidence /= evidence.sum()
        old = np.asarray(self.private_beliefs.get(observation.incident_id, evidence), dtype=float)
        # Persistent private belief update; old information has bounded weight.
        updated = 0.30 * old + 0.70 * evidence
        updated /= updated.sum()
        self.private_beliefs[observation.incident_id] = tuple(float(value) for value in updated)
        ledger.append(
            observation.step, "v6_private_observation", "simulator",
            {"recipient": self.agent_id, "observation": asdict(observation)},
            private_to=self.agent_id,
        )
        ledger.append(
            observation.step, "v6_belief_update", self.agent_id,
            {"incident_id": observation.incident_id, "belief_distribution": list(updated)},
            private_to=self.agent_id,
        )

    def receive(self, message: Message) -> None:
        if message.recipient not in (self.agent_id, "broadcast"):
            raise PrivacyViolation("message crossed the V6 private authority boundary")
        self.inbox.append(deepcopy(message))

    def context(self, incident_id: str) -> Dict[str, Any]:
        return {
            "identity": asdict(self.identity),
            "utility": asdict(self.utility),
            "private_observation": asdict(self.vault.observation(self.agent_id, incident_id)),
            "private_memory": [asdict(value) for value in self.vault.memory(self.agent_id)],
            "private_belief": list(self.private_beliefs[incident_id]),
            "inbox": [asdict(value) for value in self.inbox],
            "commitments": [asdict(value) for value in self.commitments.values()],
        }

    def action_mask(self, registry: V6ToolRegistry) -> np.ndarray:
        return registry.action_mask(self.identity.role)

    def propose(self, incident_id: str, registry: V6ToolRegistry) -> V6ActionProposal:
        observation = self.vault.observation(self.agent_id, incident_id)
        belief = np.asarray(self.private_beliefs[incident_id], dtype=float)
        order = np.argsort(belief)[::-1]
        allowed = set(registry.allowed_actions(self.identity.role))
        # Level 1 proposes a consequential operational action. Verification,
        # communication, deferral, and abstention belong to the Level 2
        # delegation controller and must not become trivially identifiable
        # "unsafe action" labels in the primary selective-risk task.
        action = "no_action"
        chosen_probability = 0.0
        mode_index = int(order[0])
        # A high-probability nominal belief is a valid decentralized decision,
        # not a missing action.  It must remain available so false escalation
        # can be measured before disruption and throughout nominal episodes.
        if INCIDENT_MODES[mode_index] == "nominal":
            return V6ActionProposal(
                agent_id=self.agent_id,
                role=self.identity.role,
                incident_id=incident_id,
                action="no_action",
                quantity=0.0,
                action_probability=float(belief[mode_index]),
                action_value=0.0,
                value_margin=float(np.sort(belief)[::-1][0] - np.sort(belief)[::-1][1]),
                reason_code="private_belief_nominal",
            )
        for candidate_index in order:
            mode = INCIDENT_MODES[int(candidate_index)]
            if mode in ("observation_ambiguity", "nominal"):
                continue
            primary = PRIMARY_ACTION_FOR_MODE[mode]
            secondary = SECONDARY_ACTION_FOR_MODE[mode]
            if primary in allowed and primary in V6ToolRegistry.PHYSICAL_ACTIONS:
                action = primary
            elif secondary in allowed and secondary in V6ToolRegistry.PHYSICAL_ACTIONS:
                action = secondary
            if action != "no_action":
                mode_index = int(candidate_index)
                chosen_probability = float(belief[mode_index])
                break
        top = np.sort(belief)[::-1]
        margin = float(top[0] - top[1])
        action_value = float(
            self.utility.service_weight * observation.visible_severity
            + self.utility.safety_weight * observation.safety_risk
            + 0.25 * chosen_probability
            - self.utility.cost_weight * observation.private_cost
            - self.utility.delay_weight * observation.visible_delay
        )
        return V6ActionProposal(
            agent_id=self.agent_id,
            role=self.identity.role,
            incident_id=incident_id,
            action=action,
            quantity=1.0,
            action_probability=chosen_probability,
            action_value=action_value,
            value_margin=margin,
            reason_code="private_belief_and_utility",
        )

    def select_offer(self, offers: Sequence[V6ActionProposal]) -> V6ActionProposal:
        """Choose among explicitly received peer proposals; no peer vault access."""
        if not offers:
            raise ValueError("an agent cannot select from an empty offer set")
        return max(
            offers,
            key=lambda value: (
                value.action_value - self.utility.disclosure_cost,
                value.value_margin,
                value.agent_id,
            ),
        )

    def evaluate_commitment(self, commitment: V6Commitment, incident_id: str) -> str:
        observation = self.vault.observation(self.agent_id, incident_id)
        value = (
            self.utility.service_weight * observation.visible_severity
            + self.utility.safety_weight * observation.safety_risk
            - self.utility.cost_weight * observation.private_cost
        )
        burden = commitment.quantity / max(observation.private_inventory, 0.2)
        if value >= 0.62 * burden:
            return "accept"
        if value >= 0.34 * burden and commitment.revision < 2:
            return "counter"
        return "reject"
