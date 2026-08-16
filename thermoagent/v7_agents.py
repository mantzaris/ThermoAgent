"""Persistent independent V7 agents with multi-asset private scopes."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .agents import PrivacyViolation
from .events import EventLedger
from .types import MemoryRecord, Message
from .v7_types import (
    COMMUNICATION_ACTIONS, DELEGATION_ACTIONS, HUMANITARIAN_ACTIONS,
    INFORMATION_ACTIONS, UTILITY_ACTIONS, V7Commitment, V7Identity,
    V7OperationalProposal, V7PrivateObservation, V7StructuredDecision,
    V7Utility,
)


HUMANITARIAN_MODE_ACTION = {
    "nominal": "no_operational_action",
    "demand_surge": "allocate_shipment",
    "route_failure": "redirect_vehicle",
    "stock_shortage": "release_emergency_reserve",
    "access_uncertain": "cancel_risky_dispatch",
    "commitment_conflict": "revise_delivery_priority",
}
UTILITY_MODE_ACTION = {
    "nominal": "no_operational_action",
    "physical_failure": "dispatch_repair_crew",
    "telemetry_corrupt": "isolate_component",
    "communication_failure": "restore_communication_relay",
    "capacity_overload": "reconfigure_service_edge",
    "cascading_risk": "deploy_mobile_generation",
}


class V7PrivateVault:
    def __init__(self, owner: str) -> None:
        self._owner = str(owner)
        self._observations: Dict[str, V7PrivateObservation] = {}
        self._memory: Deque[MemoryRecord] = deque(maxlen=512)
        self._working: Dict[str, Any] = {}

    def _authorize(self, requester: str) -> None:
        if str(requester) != self._owner:
            raise PrivacyViolation(
                "%s cannot inspect %s V7 private vault" % (requester, self._owner)
            )

    def set_observation(
        self, requester: str, observation: V7PrivateObservation,
    ) -> None:
        self._authorize(requester)
        self._observations[observation.focal_asset] = deepcopy(observation)

    def observation(self, requester: str, asset: str) -> V7PrivateObservation:
        self._authorize(requester)
        if asset not in self._observations:
            raise RuntimeError("V7 observation was not explicitly delivered")
        return deepcopy(self._observations[asset])

    def observations(self, requester: str) -> Dict[str, V7PrivateObservation]:
        self._authorize(requester)
        return deepcopy(self._observations)

    def remember(self, requester: str, record: MemoryRecord) -> None:
        self._authorize(requester)
        self._memory.append(deepcopy(record))

    def memory(self, requester: str) -> List[MemoryRecord]:
        self._authorize(requester)
        return deepcopy(list(self._memory))

    def update_working(self, requester: str, values: Mapping[str, Any]) -> None:
        self._authorize(requester)
        self._working.update(deepcopy(dict(values)))

    def working(self, requester: str) -> Dict[str, Any]:
        self._authorize(requester)
        return deepcopy(self._working)


class IndependentV7Agent:
    """A decentralized decision process without references to peer vaults."""

    def __init__(self, identity: V7Identity, utility: V7Utility, seed: int) -> None:
        self.identity = deepcopy(identity)
        self.utility = deepcopy(utility)
        self.vault = V7PrivateVault(identity.agent_id)
        self.inbox: Deque[Message] = deque()
        self.outbox: Deque[Message] = deque()
        self.commitments: Dict[str, V7Commitment] = {}
        self.private_beliefs: Dict[str, Tuple[float, ...]] = {}
        self.rng = np.random.RandomState(int(seed))

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    def deliver_observation(
        self, observation: V7PrivateObservation, ledger: EventLedger,
    ) -> None:
        if observation.focal_asset not in self.identity.asset_scope:
            raise PrivacyViolation("observation is outside the agent asset scope")
        self.vault.set_observation(self.agent_id, observation)
        evidence = np.asarray(observation.belief_distribution, dtype=float)
        evidence = np.maximum(evidence, 1e-12)
        evidence /= evidence.sum()
        previous = np.asarray(
            self.private_beliefs.get(observation.focal_asset, tuple(evidence)),
            dtype=float,
        )
        updated = 0.45 * previous + 0.55 * evidence
        updated /= updated.sum()
        self.private_beliefs[observation.focal_asset] = tuple(
            float(value) for value in updated
        )
        ledger.append(
            observation.step, "v7_private_observation", "simulator",
            {"recipient": self.agent_id, "observation": asdict(observation)},
            private_to=self.agent_id,
        )
        ledger.append(
            observation.step, "v7_belief_update", self.agent_id,
            {
                "focal_asset": observation.focal_asset,
                "belief_distribution": list(updated),
            },
            private_to=self.agent_id,
        )

    def receive(self, message: Message) -> None:
        if message.recipient not in (self.agent_id, "broadcast"):
            raise PrivacyViolation("message was delivered outside its recipient boundary")
        self.inbox.append(deepcopy(message))

    def integrate_delivered_evidence(self, message: Message) -> bool:
        """Update a local belief only from an explicitly delivered message."""
        asset = message.payload.get("target")
        values = message.payload.get("belief_distribution")
        if asset not in self.identity.asset_scope or values is None:
            return False
        evidence = np.asarray(values, dtype=float)
        if evidence.ndim != 1 or len(evidence) < 2 or not np.isfinite(evidence).all():
            return False
        evidence = np.maximum(evidence, 1e-12)
        evidence /= evidence.sum()
        previous = np.asarray(self.private_beliefs.get(str(asset), tuple(evidence)), dtype=float)
        if len(previous) != len(evidence):
            return False
        updated = 0.72 * previous + 0.28 * evidence
        updated /= updated.sum()
        self.private_beliefs[str(asset)] = tuple(float(value) for value in updated)
        return True

    def context(self, focal_asset: str) -> Dict[str, Any]:
        return {
            "identity": asdict(self.identity),
            "utility": asdict(self.utility),
            "private_observation": asdict(
                self.vault.observation(self.agent_id, focal_asset)
            ),
            "private_belief": list(self.private_beliefs[focal_asset]),
            "private_memory": [
                asdict(value) for value in self.vault.memory(self.agent_id)
            ],
            "inbox": [asdict(value) for value in self.inbox],
            "commitments": [asdict(value) for value in self.commitments.values()],
        }

    def _mode_action(self, belief: np.ndarray) -> Tuple[str, float, float]:
        mapping = (
            HUMANITARIAN_MODE_ACTION
            if self.identity.application == "humanitarian"
            else UTILITY_MODE_ACTION
        )
        modes = tuple(mapping)
        order = np.argsort(belief)[::-1]
        feasible = set(self.identity.physical_authority)
        selected = "no_operational_action"
        probability = 0.0
        for index in order:
            candidate = mapping[modes[int(index)]]
            if candidate == "no_operational_action" or candidate in feasible:
                selected = candidate
                probability = float(belief[int(index)])
                break
        ordered = np.sort(belief)[::-1]
        margin = float(ordered[0] - ordered[1])
        return selected, probability, margin

    def propose(self, focal_asset: str) -> V7StructuredDecision:
        observation = self.vault.observation(self.agent_id, focal_asset)
        belief = np.asarray(self.private_beliefs[focal_asset], dtype=float)
        action, probability, margin = self._mode_action(belief)
        feasible = set(observation.feasible_physical_actions)
        if action not in feasible:
            action = "no_operational_action"
        severity = float(observation.local_kpis.get("severity", 0.0))
        safety = float(observation.local_kpis.get("safety_risk", 0.0))
        scarcity = float(observation.local_kpis.get("resource_scarcity", 0.0))
        action_value = (
            self.utility.service_weight * severity
            + self.utility.safety_weight * safety
            + 0.25 * probability
            - self.utility.cost_weight * scarcity
            - 0.10 * len(self.commitments)
        )
        uncertain = observation.telemetry_confidence < 0.58 or margin < 0.14
        information_action = (
            "request_peer_evidence" if uncertain else "no_information_action"
        )
        communication_action = (
            "negotiate_commitment"
            if action != "no_operational_action" and len(observation.active_commitments) > 0
            else (
                "send_targeted_summary" if uncertain else "no_communication_action"
            )
        )
        if action == "no_operational_action":
            delegation = "defer" if uncertain else "execute_autonomously"
        elif action_value <= 0.0:
            delegation = "abstain"
        elif uncertain and safety > self.utility.risk_tolerance:
            delegation = "escalate_operator"
        else:
            delegation = "execute_autonomously"
        proposal = V7OperationalProposal(
            agent_id=self.agent_id,
            application=self.identity.application,
            role=self.identity.role,
            proposed_operational_action=action,
            target_asset_or_location=focal_asset,
            source_asset_or_location=None,
            commodity_or_resource=None,
            quantity_or_capacity=max(0.0, min(2.0, severity + 0.35)),
            expected_delay=max(1, int(round(observation.local_kpis.get("delay", 1.0)))),
            action_probability=probability,
            action_value=float(action_value),
            value_margin=margin,
            reason_code="private_belief_local_utility",
        )
        return V7StructuredDecision(
            proposal=proposal,
            information_action=information_action,
            communication_action=communication_action,
            delegation_action=delegation,
            confidence=float(np.clip(0.5 * probability + 0.5 * margin, 0.0, 1.0)),
            compact_plan_summary="local proposal under private evidence",
        )

    def evaluate_commitment(self, commitment: V7Commitment) -> str:
        if commitment.recipient != self.agent_id:
            raise PrivacyViolation("agent cannot evaluate a peer commitment")
        observations = self.vault.observations(self.agent_id)
        scarcity = np.mean([
            value.local_kpis.get("resource_scarcity", 0.0)
            for value in observations.values()
        ]) if observations else 1.0
        burden = float(commitment.quantity) * (0.5 + float(scarcity))
        benefit = self.utility.service_weight + self.utility.commitment_weight
        if benefit >= burden:
            return "accept"
        if benefit >= 0.6 * burden and commitment.revision < 2:
            return "counter"
        return "reject"


def role_authority(application: str, role: str) -> Tuple[str, ...]:
    if application == "humanitarian":
        actions = set(HUMANITARIAN_ACTIONS)
        if role in ("clinic", "shelter", "assessment"):
            actions -= {"redirect_vehicle", "release_emergency_reserve"}
        if role == "transport":
            actions -= {"release_emergency_reserve", "revise_delivery_priority"}
        if role == "local_authority":
            actions -= {"allocate_shipment", "redirect_vehicle"}
        return tuple(sorted(actions))
    # V7 pilot iteration 2 showed that leaving the default utility roles with
    # every physical authority generated a pool dominated by inappropriate
    # isolation proposals.  These are domain authority boundaries, not an
    # outcome-dependent action filter: the simulator still permits each
    # authorized action to help or harm depending on delayed system state.
    authorities = {
        "zone_operator": {
            "reconfigure_service_edge", "isolate_component",
            "no_operational_action",
        },
        "crew_dispatch": {
            "dispatch_repair_crew", "allocate_spare_component",
            "no_operational_action",
        },
        "cyber_defense": {
            "isolate_component", "restore_communication_relay",
            "no_operational_action",
        },
        "communications": {
            "restore_communication_relay", "reconfigure_service_edge",
            "no_operational_action",
        },
        "resource_allocation": {
            "deploy_mobile_generation", "allocate_spare_component",
            "no_operational_action",
        },
        "critical_load": {
            "deploy_mobile_generation", "no_operational_action",
        },
    }
    return tuple(sorted(authorities.get(role, set(UTILITY_ACTIONS))))
