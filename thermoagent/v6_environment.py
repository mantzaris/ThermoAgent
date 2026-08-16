"""Dynamic replayable V6 selective-autonomy environment.

Cyber-physical events are abstract defensive simulator states.  This module
does not contain real protocols, targets, credentials, exploits, or deployable
attack procedures.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .events import EventLedger
from .types import Message
from .v6_agents import IndependentV6Agent, V6ToolRegistry
from .v6_entropy import (
    average_local_uncertainty, consensus_score, generalized_disagreement,
    gini_simpson_impurity, graph_weighted_disagreement, shannon_entropy,
    temporal_information_state, tsallis_entropy, weighted_pooled_belief,
)
from .v6_types import (
    APPLICATIONS, INCIDENT_MODES, INFORMATION_CONDITIONS, OPERATIONAL_ACTIONS,
    PRIMARY_ACTION_FOR_MODE, REGIMES, SECONDARY_ACTION_FOR_MODE, SKETCH_POLICIES,
    ResourceAccount, V6ActionProposal, V6ActionResult, V6DecisionContext,
    V6Commitment, V6Identity, V6Incident, V6PrivateObservation, V6ToolCall,
    V6Utility,
)


APP_ROLES: Dict[str, Tuple[str, ...]] = {
    "commercial": ("supplier", "carrier", "warehouse", "retailer"),
    "humanitarian": ("ngo", "regional_hub", "clinic"),
    "utility_restoration": (
        "distribution_node", "field_crew", "communications", "cyber_defense",
        "resource_allocation", "critical_load",
    ),
}

RESOURCE_FOR_ACTION = {
    "verify": "verification_capacity",
    "authorize_emergency_resource": "emergency_units",
    "reroute_or_reconfigure": "route_authorizations",
    "deploy_repair_capacity": "crew_units",
    "isolate_or_quarantine": "isolation_authorizations",
}

PHYSICAL_ACTIONS = {
    "authorize_emergency_resource", "reroute_or_reconfigure",
    "deploy_repair_capacity", "isolate_or_quarantine", "revise_commitment",
}


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(value) for value in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def payload_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


class V6PanelEnvironment:
    """A panel of concurrent incidents and genuinely separate organizations."""

    # Step zero supplies the prospective pre-disruption decision needed for a
    # real false-activation test. Disruptions begin at step two.
    decision_steps = (0, 2, 4, 6, 8, 10)

    def __init__(
        self,
        application: str,
        regime: str,
        information_condition: str,
        seed: int,
        sketch_policy: str = "event_triggered",
        horizon: int = 12,
        incident_count: int = 4,
    ) -> None:
        if application not in APPLICATIONS:
            raise ValueError("unknown V6 application")
        if regime not in REGIMES:
            raise ValueError("unknown V6 regime")
        if information_condition not in INFORMATION_CONDITIONS:
            raise ValueError("unknown V6 information condition")
        if sketch_policy not in SKETCH_POLICIES:
            raise ValueError("unknown V6 sketch policy")
        if horizon < 8 or incident_count < 3:
            raise ValueError("V6 requires a dynamic horizon and competitive incidents")
        self.application = application
        self.regime = regime
        self.information_condition = information_condition
        self.seed = int(seed)
        self.sketch_policy = sketch_policy
        self.horizon = int(horizon)
        self.incident_count = int(incident_count)
        self.split_family = "%s_v6_split_%d" % (application, self.seed % 5)
        self.topology_family = "%s_v6_topology_%d" % (application, self.seed % 5)
        self.scenario_family = "%s_%s_family_%d" % (application, regime, self.seed % 5)
        self.cluster_id = "%s|%s|%s|%d" % (
            application, regime, information_condition, self.seed,
        )
        # Match the exogenous scenario and stochastic tape across information
        # conditions. The treatment changes only evidence sharing, never the
        # incidents, utilities, or action-outcome randomness.
        self.rng = np.random.RandomState(stable_seed("v6", application, regime, seed))
        self.ledger = EventLedger()
        self.registry = V6ToolRegistry()
        self.current_step = 0
        self.operational_messages = 0
        self.operational_bytes = 0
        self.sketch_messages = 0
        self.sketch_bytes = 0
        self.sketch_latency = 0.0
        self.operator_minutes = 0.0
        self.operator_queue: List[Dict[str, Any]] = []
        self.maximum_queue_length = 0
        self.operator_budget_remaining = 4
        self.operator_busy_until = 0
        self.pending_actions: List[Dict[str, Any]] = []
        self.action_records: List[Dict[str, Any]] = []
        self.candidate_records: List[Dict[str, Any]] = []
        self.delegation_records: List[Dict[str, Any]] = []
        self.consensus_records: List[Dict[str, Any]] = []
        self.information_history: Dict[str, Dict[str, List[float]]] = {}
        self.information_recorded_step: Dict[str, int] = {}
        self.sketch_cache: Dict[str, Dict[str, Tuple[Tuple[float, ...], int, float]]] = {}
        self.last_sent_belief: Dict[str, np.ndarray] = {}
        self.last_sent_step: Dict[str, int] = {}
        self.resources = {
            "verification_capacity": ResourceAccount(8.0, 8.0),
            "emergency_units": ResourceAccount(6.0, 6.0),
            "route_authorizations": ResourceAccount(7.0, 7.0),
            "crew_units": ResourceAccount(6.0, 6.0),
            "isolation_authorizations": ResourceAccount(5.0, 5.0),
        }
        self._tape = self._make_stochastic_tape()
        self.stochastic_tape_digest = payload_digest(self._tape)
        self.incidents = self._make_incidents()
        self.agents, self.incident_agents = self._make_agents()
        self.communication_edges = self._make_communication_edges()
        self._initialize_private_state()
        self._initialize_commitments()
        self.ledger.append(
            0, "v6_stochastic_tape", "simulator",
            {"digest": self.stochastic_tape_digest, "seed": self.seed},
        )
        self.ledger.append(
            0, "v6_panel_snapshot", "simulator",
            {
                "cluster_id": self.cluster_id,
                "application": application,
                "regime": regime,
                "information_condition": information_condition,
                "topology_family": self.topology_family,
                "scenario_family": self.scenario_family,
                "split_family": self.split_family,
                "incident_count": self.incident_count,
                "agent_ids": sorted(self.agents),
                "resource_initial": {key: value.initial for key, value in self.resources.items()},
            },
        )

    def _make_stochastic_tape(self) -> Dict[str, Any]:
        count_agents = self.incident_count * 3
        return {
            "observation": self.rng.uniform(0.0, 1.0, size=(self.horizon, count_agents, len(INCIDENT_MODES))).round(12).tolist(),
            "action": self.rng.uniform(0.0, 1.0, size=(self.horizon, self.incident_count, len(OPERATIONAL_ACTIONS))).round(12).tolist(),
            "verification": self.rng.uniform(0.0, 1.0, size=(self.horizon, self.incident_count, 3)).round(12).tolist(),
            "service": self.rng.uniform(0.82, 1.18, size=(self.horizon, self.incident_count)).round(12).tolist(),
            "operator": self.rng.uniform(0.0, 1.0, size=(self.horizon, self.incident_count)).round(12).tolist(),
        }

    def _make_incidents(self) -> Dict[str, V6Incident]:
        incidents: Dict[str, V6Incident] = {}
        regime_severity = {
            "nominal": 0.25, "isolated_physical": 0.48,
            "telemetry_integrity": 0.50, "partition": 0.52,
            "correlated": 0.64, "compound": 0.74, "ood": 0.78,
        }[self.regime]
        for index in range(self.incident_count):
            incident_id = "%s_incident_%02d" % (self.application, index + 1)
            non_nominal_modes = tuple(
                value for value in INCIDENT_MODES if value != "nominal"
            )
            mode_index = int(self.rng.randint(0, len(non_nominal_modes)))
            # Utility scenarios contain a somewhat higher prevalence of
            # integrity conflicts, but no observed feature deterministically
            # reveals the mode or correct action.
            if self.application == "utility_restoration" and self.regime in (
                "telemetry_integrity", "compound", "ood",
            ) and self.rng.uniform() < 0.35:
                mode_index = int(self.rng.choice([0, 4]))
            true_mode = (
                "nominal" if self.regime == "nominal"
                else non_nominal_modes[mode_index]
            )
            severity = float(np.clip(regime_severity + self.rng.normal(0.0, 0.14), 0.16, 0.96))
            priority = float(np.clip(self.rng.beta(2.4, 1.9), 0.18, 0.98))
            # Evidence difficulty belongs to the matched incident. A broad
            # distribution supplies coherent and fragmented cases at similar
            # KPI severity. Public sharing changes belief coherence without
            # making the underlying intervention inert.
            fragmentation_base = {
                "nominal": 0.18,
                "isolated_physical": 0.30,
                "telemetry_integrity": 0.52,
                "partition": 0.48,
                "correlated": 0.43,
                "compound": 0.60,
                "ood": 0.68,
            }[self.regime]
            fragmentation = float(np.clip(
                fragmentation_base + self.rng.normal(0.0, 0.23), 0.02, 0.96,
            ))
            integrity = float(np.clip(1.0 - 0.62 * fragmentation + self.rng.normal(0.0, 0.12), 0.08, 0.98))
            incidents[incident_id] = V6Incident(
                incident_id=incident_id,
                application=self.application,
                regime=self.regime,
                scenario_family="%s_incident_mode_mix_%d" % (self.scenario_family, index % 3),
                topology_family=self.topology_family,
                true_mode=true_mode,
                correct_action=PRIMARY_ACTION_FOR_MODE[true_mode],
                secondary_action=SECONDARY_ACTION_FOR_MODE[true_mode],
                severity=severity,
                priority=priority,
                fragmentation=fragmentation,
                telemetry_integrity=integrity,
                disruption_step=2,
            )
        return incidents

    def _make_agents(self) -> Tuple[Dict[str, IndependentV6Agent], Dict[str, Tuple[str, ...]]]:
        agents: Dict[str, IndependentV6Agent] = {}
        incident_agents: Dict[str, Tuple[str, ...]] = {}
        roles = APP_ROLES[self.application]
        for incident_index, incident_id in enumerate(sorted(self.incidents)):
            ids: List[str] = []
            for local_index in range(3):
                role = roles[(incident_index + local_index) % len(roles)]
                agent_id = "%s_%s_%02d_%d" % (self.application, role, incident_index + 1, local_index + 1)
                identity = V6Identity(
                    agent_id=agent_id,
                    application=self.application,
                    role=role,
                    incident_scope=(incident_id,),
                    authority=self.registry.allowed_actions(role),
                )
                utility = V6Utility(
                    service_weight=float(0.75 + 0.35 * self.rng.uniform()),
                    safety_weight=float(0.55 + 0.55 * self.rng.uniform()),
                    cost_weight=float(0.32 + 0.48 * self.rng.uniform()),
                    delay_weight=float(0.22 + 0.38 * self.rng.uniform()),
                    disclosure_cost=float(0.03 + 0.12 * self.rng.uniform()),
                    risk_tolerance=float(0.25 + 0.60 * self.rng.uniform()),
                )
                agents[agent_id] = IndependentV6Agent(
                    identity, utility, stable_seed(self.seed, agent_id),
                )
                ids.append(agent_id)
            incident_agents[incident_id] = tuple(ids)
        return agents, incident_agents

    def _make_communication_edges(self) -> Dict[Tuple[str, str], float]:
        edges: Dict[Tuple[str, str], float] = {}
        for agent_ids in self.incident_agents.values():
            for first in agent_ids:
                for second in agent_ids:
                    if first != second:
                        edges[(first, second)] = float(0.72 + 0.25 * self.rng.uniform())
        # Sparse cross-incident ad-hoc links provide propagation without a
        # hidden central coordinator.
        leads = [ids[0] for ids in self.incident_agents.values()]
        for index, first in enumerate(leads):
            second = leads[(index + 1) % len(leads)]
            edges[(first, second)] = edges[(second, first)] = float(0.62 + 0.28 * self.rng.uniform())
        return edges

    def _edge_available(self, first: str, second: str, step: int) -> bool:
        if (first, second) not in self.communication_edges:
            return False
        if self.regime == "partition" and step >= 2:
            # The local-agent suffix makes every incident span both sides of
            # the deterministic partition; this prevents a nominal
            # "partition" episode from accidentally retaining a full clique.
            first_group = int(first.rsplit("_", 1)[-1]) % 2
            second_group = int(second.rsplit("_", 1)[-1]) % 2
            return first_group == second_group
        if self.regime in ("compound", "ood") and step >= 2:
            return stable_seed(self.seed, first, second, step // 3) % 5 != 0
        return True

    def _initial_belief(self, incident: V6Incident, agent_index: int, step: int) -> np.ndarray:
        incident_index = list(sorted(self.incidents)).index(incident.incident_id)
        global_agent_index = incident_index * 3 + agent_index
        noise = np.asarray(self._tape["observation"][step][global_agent_index], dtype=float)
        nominal_index = INCIDENT_MODES.index("nominal")
        # Before the registered disruption onset, no agent receives a clue
        # about the future incident mode. Nominal episodes remain nominal for
        # their full horizon. Small private tail variation prevents a constant
        # feature block while preserving a meaningful no-alert state.
        if step < incident.disruption_step or incident.true_mode == "nominal":
            common_noise = np.asarray(
                self._tape["observation"][step][incident_index * 3], dtype=float,
            )
            # Rare, incident-level ambiguous observations make nominal and
            # pre-disruption false-alert rates empirically testable. They do
            # not encode the future mode and occur with a fixed 2.5% rate
            # chosen before formal development.
            if common_noise[0] < 0.025:
                nominal_mass = float(0.18 + 0.08 * common_noise[1])
            else:
                nominal_mass = float(0.70 + 0.19 * noise[0])
            tail = np.maximum(noise + 0.05, 1e-6)
            tail[nominal_index] = 0.0
            tail /= tail.sum()
            belief = (1.0 - nominal_mass) * tail
            belief[nominal_index] = nominal_mass
            return belief / belief.sum()
        signal_noise = noise
        if self.information_condition == "public_shared":
            # Public agents receive the same incident signal (with small
            # private tail variation). The shared signal can still be wrong,
            # so public interventions remain consequential rather than inert.
            signal_noise = np.asarray(
                self._tape["observation"][step][incident_index * 3], dtype=float,
            )
        true_index = INCIDENT_MODES.index(incident.true_mode)
        corruption = 0.06 + 0.42 * incident.fragmentation + 0.20 * (1.0 - incident.telemetry_integrity)
        if self.information_condition == "public_shared":
            corruption *= 0.30
        corrupted = signal_noise[0] < corruption
        signal_index = true_index
        if corrupted:
            alternatives = [value for value in range(len(INCIDENT_MODES)) if value != true_index]
            signal_index = alternatives[int(signal_noise[1] * len(alternatives)) % len(alternatives)]
        # Make the private signal the modal belief while retaining continuous,
        # overlapping tail mass. Fragmentation governs whether agents receive
        # conflicting signals, not a one-feature encoding of the true label.
        if signal_noise[3] < 0.34:
            signal_mass = float(0.34 + 0.13 * signal_noise[2])
        else:
            signal_mass = float(0.62 + 0.20 * signal_noise[2])
        tail = np.maximum(noise + 0.08, 1e-6)
        tail[signal_index] = 0.0
        tail /= tail.sum()
        belief = (1.0 - signal_mass) * tail
        belief[signal_index] = signal_mass
        belief /= belief.sum()
        return belief

    def _observation(self, incident: V6Incident, agent_index: int, step: int) -> V6PrivateObservation:
        belief = self._initial_belief(incident, agent_index, step)
        agent_id = self.incident_agents[incident.incident_id][agent_index]
        utility = self.agents[agent_id].utility
        visible_noise = float(belief.max() - belief.min())
        # KPI fields overlap across coherent and conflicting incidents and do
        # not encode the true action. In the public condition, shared evidence
        # improves local action confidence without making interventions inert.
        severity = float(np.clip(incident.severity + 0.08 * (visible_noise - 0.35), 0.0, 1.0))
        shortage_index = INCIDENT_MODES.index("resource_shortage")
        route_index = INCIDENT_MODES.index("route_or_service_failure")
        unsafe_index = INCIDENT_MODES.index("unsafe_or_compromised_component")
        deadlock_index = INCIDENT_MODES.index("commitment_deadlock")
        backlog = float(np.clip(incident.backlog + 0.10 * belief[shortage_index], 0.0, 1.4))
        delay = float(np.clip(0.18 + 0.55 * incident.service_deficit + 0.12 * belief[route_index], 0.0, 1.0))
        scarcity = float(np.clip(0.20 + 0.50 * (1.0 - self.resources["emergency_units"].remaining / self.resources["emergency_units"].initial) + 0.12 * belief[shortage_index], 0.0, 1.0))
        safety = float(np.clip(0.16 + 0.62 * incident.service_deficit + 0.10 * belief[unsafe_index], 0.0, 1.0))
        commitment = float(np.clip(0.12 + 0.44 * incident.backlog + 0.12 * belief[deadlock_index], 0.0, 1.0))
        return V6PrivateObservation(
            step=step,
            incident_id=incident.incident_id,
            visible_severity=severity,
            visible_backlog=backlog,
            visible_delay=delay,
            resource_scarcity=scarcity,
            safety_risk=safety,
            commitment_strain=commitment,
            # Apparent telemetry confidence is locally observable and may be
            # confidently wrong under integrity loss. True integrity remains
            # evaluator-only, making peer disagreement potentially useful.
            telemetry_confidence=float(np.clip(0.22 + 0.86 * belief.max(), 0.05, 0.99)),
            communication_reliability=float(np.mean([
                value for (first, _), value in self.communication_edges.items() if first == agent_id
            ]) if any(first == agent_id for first, _ in self.communication_edges) else 0.0),
            private_evidence=tuple(float(value) for value in belief),
            private_inventory=float(0.7 + 1.2 * (1.0 - utility.cost_weight)),
            private_cost=float(np.clip(utility.cost_weight + 0.10 * belief[shortage_index], 0.0, 1.0)),
            private_priority=float(np.clip(incident.priority * utility.service_weight, 0.0, 1.2)),
        )

    def _initialize_private_state(self) -> None:
        for incident_id, agent_ids in self.incident_agents.items():
            self.information_history[incident_id] = {"entropy": [], "disagreement": [], "consensus": []}
            self.information_recorded_step[incident_id] = -1
            for agent_id in agent_ids:
                self.sketch_cache[agent_id] = {}
            for index, agent_id in enumerate(agent_ids):
                self.agents[agent_id].deliver(self._observation(self.incidents[incident_id], index, 0), self.ledger)

    def _initialize_commitments(self) -> None:
        """Create explicit bilateral offers with autonomous accept/counter/reject."""
        for incident_index, (incident_id, agent_ids) in enumerate(sorted(self.incident_agents.items())):
            proposer, recipient = agent_ids[:2]
            commitment = V6Commitment(
                commitment_id="V6C-%s-%02d" % (self.seed, incident_index + 1),
                proposer=proposer,
                recipient=recipient,
                incident_id=incident_id,
                action="revise_commitment",
                quantity=float(0.55 + 0.20 * self.incidents[incident_id].priority),
            )
            self.ledger.append(0, "offer", proposer, {
                "commitment": asdict(commitment), "recipient": recipient,
                "private_terms_not_disclosed": True,
            })
            decision = self.agents[recipient].evaluate_commitment(commitment, incident_id)
            if decision == "counter":
                commitment.quantity *= 0.72
                commitment.revision += 1
                self.ledger.append(0, "counteroffer", recipient, {
                    "commitment": asdict(commitment), "recipient": proposer,
                })
                decision = self.agents[proposer].evaluate_commitment(commitment, incident_id)
            commitment.status = "accepted" if decision == "accept" else "rejected"
            self.agents[proposer].commitments[commitment.commitment_id] = deepcopy(commitment)
            self.agents[recipient].commitments[commitment.commitment_id] = deepcopy(commitment)
            self.ledger.append(0, "commitment", recipient, {
                "commitment_id": commitment.commitment_id,
                "proposer": proposer,
                "recipient": recipient,
                "status": commitment.status,
                "revision": commitment.revision,
            })

    def deliver_observations(self, step: int) -> None:
        for incident_id, agent_ids in self.incident_agents.items():
            incident = self.incidents[incident_id]
            for index, agent_id in enumerate(agent_ids):
                self.agents[agent_id].deliver(self._observation(incident, index, step), self.ledger)

    def _sketch_due(self, agent_id: str, belief: np.ndarray, step: int) -> bool:
        if self.sketch_policy == "none":
            return False
        if self.sketch_policy == "always_on":
            return True
        if self.sketch_policy == "periodic":
            return step % 3 == 0
        prior = self.last_sent_belief.get(agent_id)
        prior_step = self.last_sent_step.get(agent_id, -100)
        # Event sketches are limited to one transmission every two simulator
        # steps. A total-variation change above 0.35 (L1 > 0.70) is the
        # interpretable information threshold; disruption onset is always
        # sampled. This threshold was selected from the retained sketch-cost
        # pilot before formal development and is not tuned on safety outcomes.
        return bool(
            step in (0, 2)
            or prior is None
            or (step - prior_step >= 2 and float(np.abs(belief - prior).sum()) >= 0.70)
        )

    def exchange_sketches(self, incident_id: str, step: int) -> None:
        agent_ids = self.incident_agents[incident_id]
        for sender in agent_ids:
            belief = np.asarray(self.agents[sender].private_beliefs[incident_id], dtype=float)
            if not self._sketch_due(sender, belief, step):
                continue
            self.last_sent_belief[sender] = belief.copy()
            self.last_sent_step[sender] = step
            observation = self.agents[sender].vault.observation(sender, incident_id)
            for recipient in agent_ids:
                if sender == recipient or not self._edge_available(sender, recipient, step):
                    continue
                reliability = self.communication_edges[(sender, recipient)]
                payload_bytes = 58 + 6 * len(belief)
                self.sketch_cache[recipient][sender] = (
                    tuple(float(value) for value in belief), step,
                    float(observation.telemetry_confidence * reliability),
                )
                self.sketch_messages += 1
                self.sketch_bytes += payload_bytes
                self.sketch_latency += 0.0015 + 0.003 * (1.0 - reliability)
                self.ledger.append(
                    step, "v6_sketch", sender,
                    {
                        "recipient": recipient,
                        "incident_id": incident_id,
                        "belief_summary": list(belief),
                        "telemetry_confidence": observation.telemetry_confidence,
                        "reliability": reliability,
                        "bytes": payload_bytes,
                    },
                )

    def information_state(self, incident_id: str, recipient: str, step: int) -> Dict[str, Any]:
        own = np.asarray(self.agents[recipient].private_beliefs[incident_id], dtype=float)
        beliefs: List[np.ndarray] = [own]
        contributors = [recipient]
        own_observation = self.agents[recipient].vault.observation(recipient, incident_id)
        weights = [max(0.05, own_observation.telemetry_confidence)]
        ages: List[float] = [0.0]
        for sender, (belief, sent_step, confidence) in sorted(self.sketch_cache[recipient].items()):
            age = max(0, step - sent_step)
            age_weight = math.exp(-age / 3.0)
            beliefs.append(np.asarray(belief, dtype=float))
            contributors.append(sender)
            weights.append(max(0.01, confidence * age_weight))
            ages.append(float(age))
        missing = tuple(sorted(set(self.incident_agents[incident_id]) - set(contributors)))
        pooled = weighted_pooled_belief(beliefs, weights)
        js = generalized_disagreement(beliefs, weights, 1.0)
        graph_edges: List[Tuple[str, str, float]] = []
        belief_map = {agent: belief for agent, belief in zip(contributors, beliefs)}
        for first in contributors:
            for second in contributors:
                if first < second and self._edge_available(first, second, step):
                    graph_edges.append((first, second, self.communication_edges.get((first, second), 0.0)))
        graph = graph_weighted_disagreement(belief_map, graph_edges, 1.0)
        history = self.information_history[incident_id]
        entropy_value = shannon_entropy(pooled)
        consensus = consensus_score(beliefs, weights, 1.0)
        if self.information_recorded_step[incident_id] != step:
            history["entropy"].append(entropy_value)
            history["disagreement"].append(js)
            history["consensus"].append(consensus)
            self.information_recorded_step[incident_id] = step
        entropy_temporal = temporal_information_state(history["entropy"] or [entropy_value], 0.60)
        disagreement_temporal = temporal_information_state(history["disagreement"] or [js], 0.22)
        consensus_temporal = temporal_information_state(history["consensus"] or [consensus], 0.55)
        global_beliefs = [
            np.asarray(self.agents[value].private_beliefs[incident_id], dtype=float)
            for value in self.incident_agents[incident_id]
        ]
        global_weights = [
            self.agents[value].vault.observation(value, incident_id).telemetry_confidence
            for value in self.incident_agents[incident_id]
        ]
        global_pooled = weighted_pooled_belief(global_beliefs, global_weights)
        return {
            "pooled_belief": pooled,
            "average_local_uncertainty": average_local_uncertainty(beliefs, weights, 1.0),
            "pooled_uncertainty": entropy_value,
            "js_disagreement": js,
            "jt_disagreement_0_5": generalized_disagreement(beliefs, weights, 0.5),
            "jt_disagreement_1_5": generalized_disagreement(beliefs, weights, 1.5),
            "jt_disagreement_2": generalized_disagreement(beliefs, weights, 2.0),
            "jt_disagreement_3": generalized_disagreement(beliefs, weights, 3.0),
            "graph_disagreement": graph,
            "consensus": consensus,
            "consensus_residual": float(np.mean(ages) / max(self.horizon, 1) + len(missing) / len(self.incident_agents[incident_id])),
            "entropy_slope": entropy_temporal.slope,
            "entropy_acceleration": entropy_temporal.acceleration,
            "entropy_ewma": entropy_temporal.ewma,
            "entropy_time_above": entropy_temporal.time_above_threshold,
            "disagreement_slope": disagreement_temporal.slope,
            "disagreement_acceleration": disagreement_temporal.acceleration,
            "disagreement_ewma": disagreement_temporal.ewma,
            "disagreement_time_above": disagreement_temporal.time_above_threshold,
            "consensus_slope": consensus_temporal.slope,
            "consensus_ewma": consensus_temporal.ewma,
            "contributors": tuple(contributors),
            "missing_agents": missing,
            # Evaluator-only diagnostic; never returned in operator payloads.
            "evaluator_distributed_error": float(np.abs(pooled - global_pooled).sum() / 2.0),
        }

    def decision_context(
        self,
        incident_id: str,
        step: int,
        offers_override: Optional[Sequence[V6ActionProposal]] = None,
        lead_override: Optional[str] = None,
    ) -> V6DecisionContext:
        agent_ids = self.incident_agents[incident_id]
        lead = lead_override or agent_ids[(step // 2) % len(agent_ids)]
        if lead not in agent_ids:
            raise ValueError("lead is outside the incident authority scope")
        if offers_override is None:
            offers = [self.agents[lead].propose(incident_id, self.registry)]
            for sender in agent_ids:
                if sender == lead or not self._edge_available(sender, lead, step):
                    continue
                offer = self.agents[sender].propose(incident_id, self.registry)
                offers.append(offer)
                self.operational_messages += 1
                self.operational_bytes += 148
                self.ledger.append(step, "offer", sender, {
                    "recipient": lead,
                    "incident_id": incident_id,
                    "action": offer.action,
                    "action_value": offer.action_value,
                    "value_margin": offer.value_margin,
                    "private_belief_not_disclosed": True,
                })
        else:
            offers = list(offers_override)
            if not offers:
                raise ValueError("offer override cannot be empty")
            for offer in offers:
                if offer.agent_id not in agent_ids or offer.incident_id != incident_id:
                    raise ValueError("offer override crossed an authority boundary")
        proposal = self.agents[lead].select_offer(offers)
        # The selected actor remains the authority holder; the lead does not
        # acquire its private observation or execute on its behalf.
        observation = self.agents[proposal.agent_id].vault.observation(proposal.agent_id, incident_id)
        state = self.information_state(incident_id, lead, step)
        local_belief = self.agents[proposal.agent_id].private_beliefs[incident_id]
        energy = float(np.clip(
            0.31 * observation.visible_severity
            + 0.22 * observation.visible_backlog
            + 0.17 * observation.visible_delay
            + 0.14 * observation.resource_scarcity
            + 0.10 * observation.safety_risk
            + 0.06 * observation.commitment_strain,
            0.0, 1.2,
        ))
        # Dimensionless effective decision temperature: locally perceived
        # volatility/urgency, not a physical temperature. Free energy remains
        # an exploratory diagnostic and is excluded from primary controllers.
        effective_temperature = float(np.clip(
            0.25 + 0.375 * observation.visible_delay
            + 0.375 * observation.safety_risk,
            0.25, 1.0,
        ))
        free_energy = float(
            energy - effective_temperature * state["pooled_uncertainty"]
        )
        context = V6DecisionContext(
            step=step,
            proposal=proposal,
            local_kpis=observation.kpis(),
            operational_energy=energy,
            effective_temperature=effective_temperature,
            free_energy_diagnostic=free_energy,
            shannon_local=shannon_entropy(local_belief),
            tsallis_0_5_local=tsallis_entropy(local_belief, 0.5),
            tsallis_1_5_local=tsallis_entropy(local_belief, 1.5),
            tsallis_2_local=tsallis_entropy(local_belief, 2.0),
            tsallis_3_local=tsallis_entropy(local_belief, 3.0),
            gini_simpson_local=gini_simpson_impurity(local_belief),
            average_local_uncertainty=state["average_local_uncertainty"],
            pooled_uncertainty=state["pooled_uncertainty"],
            js_disagreement=state["js_disagreement"],
            jt_disagreement_0_5=state["jt_disagreement_0_5"],
            jt_disagreement_1_5=state["jt_disagreement_1_5"],
            jt_disagreement_2=state["jt_disagreement_2"],
            jt_disagreement_3=state["jt_disagreement_3"],
            graph_disagreement=state["graph_disagreement"],
            consensus=state["consensus"],
            consensus_residual=state["consensus_residual"],
            entropy_slope=state["entropy_slope"],
            entropy_acceleration=state["entropy_acceleration"],
            entropy_ewma=state["entropy_ewma"],
            entropy_time_above=state["entropy_time_above"],
            disagreement_slope=state["disagreement_slope"],
            disagreement_acceleration=state["disagreement_acceleration"],
            disagreement_ewma=state["disagreement_ewma"],
            disagreement_time_above=state["disagreement_time_above"],
            consensus_slope=state["consensus_slope"],
            consensus_ewma=state["consensus_ewma"],
            communication_reliability=observation.communication_reliability,
            contributors=state["contributors"],
            missing_agents=state["missing_agents"],
        )
        self.ledger.append(
            step, "v6_operational_proposal", proposal.agent_id,
            {
                "proposal": asdict(proposal),
                "selected_by": lead,
                "received_offer_count": len(offers),
                "deployable_context": context.deployable(),
            },
            private_to=proposal.agent_id,
        )
        self.consensus_records.append({
            "cluster_id": self.cluster_id,
            "application": self.application,
            "regime": self.regime,
            "information_condition": self.information_condition,
            "environment_seed": self.seed,
            "step": step,
            "incident_id": incident_id,
            "recipient": lead,
            "sketch_policy": self.sketch_policy,
            "pooled_uncertainty": state["pooled_uncertainty"],
            "js_disagreement": state["js_disagreement"],
            "graph_disagreement": state["graph_disagreement"],
            "consensus": state["consensus"],
            "consensus_residual": state["consensus_residual"],
            "entropy_slope": state["entropy_slope"],
            "disagreement_slope": state["disagreement_slope"],
            "consensus_slope": state["consensus_slope"],
            "contributor_count": len(state["contributors"]),
            "missing_agent_count": len(state["missing_agents"]),
            "evaluator_distributed_error": state["evaluator_distributed_error"],
        })
        self.ledger.append(
            step, "v6_consensus_state", lead,
            {
                "incident_id": incident_id,
                "deployable": {
                    "pooled_uncertainty": state["pooled_uncertainty"],
                    "js_disagreement": state["js_disagreement"],
                    "consensus": state["consensus"],
                    "consensus_residual": state["consensus_residual"],
                    "contributors": list(state["contributors"]),
                    "missing_agents": list(state["missing_agents"]),
                },
                "evaluator_distributed_error": state["evaluator_distributed_error"],
            },
            private_to="evaluator",
        )
        return context

    def _resource_transition(self, action: str, quantity: float, incident_id: str, step: int) -> Tuple[bool, Optional[str]]:
        resource_key = RESOURCE_FOR_ACTION.get(action)
        if resource_key is None:
            return True, None
        account = self.resources[resource_key]
        if account.remaining + 1e-12 < quantity:
            return False, resource_key
        account.remaining -= quantity
        if action in ("authorize_emergency_resource", "deploy_repair_capacity"):
            account.transferred += quantity
        else:
            account.consumed += quantity
        self.ledger.append(
            step, "v6_resource_transition", "simulator",
            {
                "incident_id": incident_id,
                "action": action,
                "resource": resource_key,
                "quantity": quantity,
                "remaining": account.remaining,
                "consumed": account.consumed,
                "transferred": account.transferred,
                "losses": account.losses,
            },
        )
        return True, resource_key

    def schedule_action(self, proposal: V6ActionProposal, step: int, source: str) -> V6ActionResult:
        call = V6ToolCall(
            action=proposal.action, incident_id=proposal.incident_id,
            quantity=proposal.quantity, reason_code=proposal.reason_code,
        )
        identity = self.agents[proposal.agent_id].identity
        valid, code, normalized = self.registry.validate(identity, call)
        self.ledger.append(step, "tool_call", proposal.agent_id, {"call": call.as_dict(), "source": source, "v6": True})
        if not valid or normalized is None:
            result = V6ActionResult(False, False, call.action, call.incident_id, None, None, 0.0, False, False, False, False, False, code, None, 0.0)
            self.ledger.append(step, "tool_result", "simulator", result.as_dict(), private_to=proposal.agent_id)
            return result
        available, resource_key = self._resource_transition(call.action, call.quantity, call.incident_id, step)
        if not available:
            result = V6ActionResult(False, False, call.action, call.incident_id, None, None, 0.0, False, False, False, False, False, "resource_unavailable", resource_key, 0.0)
            self.ledger.append(step, "tool_result", "simulator", result.as_dict(), private_to=proposal.agent_id)
            return result
        delay = {
            "verify": 2, "request_peer_evidence": 1,
            "authorize_emergency_resource": 2, "reroute_or_reconfigure": 2,
            "deploy_repair_capacity": 3, "isolate_or_quarantine": 1,
            "revise_commitment": 1, "defer": 1, "no_action": 0,
        }[call.action]
        complete_step = min(self.horizon - 1, step + delay)
        pending = {
            "proposal": asdict(proposal), "source": source,
            "scheduled_step": step, "complete_step": complete_step,
            "resource_key": resource_key, "resource_quantity": call.quantity if resource_key else 0.0,
        }
        self.pending_actions.append(pending)
        self.ledger.append(step, "v6_action_scheduled", proposal.agent_id, pending)
        accepted_physical = call.action in PHYSICAL_ACTIONS
        result = V6ActionResult(
            True, accepted_physical, call.action, call.incident_id, step, complete_step,
            0.0, False, False, False, False, False, "scheduled",
            resource_key, call.quantity if resource_key else 0.0,
        )
        self.ledger.append(step, "tool_result", "simulator", result.as_dict(), private_to=proposal.agent_id)
        return result

    def _complete_pending(self, step: int) -> None:
        remaining: List[Dict[str, Any]] = []
        for pending in self.pending_actions:
            if int(pending["complete_step"]) > step:
                remaining.append(pending)
                continue
            proposal = V6ActionProposal(**pending["proposal"])
            incident = self.incidents[proposal.incident_id]
            action_index = OPERATIONAL_ACTIONS.index(proposal.action)
            incident_index = list(sorted(self.incidents)).index(proposal.incident_id)
            action = proposal.action
            correct = action == incident.correct_action
            secondary = action == incident.secondary_action
            changed_commitment = action == "revise_commitment" and (correct or secondary)
            direct_effect = self.preview_direct_effect(proposal, int(pending["scheduled_step"]))
            reached_next = False
            reached_service = False
            if action == "verify":
                agent_ids = self.incident_agents[proposal.incident_id]
                agent_index = agent_ids.index(proposal.agent_id)
                verified = float(self._tape["verification"][step][incident_index][agent_index]) < (0.76 - 0.18 * incident.fragmentation)
                target_mode = incident.true_mode if verified else INCIDENT_MODES[(INCIDENT_MODES.index(incident.true_mode) + 1 + agent_index) % len(INCIDENT_MODES)]
                belief = np.full(len(INCIDENT_MODES), 0.06)
                belief[INCIDENT_MODES.index(target_mode)] = 0.70
                belief /= belief.sum()
                self.agents[proposal.agent_id].private_beliefs[proposal.incident_id] = tuple(float(value) for value in belief)
                self.ledger.append(step, "telemetry_verification", proposal.agent_id, {"incident_id": proposal.incident_id, "verified_signal_delivered": True, "truth_not_exposed": True})
            elif action == "request_peer_evidence":
                self._deliver_peer_evidence(proposal.incident_id, proposal.agent_id, step)
            elif action in ("defer", "no_action"):
                pass
            elif action == "revise_commitment" and changed_commitment:
                for agent_id in self.incident_agents[proposal.incident_id]:
                    for commitment in self.agents[agent_id].commitments.values():
                        if commitment.incident_id == proposal.incident_id:
                            commitment.revision += 1
                            commitment.status = "revised"
                self.ledger.append(step, "commitment", proposal.agent_id, {
                    "incident_id": proposal.incident_id,
                    "status": "revised",
                    "bounded_revision": True,
                })
                reached_next = True
                reached_service = self._action_draw(
                    proposal, int(pending["scheduled_step"]),
                ) > 0.25
            elif correct:
                reached_next = True
                reached_service = self._action_draw(proposal, int(pending["scheduled_step"])) > 0.10
            elif secondary:
                draw = self._action_draw(proposal, int(pending["scheduled_step"]))
                reached_next = draw > 0.12
                reached_service = draw > 0.28
            else:
                draw = self._action_draw(proposal, int(pending["scheduled_step"]))
                reached_next = draw > 0.72
                reached_service = False
            before_deficit = incident.service_deficit
            incident.service_deficit = float(np.clip(incident.service_deficit - direct_effect, 0.0, 1.6))
            incident.backlog = float(np.clip(incident.backlog - 0.65 * direct_effect, 0.0, 2.0))
            incident.resolved_fraction = float(np.clip(incident.resolved_fraction + max(direct_effect, 0.0), 0.0, 1.0))
            incident.last_effect = float(direct_effect)
            record = {
                **pending,
                "completed_step": step,
                "causal_effect": float(direct_effect),
                "harmful": bool(direct_effect < -1e-9),
                "beneficial": bool(direct_effect > 1e-9),
                "neutral": bool(abs(direct_effect) <= 1e-9),
                "changed_commitment": changed_commitment,
                "accepted_typed_action": True,
                "accepted_physical_action": action in PHYSICAL_ACTIONS,
                "reached_next_stage": reached_next,
                "reached_service": reached_service,
                "service_deficit_before": before_deficit,
                "service_deficit_after": incident.service_deficit,
            }
            self.action_records.append(record)
            self.ledger.append(step, "v6_action_completed", proposal.agent_id, record)
            self.ledger.append(step, "v6_service_transition", "simulator", {
                "incident_id": proposal.incident_id,
                "action": action,
                "deficit_before": before_deficit,
                "deficit_after": incident.service_deficit,
                "reached_service": reached_service,
            })
        self.pending_actions = remaining

    def _action_draw(self, proposal: V6ActionProposal, scheduled_step: int) -> float:
        incident_index = list(sorted(self.incidents)).index(proposal.incident_id)
        action_index = OPERATIONAL_ACTIONS.index(proposal.action)
        return float(self._tape["action"][scheduled_step][incident_index][action_index])

    def preview_direct_effect(self, proposal: V6ActionProposal, scheduled_step: int) -> float:
        """Evaluator-only immediate effect for learnability and safety labels.

        Controllers never receive this value. Formal service outcomes still
        arise by scheduling and completing the action in the dynamic system.
        """
        incident = self.incidents[proposal.incident_id]
        draw = self._action_draw(proposal, scheduled_step)
        action = proposal.action
        if action == "verify":
            return float(-0.012 * (0.7 + incident.priority))
        if action == "request_peer_evidence":
            return float(-0.008 * (0.7 + incident.priority))
        if action in ("defer", "no_action"):
            return 0.0
        if action == incident.correct_action:
            return float((0.14 + 0.34 * incident.severity) * (0.82 + 0.28 * draw))
        if action == incident.secondary_action:
            return float((0.07 + 0.17 * incident.severity) * (0.72 + 0.24 * draw))
        return float(-(0.035 + 0.14 * draw) * (0.55 + incident.severity))

    def _deliver_peer_evidence(self, incident_id: str, requester: str, step: int) -> None:
        peers = [value for value in self.incident_agents[incident_id] if value != requester]
        delivered: List[np.ndarray] = []
        for sender in peers:
            if not self._edge_available(sender, requester, step):
                continue
            belief = np.asarray(self.agents[sender].private_beliefs[incident_id], dtype=float)
            delivered.append(belief)
            message = Message(
                message_id="V6M%08d" % (self.operational_messages + 1),
                sender=sender, recipient=requester,
                kind="peer_evidence", payload={"incident_id": incident_id, "belief_summary": belief.tolist()},
                sent_step=step, deliver_step=step, public=False,
            )
            self.agents[requester].receive(message)
            self.operational_messages += 1
            self.operational_bytes += 126
            self.ledger.append(step, "message", sender, asdict(message))
            self.ledger.append(step, "message_delivery", "network", asdict(message), private_to=requester)
        if delivered:
            own = np.asarray(self.agents[requester].private_beliefs[incident_id], dtype=float)
            pooled = np.mean(np.vstack([own, *delivered]), axis=0)
            pooled /= pooled.sum()
            self.agents[requester].private_beliefs[incident_id] = tuple(float(value) for value in pooled)

    def _operator_proposal(self, context: V6DecisionContext, step: int) -> V6ActionProposal:
        incident_id = context.proposal.incident_id
        agent_id = context.proposal.agent_id
        # Bounded simulated operator sees only the authorized distributed
        # context. It has no true state, future tape, or counterfactual effect.
        state = self.information_state(incident_id, agent_id, step)
        belief = np.asarray(state["pooled_belief"], dtype=float)
        order = np.argsort(belief)[::-1]
        draw = float(self._tape["operator"][step][list(sorted(self.incidents)).index(incident_id)])
        chosen_index = int(order[0] if draw < 0.82 else order[min(1, len(order) - 1)])
        action = PRIMARY_ACTION_FOR_MODE[INCIDENT_MODES[chosen_index]]
        allowed = set(self.registry.allowed_actions(self.agents[agent_id].identity.role))
        if action not in allowed:
            secondary = SECONDARY_ACTION_FOR_MODE[INCIDENT_MODES[chosen_index]]
            action = secondary if secondary in allowed else context.proposal.action
        return V6ActionProposal(
            agent_id=agent_id, role=context.proposal.role, incident_id=incident_id,
            action=action, quantity=1.0, action_probability=float(belief[chosen_index]),
            action_value=context.proposal.action_value, value_margin=context.proposal.value_margin,
            reason_code="bounded_simulated_operator_authorized_view",
        )

    def _process_operator_queue(self, step: int) -> None:
        if step < self.operator_busy_until or not self.operator_queue:
            return
        request = self.operator_queue.pop(0)
        context = V6DecisionContext(**request["context"])
        proposal = self._operator_proposal(context, step)
        minutes = 5.0 + 4.0 * (1.0 - context.consensus)
        self.operator_minutes += minutes
        self.operator_busy_until = step + 1
        self.ledger.append(step, "v6_operator_response", "simulated_operator", {
            "incident_id": proposal.incident_id,
            "action": proposal.action,
            "queue_delay_steps": step - int(request["requested_step"]),
            "operator_minutes": minutes,
            "simulated_operator": True,
            "real_human": False,
        })
        self.schedule_action(proposal, step, "bounded_simulated_operator")

    def _advance_service(self, step: int) -> None:
        for incident_index, incident_id in enumerate(sorted(self.incidents)):
            incident = self.incidents[incident_id]
            before = incident.service_deficit
            if step >= incident.disruption_step and incident.true_mode != "nominal":
                shock = incident.severity * float(self._tape["service"][step][incident_index])
                increment = 0.055 * shock * (1.0 - 0.70 * incident.resolved_fraction)
            else:
                increment = 0.0015 * incident.severity
            incident.service_deficit = float(np.clip(incident.service_deficit + increment, 0.0, 1.6))
            incident.backlog = float(np.clip(incident.backlog + 0.045 * incident.service_deficit, 0.0, 2.0))
            step_loss = incident.priority * (incident.service_deficit + 0.28 * incident.backlog)
            incident.cumulative_loss += float(step_loss)
            self.ledger.append(step, "environment_transition", "simulator", {
                "incident_id": incident_id,
                "service_deficit_before": before,
                "service_deficit_after": incident.service_deficit,
                "backlog_after": incident.backlog,
                "step_loss": step_loss,
            })

    def record_candidate(self, context: V6DecisionContext) -> None:
        """Record a matched dynamic evaluator label outside deployable state."""
        immediate_effect = self.preview_direct_effect(
            context.proposal, context.step,
        )
        counterfactual = self.evaluator_counterfactual_branch(
            context.proposal, context.step,
        )
        causal_utility = float(counterfactual["loss_reduction"])
        self.candidate_records.append({
            "cluster_id": self.cluster_id,
            "application": self.application,
            "regime": self.regime,
            "information_condition": self.information_condition,
            "environment_seed": self.seed,
            "topology_family": self.topology_family,
            "scenario_family": self.scenario_family,
            "split_family": self.split_family,
            "step": context.step,
            "incident_id": context.proposal.incident_id,
            "agent_id": context.proposal.agent_id,
            "role": context.proposal.role,
            "proposed_action": context.proposal.action,
            **context.local_kpis,
            **{key: value for key, value in context.deployable().items() if key != "proposal"},
            "action_probability": context.proposal.action_probability,
            "action_value": context.proposal.action_value,
            "value_margin": context.proposal.value_margin,
            "evaluator_immediate_effect_if_executed": immediate_effect,
            "evaluator_causal_utility_if_executed": causal_utility,
            "evaluator_harmful_if_executed": bool(causal_utility < -1e-9),
            "evaluator_beneficial_if_executed": bool(causal_utility > 1e-9),
        })

    def apply_delegation(
        self,
        context: V6DecisionContext,
        delegation: str,
        controller_name: str,
    ) -> None:
        """Apply one Level-2 choice without substituting a central action."""
        step = context.step
        incident_id = context.proposal.incident_id
        if delegation not in (
            "execute_autonomously", "communicate", "request_evidence",
            "defer", "abstain", "escalate_operator",
        ):
            raise ValueError("invalid V6 delegation action")
        record = {
            "step": step,
            "incident_id": incident_id,
            "agent_id": context.proposal.agent_id,
            "proposed_action": context.proposal.action,
            "delegation_action": delegation,
            "controller": controller_name,
            "consensus": context.consensus,
            "js_disagreement": context.js_disagreement,
            "action_value": context.proposal.action_value,
            "value_margin": context.proposal.value_margin,
        }
        self.delegation_records.append(record)
        self.ledger.append(
            step, "v6_delegation_decision", context.proposal.agent_id, record,
        )
        if delegation == "execute_autonomously":
            self.schedule_action(context.proposal, step, "autonomous_agent")
        elif delegation in ("communicate", "request_evidence"):
            if "request_peer_evidence" in self.registry.allowed_actions(context.proposal.role):
                proposal = V6ActionProposal(**{
                    **asdict(context.proposal),
                    "action": "request_peer_evidence",
                    "reason_code": "delegation_requested_evidence",
                })
                self.schedule_action(proposal, step, "delegation_controller")
        elif delegation == "defer":
            if "defer" in self.registry.allowed_actions(context.proposal.role):
                proposal = V6ActionProposal(**{
                    **asdict(context.proposal),
                    "action": "defer", "reason_code": "delegation_defer",
                })
                self.schedule_action(proposal, step, "delegation_controller")
        elif delegation == "escalate_operator" and self.operator_budget_remaining > 0:
            self.operator_budget_remaining -= 1
            context_payload = asdict(context)
            context_payload["proposal"] = V6ActionProposal(**context_payload["proposal"])
            authorized_view = context.deployable()
            self.ledger.append(
                step, "operator_view", "simulator", {
                    "v6": True,
                    "incident_id": incident_id,
                    "authorized_view": authorized_view,
                    "payload_sha256": payload_digest(authorized_view),
                    "simulated_operator": True,
                    "real_human": False,
                }, private_to="simulated_operator",
            )
            self.operator_queue.append({
                "requested_step": step, "context": context_payload,
            })
            self.maximum_queue_length = max(
                self.maximum_queue_length, len(self.operator_queue),
            )
            self.operational_messages += 1
            self.operational_bytes += 284
            self.ledger.append(step, "v6_operator_escalation", context.proposal.agent_id, {
                "incident_id": incident_id,
                "authorized_view_digest": payload_digest(context.deployable()),
                "remaining_budget": self.operator_budget_remaining,
                "queue_length": len(self.operator_queue),
            })

    def finalize(self, controller_name: str) -> Dict[str, Any]:
        """Complete pending work and produce the authoritative episode summary."""
        self._complete_pending(self.horizon - 1)
        conservation = self.conservation_report()
        summary = {
            "cluster_id": self.cluster_id,
            "application": self.application,
            "regime": self.regime,
            "information_condition": self.information_condition,
            "environment_seed": self.seed,
            "topology_family": self.topology_family,
            "scenario_family": self.scenario_family,
            "split_family": self.split_family,
            "sketch_policy": self.sketch_policy,
            "controller": controller_name,
            "horizon": self.horizon,
            "incidents": self.incident_count,
            "agents": len(self.agents),
            "service_loss": float(sum(value.cumulative_loss for value in self.incidents.values())),
            "final_service_deficit": float(sum(value.service_deficit for value in self.incidents.values())),
            "accepted_typed_actions": int(len(self.action_records)),
            "accepted_physical_actions": int(sum(bool(value["accepted_physical_action"]) for value in self.action_records)),
            "beneficial_actions": int(sum(bool(value["beneficial"]) for value in self.action_records)),
            "neutral_actions": int(sum(bool(value["neutral"]) for value in self.action_records)),
            "harmful_actions": int(sum(bool(value["harmful"]) for value in self.action_records)),
            "service_reaching_actions": int(sum(bool(value["reached_service"]) for value in self.action_records)),
            "net_causal_utility": float(sum(float(value["causal_effect"]) for value in self.action_records)),
            "autonomous_completed_actions": int(sum(value["source"] == "autonomous_agent" for value in self.action_records)),
            "autonomous_harmful_actions": int(sum(value["source"] == "autonomous_agent" and bool(value["harmful"]) for value in self.action_records)),
            "autonomous_beneficial_actions": int(sum(value["source"] == "autonomous_agent" and bool(value["beneficial"]) for value in self.action_records)),
            "autonomous_causal_utility": float(sum(float(value["causal_effect"]) for value in self.action_records if value["source"] == "autonomous_agent")),
            "operator_completed_actions": int(sum(value["source"] == "bounded_simulated_operator" for value in self.action_records)),
            "operator_harmful_actions": int(sum(value["source"] == "bounded_simulated_operator" and bool(value["harmful"]) for value in self.action_records)),
            "commitment_revisions": int(sum(bool(value["changed_commitment"]) for value in self.action_records)),
            "autonomous_executions": int(sum(value["delegation_action"] == "execute_autonomously" for value in self.delegation_records)),
            "eligible_operational_proposals": int(sum(
                value["proposed_action"] in PHYSICAL_ACTIONS
                for value in self.delegation_records
            )),
            "decision_epoch_count": len(self.decision_steps),
            "incident_decision_count": len(self.delegation_records),
            "abstentions": int(sum(value["delegation_action"] == "abstain" for value in self.delegation_records)),
            "escalation_attempts": int(sum(value["delegation_action"] == "escalate_operator" for value in self.delegation_records)),
            "escalations": int(sum(event.kind == "v6_operator_escalation" for event in self.ledger.events)),
            "pre_disruption_escalations": int(sum(
                event.kind == "v6_operator_escalation" and event.step < 2
                for event in self.ledger.events
            )),
            "post_disruption_escalations": int(sum(
                event.kind == "v6_operator_escalation" and event.step >= 2
                for event in self.ledger.events
            )),
            "first_escalation_step": next((
                int(event.step) for event in self.ledger.events
                if event.kind == "v6_operator_escalation"
            ), None),
            "first_post_disruption_escalation_step": next((
                int(event.step) for event in self.ledger.events
                if event.kind == "v6_operator_escalation" and event.step >= 2
            ), None),
            "timely_post_disruption_activation_by_step_4": any(
                event.kind == "v6_operator_escalation" and 2 <= event.step <= 4
                for event in self.ledger.events
            ),
            "nominal_false_activation": bool(
                self.regime == "nominal" and any(
                    event.kind == "v6_operator_escalation"
                    for event in self.ledger.events
                )
            ),
            "operator_minutes": self.operator_minutes,
            "maximum_queue_length": self.maximum_queue_length,
            "operator_queue_unserved": len(self.operator_queue),
            "operational_messages": self.operational_messages,
            "operational_bytes": self.operational_bytes,
            "sketch_messages": self.sketch_messages,
            "sketch_bytes": self.sketch_bytes,
            "sketch_latency": self.sketch_latency,
            "total_messages": self.operational_messages + self.sketch_messages,
            "total_bytes": self.operational_bytes + self.sketch_bytes,
            "conservation_feasible": conservation["feasible"],
            "maximum_conservation_residual": conservation["maximum_residual"],
            "event_count": len(self.ledger.events),
            "event_ledger_digest": self.ledger.digest(),
            "stochastic_tape_digest": self.stochastic_tape_digest,
        }
        self.ledger.append(self.horizon, "v6_conservation_audit", "simulator", conservation)
        self.ledger.append(self.horizon, "metric", "evaluator", summary)
        summary["event_count"] = len(self.ledger.events)
        summary["event_ledger_digest"] = self.ledger.digest()
        return summary

    def evaluator_counterfactual_branch(
        self, proposal: V6ActionProposal, step: int,
    ) -> Dict[str, Any]:
        """Evaluator-only matched dynamic action/no-action branch.

        Both branches copy the complete current simulator, agent, queue, and
        RNG-tape state.  Only the selected action differs.  This method is
        never exposed through an agent or operator view.
        """
        # The historical ledger is immutable evidence, not causal state.
        # Supplying an empty replacement avoids copying a growing event list
        # for every candidate while every simulator, agent, queue, resource,
        # commitment, and random-tape state is still independently cloned.
        intervention = deepcopy(self, {id(self.ledger): EventLedger()})
        control = deepcopy(self, {id(self.ledger): EventLedger()})
        intervention.schedule_action(proposal, int(step), "counterfactual_probe")
        for future_step in range(int(step) + 1, self.horizon):
            for branch in (intervention, control):
                branch.current_step = future_step
                branch._complete_pending(future_step)
                branch._process_operator_queue(future_step)
                branch._advance_service(future_step)
        intervention_conservation = intervention.conservation_report()
        control_conservation = control.conservation_report()
        if not intervention_conservation["feasible"] or not control_conservation["feasible"]:
            raise RuntimeError("counterfactual branch violated resource conservation")
        loss_with_action = float(sum(
            value.cumulative_loss for value in intervention.incidents.values()
        ))
        loss_without_action = float(sum(
            value.cumulative_loss for value in control.incidents.values()
        ))
        result = {
            "incident_id": proposal.incident_id,
            "action": proposal.action,
            "branch_step": int(step),
            "stochastic_tape_digest_action": intervention.stochastic_tape_digest,
            "stochastic_tape_digest_no_action": control.stochastic_tape_digest,
            "loss_with_action": loss_with_action,
            "loss_without_action": loss_without_action,
            "loss_reduction": loss_without_action - loss_with_action,
            "action_ledger_digest": intervention.ledger.digest(),
            "no_action_ledger_digest": control.ledger.digest(),
        }
        self.ledger.append(
            int(step), "v6_counterfactual_branch", "evaluator", result,
            private_to="evaluator",
        )
        return result

    def run(
        self,
        controller: Callable[[Sequence[V6DecisionContext], int], Mapping[str, str]],
        controller_name: str,
    ) -> Dict[str, Any]:
        for step in range(self.horizon):
            self.current_step = step
            self._complete_pending(step)
            self._process_operator_queue(step)
            self._advance_service(step)
            self.deliver_observations(step)
            for incident_id in sorted(self.incidents):
                self.exchange_sketches(incident_id, step)
            if step not in self.decision_steps:
                continue
            contexts = [self.decision_context(incident_id, step) for incident_id in sorted(self.incidents)]
            for context in contexts:
                self.record_candidate(context)
            decisions = dict(controller(contexts, step))
            for context in contexts:
                self.apply_delegation(
                    context, decisions.get(context.proposal.incident_id, "abstain"),
                    controller_name,
                )
        return self.finalize(controller_name)

    def conservation_report(self) -> Dict[str, Any]:
        residuals = {key: account.residual() for key, account in self.resources.items()}
        nonnegative = all(
            min(account.remaining, account.consumed, account.transferred, account.losses) >= -1e-10
            for account in self.resources.values()
        )
        maximum = max((abs(value) for value in residuals.values()), default=0.0)
        return {
            "feasible": bool(nonnegative and maximum <= 1e-9),
            "residuals": residuals,
            "maximum_residual": float(maximum),
            "components": {key: asdict(value) for key, value in self.resources.items()},
        }

    def inject_conservation_fault_for_test(self, resource: str, quantity: float) -> None:
        """Deliberate fault hook used only by the negative engineering test."""
        self.resources[resource].remaining += float(quantity)

    def evaluator_private_state(self) -> Dict[str, Any]:
        """Analysis-only state that must never enter an agent/operator view."""
        return {
            "true_modes": {key: value.true_mode for key, value in self.incidents.items()},
            "stochastic_tape": deepcopy(self._tape),
            "counterfactual_available": True,
        }
