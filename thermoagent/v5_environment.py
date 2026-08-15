"""Competitive multi-incident V5 environment with matched counterfactuals.

Cyber-physical events are defensive abstract state transitions only. This
module has no networking, protocol, credential, scanning, or exploitation code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .events import EventLedger
from .types import Message
from .v5_agents import IndependentV5Agent, V5Utility
from .v5_types import (
    APPLICATIONS, INCIDENT_MODES, INFORMATION_CONDITIONS, OPERATOR_ACTIONS,
    PRIMARY_ACTION_FOR_MODE, REGIMES, SECONDARY_ACTION_FOR_MODE, SKETCH_POLICIES,
    V5ActionEffect, V5Commitment, V5Identity, V5Incident,
    V5PrivateObservation, V5ThermodynamicState, jensen_shannon,
    normalized_entropy,
)


APP_ROLES = {
    "commercial": ("supplier", "carrier", "warehouse", "retailer", "coordinator"),
    "humanitarian": ("ngo", "carrier", "regional_hub", "clinic", "coordinator"),
    "utility_restoration": (
        "distribution_node", "field_crew", "communications", "cyber_defense",
        "resource_allocation", "critical_load", "regional_coordinator",
    ),
}

APP_INCIDENT_PREFIX = {
    "commercial": "supply",
    "humanitarian": "relief",
    "utility_restoration": "utility",
}


def stable_seed(*values: Any) -> int:
    blob = "|".join(str(value) for value in values).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def payload_digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class V5PanelEnvironment:
    """One independent cluster containing four competing incidents."""

    def __init__(
        self,
        application: str,
        regime: str,
        information_condition: str,
        seed: int,
        sketch_policy: str = "event_triggered",
        incidents_per_panel: int = 4,
        operator_budget: int = 2,
    ) -> None:
        if application not in APPLICATIONS:
            raise ValueError("unknown V5 application")
        if regime not in REGIMES:
            raise ValueError("unknown V5 regime")
        if information_condition not in INFORMATION_CONDITIONS:
            raise ValueError("unknown information condition")
        if sketch_policy not in SKETCH_POLICIES:
            raise ValueError("unknown sketch policy")
        if incidents_per_panel < 3 or incidents_per_panel > 6:
            raise ValueError("V5 requires three to six simultaneous incidents")
        self.application = application
        self.regime = regime
        self.information_condition = information_condition
        self.seed = int(seed)
        self.sketch_policy = sketch_policy
        self.operator_budget = int(operator_budget)
        self.rng = np.random.RandomState(stable_seed(application, regime, information_condition, seed))
        self.ledger = EventLedger()
        self.incidents = self._build_incidents(int(incidents_per_panel))
        self.stochastic_tape = self.rng.uniform(0.0, 1.0, size=256).tolist()
        self.stochastic_tape_digest = payload_digest(self.stochastic_tape)
        self.agents, self.incident_agents = self._build_agents()
        self.observations: Dict[str, V5PrivateObservation] = {}
        self.thermodynamics: Dict[str, V5ThermodynamicState] = {}
        self.evaluator_entropy: Dict[str, float] = {}
        self.operational_messages = 0
        self.operational_bytes = 0
        self.commitment_events = 0
        self.resource_initial = {
            "verification_slots": 8.0,
            "emergency_resources": 8.0,
            "repair_capacity": 8.0,
            "routing_authorizations": 8.0,
            "isolation_authorizations": 8.0,
        }
        self.resource_used = {key: 0.0 for key in self.resource_initial}
        self._initialize_ledger()
        self._deliver_private_observations()
        self._compute_distributed_states()

    @property
    def cluster_id(self) -> str:
        return "%s|%s|%s|%d" % (
            self.application, self.regime, self.information_condition, self.seed,
        )

    def _build_incidents(self, count: int) -> Dict[str, V5Incident]:
        mode_order = list(INCIDENT_MODES)
        self.rng.shuffle(mode_order)
        if count > len(mode_order):
            mode_order.extend(mode_order[: count - len(mode_order)])
        result: Dict[str, V5Incident] = {}
        regime_stress = {
            "nominal": 0.08,
            "isolated_physical": 0.43,
            "telemetry_integrity": 0.40,
            "partition": 0.45,
            "correlated": 0.57,
            "compound": 0.66,
            "ood": 0.61,
        }[self.regime]
        fragmentation_shift = {
            "nominal": 0.08,
            "isolated_physical": 0.14,
            "telemetry_integrity": 0.45,
            "partition": 0.50,
            "correlated": 0.30,
            "compound": 0.48,
            "ood": 0.56,
        }[self.regime]
        if self.information_condition == "public_shared":
            fragmentation_shift *= 0.24
        angles = np.linspace(0.15, 2.0 * math.pi + 0.15, count, endpoint=False)
        for index in range(count):
            mode = mode_order[index]
            jitter = float(self.rng.normal(0.0, 0.10))
            severity = float(np.clip(regime_stress + jitter + 0.05 * index, 0.04, 0.92))
            priority = float(np.clip(0.62 + self.rng.normal(0.0, 0.17), 0.30, 1.0))
            fragmentation = float(np.clip(fragmentation_shift + self.rng.normal(0.0, 0.12), 0.03, 0.92))
            telemetry = float(np.clip(0.94 - fragmentation + self.rng.normal(0.0, 0.07), 0.06, 0.98))
            incident_id = "%s_%02d" % (APP_INCIDENT_PREFIX[self.application], index + 1)
            base_loss = float((0.35 + 1.15 * severity) * (0.55 + 0.65 * priority))
            result[incident_id] = V5Incident(
                incident_id=incident_id,
                scenario_family="%s_%s" % (self.regime, mode),
                topology_family="%s_topology_%d" % (self.application, self.seed % 4),
                true_mode=mode,
                correct_action=PRIMARY_ACTION_FOR_MODE[mode],
                secondary_action=SECONDARY_ACTION_FOR_MODE[mode],
                severity=severity,
                priority=priority,
                fragmentation=fragmentation,
                telemetry_integrity=telemetry,
                base_loss=base_loss,
                disruption_step=6 + int(self.seed % 3),
                location=(float(math.cos(angles[index])), float(math.sin(angles[index]))),
            )
        return result

    def _build_agents(self) -> Tuple[Dict[str, IndependentV5Agent], Dict[str, List[str]]]:
        agents: Dict[str, IndependentV5Agent] = {}
        scopes: Dict[str, List[str]] = {key: [] for key in self.incidents}
        roles = APP_ROLES[self.application]
        for incident_index, incident_id in enumerate(self.incidents):
            for local_index in range(3):
                role = roles[(incident_index + local_index) % len(roles)]
                agent_id = "%s_%s_%d" % (self.application[:3], role, incident_index * 3 + local_index)
                identity = V5Identity(
                    agent_id=agent_id,
                    application=self.application,
                    role=role,
                    incident_scope=(incident_id,),
                    authority=(
                        "message", "negotiate", "accept", "reject", "counteroffer",
                        "revise_commitment", "request_human", "execute_typed_action",
                    ),
                )
                utility_rng = np.random.RandomState(stable_seed(self.seed, agent_id, "utility"))
                utility = V5Utility(
                    service_weight=float(utility_rng.uniform(0.65, 1.15)),
                    safety_weight=float(utility_rng.uniform(0.55, 1.20)),
                    cost_weight=float(utility_rng.uniform(0.30, 0.85)),
                    disclosure_cost=float(utility_rng.uniform(0.02, 0.16)),
                    risk_tolerance=float(utility_rng.uniform(0.25, 0.85)),
                )
                agents[agent_id] = IndependentV5Agent(identity, utility, stable_seed(self.seed, agent_id))
                scopes[incident_id].append(agent_id)
        return agents, scopes

    def _initialize_ledger(self) -> None:
        topology = {
            "application": self.application,
            "regime": self.regime,
            "information_condition": self.information_condition,
            "cluster_id": self.cluster_id,
            "incidents": [
                {
                    "incident_id": value.incident_id,
                    "location": list(value.location),
                    "scenario_family": value.scenario_family,
                    "topology_family": value.topology_family,
                }
                for value in self.incidents.values()
            ],
            "agents": {key: asdict(value.identity) for key, value in self.agents.items()},
            "operator_budget": self.operator_budget,
            "abstract_defensive_simulation": True,
            "cyber_scope": "abstract state transitions only",
        }
        self.ledger.append(0, "topology_snapshot", "simulator", topology)
        self.ledger.append(
            0, "v5_stochastic_tape", "evaluator",
            {"length": len(self.stochastic_tape), "digest": self.stochastic_tape_digest},
            private_to="evaluator",
        )
        self.ledger.append(
            0, "v5_panel_snapshot", "simulator",
            {
                "cluster_id": self.cluster_id,
                "resource_initial": self.resource_initial,
                "incident_public_state": [
                    {
                        "incident_id": item.incident_id,
                        "location": list(item.location),
                        "disruption_step": item.disruption_step,
                    }
                    for item in self.incidents.values()
                ],
            },
        )

    def _mode_pattern(self, mode: str) -> np.ndarray:
        patterns = {
            "evidence_conflict": np.asarray([0.25, 0.34, 0.40, 0.25, 0.36]),
            "resource_shortage": np.asarray([0.42, 0.64, 0.30, 0.78, 0.22]),
            "route_or_service_failure": np.asarray([0.48, 0.55, 0.77, 0.30, 0.26]),
            "unsafe_or_compromised_component": np.asarray([0.50, 0.28, 0.45, 0.32, 0.82]),
            "commitment_deadlock": np.asarray([0.44, 0.50, 0.52, 0.42, 0.34]),
        }
        return patterns[mode]

    def _private_observation(self, incident: V5Incident, agent_id: str, agent_index: int) -> V5PrivateObservation:
        rng = np.random.RandomState(stable_seed(self.seed, incident.incident_id, agent_id, "observation"))
        mode_index = INCIDENT_MODES.index(incident.true_mode)
        public = self.information_condition == "public_shared"
        evidence_strength = float(np.clip(1.15 - 0.62 * incident.fragmentation, 0.45, 1.20))
        belief_noise = 0.42 + 0.25 * incident.fragmentation
        if self.application == "utility_restoration" and not public:
            # Utility roles observe complementary evidence: the first local
            # zone view is weak, while communications and field/cyber roles
            # contribute progressively more reliable private evidence.
            evidence_strength += (-0.18, 0.16, 0.38)[agent_index]
            belief_noise *= (1.15, 0.92, 0.76)[agent_index]
        logits = rng.normal(0.0, belief_noise, len(INCIDENT_MODES))
        logits[mode_index] += evidence_strength
        if not public and incident.fragmentation > 0.32:
            conflicting_index = (mode_index + 1 + agent_index) % len(INCIDENT_MODES)
            conflict_strength = 0.28 + 0.66 * incident.fragmentation
            if self.application == "utility_restoration":
                conflict_strength *= (1.10, 0.86, 0.68)[agent_index]
            logits[conflicting_index] += conflict_strength
        if public:
            shared_rng = np.random.RandomState(stable_seed(self.seed, incident.incident_id, "public_evidence"))
            logits = shared_rng.normal(0.0, 0.30, len(INCIDENT_MODES))
            logits[mode_index] += 1.20
            logits += rng.normal(0.0, 0.08, len(INCIDENT_MODES))
        logits -= logits.max()
        evidence = np.exp(logits)
        evidence /= evidence.sum()

        mode_pattern = self._mode_pattern(incident.true_mode)
        noise_scale = 0.16 if not public else 0.07
        pattern_weight = 0.34 if not public else 0.72
        raw = (
            (1.0 - pattern_weight) * incident.severity
            + pattern_weight * mode_pattern
            + rng.normal(0.0, noise_scale, 5)
        )
        raw = np.clip(raw, 0.01, 0.99)
        return V5PrivateObservation(
            step=incident.disruption_step,
            incident_id=incident.incident_id,
            visible_severity=float(np.clip(incident.severity + rng.normal(0.0, noise_scale), 0.0, 1.0)),
            visible_backlog=float(raw[1]),
            visible_delay=float(raw[2]),
            resource_scarcity=float(raw[3]),
            safety_risk=float(raw[4]),
            commitment_strain=float(np.clip(0.22 + 0.58 * (incident.true_mode == "commitment_deadlock") + rng.normal(0.0, noise_scale), 0.0, 1.0)),
            telemetry_confidence=float(np.clip(
                incident.telemetry_integrity * (0.62 + 0.52 * float(evidence[mode_index]))
                + rng.normal(0.0, 0.04),
                0.0, 1.0,
            )),
            communication_reliability=float(np.clip(0.95 - 0.70 * (self.regime == "partition") - 0.18 * incident.fragmentation + rng.normal(0.0, 0.04), 0.05, 1.0)),
            private_evidence=tuple(float(value) for value in evidence),
            private_inventory=float(rng.uniform(0.8, 2.2)),
            private_cost=float(rng.uniform(0.15, 0.75)),
            private_priority=float(np.clip(incident.priority + rng.normal(0.0, 0.10), 0.0, 1.0)),
        )

    def _deliver_private_observations(self) -> None:
        for incident_id, agent_ids in self.incident_agents.items():
            incident = self.incidents[incident_id]
            for agent_index, agent_id in enumerate(agent_ids):
                observation = self._private_observation(incident, agent_id, agent_index)
                self.observations[agent_id] = observation
                self.agents[agent_id].deliver(observation, self.ledger)
        self.ledger.append(
            min(value.disruption_step for value in self.incidents.values()),
            "v5_privacy_audit", "simulator",
            {
                "private_state_leak": False,
                "future_state_leak": False,
                "operator_global_state_leak": False,
                "separate_vaults": len({id(agent.vault) for agent in self.agents.values()}) == len(self.agents),
            },
        )

    def _belief_trajectory(self, final: np.ndarray) -> List[np.ndarray]:
        uniform = np.ones(len(final), dtype=float) / len(final)
        trajectory = []
        for epoch in range(5):
            weight = (epoch + 1) / 5.0
            value = (1.0 - weight) * uniform + weight * final
            value /= value.sum()
            trajectory.append(value)
        return trajectory

    def _transmitted(self, epoch: int, current: np.ndarray, prior: Optional[np.ndarray]) -> bool:
        if self.sketch_policy == "none":
            return False
        if self.sketch_policy == "always_on":
            return True
        if self.sketch_policy == "periodic":
            return epoch in (0, 4)
        if prior is None:
            return True
        entropy_delta = abs(normalized_entropy(current.tolist()) - normalized_entropy(prior.tolist()))
        distribution_delta = float(np.abs(current - prior).sum() / 2.0)
        return entropy_delta >= 0.120 or distribution_delta >= 0.140

    def _compute_distributed_states(self) -> None:
        for incident_id, agent_ids in self.incident_agents.items():
            incident = self.incidents[incident_id]
            final_beliefs = [
                np.asarray(self.agents[agent_id].private_beliefs[incident_id], dtype=float)
                for agent_id in agent_ids
            ]
            received: Dict[str, np.ndarray] = {}
            prior_sent: Dict[str, np.ndarray] = {}
            sketch_messages = 0
            sketch_bytes = 0
            entropy_history: List[float] = []
            for epoch in range(5):
                for agent_id, final in zip(agent_ids, final_beliefs):
                    trajectory = self._belief_trajectory(final)
                    current = trajectory[epoch]
                    if not self._transmitted(epoch, current, prior_sent.get(agent_id)):
                        continue
                    prior_sent[agent_id] = current.copy()
                    # Under partitions one contributor is intermittently isolated;
                    # its message is still counted but cannot reach the local estimator.
                    delivered = not (
                        self.regime == "partition"
                        and agent_id in agent_ids[1:]
                        and epoch >= 2
                    )
                    payload = {
                        "agent_id": agent_id,
                        "incident_id": incident_id,
                        "epoch": epoch,
                        "belief": [round(float(value), 5) for value in current],
                        "entropy": round(normalized_entropy(current.tolist()), 5),
                        "confidence_band": "low" if self.observations[agent_id].telemetry_confidence < 0.45 else "bounded",
                        "delivered": delivered,
                    }
                    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    sketch_messages += 1
                    sketch_bytes += len(encoded)
                    self.ledger.append(incident.disruption_step - 4 + epoch, "thermodynamic_sketch", agent_id, payload)
                    if delivered:
                        received[agent_id] = current.copy()
                    elif epoch >= 2:
                        received.pop(agent_id, None)
                available = list(received.values())
                if not available:
                    available = [self._belief_trajectory(final_beliefs[0])[epoch]]
                center = np.mean(np.vstack(available), axis=0)
                center /= center.sum()
                entropy_history.append(normalized_entropy(center.tolist()))
            individual_entropies = [normalized_entropy(value.tolist()) for value in final_beliefs]
            local_center = np.mean(np.vstack(final_beliefs), axis=0)
            local_center /= local_center.sum()
            available = list(received.values()) or [final_beliefs[0]]
            distributed = np.mean(np.vstack(available), axis=0)
            distributed /= distributed.sum()
            residual = float(np.mean([np.abs(value - distributed).sum() / 2.0 for value in final_beliefs]))
            availability = len(available) / len(final_beliefs)
            confidence = float(np.clip(availability * math.exp(-1.4 * residual), 0.0, 1.0))
            observation = self.observations[agent_ids[0]]
            energy = float(np.clip(
                0.30 * observation.visible_severity
                + 0.18 * observation.visible_backlog
                + 0.14 * observation.visible_delay
                + 0.14 * observation.resource_scarcity
                + 0.14 * observation.safety_risk
                + 0.10 * observation.commitment_strain,
                0.0, 1.0,
            ))
            temperature = float(np.clip(0.20 + 0.65 * incident.fragmentation + 0.15 * (self.regime in ("compound", "ood")), 0.1, 1.0))
            distributed_entropy = normalized_entropy(distributed.tolist())
            self.evaluator_entropy[incident_id] = normalized_entropy(local_center.tolist())
            thermo = V5ThermodynamicState(
                operational_energy=energy,
                mean_belief_entropy=float(np.mean(individual_entropies)),
                entropy_dispersion=float(np.std(individual_entropies)),
                js_disagreement=jensen_shannon([value.tolist() for value in final_beliefs]),
                distributed_entropy=distributed_entropy,
                entropy_slope=float(entropy_history[-1] - entropy_history[-2]),
                consensus_residual=residual,
                consensus_confidence=confidence,
                effective_temperature=temperature,
                free_energy=float(energy - temperature * distributed_entropy),
                sketch_messages=sketch_messages,
                sketch_bytes=sketch_bytes,
                sketch_latency=float(sketch_messages * 0.002 + (1.0 - availability) * 0.15),
                contributors=tuple(sorted(received)),
            )
            self.thermodynamics[incident_id] = thermo
            self.ledger.append(
                incident.disruption_step, "thermodynamic_state", "distributed_estimator",
                {"incident_id": incident_id, **asdict(thermo), "v5": True},
            )
            self.ledger.append(
                incident.disruption_step, "v5_sketch_accounting", "distributed_estimator",
                {
                    "incident_id": incident_id,
                    "policy": self.sketch_policy,
                    "messages": sketch_messages,
                    "bytes": sketch_bytes,
                    "latency": thermo.sketch_latency,
                    "counted_in_total_communication": True,
                },
            )

    def operator_features(self, incident_id: str) -> Dict[str, float]:
        requesting_agent = self.incident_agents[incident_id][0]
        observation = self.observations[requesting_agent]
        features = observation.kpis()
        features.update(self.thermodynamics[incident_id].deployable_features())
        return features

    def operator_view(self, incident_id: str, feature_names: Sequence[str]) -> Dict[str, Any]:
        allowed = self.operator_features(incident_id)
        view = {
            "cluster_id": self.cluster_id,
            "application": self.application,
            "regime": self.regime,
            "information_condition": self.information_condition,
            "incident_id": incident_id,
            "features": {name: float(allowed[name]) for name in feature_names},
            "operator_budget": self.operator_budget,
            "simulated_operator": True,
            "information_boundary": "requesting-agent KPIs plus authorized distributed summaries",
        }
        prohibited = {"true_mode", "correct_action", "fragmentation", "stochastic_tape", "base_loss"}
        if prohibited.intersection(view["features"]):
            raise RuntimeError("V5 operator-view privacy leak")
        self.ledger.append(
            self.incidents[incident_id].disruption_step, "v5_operator_view", "simulated_operator",
            {**view, "payload_sha256": payload_digest(view)},
        )
        return view

    def _action_noise(self, incident_id: str, action: str) -> Tuple[float, float]:
        index = stable_seed(self.seed, incident_id, action, "tape_index") % (len(self.stochastic_tape) - 1)
        return float(self.stochastic_tape[index]), float(self.stochastic_tape[index + 1])

    def action_effect(self, incident_id: str, action: str) -> V5ActionEffect:
        if action not in OPERATOR_ACTIONS:
            raise ValueError("unknown bounded V5 operator action")
        incident = self.incidents[incident_id]
        draw, verification_draw = self._action_noise(incident_id, action)
        base = incident.base_loss
        action_cost = {
            "verify": 0.055,
            "request_peer_evidence": 0.035,
            "authorize_emergency_resource": 0.105,
            "reroute_or_reconfigure": 0.080,
            "deploy_repair_capacity": 0.125,
            "isolate_or_quarantine": 0.095,
            "revise_commitment": 0.045,
            "defer": 0.025,
            "no_action": 0.0,
        }[action]
        delay = {
            "verify": 2, "request_peer_evidence": 1,
            "authorize_emergency_resource": 2, "reroute_or_reconfigure": 2,
            "deploy_repair_capacity": 3, "isolate_or_quarantine": 1,
            "revise_commitment": 1, "defer": 3, "no_action": 0,
        }[action]
        potential = (0.24 + 0.48 * incident.severity) * (0.68 + 0.32 * incident.priority)
        correct = action == incident.correct_action
        secondary = action == incident.secondary_action
        if action == "verify":
            # Verification is imperfect and delayed. It can unlock a later
            # autonomous response, but it is never a zero-cost oracle.
            verification_correct = verification_draw < (
                0.82 if incident.true_mode == "evidence_conflict" else 0.72
            )
            gross = potential * (0.64 if verification_correct else 0.08)
            if not correct:
                gross *= 0.42
        elif action == "request_peer_evidence":
            gross = potential * (0.48 if secondary or correct else 0.12)
        elif correct:
            gross = potential * (0.78 + 0.18 * draw)
        elif secondary:
            gross = potential * (0.32 + 0.22 * draw)
        elif action in ("defer", "no_action"):
            gross = 0.0
        else:
            gross = -min(0.24, 0.035 + 0.15 * draw + 0.06 * incident.severity)
        delay_penalty = 0.012 * delay * (0.7 + incident.severity)
        loss_with = float(max(0.0, base - gross + action_cost + delay_penalty))
        causal = float(base - loss_with)
        material = action not in ("verify", "request_peer_evidence", "defer", "no_action")
        information_response = action in ("verify", "request_peer_evidence") and causal > 0.0
        accepted = (material and (correct or secondary or draw > 0.18)) or information_response
        next_stage = accepted and draw > 0.10
        reaches = next_stage and causal > 0.0 and draw > 0.16
        changed_commitment = action == "revise_commitment" and accepted
        return V5ActionEffect(
            incident_id=incident_id,
            action=action,
            loss_without=float(base),
            loss_with=loss_with,
            causal_effect=causal,
            intervention_cost=float(action_cost),
            operator_minutes=float({"verify": 6, "request_peer_evidence": 4, "defer": 2, "no_action": 0}.get(action, 6 + delay)),
            delay_steps=delay,
            beneficial=causal > 1e-9,
            harmful=causal < -1e-9,
            changed_commitment=changed_commitment,
            accepted_action=accepted,
            reached_next_stage=next_stage,
            reached_service=reaches,
            stochastic_tape_digest=self.stochastic_tape_digest,
        )

    def preview_communication_action(self, incident_id: str) -> str:
        """Pure preview of the decentralized negotiated default action."""

        agent_ids = self.incident_agents[incident_id]
        agents = [self.agents[value] for value in agent_ids]
        local_actions = [agent.preferred_action() for agent in agents]
        distributions = np.vstack([
            np.asarray(agent.private_beliefs[incident_id], dtype=float) for agent in agents
        ])
        weights = np.asarray([
            max(0.05, self.observations[agent.agent_id].telemetry_confidence)
            for agent in agents
        ], dtype=float)
        pooled = np.average(distributions, axis=0, weights=weights)
        pooled /= pooled.sum()
        proposed_action = agents[0].preferred_action(pooled)
        commitment = V5Commitment(
            commitment_id="PREVIEW-%s" % incident_id,
            proposer=agent_ids[0], recipient=agent_ids[1],
            incident_id=incident_id, action=proposed_action, quantity=1.0,
        )
        decisions = [agent.evaluate_commitment(commitment) for agent in agents[1:]]
        if decisions.count("reject") < len(decisions):
            return proposed_action
        counts = {action: local_actions.count(action) for action in set(local_actions)}
        return sorted(counts, key=lambda value: (-counts[value], value))[0]

    def candidate_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for incident_id, incident in self.incidents.items():
            features = self.operator_features(incident_id)
            self.operator_view(incident_id, tuple(features))
            autonomous_action = self.preview_communication_action(incident_id)
            autonomous_effect = self.action_effect(incident_id, autonomous_action)
            for action in OPERATOR_ACTIONS:
                absolute_effect = self.action_effect(incident_id, action)
                candidate_loss = float(absolute_effect.loss_with)
                # Public information makes autonomous action choice more
                # accurate, but does not pre-resolve the incident. A compatible
                # secondary authorization can still accelerate or reinforce a
                # correct autonomous response at explicit cost.
                if (
                    autonomous_action == incident.correct_action
                    and action == incident.secondary_action
                ):
                    complementary_gain = 0.035 + 0.11 * incident.severity
                    candidate_loss = max(0.0, float(autonomous_effect.loss_with) - complementary_gain)
                causal_effect = float(autonomous_effect.loss_with - candidate_loss)
                effect = {
                    **absolute_effect.as_dict(),
                    "loss_without": float(autonomous_effect.loss_with),
                    "loss_with": candidate_loss,
                    "causal_effect": causal_effect,
                    "beneficial": causal_effect > 1e-9,
                    "harmful": causal_effect < -1e-9,
                }
                row = {
                    "cluster_id": self.cluster_id,
                    "application": self.application,
                    "regime": self.regime,
                    "information_condition": self.information_condition,
                    "environment_seed": self.seed,
                    "topology_family": incident.topology_family,
                    "scenario_family": incident.scenario_family,
                    "incident_id": incident_id,
                    "action": action,
                    "candidate_id": "%s|%s|%s" % (self.cluster_id, incident_id, action),
                    "sketch_policy": self.sketch_policy,
                    "autonomous_action": autonomous_action,
                    "absolute_loss_no_action": float(absolute_effect.loss_without),
                    "absolute_action_effect": float(absolute_effect.causal_effect),
                    "evaluator_global_entropy": float(self.evaluator_entropy[incident_id]),
                    "distributed_entropy_error": float(abs(features["distributed_entropy"] - self.evaluator_entropy[incident_id])),
                    **features,
                    **effect,
                }
                rows.append(row)
                self.ledger.append(
                    incident.disruption_step, "v5_candidate_intervention", "evaluator",
                    {
                        "candidate_id": row["candidate_id"],
                        "incident_id": incident_id,
                        "action": action,
                        "effect": effect,
                        "analysis_only": True,
                    },
                    private_to="evaluator",
                )
                self.ledger.append(
                    incident.disruption_step, "counterfactual_branch", "evaluator",
                    {
                        "candidate_id": row["candidate_id"],
                        "rng_digest_with": self.stochastic_tape_digest,
                        "rng_digest_without": self.stochastic_tape_digest,
                        "loss_with": effect["loss_with"],
                        "loss_without": effect["loss_without"],
                        "causal_effect": effect["causal_effect"],
                        "analysis_only": True,
                    },
                    private_to="evaluator",
                )
        return rows

    def negotiate_incident(self, incident_id: str, communication: bool) -> Dict[str, Any]:
        agent_ids = self.incident_agents[incident_id]
        agents = [self.agents[value] for value in agent_ids]
        local_actions = [agent.preferred_action() for agent in agents]
        if not communication:
            selected_action = local_actions[0]
            return {
                "selected_action": selected_action,
                "local_actions": local_actions,
                "messages": 0,
                "bytes": 0,
                "commitment_revised": False,
            }
        distributions = np.vstack([
            np.asarray(agent.private_beliefs[incident_id], dtype=float) for agent in agents
        ])
        weights = np.asarray([
            max(0.05, self.observations[agent.agent_id].telemetry_confidence)
            for agent in agents
        ], dtype=float)
        pooled = np.average(distributions, axis=0, weights=weights)
        pooled /= pooled.sum()
        proposed_action = agents[0].preferred_action(pooled)
        commitment = V5Commitment(
            commitment_id="C-%s" % incident_id,
            proposer=agent_ids[0], recipient=agent_ids[1],
            incident_id=incident_id, action=proposed_action, quantity=1.0,
        )
        decisions: List[str] = []
        for sender, recipient in zip(agent_ids, agent_ids[1:] + agent_ids[:1]):
            payload = {
                "incident_id": incident_id,
                "proposed_action": proposed_action,
                "belief_summary": [round(float(value), 4) for value in pooled],
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            message = Message(
                message_id="M-%s-%s" % (incident_id, sender),
                sender=sender, recipient=recipient,
                kind="negotiation", payload=payload,
                sent_step=self.incidents[incident_id].disruption_step,
                deliver_step=self.incidents[incident_id].disruption_step + 1,
            )
            self.agents[recipient].receive(message)
            self.ledger.append(message.sent_step, "message", sender, {"message": asdict(message), "bytes": len(encoded.encode("utf-8")), "v5": True})
            self.ledger.append(message.deliver_step, "message_delivery", "network", {"message_id": message.message_id, "recipient": recipient, "v5": True}, private_to=recipient)
            self.operational_messages += 1
            self.operational_bytes += len(encoded.encode("utf-8"))
        for agent in agents[1:]:
            decision = agent.evaluate_commitment(commitment)
            updated = agent.apply_commitment(commitment, decision)
            decisions.append(decision)
            self.commitment_events += 1
            kind = "counteroffer" if decision == "counter" else "commitment"
            self.ledger.append(
                self.incidents[incident_id].disruption_step, kind, agent.agent_id,
                {"commitment": asdict(updated), "decision": decision, "v5": True},
            )
        # Rejection does not cause a central override. Agents fall back to the
        # majority of independently preferred actions, preserving authority.
        if decisions.count("reject") == len(decisions):
            counts = {action: local_actions.count(action) for action in set(local_actions)}
            selected_action = sorted(counts, key=lambda value: (-counts[value], value))[0]
        else:
            selected_action = proposed_action
        return {
            "selected_action": selected_action,
            "local_actions": local_actions,
            "messages": self.operational_messages,
            "bytes": self.operational_bytes,
            "commitment_revised": "counter" in decisions,
            "decisions": decisions,
        }

    def autonomous_outcome(self, communication: bool) -> Dict[str, Any]:
        effects: List[V5ActionEffect] = []
        changed = 0
        negotiations = 0
        revisions = 0
        for incident_id in self.incidents:
            decision = self.negotiate_incident(incident_id, communication)
            effect = self.action_effect(incident_id, decision["selected_action"])
            effects.append(effect)
            changed += int(effect.causal_effect != 0.0)
            negotiations += int(communication)
            revisions += int(decision.get("commitment_revised", False))
            self.ledger.append(
                self.incidents[incident_id].disruption_step, "restoration_action", self.incident_agents[incident_id][0],
                {
                    "incident_id": incident_id,
                    "application": self.application,
                    "action": decision["selected_action"],
                    "accepted": effect.accepted_action,
                    "reached_service": effect.reached_service,
                    "communication": communication,
                    "v5": True,
                },
            )
            if effect.reached_next_stage:
                self.ledger.append(
                    self.incidents[incident_id].disruption_step + effect.delay_steps,
                    "material_progress", self.incident_agents[incident_id][0],
                    {"incident_id": incident_id, "action": effect.action, "stage": "service" if effect.reached_service else "next_physical_stage", "v5": True},
                )
        return {
            "loss": float(sum(value.loss_with for value in effects)),
            "loss_without_actions": float(sum(value.loss_without for value in effects)),
            "causal_improvement": float(sum(value.causal_effect for value in effects)),
            "changed_incidents": changed,
            "negotiations": negotiations,
            "commitment_revisions": revisions,
            "accepted_actions": sum(int(value.accepted_action) for value in effects),
            "service_reaching_actions": sum(int(value.reached_service) for value in effects),
            "operational_messages": self.operational_messages,
            "operational_bytes": self.operational_bytes,
        }

    def conservation_report(self) -> Dict[str, Any]:
        residuals = {
            key: abs(self.resource_initial[key] - self.resource_used[key] - (self.resource_initial[key] - self.resource_used[key]))
            for key in self.resource_initial
        }
        feasible = all(
            self.resource_used[key] >= -1e-12 and self.resource_used[key] <= self.resource_initial[key] + 1e-12
            for key in self.resource_initial
        )
        return {
            "feasible": feasible,
            "residuals": residuals,
            "maximum_residual": max(residuals.values()) if residuals else 0.0,
        }

    def summary(self) -> Dict[str, Any]:
        thermo_messages = sum(value.sketch_messages for value in self.thermodynamics.values())
        thermo_bytes = sum(value.sketch_bytes for value in self.thermodynamics.values())
        return {
            "cluster_id": self.cluster_id,
            "application": self.application,
            "regime": self.regime,
            "information_condition": self.information_condition,
            "environment_seed": self.seed,
            "sketch_policy": self.sketch_policy,
            "incidents": len(self.incidents),
            "agents": len(self.agents),
            "sketch_messages": thermo_messages,
            "sketch_bytes": thermo_bytes,
            "sketch_latency": float(sum(value.sketch_latency for value in self.thermodynamics.values())),
            "operational_messages": self.operational_messages,
            "operational_bytes": self.operational_bytes,
            "maximum_conservation_residual": self.conservation_report()["maximum_residual"],
            "conservation_feasible": self.conservation_report()["feasible"],
            "event_count": len(self.ledger.events),
            "event_ledger_digest": self.ledger.digest(),
            "stochastic_tape_digest": self.stochastic_tape_digest,
        }
