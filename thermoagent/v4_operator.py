"""Bounded simulated-operator attention allocation for ThermoHITL v4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .events import EventLedger
from .v4_environment import FragmentedOversightEnvironment, IncidentState
from .v4_types import (
    AttentionRequestV4,
    OPERATOR_VIEW_FIELDS,
    OperatorInterventionV4,
    OperatorViewCondition,
    OperatorViewV4,
    ThermodynamicFeaturesV4,
    V4Application,
    V4Method,
    validate_operator_view_v4,
)


@dataclass(frozen=True)
class SimulatedOperatorProfileV4:
    name: str
    slots: int = 1
    base_accuracy: float = 0.90
    response_latency_steps: int = 1
    fatigue_sensitivity: float = 0.18
    workload_recovery: float = 0.10
    minutes_per_intervention: float = 8.0
    risk_aversion: float = 0.55


OPERATOR_PROFILES_V4: Dict[str, SimulatedOperatorProfileV4] = {
    "high_accuracy_bounded": SimulatedOperatorProfileV4("high_accuracy_bounded", base_accuracy=0.94),
    "fast_imperfect": SimulatedOperatorProfileV4(
        "fast_imperfect", base_accuracy=0.78, response_latency_steps=0,
        fatigue_sensitivity=0.22, minutes_per_intervention=5.0,
    ),
    "slow_accurate": SimulatedOperatorProfileV4(
        "slow_accurate", base_accuracy=0.96, response_latency_steps=2,
        fatigue_sensitivity=0.12, minutes_per_intervention=11.0,
    ),
    "fatigue_sensitive": SimulatedOperatorProfileV4(
        "fatigue_sensitive", base_accuracy=0.91, fatigue_sensitivity=0.38,
        workload_recovery=0.07,
    ),
    "risk_averse": SimulatedOperatorProfileV4(
        "risk_averse", base_accuracy=0.90, risk_aversion=0.82,
        minutes_per_intervention=9.0,
    ),
    "oracle": SimulatedOperatorProfileV4("oracle", base_accuracy=1.0, response_latency_steps=0),
}


METHOD_VIEW_V4 = {
    V4Method.LOCAL_KPI_TRIGGER.value: OperatorViewCondition.KPI_ONLY,
    V4Method.KPI_CAUSAL_TRIAGE.value: OperatorViewCondition.KPI_ONLY,
    V4Method.ENERGY_TRIAGE.value: OperatorViewCondition.ENERGY_ONLY,
    V4Method.ENTROPY_DISAGREEMENT_TRIAGE.value: OperatorViewCondition.ENTROPY_DISAGREEMENT,
    V4Method.KPI_ENERGY_TRIAGE.value: OperatorViewCondition.KPI_ENERGY,
    V4Method.KPI_THERMO_TRIAGE.value: OperatorViewCondition.KPI_THERMO,
    V4Method.COMPLETE_THERMO_TRIAGE.value: OperatorViewCondition.COMPLETE_THERMO,
    V4Method.THERMOHITL_RULE.value: OperatorViewCondition.COMPLETE_THERMO,
    V4Method.LEARNED_NON_THERMO.value: OperatorViewCondition.KPI_ONLY,
    V4Method.THERMOHITL_RL.value: OperatorViewCondition.COMPLETE_THERMO,
    V4Method.BOUNDED_ORACLE.value: OperatorViewCondition.ORACLE,
    V4Method.FULL_ORACLE.value: OperatorViewCondition.ORACLE,
}


def request_score_v4(
    method: str,
    local_kpis: Mapping[str, float],
    thermo: ThermodynamicFeaturesV4,
    workload: float,
) -> float:
    severity = (
        0.42 * float(local_kpis["service_deficit"])
        + 0.18 * float(local_kpis["backlog"])
        + 0.14 * float(local_kpis["safety_stress"])
        + 0.12 * float(local_kpis["resource_scarcity"])
        + 0.14 * float(local_kpis["actionability_flag"])
    )
    if method in {V4Method.LOCAL_KPI_TRIGGER.value, V4Method.KPI_CAUSAL_TRIAGE.value, V4Method.LEARNED_NON_THERMO.value}:
        return 2.35 * severity - 0.20 * workload
    if method == V4Method.ENERGY_TRIAGE.value:
        return 0.72 * max(0.0, thermo.standardized_energy) - 0.20 * workload
    if method == V4Method.ENTROPY_DISAGREEMENT_TRIAGE.value:
        return 0.48 * thermo.entropy_anomaly + 2.80 * thermo.belief_disagreement - 0.24 * (1.0 - thermo.consensus_confidence)
    if method == V4Method.KPI_ENERGY_TRIAGE.value:
        return 1.35 * severity + 0.38 * max(0.0, thermo.standardized_energy) - 0.20 * workload
    if method in {V4Method.KPI_THERMO_TRIAGE.value, V4Method.COMPLETE_THERMO_TRIAGE.value, V4Method.THERMOHITL_RULE.value, V4Method.THERMOHITL_RL.value}:
        return (
            1.05 * severity
            + 0.30 * max(0.0, thermo.standardized_energy)
            + 0.20 * thermo.entropy_anomaly
            + 0.10 * max(0.0, thermo.entropy_slope / 0.05)
            + 2.80 * thermo.belief_disagreement
            - 0.25 * (1.0 - thermo.consensus_confidence)
            - 0.20 * workload
        )
    if method in {V4Method.BOUNDED_ORACLE.value, V4Method.FULL_ORACLE.value}:
        return 3.0 * severity + 2.0 * thermo.belief_disagreement
    return 0.0


class SimulatedOperatorV4:
    """Capacity-constrained simulated operator; never a real-human claim."""

    def __init__(
        self,
        profile: SimulatedOperatorProfileV4,
        seed: int,
        intervention_budget: int = 2,
    ) -> None:
        self.profile = profile
        self.rng = np.random.RandomState(int(seed))
        self.intervention_budget = int(intervention_budget)
        self.queue: List[AttentionRequestV4] = []
        self.active_until: List[int] = []
        self.workload = 0.0
        self.fatigue = 0.0
        self.operator_minutes = 0.0
        self.interventions = 0
        self.false_or_harmful = 0
        self.queue_wait_steps: List[int] = []
        self._intervention_counter = 0

    @property
    def available_slots(self) -> int:
        return max(0, self.profile.slots - len(self.active_until))

    def enqueue(self, request: AttentionRequestV4, ledger: EventLedger) -> None:
        if any(value.request_id == request.request_id for value in self.queue):
            return
        self.queue.append(request)
        ledger.append(request.step, "operator_queue", request.requesting_agent, {
            "action": "enqueued",
            "request": request.as_dict(),
            "queue_length": len(self.queue),
            "v4": True,
        })

    def advance_workload(self, step: int) -> None:
        self.active_until = [value for value in self.active_until if value > step]
        self.workload = max(0.0, self.workload - self.profile.workload_recovery)
        self.fatigue = max(0.0, self.fatigue - 0.5 * self.profile.workload_recovery)

    def allocate(
        self,
        step: int,
        policy: str,
        request_features: Mapping[str, ThermodynamicFeaturesV4],
        incidents: Mapping[str, IncidentState],
        ledger: EventLedger,
    ) -> List[AttentionRequestV4]:
        self.advance_workload(step)
        if self.interventions >= self.intervention_budget or self.available_slots <= 0 or not self.queue:
            return []

        def key(request: AttentionRequestV4) -> Tuple[float, str]:
            thermo = request_features[request.incident_id]
            incident = incidents[request.incident_id]
            if policy == "first_come_first_served":
                return (-float(request.step), request.request_id)
            if policy == "highest_energy":
                return (thermo.operational_energy, request.request_id)
            if policy == "highest_entropy_anomaly":
                return (thermo.entropy_anomaly, request.request_id)
            if policy == "highest_disagreement":
                return (thermo.belief_disagreement, request.request_id)
            if policy == "random":
                return (float(self.rng.rand()), request.request_id)
            if policy in {"local_kpi", "learned_non_thermo"}:
                return (request.severity + 0.25 * incident.commitment_strain, request.request_id)
            if policy == "oracle":
                return ((1.0 if incident.ambiguous and not incident.verified else 0.0) + incident.criticality, request.request_id)
            # Thermodynamic expected benefit per simulated-operator minute.
            priority = request.priority_score * thermo.consensus_confidence
            return (priority / max(request.estimated_operator_minutes, 1e-9), request.request_id)

        selected = sorted(self.queue, key=key, reverse=True)[: self.available_slots]
        for request in selected:
            self.queue.remove(request)
            self.queue_wait_steps.append(max(0, step - request.step))
            ledger.append(step, "attention_allocation", "simulated_operator", {
                "request_id": request.request_id,
                "incident_id": request.incident_id,
                "policy": policy,
                "queue_wait_steps": step - request.step,
                "available_slots_before": self.available_slots,
                "v4": True,
            })
            ledger.append(step, "attention_decision_v4", "simulated_operator", {
                "request_id": request.request_id,
                "incident_id": request.incident_id,
                "allocation_policy": policy,
                "selected": True,
            })
        return selected

    def build_view(
        self,
        environment: FragmentedOversightEnvironment,
        request: AttentionRequestV4,
        thermo: ThermodynamicFeaturesV4,
        condition: OperatorViewCondition,
    ) -> OperatorViewV4:
        incident = environment.incidents[request.incident_id]
        local = next(
            agent.vault.observation(agent.agent_id)
            for agent in environment.agents.values()
            if agent.agent_id == request.requesting_agent
        )
        available: Dict[str, Any] = {
            **local.local_kpis(),
            "operational_energy": thermo.operational_energy,
            "standardized_energy": thermo.standardized_energy,
            "belief_entropy": thermo.belief_entropy,
            "alternative_entropy": thermo.alternative_entropy,
            "commitment_entropy": thermo.commitment_entropy,
            "distributed_entropy": thermo.distributed_entropy,
            "entropy_residual": thermo.entropy_residual,
            "entropy_anomaly": thermo.entropy_anomaly,
            "entropy_slope": thermo.entropy_slope,
            "entropy_acceleration": thermo.entropy_acceleration,
            "belief_disagreement": thermo.belief_disagreement,
            "consensus_confidence": thermo.consensus_confidence,
            "temperature_diagnostic": thermo.temperature_diagnostic,
            "free_energy_diagnostic": thermo.free_energy_diagnostic,
            "free_energy_residual": thermo.free_energy_residual,
        }
        if condition == OperatorViewCondition.ORACLE:
            available["oracle_state"] = {
                "true_mode": incident.true_mode,
                "ambiguous": incident.ambiguous,
                "resource_required": incident.resource_required,
            }
        allowed = set(OPERATOR_VIEW_FIELDS[condition])
        features = {key: value for key, value in available.items() if key in allowed}
        nodes = [
            {
                "agent_id": agent.agent_id,
                "role": agent.identity.role,
                "location": list(agent.identity.location),
                "incident_scope": list(agent.identity.incident_scope),
            }
            for agent in environment.agents.values()
        ]
        public_network = {
            "nodes": nodes,
            "service_edges": [list(edge) for edge in environment.service_edges],
            "communication_edges": [list(edge) for edge in environment.communication_edges],
            "logistics_edges": [list(edge) for edge in environment.logistics_edges],
            "authorized_emergency_edges": [list(key) for key in environment.authorized_edges],
            "visible_incidents": [
                {
                    "incident_id": value.incident_id,
                    "location": list(value.location),
                    "service_deficit": value.service_deficit,
                    "visible_collapse": value.visible_collapse,
                    "telemetry_confidence_state": (
                        "low" if thermo.consensus_confidence < 0.45 else "bounded_distributed_estimate"
                    ),
                }
                for value in environment.incidents.values()
            ],
        }
        view = OperatorViewV4(
            condition=condition.value,
            step=environment.step_index,
            application=environment.application,
            incident_id=request.incident_id,
            features=features,
            alert={
                "request_id": request.request_id,
                "requesting_agent": request.requesting_agent,
                "reason_code": request.reason_code,
                "severity": request.severity,
                "predicted_benefit": request.predicted_benefit,
                "prediction_uncertainty": request.uncertainty,
                "estimated_operator_minutes": request.estimated_operator_minutes,
                "priority_score": request.priority_score,
                "predicted_steps_until_collapse": request.predicted_steps_until_collapse,
            },
            public_network=public_network,
            workload={
                "workload": self.workload,
                "fatigue": self.fatigue,
                "queue_length": len(self.queue),
                "available_slots": self.available_slots,
                "operator_minutes": self.operator_minutes,
                "intervention_budget_remaining": max(0, self.intervention_budget - self.interventions),
            },
            provenance={
                "information_boundary": "requesting_agent_local_kpis_plus_authorized_distributed_sketch_fields",
                "timestamp_step": environment.step_index,
                "sketch_contributors": thermo.sketch_contributors,
                "evaluator_only_fields_excluded": condition != OperatorViewCondition.ORACLE,
            },
            oracle=condition == OperatorViewCondition.ORACLE,
        )
        validate_operator_view_v4(view)
        return view

    def choose_intervention(
        self,
        environment: FragmentedOversightEnvironment,
        request: AttentionRequestV4,
        view: OperatorViewV4,
    ) -> OperatorInterventionV4:
        incident = environment.incidents[request.incident_id]
        features = view.features
        effective_accuracy = max(
            0.5,
            self.profile.base_accuracy - self.profile.fatigue_sensitivity * self.fatigue,
        )
        correct = bool(self.rng.rand() <= effective_accuracy) or view.oracle
        disagreement = float(features.get("belief_disagreement", 0.0))
        anomaly = float(features.get("entropy_anomaly", 0.0))
        confidence = float(features.get("consensus_confidence", 1.0))
        actionability = float(features.get("actionability_flag", 0.0))
        energy = float(features.get("standardized_energy", 0.0))
        if view.oracle:
            action = "authorize_verification" if incident.ambiguous else "authorize_emergency_resource"
        elif (disagreement >= 0.08 or anomaly >= 1.15 or confidence < 0.55 or actionability >= 0.85):
            action = "authorize_verification"
        elif energy >= 1.10 or request.severity >= 0.45:
            action = "authorize_emergency_resource"
        else:
            action = "authorize_emergency_logistics_edge"
        if not correct:
            # Imperfect simulated operators remain bounded: an error is an
            # unnecessary resource/edge decision, never knowledge of a hidden future.
            action = "authorize_emergency_resource" if action == "authorize_verification" else "authorize_verification"
        if confidence < 0.25 and action not in {"authorize_verification", "authorize_information_sharing"}:
            action = "abstain"
        self._intervention_counter += 1
        return OperatorInterventionV4(
            intervention_id="V4HI%06d" % self._intervention_counter,
            incident_id=request.incident_id,
            step=environment.step_index,
            action=action,
            target_agent=request.requesting_agent,
            arguments={"bounded": True, "source_view_sha256": view.digest()},
            mandatory=False,
            service_steps=self.profile.response_latency_steps + 1,
            estimated_minutes=self.profile.minutes_per_intervention,
            reason_code=request.reason_code,
        )

    def register_intervention(self, step: int, intervention: OperatorInterventionV4) -> None:
        self.interventions += 1
        self.operator_minutes += intervention.estimated_minutes
        self.workload = min(1.0, self.workload + 0.34)
        self.fatigue = min(1.0, self.fatigue + 0.16 + 0.08 * self.workload)
        self.active_until.append(step + intervention.service_steps)


def view_condition_for_method(method: str) -> OperatorViewCondition:
    return METHOD_VIEW_V4.get(method, OperatorViewCondition.KPI_ONLY)


def allocation_policy_for_method(method: str) -> str:
    if method == V4Method.PERIODIC_REVIEW.value:
        return "first_come_first_served"
    if method == V4Method.RANDOM_REVIEW.value:
        return "random"
    if method in {
        V4Method.LOCAL_KPI_TRIGGER.value,
        V4Method.KPI_CAUSAL_TRIAGE.value,
        V4Method.KPI_ENERGY_TRIAGE.value,
        V4Method.LEARNED_NON_THERMO.value,
    }:
        return "local_kpi"
    if method == V4Method.ENERGY_TRIAGE.value:
        return "highest_energy"
    if method == V4Method.ENTROPY_DISAGREEMENT_TRIAGE.value:
        return "highest_disagreement"
    if method in {V4Method.BOUNDED_ORACLE.value, V4Method.FULL_ORACLE.value}:
        return "oracle"
    return "thermodynamic_expected_benefit"
