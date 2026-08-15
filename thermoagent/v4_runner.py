"""Episode execution and paired causal branching for ThermoHITL v4."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .v4_environment import FragmentedOversightEnvironment
from .v4_operator import (
    OPERATOR_PROFILES_V4,
    SimulatedOperatorV4,
    allocation_policy_for_method,
    request_score_v4,
    view_condition_for_method,
)
from .v4_types import (
    AttentionRequestV4,
    CausalChainV4,
    InformationCondition,
    OperatorInterventionV4,
    OperatorViewCondition,
    ThermodynamicFeaturesV4,
    V4Application,
    V4Method,
)


HUMAN_METHODS = {
    V4Method.PERIODIC_REVIEW.value,
    V4Method.RANDOM_REVIEW.value,
    V4Method.LOCAL_KPI_TRIGGER.value,
    V4Method.KPI_CAUSAL_TRIAGE.value,
    V4Method.ENERGY_TRIAGE.value,
    V4Method.ENTROPY_DISAGREEMENT_TRIAGE.value,
    V4Method.KPI_ENERGY_TRIAGE.value,
    V4Method.KPI_THERMO_TRIAGE.value,
    V4Method.COMPLETE_THERMO_TRIAGE.value,
    V4Method.THERMOHITL_RULE.value,
    V4Method.LEARNED_NON_THERMO.value,
    V4Method.THERMOHITL_RL.value,
    V4Method.BOUNDED_ORACLE.value,
    V4Method.FULL_ORACLE.value,
}


@dataclass(frozen=True)
class V4EpisodeConfig:
    application: str
    regime: str
    information_condition: str
    method: str
    environment_seed: int
    operator_seed: int
    planner_seed: int = 0
    rl_seed: Optional[int] = None
    horizon: int = 20
    disruption_step: int = 6
    operator_profile: str = "high_accuracy_bounded"
    operator_budget: int = 2
    counterfactual_probes: bool = True
    dense_candidates: bool = False
    stage: str = "development"

    @property
    def run_id(self) -> str:
        return (
            "%s-%s-%s-%s-%s-e%d-o%d-r%s"
            % (
                self.stage,
                self.application,
                self.method,
                self.regime,
                self.information_condition,
                self.environment_seed,
                self.operator_seed,
                "none" if self.rl_seed is None else self.rl_seed,
            )
        )


@dataclass
class V4EpisodeResult:
    run_id: str
    application: str
    regime: str
    information_condition: str
    method: str
    environment_seed: int
    operator_seed: int
    rl_seed: Optional[int]
    status: str
    metrics: Dict[str, Any]
    time_series: List[Dict[str, Any]]
    candidate_interventions: List[Dict[str, Any]]
    counterfactuals: List[Dict[str, Any]]
    manifest_fields: Dict[str, Any]
    ledger: Any

    def episode_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "application": self.application,
            "regime": self.regime,
            "information_condition": self.information_condition,
            "method": self.method,
            "environment_seed": self.environment_seed,
            "operator_seed": self.operator_seed,
            "rl_seed": self.rl_seed,
            "status": self.status,
            "metrics": self.metrics,
            "time_series": self.time_series,
            "candidate_interventions": self.candidate_interventions,
            "counterfactuals": self.counterfactuals,
            "manifest_fields": self.manifest_fields,
            "event_ledger_digest": self.ledger.digest(),
            "evidence_boundary": "simulated operator; deterministic independent agents unless planner metadata says real Qwen",
        }


class V4EpisodeRunner:
    def __init__(self, config: V4EpisodeConfig) -> None:
        self.config = config
        communication = config.method not in {V4Method.NO_COMMUNICATION.value}
        self.environment = FragmentedOversightEnvironment(
            application=config.application,
            regime=config.regime,
            information_condition=config.information_condition,
            seed=config.environment_seed,
            horizon=config.horizon,
            disruption_step=config.disruption_step,
            communication_enabled=communication,
        )
        self.operator = SimulatedOperatorV4(
            OPERATOR_PROFILES_V4[config.operator_profile],
            seed=config.operator_seed,
            intervention_budget=config.operator_budget,
        )
        self.request_counter = 0
        self.dense_candidate_counter = 0
        self.agent_trigger_active: Dict[str, bool] = {agent_id: False for agent_id in self.environment.agents}
        self.agent_trigger_since: Dict[str, Optional[int]] = {agent_id: None for agent_id in self.environment.agents}
        self.agent_last_request: Dict[str, Optional[int]] = {agent_id: None for agent_id in self.environment.agents}
        self.responded_incidents: Set[str] = set()
        self.response_correct_by_incident: Dict[str, bool] = {}
        self.first_request_step: Optional[int] = None
        self.intervention_steps: List[int] = []
        self.counterfactuals: List[Dict[str, Any]] = []
        self.candidate_interventions: List[Dict[str, Any]] = []
        self._dense_candidates_recorded = False
        self.low_confidence_operator_decisions = 0
        self.safe_low_confidence_decisions = 0

    @staticmethod
    def _request_threshold(method: str) -> float:
        if method in {V4Method.LOCAL_KPI_TRIGGER.value, V4Method.KPI_CAUSAL_TRIAGE.value, V4Method.LEARNED_NON_THERMO.value}:
            return 0.72
        if method == V4Method.ENERGY_TRIAGE.value:
            return 0.90
        return 1.15

    def _eligible_agent_for_incident(self, incident_id: str) -> str:
        scoped = [
            agent for agent in self.environment.agents.values()
            if incident_id in agent.identity.incident_scope
            and agent.vault.observation(agent.agent_id).incident_id == incident_id
        ]
        if not scoped:
            raise RuntimeError("no agent currently observes incident %s" % incident_id)
        return max(
            scoped,
            key=lambda agent: (
                agent.vault.observation(agent.agent_id).local_service_deficit
                * agent.vault.observation(agent.agent_id).private_priority,
                agent.agent_id,
            ),
        ).agent_id

    def _build_request(
        self,
        agent_id: str,
        incident_id: str,
        thermo: ThermodynamicFeaturesV4,
        score: float,
        count_as_request: bool = True,
    ) -> AttentionRequestV4:
        observation = self.environment.agents[agent_id].vault.observation(agent_id)
        if count_as_request:
            self.request_counter += 1
            request_id = "V4HR%07d" % self.request_counter
        else:
            self.dense_candidate_counter += 1
            request_id = "V4DC%07d" % self.dense_candidate_counter
        severity = (
            0.55 * observation.local_service_deficit
            + 0.25 * observation.local_backlog
            + 0.20 * observation.local_safety_stress
        )
        predicted_benefit = max(
            0.0,
            0.55 * severity
            + 0.20 * thermo.entropy_anomaly
            + 0.65 * thermo.belief_disagreement
            - 0.18 * (1.0 - thermo.consensus_confidence),
        )
        reason = "severity"
        action = "authorize_emergency_resource"
        if thermo.belief_disagreement >= 0.08 or thermo.entropy_anomaly >= 1.15:
            reason = "fragmented_belief_disagreement"
            action = "authorize_verification"
        if thermo.consensus_confidence < 0.45:
            reason = "low_consensus_confidence"
            action = "authorize_verification"
        return AttentionRequestV4(
            request_id=request_id,
            incident_id=incident_id,
            requesting_agent=agent_id,
            step=self.environment.step_index,
            reason_code=reason,
            requested_action=action,
            severity=float(severity),
            predicted_benefit=float(predicted_benefit),
            uncertainty=float(1.0 - thermo.consensus_confidence),
            estimated_operator_minutes=self.operator.profile.minutes_per_intervention,
            priority_score=float(score),
            consensus_confidence=thermo.consensus_confidence,
            predicted_steps_until_collapse=max(1, int(round(5.0 * (1.0 - severity)))),
        )

    def _should_request(
        self,
        agent_id: str,
        thermo: ThermodynamicFeaturesV4,
    ) -> Tuple[bool, float]:
        method = self.config.method
        observation = self.environment.agents[agent_id].vault.observation(agent_id)
        score = request_score_v4(method, observation.local_kpis(), thermo, self.operator.workload)
        if method not in HUMAN_METHODS:
            return False, score
        if method == V4Method.PERIODIC_REVIEW.value:
            return self.environment.step_index % 4 == 0, score
        if method == V4Method.RANDOM_REVIEW.value:
            return bool(self.operator.rng.rand() < 0.13), score
        threshold = self._request_threshold(method)
        active = self.agent_trigger_active[agent_id]
        activated = False
        if not active and score >= threshold:
            self.agent_trigger_active[agent_id] = True
            self.agent_trigger_since[agent_id] = self.environment.step_index
            activated = True
        elif active:
            since = self.agent_trigger_since[agent_id]
            if since is not None and self.environment.step_index - since >= 2 and score <= 0.45:
                self.agent_trigger_active[agent_id] = False
                self.agent_trigger_since[agent_id] = None
        if not activated:
            return False, score
        last = self.agent_last_request[agent_id]
        if last is not None and self.environment.step_index - last < 3:
            return False, score
        self.agent_last_request[agent_id] = self.environment.step_index
        return True, score

    def _counterfactual_probe(
        self,
        request: AttentionRequestV4,
        intervention: OperatorInterventionV4,
        probe_horizon: int = 5,
    ) -> Dict[str, Any]:
        with_intervention = deepcopy(self.environment)
        without_intervention = deepcopy(self.environment)
        initial_rng_with = with_intervention.rng_digest()
        initial_rng_without = without_intervention.rng_digest()
        before_with = deepcopy(with_intervention.metric_counters)
        before_without = deepcopy(without_intervention.metric_counters)
        commitment_before_with = sum(len(agent.commitments) for agent in with_intervention.agents.values())
        commitment_before_without = sum(len(agent.commitments) for agent in without_intervention.agents.values())
        result = with_intervention.apply_operator_intervention(intervention)
        with_intervention.automated_response(
            request.incident_id,
            intervention_id=intervention.intervention_id,
            coordinated=with_intervention.communication_enabled,
        )
        without_intervention.automated_response(
            request.incident_id,
            coordinated=without_intervention.communication_enabled,
        )
        loss_with = 0.0
        loss_without = 0.0
        for _ in range(probe_horizon):
            loss_with += with_intervention.step()["loss"]
            loss_without += without_intervention.step()["loss"]
        accepted_with = with_intervention.metric_counters["material_actions_accepted"] - before_with["material_actions_accepted"]
        accepted_without = without_intervention.metric_counters["material_actions_accepted"] - before_without["material_actions_accepted"]
        service_with = with_intervention.metric_counters["material_actions_reached_service"] - before_with["material_actions_reached_service"]
        service_without = without_intervention.metric_counters["material_actions_reached_service"] - before_without["material_actions_reached_service"]
        commitments_with = sum(len(agent.commitments) for agent in with_intervention.agents.values()) - commitment_before_with
        commitments_without = sum(len(agent.commitments) for agent in without_intervention.agents.values()) - commitment_before_without
        effect = float(loss_without - loss_with)
        chain = CausalChainV4(
            request_entered_queue=True,
            allocator_selected=True,
            operator_received_authorized_view=True,
            operator_acted=result.ok,
            agent_commitment_changed=commitments_with != commitments_without,
            accepted_action_changed=accepted_with != accepted_without,
            material_or_service_flow_changed=service_with != service_without,
            reached_demand_or_critical_service=service_with > 0,
            primary_outcome_changed=abs(effect) > 1e-12,
            intervention_effect=effect,
            harmful=effect < -1e-12,
            common_randomness_verified=initial_rng_with == initial_rng_without,
        )
        row = {
            "request_id": request.request_id,
            "intervention_id": intervention.intervention_id,
            "incident_id": request.incident_id,
            "step": self.environment.step_index,
            "loss_with_intervention": loss_with,
            "loss_without_intervention": loss_without,
            "intervention_effect": effect,
            "operator_minutes": intervention.estimated_minutes,
            "causal_utility": effect - 0.0025 * intervention.estimated_minutes,
            "rng_digest_with": initial_rng_with,
            "rng_digest_without": initial_rng_without,
            "accepted_actions_with": accepted_with,
            "accepted_actions_without": accepted_without,
            "service_arrivals_with": service_with,
            "service_arrivals_without": service_without,
            "commitment_changes_with": commitments_with,
            "commitment_changes_without": commitments_without,
            **chain.as_dict(),
        }
        self.environment.ledger.append(
            self.environment.step_index,
            "counterfactual_snapshot",
            "evaluator",
            {
                "request_id": request.request_id,
                "incident_id": request.incident_id,
                "state_digest": self.environment.state_digest(),
                "rng_digest": self.environment.rng_digest(),
                "operator_view_unavailable_to_branches": True,
            },
        )
        self.environment.ledger.append(
            self.environment.step_index,
            "counterfactual_branch",
            "evaluator",
            row,
        )
        for stage, passed in (
            ("request_entered_queue", chain.request_entered_queue),
            ("allocator_selected", chain.allocator_selected),
            ("operator_action", chain.operator_acted),
            ("agent_commitment_or_action", chain.agent_commitment_changed or chain.accepted_action_changed),
            ("material_or_service_flow", chain.material_or_service_flow_changed),
            ("reached_demand_or_critical_service", chain.reached_demand_or_critical_service),
            ("primary_outcome_changed", chain.primary_outcome_changed),
        ):
            self.environment.ledger.append(
                self.environment.step_index,
                "intervention_causal_stage",
                "evaluator",
                {"intervention_id": intervention.intervention_id, "stage": stage, "passed": passed},
            )
        return row

    def _dense_candidate_rows(
        self,
        thermo_by_incident: Mapping[str, ThermodynamicFeaturesV4],
    ) -> None:
        if self._dense_candidates_recorded or not self.config.dense_candidates:
            return
        if self.environment.step_index < self.config.disruption_step + 1:
            return
        self._dense_candidates_recorded = True
        for incident_id, incident in self.environment.incidents.items():
            if not incident.active:
                continue
            agent_id = self._eligible_agent_for_incident(incident_id)
            thermo = thermo_by_incident[incident_id]
            observation = self.environment.agents[agent_id].vault.observation(agent_id)
            score = request_score_v4(
                V4Method.THERMOHITL_RULE.value,
                observation.local_kpis(),
                thermo,
                self.operator.workload,
            )
            request = self._build_request(
                agent_id, incident_id, thermo, score, count_as_request=False
            )
            condition = OperatorViewCondition.COMPLETE_THERMO
            view = self.operator.build_view(self.environment, request, thermo, condition)
            # The candidate action is fixed prospectively across views; models
            # rank incidents, not hindsight-select an action.
            intervention = OperatorInterventionV4(
                intervention_id="V4DENSE-%s" % request.request_id,
                incident_id=incident_id,
                step=self.environment.step_index,
                action="authorize_verification",
                target_agent=agent_id,
                arguments={"bounded": True, "source_view_sha256": view.digest()},
                mandatory=False,
                service_steps=2,
                estimated_minutes=8.0,
                reason_code="prospective_dense_verification_candidate",
            )
            probe = self._counterfactual_probe(request, intervention)
            row = {
                "run_id": self.config.run_id,
                "application": self.config.application,
                "regime": self.config.regime,
                "information_condition": self.config.information_condition,
                "environment_seed": self.config.environment_seed,
                "operator_seed": self.config.operator_seed,
                "incident_id": incident_id,
                "criticality": incident.criticality,
                **observation.local_kpis(),
                **thermo.as_dict(),
                "view_sha256": view.digest(),
                "operator_minutes": intervention.estimated_minutes,
                "intervention_effect": probe["intervention_effect"],
                "loss_with_intervention": probe["loss_with_intervention"],
                "loss_without_intervention": probe["loss_without_intervention"],
                "causal_utility": probe["causal_utility"],
                "beneficial": int(probe["causal_utility"] > 0.0),
                "harmful": int(probe["intervention_effect"] < -1e-12),
                "complete_causal_chain": int(probe["complete"]),
                "privileged_feature_used": False,
                "cluster_id": "%s|%s|%d|%s" % (
                    self.config.application,
                    self.config.regime,
                    self.config.environment_seed,
                    self.config.information_condition,
                ),
            }
            self.candidate_interventions.append(row)

    def _online_requests(
        self,
        thermo_by_incident: Mapping[str, ThermodynamicFeaturesV4],
    ) -> Set[str]:
        selected_incidents: Set[str] = set()
        if self.config.method not in HUMAN_METHODS:
            return selected_incidents
        requested_incidents: Set[str] = set()
        for incident_id, incident in self.environment.incidents.items():
            if not incident.active:
                continue
            agent_id = self._eligible_agent_for_incident(incident_id)
            should, score = self._should_request(agent_id, thermo_by_incident[incident_id])
            if not should or incident_id in requested_incidents:
                continue
            request = self._build_request(agent_id, incident_id, thermo_by_incident[incident_id], score)
            self.operator.enqueue(request, self.environment.ledger)
            self.environment.ledger.append(
                self.environment.step_index,
                "human_request",
                agent_id,
                {**request.as_dict(), "independent_agent_decision": True, "v4": True},
            )
            requested_incidents.add(incident_id)
            if self.first_request_step is None:
                self.first_request_step = self.environment.step_index
        selected = self.operator.allocate(
            self.environment.step_index,
            allocation_policy_for_method(self.config.method),
            thermo_by_incident,
            self.environment.incidents,
            self.environment.ledger,
        )
        for request in selected:
            condition = view_condition_for_method(self.config.method)
            view = self.operator.build_view(
                self.environment,
                request,
                thermo_by_incident[request.incident_id],
                condition,
            )
            self.environment.ledger.append(
                self.environment.step_index,
                "operator_view_v4",
                "dashboard",
                {"payload": view.as_dict(), "sha256": view.digest()},
            )
            self.environment.ledger.append(
                self.environment.step_index,
                "information_boundary_audit",
                "dashboard",
                {
                    "view_sha256": view.digest(),
                    "condition": view.condition,
                    "private_state_leak": False,
                    "future_state_leak": False,
                    "evaluator_global_leak": False if not view.oracle else "oracle_labeled",
                },
            )
            intervention = self.operator.choose_intervention(self.environment, request, view)
            confidence = float(view.features.get("consensus_confidence", 1.0))
            if confidence < 0.25:
                self.low_confidence_operator_decisions += 1
                if intervention.action in {
                    "abstain", "authorize_verification",
                    "authorize_information_sharing", "resolve_conflicting_reports",
                }:
                    self.safe_low_confidence_decisions += 1
            if self.config.counterfactual_probes:
                probe = self._counterfactual_probe(request, intervention)
                self.counterfactuals.append(probe)
            result = self.environment.apply_operator_intervention(intervention)
            self.environment.ledger.append(
                self.environment.step_index,
                "operator_result",
                "simulator",
                {"intervention_id": intervention.intervention_id, **result.as_dict()},
            )
            if result.ok and intervention.action != "abstain":
                self.operator.register_intervention(self.environment.step_index, intervention)
                self.intervention_steps.append(self.environment.step_index)
                selected_incidents.add(request.incident_id)
                # An operator response can arrive after the agent's first
                # autonomous material action.  When that action used the
                # wrong resource, the newly authorized evidence must cause a
                # bounded replan; otherwise the operator intervention would
                # change state but never enter the logistics causal chain.
                # Correct prior actions are not duplicated.
                if (
                    request.incident_id in self.responded_incidents
                    and not self.response_correct_by_incident.get(request.incident_id, False)
                ):
                    revised = self.environment.automated_response(
                        request.incident_id,
                        intervention_id=intervention.intervention_id,
                        coordinated=self.environment.communication_enabled,
                    )
                    self.environment.ledger.append(
                        self.environment.step_index,
                        "plan_revision",
                        request.requesting_agent,
                        {
                            "incident_id": request.incident_id,
                            "reason": "bounded_operator_evidence_after_failed_material_action",
                            "intervention_id": intervention.intervention_id,
                            "result": revised.as_dict(),
                        },
                    )
                    if revised.ok:
                        self.response_correct_by_incident[request.incident_id] = bool(
                            revised.data.get("correct_resource", False)
                        )
        return selected_incidents

    def run(self) -> V4EpisodeResult:
        started = time.perf_counter()
        time_series: List[Dict[str, Any]] = []
        for _ in range(self.config.horizon):
            transition = self.environment.step()
            self.environment.deliver_observations()
            thermo = self.environment.exchange_sketches(gossip_rounds=3)
            self._dense_candidate_rows(thermo)
            selected_incidents = self._online_requests(thermo)
            if self.environment.step_index == self.config.disruption_step + 1:
                for incident_id, incident in self.environment.incidents.items():
                    if not incident.active or incident_id in self.responded_incidents:
                        continue
                    result = self.environment.automated_response(
                        incident_id,
                        intervention_id=("operator_selected" if incident_id in selected_incidents else None),
                        coordinated=self.environment.communication_enabled,
                    )
                    if result.ok:
                        self.responded_incidents.add(incident_id)
                        self.response_correct_by_incident[incident_id] = bool(
                            result.data.get("correct_resource", False)
                        )
            active_features = [thermo[key] for key, value in self.environment.incidents.items() if value.active and key in thermo]
            time_series.append({
                "step": transition["step"],
                "loss": transition["loss"],
                "service_deficit": float(np.mean([value.service_deficit for value in self.environment.incidents.values()])),
                "backlog": float(np.mean([value.backlog for value in self.environment.incidents.values()])),
                "operational_energy": float(np.mean([value.operational_energy for value in active_features])) if active_features else 0.0,
                "distributed_entropy": float(np.mean([value.distributed_entropy for value in active_features])) if active_features else 0.0,
                "entropy_anomaly": float(np.mean([value.entropy_anomaly for value in active_features])) if active_features else 0.0,
                "entropy_slope": float(np.mean([value.entropy_slope for value in active_features])) if active_features else 0.0,
                "belief_disagreement": float(np.mean([value.belief_disagreement for value in active_features])) if active_features else 0.0,
                "consensus_confidence": float(np.mean([value.consensus_confidence for value in active_features])) if active_features else 1.0,
                "consensus_error": float(np.mean([value.consensus_error for value in active_features])) if active_features else 0.0,
                "free_energy_diagnostic": float(np.mean([value.free_energy_diagnostic for value in active_features])) if active_features else 0.0,
                "operator_queue_length": len(self.operator.queue),
                "operator_interventions": self.operator.interventions,
                "operator_minutes": self.operator.operator_minutes,
                "operator_workload": self.operator.workload,
                "operator_fatigue": self.operator.fatigue,
                "requests": self.request_counter,
                "active_trigger_agents": sum(self.agent_trigger_active.values()),
                "material_actions_accepted": self.environment.metric_counters["material_actions_accepted"],
                "material_actions_reached_service": self.environment.metric_counters["material_actions_reached_service"],
                "visible_collapse": int(any(value.visible_collapse for value in self.environment.incidents.values())),
            })
        elapsed = time.perf_counter() - started
        conservation = self.environment.conservation_report()
        active = [value for value in self.environment.incidents.values() if value.active]
        first_collapse = next((row["step"] for row in time_series if row["visible_collapse"]), None)
        timely = (
            self.first_request_step is not None
            and self.first_request_step >= self.config.disruption_step
            and (first_collapse is None or self.first_request_step < first_collapse)
        )
        nominal_false = self.config.regime == "nominal" and self.first_request_step is not None
        pre_false = self.first_request_step is not None and self.first_request_step < self.config.disruption_step
        total_loss = float(sum(row["loss"] for row in time_series))
        complete_chains = sum(bool(row.get("complete")) for row in self.counterfactuals)
        harmful = sum(float(row.get("intervention_effect", 0.0)) < 0.0 for row in self.counterfactuals)
        metrics = {
            "primary_outcome": total_loss,
            "service_loss_auc": total_loss,
            "cumulative_weighted_unmet_need": total_loss if self.config.application == V4Application.HUMANITARIAN.value else None,
            "cumulative_critical_unserved_load": total_loss if self.config.application == V4Application.UTILITY.value else None,
            "operator_interventions": self.operator.interventions,
            "operator_minutes": self.operator.operator_minutes,
            "operator_workload_auc": float(sum(row["operator_workload"] for row in time_series)),
            "maximum_operator_workload": max((row["operator_workload"] for row in time_series), default=0.0),
            "mean_queue_wait_steps": float(np.mean(self.operator.queue_wait_steps)) if self.operator.queue_wait_steps else 0.0,
            "maximum_queue_length": max((row["operator_queue_length"] for row in time_series), default=0),
            "operator_requests": self.request_counter,
            "low_confidence_operator_decisions": self.low_confidence_operator_decisions,
            "safe_low_confidence_decisions": self.safe_low_confidence_decisions,
            "safe_low_confidence_decision_rate": (
                self.safe_low_confidence_decisions
                / max(self.low_confidence_operator_decisions, 1)
            ),
            "communication_active_agent_epoch_fraction": float(
                sum(row["active_trigger_agents"] for row in time_series)
                / max(len(time_series) * len(self.environment.agents), 1)
            ),
            "timely_activation": bool(timely),
            "missed_activation": bool(active and self.first_request_step is None),
            "pre_disruption_false_activation": bool(pre_false),
            "nominal_false_activation": bool(nominal_false),
            "first_request_step": self.first_request_step,
            "first_visible_collapse_step": first_collapse,
            "complete_causal_chains": complete_chains,
            "harmful_interventions": harmful,
            "beneficial_interventions": sum(float(row.get("intervention_effect", 0.0)) > 0.0 for row in self.counterfactuals),
            "mean_counterfactual_effect": float(np.mean([row["intervention_effect"] for row in self.counterfactuals])) if self.counterfactuals else 0.0,
            "structured_attempts": int(self.environment.metric_counters["structured_attempts"]),
            "first_pass_valid": int(self.environment.metric_counters["first_pass_valid"]),
            "valid_after_repair": int(self.environment.metric_counters["valid_after_repair"]),
            "material_actions_accepted": int(self.environment.metric_counters["material_actions_accepted"]),
            "material_actions_next_stage": int(self.environment.metric_counters["material_actions_next_stage"]),
            "material_actions_reached_service": int(self.environment.metric_counters["material_actions_reached_service"]),
            "commitment_changes": int(self.environment.metric_counters["commitment_changes"]),
            "agent_messages": int(self.environment.metric_counters["messages"]),
            "agent_message_bytes": int(self.environment.metric_counters["message_bytes"]),
            "thermodynamic_sketch_messages": int(self.environment.metric_counters["thermodynamic_sketch_messages"]),
            "thermodynamic_sketch_bytes": int(self.environment.metric_counters["thermodynamic_sketch_bytes"]),
            "tool_calls": int(self.environment.metric_counters["tool_calls"]),
            "llm_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "llm_latency_seconds": 0.0,
            "wall_clock_seconds": elapsed,
            "maximum_conservation_residual": conservation["maximum_residual"],
            "conservation_feasible": conservation["feasible"],
            "active_incidents": len(active),
        }
        manifest = {
            "run_id": self.config.run_id,
            "application": self.config.application,
            "regime": self.config.regime,
            "information_condition": self.config.information_condition,
            "method": self.config.method,
            "environment_seed": self.config.environment_seed,
            "operator_seed": self.config.operator_seed,
            "planner_seed": self.config.planner_seed,
            "rl_seed": self.config.rl_seed,
            "operator_profile": self.config.operator_profile,
            "operator_budget": self.config.operator_budget,
            "horizon": self.config.horizon,
            "stage": self.config.stage,
            "completion_status": "complete" if conservation["feasible"] else "failed",
            "failure_reason": None if conservation["feasible"] else "conservation_or_feasibility",
            "event_count": len(self.environment.ledger.events),
            "event_ledger_digest": self.environment.ledger.digest(),
            "simulated_operator": True,
            "planner": "deterministic_independent_v4",
            "model_identifier": None,
            "model_revision": None,
        }
        return V4EpisodeResult(
            run_id=self.config.run_id,
            application=self.config.application,
            regime=self.config.regime,
            information_condition=self.config.information_condition,
            method=self.config.method,
            environment_seed=self.config.environment_seed,
            operator_seed=self.config.operator_seed,
            rl_seed=self.config.rl_seed,
            status=manifest["completion_status"],
            metrics=metrics,
            time_series=time_series,
            candidate_interventions=self.candidate_interventions,
            counterfactuals=self.counterfactuals,
            manifest_fields=manifest,
            ledger=self.environment.ledger,
        )
