"""Episode orchestration for the ThermoHITL v3 study."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .environment import DEMAND_ROLES, SOURCE_ROLES, derived_rng_seed
from .events import sha256_file
from .human_environment import HumanOversightEnvironment, HumanScenarioConfig
from .human_operator import (
    AssistanceKind,
    AttentionAllocator,
    AutonomyLevel,
    DistributedThermodynamicMonitor,
    EnergyWeights,
    EscalationConfig,
    HumanMethod,
    IndependentEscalationController,
    OPERATOR_PROFILES,
    OperatorIntervention,
    OperatorViewCondition,
    SimulatedOperator,
    ThermodynamicCalibration,
    build_operator_view,
)
from .planners import MockPlanner
from .runner import EpisodeRunner
from .types import CoordinationOption, Method


THERMODYNAMIC_METHODS = {
    HumanMethod.ENTROPY_ONLY_TRIGGER,
    HumanMethod.ENERGY_ONLY_TRIGGER,
    HumanMethod.FREE_ENERGY_TRIGGER,
    HumanMethod.DISAGREEMENT_TRIGGER,
    HumanMethod.THERMOHITL_RULE,
    HumanMethod.THERMOHITL_RL,
    HumanMethod.BOUNDED_HUMAN_ORACLE,
    HumanMethod.FULL_INFORMATION_ORACLE,
}


HUMAN_METHODS = {
    HumanMethod.ALWAYS_ON_HUMAN_REVIEW,
    HumanMethod.PERIODIC_HUMAN_REVIEW,
    HumanMethod.RANDOM_BUDGET_MATCHED_HUMAN,
    HumanMethod.LOCAL_KPI_TRIGGER,
    HumanMethod.ENTROPY_ONLY_TRIGGER,
    HumanMethod.ENERGY_ONLY_TRIGGER,
    HumanMethod.FREE_ENERGY_TRIGGER,
    HumanMethod.DISAGREEMENT_TRIGGER,
    HumanMethod.THERMOHITL_RULE,
    HumanMethod.LEARNED_NO_THERMODYNAMICS,
    HumanMethod.THERMOHITL_RL,
    HumanMethod.BOUNDED_HUMAN_ORACLE,
    HumanMethod.FULL_INFORMATION_ORACLE,
}


@dataclass
class HumanEpisodeResult:
    run_id: str
    application: str
    method: str
    scenario: str
    environment_seed: int
    llm_seed: int
    rl_seed: Optional[int]
    operator_seed: int
    operator_profile: str
    operator_view: str
    metrics: Dict[str, Any]
    time_series: List[Dict[str, Any]]
    actionability: Dict[str, Any]
    operator_metrics: Dict[str, Any]
    planner_metrics: Dict[str, Any]
    counterfactuals: List[Dict[str, Any]]
    completion_status: str
    wall_clock_seconds: float
    trajectory: List[Dict[str, Any]]


def _underlying_method(method: HumanMethod) -> Method:
    if method == HumanMethod.NO_COMMUNICATION:
        return Method.NO_COMM
    if method == HumanMethod.FIXED_COMMUNICATION_NO_HUMAN:
        return Method.FIXED_ALWAYS_ON
    if method == HumanMethod.CENTRALIZED_FULL_INFORMATION:
        return Method.CENTRALIZED
    # All supervisory comparisons use the same ordinary periodic independent
    # agent communication policy. Human attention is the manipulated layer.
    return Method.FIXED_COMM


def _view_condition(method: HumanMethod) -> OperatorViewCondition:
    if method in (
        HumanMethod.ALWAYS_ON_HUMAN_REVIEW,
        HumanMethod.PERIODIC_HUMAN_REVIEW,
        HumanMethod.RANDOM_BUDGET_MATCHED_HUMAN,
        HumanMethod.LOCAL_KPI_TRIGGER,
        HumanMethod.LEARNED_NO_THERMODYNAMICS,
    ):
        return OperatorViewCondition.LOCAL_KPI
    if method == HumanMethod.ENTROPY_ONLY_TRIGGER:
        return OperatorViewCondition.ENTROPY_ONLY
    if method == HumanMethod.ENERGY_ONLY_TRIGGER:
        return OperatorViewCondition.ENERGY_ONLY
    if method in (HumanMethod.FREE_ENERGY_TRIGGER,):
        return OperatorViewCondition.THERMODYNAMIC
    if method in (
        HumanMethod.BOUNDED_HUMAN_ORACLE,
        HumanMethod.FULL_INFORMATION_ORACLE,
    ):
        return OperatorViewCondition.EVALUATOR_ORACLE
    return OperatorViewCondition.THERMODYNAMIC_DISAGREEMENT


def _allocator_policy(method: HumanMethod) -> str:
    if method in (
        HumanMethod.ALWAYS_ON_HUMAN_REVIEW,
        HumanMethod.PERIODIC_HUMAN_REVIEW,
    ):
        return "fcfs"
    if method == HumanMethod.RANDOM_BUDGET_MATCHED_HUMAN:
        return "random"
    if method in (HumanMethod.LOCAL_KPI_TRIGGER, HumanMethod.LEARNED_NO_THERMODYNAMICS):
        return "local_kpi" if method == HumanMethod.LOCAL_KPI_TRIGGER else "learned_non_entropic"
    if method == HumanMethod.ENTROPY_ONLY_TRIGGER:
        return "highest_entropy"
    if method in (HumanMethod.ENERGY_ONLY_TRIGGER, HumanMethod.FREE_ENERGY_TRIGGER):
        return "highest_energy"
    if method == HumanMethod.DISAGREEMENT_TRIGGER:
        return "highest_disagreement"
    if method in (HumanMethod.BOUNDED_HUMAN_ORACLE, HumanMethod.FULL_INFORMATION_ORACLE):
        return "oracle"
    return "thermodynamic_expected_benefit"


class HumanOperatorEpisodeRunner(EpisodeRunner):
    """Independent-agent episode with bounded human-on-the-loop triage."""

    def __init__(
        self,
        config: HumanScenarioConfig,
        human_method: str,
        planner: Optional[Any] = None,
        thermodynamic_calibration: Optional[ThermodynamicCalibration] = None,
        energy_weights: Optional[EnergyWeights] = None,
        escalation_config: Optional[EscalationConfig] = None,
        learned_score: Optional[Callable[[str, Mapping[str, float]], float]] = None,
        rl_seed: Optional[int] = None,
        llm_seed: int = 0,
        operator_view: Optional[str] = None,
        operator_profile: Optional[str] = None,
        gossip_rounds: int = 3,
        enable_counterfactual_probes: bool = True,
        dense_counterfactual_probes: bool = False,
        dense_probe_interval: int = 2,
    ) -> None:
        self.human_method = HumanMethod(human_method)
        environment = HumanOversightEnvironment(config)
        super().__init__(
            config=config,
            method=_underlying_method(self.human_method).value,
            planner=planner or MockPlanner(),
            gossip_rounds=gossip_rounds,
            environment=environment,
        )
        self.env: HumanOversightEnvironment
        self.thermodynamic_monitor = DistributedThermodynamicMonitor(
            self.env.agent_ids,
            calibration=thermodynamic_calibration,
            energy_weights=energy_weights,
            gossip_rounds=gossip_rounds,
        )
        self.escalation = IndependentEscalationController(
            self.env.agent_ids, escalation_config
        )
        self.learned_score = learned_score
        self.rl_seed = rl_seed
        self.llm_seed = int(llm_seed)
        self.operator_seed = int(config.operator_seed)
        self.view_condition = OperatorViewCondition(
            operator_view or _view_condition(self.human_method).value
        )
        profile_name = operator_profile or config.operator_profile
        if self.human_method in (
            HumanMethod.BOUNDED_HUMAN_ORACLE,
            HumanMethod.FULL_INFORMATION_ORACLE,
        ):
            profile_name = "oracle"
        if profile_name not in OPERATOR_PROFILES:
            raise ValueError("unknown operator profile: %s" % profile_name)
        self.operator_profile_name = profile_name
        self.operator: Optional[SimulatedOperator] = None
        if self.human_method in HUMAN_METHODS:
            self.operator = SimulatedOperator(
                OPERATOR_PROFILES[profile_name],
                AttentionAllocator(
                    _allocator_policy(self.human_method),
                    seed=derived_rng_seed(config.operator_seed, "communication"),
                ),
                seed=config.operator_seed,
            )
        self.enable_counterfactual_probes = bool(enable_counterfactual_probes)
        self.dense_counterfactual_probes = bool(dense_counterfactual_probes)
        self.dense_probe_interval = max(1, int(dense_probe_interval))
        self.counterfactual_rows: List[Dict[str, Any]] = []
        self.human_request_steps: List[int] = []
        self.human_intervention_steps: List[int] = []
        self.human_trigger_activations = 0
        self.human_trigger_deactivations = 0
        self.first_pass_valid = 0
        self.repaired_valid = 0
        self.structured_attempts = 0
        self._accepted_directive_agents: set[str] = set()
        self._handled_directive_interventions: set[str] = set()
        self._processed_need_messages: set[str] = set()
        self._latest_learned_actions: Dict[str, int] = {}

    def _option(self, agent_id: str) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
        if agent_id in self._accepted_directive_agents and agent_id in self.env.pending_human_directives:
            agent = self.env.agents[agent_id]
            observation = agent.observation_vector(0, include_entropy=False)
            mask = self._local_option_mask(agent_id)
            mask[int(CoordinationOption.EMERGENCY)] = True
            return int(CoordinationOption.EMERGENCY), 0.0, 0.0, observation, mask
        agent = self.env.agents[agent_id]
        if agent.identity.role in SOURCE_ROLES:
            actionable_need = next((
                message for message in reversed(agent.inbox)
                if message.kind == "need"
                and message.message_id not in self._processed_need_messages
                and (agent_id, message.sender) in self.env.physical_edges
            ), None)
            if actionable_need is not None:
                # One explicit delivered need is consumed by one independent
                # source decision. The simulator does not synthesize a target.
                self._processed_need_messages.add(actionable_need.message_id)
                observation = agent.observation_vector(0, include_entropy=False)
                mask = self._local_option_mask(agent_id)
                mask[int(CoordinationOption.CONTINUE)] = True
                return int(CoordinationOption.CONTINUE), 0.0, 0.0, observation, mask
        return super()._option(agent_id)

    def _v3_fixed_need_protocol(self) -> None:
        """Agent-local typed need reports used by the strong communication control."""

        if self.human_method in (
            HumanMethod.NO_COMMUNICATION,
            HumanMethod.CENTRALIZED_FULL_INFORMATION,
        ):
            return
        interval = 2 if self.human_method == HumanMethod.FIXED_COMMUNICATION_NO_HUMAN else 4
        if self.env.step_index % interval != 0:
            return
        active_edges = self.env.active_communication_edges()
        for demand_id in self.env.agent_ids:
            demand_agent = self.env.agents[demand_id]
            if demand_agent.identity.role not in DEMAND_ROLES:
                continue
            observation = demand_agent.vault.observation(demand_id)
            if observation.backlog <= 0.01:
                continue
            reachable_sources = sorted(
                source for source, target in self.env.initial_physical_edges
                if target == demand_id
                and tuple(sorted((source, demand_id))) in active_edges
            )
            for source in reachable_sources[:2]:
                quantity = max(0.01, min(
                    50.0,
                    float(observation.backlog) + float(observation.demand),
                ))
                self.env.execute_tool(demand_id, "report_local_need", {
                    "target": source,
                    "quantity": quantity,
                    "urgency": "critical" if observation.service_shortfall >= 0.6 else "urgent",
                })

    def _human_gossip_edges(self) -> Sequence[set[Tuple[str, str]]]:
        if self.human_method not in THERMODYNAMIC_METHODS:
            return []
        return self._gossip_round_edges()

    def _learned_agent_score(self, agent_id: str, state: Any) -> Optional[float]:
        if self.human_method not in (
            HumanMethod.LEARNED_NO_THERMODYNAMICS,
            HumanMethod.THERMOHITL_RL,
        ):
            return None
        if self.learned_score is None:
            raise ValueError("learned human method requires a trained scoring policy")
        features = {
            "local_kpi_risk": state.local_kpi_risk,
            "local_disruption_risk": state.local_disruption_risk,
            "actionability_evidence": state.actionability_evidence,
            "consensus_confidence": state.consensus_confidence,
            "local_energy_residual": state.local_energy_residual if self.human_method == HumanMethod.THERMOHITL_RL else 0.0,
            "energy_residual": state.energy_residual if self.human_method == HumanMethod.THERMOHITL_RL else 0.0,
            "entropy_residual": state.entropy_residual if self.human_method == HumanMethod.THERMOHITL_RL else 0.0,
            "entropy_slope": state.entropy_slope if self.human_method == HumanMethod.THERMOHITL_RL else 0.0,
            "disagreement": state.disagreement if self.human_method == HumanMethod.THERMOHITL_RL else 0.0,
        }
        value = self.learned_score(agent_id, features)
        if isinstance(value, tuple):
            score, action = value
            self._latest_learned_actions[agent_id] = int(action)
            # Actions 0/1 retain machine autonomy and must not be converted to
            # a human request by the outer hysteresis layer.
            return float(score) if int(action) >= 2 else 0.0
        return float(value)

    def _oracle_payload(self, state: Any, evaluator: Any) -> Dict[str, Any]:
        return {
            "true_intervention_benefit": max(0.0, state.local_kpi_risk * 4.0),
            "exact_global_entropy": evaluator.exact_entropy,
            "exact_global_energy": evaluator.exact_energy,
            "true_disruption_label": self.config.disruption,
            "raw_private_state": self.env.full_state_for_evaluator(),
        }

    def _enqueue_requests(self, thermodynamic: Any) -> None:
        if self.operator is None:
            return
        workload = self.operator.workload_snapshot()
        public_network = self.env.public_operator_network(
            thermodynamic.local
            if self.view_condition in (
                OperatorViewCondition.THERMODYNAMIC,
                OperatorViewCondition.THERMODYNAMIC_DISAGREEMENT,
                OperatorViewCondition.ENTROPY_ONLY,
                OperatorViewCondition.ENERGY_ONLY,
            ) else None
        )
        for agent_id in self.env.agent_ids:
            state = thermodynamic.local[agent_id]
            learned = self._learned_agent_score(agent_id, state)
            request, score, activated, deactivated = self.escalation.should_request(
                agent_id,
                self.human_method,
                state,
                self.operator.workload,
                self.env.agents[agent_id].rng,
                learned_score=learned,
            )
            self.human_trigger_activations += int(activated)
            self.human_trigger_deactivations += int(deactivated)
            if activated or deactivated:
                self.env.ledger.append(
                    self.env.step_index,
                    "autonomy_transition",
                    agent_id,
                    {
                        "trigger_transition": "activated" if activated else "deactivated",
                        "score": score,
                        "method": self.human_method.value,
                    },
                    private_to=agent_id,
                )
            if not request:
                continue
            learned_kind: Optional[AssistanceKind] = None
            if agent_id in self._latest_learned_actions:
                learned_kind = {
                    2: AssistanceKind.INFORMATION,
                    3: AssistanceKind.RECOMMENDATION,
                    4: AssistanceKind.APPROVAL,
                    5: AssistanceKind.CONFLICT_RESOLUTION,
                    6: AssistanceKind.EMERGENCY_OVERRIDE,
                }.get(self._latest_learned_actions[agent_id])
            assistance = self.env.create_assistance_request(
                agent_id, state, score, assistance_kind=learned_kind
            )
            oracle_payload = (
                self._oracle_payload(state, thermodynamic.evaluator)
                if self.view_condition == OperatorViewCondition.EVALUATOR_ORACLE else None
            )
            view = build_operator_view(
                assistance,
                state,
                self.view_condition,
                workload,
                public_network,
                oracle_payload=oracle_payload,
            )
            self.env.ledger.append(
                self.env.step_index,
                "operator_view",
                "operator_dashboard",
                view.as_dict(),
                private_to="simulated_human_operator",
            )
            if self.operator.enqueue(assistance, view):
                self.human_request_steps.append(self.env.step_index)
                self.env.ledger.append(
                    self.env.step_index,
                    "operator_queue",
                    "simulated_human_operator",
                    {
                        "action": "enqueued",
                        "incident_id": assistance.incident_id,
                        "queue_length": len(self.operator.queue),
                        "view_sha256": view.sha256,
                    },
                )

    def _process_operator(self) -> set[str]:
        if self.operator is None:
            return set()
        before_active = {row.request.incident_id for row in self.operator.active}
        interventions = self.operator.step(self.env.step_index)
        after_active = {row.request.incident_id for row in self.operator.active}
        for incident_id in sorted(after_active - before_active):
            decision = next(row for row in self.operator.active if row.request.incident_id == incident_id)
            self.env.ledger.append(
                self.env.step_index,
                "attention_allocation",
                "simulated_human_operator",
                {
                    "incident_id": incident_id,
                    "allocated_step": self.env.step_index,
                    "completion_step": decision.completion_step,
                    "policy": self.operator.allocator.policy,
                    "queue_length_after": len(self.operator.queue),
                    "view_sha256": decision.view.sha256,
                },
            )
        replans: set[str] = set()
        for intervention in interventions:
            if self.enable_counterfactual_probes:
                self.counterfactual_rows.append(
                    self._counterfactual_probe(intervention)
                )
            result = self.env.execute_human_intervention(intervention)
            self.human_intervention_steps.append(self.env.step_index)
            if result.ok:
                directive_target = result.data.get("directive_target")
                if directive_target in self.env.agents:
                    accepted, directive = self.env.directive_response(directive_target)
                    if accepted and directive is not None:
                        self._accepted_directive_agents.add(directive_target)
                        replans.add(directive_target)
        return replans

    @staticmethod
    def _execute_probe_directive(env: HumanOversightEnvironment, intervention: OperatorIntervention) -> Dict[str, Any]:
        result = env.execute_human_intervention(intervention)
        record: Dict[str, Any] = {
            "operator_result": result.as_dict(),
            "agent_accepted": False,
            "material_action_accepted": False,
            "shipment_id": None,
        }
        if not result.ok:
            return record
        agent_id = result.data.get("directive_target")
        if agent_id not in env.agents:
            return record
        accepted, directive = env.directive_response(agent_id)
        record["agent_accepted"] = accepted
        if not accepted or directive is None:
            return record
        if directive.get("tool") not in (
            "authorize_emergency_route",
            "temporary_emergency_override",
        ):
            return record
        target = str(directive["target"])
        state = env.states[agent_id]
        quantity = min(
            float(directive.get("maximum_quantity", 12.0)),
            state.inventory,
            state.capacity,
        )
        if quantity <= 0.01:
            return record
        tool = "transfer_resource" if env.application.value == "humanitarian" else "schedule_shipment"
        arrival = env.step_index + 1 + env.route_lead_time_penalty
        action = env.execute_tool(agent_id, tool, {
            "target": target,
            "quantity": float(quantity),
            "arrival_step": int(arrival),
        })
        record["material_action"] = action.as_dict()
        record["material_action_accepted"] = action.ok
        record["shipment_id"] = action.data.get("shipment_id") if action.ok else None
        return record

    def _counterfactual_probe(self, intervention: OperatorIntervention) -> Dict[str, Any]:
        snapshot, rng_digests = self.env.counterfactual_snapshot()
        treated = snapshot
        control = deepcopy_environment(snapshot)
        treated_response = self._execute_probe_directive(treated, intervention)
        horizon = min(6, max(1, self.config.horizon - self.env.step_index - 1))
        treated_losses: List[float] = []
        control_losses: List[float] = []
        for _ in range(horizon):
            for branch, losses in ((treated, treated_losses), (control, control_losses)):
                branch.advance()
                branch.transition()
                branch.deliver_observations()
                metrics = branch.public_metrics()
                losses.append(
                    float(metrics["service_loss"])
                    if self.config.application == "commercial"
                    else float(metrics["weighted_backlog"])
                )
        effect = float(sum(control_losses) - sum(treated_losses))
        shipment_id = treated_response.get("shipment_id")
        record = treated.material_action_records.get(shipment_id, {}) if shipment_id else {}
        row = {
            "incident_id": intervention.incident_id,
            "intervention_id": intervention.intervention_id,
            "step": self.env.step_index,
            "rng_digests": rng_digests,
            "probe_horizon": horizon,
            "loss_without_intervention": float(sum(control_losses)),
            "loss_with_intervention": float(sum(treated_losses)),
            "intervention_effect": effect,
            "classification": "beneficial" if effect > 1e-9 else "harmful" if effect < -1e-9 else "no_causal_benefit",
            "agent_accepted": treated_response.get("agent_accepted", False),
            "material_action_accepted": treated_response.get("material_action_accepted", False),
            "material_reached_next_stage": bool(record.get("reached_next_stage")),
            "material_reached_demand": bool(record.get("reached_final_demand")),
            "primary_outcome_changed": abs(effect) > 1e-9,
            "common_randomness_verified": all(
                self.env._rng_state_digest(getattr(control, name))
                == self.env._rng_state_digest(getattr(treated, name))
                for name in ("rng", "exogenous_rng", "observation_rng", "communication_rng")
            ),
        }
        self.env.ledger.append(
            self.env.step_index,
            "counterfactual_branch",
            "evaluator",
            row,
        )
        return row

    def _dense_counterfactual_probe(self, agent_id: str, state: Any) -> Dict[str, Any]:
        """Evaluate one authorized candidate without exposing its outcome.

        This evaluator-only development probe starts two branches from the
        same quantitative, agent, and RNG state. It never feeds its label or
        selected route back to an execution-time policy.
        """

        snapshot, rng_digests = self.env.counterfactual_snapshot()
        treated = snapshot
        control = deepcopy_environment(snapshot)
        request = treated.create_assistance_request(agent_id, state, score=0.0)
        intervention = OperatorIntervention(
            intervention_id="DENSE-%s-%04d" % (agent_id, self.env.step_index),
            incident_id=request.incident_id,
            requesting_agent=agent_id,
            step=self.env.step_index,
            action="evaluator_development_probe",
            bounded_tool=request.suggested_intervention,
            arguments=dict(request.intervention_arguments),
            mandatory=request.suggested_intervention == "temporary_emergency_override",
            view_sha256="evaluator-only-not-an-operator-view",
            expected_benefit=request.expected_benefit,
            operator_minutes=request.estimated_operator_minutes,
        )
        candidate_edge = (
            str(request.intervention_arguments.get("source", "")),
            str(request.intervention_arguments.get("target", "")),
        )
        redundant_public_route = bool(
            request.suggested_intervention in (
                "authorize_emergency_route", "temporary_emergency_override",
            )
            and candidate_edge in treated.physical_edges
        )
        if redundant_public_route:
            response = {
                "operator_result": {
                    "ok": False,
                    "code": "operator_rejected_redundant_public_route",
                },
                "agent_accepted": False,
                "material_action_accepted": False,
                "shipment_id": None,
            }
        else:
            response = self._execute_probe_directive(treated, intervention)
        horizon = min(6, max(1, self.config.horizon - self.env.step_index - 1))
        treated_losses: List[float] = []
        control_losses: List[float] = []
        for _ in range(horizon):
            for branch, losses in ((treated, treated_losses), (control, control_losses)):
                branch.advance()
                branch.transition()
                branch.deliver_observations()
                metrics = branch.public_metrics()
                losses.append(
                    float(metrics["service_loss"])
                    if self.config.application == "commercial"
                    else float(metrics["weighted_backlog"])
                )
        effect = float(sum(control_losses) - sum(treated_losses))
        return {
            "probe_type": "dense_development_candidate",
            "information_boundary": "features_are_actor_local; effect_is_evaluator_only",
            "application": self.config.application,
            "scenario": self.config.disruption,
            "communication": self.config.communication,
            "environment_seed": self.config.seed,
            "operator_seed": self.operator_seed,
            "step": self.env.step_index,
            "agent_id": agent_id,
            "role": state.role,
            "candidate_tool": request.suggested_intervention,
            "candidate_publicly_redundant": redundant_public_route,
            "candidate_arguments_sha256": hashlib.sha256(
                json.dumps(request.intervention_arguments, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "features": state.as_dict(),
            "loss_without_intervention": float(sum(control_losses)),
            "loss_with_intervention": float(sum(treated_losses)),
            "intervention_effect": effect,
            "beneficial_intervention": bool(effect > 1e-9),
            "agent_accepted": response.get("agent_accepted", False),
            "material_action_accepted": response.get("material_action_accepted", False),
            "common_randomness_verified": all(
                self.env._rng_state_digest(getattr(control, name))
                == self.env._rng_state_digest(getattr(treated, name))
                for name in ("rng", "exogenous_rng", "observation_rng", "communication_rng")
            ),
            "initial_rng_digests": rng_digests,
        }

    def _finalize_successful_directives(self, material_before: set[str]) -> None:
        new_shipments = set(self.env.material_action_records) - material_before
        completed_intervention_ids = {
            str(self.env.material_action_records[shipment_id].get("human_intervention_id"))
            for shipment_id in new_shipments
            if self.env.material_action_records[shipment_id].get("human_intervention_id")
        }
        for agent_id in list(self._accepted_directive_agents):
            directive = self.env.pending_human_directives.get(agent_id)
            if directive is None:
                self._accepted_directive_agents.discard(agent_id)
                continue
            intervention_id = str(directive.get("intervention_id"))
            if intervention_id not in completed_intervention_ids:
                continue
            self.env.pending_human_directives.pop(agent_id, None)
            self._accepted_directive_agents.discard(agent_id)
            self.env.set_autonomy_level(
                agent_id,
                int(AutonomyLevel.TARGETED_COORDINATION),
                "bounded_directive_executed",
                intervention_id,
            )

    def run(self, run_id: Optional[str] = None) -> HumanEpisodeResult:
        started = time.perf_counter()
        run_id = run_id or "%s-%s-s%05d-o%05d" % (
            self.config.application,
            self.human_method.value,
            self.config.seed,
            self.operator_seed,
        )
        time_series: List[Dict[str, Any]] = []
        previous_metrics: Optional[Dict[str, Any]] = None
        status = "complete"
        requested_replans: set[str] = set()
        try:
            event_cursor = len(self.env.ledger.events)
            for step in range(self.config.horizon):
                self.env.transition()
                new_events = self.env.ledger.events[event_cursor:]
                event_cursor = len(self.env.ledger.events)
                self.env.deliver_observations()
                self._prepare_step_modes()
                thermo = self.thermodynamic_monitor.update(
                    self.env,
                    self._human_gossip_edges(),
                )
                self._v3_fixed_need_protocol()
                if (
                    self.dense_counterfactual_probes
                    and self.env.step_index % self.dense_probe_interval == 0
                ):
                    for agent_id in self.env.agent_ids:
                        state = thermo.local[agent_id]
                        if (
                            state.role not in DEMAND_ROLES
                            and state.actionability_evidence < 0.95
                        ):
                            continue
                        self.counterfactual_rows.append(
                            self._dense_counterfactual_probe(
                                agent_id, state
                            )
                        )
                self._enqueue_requests(thermo)
                operator_replans = self._process_operator()
                metrics = self.env.public_metrics()
                operator_snapshot = (
                    self.operator.workload_snapshot() if self.operator is not None
                    else {
                        "workload": 0.0, "fatigue": 0.0,
                        "active_interventions": 0, "queue_length": 0,
                        "available_attention_slots": 0, "operator_minutes": 0.0,
                    }
                )
                local_values = list(thermo.local.values())
                row = {
                    **metrics,
                    "disruption_active": self.env._disruption_applied,
                    "exact_operational_energy": thermo.evaluator.exact_energy,
                    "exact_operational_entropy": thermo.evaluator.exact_entropy,
                    "exact_flow_entropy": thermo.evaluator.exact_flow_entropy,
                    "exact_belief_entropy": thermo.evaluator.exact_belief_entropy,
                    "exact_disagreement": thermo.evaluator.exact_disagreement,
                    "exact_free_energy_diagnostic": thermo.evaluator.exact_free_energy,
                    "distributed_energy_mean": float(np.mean([state.distributed_energy for state in local_values])),
                    "distributed_entropy_mean": float(np.mean([state.distributed_entropy for state in local_values])),
                    "entropy_anomaly_mean": float(np.mean([state.entropy_residual for state in local_values])),
                    "entropy_slope_mean": float(np.mean([state.entropy_slope for state in local_values])),
                    "disagreement_mean": float(np.mean([state.disagreement for state in local_values])),
                    "consensus_confidence_mean": float(np.mean([state.consensus_confidence for state in local_values])),
                    "entropy_estimation_rmse": thermo.evaluator.entropy_rmse,
                    "energy_estimation_rmse": thermo.evaluator.energy_rmse,
                    "thermodynamic_sketch_messages": self.thermodynamic_monitor.cumulative_sketch_messages,
                    "thermodynamic_sketch_bytes": self.thermodynamic_monitor.cumulative_sketch_bytes,
                    "human_requests": len(self.human_request_steps),
                    "human_interventions": len(self.human_intervention_steps),
                    "operator_workload": operator_snapshot["workload"],
                    "operator_fatigue": operator_snapshot["fatigue"],
                    "operator_queue_length": operator_snapshot["queue_length"],
                    "operator_active": operator_snapshot["active_interventions"],
                    "operator_minutes": operator_snapshot["operator_minutes"],
                    "maximum_autonomy_level": max(self.env.autonomy_levels.values()),
                }
                time_series.append(row)
                self.env.ledger.append(
                    self.env.step_index,
                    "metric",
                    "v3_evaluator",
                    row,
                )
                self._assign_rewards(previous_metrics, metrics)

                scheduled_agents = self._scheduled_agent_ids(step)
                triggered_agents = set(requested_replans) | operator_replans
                for event in new_events:
                    if event.kind == "message_delivery" and event.payload.get("kind") in (
                        "offer", "counteroffer", "offer_accepted", "offer_rejected",
                        "coalition_proposal", "coalition_joined", "commitment_breach",
                    ):
                        triggered_agents.add(str(event.payload["recipient"]))
                selected = sorted(set(scheduled_agents) | triggered_agents)
                material_before = set(self.env.material_action_records)
                structured_before = self.total_structured_outputs
                first_pass_before = self.first_pass_structured_valid
                final_valid_before = self.valid_structured_outputs
                if selected:
                    self._decision_epoch(selected)
                new_attempts = self.total_structured_outputs - structured_before
                self.structured_attempts += new_attempts
                self.first_pass_valid += (
                    self.first_pass_structured_valid - first_pass_before
                )
                self.repaired_valid += (
                    self.valid_structured_outputs - final_valid_before
                )
                self._finalize_successful_directives(material_before)
                requested_replans = {
                    agent_id for agent_id, agent in self.env.agents.items()
                    if not agent.last_tool_ok
                }
                previous_metrics = metrics
                self.env.advance()
        except Exception:
            status = "failed"
            raise
        finally:
            if self.trajectory:
                last_by_agent: Dict[str, int] = {}
                for index, row in enumerate(self.trajectory):
                    last_by_agent[row["agent_id"]] = index
                for index in last_by_agent.values():
                    self.trajectory[index]["done"] = True

        wall_clock = time.perf_counter() - started
        final = time_series[-1]
        disruption_step = max(2, self.config.horizon // 3)
        post_requests = [step for step in self.human_request_steps if step >= disruption_step]
        pre_requests = [step for step in self.human_request_steps if step < disruption_step]
        collapse_steps = [
            row["step"] for index, row in enumerate(time_series)
            if row["step"] >= disruption_step + self.config.collapse_persistence - 1
            and index >= self.config.collapse_persistence - 1
            and all(
                time_series[index - offset]["step"] >= disruption_step
                and time_series[index - offset]["service_loss"] >= self.config.collapse_threshold
                for offset in range(self.config.collapse_persistence)
            )
        ]
        collapse_step = min(collapse_steps) if collapse_steps else None
        timely = bool(post_requests) and (
            collapse_step is None or min(post_requests) < collapse_step
        )
        service_loss_auc = float(sum(row["service_loss"] for row in time_series))
        cumulative_unmet = float(sum(row["weighted_backlog"] for row in time_series))
        primary = service_loss_auc if self.config.application == "commercial" else cumulative_unmet
        v3 = self.env.v3_metrics()
        operator_metrics = {
            "operator_profile": self.operator_profile_name,
            "operator_view": self.view_condition.value,
            "requests": len(self.human_request_steps),
            "interventions": len(self.human_intervention_steps),
            "operator_minutes": self.operator.operator_minutes if self.operator else 0.0,
            "operator_workload_auc": float(sum(row["operator_workload"] for row in time_series)),
            "maximum_workload": self.operator.maximum_workload if self.operator else 0.0,
            "maximum_queue": self.operator.maximum_queue if self.operator else 0,
            "mean_queue_wait_steps": float(np.mean(self.operator.queue_wait_steps)) if self.operator and self.operator.queue_wait_steps else 0.0,
            "rejected_requests": self.operator.rejected_count if self.operator else 0,
            "trigger_activations": self.human_trigger_activations,
            "trigger_deactivations": self.human_trigger_deactivations,
            "first_post_disruption_request_step": min(post_requests) if post_requests else None,
            "activation_delay": min(post_requests) - disruption_step if post_requests else None,
            "pre_disruption_false_activation": bool(pre_requests),
            "nominal_false_activation": bool(self.config.disruption == "nominal" and self.human_request_steps),
            "timely_activation": timely,
            "missed_activation": bool(self.config.disruption != "nominal" and not post_requests),
            "collapse_step": collapse_step,
        }
        actionability = {
            "structured_attempts": self.structured_attempts,
            "first_pass_valid": self.first_pass_valid,
            "first_pass_valid_rate": self.first_pass_valid / max(self.structured_attempts, 1),
            "valid_after_one_repair": max(self.first_pass_valid, self.repaired_valid),
            "valid_after_one_repair_rate": max(self.first_pass_valid, self.repaired_valid) / max(self.structured_attempts, 1),
            **{key: value for key, value in v3.items() if key.startswith("material_")},
        }
        metrics = {
            "primary_outcome": primary,
            "service_loss_auc": service_loss_auc,
            "cumulative_unmet_weighted_need": cumulative_unmet,
            "fulfillment_rate": final["fulfillment_rate"],
            "final_backlog": final["backlog"],
            "fairness": final["fairness"],
            "total_cost": final["total_cost"],
            "conservation_error": final["conservation_error"],
            "agent_messages": final["messages"],
            "agent_message_bytes": final["message_bytes"],
            "thermodynamic_sketch_messages": self.thermodynamic_monitor.cumulative_sketch_messages,
            "thermodynamic_sketch_bytes": self.thermodynamic_monitor.cumulative_sketch_bytes,
            "operator_messages": self.env.operator_message_count,
            "operator_message_bytes": self.env.operator_message_bytes,
            "total_communication_messages": final["messages"] + self.thermodynamic_monitor.cumulative_sketch_messages + self.env.operator_message_count,
            "total_communication_bytes": final["message_bytes"] + self.thermodynamic_monitor.cumulative_sketch_bytes + self.env.operator_message_bytes,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "llm_calls": self.llm_calls,
            "llm_latency_seconds": self.llm_latency,
            "counterfactual_interventions": len(self.counterfactual_rows),
            "counterfactual_beneficial": sum(row["intervention_effect"] > 0 for row in self.counterfactual_rows),
            "mean_counterfactual_effect": float(np.mean([row["intervention_effect"] for row in self.counterfactual_rows])) if self.counterfactual_rows else 0.0,
            **v3,
        }
        planner_metrics = {
            "planner_revision": getattr(self.planner, "revision", "unknown"),
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "llm_latency_seconds": self.llm_latency,
        }
        scenario = "%s-%s-%s-p%.1f-o%.1f" % (
            self.config.topology,
            self.config.communication,
            self.config.disruption,
            self.config.private_information,
            self.config.objective_misalignment,
        )
        return HumanEpisodeResult(
            run_id=run_id,
            application=self.config.application,
            method=self.human_method.value,
            scenario=scenario,
            environment_seed=self.config.seed,
            llm_seed=self.llm_seed,
            rl_seed=self.rl_seed,
            operator_seed=self.operator_seed,
            operator_profile=self.operator_profile_name,
            operator_view=self.view_condition.value,
            metrics=metrics,
            time_series=time_series,
            actionability=actionability,
            operator_metrics=operator_metrics,
            planner_metrics=planner_metrics,
            counterfactuals=self.counterfactual_rows,
            completion_status=status,
            wall_clock_seconds=wall_clock,
            trajectory=self.trajectory,
        )


def deepcopy_environment(environment: HumanOversightEnvironment) -> HumanOversightEnvironment:
    """Named seam used by tests to verify branch isolation."""

    from copy import deepcopy

    return deepcopy(environment)


def write_human_episode(
    result: HumanEpisodeResult,
    ledger: Any,
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "episode.json"
    events_path = output_dir / "events.jsonl.gz"
    counterfactual_path = output_dir / "counterfactuals.json"
    episode_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    counterfactual_path.write_text(
        json.dumps(result.counterfactuals, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ledger.write_jsonl(events_path)
    return {
        "episode.json": sha256_file(episode_path),
        "events.jsonl.gz": sha256_file(events_path),
        "counterfactuals.json": sha256_file(counterfactual_path),
    }
