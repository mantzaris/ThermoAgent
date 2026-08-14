"""V3-only actionable logistics environment and typed human authority.

This module subclasses the frozen simulator without changing the meaning of
v1/v2 methods or artifacts. Human interventions change a documented feasible
action, information, priority, or authority boundary. They do not directly
invent an ordinary autonomous-agent action.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .environment import (
    DEMAND_ROLES,
    SOURCE_ROLES,
    LogisticsEnvironment,
    ScenarioConfig,
)
from .human_operator import (
    AssistanceKind,
    AssistanceRequest,
    AutonomyLevel,
    LocalThermodynamicState,
    OperatorIntervention,
    autonomy_level_for_request,
    request_assistance_kind,
)
from .tools import FieldRule, ToolSpec
from .types import Message, ToolResult


@dataclass
class HumanScenarioConfig(ScenarioConfig):
    """Scenario parameters introduced only for the ThermoHITL study."""

    human_remediable: bool = True
    actionability_mode: str = "authority_bottleneck"
    operator_profile: str = "high_accuracy_bounded"
    operator_seed: int = 0
    operator_slots: int = 2
    intervention_budget: int = 6
    emergency_resource_limit: float = 24.0
    maximum_authorization_duration: int = 8
    collapse_threshold: float = 0.72
    collapse_persistence: int = 3


def _human_specs() -> Dict[str, ToolSpec]:
    agent = FieldRule(str)
    duration = FieldRule(int, minimum=1, maximum=8)
    quantity = FieldRule(float, minimum=0.01, maximum=100.0)
    return {
        "authorize_emergency_route": ToolSpec(
            "authorize_emergency_route",
            {"source": agent, "target": agent, "duration": duration},
            {"human_operator"}, True,
            "Authorize one explicitly identified emergency logistics edge.",
        ),
        "approve_emergency_resource": ToolSpec(
            "approve_emergency_resource",
            {"recipient": agent, "quantity": quantity},
            {"human_operator"}, True,
            "Release bounded emergency inventory as a recorded exogenous inflow.",
        ),
        "relax_operational_constraint": ToolSpec(
            "relax_operational_constraint",
            {
                "agent_id": agent,
                "constraint": FieldRule(str, choices=("cost_ceiling", "lead_time_ceiling")),
                "new_limit": FieldRule(float, minimum=0.1, maximum=20.0),
                "duration": duration,
            },
            {"human_operator"}, True,
            "Relax one documented local operating constraint for a bounded period.",
        ),
        "authorize_information_sharing": ToolSpec(
            "authorize_information_sharing",
            {"source": agent, "target": agent, "scope": FieldRule(str, choices=("coarse_need", "coarse_capacity", "route_status")), "duration": duration},
            {"human_operator"}, True,
            "Authorize one coarse cross-organization disclosure.",
        ),
        "resolve_contract_conflict": ToolSpec(
            "resolve_contract_conflict",
            {"commitment_id": FieldRule(str), "resolution": FieldRule(str, choices=("accept", "cancel", "extend"))},
            {"human_operator"}, True,
            "Resolve one explicit multi-party contractual deadlock.",
        ),
        "adjust_priorities": ToolSpec(
            "adjust_priorities",
            {"target": agent, "priority_weight": FieldRule(float, minimum=0.5, maximum=3.0), "duration": duration},
            {"human_operator"}, True,
            "Temporarily change one demand location's priority class.",
        ),
        "temporary_emergency_override": ToolSpec(
            "temporary_emergency_override",
            {"source": agent, "target": agent, "quantity": quantity, "duration": duration},
            {"human_operator"}, True,
            "Mandate a scoped emergency route and maximum dispatch quantity.",
        ),
        "return_control": ToolSpec(
            "return_control",
            {"agent_id": agent},
            {"human_operator"}, True,
            "Return scoped authority to an autonomous organization.",
        ),
        "request_more_information": ToolSpec(
            "request_more_information",
            {"agent_id": agent, "topic": FieldRule(str)},
            {"human_operator"}, False,
            "Ask one organization for an additional bounded report.",
        ),
        "reject_request": ToolSpec(
            "reject_request",
            {"reason": FieldRule(str)},
            {"human_operator"}, False,
            "Decline a request while autonomous operation continues.",
        ),
    }


HUMAN_TOOL_SPECS = _human_specs()


class HumanOperatorToolRegistry:
    def validate(self, tool: str, arguments: Mapping[str, Any]) -> ToolResult:
        spec = HUMAN_TOOL_SPECS.get(str(tool))
        if spec is None:
            return ToolResult(False, "unknown_human_tool", "operator tool does not exist", {"tool": tool})
        return spec.validate(arguments)

    def schema(self) -> Dict[str, Any]:
        return {
            name: {
                "description": spec.description,
                "arguments": {
                    field: {
                        "type": rule.kind.__name__,
                        "minimum": rule.minimum,
                        "maximum": rule.maximum,
                        "choices": rule.choices,
                    }
                    for field, rule in spec.fields.items()
                },
            }
            for name, spec in HUMAN_TOOL_SPECS.items()
        }


class HumanOversightEnvironment(LogisticsEnvironment):
    """Actionable v3 environment with bounded human authorization state."""

    def __init__(self, config: HumanScenarioConfig) -> None:
        self.human_config = config
        self.human_registry = HumanOperatorToolRegistry()
        self.route_authorizations: Dict[Tuple[str, str], int] = {}
        self.original_priority_weights: Dict[str, float] = {}
        self.priority_authorizations: Dict[str, Tuple[float, int]] = {}
        self.constraint_authorizations: Dict[str, Dict[str, Any]] = {}
        self.pending_human_directives: Dict[str, Dict[str, Any]] = {}
        self.directive_history: List[Dict[str, Any]] = []
        self.autonomy_levels: Dict[str, int] = {}
        self.operator_message_count = 0
        self.operator_message_bytes = 0
        self.operator_interventions = 0
        self.operator_requests = 0
        self.emergency_material_added = 0.0
        self.material_action_records: Dict[str, Dict[str, Any]] = {}
        self._observed_completed_shipments: Set[str] = set()
        self._human_message_counter = 0
        self._request_counter = 0
        self._intervention_budget_used = 0
        super().__init__(config)
        self.autonomy_levels = {
            agent_id: int(AutonomyLevel.QUIET_DECENTRALIZED)
            for agent_id in self.agent_ids
        }
        self.original_priority_weights = {
            agent_id: state.priority_weight for agent_id, state in self.states.items()
        }
        self.emergency_route_candidates = self._build_emergency_candidates()

    def _build(self) -> None:
        requested_topology = self.config.topology
        if requested_topology in ("human_v3_development", "human_v3_holdout"):
            self.config.topology = "ring_plus_hubs"
            super()._build()
            self.config.topology = requested_topology
            ids = list(self.agents)
            self.communication_edges.clear()
            if requested_topology == "human_v3_development":
                # Two locally dense organizations joined by two independent
                # bridges. Partitions remove one bridge but not hidden state.
                midpoint = len(ids) // 2
                for region in (ids[:midpoint], ids[midpoint:]):
                    for left, right in zip(region[:-1], region[1:]):
                        self.communication_edges.add(tuple(sorted((left, right))))
                    if len(region) > 2:
                        self.communication_edges.add(tuple(sorted((region[0], region[-1]))))
                self.communication_edges.add(tuple(sorted((ids[midpoint - 1], ids[midpoint]))))
                if midpoint >= 2 and midpoint + 1 < len(ids):
                    self.communication_edges.add(tuple(sorted((ids[midpoint - 2], ids[midpoint + 1]))))
            else:
                # Fresh ladder topology reserved for a future eligible holdout.
                upper = ids[::2]
                lower = ids[1::2]
                for rail in (upper, lower):
                    for left, right in zip(rail[:-1], rail[1:]):
                        self.communication_edges.add(tuple(sorted((left, right))))
                for left, right in zip(upper, lower):
                    self.communication_edges.add(tuple(sorted((left, right))))
            sources = [agent_id for agent_id in ids if self.agents[agent_id].identity.role in SOURCE_ROLES]
            demands = [agent_id for agent_id in ids if self.agents[agent_id].identity.role in DEMAND_ROLES]
            self.physical_edges.clear()
            for demand_index, demand in enumerate(demands):
                degree = min(3, len(sources))
                stride = 2 if requested_topology == "human_v3_development" else 3
                for offset in range(degree):
                    source = sources[(stride * demand_index + offset) % len(sources)]
                    self.physical_edges.add((source, demand))
            # V3 advertises a bounded operational contact for every physical
            # supplier-demand relation. The lossy/partitioned channel still
            # governs delivery; this only makes a legitimate explicit need
            # report possible.
            for source, demand in self.physical_edges:
                self.communication_edges.add(tuple(sorted((source, demand))))
            # V3 starts demand nodes lean enough that correct replenishment can
            # affect the finite horizon. Adjust the conservation baseline.
            for demand in demands:
                old_inventory = self.states[demand].inventory
                new_inventory = min(old_inventory, 2.0)
                self.states[demand].inventory = new_inventory
                self.initial_material += new_inventory - old_inventory
            return
        super()._build()

    def _build_emergency_candidates(self) -> Set[Tuple[str, str]]:
        sources = [agent_id for agent_id in self.agent_ids if self.agents[agent_id].identity.role in SOURCE_ROLES]
        demands = [agent_id for agent_id in self.agent_ids if self.agents[agent_id].identity.role in DEMAND_ROLES]
        candidates = set(self.initial_physical_edges)
        for demand_index, demand in enumerate(demands):
            if sources:
                candidates.add((sources[(demand_index + 1) % len(sources)], demand))
        return candidates

    def apply_disruption(self) -> None:
        already_applied = self._disruption_applied
        super().apply_disruption()
        if already_applied or not self._disruption_applied or not self.human_config.human_remediable:
            return
        demands = [agent_id for agent_id in self.agent_ids if self.agents[agent_id].identity.role in DEMAND_ROLES]
        if not demands:
            return
        affected_demands = demands[:1] if self.config.disruption == "moderate" else demands[: min(2, len(demands))]
        additionally_closed: List[Tuple[str, str]] = []
        for demand in affected_demands:
            inbound = sorted(edge for edge in self.physical_edges if edge[1] == demand)
            close_count = 1 if self.config.disruption == "moderate" else max(1, len(inbound) - 1)
            if self.config.disruption == "compound":
                close_count = len(inbound)
            for edge in inbound[:close_count]:
                self.physical_edges.discard(edge)
                self.closed_physical_edges.add(edge)
                additionally_closed.append(edge)
        self.ledger.append(
            self.step_index,
            "disruption",
            "v3_actionability_mechanism",
            {
                "regime": self.config.disruption,
                "affected_demand_nodes": affected_demands,
                "additional_authority_bottleneck_edges": [list(edge) for edge in additionally_closed],
                "human_remediable": True,
                "information_boundary": "evaluator_event_not_delivered_as_true_label",
            },
        )

    def private_observation(self, agent_id: str) -> Any:
        """Return conservative v3 action affordances for continuous resources.

        V2 exposed rounded inventory/capacity values that could be a few
        floating-point ulps above the executable simulator value. V3 floors
        those two private fields, preserving privacy while guaranteeing that a
        proposal at the observed bound is actually feasible.
        """

        observation = super().private_observation(agent_id)
        # The base observation has already been rounded, so subtract one unit
        # at the reported precision after flooring to make it a certified
        # lower bound even when the rounded value crossed upward.
        observation.inventory = max(
            0.0,
            math.floor(float(observation.inventory) * 1000.0) / 1000.0 - 0.001,
        )
        observation.capacity = max(
            0.0,
            math.floor(float(observation.capacity) * 1000.0) / 1000.0 - 0.001,
        )
        return observation

    def _expire_authorizations(self) -> None:
        for edge, expires in list(self.route_authorizations.items()):
            if self.step_index <= expires:
                continue
            self.route_authorizations.pop(edge)
            if edge not in self.initial_physical_edges or edge in self.closed_physical_edges:
                self.physical_edges.discard(edge)
        for agent_id, (_, expires) in list(self.priority_authorizations.items()):
            if self.step_index <= expires:
                continue
            self.states[agent_id].priority_weight = self.original_priority_weights[agent_id]
            self.priority_authorizations.pop(agent_id)
        for agent_id, authorization in list(self.constraint_authorizations.items()):
            if self.step_index > int(authorization["expires_step"]):
                self.constraint_authorizations.pop(agent_id)

    def transition(self) -> None:
        self._expire_authorizations()
        before_completed = set(self.completed_shipments)
        super().transition()
        new_completed = sorted(set(self.completed_shipments) - before_completed)
        for shipment_id in new_completed:
            shipment = self.completed_shipments[shipment_id]
            record = self.material_action_records.get(shipment_id)
            if record is None:
                continue
            demand_reached = self.agents[shipment.recipient].identity.role in DEMAND_ROLES
            record["arrival_step"] = self.step_index
            record["reached_next_stage"] = True
            record["reached_final_demand"] = demand_reached
            self.ledger.append(
                self.step_index,
                "material_progress",
                shipment.sender,
                {
                    "proposal_id": record["proposal_id"],
                    "shipment_id": shipment_id,
                    "stage": "final_demand_arrival" if demand_reached else "next_logistics_node",
                    "recipient": shipment.recipient,
                    "quantity": shipment.quantity,
                    "human_intervention_id": record.get("human_intervention_id"),
                },
            )

    def execute_tool(self, agent_id: str, tool: str, args: Mapping[str, Any]) -> ToolResult:
        if tool == "request_human_assistance":
            self.tool_calls += 1
            self.ledger.append(
                self.step_index, "tool_call", agent_id,
                {"tool": tool, "arguments": dict(args)}, private_to=agent_id,
            )
            if args.get("assistance_kind") not in {kind.value for kind in AssistanceKind}:
                result = ToolResult(False, "invalid_assistance_kind", "unknown human-assistance request")
            else:
                result = ToolResult(True, "human_assistance_requested", "request entered the local escalation path")
                self.valid_tool_calls += 1
            self.ledger.append(
                self.step_index, "tool_result", agent_id,
                {"tool": tool, **result.as_dict()}, private_to=agent_id,
            )
            return result
        result = super().execute_tool(agent_id, tool, args)
        if result.ok and tool in ("schedule_shipment", "transfer_resource"):
            shipment_id = str(result.data["shipment_id"])
            proposal_id = "MA-%s" % shipment_id
            directive = self.pending_human_directives.get(agent_id)
            self.material_action_records[shipment_id] = {
                "proposal_id": proposal_id,
                "shipment_id": shipment_id,
                "agent_id": agent_id,
                "target": args["target"],
                "quantity": float(args["quantity"]),
                "proposed_step": self.step_index,
                "first_pass_valid": True,
                "valid_after_repair": True,
                "accepted": True,
                "entered_transit": True,
                "reached_next_stage": False,
                "reached_final_demand": False,
                "human_intervention_id": (
                    directive.get("intervention_id") if directive else None
                ),
            }
            self.ledger.append(
                self.step_index,
                "material_progress",
                agent_id,
                {
                    "proposal_id": proposal_id,
                    "shipment_id": shipment_id,
                    "stage": "entered_transit",
                    "target": args["target"],
                    "quantity": float(args["quantity"]),
                    "human_intervention_id": self.material_action_records[shipment_id]["human_intervention_id"],
                },
            )
        return result

    def _send_operator_message(self, recipient: str, kind: str, payload: Mapping[str, Any]) -> ToolResult:
        if recipient not in self.agents:
            return ToolResult(False, "invalid_recipient", "operator message recipient does not exist")
        self._human_message_counter += 1
        message = Message(
            message_id="HM%06d" % self._human_message_counter,
            sender="simulated_human_operator",
            recipient=recipient,
            kind=kind,
            payload=dict(payload),
            sent_step=self.step_index,
            deliver_step=self.step_index,
            public=False,
        )
        encoded = json.dumps(asdict(message), sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.operator_message_count += 1
        self.operator_message_bytes += len(encoded)
        self.agents[recipient].deliver_message(message)
        self.ledger.append(
            self.step_index,
            "human_directive",
            "simulated_human_operator",
            {**asdict(message), "delivery": "authenticated_operator_channel"},
            private_to=recipient,
        )
        return ToolResult(True, "operator_message_delivered", "signed operator directive delivered", {"message_id": message.message_id})

    def set_autonomy_level(self, agent_id: str, level: int, reason: str, intervention_id: Optional[str] = None) -> None:
        if agent_id not in self.agents:
            raise ValueError("unknown autonomy target")
        parsed = AutonomyLevel(int(level))
        previous = self.autonomy_levels[agent_id]
        self.autonomy_levels[agent_id] = int(parsed)
        self.ledger.append(
            self.step_index,
            "autonomy_transition",
            agent_id,
            {
                "from": previous,
                "to": int(parsed),
                "reason": reason,
                "intervention_id": intervention_id,
            },
            private_to=agent_id,
        )

    def create_assistance_request(
        self,
        agent_id: str,
        state: LocalThermodynamicState,
        score: float,
        assistance_kind: Optional[AssistanceKind] = None,
    ) -> AssistanceRequest:
        """Build a request solely from the agent's local/distributed state."""

        if agent_id != state.agent_id:
            raise PermissionError("agent may create only its own request")
        self._request_counter += 1
        self.operator_requests += 1
        kind = assistance_kind or request_assistance_kind(state)
        intervention, arguments = self._suggest_intervention(agent_id, state, kind)
        # Expected intervention value is not the anomaly score itself. A
        # system-wide anomaly at a well-stocked source may warrant monitoring
        # but offers little immediate supervisory benefit. Local service risk
        # therefore anchors the loss forecast, while thermodynamic measures
        # contribute bounded incremental evidence.
        predicted_without = float(
            5.0 * state.local_kpi_risk
            + 0.35 * min(4.0, max(0.0, state.energy_residual))
            + 0.25 * min(4.0, state.entropy_residual)
            + 0.50 * state.local_disruption_risk
        )
        predicted_fraction = {
            "authorize_emergency_route": 0.42,
            "approve_emergency_resource": 0.36,
            "authorize_information_sharing": 0.20,
            "resolve_contract_conflict": 0.25,
            "adjust_priorities": 0.18,
            "temporary_emergency_override": 0.50,
        }.get(intervention, 0.10)
        if intervention == "approve_emergency_resource" and state.local_kpi_risk < 0.25:
            predicted_fraction = 0.08
        predicted_with = max(0.0, predicted_without * (1.0 - predicted_fraction))
        uncertainty = float(min(1.0, max(
            0.0,
            0.15
            + 0.55 * (1.0 - state.consensus_confidence)
            + 0.30 * state.disagreement,
        )))
        operator_minutes = 6.0 + 4.0 * int(kind in (AssistanceKind.CONFLICT_RESOLUTION, AssistanceKind.EMERGENCY_OVERRIDE))
        request = AssistanceRequest(
            incident_id="HA%06d" % self._request_counter,
            requesting_agent=agent_id,
            application=self.application.value,
            step=self.step_index,
            assistance_kind=kind.value,
            reason=(
                "severity" if state.energy_residual >= state.entropy_residual
                else "uncertainty_or_rapid_change"
            ),
            severity=float(min(1.0, max(0.0, max(state.energy, state.local_kpi_risk)))),
            entropy_anomaly=state.entropy_residual,
            disagreement=state.disagreement,
            consensus_confidence=state.consensus_confidence,
            local_kpi_risk=state.local_kpi_risk,
            expected_loss_without=predicted_without,
            expected_loss_with=predicted_with,
            expected_benefit=predicted_without - predicted_with,
            prediction_uncertainty=uncertainty,
            estimated_operator_minutes=operator_minutes,
            priority_score=float(score),
            predicted_steps_until_collapse=max(1, int(round(5.0 * (1.0 - state.local_kpi_risk)))),
            suggested_intervention=intervention,
            intervention_arguments=arguments,
            requested_autonomy_level=int(autonomy_level_for_request(kind)),
        )
        self.ledger.append(
            self.step_index,
            "human_request",
            agent_id,
            {**request.as_dict(), "information_boundary": "private_local_plus_distributed_sketches"},
            private_to=agent_id,
        )
        self.set_autonomy_level(agent_id, request.requested_autonomy_level, "agent_requested_assistance")
        return request

    def _suggest_intervention(
        self,
        agent_id: str,
        state: LocalThermodynamicState,
        kind: AssistanceKind,
    ) -> Tuple[str, Dict[str, Any]]:
        role = self.agents[agent_id].identity.role
        if role in DEMAND_ROLES:
            known_inbound = sorted(edge for edge in self.initial_physical_edges if edge[1] == agent_id)
            # The requester knows advertised infrastructure and its own failed
            # attempts/messages, not the evaluator's closed-edge set. It
            # nominates a public inbound edge without being told whether the
            # simulator currently classifies that edge as disrupted.
            failed_targets = {
                message.sender for message in self.agents[agent_id].inbox
                if message.kind in ("route_failure", "late_delivery")
            }
            prioritized = [edge for edge in known_inbound if edge[0] in failed_targets]
            candidate = (prioritized or known_inbound)[0] if known_inbound else None
            if candidate is not None and state.local_disruption_risk >= 0.25:
                tool = "temporary_emergency_override" if kind == AssistanceKind.EMERGENCY_OVERRIDE else "authorize_emergency_route"
                arguments: Dict[str, Any] = {
                    "source": candidate[0], "target": candidate[1], "duration": min(6, self.human_config.maximum_authorization_duration),
                }
                if tool == "temporary_emergency_override":
                    arguments["quantity"] = min(12.0, self.human_config.emergency_resource_limit)
                return tool, arguments
            if state.disagreement >= 0.15 and known_inbound:
                return "authorize_information_sharing", {
                    "source": agent_id,
                    "target": known_inbound[0][0],
                    "scope": "coarse_need",
                    "duration": 4,
                }
            if self.application.value == "humanitarian":
                return "adjust_priorities", {
                    "target": agent_id,
                    "priority_weight": 2.5,
                    "duration": 5,
                }
        if role in SOURCE_ROLES and state.actionability_evidence >= 0.95:
            # A source nominates a route only after its own validated material
            # action failed. The target comes from an explicitly received
            # need/quote and public advertised topology, never evaluator state.
            reported_targets = [
                message.sender for message in reversed(self.agents[agent_id].inbox)
                if message.kind in ("need", "quote_request")
                and (agent_id, message.sender) in self.initial_physical_edges
            ]
            if reported_targets:
                return "authorize_emergency_route", {
                    "source": agent_id,
                    "target": reported_targets[0],
                    "duration": min(6, self.human_config.maximum_authorization_duration),
                }
        if role in SOURCE_ROLES and state.energy_residual >= 1.0:
            return "approve_emergency_resource", {
                "recipient": agent_id,
                "quantity": min(10.0, self.human_config.emergency_resource_limit),
            }
        # An information authorization is the bounded fallback, using only
        # public initial topology and the requester's identity.
        neighbors = sorted({
            right if left == agent_id else left
            for left, right in self.initial_communication_edges
            if agent_id in (left, right)
        })
        target = neighbors[0] if neighbors else next(other for other in self.agent_ids if other != agent_id)
        return "authorize_information_sharing", {
            "source": agent_id,
            "target": target,
            "scope": "route_status",
            "duration": 3,
        }

    def execute_human_intervention(self, intervention: OperatorIntervention) -> ToolResult:
        self.ledger.append(
            self.step_index,
            "operator_action",
            "simulated_human_operator",
            intervention.as_dict(),
        )
        validation = self.human_registry.validate(intervention.bounded_tool, intervention.arguments)
        if not validation.ok:
            self.ledger.append(
                self.step_index, "operator_result", "simulated_human_operator",
                {"intervention_id": intervention.intervention_id, **validation.as_dict()},
            )
            return validation
        if self._intervention_budget_used >= self.human_config.intervention_budget and intervention.bounded_tool not in ("reject_request", "request_more_information", "return_control"):
            result = ToolResult(False, "operator_budget_exhausted", "no bounded intervention budget remains")
        else:
            result = self._apply_validated_human_tool(intervention, validation.data)
        if result.ok and intervention.bounded_tool not in ("reject_request", "request_more_information", "return_control"):
            self._intervention_budget_used += 1
            self.operator_interventions += 1
        self.ledger.append(
            self.step_index,
            "operator_result",
            "simulated_human_operator",
            {"intervention_id": intervention.intervention_id, "tool": intervention.bounded_tool, **result.as_dict()},
        )
        return result

    def _apply_validated_human_tool(self, intervention: OperatorIntervention, args: Mapping[str, Any]) -> ToolResult:
        tool = intervention.bounded_tool
        requester = intervention.requesting_agent
        if tool == "authorize_emergency_route":
            edge = (str(args["source"]), str(args["target"]))
            if edge not in self.emergency_route_candidates and edge not in self.closed_physical_edges:
                return ToolResult(False, "unauthorized_route_scope", "route is outside the bounded emergency candidate set")
            expires = self.step_index + int(args["duration"])
            self.physical_edges.add(edge)
            self.route_authorizations[edge] = expires
            directive_target = edge[0]
            directive = {
                "intervention_id": intervention.intervention_id,
                "tool": tool,
                "source": edge[0],
                "target": edge[1],
                "maximum_quantity": 12.0,
                # Public spot-market reimbursement; the organization compares
                # it privately with its own marginal cost.
                "unit_compensation": 1.35 if self.application.value == "commercial" else 0.0,
                "relief_priority": 0.80 if self.application.value == "humanitarian" else 0.0,
                "expires_step": expires,
                "mandatory": False,
            }
        elif tool == "temporary_emergency_override":
            edge = (str(args["source"]), str(args["target"]))
            if edge not in self.emergency_route_candidates and edge not in self.closed_physical_edges:
                return ToolResult(False, "unauthorized_route_scope", "override route is outside the bounded candidate set")
            expires = self.step_index + int(args["duration"])
            self.physical_edges.add(edge)
            self.route_authorizations[edge] = expires
            directive_target = edge[0]
            directive = {
                "intervention_id": intervention.intervention_id,
                "tool": tool,
                "source": edge[0],
                "target": edge[1],
                "maximum_quantity": float(args["quantity"]),
                "expires_step": expires,
                "mandatory": True,
            }
        elif tool == "approve_emergency_resource":
            recipient = str(args["recipient"])
            if recipient not in self.states or self.agents[recipient].identity.role not in SOURCE_ROLES:
                return ToolResult(False, "emergency_resource_scope", "resource recipient must be a source organization")
            quantity = min(float(args["quantity"]), self.human_config.emergency_resource_limit - self.emergency_material_added)
            if quantity <= 0.0:
                return ToolResult(False, "emergency_resource_limit", "emergency material limit exhausted")
            self.states[recipient].inventory += quantity
            self.produced_material += quantity
            self.emergency_material_added += quantity
            self.total_cost += 2.5 * quantity
            directive_target = recipient
            directive = {
                "intervention_id": intervention.intervention_id,
                "tool": tool,
                "maximum_quantity": quantity,
                "expires_step": self.step_index + 4,
                "mandatory": False,
            }
        elif tool == "authorize_information_sharing":
            source = str(args["source"])
            target = str(args["target"])
            if source not in self.agents or target not in self.agents or source == target:
                return ToolResult(False, "information_scope", "invalid information-sharing parties")
            source_observation = self.agents[source].vault.observation(source)
            scope = str(args["scope"])
            coarse = {
                "scope": scope,
                "pressure": "high" if source_observation.service_shortfall >= 0.6 else "nominal" if source_observation.service_shortfall >= 0.2 else "low",
                "capacity": "impaired" if source_observation.impairment >= 0.3 else "available",
                "authorization_expires_step": self.step_index + int(args["duration"]),
            }
            self._send_operator_message(target, "authorized_coarse_information", coarse)
            directive_target = source
            directive = {
                "intervention_id": intervention.intervention_id,
                "tool": tool,
                "target": target,
                "scope": scope,
                "expires_step": self.step_index + int(args["duration"]),
                "mandatory": False,
            }
        elif tool == "adjust_priorities":
            target = str(args["target"])
            if target not in self.states or self.agents[target].identity.role not in DEMAND_ROLES:
                return ToolResult(False, "priority_scope", "priority target must be a demand organization")
            expires = self.step_index + int(args["duration"])
            self.states[target].priority_weight = float(args["priority_weight"])
            self.priority_authorizations[target] = (float(args["priority_weight"]), expires)
            directive_target = target
            directive = {
                "intervention_id": intervention.intervention_id,
                "tool": tool,
                "priority_weight": float(args["priority_weight"]),
                "expires_step": expires,
                "mandatory": True,
            }
        elif tool == "relax_operational_constraint":
            agent_id = str(args["agent_id"])
            if agent_id not in self.agents:
                return ToolResult(False, "constraint_scope", "constraint target does not exist")
            self.constraint_authorizations[agent_id] = {
                "constraint": args["constraint"],
                "new_limit": float(args["new_limit"]),
                "expires_step": self.step_index + int(args["duration"]),
            }
            if args["constraint"] == "cost_ceiling":
                self.agents[agent_id].utility.reservation_price = max(
                    self.agents[agent_id].utility.reservation_price,
                    float(args["new_limit"]),
                )
            directive_target = agent_id
            directive = {"intervention_id": intervention.intervention_id, "tool": tool, **self.constraint_authorizations[agent_id], "mandatory": False}
        elif tool == "resolve_contract_conflict":
            commitment_id = str(args["commitment_id"])
            commitment = self.commitments.get(commitment_id)
            if commitment is None:
                return ToolResult(False, "contract_scope", "commitment does not exist")
            resolution = str(args["resolution"])
            if resolution == "accept":
                commitment.status = "accepted"
            elif resolution == "cancel":
                commitment.status = "cancelled_by_operator"
            else:
                commitment.due_step += 2
            directive_target = commitment.resource_owner or commitment.proposer
            directive = {
                "intervention_id": intervention.intervention_id,
                "tool": tool,
                "commitment_id": commitment_id,
                "resolution": resolution,
                "expires_step": self.step_index + 3,
                "mandatory": False,
            }
        elif tool == "return_control":
            agent_id = str(args["agent_id"])
            if agent_id not in self.agents:
                return ToolResult(False, "authority_scope", "return-control target does not exist")
            self.pending_human_directives.pop(agent_id, None)
            self.set_autonomy_level(agent_id, int(AutonomyLevel.QUIET_DECENTRALIZED), "operator_returned_control", intervention.intervention_id)
            return ToolResult(True, "control_returned", "ordinary autonomous authority restored", {"agent_id": agent_id})
        elif tool == "request_more_information":
            agent_id = str(args["agent_id"])
            return self._send_operator_message(agent_id, "operator_information_request", {"topic": str(args["topic"]), "intervention_id": intervention.intervention_id})
        elif tool == "reject_request":
            self.set_autonomy_level(requester, int(AutonomyLevel.TARGETED_COORDINATION), "operator_declined_request", intervention.intervention_id)
            return ToolResult(True, "request_rejected", "request declined; autonomous operation continues")
        else:
            return ToolResult(False, "unimplemented_human_tool", "human tool has no execution semantics")

        self.pending_human_directives[directive_target] = dict(directive)
        self.directive_history.append(dict(directive))
        self._send_operator_message(directive_target, "operator_directive", directive)
        level = AutonomyLevel.EMERGENCY_OVERRIDE if directive.get("mandatory") else AutonomyLevel.HUMAN_APPROVAL
        self.set_autonomy_level(directive_target, int(level), "bounded_operator_intervention", intervention.intervention_id)
        return ToolResult(
            True,
            "human_intervention_applied",
            "bounded authority or information state changed",
            {"directive_target": directive_target, "directive": directive},
        )

    def directive_response(self, agent_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Independent organization accepts/rejects an advisory directive."""

        directive = self.pending_human_directives.get(agent_id)
        if directive is None:
            return False, None
        if int(directive.get("expires_step", self.step_index)) < self.step_index:
            self.pending_human_directives.pop(agent_id, None)
            return False, None
        if directive.get("mandatory"):
            accepted = True
        else:
            observation = self.agents[agent_id].vault.observation(agent_id)
            utility = self.agents[agent_id].utility
            benefit_pressure = max(
                float(observation.service_shortfall),
                float(observation.backlog) / max(float(observation.local_forecast), 1.0),
            )
            private_economic_acceptance = float(
                directive.get("unit_compensation", 0.0)
            ) >= float(observation.private_cost)
            relief_mandate_acceptance = (
                self.application.value == "humanitarian"
                and float(directive.get("relief_priority", 0.0))
                * utility.service >= 0.35 * utility.cost
            )
            accepted = bool(
                private_economic_acceptance
                or relief_mandate_acceptance
                or benefit_pressure * utility.service
                + float(observation.impairment) * utility.risk
                >= 0.15 * utility.cost
            )
        self.ledger.append(
            self.step_index,
            "human_directive",
            agent_id,
            {
                "intervention_id": directive.get("intervention_id"),
                "response": "accepted" if accepted else "rejected",
                "mandatory": bool(directive.get("mandatory")),
            },
            private_to=agent_id,
        )
        if not accepted:
            self.pending_human_directives.pop(agent_id, None)
            self.set_autonomy_level(agent_id, int(AutonomyLevel.TARGETED_COORDINATION), "agent_rejected_advisory", directive.get("intervention_id"))
        return accepted, deepcopy(directive)

    def authorized_edges_for(self, agent_id: str) -> Set[Tuple[str, str]]:
        return {
            edge for edge, expires in self.route_authorizations.items()
            if edge[0] == agent_id and self.step_index <= expires
        }

    def public_operator_network(
        self,
        thermodynamic_states: Optional[Mapping[str, LocalThermodynamicState]] = None,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        for agent_id, agent in self.agents.items():
            row: Dict[str, Any] = {
                "agent_id": agent_id,
                "role": agent.identity.role,
                "organization": agent.identity.organization,
                "location": list(agent.identity.location),
                "autonomy_level": self.autonomy_levels.get(agent_id, 0),
            }
            if thermodynamic_states and agent_id in thermodynamic_states:
                state = thermodynamic_states[agent_id]
                # Coarse values correspond to shared sketches, not raw private
                # observations or evaluator state.
                row.update({
                    "energy_band": "high" if state.distributed_energy >= 0.6 else "nominal" if state.distributed_energy >= 0.3 else "low",
                    "entropy_anomaly_band": "high" if state.entropy_residual >= 2.0 else "nominal" if state.entropy_residual >= 1.0 else "low",
                    "disagreement_band": "high" if state.disagreement >= 0.2 else "nominal" if state.disagreement >= 0.08 else "low",
                    "consensus_confidence_band": "high" if state.consensus_confidence >= 0.75 else "nominal" if state.consensus_confidence >= 0.4 else "low",
                })
            nodes.append(row)
        return {
            "nodes": nodes,
            "physical_edges": [list(edge) for edge in sorted(self.physical_edges)],
            "communication_edges": [list(edge) for edge in sorted(self.active_communication_edges())],
            "authorized_emergency_edges": [list(edge) for edge in sorted(self.route_authorizations)],
            "active_shipments": [
                {
                    "shipment_id": shipment.shipment_id,
                    "sender": shipment.sender,
                    "recipient": shipment.recipient,
                    "quantity_band": "large" if shipment.quantity >= 10 else "small",
                }
                for shipment in self.shipments.values()
            ],
        }

    @staticmethod
    def _rng_state_digest(rng: np.random.RandomState) -> str:
        state = rng.get_state()
        payload = {
            "algorithm": state[0],
            "keys": state[1].tolist(),
            "position": int(state[2]),
            "gaussian": int(state[3]),
            "cached": float(state[4]),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def counterfactual_snapshot(self) -> Tuple["HumanOversightEnvironment", Dict[str, str]]:
        digests = {
            "initialization": self._rng_state_digest(self.rng),
            "exogenous": self._rng_state_digest(self.exogenous_rng),
            "observation": self._rng_state_digest(self.observation_rng),
            "communication": self._rng_state_digest(self.communication_rng),
        }
        self.ledger.append(
            self.step_index,
            "counterfactual_snapshot",
            "evaluator",
            {"step": self.step_index, "rng_digests": digests, "state_digest": self.state_digest()},
        )
        return deepcopy(self), digests

    def state_digest(self) -> str:
        value = self.full_state_for_evaluator()
        value["route_authorizations"] = {"%s->%s" % edge: expiry for edge, expiry in sorted(self.route_authorizations.items())}
        value["autonomy_levels"] = dict(sorted(self.autonomy_levels.items()))
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()

    def v3_metrics(self) -> Dict[str, Any]:
        material = list(self.material_action_records.values())
        accepted = [record for record in material if record.get("accepted")]
        return {
            "operator_requests": self.operator_requests,
            "operator_interventions": self.operator_interventions,
            "operator_messages": self.operator_message_count,
            "operator_message_bytes": self.operator_message_bytes,
            "emergency_material_added": self.emergency_material_added,
            "material_actions_accepted": len(accepted),
            "material_actions_entered_transit": sum(bool(row.get("entered_transit")) for row in accepted),
            "material_actions_next_stage": sum(bool(row.get("reached_next_stage")) for row in accepted),
            "material_actions_reached_demand": sum(bool(row.get("reached_final_demand")) for row in accepted),
            "autonomy_levels": dict(self.autonomy_levels),
            "intervention_budget_used": self._intervention_budget_used,
        }
