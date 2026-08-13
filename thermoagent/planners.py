"""Deterministic and open-weight language-model planners."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .tools import OPTION_TOOLS, ToolRegistry
from .types import CoordinationOption, PlanOutput, ToolResult


PLANNER_PROMPT_REVISION = "planner-json-v7-route-affordances"


@dataclass
class PlannerRequest:
    agent_id: str
    role: str
    application: str
    option: int
    context: Dict[str, Any]
    candidate_agents: List[Dict[str, Any]]


@dataclass
class PlannerResponse:
    output: PlanOutput
    valid_json: bool
    prompt_tokens: int = 0
    generated_tokens: int = 0
    latency_seconds: float = 0.0
    raw_text: str = ""
    recovery: str = "none"


def request_affordances(request: PlannerRequest) -> Tuple[set[str], Dict[str, Any]]:
    """Return state-dependent tools and private decision constraints.

    This runs inside one agent's planning boundary: it uses only that agent's
    delivered context and never consults the simulator or another private
    vault.  In particular, offer guidance is derived from the agent's own
    reservation price or marginal cost rather than from a shared welfare
    objective.  Counteroffers retain the original resource direction, so a
    seller responding to a buyer counter is evaluated on the seller side.
    """

    tools = set(OPTION_TOOLS[int(request.option)])
    guidance: Dict[str, Any] = {}
    action_guidance = request.context.get("material_action_guidance", {})
    if action_guidance:
        # This contains public topology plus this agent's own locally known
        # coalition memberships. It contains no other organization's state.
        guidance["material_action"] = dict(action_guidance)
    messages = request.context.get("messages", [])
    commitments = request.context.get("commitments", [])
    if int(request.option) == int(CoordinationOption.RESPOND_OFFER):
        pending = [
            commitment for commitment in commitments
            if commitment.get("status") == "proposed"
            and commitment.get("partner") == request.agent_id
        ]
        if pending:
            # A recipient evaluates its best delivered price; a resource owner
            # evaluates the best buyer counter. This prevents message arrival
            # order from deciding which private offer receives attention.
            recipient_offers = [
                offer for offer in pending
                if request.agent_id != str(offer.get("resource_owner") or offer["proposer"])
            ]
            owner_counters = [offer for offer in pending if offer not in recipient_offers]
            offer = (
                max(owner_counters, key=lambda row: float(row["unit_price"]))
                if owner_counters else
                min(recipient_offers, key=lambda row: float(row["unit_price"]))
            )
            offered = float(offer["unit_price"])
            negotiation_round = int(offer.get("negotiation_round", 0))
            resource_owner = str(offer.get("resource_owner") or offer["proposer"])
            if request.agent_id == resource_owner:
                private_cost = float(request.context["observation"]["private_cost"])
                if offered >= private_cost:
                    required = "accept_offer"
                elif negotiation_round < 2 and offered >= (2.0 / 3.0) * private_cost:
                    required = "counter_offer"
                else:
                    required = "reject_offer"
                guidance["private_offer_rule"] = {
                    "commitment_id": offer["commitment_id"],
                    "decision_side": "resource_owner",
                    "offered_unit_price": offered,
                    "private_marginal_cost": private_cost,
                    "required_tool": required,
                    "minimum_counter_price": private_cost,
                    "negotiation_round": negotiation_round,
                    "exact_argument_names": {
                        "accept_offer": ["commitment_id"],
                        "reject_offer": ["commitment_id", "reason"],
                        "counter_offer": [
                            "commitment_id", "quantity", "unit_price", "due_step"
                        ],
                    }[required],
                    "maximum_counter_quantity": float(offer["quantity"]),
                    "recommended_due_step": int(
                        request.context.get("observation", {}).get("step", 0)
                    ) + 3,
                }
            else:
                reservation = float(request.context["utility"]["reservation_price"])
                if offered <= reservation:
                    required = "accept_offer"
                elif negotiation_round < 2 and offered <= 1.5 * reservation:
                    required = "counter_offer"
                else:
                    required = "reject_offer"
                guidance["private_offer_rule"] = {
                    "commitment_id": offer["commitment_id"],
                    "decision_side": "resource_recipient",
                    "offered_unit_price": offered,
                    "private_reservation_price": reservation,
                    "required_tool": required,
                    "maximum_counter_price": reservation,
                    "negotiation_round": negotiation_round,
                    "exact_argument_names": {
                        "accept_offer": ["commitment_id"],
                        "reject_offer": ["commitment_id", "reason"],
                        "counter_offer": [
                            "commitment_id", "quantity", "unit_price", "due_step"
                        ],
                    }[required],
                    "maximum_counter_quantity": float(offer["quantity"]),
                    "recommended_due_step": int(
                        request.context.get("observation", {}).get("step", 0)
                    ) + 3,
                }
    if int(request.option) in (
        int(CoordinationOption.PROPOSE_COALITION),
        int(CoordinationOption.REQUEST_REALLOCATION),
    ) and "propose_coalition" in tools:
        coalition_ledger = request.context.get("coalitions", {})
        current_step = int(request.context.get("observation", {}).get("step", 0))
        proposals = [
            message for message in messages
            if message.get("kind") == "coalition_proposal"
            and int(message.get("payload", {}).get("expires_step", current_step)) >= current_step
            and message.get("payload", {}).get("coalition_id") not in coalition_ledger
        ]
        withdrawable = [
            (coalition_id, state)
            for coalition_id, state in coalition_ledger.items()
            if state.get("status") == "member"
            and state.get("proposer") != request.agent_id
            and int(state.get("expires_step", current_step - 1)) >= current_step
        ]
        if proposals:
            tools &= {"join_coalition", "refuse_coalition"}
            guidance["coalition_state"] = {
                "instruction": "An explicit coalition proposal was delivered. Respond using its exact coalition_id.",
                "coalition_id": proposals[-1]["payload"]["coalition_id"],
                "permitted_tools": ["join_coalition", "refuse_coalition"],
                "exact_argument_names": {
                    "join_coalition": ["coalition_id"],
                    "refuse_coalition": ["coalition_id", "reason"],
                },
            }
        elif withdrawable:
            coalition_id, _ = sorted(withdrawable)[-1]
            tools &= {"withdraw_coalition"}
            guidance["coalition_state"] = {
                "instruction": "You are a non-proposer member and may independently withdraw from this active temporary coalition.",
                "coalition_id": coalition_id,
                "permitted_tools": ["withdraw_coalition"],
                "exact_argument_names": ["coalition_id", "reason"],
            }
        else:
            tools &= {"propose_coalition"}
            direct_neighbors = set(action_guidance.get("direct_message_ids", []))
            eligible_invitees = sorted({
                str(candidate.get("agent_id"))
                for candidate in request.candidate_agents
                if candidate.get("agent_id")
                and str(candidate.get("agent_id")) != request.agent_id
                and (
                    not action_guidance
                    or str(candidate.get("agent_id")) in direct_neighbors
                )
            })
            guidance["coalition_state"] = {
                "instruction": (
                    "No coalition proposal was delivered. You are already the proposer/member. "
                    "In members, list only other invited agent IDs; never include your own agent_id "
                    "and do not invent a coalition_id."
                ),
                "proposer_already_member": request.agent_id,
                "eligible_invitee_ids": eligible_invitees,
                "exact_argument_names": ["members", "purpose", "expires_step"],
                "minimum_expires_step": current_step + 2,
                "maximum_expires_step": current_step + 8,
                "recommended_expires_step": current_step + 4,
            }
    # A structurally unavailable option can always pause safely. This is not a
    # domain decision invented by the simulator and prevents empty schemas for
    # roles that cannot execute any tool attached to an option.
    role_tools = set(ToolRegistry().allowed(request.role))
    tools &= role_tools
    if action_guidance:
        target_requirements = {
            "request_info": "direct_message_ids",
            "disclose_summary": "direct_message_ids",
            "report_local_need": "direct_message_ids",
            "request_priority": "direct_message_ids",
            "challenge_allocation": "direct_message_ids",
            "request_quote": "eligible_quote_source_ids",
            "submit_offer": "eligible_offer_target_ids",
            "pledge_resource": "eligible_offer_target_ids",
            "schedule_shipment": "known_outbound_material_ids",
            "transfer_resource": "known_outbound_material_ids",
        }
        for tool, target_group in target_requirements.items():
            if tool in tools and not action_guidance.get(target_group):
                tools.remove(tool)
    if not tools:
        tools = {"no_op"}
    return tools, guidance


def validate_request_plan(request: PlannerRequest, plan: PlanOutput) -> Optional[ToolResult]:
    """Validate a proposal against this agent's option and private constraints."""

    tools, guidance = request_affordances(request)
    if plan.tool not in tools:
        return ToolResult(
            False,
            "tool_outside_affordance",
            "tool is not available for the selected option and delivered state",
            {"allowed_tools": sorted(tools)},
        )
    material_rule = guidance.get("material_action", {})
    target_requirements = {
        "request_info": "direct_message_ids",
        "disclose_summary": "direct_message_ids",
        "report_local_need": "direct_message_ids",
        "request_priority": "direct_message_ids",
        "challenge_allocation": "direct_message_ids",
        "request_quote": "eligible_quote_source_ids",
        "submit_offer": "eligible_offer_target_ids",
        "pledge_resource": "eligible_offer_target_ids",
        "schedule_shipment": "known_outbound_material_ids",
        "transfer_resource": "known_outbound_material_ids",
    }
    target_group = target_requirements.get(plan.tool)
    if material_rule and target_group:
        allowed_targets = set(material_rule.get(target_group, []))
        if str(plan.arguments.get("target")) not in allowed_targets:
            return ToolResult(
                False,
                "target_outside_local_affordance",
                "target is not reachable through the agent's public topology or local coalition state",
                {
                    "target_group": target_group,
                    "allowed_targets": sorted(allowed_targets),
                },
            )
    offer_rule = guidance.get("private_offer_rule")
    if offer_rule and plan.tool != offer_rule["required_tool"]:
        return ToolResult(
            False,
            "private_utility_constraint",
            "proposed offer response conflicts with the agent's private utility constraint",
            {"required_tool": offer_rule["required_tool"]},
        )
    if offer_rule and plan.arguments.get("commitment_id") != offer_rule["commitment_id"]:
        return ToolResult(
            False,
            "private_commitment_mismatch",
            "proposed response does not reference the pending private commitment",
            {"required_commitment_id": offer_rule["commitment_id"]},
        )
    if offer_rule and plan.tool == "counter_offer":
        counter_price = float(plan.arguments.get("unit_price", float("nan")))
        if "maximum_counter_price" in offer_rule and (
            not math.isfinite(counter_price)
            or counter_price > float(offer_rule["maximum_counter_price"])
        ):
            return ToolResult(
                False,
                "private_counter_constraint",
                "counter price exceeds the agent's private reservation price",
                {"maximum_counter_price": offer_rule["maximum_counter_price"]},
            )
        if "minimum_counter_price" in offer_rule and (
            not math.isfinite(counter_price)
            or counter_price < float(offer_rule["minimum_counter_price"])
        ):
            return ToolResult(
                False,
                "private_counter_constraint",
                "counter price is below the resource owner's private marginal cost",
                {"minimum_counter_price": offer_rule["minimum_counter_price"]},
            )
    coalition_rule = guidance.get("coalition_state")
    if isinstance(coalition_rule, dict) and plan.tool in (
        "join_coalition", "refuse_coalition", "withdraw_coalition"
    ):
        if plan.arguments.get("coalition_id") != coalition_rule["coalition_id"]:
            return ToolResult(
                False,
                "private_coalition_mismatch",
                "coalition response does not reference the explicitly delivered proposal",
                {"required_coalition_id": coalition_rule["coalition_id"]},
            )
    return None


class MockPlanner:
    """Deterministic role-aware planner used for tests and PPO rollouts."""

    revision = "mock-v2"

    @staticmethod
    def _target(request: PlannerRequest, preferred_roles: Sequence[str]) -> str:
        for candidate in request.candidate_agents:
            if candidate["role"] in preferred_roles and candidate["agent_id"] != request.agent_id:
                return candidate["agent_id"]
        for candidate in request.candidate_agents:
            if candidate["agent_id"] != request.agent_id:
                return candidate["agent_id"]
        return request.agent_id

    @staticmethod
    def _guided_target(
        request: PlannerRequest,
        key: str,
        preferred_roles: Sequence[str],
    ) -> str:
        action_guidance = request.context.get("material_action_guidance")
        if not action_guidance:
            return MockPlanner._target(request, preferred_roles)
        allowed = set(action_guidance.get(key, []))
        for candidate in request.candidate_agents:
            if (
                candidate.get("agent_id") in allowed
                and candidate.get("role") in preferred_roles
            ):
                return str(candidate["agent_id"])
        return sorted(allowed)[0] if allowed else request.agent_id

    def plan_batch(self, requests: Sequence[PlannerRequest]) -> List[PlannerResponse]:
        return [PlannerResponse(self.plan(request), True) for request in requests]

    def plan(self, request: PlannerRequest) -> PlanOutput:
        plan = self._plan_unchecked(request)
        allowed, _ = request_affordances(request)
        if plan.tool not in allowed and "no_op" in allowed:
            return PlanOutput(
                "Pause outside local affordance.",
                "no_op",
                {},
                "No reachable validated target is locally known.",
                0.9,
            )
        return plan

    def _plan_unchecked(self, request: PlannerRequest) -> PlanOutput:
        obs = request.context["observation"]
        option = int(request.option)
        role = request.role
        inbox = request.context.get("messages", [])
        pending = [
            c for c in request.context.get("commitments", [])
            if c.get("status") == "proposed" and c.get("partner") == request.agent_id
        ]
        demand_roles = ("retailer", "clinic", "community")
        source_roles = (
            "supplier", "manufacturer", "carrier", "warehouse",
            "ngo", "agency", "transport", "depot",
        )
        target_source = self._guided_target(
            request, "eligible_quote_source_ids", source_roles
        )
        target_demand = self._guided_target(
            request, "eligible_offer_target_ids", demand_roles
        )
        if target_source == request.agent_id:
            target_source = self._guided_target(
                request, "direct_message_ids", source_roles
            )
        if target_demand == request.agent_id:
            target_demand = self._guided_target(
                request, "direct_message_ids", demand_roles
            )
        need_messages = [m for m in inbox if m.get("kind") in ("need", "quote_request")]
        if need_messages:
            reported_target = str(need_messages[-1]["sender"])
            locally_reachable = set(
                request.context.get("material_action_guidance", {}).get(
                    "known_outbound_material_ids", []
                )
            )
            if reported_target in locally_reachable:
                target_demand = reported_target

        if option == 8:
            return PlanOutput("Conserve communication.", "no_op", {}, "Silence option selected.", 0.9)
        if role == "coordinator" and option == 7:
            assigned_target = request.context.get("coordinator_assignment", {}).get("target")

            def reported_available(candidate: Mapping[str, Any]) -> bool:
                shared = candidate.get("shared_operational_state", {})
                inventory = shared.get(
                    "inventory", candidate.get("inventory_level", "unreported")
                )
                return (
                    isinstance(inventory, (int, float)) and float(inventory) > 0
                ) or inventory in ("nominal", "high")

            def reported_need(candidate: Mapping[str, Any]) -> bool:
                shared = candidate.get("shared_operational_state", {})
                backlog = shared.get(
                    "backlog", candidate.get("need_level", "unreported")
                )
                return (
                    isinstance(backlog, (int, float)) and float(backlog) >= 0
                ) or backlog in ("low", "nominal", "high")

            sources = [
                candidate for candidate in request.candidate_agents
                if candidate.get("role") in source_roles
                and reported_available(candidate)
            ]
            demands = [
                candidate for candidate in request.candidate_agents
                if candidate.get("role") in demand_roles
                and reported_need(candidate)
                and (
                    assigned_target is None
                    or candidate.get("agent_id") == assigned_target
                )
            ]
            if sources and demands:
                shared = sources[0].get("shared_operational_state", {})
                numerical_limits = [
                    float(value) for value in (
                        shared.get("inventory"), shared.get("available_capacity")
                    ) if isinstance(value, (int, float))
                ]
                quantity = min([20.0] + numerical_limits) if numerical_limits else 5.0
                return PlanOutput(
                    "Dispatch conservatively from coarse legal reports.",
                    "central_dispatch",
                    {
                        "source": sources[0]["agent_id"],
                        "target": demands[0]["agent_id"],
                        "quantity": max(0.01, quantity),
                        "arrival_step": int(obs["step"]) + 3,
                    },
                    "The coordinator uses only reported inventory and need bins.",
                    0.7,
                )
            return PlanOutput("No legal coarse report supports dispatch.", "no_op", {}, "Pause rather than infer private state.", 0.8)
        if option == 1:
            if role in demand_roles:
                quantity = max(1.0, min(50.0, float(obs["backlog"]) + float(obs["demand"])))
                return PlanOutput("Request a capacity quote.", "request_quote", {"target": target_source, "quantity": quantity, "due_step": int(obs["step"]) + 3}, "Private shortage warrants targeted information.", 0.8)
            return PlanOutput("Request a need update.", "request_info", {"target": target_demand, "topic": "current shortage and deadline"}, "Avoid untargeted disclosure.", 0.8)
        if option == 2:
            if role in demand_roles and obs["backlog"] > 0:
                quantity = max(1.0, min(50.0, float(obs["backlog"])))
                return PlanOutput("Disclose a coarse need signal.", "report_local_need", {"target": target_source, "quantity": quantity, "urgency": "critical" if obs["service_shortfall"] > 0.5 else "urgent"}, "Only the need magnitude is disclosed.", 0.8)
            level = "high" if obs["inventory"] > obs["capacity"] else "nominal"
            return PlanOutput("Disclose coarse capacity.", "disclose_summary", {"target": target_demand, "level": level}, "Share a binned summary, not private costs.", 0.8)
        if option == 3:
            if role in source_roles and obs["inventory"] > 0:
                quantity = round(max(
                    0.01,
                    min(20.0, float(obs["inventory"]), float(obs["capacity"])),
                ), 3)
                price = round(float(obs["private_cost"]) * 1.15, 2)
                return PlanOutput("Offer a bounded shipment.", "submit_offer", {"target": target_demand, "quantity": quantity, "unit_price": price, "due_step": int(obs["step"]) + 5}, "Offer remains individually rational and leaves time for explicit response and delivery.", 0.75)
            return PlanOutput("Request a quote.", "request_quote", {"target": target_source, "quantity": max(1.0, min(30.0, float(obs["backlog"]) + 1.0)), "due_step": int(obs["step"]) + 3}, "Seek terms before committing.", 0.75)
        if option == 4 and pending:
            recipient_offers = [
                offer for offer in pending
                if request.agent_id != str(offer.get("resource_owner") or offer["proposer"])
            ]
            owner_counters = [offer for offer in pending if offer not in recipient_offers]
            offer = (
                max(owner_counters, key=lambda row: float(row["unit_price"]))
                if owner_counters else
                min(recipient_offers, key=lambda row: float(row["unit_price"]))
            )
            offered = float(offer["unit_price"])
            negotiation_round = int(offer.get("negotiation_round", 0))
            resource_owner = str(offer.get("resource_owner") or offer["proposer"])
            if request.agent_id == resource_owner:
                private_cost = float(obs["private_cost"])
                if offered >= private_cost:
                    return PlanOutput("Accept the viable buyer counter.", "accept_offer", {"commitment_id": offer["commitment_id"]}, "Counter covers private marginal cost.", 0.9)
                if negotiation_round < 2 and offered >= (2.0 / 3.0) * private_cost:
                    return PlanOutput("Counter at the private cost floor.", "counter_offer", {"commitment_id": offer["commitment_id"], "quantity": float(offer["quantity"]), "unit_price": round(private_cost, 2), "due_step": int(offer["due_step"]) + 1}, "Resource owner will not counter below private marginal cost.", 0.85)
                return PlanOutput("Reject an uneconomic buyer counter.", "reject_offer", {"commitment_id": offer["commitment_id"], "reason": "price is too far below private marginal cost"}, "Agent retains refusal authority.", 0.9)
            reservation = float(request.context["utility"]["reservation_price"])
            if offered <= reservation:
                return PlanOutput("Accept the affordable offer.", "accept_offer", {"commitment_id": offer["commitment_id"]}, "Offer meets private reservation price.", 0.9)
            if negotiation_round < 2 and offered <= reservation * 1.5:
                return PlanOutput("Counter the expensive offer.", "counter_offer", {"commitment_id": offer["commitment_id"], "quantity": float(offer["quantity"]), "unit_price": round(reservation, 2), "due_step": int(offer["due_step"]) + 1}, "Counter preserves local utility.", 0.85)
            return PlanOutput("Reject an irrational offer.", "reject_offer", {"commitment_id": offer["commitment_id"], "reason": "price exceeds private reservation value"}, "Agent retains refusal authority.", 0.9)
        if option in (5, 6) and (option == 5 or not request.context.get("last_tool_ok", True)):
            coalition_ledger = request.context.get("coalitions", {})
            active_memberships = [
                coalition_id for coalition_id, state in coalition_ledger.items()
                if state.get("status") == "member"
                and state.get("proposer") != request.agent_id
                and int(state.get("expires_step", int(obs["step"]) - 1)) >= int(obs["step"])
            ]
            proposals = [
                message for message in inbox
                if message.get("kind") == "coalition_proposal"
                and message.get("payload", {}).get("coalition_id") not in coalition_ledger
                and int(message.get("payload", {}).get("expires_step", obs["step"])) >= int(obs["step"])
            ]
            if proposals:
                proposal = proposals[-1]["payload"]
                should_join = (
                    float(obs.get("impairment", 0.0)) > 0.2
                    or float(obs.get("service_shortfall", 0.0)) > 0.2
                    or "recovery" in str(proposal.get("purpose", "")).lower()
                )
                if should_join:
                    return PlanOutput("Join the delivered recovery coalition.", "join_coalition", {"coalition_id": proposal["coalition_id"]}, "The explicit proposal can restore shared route access.", 0.8)
                return PlanOutput("Refuse the delivered coalition.", "refuse_coalition", {"coalition_id": proposal["coalition_id"], "reason": "no local recovery need"}, "The agent retains refusal authority.", 0.8)
            if active_memberships:
                return PlanOutput(
                    "Withdraw from the temporary coalition.",
                    "withdraw_coalition",
                    {"coalition_id": sorted(active_memberships)[-1], "reason": "local participation is no longer preferred"},
                    "The non-proposer member retains independent withdrawal authority.",
                    0.8,
                )
            preferred_groups = [demand_roles, ("carrier", "transport", "warehouse", "depot"), source_roles]
            members: List[str] = []
            for preferred in preferred_groups:
                target = next((
                    candidate["agent_id"] for candidate in request.candidate_agents
                    if candidate["agent_id"] != request.agent_id
                    and candidate["role"] in preferred
                    and candidate["agent_id"] not in members
                ), None)
                if target is not None:
                    members.append(target)
            if not members:
                members = [c["agent_id"] for c in request.candidate_agents if c["agent_id"] != request.agent_id][:3]
            return PlanOutput("Propose a temporary recovery coalition.", "propose_coalition", {"members": members, "purpose": "pool recovery capacity and route access for current disruption", "expires_step": int(obs["step"]) + 8}, "Stress is broader than one bilateral action.", 0.7)
        accepted_to_honor = [
            commitment for commitment in request.context.get("commitments", [])
            if commitment.get("status") in ("accepted", "breached")
            and str(commitment.get("resource_owner") or commitment.get("proposer")) == request.agent_id
        ]
        if option == 0 and accepted_to_honor:
            commitment = accepted_to_honor[-1]
            quantity = float(commitment["quantity"])
            target = str(commitment.get("resource_recipient") or commitment["partner"])
            if quantity > min(float(obs["inventory"]), float(obs["capacity"])):
                return PlanOutput("Pause until committed stock can be dispatched.", "no_op", {}, "The accepted quantity exceeds currently executable inventory or handling capacity.", 0.8)
            tool = "transfer_resource" if request.application == "humanitarian" else "schedule_shipment"
            observed_lead = 1 + int(math.ceil(2.0 * float(obs.get("delay", 0.0))))
            arrival = int(obs["step"]) + observed_lead
            return PlanOutput("Honor the accepted commitment.", tool, {"target": target, "quantity": quantity, "arrival_step": arrival}, "The validated transfer implements the agent's own accepted commitment.", 0.9)
        if option in (0, 6, 7) and role in source_roles and obs["inventory"] > 0 and need_messages:
            quantity = max(0.01, min(25.0, float(obs["inventory"]), float(obs["capacity"])))
            tool = "transfer_resource" if request.application == "humanitarian" else "schedule_shipment"
            observed_lead = 1 + int(math.ceil(2.0 * float(obs.get("delay", 0.0))))
            arrival = int(obs["step"]) + observed_lead
            return PlanOutput("Dispatch stock toward an explicitly reported need.", tool, {"target": target_demand, "quantity": quantity, "arrival_step": arrival}, "A delivered need report supports a validated transfer while respecting observed delay.", 0.8)
        if option == 7 and role in demand_roles:
            quantity = max(1.0, min(50.0, float(obs["backlog"]) + float(obs["demand"])))
            return PlanOutput("Report critical local need.", "report_local_need", {"target": target_source, "quantity": quantity, "urgency": "critical"}, "Emergency replanning follows visible shortage.", 0.9)
        return PlanOutput("Continue local operations.", "no_op", {}, "No validated external action is currently justified.", 0.7)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    starts = [i for i, char in enumerate(cleaned) if char == "{"]
    for start in starts:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start : index + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        break
    return None


def coerce_plan(value: Mapping[str, Any]) -> PlanOutput:
    required = ("plan_summary", "tool", "arguments", "justification")
    if any(name not in value for name in required):
        raise ValueError("planner JSON missing required fields")
    if not isinstance(value["arguments"], dict):
        raise ValueError("arguments must be an object")
    return PlanOutput(
        plan_summary=str(value["plan_summary"])[:240],
        tool=str(value["tool"]),
        arguments=dict(value["arguments"]),
        justification=str(value["justification"])[:240],
        confidence=max(0.0, min(1.0, float(value.get("confidence", 0.5)))),
    )


class TransformersPlanner:
    """Batched independent prompts served by one frozen local model."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        max_new_tokens: int = 128,
        max_input_tokens: int = 2560,
        load_in_4bit: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(max_input_tokens)
        self.registry = ToolRegistry()
        self.mock = MockPlanner()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        quantization = None
        if load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _prompt(self, request: PlannerRequest) -> str:
        allowed_tools, private_guidance = request_affordances(request)
        tools = self.registry.prompt_schema(request.role, allowed_tools)
        observation = request.context["observation"]
        current_step = int(observation["step"])
        earliest_arrival = current_step + 1 + int(
            math.ceil(2.0 * float(observation.get("delay", 0.0)))
        )
        compact = {
            "coordination_option": int(request.option),
            "coordination_option_name": CoordinationOption(int(request.option)).name.lower(),
            "time_bounds": {
                "current_step": current_step,
                "minimum_due_step": current_step + 1,
                "maximum_due_step": current_step + 6,
                "earliest_arrival_step": earliest_arrival,
                "maximum_arrival_step": current_step + 6,
                "minimum_coalition_expires_step": current_step + 2,
                "maximum_coalition_expires_step": current_step + 8,
                "recommended_coalition_expires_step": current_step + 4,
            },
            "private_action_guidance": private_guidance,
            "allowed_tools": tools,
            "private_observation": observation,
            "private_utility": request.context["utility"],
            "identity": request.context["identity"],
            "commitments": request.context.get("commitments", [])[-4:],
            "delivered_messages": request.context.get("messages", [])[-5:],
            "entropy_estimate": request.context.get("entropy", {}),
            "last_plan": request.context.get("last_plan_summary"),
            "last_tool_ok": request.context.get("last_tool_ok"),
            "private_memories": request.context.get("memories", [])[-2:],
            "coordinator_assignment": request.context.get("coordinator_assignment"),
            "communication_mode": request.context.get("communication_mode"),
            "trigger_state": request.context.get("trigger_state"),
            "candidate_agents_public_identity": [
                candidate for candidate in request.candidate_agents
                if candidate.get("agent_id") != request.agent_id
            ],
        }
        system = (
            "You are one independent logistics organization. You know only the private state and explicitly "
            "delivered messages below. Select exactly one allowed tool. Return one compact JSON object with keys "
            "plan_summary, tool, arguments, justification, confidence. Do not reveal chain-of-thought. Do not use "
            "markdown. Keep plan_summary and justification to at most 12 words each and the whole response under "
            "90 tokens. Never invent agent IDs, commitment IDs, quantities beyond inventory, or unavailable tools. "
            "Never target yourself. Any due_step or arrival_step must be between the current step plus 1 and "
            "the current step plus 6. Coalition expiry must be between current step plus 2 and plus 8. "
            "For shipment tools, private delay implies a minimum lead of 1 + ceil(2 * delay) periods. "
            "Shipment quantity must not exceed both private inventory and current private capacity. "
            "The private_action_guidance is an authoritative result from your own private utility and commitment "
            "modules: when it contains required_tool, use exactly that tool and exact commitment_id. For an "
            "offer or counteroffer, follow its required_tool exactly. A resource recipient uses its private "
            "reservation-price ceiling; a resource owner uses its private marginal-cost floor. Never counter "
            "outside the maximum_counter_price or minimum_counter_price stated in that guidance. Select a tool "
            "whose exact name appears in allowed_tools and include exactly its listed arguments, with no extra keys. "
            "Use the explicit integer values in time_bounds instead of reusing dates from memory or earlier plans."
        )
        if private_guidance.get("material_action"):
            system += (
                " Targets for messages, offers, and material actions must come from the corresponding ID lists "
                "in private_action_guidance.material_action. These lists are public topology and your own known "
                "coalition affordances; they do not reveal another agent's private state."
            )
        if "propose_coalition" in tools:
            system += (
                " For propose_coalition, members is an invitee-only list: the proposer is already a member. "
                "Never include your own identity agent_id; use only eligible_invitee_ids from private_action_guidance."
            )
        if request.role == "coordinator":
            system += (
                " You are a coordinator, not a resource owner. The private inventory/capacity shipment limit "
                "does not apply to central_dispatch. When coordinator_assignment contains a target, use that "
                "exact target. Choose a different source with legally reported positive or nominal/high inventory. "
                "When exact inventory and available_capacity are reported, quantity must not exceed either or 20; "
                "when only bins are reported, use quantity 5. Set arrival_step to current step plus 3. "
                "Choose source only from coordinator_assignment.eligible_source_ids; those are the public "
                "physical routes to the assigned target. Use no_op when that list is empty or no legal public "
                "report supports dispatch."
            )
        if "submit_offer" in tools:
            system += " When you own surplus inventory and receive a quote_request, prefer submit_offer over requesting another quote."
        # Preserve this insertion order so current constraints, private utility
        # guidance, and exact tool schemas remain at the front if a pathological
        # context still reaches the input-token bound.
        user = json.dumps(compact, sort_keys=False, separators=(",", ":"))
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return system + "\n" + user + "\nJSON:"

    def plan_batch(self, requests: Sequence[PlannerRequest]) -> List[PlannerResponse]:
        import torch

        if not requests:
            return []
        prompts = [self._prompt(request) for request in requests]
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        prompt_width = encoded["input_ids"].shape[1]
        responses: List[PlannerResponse] = []
        for index, request in enumerate(requests):
            generated_ids = outputs[index, prompt_width:]
            raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            value = extract_json_object(raw)
            valid = True
            recovery = "none"
            try:
                if value is None:
                    raise ValueError("no JSON object")
                plan = coerce_plan(value)
            except (ValueError, TypeError):
                # A safe no-op is the only automatic recovery; the simulator
                # never invents a domain action on behalf of the model.
                valid = False
                recovery = "safe_no_op"
                plan = PlanOutput("Planner output was invalid; pause safely.", "no_op", {}, "Schema recovery permits no operational mutation.", 0.0)
            generated_tokens = int((generated_ids != self.tokenizer.pad_token_id).sum().item())
            responses.append(
                PlannerResponse(
                    output=plan,
                    valid_json=valid,
                    prompt_tokens=int(input_lengths[index]),
                    generated_tokens=generated_tokens,
                    latency_seconds=elapsed / len(requests),
                    raw_text=raw,
                    recovery=recovery,
                )
            )
        return responses
