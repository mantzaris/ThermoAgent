"""Role-scoped typed tool registry and deterministic input validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .types import PlanOutput, ToolResult


@dataclass(frozen=True)
class FieldRule:
    kind: type
    required: bool = True
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fields: Mapping[str, FieldRule]
    roles: Set[str]
    mutates: bool = False
    description: str = ""

    def validate(self, args: Mapping[str, Any]) -> ToolResult:
        unknown = set(args) - set(self.fields)
        if unknown:
            return ToolResult(False, "unknown_field", "unknown fields", {"fields": sorted(unknown)})
        normalized: Dict[str, Any] = {}
        for name, rule in self.fields.items():
            if name not in args:
                if rule.required:
                    return ToolResult(False, "missing_field", "missing required field", {"field": name})
                continue
            value = args[name]
            if rule.kind is float and isinstance(value, int):
                value = float(value)
            if rule.kind is int and isinstance(value, bool):
                return ToolResult(False, "wrong_type", "boolean is not an integer argument", {"field": name})
            if not isinstance(value, rule.kind):
                return ToolResult(False, "wrong_type", "wrong argument type", {"field": name, "expected": rule.kind.__name__})
            if rule.minimum is not None and float(value) < rule.minimum:
                return ToolResult(False, "below_minimum", "argument below minimum", {"field": name})
            if rule.maximum is not None and float(value) > rule.maximum:
                return ToolResult(False, "above_maximum", "argument above maximum", {"field": name})
            if rule.choices is not None and value not in rule.choices:
                return ToolResult(False, "invalid_choice", "argument is not an allowed choice", {"field": name})
            normalized[name] = value
        return ToolResult(True, "validated", "arguments validated", normalized)


ALL_ROLES = {
    "supplier", "manufacturer", "carrier", "warehouse", "retailer",
    "ngo", "agency", "transport", "depot", "clinic", "community",
    "coordinator",
}
SOURCE_ROLES = {"supplier", "manufacturer", "warehouse", "ngo", "agency", "depot"}
TRANSPORT_ROLES = {"carrier", "transport"}
DEMAND_ROLES = {"retailer", "clinic", "community"}


def _specs() -> Dict[str, ToolSpec]:
    agent = FieldRule(str)
    quantity = FieldRule(float, minimum=0.01, maximum=1000.0)
    price = FieldRule(float, minimum=0.0, maximum=1000.0)
    due = FieldRule(int, minimum=0, maximum=10000)
    return {
        "no_op": ToolSpec("no_op", {}, set(ALL_ROLES), False, "Take no external action."),
        "central_dispatch": ToolSpec(
            "central_dispatch",
            {"source": agent, "target": agent, "quantity": quantity, "arrival_step": due},
            {"coordinator"},
            True,
            "Dispatch one shipment using only the coordinator's legally available coarse reports.",
        ),
        "inspect_private_inventory": ToolSpec("inspect_private_inventory", {}, set(ALL_ROLES), False),
        "forecast_local_demand": ToolSpec("forecast_local_demand", {}, set(ALL_ROLES), False),
        "request_info": ToolSpec("request_info", {"target": agent, "topic": FieldRule(str)}, set(ALL_ROLES), False),
        "request_quote": ToolSpec("request_quote", {"target": agent, "quantity": quantity, "due_step": due}, set(ALL_ROLES), False),
        "submit_offer": ToolSpec("submit_offer", {"target": agent, "quantity": quantity, "unit_price": price, "due_step": due}, set(ALL_ROLES), False),
        "accept_offer": ToolSpec("accept_offer", {"commitment_id": FieldRule(str)}, set(ALL_ROLES), True),
        "reject_offer": ToolSpec("reject_offer", {"commitment_id": FieldRule(str), "reason": FieldRule(str)}, set(ALL_ROLES), True),
        "counter_offer": ToolSpec("counter_offer", {"commitment_id": FieldRule(str), "quantity": quantity, "unit_price": price, "due_step": due}, set(ALL_ROLES), True),
        "schedule_shipment": ToolSpec("schedule_shipment", {"target": agent, "quantity": quantity, "arrival_step": due}, SOURCE_ROLES | TRANSPORT_ROLES, True),
        "reroute_shipment": ToolSpec("reroute_shipment", {"shipment_id": FieldRule(str), "new_target": agent}, TRANSPORT_ROLES, True),
        "expedite_shipment": ToolSpec("expedite_shipment", {"shipment_id": FieldRule(str)}, TRANSPORT_ROLES, True),
        "disclose_summary": ToolSpec("disclose_summary", {"target": agent, "level": FieldRule(str, choices=("low", "nominal", "high"))}, set(ALL_ROLES), False),
        "report_local_need": ToolSpec("report_local_need", {"target": agent, "quantity": quantity, "urgency": FieldRule(str, choices=("routine", "urgent", "critical"))}, DEMAND_ROLES | {"ngo", "agency"}, False),
        "pledge_resource": ToolSpec("pledge_resource", {"target": agent, "quantity": quantity, "due_step": due}, {"ngo", "agency", "depot", "transport"}, True),
        "transfer_resource": ToolSpec("transfer_resource", {"target": agent, "quantity": quantity, "arrival_step": due}, {"ngo", "agency", "depot", "transport"}, True),
        "request_priority": ToolSpec("request_priority", {"target": agent, "reason": FieldRule(str)}, DEMAND_ROLES | {"ngo", "agency"}, False),
        "verify_delivery": ToolSpec("verify_delivery", {"shipment_id": FieldRule(str)}, set(ALL_ROLES), False),
        "challenge_allocation": ToolSpec("challenge_allocation", {"target": agent, "reason": FieldRule(str)}, {"ngo", "agency", "clinic", "community"}, False),
        "propose_coalition": ToolSpec("propose_coalition", {"members": FieldRule(list), "purpose": FieldRule(str), "expires_step": due}, set(ALL_ROLES), True),
        "join_coalition": ToolSpec("join_coalition", {"coalition_id": FieldRule(str)}, set(ALL_ROLES), True),
        "refuse_coalition": ToolSpec("refuse_coalition", {"coalition_id": FieldRule(str), "reason": FieldRule(str)}, set(ALL_ROLES), True),
        "withdraw_coalition": ToolSpec("withdraw_coalition", {"coalition_id": FieldRule(str), "reason": FieldRule(str)}, set(ALL_ROLES), True),
    }


TOOL_SPECS = _specs()


class ToolRegistry:
    def allowed(self, role: str) -> Dict[str, ToolSpec]:
        return {name: spec for name, spec in TOOL_SPECS.items() if role in spec.roles}

    def validate(self, role: str, plan: PlanOutput) -> ToolResult:
        if plan.tool not in TOOL_SPECS:
            return ToolResult(False, "unknown_tool", "tool does not exist", {"tool": plan.tool})
        spec = TOOL_SPECS[plan.tool]
        if role not in spec.roles:
            return ToolResult(False, "permission_denied", "role may not use tool", {"role": role, "tool": plan.tool})
        return spec.validate(plan.arguments)

    def prompt_schema(self, role: str, allowed_names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        names = set(allowed_names) if allowed_names is not None else set(TOOL_SPECS)
        output: Dict[str, Any] = {}
        for name, spec in self.allowed(role).items():
            if name not in names:
                continue
            output[name] = {
                "description": spec.description,
                "arguments": {
                    field_name: {
                        "type": rule.kind.__name__,
                        "required": rule.required,
                        "minimum": rule.minimum,
                        "maximum": rule.maximum,
                        "choices": rule.choices,
                    }
                    for field_name, rule in spec.fields.items()
                },
            }
        return output


OPTION_TOOLS = {
    0: {"no_op", "inspect_private_inventory", "forecast_local_demand", "schedule_shipment", "transfer_resource"},
    1: {"request_info", "request_quote", "request_priority"},
    2: {"disclose_summary", "report_local_need"},
    3: {"request_quote", "submit_offer", "pledge_resource"},
    4: {"accept_offer", "reject_offer", "counter_offer"},
    5: {"propose_coalition", "join_coalition", "refuse_coalition", "withdraw_coalition"},
    6: {"schedule_shipment", "transfer_resource", "request_priority", "challenge_allocation", "propose_coalition"},
    7: {"schedule_shipment", "transfer_resource", "reroute_shipment", "expedite_shipment", "report_local_need", "propose_coalition", "central_dispatch"},
    8: {"no_op"},
}
