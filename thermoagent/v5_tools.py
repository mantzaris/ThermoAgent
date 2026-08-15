"""Deterministic typed V5 tool boundary for independent agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .v5_types import OPERATOR_ACTIONS


@dataclass(frozen=True)
class V5ToolCall:
    action: str
    incident_id: str
    quantity: float = 1.0
    target_agent: Optional[str] = None
    reason_code: str = "bounded_local_evidence"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V5ToolValidation:
    ok: bool
    code: str
    normalized: Optional[V5ToolCall]


ROLE_ACTIONS: Dict[str, Tuple[str, ...]] = {
    "supplier": ("authorize_emergency_resource", "revise_commitment", "defer", "no_action"),
    "carrier": ("reroute_or_reconfigure", "deploy_repair_capacity", "revise_commitment", "defer", "no_action"),
    "warehouse": ("authorize_emergency_resource", "reroute_or_reconfigure", "revise_commitment", "defer", "no_action"),
    "retailer": ("request_peer_evidence", "revise_commitment", "defer", "no_action"),
    "ngo": ("authorize_emergency_resource", "request_peer_evidence", "revise_commitment", "defer", "no_action"),
    "regional_hub": ("authorize_emergency_resource", "reroute_or_reconfigure", "deploy_repair_capacity", "defer", "no_action"),
    "clinic": ("verify", "request_peer_evidence", "revise_commitment", "defer", "no_action"),
    "distribution_node": ("verify", "reroute_or_reconfigure", "isolate_or_quarantine", "defer", "no_action"),
    "field_crew": ("verify", "deploy_repair_capacity", "isolate_or_quarantine", "defer", "no_action"),
    "communications": ("verify", "request_peer_evidence", "reroute_or_reconfigure", "defer", "no_action"),
    "cyber_defense": ("verify", "request_peer_evidence", "isolate_or_quarantine", "defer", "no_action"),
    "resource_allocation": ("authorize_emergency_resource", "deploy_repair_capacity", "revise_commitment", "defer", "no_action"),
    "critical_load": ("request_peer_evidence", "authorize_emergency_resource", "revise_commitment", "defer", "no_action"),
}

DEFAULT_ACTIONS = (
    "verify", "request_peer_evidence", "authorize_emergency_resource",
    "reroute_or_reconfigure", "deploy_repair_capacity",
    "isolate_or_quarantine", "revise_commitment", "defer", "no_action",
)


class V5ToolRegistry:
    def allowed_actions(self, role: str) -> Tuple[str, ...]:
        return ROLE_ACTIONS.get(str(role), DEFAULT_ACTIONS)

    def schema(self, role: str) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "incident_id", "quantity", "reason_code"],
            "properties": {
                "action": {"type": "string", "enum": list(self.allowed_actions(role))},
                "incident_id": {"type": "string", "minLength": 1, "maxLength": 96},
                "quantity": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "target_agent": {"type": ["string", "null"], "maxLength": 96},
                "reason_code": {"type": "string", "minLength": 1, "maxLength": 96},
            },
        }

    def validate(
        self,
        role: str,
        incident_scope: Sequence[str],
        payload: Mapping[str, Any],
    ) -> V5ToolValidation:
        allowed_keys = {"action", "incident_id", "quantity", "target_agent", "reason_code"}
        if set(payload) - allowed_keys:
            return V5ToolValidation(False, "unexpected_field", None)
        try:
            action = str(payload["action"])
            incident_id = str(payload["incident_id"])
            quantity = float(payload.get("quantity", 1.0))
            target = payload.get("target_agent")
            reason = str(payload.get("reason_code", "bounded_local_evidence"))
        except (KeyError, TypeError, ValueError):
            return V5ToolValidation(False, "schema_type_error", None)
        if action not in OPERATOR_ACTIONS or action not in self.allowed_actions(role):
            return V5ToolValidation(False, "action_not_permitted", None)
        if incident_id not in tuple(incident_scope):
            return V5ToolValidation(False, "incident_outside_private_scope", None)
        if not 0.0 <= quantity <= 2.0:
            return V5ToolValidation(False, "quantity_out_of_bounds", None)
        if len(reason) < 1 or len(reason) > 96:
            return V5ToolValidation(False, "reason_out_of_bounds", None)
        if target is not None and (not isinstance(target, str) or len(target) > 96):
            return V5ToolValidation(False, "target_invalid", None)
        return V5ToolValidation(
            True,
            "validated",
            V5ToolCall(action, incident_id, quantity, target, reason),
        )
