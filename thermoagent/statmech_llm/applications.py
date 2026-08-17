"""Secondary, abstract application mappings for the V10 LLM-agent model.

These deterministic transition shells make the two binary plans consequential;
they are not field-valid logistics or power-system simulators and are not used
to search for a favorable control result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .agents import StructuredDecision


@dataclass
class HumanitarianState:
    east_inventory: float = 12.0
    west_inventory: float = 12.0
    east_unmet_need: float = 10.0
    west_unmet_need: float = 10.0
    shared_vehicle_hours: float = 8.0
    cumulative_unmet_need: float = 0.0


@dataclass
class UtilityState:
    reroute_capacity: float = 8.0
    isolation_capacity: float = 6.0
    critical_unserved_load: float = 10.0
    cascade_risk: float = 0.35
    crew_hours: float = 8.0
    cumulative_unserved_load: float = 0.0


class HumanitarianCoordinationMapping:
    """Binary plans allocate one shared convoy east or west."""

    def __init__(self) -> None:
        self.state = HumanitarianState()

    def apply(self, decision: StructuredDecision) -> Dict[str, float]:
        before = self.state.east_unmet_need + self.state.west_unmet_need
        if decision.tool_action in ("commit_plan_left", "commit_plan_right") and self.state.shared_vehicle_hours >= 1.0:
            east = decision.tool_action == "commit_plan_left"
            inventory_name = "east_inventory" if east else "west_inventory"
            need_name = "east_unmet_need" if east else "west_unmet_need"
            available = getattr(self.state, inventory_name)
            need = getattr(self.state, need_name)
            delivered = min(2.0, available, need)
            setattr(self.state, inventory_name, available - delivered)
            setattr(self.state, need_name, need - delivered)
            self.state.shared_vehicle_hours -= 1.0
        # Persistent demand makes delayed coordination consequential.
        self.state.east_unmet_need += 0.15
        self.state.west_unmet_need += 0.15
        after = self.state.east_unmet_need + self.state.west_unmet_need
        self.state.cumulative_unmet_need += after
        return {
            "service_before": before,
            "service_after": after,
            "causal_service_change": after - (before + 0.30),
            "resource_remaining": self.state.shared_vehicle_hours,
        }


class DefensiveUtilityMapping:
    """Binary plans choose defensive isolation or service rerouting."""

    def __init__(self) -> None:
        self.state = UtilityState()

    def apply(self, decision: StructuredDecision) -> Dict[str, float]:
        before = self.state.critical_unserved_load
        if decision.tool_action == "commit_plan_left" and self.state.isolation_capacity >= 1.0:
            # Isolation reduces cascade risk but temporarily sheds service.
            self.state.isolation_capacity -= 1.0
            self.state.cascade_risk = max(0.0, self.state.cascade_risk - 0.12)
            self.state.critical_unserved_load += 0.35
        elif decision.tool_action == "commit_plan_right" and self.state.reroute_capacity >= 1.0:
            # Rerouting restores load but is harmful when unresolved cascade risk is high.
            self.state.reroute_capacity -= 1.0
            if self.state.cascade_risk > 0.30:
                self.state.critical_unserved_load += 0.55
                self.state.cascade_risk = min(1.0, self.state.cascade_risk + 0.08)
            else:
                self.state.critical_unserved_load = max(0.0, self.state.critical_unserved_load - 1.5)
        self.state.cumulative_unserved_load += self.state.critical_unserved_load
        return {
            "service_before": before,
            "service_after": self.state.critical_unserved_load,
            "causal_service_change": self.state.critical_unserved_load - before,
            "cascade_risk": self.state.cascade_risk,
        }


def application_roles(application: str, n_agents: int) -> Tuple[str, ...]:
    if application == "humanitarian":
        base = ("depot_coordinator", "field_assessor", "carrier", "clinic_liaison")
    elif application == "utility":
        base = ("component_operator", "crew_coordinator", "communication_relay", "safety_monitor")
    else:
        raise ValueError("unknown application")
    return tuple(base[index % len(base)] for index in range(int(n_agents)))


def private_evidence_text(application: str, latent_field: float, template: int) -> str:
    """Map a controlled field to natural language without exposing global truth."""

    strength = abs(float(latent_field))
    if strength < 0.20:
        qualifier = "weak and conflicting"
    elif strength < 0.60:
        qualifier = "moderate"
    else:
        qualifier = "strong"
    favored = "plan_right" if latent_field >= 0.0 else "plan_left"
    if application == "humanitarian":
        phrases = (
            "Your local route and demand reports give %s support for %s.",
            "The depot evidence available to you gives %s support for %s.",
            "Your private field assessment provides %s evidence favoring %s.",
        )
    elif application == "utility":
        phrases = (
            "Your local defensive telemetry gives %s support for %s.",
            "The component evidence visible to your role gives %s support for %s.",
            "Your private restoration checks provide %s evidence favoring %s.",
        )
    else:
        raise ValueError("unknown application")
    return phrases[int(template) % len(phrases)] % (qualifier, favored)


def matched_local_fields(n_agents: int, seed: int, magnitude: float = 0.55) -> np.ndarray:
    """Balanced evidence fields for counterbalanced LLM panels."""

    rng = np.random.default_rng(int(seed))
    values = rng.normal(0.0, float(magnitude), int(n_agents))
    values -= np.mean(values)
    return values
