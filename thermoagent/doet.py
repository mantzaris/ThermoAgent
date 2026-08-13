"""Decentralized communication triggers for the DOET study.

The classes in this module deliberately have no reference to a logistics
environment.  A trigger consumes only one agent's distributed entropy estimate,
private local surprisal, locally observable link reliability, and explicitly
delivered neighbour alerts.  This makes the execution-time information boundary
small enough to audit independently of the simulator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, Mapping, Optional, Sequence


class CommunicationMode(IntEnum):
    """Communication intensity available to one autonomous agent."""

    QUIET = 0
    TARGETED = 1
    CRISIS = 2


@dataclass(frozen=True)
class TriggerConfig:
    """Frozen parameters for one transparent stateful trigger.

    ``direction`` is selected on validation data and is never inferred from an
    evaluation label.  ``absolute`` is two-sided.  ``high`` and ``low`` are
    one-sided.  The hysteresis candidate uses the current positive residual;
    the CUSUM candidate accumulates it over time.
    """

    trigger_type: str = "cusum"
    direction: str = "absolute"
    nominal_center: float = 0.5
    nominal_scale: float = 0.1
    rho: float = 0.85
    kappa: float = 0.25
    tau_on: float = 2.0
    tau_off: float = 0.75
    tau_crisis: float = 3.5
    minimum_dwell: int = 2
    cooldown: int = 2
    crisis_surprisal: float = 4.0
    alert_weight: float = 0.5
    max_alert_neighbors: int = 2
    quiet_gossip_rounds: int = 1
    targeted_gossip_rounds: int = 1
    crisis_gossip_rounds: int = 1
    quiet_gossip_period: int = 8
    targeted_gossip_period: int = 4
    crisis_gossip_period: int = 2
    quiet_decision_interval: int = 8
    targeted_decision_interval: int = 4
    crisis_decision_interval: int = 2
    propagation: str = "local"
    disable_gossip: bool = False

    def __post_init__(self) -> None:
        if self.trigger_type not in ("cusum", "hysteresis"):
            raise ValueError("trigger_type must be cusum or hysteresis")
        if self.direction not in ("high", "low", "absolute", "change"):
            raise ValueError("direction must be high, low, absolute, or change")
        if not math.isfinite(self.nominal_center):
            raise ValueError("nominal_center must be finite")
        if not math.isfinite(self.nominal_scale) or self.nominal_scale <= 0:
            raise ValueError("nominal_scale must be finite and positive")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if self.kappa < 0 or not math.isfinite(self.kappa):
            raise ValueError("kappa must be finite and non-negative")
        if not 0 <= self.tau_off < self.tau_on < self.tau_crisis:
            raise ValueError("thresholds must satisfy 0 <= tau_off < tau_on < tau_crisis")
        if self.minimum_dwell < 1 or self.cooldown < 0:
            raise ValueError("minimum_dwell must be positive and cooldown non-negative")
        if self.alert_weight < 0 or self.max_alert_neighbors < 0:
            raise ValueError("alert parameters must be non-negative")
        if self.propagation not in ("local", "neighbor"):
            raise ValueError("propagation must be local or neighbor")
        for value in (
            self.quiet_gossip_rounds,
            self.targeted_gossip_rounds,
            self.crisis_gossip_rounds,
            self.quiet_gossip_period,
            self.targeted_gossip_period,
            self.crisis_gossip_period,
            self.quiet_decision_interval,
            self.targeted_decision_interval,
            self.crisis_decision_interval,
        ):
            if value < 1:
                raise ValueError("gossip rounds and decision intervals must be positive")

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "TriggerConfig":
        if value is None:
            return cls()
        permitted = set(cls.__dataclass_fields__)
        unknown = set(value) - permitted
        if unknown:
            raise ValueError("unknown trigger parameters: %s" % sorted(unknown))
        return cls(**dict(value))


@dataclass
class AgentTriggerState:
    agent_id: str
    mode: int = int(CommunicationMode.QUIET)
    cumulative_statistic: float = 0.0
    standardized_residual: float = 0.0
    trigger_residual: float = 0.0
    last_entropy: Optional[float] = None
    last_step: int = -1
    mode_since_step: int = 0
    last_activation_step: int = -1
    last_deactivation_step: int = -1
    activation_count: int = 0
    false_alert_inputs: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriggerDecision:
    agent_id: str
    step: int
    previous_mode: int
    mode: int
    transitioned: bool
    activated: bool
    deactivated: bool
    standardized_residual: float
    trigger_residual: float
    cumulative_statistic: float
    local_surprisal: float
    consensus_disagreement: float
    communication_availability: float
    delivered_alerts: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistributedEntropyTrigger:
    """Independent state machines for a population of agents."""

    def __init__(
        self,
        agent_ids: Sequence[str],
        config: Optional[TriggerConfig] = None,
        normalizers: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> None:
        if not agent_ids:
            raise ValueError("at least one agent is required")
        self.config = config or TriggerConfig()
        self.states: Dict[str, AgentTriggerState] = {
            str(agent_id): AgentTriggerState(str(agent_id))
            for agent_id in agent_ids
        }
        self.normalizers: Dict[str, Dict[str, float]] = {
            str(agent_id): {
                "center": float(values["center"]),
                "scale": float(values["scale"]),
            }
            for agent_id, values in (normalizers or {}).items()
        }
        unknown = set(self.normalizers) - set(self.states)
        if unknown:
            raise ValueError("normalizer supplied for unknown agents: %s" % sorted(unknown))
        if any(
            not math.isfinite(row["center"])
            or not math.isfinite(row["scale"])
            or row["scale"] <= 0
            for row in self.normalizers.values()
        ):
            raise ValueError("all normalizers require finite center and positive scale")

    def _standardized(self, agent_id: str, entropy: float) -> float:
        normalizer = self.normalizers.get(agent_id)
        center = (
            normalizer["center"] if normalizer is not None
            else self.config.nominal_center
        )
        scale = (
            normalizer["scale"] if normalizer is not None
            else self.config.nominal_scale
        )
        return (entropy - center) / max(scale, 1e-12)

    def _residual(self, standardized: float, change: float) -> float:
        if self.config.direction == "high":
            return standardized
        if self.config.direction == "low":
            return -standardized
        if self.config.direction == "change":
            return abs(change) / max(self.config.nominal_scale, 1e-12)
        return abs(standardized)

    def update(
        self,
        agent_id: str,
        step: int,
        entropy: float,
        local_surprisal: float,
        consensus_disagreement: float,
        communication_availability: float,
        delivered_alerts: int = 0,
    ) -> TriggerDecision:
        """Advance exactly one agent from locally available inputs.

        Alerts do not directly set a mode.  Each alert adds a bounded amount
        of evidence, after which the receiving agent applies its own state
        machine.  Repeated calls for the same or an earlier step are rejected
        to keep event replay deterministic.
        """

        if agent_id not in self.states:
            raise KeyError(agent_id)
        values = (
            float(entropy),
            float(local_surprisal),
            float(consensus_disagreement),
            float(communication_availability),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("trigger inputs must be finite")
        if delivered_alerts < 0:
            raise ValueError("delivered_alerts must be non-negative")
        if not 0.0 <= communication_availability <= 1.0:
            raise ValueError("communication_availability must be in [0, 1]")
        state = self.states[agent_id]
        if int(step) <= state.last_step:
            raise ValueError("trigger steps must increase monotonically per agent")

        standardized = self._standardized(agent_id, float(entropy))
        change = 0.0 if state.last_entropy is None else float(entropy) - state.last_entropy
        base_residual = self._residual(standardized, change)
        alert_evidence = self.config.alert_weight * min(int(delivered_alerts), 1)
        # Poor consensus reduces confidence in a level alarm.  It never creates
        # evidence and never suppresses explicit neighbour evidence entirely.
        confidence = max(0.0, min(1.0, 1.0 - float(consensus_disagreement)))
        residual = base_residual * confidence + alert_evidence
        if self.config.trigger_type == "cusum":
            statistic = max(
                0.0,
                self.config.rho * state.cumulative_statistic
                + residual
                - self.config.kappa,
            )
        else:
            # The level candidate retains limited memory solely for smooth
            # decay; activation is governed by the current residual.
            statistic = max(residual, self.config.rho * state.cumulative_statistic)

        previous_mode = int(state.mode)
        mode = previous_mode
        dwell = int(step) - state.mode_since_step
        cooldown_complete = (
            state.last_deactivation_step < 0
            or int(step) - state.last_deactivation_step >= self.config.cooldown
        )
        if previous_mode == int(CommunicationMode.QUIET):
            if cooldown_complete and statistic >= self.config.tau_on:
                mode = (
                    int(CommunicationMode.CRISIS)
                    if statistic >= self.config.tau_crisis
                    or float(local_surprisal) >= self.config.crisis_surprisal
                    else int(CommunicationMode.TARGETED)
                )
        elif previous_mode == int(CommunicationMode.TARGETED):
            if (
                statistic >= self.config.tau_crisis
                or float(local_surprisal) >= self.config.crisis_surprisal
            ):
                mode = int(CommunicationMode.CRISIS)
            elif dwell >= self.config.minimum_dwell and statistic <= self.config.tau_off:
                mode = int(CommunicationMode.QUIET)
        else:
            if dwell >= self.config.minimum_dwell:
                if statistic <= self.config.tau_off:
                    mode = int(CommunicationMode.QUIET)
                elif statistic < self.config.tau_crisis:
                    mode = int(CommunicationMode.TARGETED)

        transitioned = mode != previous_mode
        activated = previous_mode == int(CommunicationMode.QUIET) and mode > previous_mode
        deactivated = previous_mode > int(CommunicationMode.QUIET) and mode == int(CommunicationMode.QUIET)
        if transitioned:
            state.mode_since_step = int(step)
        if activated:
            state.last_activation_step = int(step)
            state.activation_count += 1
        if deactivated:
            state.last_deactivation_step = int(step)
        state.mode = mode
        state.cumulative_statistic = statistic
        state.standardized_residual = standardized
        state.trigger_residual = residual
        state.last_entropy = float(entropy)
        state.last_step = int(step)

        return TriggerDecision(
            agent_id=agent_id,
            step=int(step),
            previous_mode=previous_mode,
            mode=mode,
            transitioned=transitioned,
            activated=activated,
            deactivated=deactivated,
            standardized_residual=standardized,
            trigger_residual=residual,
            cumulative_statistic=statistic,
            local_surprisal=float(local_surprisal),
            consensus_disagreement=float(consensus_disagreement),
            communication_availability=float(communication_availability),
            delivered_alerts=int(delivered_alerts),
        )

    def mode(self, agent_id: str) -> CommunicationMode:
        return CommunicationMode(self.states[agent_id].mode)

    def gossip_rounds(self, agent_id: str) -> int:
        mode = self.mode(agent_id)
        if mode == CommunicationMode.CRISIS:
            return self.config.crisis_gossip_rounds
        if mode == CommunicationMode.TARGETED:
            return self.config.targeted_gossip_rounds
        return self.config.quiet_gossip_rounds

    def decision_interval(self, agent_id: str) -> int:
        mode = self.mode(agent_id)
        if mode == CommunicationMode.CRISIS:
            return self.config.crisis_decision_interval
        if mode == CommunicationMode.TARGETED:
            return self.config.targeted_decision_interval
        return self.config.quiet_decision_interval

    def gossip_period(self, agent_id: str) -> int:
        mode = self.mode(agent_id)
        if mode == CommunicationMode.CRISIS:
            return self.config.crisis_gossip_period
        if mode == CommunicationMode.TARGETED:
            return self.config.targeted_gossip_period
        return self.config.quiet_gossip_period

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            agent_id: state.as_dict()
            for agent_id, state in sorted(self.states.items())
        }
