"""Locally deployable V8 belief-sketch schedulers and hysteretic triggers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .v6_entropy import PRESPECIFIED_Q, pairwise_disagreement, probability_vector, tsallis_entropy


SCHEDULERS = (
    "always_on",
    "none",
    "periodic",
    "matched_random",
    "kpi_change",
    "predictive_uncertainty_change",
    "l1_belief_drift",
    "age_of_information",
    "v7_shannon_change",
    "js_drift",
    "tsallis_spectrum_change",
    "generalized_information",
    "offline_oracle",
)


@dataclass(frozen=True)
class TriggerConfig:
    method: str = "generalized_information"
    tau_on: float = 0.115
    tau_off: float = 0.045
    cooldown_steps: int = 2
    maximum_silence_steps: int = 12
    periodic_interval_steps: int = 4
    random_probability: float = 0.30
    kpi_threshold: float = 0.10
    uncertainty_threshold: float = 0.08
    l1_threshold: float = 0.18
    shannon_threshold: float = 0.045
    js_threshold: float = 0.055
    spectrum_threshold: float = 0.060
    weights: Mapping[str, float] = field(default_factory=lambda: {
        "js": 0.45,
        "spectrum": 0.25,
        "confidence": 0.15,
        "age": 0.15,
    })
    q_values: Tuple[float, ...] = PRESPECIFIED_Q

    def __post_init__(self) -> None:
        if self.method not in SCHEDULERS:
            raise ValueError("unknown V8 belief scheduler: %s" % self.method)
        if not 0.0 <= self.tau_off < self.tau_on:
            raise ValueError("trigger hysteresis requires 0 <= tau_off < tau_on")
        if self.cooldown_steps < 0 or self.maximum_silence_steps < 1:
            raise ValueError("invalid cooldown or maximum-silence setting")
        if self.periodic_interval_steps < 1:
            raise ValueError("periodic interval must be positive")
        if not 0.0 <= self.random_probability <= 1.0:
            raise ValueError("random_probability must be in [0, 1]")
        if any(float(value) < 0.0 for value in self.weights.values()):
            raise ValueError("generalized-information weights must be nonnegative")
        if sum(float(value) for value in self.weights.values()) <= 0.0:
            raise ValueError("at least one generalized-information weight is required")


@dataclass
class TriggerReference:
    belief: Tuple[float, ...]
    confidence: float
    kpi: float
    sent_step: int
    armed: bool = True


@dataclass(frozen=True)
class TriggerDecision:
    transmit: bool
    score: float
    reason: str
    js_drift: float
    entropy_spectrum_drift: float
    confidence_drift: float
    age_fraction: float
    l1_drift: float
    shannon_drift: float
    predictive_uncertainty_drift: float
    hysteresis_release_score: float = 0.0


class LocalBeliefScheduler:
    """Per-sender scheduler that never queries peer or evaluator-private state."""

    def __init__(self, config: TriggerConfig, deterministic_seed: int = 0) -> None:
        self.config = config
        self.deterministic_seed = int(deterministic_seed)
        self.references: Dict[Tuple[str, str], TriggerReference] = {}

    def _random_draw(self, sender: str, asset: str, step: int) -> float:
        value = "%d|%s|%s|%d" % (self.deterministic_seed, sender, asset, int(step))
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64 - 1)

    def _components(
        self,
        current: np.ndarray,
        confidence: float,
        local_kpi: float,
        reference: TriggerReference,
        step: int,
    ) -> Dict[str, float]:
        previous = np.asarray(reference.belief, dtype=float)
        js = pairwise_disagreement(current, previous, q=1.0)
        spectrum = max(
            abs(tsallis_entropy(current, q) - tsallis_entropy(previous, q))
            for q in self.config.q_values
        )
        confidence_drift = abs(float(confidence) - reference.confidence)
        age = max(0, int(step) - reference.sent_step)
        age_fraction = min(age / float(self.config.maximum_silence_steps), 1.0)
        l1 = float(np.sum(np.abs(current - previous)))
        shannon = abs(tsallis_entropy(current, 1.0) - tsallis_entropy(previous, 1.0))
        uncertainty = abs((1.0 - float(np.max(current))) - (1.0 - float(np.max(previous))))
        return {
            "js": float(js),
            "spectrum": float(spectrum),
            "confidence": float(confidence_drift),
            "age": float(age_fraction),
            "l1": l1,
            "shannon": float(shannon),
            "uncertainty": float(uncertainty),
            "kpi": abs(float(local_kpi) - reference.kpi),
        }

    def evaluate(
        self,
        *,
        sender: str,
        asset: str,
        belief: Sequence[float],
        confidence: float,
        local_kpi: float,
        step: int,
        partition_healed: bool = False,
        offline_oracle_change: Optional[float] = None,
    ) -> TriggerDecision:
        current = probability_vector(belief)
        key = (str(sender), str(asset))
        reference = self.references.get(key)
        if reference is None:
            if self.config.method == "none":
                return TriggerDecision(False, 0.0, "disabled", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            return TriggerDecision(True, 1.0, "initial_reference", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        components = self._components(
            current, float(confidence), float(local_kpi), reference, int(step),
        )
        # Message age can only rise between transmissions.  It belongs in the
        # activation score, but using it to release the hysteresis latch can
        # make release mathematically impossible.  The off transition is
        # therefore based on locally observable information innovation only.
        innovation_weights = {
            name: float(self.config.weights.get(name, 0.0))
            for name in ("js", "spectrum", "confidence")
        }
        innovation_denominator = sum(innovation_weights.values())
        release_score = (
            sum(innovation_weights[name] * components[name] for name in innovation_weights)
            / innovation_denominator
            if innovation_denominator > 0.0 else 0.0
        )
        method = self.config.method
        if method == "always_on":
            transmit, score, reason = True, 1.0, "always_on"
        elif method == "none":
            transmit, score, reason = False, 0.0, "disabled"
        elif partition_healed:
            transmit, score, reason = True, 1.0, "partition_recovery"
        elif int(step) - reference.sent_step >= self.config.maximum_silence_steps:
            transmit, score, reason = True, 1.0, "maximum_silence"
        elif method == "periodic":
            transmit = int(step) - reference.sent_step >= self.config.periodic_interval_steps
            score, reason = components["age"], "periodic_due" if transmit else "periodic_wait"
        elif method == "matched_random":
            score = self._random_draw(sender, asset, step)
            transmit = score < self.config.random_probability
            reason = "matched_random_selected" if transmit else "matched_random_wait"
        elif method == "kpi_change":
            score = components["kpi"]
            transmit, reason = score >= self.config.kpi_threshold, "kpi_change"
        elif method == "predictive_uncertainty_change":
            score = components["uncertainty"]
            transmit, reason = score >= self.config.uncertainty_threshold, "predictive_uncertainty_change"
        elif method == "l1_belief_drift":
            score = components["l1"]
            transmit, reason = score >= self.config.l1_threshold, "l1_belief_drift"
        elif method == "age_of_information":
            score = components["age"]
            transmit, reason = score >= 1.0, "age_deadline"
        elif method == "v7_shannon_change":
            score = components["shannon"]
            transmit, reason = score >= self.config.shannon_threshold, "v7_shannon_change"
        elif method == "js_drift":
            score = components["js"]
            transmit, reason = score >= self.config.js_threshold, "js_drift"
        elif method == "tsallis_spectrum_change":
            score = components["spectrum"]
            transmit, reason = score >= self.config.spectrum_threshold, "tsallis_spectrum_change"
        elif method == "offline_oracle":
            if offline_oracle_change is None:
                raise ValueError("offline oracle scheduler requires evaluator-only change")
            score = float(offline_oracle_change)
            transmit, reason = score >= self.config.tau_on, "offline_oracle"
        else:
            weights = self.config.weights
            denominator = sum(float(value) for value in weights.values())
            score = sum(float(weights[name]) * components[name] for name in weights) / denominator
            if not reference.armed and release_score <= self.config.tau_off:
                reference.armed = True
            # A fresh excursion above tau_on is itself actionable even while
            # the latch remains in its active state.  Hysteresis suppresses
            # only the ambiguous tau_off--tau_on band; cooldown bounds bursts.
            transmit = score >= self.config.tau_on
            reason = (
                "generalized_information_on" if transmit and reference.armed
                else "generalized_information_continuation" if transmit
                else
                "hysteresis_wait" if not reference.armed else "below_threshold"
            )
        if transmit and int(step) - reference.sent_step < self.config.cooldown_steps and reason not in (
            "partition_recovery", "maximum_silence",
        ):
            transmit = False
            reason = "cooldown"
        return TriggerDecision(
            transmit=bool(transmit), score=float(score), reason=reason,
            js_drift=components["js"],
            entropy_spectrum_drift=components["spectrum"],
            confidence_drift=components["confidence"],
            age_fraction=components["age"],
            l1_drift=components["l1"],
            shannon_drift=components["shannon"],
            predictive_uncertainty_drift=components["uncertainty"],
            hysteresis_release_score=float(release_score),
        )

    def mark_transmitted(
        self,
        *,
        sender: str,
        asset: str,
        belief: Sequence[float],
        confidence: float,
        local_kpi: float,
        step: int,
    ) -> None:
        self.references[(str(sender), str(asset))] = TriggerReference(
            belief=tuple(float(value) for value in probability_vector(belief)),
            confidence=float(confidence),
            kpi=float(local_kpi),
            sent_step=int(step),
            # A successful transmission enters the hysteresis off state.  A
            # later evaluation must first fall to or below tau_off before the
            # trigger can arm again; maximum silence and partition recovery
            # remain explicit safety overrides in evaluate().
            armed=False,
        )
