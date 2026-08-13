"""Statistical-mechanics-inspired operational monitoring.

Energy and temperature are operational constructs, not physical quantities.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


K_MACROSTATES = 27

# Fixed before the main experiment.  These evaluator-only alternatives test
# whether monitoring conclusions depend on the domain-priority weighting used
# by the primary operational-energy construct.
ENERGY_WEIGHT_SENSITIVITY: Dict[str, Tuple[float, float, float, float]] = {
    "balanced": (0.25, 0.25, 0.25, 0.25),
    "backlog_service_heavy": (0.40, 0.10, 0.40, 0.10),
    "delay_commitment_heavy": (0.15, 0.35, 0.15, 0.35),
}


@dataclass
class MacrostateCalibration:
    thresholds: np.ndarray
    alpha: float = 0.1
    temperature: float = 0.35
    energy_weights: Tuple[float, float, float, float] = (0.35, 0.20, 0.30, 0.15)
    role_references: Dict[str, List[float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, nominal_features: Sequence[Sequence[float]], alpha: float = 0.1) -> "MacrostateCalibration":
        matrix = np.asarray(nominal_features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] < 3:
            raise ValueError("nominal_features must have shape (n>=3, 3)")
        thresholds = np.quantile(matrix, [1.0 / 3.0, 2.0 / 3.0], axis=0).T
        # Degenerate nominal dimensions still receive stable, interpretable bins.
        for dim in range(3):
            lo, hi = thresholds[dim]
            if hi - lo < 0.05:
                center = float(np.median(matrix[:, dim]))
                thresholds[dim] = [max(0.05, center - 0.1), min(0.95, center + 0.1)]
        return cls(thresholds=thresholds, alpha=alpha)

    def encode(self, features: Sequence[float]) -> int:
        values = np.clip(np.asarray(features, dtype=float), 0.0, 1.0)
        if values.shape != (3,):
            raise ValueError("macrostate requires exactly three features")
        bins = [int(np.digitize(values[i], self.thresholds[i], right=False)) for i in range(3)]
        return bins[0] * 9 + bins[1] * 3 + bins[2]

    def energies(self, weights: Optional[Sequence[float]] = None) -> np.ndarray:
        w = np.asarray(weights if weights is not None else self.energy_weights, dtype=float)
        if w.shape != (4,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("energy weights must be four nonnegative values")
        w = w / w.sum()
        centers = np.asarray([1.0 / 6.0, 0.5, 5.0 / 6.0])
        energy = np.zeros(K_MACROSTATES, dtype=float)
        for state in range(K_MACROSTATES):
            backlog_bin = state // 9
            impairment_bin = (state % 9) // 3
            strain_bin = state % 3
            backlog = centers[backlog_bin]
            impairment = centers[impairment_bin]
            strain = centers[strain_bin]
            delay = 0.55 * impairment + 0.45 * strain
            energy[state] = w[0] * backlog + w[1] * delay + w[2] * backlog + w[3] * strain
        return energy

    def healthy_reference(self, weights: Optional[Sequence[float]] = None) -> np.ndarray:
        energy = self.energies(weights)
        logits = -energy / self.temperature
        logits -= logits.max()
        q = np.exp(logits)
        return q / q.sum()

    def role_reference(self, role: str) -> np.ndarray:
        values = self.role_references.get(str(role))
        if values is None:
            return self.healthy_reference()
        reference = np.asarray(values, dtype=float)
        if reference.shape != (K_MACROSTATES,) or np.any(reference <= 0):
            raise ValueError("role reference must be a positive 27-state distribution")
        return reference / reference.sum()


def occupancy_distribution(states: Sequence[int], alpha: float = 0.1, k: int = K_MACROSTATES) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    counts = np.zeros(k, dtype=float)
    for state in states:
        if int(state) < 0 or int(state) >= k:
            raise ValueError("macrostate out of bounds")
        counts[int(state)] += 1.0
    return (counts + alpha) / (counts.sum() + alpha * k)


def role_conditioned_distribution(
    states_by_role: Mapping[str, Sequence[int]], alpha: float = 0.1, shrinkage: float = 0.5
) -> np.ndarray:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    pooled_states = [s for states in states_by_role.values() for s in states]
    pooled = occupancy_distribution(pooled_states, alpha)
    weighted = np.zeros_like(pooled)
    total = max(1, len(pooled_states))
    for states in states_by_role.values():
        role_p = occupancy_distribution(states, alpha)
        shrunk = shrinkage * role_p + (1.0 - shrinkage) * pooled
        weighted += (len(states) / total) * shrunk
    return weighted / weighted.sum()


def normalized_entropy(p: Sequence[float]) -> float:
    values = np.asarray(p, dtype=float)
    values = values[values > 0]
    if values.size <= 1:
        return 0.0
    return float(-(values * np.log(values)).sum() / np.log(len(p)))


def operational_energy(p: Sequence[float], epsilon: Sequence[float]) -> float:
    return float(np.dot(np.asarray(p, dtype=float), np.asarray(epsilon, dtype=float)))


def free_energy_gap(p: Sequence[float], q: Sequence[float], temperature: float) -> float:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if temperature <= 0 or p_arr.shape != q_arr.shape or np.any(q_arr <= 0):
        raise ValueError("invalid free-energy inputs")
    mask = p_arr > 0
    value = float(temperature * np.sum(p_arr[mask] * np.log(p_arr[mask] / q_arr[mask])))
    # Roundoff can produce tiny negative values only.
    if value < -1e-12:
        raise ArithmeticError("KL divergence became negative")
    return max(0.0, value)


def local_surprisal(state: int, q: Sequence[float], epsilon: float = 1e-12) -> float:
    return float(-np.log(float(q[int(state)]) + epsilon))


def interaction_entropy(weights: Mapping[Tuple[str, str], float]) -> float:
    positive = np.asarray([v for v in weights.values() if v > 0], dtype=float)
    if positive.size <= 1:
        return 0.0
    psi = positive / positive.sum()
    return float(-(psi * np.log(psi)).sum() / np.log(positive.size))


class RollingMacrostateMonitor:
    def __init__(self, calibration: MacrostateCalibration, window: int = 3, formulation: str = "pooled") -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.calibration = calibration
        self.window: Deque[Dict[str, int]] = deque(maxlen=window)
        self.formulation = formulation

    def update(self, states_by_agent: Mapping[str, int], roles: Mapping[str, str]) -> Dict[str, object]:
        self.window.append(dict(states_by_agent))
        all_states = [state for frame in self.window for state in frame.values()]
        if self.formulation == "role_conditioned":
            by_role: Dict[str, List[int]] = defaultdict(list)
            for frame in self.window:
                for agent_id, state in frame.items():
                    by_role[roles[agent_id]].append(state)
            p = role_conditioned_distribution(by_role, self.calibration.alpha)
        else:
            p = occupancy_distribution(all_states, self.calibration.alpha)
        energy = self.calibration.energies()
        q = self.calibration.healthy_reference()
        return {
            "p": p,
            "entropy": normalized_entropy(p),
            "energy": operational_energy(p, energy),
            "free_energy": free_energy_gap(p, q, self.calibration.temperature),
            "surprisal": {
                agent_id: local_surprisal(
                    state,
                    self.calibration.role_reference(roles[agent_id]),
                )
                for agent_id, state in states_by_agent.items()
            },
        }
