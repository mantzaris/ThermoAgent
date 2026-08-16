"""Auditable generalized uncertainty and decentralized-consensus measures.

All functions operate on probability vectors and return normalized,
dimensionless information-theoretic summaries.  They are operational modeling
constructs, not literal thermodynamic state variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


PRESPECIFIED_Q = (0.5, 1.0, 1.5, 2.0, 3.0)
EPSILON = 1e-12


def probability_vector(values: Sequence[float], epsilon: float = EPSILON) -> np.ndarray:
    """Validate and normalize a finite nonnegative probability vector."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("a belief must be a one-dimensional vector with at least two states")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("belief probabilities must be finite and nonnegative")
    total = float(array.sum())
    if total <= float(epsilon):
        raise ValueError("belief probabilities must have positive mass")
    return array / total


def shannon_entropy(values: Sequence[float]) -> float:
    """Normalized Shannon entropy in [0, 1]."""
    probabilities = probability_vector(values)
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)) / np.log(len(probabilities)))


def tsallis_entropy(values: Sequence[float], q: float) -> float:
    """Normalized Tsallis entropy for q > 0, with a stable Shannon limit."""
    probabilities = probability_vector(values)
    q_value = float(q)
    if not np.isfinite(q_value) or q_value <= 0.0:
        raise ValueError("Tsallis q must be finite and positive")
    if abs(q_value - 1.0) <= 1e-7:
        return shannon_entropy(probabilities)
    denominator = 1.0 - len(probabilities) ** (1.0 - q_value)
    if abs(denominator) <= EPSILON:
        return shannon_entropy(probabilities)
    value = (1.0 - float(np.sum(probabilities ** q_value))) / denominator
    return float(np.clip(value, 0.0, 1.0))


def gini_simpson_impurity(values: Sequence[float]) -> float:
    """Normalized Gini-Simpson impurity; exactly normalized Tsallis q=2."""
    probabilities = probability_vector(values)
    maximum = 1.0 - 1.0 / len(probabilities)
    return float((1.0 - np.sum(probabilities ** 2.0)) / maximum)


def probability_gini_concentration(values: Sequence[float]) -> float:
    """Exploratory economic-style Gini coefficient over belief masses.

    This concentration statistic is deliberately named separately from the
    Gini-Simpson impurity and is not part of the primary entropy family.
    """
    probabilities = np.sort(probability_vector(values))
    count = len(probabilities)
    ranks = np.arange(1, count + 1, dtype=float)
    value = float(np.sum((2.0 * ranks - count - 1.0) * probabilities) / count)
    maximum = (count - 1.0) / count
    return float(np.clip(value / maximum, 0.0, 1.0))


def reliability_weights(values: Sequence[float]) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or not len(weights):
        raise ValueError("reliability weights must be a nonempty vector")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("reliability weights must be finite and nonnegative")
    if float(weights.sum()) <= EPSILON:
        raise ValueError("at least one reliability weight must be positive")
    return weights / float(weights.sum())


def weighted_pooled_belief(
    beliefs: Sequence[Sequence[float]], weights: Sequence[float],
) -> np.ndarray:
    matrix = np.vstack([probability_vector(value) for value in beliefs])
    normalized_weights = reliability_weights(weights)
    if len(matrix) != len(normalized_weights):
        raise ValueError("one reliability weight is required per belief")
    return probability_vector(np.sum(matrix * normalized_weights[:, None], axis=0))


def average_local_uncertainty(
    beliefs: Sequence[Sequence[float]], weights: Sequence[float], q: float,
) -> float:
    normalized_weights = reliability_weights(weights)
    entropies = np.asarray([tsallis_entropy(value, q) for value in beliefs], dtype=float)
    if len(entropies) != len(normalized_weights):
        raise ValueError("one reliability weight is required per belief")
    return float(np.dot(normalized_weights, entropies))


def generalized_disagreement(
    beliefs: Sequence[Sequence[float]], weights: Sequence[float], q: float,
) -> float:
    """Normalized Jensen-Tsallis gap; q=1 is normalized weighted JS."""
    pooled = weighted_pooled_belief(beliefs, weights)
    difference = tsallis_entropy(pooled, q) - average_local_uncertainty(
        beliefs, weights, q,
    )
    # Concavity gives nonnegativity for q>0; clip numerical roundoff only.
    if difference < -1e-9:
        raise ValueError("generalized disagreement violated concavity")
    return float(np.clip(difference, 0.0, 1.0))


def consensus_score(
    beliefs: Sequence[Sequence[float]], weights: Sequence[float], q: float,
) -> float:
    """Consensus C_q = 1 - D_q using the normalized entropy gap bound."""
    return float(1.0 - generalized_disagreement(beliefs, weights, q))


def pairwise_disagreement(
    first: Sequence[float], second: Sequence[float], q: float = 1.0,
) -> float:
    return generalized_disagreement((first, second), (0.5, 0.5), q)


def graph_weighted_disagreement(
    beliefs: Mapping[str, Sequence[float]],
    edges: Iterable[Tuple[str, str, float]],
    q: float = 1.0,
) -> float:
    """Reliability/latency/trust-weighted disagreement over delivered edges."""
    numerator = 0.0
    denominator = 0.0
    for first, second, raw_weight in edges:
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("graph edge weights must be finite and nonnegative")
        if weight == 0.0:
            continue
        if first not in beliefs or second not in beliefs:
            raise KeyError("graph edge references an unknown agent")
        numerator += weight * pairwise_disagreement(beliefs[first], beliefs[second], q)
        denominator += weight
    return float(numerator / denominator) if denominator > EPSILON else 0.0


def entropy_spectrum(values: Sequence[float]) -> Dict[str, float]:
    return {"q_%s" % str(q).replace(".", "_"): tsallis_entropy(values, q) for q in PRESPECIFIED_Q}


@dataclass(frozen=True)
class TemporalInformationState:
    value: float
    slope: float
    acceleration: float
    ewma: float
    time_above_threshold: int


def temporal_information_state(
    history: Sequence[float],
    threshold: float,
    persistence: float = 0.70,
) -> TemporalInformationState:
    if not history:
        raise ValueError("temporal information history cannot be empty")
    values = np.asarray(history, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("temporal information history must be finite")
    if not 0.0 <= persistence < 1.0:
        raise ValueError("EWMA persistence must be in [0, 1)")
    slope = float(values[-1] - values[-2]) if len(values) >= 2 else 0.0
    previous_slope = float(values[-2] - values[-3]) if len(values) >= 3 else 0.0
    ewma = float(values[0])
    for value in values[1:]:
        ewma = persistence * ewma + (1.0 - persistence) * float(value)
    above = 0
    for value in values[::-1]:
        if float(value) <= float(threshold):
            break
        above += 1
    return TemporalInformationState(
        value=float(values[-1]),
        slope=slope,
        acceleration=slope - previous_slope,
        ewma=ewma,
        time_above_threshold=above,
    )
