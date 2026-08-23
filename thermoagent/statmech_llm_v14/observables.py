"""V14 memory, information, quench, and nominal-manifold observables."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from sklearn.covariance import MinCovDet

from thermoagent.statmech_llm_v12.estimators import (
    block_time_reversal_kl,
    conditional_mutual_information_history,
    shannon_entropy,
    time_shuffle_floor,
)
from thermoagent.statmech_llm_v13.observables import (
    conditional_entropy_rate,
    disagreement_density,
    integrated_correlation_time,
    local_configuration_entropy,
    macrostate_code,
    plugin_entropy,
    reference_energy,
    spatial_correlation,
    total_correlation,
)


def binary_entropy(probability: float) -> float:
    value = float(np.clip(float(probability), 0.0, 1.0))
    if value <= 0.0 or value >= 1.0:
        return 0.0
    return float(-value * math.log(value) - (1.0 - value) * math.log(1.0 - value))


def mean_reported_uncertainty(confidences: np.ndarray) -> float:
    values = np.asarray(confidences, dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("confidence values must be finite and bounded")
    return float(np.mean([binary_entropy(value) for value in values.ravel()]))


def discrete_mutual_information(left: Sequence[int], right: Sequence[int]) -> float:
    x = np.asarray(left)
    y = np.asarray(right)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("mutual-information inputs must be aligned vectors")
    joint = plugin_entropy(list(zip(x.tolist(), y.tolist())))
    return float(max(plugin_entropy(x.tolist()) + plugin_entropy(y.tolist()) - joint, 0.0))


def pairwise_information_summary(
    configurations: np.ndarray,
    adjacency: np.ndarray,
) -> Dict[str, float]:
    values = np.asarray(configurations, dtype=int)
    support = np.asarray(adjacency, dtype=bool)
    if values.ndim != 2 or support.shape != (values.shape[1], values.shape[1]):
        raise ValueError("pairwise-information inputs do not align")
    if values.shape[0] < 2 or not np.all(np.isin(values, (-1, 1))):
        raise ValueError("pairwise-information configurations must be binary")
    binary = (values > 0).astype(np.int64)
    count = float(values.shape[0])
    plus = binary.sum(axis=0).astype(float)
    count_11 = binary.T.dot(binary).astype(float)
    count_10 = plus[:, None] - count_11
    count_01 = plus[None, :] - count_11
    count_00 = count - plus[:, None] - plus[None, :] + count_11
    p_one = plus / count
    p_zero = 1.0 - p_one
    information_matrix = np.zeros_like(count_11, dtype=float)
    for joint_count, left_probability, right_probability in (
        (count_11, p_one[:, None], p_one[None, :]),
        (count_10, p_one[:, None], p_zero[None, :]),
        (count_01, p_zero[:, None], p_one[None, :]),
        (count_00, p_zero[:, None], p_zero[None, :]),
    ):
        probability = joint_count / count
        denominator = left_probability * right_probability
        valid = (joint_count > 0.0) & (denominator > 0.0)
        information_matrix[valid] += probability[valid] * np.log(
            probability[valid] / denominator[valid]
        )
    upper = np.triu(np.ones_like(support, dtype=bool), 1)
    edge_mask = upper & (support | support.T)
    all_values = information_matrix[upper]
    edge_values = information_matrix[edge_mask]
    return {
        "mean_pairwise_belief_mutual_information": float(np.mean(all_values)) if all_values.size else 0.0,
        "mean_edge_belief_mutual_information": float(np.mean(edge_values)) if edge_values.size else 0.0,
    }


def total_correlation_raw(configurations: np.ndarray) -> float:
    """Return the unadjusted plug-in total correlation in nats.

    This deliberately does not clip the result.  The historical V14 primary
    coordinate remains :func:`total_correlation`; this companion function is
    used by the versioned finite-sample audit so that raw, null, and adjusted
    quantities are all retained.
    """

    values = np.asarray(configurations, dtype=int)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("total correlation requires a two-dimensional sample")
    marginal = float(sum(plugin_entropy(values[:, column].tolist()) for column in range(values.shape[1])))
    joint = plugin_entropy([tuple(row) for row in values.tolist()])
    return float(marginal - joint)


def independently_circular_shifted(
    configurations: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Break synchronous dependence while preserving each variable's series.

    Every column is circularly shifted independently.  Marginal occupancy and
    each column's complete temporal ordering are therefore preserved, while
    zero-lag cross-variable alignment is destroyed.  The transformation never
    truncates or substitutes observations.
    """

    values = np.asarray(configurations, dtype=int)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("circular-shift null requires at least three observations")
    shifts = rng.integers(0, values.shape[0], size=values.shape[1])
    if np.all(shifts == 0):
        shifts[0] = 1
    return np.column_stack(
        [np.roll(values[:, column], int(shifts[column])) for column in range(values.shape[1])]
    )


def dependence_bias_audit(
    configurations: np.ndarray,
    adjacency: np.ndarray,
    null_replicates: int,
    seed: int,
) -> Dict[str, float]:
    """Audit plug-in dependence measures against a circular-shift null.

    ``configurations`` contains belief columns first and action columns second;
    ``adjacency`` defines the number and support of the belief columns.  The
    normalized denominator is the sum of all single-variable marginal
    entropies.  Adjusted values are intentionally allowed to be negative.
    """

    values = np.asarray(configurations, dtype=int)
    support = np.asarray(adjacency, dtype=int)
    if values.ndim != 2 or support.ndim != 2 or support.shape[0] != support.shape[1]:
        raise ValueError("dependence-audit inputs do not align")
    belief_columns = int(support.shape[0])
    if values.shape[1] < belief_columns:
        raise ValueError("configuration has fewer columns than the belief graph")
    replicates = int(null_replicates)
    if replicates < 1:
        raise ValueError("at least one null replicate is required")

    raw_total = total_correlation_raw(values)
    raw_pairwise = pairwise_information_summary(values[:, :belief_columns], support)
    denominator = float(
        sum(plugin_entropy(values[:, column].tolist()) for column in range(values.shape[1]))
    )
    rng = np.random.default_rng(int(seed))
    null_total = np.empty(replicates, dtype=float)
    null_pairwise = np.empty(replicates, dtype=float)
    null_edge = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        shifted = independently_circular_shifted(values, rng)
        null_total[replicate] = total_correlation_raw(shifted)
        information = pairwise_information_summary(shifted[:, :belief_columns], support)
        null_pairwise[replicate] = information["mean_pairwise_belief_mutual_information"]
        null_edge[replicate] = information["mean_edge_belief_mutual_information"]

    total_floor = float(np.mean(null_total))
    pairwise_floor = float(np.mean(null_pairwise))
    edge_floor = float(np.mean(null_edge))
    adjusted_total = float(raw_total - total_floor)
    normalizer = denominator if denominator > 1e-12 else 1.0
    return {
        "total_correlation_raw": float(raw_total),
        "total_correlation_null_mean": total_floor,
        "total_correlation_bias_adjusted": adjusted_total,
        "total_correlation_normalized_raw": float(raw_total / normalizer),
        "total_correlation_normalized_adjusted": float(adjusted_total / normalizer),
        "pairwise_mutual_information_raw": float(
            raw_pairwise["mean_pairwise_belief_mutual_information"]
        ),
        "pairwise_mutual_information_null_mean": pairwise_floor,
        "pairwise_mutual_information_bias_adjusted": float(
            raw_pairwise["mean_pairwise_belief_mutual_information"] - pairwise_floor
        ),
        "edge_mutual_information_raw": float(
            raw_pairwise["mean_edge_belief_mutual_information"]
        ),
        "edge_mutual_information_null_mean": edge_floor,
        "edge_mutual_information_bias_adjusted": float(
            raw_pairwise["mean_edge_belief_mutual_information"] - edge_floor
        ),
        "marginal_entropy_sum": denominator,
        "null_replicates": float(replicates),
    }


def binder_cumulant(magnetization: Sequence[float]) -> float:
    values = np.asarray(magnetization, dtype=float)
    second = float(np.mean(values ** 2)) if values.size else 0.0
    if second <= 1e-12:
        return float("nan")
    return float(1.0 - np.mean(values ** 4) / (3.0 * second ** 2))


def belief_action_lag(beliefs: np.ndarray, actions: np.ndarray, lag: int = 1) -> float:
    b = np.asarray(beliefs, dtype=float)
    a = np.asarray(actions, dtype=float)
    if b.shape != a.shape or b.ndim != 2:
        raise ValueError("belief-action lag inputs do not align")
    depth = int(lag)
    if depth < 0 or len(b) <= depth:
        return float("nan")
    if depth == 0:
        return float(np.mean(b * a))
    return float(np.mean(b[:-depth] * a[depth:]))


def normalized_cross_correlation(left: Sequence[float], right: Sequence[float], lag: int) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    depth = int(lag)
    if x.shape != y.shape or x.ndim != 1 or x.size <= abs(depth) + 2:
        return float("nan")
    if depth > 0:
        x, y = x[:-depth], y[depth:]
    elif depth < 0:
        x, y = x[-depth:], y[:depth]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def irreversibility_sensitivity(
    codes: Sequence[int],
    block_lengths: Sequence[int],
    pseudocounts: Sequence[float],
    shuffle_replicates: int,
    seed: int,
) -> list[Dict[str, float]]:
    values = np.asarray(codes, dtype=int)
    output: list[Dict[str, float]] = []
    for block in block_lengths:
        for pseudocount in pseudocounts:
            raw = block_time_reversal_kl(values, int(block), float(pseudocount))
            floor = time_shuffle_floor(
                values,
                int(block),
                float(pseudocount),
                int(shuffle_replicates),
                int(seed + 101 * int(block) + round(100 * float(pseudocount))),
            )
            output.append(
                {
                    "block_length": int(block),
                    "pseudocount": float(pseudocount),
                    "raw_block_divergence_nats_per_update": float(raw),
                    "shuffle_floor_nats_per_update": float(floor["mean"]),
                    "adjusted_irreversibility_nats_per_update": float(raw - floor["mean"]),
                }
            )
    return output


def conditional_memory_depths(codes: Sequence[int], depths: Sequence[int]) -> Dict[int, float]:
    values = np.asarray(codes, dtype=int)
    return {
        int(depth): float(conditional_mutual_information_history(values, int(depth), 0.1))
        if values.size > int(depth) + 3
        else float("nan")
        for depth in depths
    }


def standardized_nominal_fit(
    training: np.ndarray,
    estimator: str,
    ridge_fraction: float,
) -> Dict[str, np.ndarray]:
    values = np.asarray(training, dtype=float)
    if values.ndim != 2 or values.shape[0] <= values.shape[1] + 1 or np.any(~np.isfinite(values)):
        raise ValueError("nominal fitting requires a finite overdetermined matrix")
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    standardized = (values - center) / scale
    method = str(estimator)
    ridge = float(ridge_fraction)
    if not 0.0 <= ridge <= 1.0:
        raise ValueError("ridge fraction must be in [0,1]")
    if method == "euclidean":
        precision = np.eye(values.shape[1])
    elif method == "diagonal":
        variance = np.var(standardized, axis=0, ddof=1)
        precision = np.diag(1.0 / np.maximum(variance, 1e-8))
    elif method == "shrinkage":
        covariance = np.cov(standardized, rowvar=False)
        target = float(np.trace(covariance) / covariance.shape[0]) * np.eye(covariance.shape[0])
        precision = np.linalg.pinv((1.0 - ridge) * covariance + ridge * target)
    elif method == "robust":
        fitted = MinCovDet(random_state=1414, support_fraction=0.75).fit(standardized)
        covariance = fitted.covariance_
        target = float(np.trace(covariance) / covariance.shape[0]) * np.eye(covariance.shape[0])
        precision = np.linalg.pinv((1.0 - ridge) * covariance + ridge * target)
    else:
        raise ValueError("unknown nominal covariance estimator")
    return {"center": center, "scale": scale, "precision": precision}


def standardized_nominal_distance(values: np.ndarray, fit: Mapping[str, np.ndarray]) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(values, dtype=float))
    standardized = (matrix - np.asarray(fit["center"])) / np.asarray(fit["scale"])
    precision = np.asarray(fit["precision"], dtype=float)
    return np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", standardized, precision, standardized), 0.0)
    )


def recovery_time(values: Sequence[float], threshold: float, consecutive: int = 2) -> float:
    series = np.asarray(values, dtype=float)
    run = int(consecutive)
    if run < 1:
        raise ValueError("consecutive recovery count must be positive")
    for index in range(max(series.size - run + 1, 0)):
        if np.all(series[index : index + run] <= float(threshold)):
            return float(index + 1)
    return float(series.size)


def phase_path_length(values: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(matrix, axis=0), axis=1)))


def signed_polygon_area(x: Sequence[float], y: Sequence[float]) -> float:
    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    if first.shape != second.shape or first.size < 3:
        return 0.0
    return float(0.5 * (np.dot(first, np.roll(second, -1)) - np.dot(second, np.roll(first, -1))))


__all__ = [
    "belief_action_lag",
    "binary_entropy",
    "binder_cumulant",
    "conditional_entropy_rate",
    "conditional_memory_depths",
    "dependence_bias_audit",
    "discrete_mutual_information",
    "disagreement_density",
    "integrated_correlation_time",
    "independently_circular_shifted",
    "irreversibility_sensitivity",
    "local_configuration_entropy",
    "macrostate_code",
    "mean_reported_uncertainty",
    "normalized_cross_correlation",
    "pairwise_information_summary",
    "phase_path_length",
    "plugin_entropy",
    "recovery_time",
    "reference_energy",
    "signed_polygon_area",
    "spatial_correlation",
    "standardized_nominal_distance",
    "standardized_nominal_fit",
    "total_correlation",
    "total_correlation_raw",
]
