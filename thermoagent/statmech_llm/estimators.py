"""Finite-sample estimators for empirical LLM transition irreversibility.

The estimators here deliberately use qualified names.  A transition-pair KL
computed after projecting an LLM trajectory to ``(b, a)`` is a coarse-grained
time-reversal irreversibility lower bound, not automatically the total physical
entropy production of the hidden prompt-and-memory process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TransitionIrreversibility:
    estimate_per_transition: float
    transition_count: int
    occupied_directed_pairs: int
    pseudocount: float


def transition_counts(states: Sequence[int], n_states: int) -> Array:
    values = np.asarray(states, dtype=int)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two scalar states are required")
    if np.any(values < 0) or np.any(values >= int(n_states)):
        raise ValueError("state index out of bounds")
    counts = np.zeros((int(n_states), int(n_states)), dtype=np.int64)
    np.add.at(counts, (values[:-1], values[1:]), 1)
    return counts


def transition_pair_irreversibility(
    counts: Array,
    pseudocount: float = 0.5,
    support: Optional[Array] = None,
) -> TransitionIrreversibility:
    """Estimate ``D_KL(q(x,y) || q(y,x))`` from transition-pair counts.

    Pseudocounts are added only to a symmetric declared support.  If support is
    omitted, the union of observed forward and reverse pairs is used.  This
    avoids assigning mass to physically impossible transitions while ensuring
    a finite pairwise estimate.  Self transitions contribute zero.
    """

    raw = np.asarray(counts, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("counts must be a square matrix")
    if np.any(raw < 0.0) or pseudocount < 0.0:
        raise ValueError("counts and pseudocount must be nonnegative")
    if support is None:
        active = (raw + raw.T) > 0.0
        np.fill_diagonal(active, np.diag(raw) > 0.0)
    else:
        active = np.asarray(support, dtype=bool)
        if active.shape != raw.shape or not np.array_equal(active, active.T):
            raise ValueError("support must be symmetric and match counts")
    smoothed = raw.copy()
    smoothed[active] += float(pseudocount)
    total = float(np.sum(smoothed))
    if total <= 0.0:
        raise ValueError("no transitions are present")
    joint = smoothed / total
    reverse = joint.T
    mask = active & (joint > 0.0) & (reverse > 0.0)
    estimate = float(np.sum(joint[mask] * np.log(joint[mask] / reverse[mask])))
    return TransitionIrreversibility(
        estimate_per_transition=max(0.0, estimate),
        transition_count=int(np.sum(raw)),
        occupied_directed_pairs=int(np.count_nonzero(active)),
        pseudocount=float(pseudocount),
    )


def block_time_reversal_kl(
    states: Sequence[int],
    block_length: int,
    pseudocount: float = 0.5,
) -> float:
    """Order-k block irreversibility divided by ``k-1`` transitions."""

    values = tuple(int(value) for value in states)
    k = int(block_length)
    if k < 2 or len(values) < k:
        raise ValueError("block length must be at least two and fit the trajectory")
    counts: Dict[Tuple[int, ...], float] = {}
    for start in range(len(values) - k + 1):
        block = values[start : start + k]
        counts[block] = counts.get(block, 0.0) + 1.0
    support = set(counts)
    support.update(tuple(reversed(block)) for block in tuple(support))
    adjusted = {block: counts.get(block, 0.0) + pseudocount for block in support}
    total = float(sum(adjusted.values()))
    divergence = 0.0
    for block, count in adjusted.items():
        reverse_count = adjusted[tuple(reversed(block))]
        probability = count / total
        divergence += probability * np.log(count / reverse_count)
    return float(max(0.0, divergence) / float(k - 1))


def shuffled_bias_floor(
    states: Sequence[int],
    n_states: int,
    pseudocount: float,
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    """Finite-sample null obtained by time-order permutation."""

    values = np.asarray(states, dtype=int)
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(replicates)):
        permuted = values[rng.permutation(values.size)]
        counts = transition_counts(permuted, n_states)
        estimates.append(transition_pair_irreversibility(counts, pseudocount).estimate_per_transition)
    samples = np.asarray(estimates, dtype=float)
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "quantile_95": float(np.quantile(samples, 0.95)),
        "replicates": float(samples.size),
    }


def conditional_mutual_information_markov(states: Sequence[int], pseudocount: float = 0.0) -> float:
    """Plug-in ``I(X[t+1]; X[t-1] | X[t])`` for Markov-state diagnostics."""

    values = np.asarray(states, dtype=int)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("at least three states are required")
    unique = np.unique(values)
    remap = {int(value): index for index, value in enumerate(unique)}
    mapped = np.asarray([remap[int(value)] for value in values], dtype=int)
    size = len(unique)
    triples = np.full((size, size, size), float(pseudocount), dtype=float)
    np.add.at(triples, (mapped[:-2], mapped[1:-1], mapped[2:]), 1.0)
    total = float(np.sum(triples))
    p_xyz = triples / total
    p_xy = np.sum(p_xyz, axis=2)
    p_yz = np.sum(p_xyz, axis=0)
    p_y = np.sum(p_xy, axis=0)
    result = 0.0
    for x in range(size):
        for y in range(size):
            for z in range(size):
                probability = p_xyz[x, y, z]
                denominator = p_xy[x, y] * p_yz[y, z]
                numerator = probability * p_y[y]
                if probability > 0.0 and denominator > 0.0 and numerator > 0.0:
                    result += probability * np.log(numerator / denominator)
    return float(max(0.0, result))


def stationary_chain_sample(kernel: Array, steps: int, seed: int, initial: Optional[int] = None) -> Array:
    """Sample a finite row-stochastic kernel for estimator validation."""

    matrix = np.asarray(kernel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel must be square")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("kernel rows must sum to one")
    rng = np.random.default_rng(int(seed))
    current = int(rng.integers(matrix.shape[0]) if initial is None else initial)
    output = np.empty(int(steps) + 1, dtype=int)
    output[0] = current
    for index in range(int(steps)):
        current = int(rng.choice(matrix.shape[0], p=matrix[current]))
        output[index + 1] = current
    return output


def known_three_state_cycle(clockwise: float, counterclockwise: float, stay: float) -> Array:
    """Return an exactly analyzable homogeneous three-state cycle."""

    if min(clockwise, counterclockwise, stay) < 0.0:
        raise ValueError("probabilities must be nonnegative")
    if not np.isclose(clockwise + counterclockwise + stay, 1.0):
        raise ValueError("probabilities must sum to one")
    kernel = np.zeros((3, 3), dtype=float)
    for state in range(3):
        kernel[state, state] = stay
        kernel[state, (state + 1) % 3] = clockwise
        kernel[state, (state - 1) % 3] = counterclockwise
    return kernel


def exact_cycle_entropy_production(clockwise: float, counterclockwise: float) -> float:
    """Exact EPR per transition opportunity for the homogeneous cycle."""

    if clockwise <= 0.0 or counterclockwise <= 0.0:
        raise ValueError("both directed rates must be positive")
    return float((clockwise - counterclockwise) * np.log(clockwise / counterclockwise))


def fit_logistic_response(
    local_fields: Sequence[float],
    choices: Sequence[int],
    previous_choices: Optional[Sequence[int]] = None,
    option_order: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Fit a transparent logistic LLM response and report effective temperature.

    The analytical heat bath has ``P(+1|ell)=logit^{-1}(2 ell/T)``.  Therefore
    the fitted effective decision temperature is ``2 / beta_field`` when the
    field slope is positive.  This parameter is unrelated to inference sampling
    temperature.
    """

    from scipy.optimize import minimize

    fields = np.asarray(local_fields, dtype=float)
    targets = (np.asarray(choices, dtype=int) > 0).astype(float)
    if fields.shape != targets.shape or fields.ndim != 1:
        raise ValueError("fields and choices must be aligned vectors")
    columns: List[Array] = [np.ones(fields.size), fields]
    names = ["intercept", "field_slope"]
    if previous_choices is not None:
        previous = np.asarray(previous_choices, dtype=float)
        if previous.shape != fields.shape:
            raise ValueError("previous choices shape mismatch")
        columns.append(previous)
        names.append("hysteresis_slope")
    if option_order is not None:
        order = np.asarray(option_order, dtype=float)
        if order.shape != fields.shape:
            raise ValueError("option order shape mismatch")
        columns.append(order)
        names.append("option_order_slope")
    design = np.column_stack(columns)

    def objective(coefficients: Array) -> float:
        linear = np.clip(design.dot(coefficients), -40.0, 40.0)
        return float(np.sum(np.logaddexp(0.0, linear) - targets * linear))

    def gradient(coefficients: Array) -> Array:
        linear = np.clip(design.dot(coefficients), -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        return np.asarray(design.T.dot(probability - targets), dtype=float)

    # Supplying the analytic score avoids false ``precision loss`` failures
    # from finite-difference BFGS on large, otherwise well-conditioned pilots.
    fit = minimize(objective, np.zeros(design.shape[1]), jac=gradient, method="BFGS")
    coefficients = np.asarray(fit.x, dtype=float)
    linear = np.clip(design.dot(coefficients), -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-linear))
    brier = float(np.mean((probability - targets) ** 2))
    result = {name: float(value) for name, value in zip(names, coefficients)}
    result.update(
        {
            "effective_decision_temperature": float(2.0 / coefficients[1])
            if coefficients[1] > 0.0
            else float("nan"),
            "negative_log_likelihood": objective(coefficients),
            "brier_score": brier,
            "converged": float(bool(fit.success)),
            "sample_count": float(fields.size),
        }
    )
    return result
