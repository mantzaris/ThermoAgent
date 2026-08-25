"""Secondary finite-system observables for reconstructed V15 trajectories.

The functions in this module operate on one complete graph trajectory at a
time.  Pooling and uncertainty are performed only after these trajectory-level
quantities have been calculated; individual updates and node pairs are never
treated as independent experimental replicates.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


PHASES = ("baseline", "disruption", "recovery")


def parse_spin_vector(value: object) -> np.ndarray:
    """Decode a semicolon-separated binary state vector."""

    vector = np.asarray([int(item) for item in str(value).split(";")], dtype=int)
    if vector.ndim != 1 or vector.size == 0 or np.any(~np.isin(vector, (-1, 1))):
        raise ValueError("expected a nonempty semicolon-separated binary vector")
    return vector


def spin_matrix(values: Iterable[object]) -> np.ndarray:
    """Decode aligned state vectors into a time-by-agent matrix."""

    rows = [parse_spin_vector(value) for value in values]
    if not rows:
        raise ValueError("at least one state vector is required")
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError("state-vector widths differ")
    return np.vstack(rows)


def connected_correlation_matrix(configurations: np.ndarray) -> np.ndarray:
    r"""Return the finite-window connected correlation matrix.

    For binary belief configurations ``b[t, i]`` this is

    ``<b_i b_j> - <b_i><b_j>``,

    where every average is over attempted-update observations in one phase of
    one complete trajectory.
    """

    values = np.asarray(configurations, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("connected correlation needs at least two times and agents")
    if np.any(~np.isfinite(values)) or np.any(~np.isin(values, (-1.0, 1.0))):
        raise ValueError("connected correlation requires finite binary states")
    means = values.mean(axis=0)
    return values.T.dot(values) / float(values.shape[0]) - np.outer(means, means)


def shortest_path_pairs(adjacency: np.ndarray) -> Dict[int, Tuple[Tuple[int, int], ...]]:
    """Group unordered node pairs by unweighted shortest-path distance."""

    support = np.asarray(adjacency, dtype=int)
    if (
        support.ndim != 2
        or support.shape[0] != support.shape[1]
        or np.any(np.diag(support))
        or not np.array_equal(support, support.T)
    ):
        raise ValueError("shortest-path support must be a simple undirected graph")
    graph = nx.from_numpy_array((support > 0).astype(int))
    if not nx.is_connected(graph):
        raise ValueError("graph-distance correlation requires connected support")
    output: Dict[int, List[Tuple[int, int]]] = {}
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    for left in range(support.shape[0]):
        for right in range(left + 1, support.shape[0]):
            distance = int(distances[left][right])
            output.setdefault(distance, []).append((left, right))
    return {distance: tuple(pairs) for distance, pairs in sorted(output.items())}


def graph_distance_correlation(
    correlation: np.ndarray,
    adjacency: np.ndarray,
) -> List[Dict[str, float]]:
    """Average a connected correlation matrix over graph-distance shells."""

    matrix = np.asarray(correlation, dtype=float)
    support = np.asarray(adjacency, dtype=int)
    if matrix.shape != support.shape or np.any(~np.isfinite(matrix)):
        raise ValueError("correlation matrix and graph do not align")
    rows: List[Dict[str, float]] = []
    for distance, pairs in shortest_path_pairs(support).items():
        values = np.asarray([matrix[left, right] for left, right in pairs], dtype=float)
        rows.append(
            {
                "graph_distance": float(distance),
                "connected_correlation": float(values.mean()),
                "pair_count": float(values.size),
                "within_pair_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return rows


def normalized_autocorrelation(values: Sequence[float], maximum_lag: int) -> np.ndarray:
    r"""Estimate normalized autocorrelation through a fixed lag truncation.

    The denominator is the full-window second central moment.  Lagged
    numerators use all available pairs at that lag.  A zero-variance series is
    returned as all-NaN rather than being assigned artificial persistence.
    """

    series = np.asarray(values, dtype=float)
    lag_maximum = int(maximum_lag)
    if series.ndim != 1 or series.size < 3 or np.any(~np.isfinite(series)):
        raise ValueError("autocorrelation requires a finite one-dimensional series")
    if lag_maximum < 0 or lag_maximum >= series.size:
        raise ValueError("maximum lag must lie between zero and series length minus one")
    centered = series - series.mean()
    denominator = float(np.mean(centered * centered))
    if denominator <= 1.0e-15:
        return np.full(lag_maximum + 1, np.nan, dtype=float)
    output = np.empty(lag_maximum + 1, dtype=float)
    output[0] = 1.0
    for lag in range(1, lag_maximum + 1):
        output[lag] = float(np.mean(centered[:-lag] * centered[lag:]) / denominator)
    return output


def truncated_integrated_autocorrelation_time(
    values: Sequence[float], maximum_lag: int
) -> float:
    r"""Return ``1/2 + sum_{lag=1}^{maximum_lag} rho(lag)``.

    This deliberately retains negative finite-window estimates.  It is a
    descriptive truncated correlation sum, not an automatic effective sample
    size or a claim of critical slowing down.
    """

    correlation = normalized_autocorrelation(values, int(maximum_lag))
    if np.any(~np.isfinite(correlation)):
        return float("nan")
    return float(0.5 + correlation[1:].sum())


def binder_cumulant(
    magnetization: Sequence[float], denominator_epsilon: float = 1.0e-12
) -> float:
    r"""Return ``1 - <m^4> / (3 <m^2>^2)`` for one trajectory phase."""

    values = np.asarray(magnetization, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("Binder cumulant requires a finite one-dimensional series")
    second = float(np.mean(values ** 2))
    if second <= float(denominator_epsilon):
        return float("nan")
    return float(1.0 - float(np.mean(values ** 4)) / (3.0 * second ** 2))


def phase_collective_observables(
    frame: pd.DataFrame,
    adjacency: np.ndarray,
    n_agents: int,
    primary_lag_sweeps: int,
    sensitivity_lag_sweeps: Sequence[int],
    binder_epsilon: float = 1.0e-12,
) -> Dict[str, List[Dict[str, object]]]:
    """Calculate all extension observables for one complete panel."""

    required = {"phase", "beliefs", "belief_magnetization"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("trajectory lacks extension columns: %s" % sorted(missing))
    if int(n_agents) < 2:
        raise ValueError("at least two agents are required")
    lag_sweeps = sorted(
        {int(primary_lag_sweeps), *[int(value) for value in sensitivity_lag_sweeps]}
    )
    if min(lag_sweeps) < 1:
        raise ValueError("lag truncations must be positive sweep counts")

    profiles: List[Dict[str, object]] = []
    matrices: List[Dict[str, object]] = []
    autocorrelations: List[Dict[str, object]] = []
    persistence: List[Dict[str, object]] = []
    binders: List[Dict[str, object]] = []
    binder_sensitivity: List[Dict[str, object]] = []
    distributions: List[Dict[str, object]] = []
    for phase in PHASES:
        selected = frame[frame["phase"] == phase]
        if selected.empty:
            raise ValueError("trajectory is missing phase %s" % phase)
        configurations = spin_matrix(selected["beliefs"])
        if configurations.shape[1] != int(n_agents):
            raise ValueError("belief vector width does not equal n_agents")
        correlation = connected_correlation_matrix(configurations)
        for row in graph_distance_correlation(correlation, adjacency):
            profiles.append({"phase": phase, **row})
        for left in range(int(n_agents)):
            for right in range(int(n_agents)):
                matrices.append(
                    {
                        "phase": phase,
                        "agent_i": left,
                        "agent_j": right,
                        "community_i": 0 if left < int(n_agents) // 2 else 1,
                        "community_j": 0 if right < int(n_agents) // 2 else 1,
                        "connected_correlation": float(correlation[left, right]),
                    }
                )

        magnetization = selected["belief_magnetization"].to_numpy(float)
        primary_lag = int(primary_lag_sweeps) * int(n_agents)
        if primary_lag >= magnetization.size:
            raise ValueError("primary autocorrelation lag exceeds phase length")
        rho = normalized_autocorrelation(magnetization, primary_lag)
        for lag, value in enumerate(rho):
            autocorrelations.append(
                {
                    "phase": phase,
                    "lag_updates": lag,
                    "lag_sweeps": float(lag / float(n_agents)),
                    "autocorrelation": float(value),
                }
            )
        for lag_window in lag_sweeps:
            maximum_lag = lag_window * int(n_agents)
            estimable = maximum_lag < magnetization.size
            persistence.append(
                {
                    "phase": phase,
                    "lag_truncation_sweeps": lag_window,
                    "lag_truncation_updates": maximum_lag,
                    "is_primary": lag_window == int(primary_lag_sweeps),
                    "lag_estimable": bool(estimable),
                    "integrated_autocorrelation_time_updates": float(
                        truncated_integrated_autocorrelation_time(magnetization, maximum_lag)
                    )
                    if estimable
                    else float("nan"),
                }
            )

        midpoint = int(len(magnetization) // 2)
        binder_windows = {
            "full_phase": magnetization,
            "early_half": magnetization[:midpoint],
            "late_half": magnetization[midpoint:],
        }
        phase_binder_rows: List[Dict[str, object]] = []
        for window_name, window_values in binder_windows.items():
            second = float(np.mean(window_values ** 2))
            fourth = float(np.mean(window_values ** 4))
            value = binder_cumulant(window_values, binder_epsilon)
            row = {
                "phase": phase,
                "temporal_window": window_name,
                "binder_cumulant": value,
                "magnetization_second_moment": second,
                "magnetization_fourth_moment": fourth,
                "near_zero_denominator": bool(second <= float(binder_epsilon)),
                "window_updates": int(len(window_values)),
            }
            phase_binder_rows.append(row)
            binder_sensitivity.append(row)

        full_window = phase_binder_rows[0]
        second_moment = float(full_window["magnetization_second_moment"])
        binder = float(full_window["binder_cumulant"])
        binders.append(
            {
                "phase": phase,
                "binder_cumulant": binder,
                "magnetization_second_moment": second_moment,
                "near_zero_denominator": bool(second_moment <= float(binder_epsilon)),
                "phase_updates": int(len(magnetization)),
            }
        )
        observed_support, observed_counts = np.unique(magnetization, return_counts=True)
        count_by_value = {
            round(float(value), 12): int(count)
            for value, count in zip(observed_support, observed_counts)
        }
        complete_support = np.linspace(-1.0, 1.0, int(n_agents) + 1)
        for value in complete_support:
            count = count_by_value.get(round(float(value), 12), 0)
            distributions.append(
                {
                    "phase": phase,
                    "belief_magnetization": float(value),
                    "count": int(count),
                    "probability": float(count / len(magnetization)),
                }
            )
    return {
        "correlation_profiles": profiles,
        "correlation_matrices": matrices,
        "autocorrelation_curves": autocorrelations,
        "integrated_autocorrelation": persistence,
        "binder_cumulants": binders,
        "binder_cumulant_sensitivity": binder_sensitivity,
        "magnetization_distributions": distributions,
    }


__all__ = [
    "PHASES",
    "binder_cumulant",
    "connected_correlation_matrix",
    "graph_distance_correlation",
    "normalized_autocorrelation",
    "parse_spin_vector",
    "phase_collective_observables",
    "shortest_path_pairs",
    "spin_matrix",
    "truncated_integrated_autocorrelation_time",
]
