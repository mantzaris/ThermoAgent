import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v15.analysis import (
    _pooled_binder_bootstrap_summary,
)
from thermoagent.statmech_llm_v15.collective_observables import (
    binder_cumulant,
    connected_correlation_matrix,
    graph_distance_correlation,
    normalized_autocorrelation,
    phase_collective_observables,
    shortest_path_pairs,
    truncated_integrated_autocorrelation_time,
)


def test_connected_correlation_and_graph_distance_are_exact_on_binary_fixture():
    configurations = np.asarray(
        [
            [1, 1, -1, -1],
            [-1, -1, 1, 1],
            [1, 1, -1, -1],
            [-1, -1, 1, 1],
        ],
        dtype=int,
    )
    correlation = connected_correlation_matrix(configurations)
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.isclose(correlation[0, 1], 1.0)
    assert np.isclose(correlation[0, 2], -1.0)
    ring = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=int,
    )
    assert {distance: len(pairs) for distance, pairs in shortest_path_pairs(ring).items()} == {
        1: 4,
        2: 2,
    }
    profile = pd.DataFrame(graph_distance_correlation(correlation, ring)).set_index(
        "graph_distance"
    )
    assert np.isclose(profile.loc[1.0, "connected_correlation"], 0.0)
    assert np.isclose(profile.loc[2.0, "connected_correlation"], -1.0)


def test_autocorrelation_and_truncated_sum_match_alternating_series():
    series = np.asarray([-1.0, 1.0] * 6)
    correlation = normalized_autocorrelation(series, 4)
    assert np.allclose(correlation, [1.0, -1.0, 1.0, -1.0, 1.0])
    assert np.isclose(truncated_integrated_autocorrelation_time(series, 2), 0.5)
    constant = normalized_autocorrelation(np.ones(8), 2)
    assert np.all(np.isnan(constant))


def test_binder_cumulant_has_known_two_state_limit_and_explicit_zero_case():
    ordered_two_state = np.asarray([-1.0, 1.0] * 10)
    assert np.isclose(binder_cumulant(ordered_two_state), 2.0 / 3.0)
    assert np.isnan(binder_cumulant(np.zeros(8)))


def test_pooled_binder_sensitivity_resamples_complete_cluster_moments():
    summary = _pooled_binder_bootstrap_summary(
        second_moments=[1.0, 1.0, 1.0],
        fourth_moments=[1.0, 1.0, 1.0],
        weights=[120.0, 120.0, 120.0],
        denominator_epsilon=1.0e-12,
        seed=1517,
        replicates=100,
    )
    assert np.isclose(summary["estimate"], 2.0 / 3.0)
    assert np.isclose(summary["ci_low"], 2.0 / 3.0)
    assert np.isclose(summary["ci_high"], 2.0 / 3.0)
    assert summary["independent_clusters"] == 3
    undefined = _pooled_binder_bootstrap_summary(
        second_moments=[0.0, 0.0],
        fourth_moments=[0.0, 0.0],
        weights=[10.0, 10.0],
        denominator_epsilon=1.0e-12,
        seed=1517,
        replicates=20,
    )
    assert np.isnan(undefined["estimate"])
    assert undefined["valid_bootstrap_replicates"] == 0


def test_phase_extension_calculates_each_trajectory_before_pooling():
    rows = []
    phases = ("baseline", "disruption", "recovery")
    state_cycle = ("1;1;-1;-1", "-1;-1;1;1")
    for phase in phases:
        for update in range(12):
            state = state_cycle[update % 2]
            vector = np.asarray([int(item) for item in state.split(";")])
            rows.append(
                {
                    "phase": phase,
                    "beliefs": state,
                    "belief_magnetization": float(vector.mean()),
                }
            )
    frame = pd.DataFrame(rows)
    ring = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=int,
    )
    output = phase_collective_observables(frame, ring, 4, 1, [2])
    assert len(output["correlation_matrices"]) == 3 * 4 * 4
    assert len(output["correlation_profiles"]) == 3 * 2
    assert len(output["integrated_autocorrelation"]) == 3 * 2
    assert len(output["binder_cumulants"]) == 3
    assert len(output["binder_cumulant_sensitivity"]) == 3 * 3
    assert len(output["magnetization_distributions"]) == 3 * 5
    assert all(
        np.isclose(
            sum(
                row["probability"]
                for row in output["magnetization_distributions"]
                if row["phase"] == phase
            ),
            1.0,
        )
        for phase in phases
    )
    assert {row["phase"] for row in output["binder_cumulants"]} == set(phases)
    assert {
        row["temporal_window"]
        for row in output["binder_cumulant_sensitivity"]
    } == {"full_phase", "early_half", "late_half"}
