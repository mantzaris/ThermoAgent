import numpy as np

from thermoagent.statmech_llm.corrected_quench.observables import (
    belief_action_lag,
    binary_entropy,
    binder_cumulant,
    conditional_memory_depths,
    dependence_bias_audit,
    discrete_mutual_information,
    mean_reported_uncertainty,
    pairwise_information_summary,
    phase_path_length,
    recovery_time,
    signed_polygon_area,
    standardized_nominal_distance,
    standardized_nominal_fit,
)


def test_entropy_and_uncertainty_bounds():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert np.isclose(binary_entropy(0.5), np.log(2.0))
    assert 0.0 <= mean_reported_uncertainty(np.asarray([0.1, 0.5, 0.9])) <= np.log(2.0)


def test_mutual_information_detects_locked_but_not_factorial_signals():
    locked = np.asarray([-1, 1] * 40)
    independent_left = np.asarray([-1, -1, 1, 1] * 20)
    independent_right = np.asarray([-1, 1, -1, 1] * 20)
    assert np.isclose(discrete_mutual_information(locked, locked), np.log(2.0))
    assert abs(discrete_mutual_information(independent_left, independent_right)) < 1e-12
    matrix = np.column_stack([locked, locked, -locked])
    graph = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = pairwise_information_summary(matrix, graph)
    assert result["mean_edge_belief_mutual_information"] > 0.6


def test_binder_lag_path_and_recovery_observables():
    ordered = np.asarray([-1.0] * 20 + [1.0] * 20)
    assert np.isclose(binder_cumulant(ordered), 2.0 / 3.0)
    beliefs = np.asarray([[-1, 1], [1, -1], [-1, 1]])
    actions = np.vstack([[1, 1], beliefs[:-1]])
    assert np.isclose(belief_action_lag(beliefs, actions, 1), 1.0)
    assert recovery_time([4.0, 1.0, 0.5, 0.4], 1.0, 2) == 2.0
    assert np.isclose(phase_path_length(np.asarray([[0.0, 0.0], [3.0, 4.0]])), 5.0)
    assert signed_polygon_area([0, 1, 1, 0], [0, 0, 1, 1]) > 0.0


def test_nominal_covariance_estimators_are_finite_and_training_centered():
    rng = np.random.default_rng(1426)
    training = rng.normal(size=(80, 5))
    for estimator in ("euclidean", "diagonal", "shrinkage", "robust"):
        fitted = standardized_nominal_fit(training, estimator, 0.1)
        center_distance = standardized_nominal_distance(training.mean(axis=0), fitted)[0]
        outlier_distance = standardized_nominal_distance(np.full(5, 8.0), fitted)[0]
        assert center_distance < 1e-10
        assert np.isfinite(outlier_distance) and outlier_distance > 1.0


def test_conditional_memory_depth_output_is_defined():
    states = np.asarray([0, 1, 0, 1] * 20)
    values = conditional_memory_depths(states, [1, 2, 3])
    assert set(values) == {1, 2, 3}
    assert all(np.isfinite(value) for value in values.values())


def test_circular_shift_adjusted_dependence_is_near_zero_for_independent_processes():
    rng = np.random.default_rng(1493)
    independent = rng.choice((-1, 1), size=(4000, 6))
    adjacency = np.ones((3, 3), dtype=int) - np.eye(3, dtype=int)
    result = dependence_bias_audit(independent, adjacency, 200, 1494)
    assert abs(result["total_correlation_bias_adjusted"]) < 0.03
    assert abs(result["pairwise_mutual_information_bias_adjusted"]) < 0.01


def test_circular_shift_adjusted_dependence_is_positive_for_locked_processes():
    rng = np.random.default_rng(1495)
    latent = rng.choice((-1, 1), size=2000)
    dependent = np.column_stack([latent, latent, -latent, latent, -latent, latent])
    adjacency = np.ones((3, 3), dtype=int) - np.eye(3, dtype=int)
    result = dependence_bias_audit(dependent, adjacency, 100, 1496)
    assert result["total_correlation_bias_adjusted"] > 2.0
    assert result["pairwise_mutual_information_bias_adjusted"] > 0.5


def test_adjusted_information_is_not_truncated():
    rng = np.random.default_rng(2)
    values = rng.choice((-1, 1), size=(12, 6))
    adjacency = np.ones((3, 3), dtype=int) - np.eye(3, dtype=int)
    result = dependence_bias_audit(values, adjacency, 100, 42)
    assert result["total_correlation_bias_adjusted"] < 0.0
    assert np.isclose(
        result["total_correlation_bias_adjusted"],
        result["total_correlation_raw"] - result["total_correlation_null_mean"],
    )
