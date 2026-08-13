import numpy as np
import pytest

from thermoagent.consensus import (
    consensus_rmse,
    gossip_distributions,
    gossip_distributions_with_trace,
    local_consensus_residuals,
    metropolis_matrix,
    one_hot_sketch,
)
from thermoagent.mechanics import (
    MacrostateCalibration,
    free_energy_gap,
    interaction_entropy,
    normalized_entropy,
    occupancy_distribution,
    role_conditioned_distribution,
)


def test_probability_entropy_and_free_energy_identities():
    p = occupancy_distribution([0, 0, 4, 26], alpha=0.1)
    assert np.isclose(p.sum(), 1.0)
    assert 0.0 <= normalized_entropy(p) <= 1.0
    calibration = MacrostateCalibration(np.array([[0.2, 0.6]] * 3))
    q = calibration.healthy_reference()
    assert np.isclose(q.sum(), 1.0)
    assert free_energy_gap(q, q, calibration.temperature) < 1e-12
    assert free_energy_gap(p, q, calibration.temperature) >= 0.0


def test_calibration_uses_nominal_features_and_encodes_bounds():
    nominal = np.array([
        [0.05, 0.05, 0.05], [0.10, 0.10, 0.10], [0.15, 0.15, 0.15],
        [0.20, 0.20, 0.20], [0.25, 0.25, 0.25], [0.30, 0.30, 0.30],
    ])
    calibration = MacrostateCalibration.fit(nominal)
    assert calibration.encode([0, 0, 0]) == 0
    assert calibration.encode([1, 1, 1]) == 26
    assert calibration.thresholds.shape == (3, 2)


def test_role_conditioned_distribution_is_normalized():
    p = role_conditioned_distribution({"source": [0, 1], "demand": [20, 26]}, shrinkage=0.6)
    assert p.shape == (27,)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p > 0)


def test_role_specific_healthy_reference_is_normalized_and_distinct():
    source = occupancy_distribution([0, 0, 1], alpha=0.1)
    demand = occupancy_distribution([25, 26, 26], alpha=0.1)
    calibration = MacrostateCalibration(
        np.array([[0.2, 0.6]] * 3),
        role_references={"source": source.tolist(), "demand": demand.tolist()},
    )
    assert np.isclose(calibration.role_reference("source").sum(), 1.0)
    assert not np.allclose(
        calibration.role_reference("source"),
        calibration.role_reference("demand"),
    )
    assert np.allclose(
        calibration.role_reference("unknown"),
        calibration.healthy_reference(),
    )


def test_interaction_entropy_distinguishes_concentration():
    concentrated = interaction_entropy({("a", "b"): 10.0, ("a", "c"): 0.1})
    broad = interaction_entropy({("a", "b"): 5.0, ("a", "c"): 5.0})
    assert concentrated < broad
    assert np.isclose(broad, 1.0)


def test_metropolis_consensus_converges_when_connected():
    ids = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    matrix = metropolis_matrix(ids, edges)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    sketches = {agent_id: one_hot_sketch(index) for index, agent_id in enumerate(ids)}
    exact = np.mean(list(sketches.values()), axis=0)
    initial = consensus_rmse(sketches, exact)
    estimates = gossip_distributions(sketches, [edges] * 80)
    final = consensus_rmse(estimates, exact)
    assert final < initial * 0.01


def test_population_scaled_smoothing_matches_global_occupancy():
    states = [0, 0, 4, 26]
    sketches = [one_hot_sketch(state, alpha=0.1, population_size=len(states)) for state in states]
    assert np.allclose(np.mean(sketches, axis=0), occupancy_distribution(states, alpha=0.1))


def test_partition_prevents_global_consensus():
    sketches = {"a": one_hot_sketch(0), "b": one_hot_sketch(0), "c": one_hot_sketch(26), "d": one_hot_sketch(26)}
    exact = np.mean(list(sketches.values()), axis=0)
    partition = [("a", "b"), ("c", "d")]
    estimates = gossip_distributions(sketches, [partition] * 20)
    assert consensus_rmse(estimates, exact) > 0.01


def test_gossip_trace_and_local_residual_do_not_require_global_occupancy():
    sketches = {
        "a": one_hot_sketch(0),
        "b": one_hot_sketch(0),
        "c": one_hot_sketch(26),
    }
    edges = [("a", "b")]
    estimates, trace = gossip_distributions_with_trace(sketches, [edges] * 3)
    assert len(trace) == 3
    residuals = local_consensus_residuals(estimates, edges)
    assert residuals["a"] == pytest.approx(0.0)
    assert residuals["b"] == pytest.approx(0.0)
    assert residuals["c"] == 1.0
