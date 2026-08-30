import numpy as np

from thermoagent.statmech_llm.discovery.estimators import (
    exact_cycle_entropy_production,
    known_three_state_cycle,
    markov_entropy_production,
    stationary_distribution,
)
from thermoagent.statmech_llm.replication.observables import (
    conditional_entropy_rate,
    instantaneous_state,
    macrostate_code,
    mahalanobis_distance,
    plugin_entropy,
    reference_energy,
    regularized_mahalanobis_fit,
    total_correlation,
)
from thermoagent.statmech_llm.replication.simulation import build_reciprocal_graph


def test_entropy_rate_and_plugin_entropy_limits():
    assert plugin_entropy([0, 0, 0]) == 0.0
    assert np.isclose(plugin_entropy([0, 1]), np.log(2.0))
    assert conditional_entropy_rate([0, 0, 0, 0], 1, 0.0) == 0.0
    alternating = conditional_entropy_rate([0, 1, 0, 1, 0, 1], 1, 0.0)
    assert np.isclose(alternating, 0.0)


def test_total_correlation_distinguishes_independent_and_locked_variables():
    independent = np.asarray([[a, b] for a in (-1, 1) for b in (-1, 1)] * 20)
    locked = np.asarray([[-1, -1], [1, 1]] * 40)
    assert abs(total_correlation(independent)) < 1e-12
    assert np.isclose(total_correlation(locked), np.log(2.0))


def test_reference_energy_matches_direct_definition():
    graph = build_reciprocal_graph(8, "modular", 51)
    b = np.asarray([1, -1, 1, -1, 1, -1, 1, -1])
    a = -b
    h = b.copy()
    expected = -0.5 * b.dot(graph.symmetric).dot(b) - 0.5 * 0.65 * a.dot(graph.symmetric).dot(a) - 0.8 * a.dot(b) - h.dot(b)
    assert np.isclose(reference_energy(b, a, graph.symmetric, h), expected)
    state = instantaneous_state(b, a, graph.adjacency, graph.symmetric, h)
    assert np.isclose(state["reference_energy"], expected)
    assert -1.0 <= state["spatial_belief_correlation"] <= 1.0


def test_macrostate_code_uses_frozen_widths():
    row = {
        "belief_magnetization": 0.49,
        "action_magnetization": -0.26,
        "belief_action_overlap": 0.74,
        "reference_energy_per_agent": -1.26,
        "belief_disagreement": 0.24,
    }
    widths = {"magnetization_width": 0.25, "overlap_width": 0.25, "energy_per_agent_width": 0.25, "disagreement_width": 0.125}
    assert macrostate_code(row, widths) == (2, -1, 3, -5, 2)


def test_regularized_nominal_distance_is_finite_and_centered():
    values = np.asarray([[0.0, 0.0], [1.0, 1.1], [-1.0, -0.9], [0.5, 0.4]])
    center, precision = regularized_mahalanobis_fit(values, 0.1)
    distances = mahalanobis_distance(np.vstack([center, [5.0, 5.0]]), center, precision)
    assert distances[0] < 1e-10
    assert distances[1] > 1.0


def test_known_cycle_entropy_production_convention():
    kernel = known_three_state_cycle(0.55, 0.25, 0.20)
    observed = markov_entropy_production(stationary_distribution(kernel), kernel)
    assert np.isclose(observed, exact_cycle_entropy_production(0.55, 0.25))
    reciprocal = known_three_state_cycle(0.4, 0.4, 0.2)
    assert markov_entropy_production(stationary_distribution(reciprocal), reciprocal) < 1e-14
