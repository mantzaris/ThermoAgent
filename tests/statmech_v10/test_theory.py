import numpy as np

from thermoagent.statmech.exact import exact_transition_matrix, stationary_distribution
from thermoagent.statmech.model import ModelParameters, MultiplexModel, topology_adjacency
from thermoagent.statmech_llm.theory import (
    directed_family,
    entropy_production_by_layer,
    exact_family_point,
    finite_difference_kernel_derivative,
    perturbative_entropy_production,
)


def reference_model(n_agents=3, temperature=1.35):
    adjacency = topology_adjacency(n_agents, "ring", 91)
    fields = np.linspace(0.15, -0.10, n_agents)
    return MultiplexModel(
        adjacency,
        adjacency,
        ModelParameters(0.70, 0.30, 0.40, temperature),
        private_fields=fields,
    )


def test_directed_family_preserves_support_pair_weight_and_reciprocal_limit():
    base = topology_adjacency(6, "ring", 1)
    family = directed_family(base, 12)
    assert np.array_equal(family.at(0.0), base)
    assert np.allclose(family.antisymmetric, -family.antisymmetric.T)
    for alpha in (0.1, 0.5, 0.8):
        directed = family.at(alpha)
        assert np.array_equal(directed > 0.0, base > 0.0)
        assert np.allclose(directed + directed.T, 2.0 * base)
        assert np.isclose(directed.sum(), base.sum())


def test_kernel_derivative_is_analytical_and_row_conserving():
    model = reference_model()
    family = directed_family(model.communication, 14)
    result = perturbative_entropy_production(model, family.antisymmetric)
    numerical = finite_difference_kernel_derivative(model, family.antisymmetric, 2e-6)
    assert np.max(np.abs(result.kernel_derivative - numerical)) < 2e-9
    assert np.max(np.abs(result.kernel_derivative.sum(axis=1))) < 1e-13


def test_stationary_first_order_equation_and_normalization_close():
    model = reference_model()
    family = directed_family(model.communication, 21)
    result = perturbative_entropy_production(model, family.antisymmetric)
    assert result.stationary_response_residual < 1e-12
    assert result.normalization_residual < 1e-12


def test_reciprocal_epr_is_numerically_zero():
    point = exact_family_point(
        reference_model(),
        directed_family(reference_model().communication, 31).antisymmetric,
        0.0,
    )
    assert point["total_per_update"] < 1e-24


def test_quadratic_coefficient_matches_small_alpha_exact_epr():
    model = reference_model()
    family = directed_family(model.communication, 14)
    perturbation = perturbative_entropy_production(model, family.antisymmetric)
    exact = exact_family_point(model, family.antisymmetric, 0.005)
    observed = exact["total_per_update"] / 0.005 ** 2
    assert perturbation.coefficient_per_update > 0.0
    assert abs(observed / perturbation.coefficient_per_update - 1.0) < 2e-4


def test_layer_decomposition_sums_to_total():
    model = reference_model()
    family = directed_family(model.communication, 18)
    directed_model = MultiplexModel(
        family.at(0.35),
        model.dependency,
        model.parameters,
        model.private_fields,
        model.task_fields,
    )
    kernel = exact_transition_matrix(directed_model)
    stationary = stationary_distribution(kernel)
    layers = entropy_production_by_layer(stationary, kernel, model.n_agents)
    assert layers["belief_per_update"] > 0.0
    assert layers["action_per_update"] >= 0.0
    assert np.isclose(
        layers["total_per_update"],
        layers["belief_per_update"] + layers["action_per_update"],
        atol=1e-13,
    )


def test_normalization_units_are_consistent():
    model = reference_model()
    family = directed_family(model.communication, 18)
    point = exact_family_point(model, family.antisymmetric, 0.2)
    assert np.isclose(point["total_per_agent_sweep"], 2.0 * point["total_per_update"])
    assert np.isclose(point["total_per_sweep"], 2 * model.n_agents * point["total_per_update"])

