import numpy as np

from thermoagent.statmech.exact import (
    decode_state,
    detailed_balance_residual,
    distribution_distances,
    empirical_distribution,
    encode_state,
    entropy_production_rate,
    exact_transition_matrix,
    gibbs_distribution,
    stationary_distribution,
    verify_free_energy_identity,
)
from thermoagent.statmech.model import (
    ModelParameters,
    MultiplexModel,
    mean_field_critical_temperature,
    topology_adjacency,
)


def small_model(temperature=1.7):
    adjacency = topology_adjacency(3, "ring", 7)
    parameters = ModelParameters(
        belief_coupling=0.4,
        action_coupling=0.3,
        belief_action_coupling=0.5,
        temperature=temperature,
    )
    return MultiplexModel(adjacency, adjacency, parameters)


def test_state_encoding_round_trip():
    for index in range(64):
        assert encode_state(decode_state(index, 3)) == index


def test_heat_bath_kernel_is_stochastic():
    kernel = exact_transition_matrix(small_model())
    assert np.all(kernel >= 0.0)
    assert np.allclose(kernel.sum(axis=1), 1.0, atol=1e-13)


def test_gibbs_measure_satisfies_detailed_balance():
    model = small_model()
    gibbs, _ = gibbs_distribution(model)
    kernel = exact_transition_matrix(model)
    residual = detailed_balance_residual(gibbs, kernel)
    assert residual["maximum"] < 1e-14
    assert residual["l1"] < 1e-12


def test_exact_stationary_distribution_is_gibbs():
    model = small_model()
    gibbs, _ = gibbs_distribution(model)
    stationary = stationary_distribution(exact_transition_matrix(model))
    assert distribution_distances(stationary, gibbs)["total_variation"] < 1e-12


def test_equilibrium_entropy_production_is_zero():
    model = small_model()
    gibbs, _ = gibbs_distribution(model)
    assert entropy_production_rate(gibbs, exact_transition_matrix(model)) < 1e-24


def test_empirical_chain_approaches_gibbs_distribution():
    model = small_model(temperature=2.2)
    gibbs, _ = gibbs_distribution(model)
    empirical = empirical_distribution(model, seed=31, burn_in_steps=20000, sample_steps=180000)
    distances = distribution_distances(empirical, gibbs)
    assert distances["total_variation"] < 0.035
    assert distances["kl_empirical_reference"] < 0.006


def test_free_energy_kl_identity_is_exact_numerically():
    model = small_model()
    equilibrium, energies = gibbs_distribution(model)
    trial = 0.75 * equilibrium + 0.25 / equilibrium.size
    identity = verify_free_energy_identity(trial, equilibrium, energies, model.parameters.temperature)
    assert identity["absolute_residual"] < 1e-12


def test_local_rate_ratio_matches_energy_change():
    model = small_model()
    state = decode_state(17, 3)
    result = model.update_one(
        state,
        np.random.default_rng(1),
        variable_index=2,
        uniform_draw=0.0 if state.beliefs[2] < 0 else 1.0,
    )
    assert np.isclose(
        result["log_rate_ratio"],
        -result["energy_change"] / model.parameters.temperature,
        atol=1e-12,
    )


def test_mean_field_boundary_decreases_with_communication_dilution():
    parameters = ModelParameters()
    full = mean_field_critical_temperature(parameters, 4.0, 4.0)
    diluted = mean_field_critical_temperature(parameters, 1.5, 4.0)
    assert diluted < full
    assert full > 0.0
