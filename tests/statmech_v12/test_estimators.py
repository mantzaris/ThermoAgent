import numpy as np
import pytest

from thermoagent.statmech_llm_v12.estimators import (
    block_time_reversal_kl,
    conditional_mutual_information_history,
    exact_cycle_entropy_production,
    gini_simpson,
    information_permutation_floor,
    jensen_shannon_divergence,
    known_three_state_cycle,
    markov_entropy_production,
    miller_madow_entropy,
    occupied_transition_counts,
    probability_currents,
    row_stochastic,
    shannon_entropy,
    stationary_chain_sample,
    stationary_distribution,
    susceptibility,
    time_shuffle_floor,
    transition_pair_irreversibility,
    tsallis_entropy,
)


def test_reciprocal_kernel_has_zero_current_and_entropy_production():
    kernel = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    stationary = stationary_distribution(kernel)
    assert np.max(np.abs(probability_currents(stationary, kernel))) < 1e-12
    assert markov_entropy_production(stationary, kernel) < 1e-12


def test_known_cycle_exact_entropy_production_and_empirical_recovery():
    kernel = known_three_state_cycle(0.55, 0.15, 0.30)
    exact = exact_cycle_entropy_production(0.55, 0.15)
    assert markov_entropy_production(stationary_distribution(kernel), kernel) == pytest.approx(exact)
    path = stationary_chain_sample(kernel, 100000, 12)
    counts, _ = occupied_transition_counts(path)
    empirical = transition_pair_irreversibility(counts, 0.5)
    assert empirical == pytest.approx(exact, abs=0.02)


def test_pathwise_estimator_and_null_floor_are_finite():
    path = stationary_chain_sample(known_three_state_cycle(0.55, 0.15, 0.30), 5000, 7)
    observed = block_time_reversal_kl(path, 3, 0.5)
    floor = time_shuffle_floor(path, 3, 0.5, 20, 9)
    assert observed > floor["mean"]
    assert floor["ci_high"] >= floor["ci_low"] >= 0.0


def test_entropy_identities_and_susceptibility():
    p = np.asarray([0.1, 0.2, 0.7])
    assert tsallis_entropy(p, 1.0) == pytest.approx(shannon_entropy(p))
    assert tsallis_entropy(p, 2.0) == pytest.approx(gini_simpson(p))
    assert susceptibility([-1.0, 1.0, -1.0, 1.0], 4) > 0.0
    assert miller_madow_entropy([10, 10]) > shannon_entropy([10, 10])
    assert jensen_shannon_divergence([0.8, 0.2], [0.8, 0.2]) == pytest.approx(0.0)


def test_information_permutation_floor_is_seeded_and_finite():
    x = np.asarray([-1, 1] * 20)
    y = np.roll(x, 1)
    first = information_permutation_floor(x, y, 0.5, 20, 17)
    second = information_permutation_floor(x, y, 0.5, 20, 17)
    assert first == second
    assert first["mutual_information_mean"] >= 0.0
    assert first["transfer_entropy_mean"] >= 0.0


def test_transition_normalization_and_history_metric():
    counts = np.asarray([[5.0, 1.0], [2.0, 4.0]])
    kernel = row_stochastic(counts, 0.5)
    assert np.allclose(kernel.sum(axis=1), 1.0)
    value = conditional_mutual_information_history([0, 1, 0, 1, 1, 0, 1, 0], 1, 0.1)
    assert np.isfinite(value) and value >= 0.0
