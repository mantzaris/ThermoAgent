import numpy as np

from thermoagent.statmech.exact import entropy_production_rate, stationary_distribution
from thermoagent.statmech_llm.estimators import (
    block_time_reversal_kl,
    conditional_mutual_information_markov,
    exact_cycle_entropy_production,
    fit_logistic_response,
    known_three_state_cycle,
    shuffled_bias_floor,
    stationary_chain_sample,
    transition_counts,
    transition_pair_irreversibility,
)


def test_known_cycle_formula_matches_schnakenberg_exact_value():
    kernel = known_three_state_cycle(0.55, 0.25, 0.20)
    stationary = stationary_distribution(kernel)
    exact = entropy_production_rate(stationary, kernel)
    assert np.isclose(exact, exact_cycle_entropy_production(0.55, 0.25), atol=1e-14)


def test_empirical_transition_estimator_converges_on_known_cycle():
    kernel = known_three_state_cycle(0.55, 0.25, 0.20)
    trajectory = stationary_chain_sample(kernel, 300000, 8201)
    observed = transition_pair_irreversibility(transition_counts(trajectory, 3), 0.5)
    exact = exact_cycle_entropy_production(0.55, 0.25)
    assert abs(observed.estimate_per_transition - exact) < 0.012


def test_reversible_chain_and_shuffled_null_have_small_estimates():
    kernel = known_three_state_cycle(0.4, 0.4, 0.2)
    trajectory = stationary_chain_sample(kernel, 80000, 8202)
    observed = transition_pair_irreversibility(transition_counts(trajectory, 3), 0.5)
    floor = shuffled_bias_floor(trajectory, 3, 0.5, 30, 8203)
    assert observed.estimate_per_transition < 5e-4
    assert floor["quantile_95"] < 8e-4


def test_block_time_reversal_detects_cycle_direction():
    kernel = known_three_state_cycle(0.62, 0.18, 0.20)
    trajectory = stationary_chain_sample(kernel, 100000, 8204)
    assert block_time_reversal_kl(trajectory, 3) > 0.1


def test_conditional_mutual_information_flags_hidden_second_order_dependence():
    rng = np.random.default_rng(8205)
    first_order = rng.integers(0, 2, 30000)
    second_order = np.zeros(30000, dtype=int)
    second_order[:2] = [0, 1]
    for index in range(2, second_order.size):
        second_order[index] = second_order[index - 2] if rng.random() < 0.95 else 1 - second_order[index - 2]
    assert conditional_mutual_information_markov(second_order, 0.1) > conditional_mutual_information_markov(first_order, 0.1) + 0.2


def test_logistic_fit_recovers_effective_temperature_and_option_bias():
    rng = np.random.default_rng(8206)
    fields = np.tile(np.linspace(-1.5, 1.5, 13), 800)
    order = rng.choice([-1, 1], fields.size)
    logits = 0.25 + 1.6 * fields + 0.35 * order
    choices = np.where(rng.random(fields.size) < 1.0 / (1.0 + np.exp(-logits)), 1, -1)
    fit = fit_logistic_response(fields, choices, option_order=order)
    assert fit["converged"] == 1.0
    assert abs(fit["field_slope"] - 1.6) < 0.08
    assert abs(fit["option_order_slope"] - 0.35) < 0.08
    assert abs(fit["effective_decision_temperature"] - 1.25) < 0.08

