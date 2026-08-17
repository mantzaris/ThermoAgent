import numpy as np

from thermoagent.statmech.exact import entropy_production_rate, exact_transition_matrix, stationary_distribution
from thermoagent.statmech.model import ModelParameters, MultiplexModel, topology_adjacency
from thermoagent.statmech_llm.theory import directed_family
from thermoagent.statmech_llm.trajectories import simulate_stationary_pathwise


def test_pathwise_estimator_matches_exact_small_system_with_uncertainty():
    base = topology_adjacency(3, "ring", 10)
    family = directed_family(base, 91)
    communication = family.at(0.5)
    parameters = ModelParameters(0.70, 0.30, 0.40, 1.5)
    model = MultiplexModel(communication, base, parameters)
    kernel = exact_transition_matrix(model)
    exact = entropy_production_rate(stationary_distribution(kernel), kernel)
    metric, blocks = simulate_stationary_pathwise(
        communication,
        base,
        parameters,
        8301,
        burn_in_sweeps=3000,
        sample_sweeps=30000,
        block_sweeps=300,
    )
    standard_error = blocks.std(ddof=1) / np.sqrt(blocks.size)
    assert abs(metric["pathwise_irreversibility_per_update"] - exact) < max(5e-4, 4 * standard_error)


def test_reciprocal_pathwise_null_is_close_to_zero():
    base = topology_adjacency(8, "ring", 11)
    metric, _ = simulate_stationary_pathwise(
        base,
        base,
        ModelParameters(0.45, 0.35, 0.40, 1.7),
        8302,
        burn_in_sweeps=400,
        sample_sweeps=4000,
    )
    assert abs(metric["pathwise_irreversibility_per_update"]) < 0.003


def test_pathwise_units_and_determinism():
    base = topology_adjacency(10, "ring", 12)
    family = directed_family(base, 92)
    arguments = (family.at(0.35), base, ModelParameters(0.55, 0.30, 0.40, 1.8), 8303, 30, 80)
    first, first_blocks = simulate_stationary_pathwise(*arguments)
    second, second_blocks = simulate_stationary_pathwise(*arguments)
    assert first == second
    assert np.array_equal(first_blocks, second_blocks)
    assert np.isclose(first["pathwise_irreversibility_per_agent_sweep"], 2 * first["pathwise_irreversibility_per_update"])
    assert np.isclose(first["pathwise_irreversibility_per_sweep"], 20 * first["pathwise_irreversibility_per_update"])
