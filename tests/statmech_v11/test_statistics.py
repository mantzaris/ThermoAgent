import numpy as np

from thermoagent.statmech_llm_v11.statistics import (
    block_time_reversal_kl,
    calibration_summary,
    entropy_production_per_update,
    normalize_transition_counts,
    paired_cluster_bootstrap,
    stationary_distribution,
    conditional_mutual_information_history,
)


def test_reversible_kernel_has_zero_entropy_production():
    kernel = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    stationary = stationary_distribution(kernel)
    assert entropy_production_per_update(stationary, kernel) < 1e-12


def test_directed_cycle_has_positive_entropy_production():
    kernel = np.asarray([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    assert entropy_production_per_update(stationary_distribution(kernel), kernel) > 0.5


def test_transition_kernel_rows_normalize():
    kernel = normalize_transition_counts(np.asarray([[2, 1], [1, 4]], dtype=float), 0.5)
    assert np.allclose(kernel.sum(axis=1), 1.0)


def test_paired_bootstrap_uses_cluster_not_row_count():
    summary = paired_cluster_bootstrap({"a": [1.0] * 100, "b": [3.0]}, 1000, 8)
    assert summary["estimate"] == 2.0
    assert summary["independent_clusters"] == 2.0


def test_calibration_and_block_reversal_are_finite():
    result = calibration_summary([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], 4)
    assert result["brier"] < 0.05
    assert np.isfinite(block_time_reversal_kl([0, 1, 2, 0, 1, 2, 0], 3))


def test_history_diagnostic_detects_second_order_memory():
    rng = np.random.default_rng(39)
    states = [0, 1]
    for _ in range(4000):
        states.append(states[-2] if rng.random() < 0.95 else 1 - states[-2])
    assert conditional_mutual_information_history(states, 1) > 0.2


def test_block_reversal_estimator_separates_known_directed_chain_from_reversible_null():
    rng = np.random.default_rng(912)
    reversible = np.asarray([[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]])
    directed = np.asarray([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])

    def sample(kernel):
        state = 0
        path = []
        for _ in range(40000):
            path.append(state)
            state = int(rng.choice(3, p=kernel[state]))
        return path

    reversible_kl = block_time_reversal_kl(sample(reversible), 3, 0.5)
    directed_kl = block_time_reversal_kl(sample(directed), 3, 0.5)
    assert reversible_kl < 0.002
    assert directed_kl > reversible_kl + 0.2


def test_quadratic_model_comparison_recovers_declared_shape():
    import pandas as pd

    from thermoagent.statmech_llm_v11.formal import _curve_model_comparison

    rows = []
    for cluster in range(8):
        for alpha in (0.0, 0.15, 0.40):
            rows.append(
                {
                    "matched_cluster": "c%d" % cluster,
                    "alpha": alpha,
                    "adjusted_block_kl": 0.02 * cluster + (0.4 + 0.01 * cluster) * alpha ** 2,
                }
            )
    results = {row["model"]: row for row in _curve_model_comparison(pd.DataFrame(rows))}
    assert results["quadratic"]["leave_cluster_out_rmse"] < results["linear"]["leave_cluster_out_rmse"]
    assert results["quadratic"]["quadratic_coefficient"] > 0.4
