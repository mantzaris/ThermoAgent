import numpy as np

from thermoagent.statmech_llm.llm_analysis import (
    bootstrap_empirical_kernel_epr,
    curve_model_comparison,
    dynamic_irreversibility_panels,
    heldout_paraphrase_logit,
)


def test_heldout_paraphrase_fit_never_trains_on_the_scored_wording():
    rng = np.random.default_rng(991)
    rows = []
    for paraphrase in range(3):
        for index, field in enumerate(np.tile(np.linspace(-1.0, 1.0, 9), 20)):
            previous = -1 if index % 2 else 1
            order = index % 2
            probability = 1.0 / (1.0 + np.exp(-(0.1 + 1.3 * field + 0.2 * previous)))
            rows.append(
                {
                    "paraphrase": paraphrase,
                    "local_field": field,
                    "previous_belief": previous,
                    "option_order_right_first": order,
                    "belief_spin": 1 if rng.random() < probability else -1,
                }
            )
    result = heldout_paraphrase_logit(rows)
    assert {int(row["heldout_paraphrase"]) for row in result} == {0, 1, 2}
    assert all(row["train_n"] == 360 and row["test_n"] == 180 for row in result)


def test_empirical_kernel_bootstrap_preserves_state_variable_strata():
    rows = []
    for alpha in (0.0, 0.2):
        for state in range(4):
            for variable in range(2):
                for replicate in range(8):
                    rows.append(
                        {
                            "alpha": alpha,
                            "state_index": state,
                            "variable": variable,
                            "replicate": replicate,
                            "destination_state": state,
                        }
                    )
    summary, samples = bootstrap_empirical_kernel_epr(rows, 1, [0.0, 0.2], 20, 992)
    assert len(summary) == 2
    assert len(samples) == 20
    assert np.allclose(samples["quadratic_delta_coefficient"], 0.0)


def test_curve_comparison_and_dynamic_null_return_declared_units():
    models = curve_model_comparison([0.0, 0.1, 0.2, 0.4], [0.01, 0.011, 0.014, 0.026])
    assert {row["model"] for row in models} == {
        "intercept_plus_linear",
        "intercept_plus_quadratic",
        "intercept_plus_linear_plus_quadratic",
    }
    rows = []
    states = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    for turn, state in enumerate(states):
        rows.append(
            {
                "application": "humanitarian",
                "n_agents": 4,
                "panel": 0,
                "alpha": 0.2,
                "turn": turn,
                "coarse_macrostate": state,
                "messages_sent": 1,
                "message_wire_bytes": 40,
                "prompt_tokens": 100,
                "generated_tokens": 20,
                "service_after": 10.0 - 0.1 * turn,
                "causal_service_change": -0.1,
            }
        )
    result = dynamic_irreversibility_panels(rows, 10, 993)
    assert len(result) == 1
    assert result[0]["turns"] == len(states)
    assert result[0]["messages"] == len(states)
    assert result[0]["beneficial_tool_actions"] == len(states)
