import numpy as np

from thermoagent.v6_learnability import (
    evaluate_action_value_learnability, generate_action_value_dataset,
)


def test_action_value_dataset_contains_multiple_authorized_actions_and_real_effects():
    frame = generate_action_value_dataset(
        (66101, 66102, 66103, 66104, 66105),
        applications=("humanitarian",), regimes=("compound",),
    )
    assert frame.candidate_action.nunique() >= 3
    assert (frame.causal_effect > 0).any()
    assert (frame.causal_effect < 0).any()
    assert (frame.causal_effect == 0).any()
    result = evaluate_action_value_learnability(frame)
    assert result["applications"][0]["authorized_oracle_mean_action_utility"] > 0
    assert all(all(value[key] for key in (
        "environment_seed_disjoint", "topology_family_disjoint",
        "scenario_family_disjoint",
    )) for value in result["folds"])
