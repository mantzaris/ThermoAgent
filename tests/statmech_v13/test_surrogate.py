import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v13.surrogate import fit_kinetic_surrogate, mean_field_fixed_point


def _response_frame():
    rng = np.random.default_rng(13)
    rows = []
    for temperature in (0.5, 0.85):
        for _ in range(400):
            private = int(rng.choice((-1, 0, 1)))
            neighbor = int(rng.choice((-1, 0, 1)))
            belief = int(rng.choice((-1, 1)))
            action = int(rng.choice((-1, 1)))
            coupling = float(rng.choice((0.35, 0.8)))
            linear = 0.7 * private + 1.1 * coupling * neighbor + 0.3 * belief - 0.1 * action
            p = 1.0 / (1.0 + np.exp(-linear))
            after = 1 if rng.random() < p else -1
            action_after = after if rng.random() < 0.8 else -after
            rows.append({
                "valid_after_repair": 1,
                "sampling_temperature": temperature,
                "private_field": private,
                "neighbor_field": neighbor,
                "coupling_strength": coupling,
                "current_belief": belief,
                "current_action": action,
                "belief_after": after,
                "action_after": action_after,
            })
    return pd.DataFrame(rows)


def test_surrogate_recovers_neighbor_direction_and_finite_mean_field():
    fitted = fit_kinetic_surrogate(_response_frame())
    for value in fitted["fits_by_decoding_noise"].values():
        assert value["belief_coefficients"][2] > 0.0
        result = mean_field_fixed_point(value["belief_coefficients"], value["action_coefficients"], 0.8)
        assert -1.0 <= result["mean_field_belief"] <= 1.0
        assert np.isfinite(result["local_belief_stability_index"])
