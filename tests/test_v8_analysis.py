import numpy as np
import pandas as pd

from thermoagent.v8_analysis import (
    _bootstrap_mean_interval,
    _holm_adjust,
    _paired_comparisons,
)


def test_panel_bootstrap_is_reproducible_and_uses_input_panels():
    values = np.asarray([0.20, 0.25, 0.30, 0.35])
    first = _bootstrap_mean_interval(values, seed=88, replicates=1000)
    second = _bootstrap_mean_interval(values, seed=88, replicates=1000)
    assert first == second
    assert np.isclose(first["mean"], 0.275)
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    raw = [0.04, 0.001, 0.02]
    adjusted = _holm_adjust(raw)
    order = np.argsort(raw)
    ordered = np.asarray(adjusted)[order]
    assert np.all(np.diff(ordered) >= -1e-12)
    assert all(value >= raw[index] for index, value in enumerate(adjusted))


def test_paired_log_ratios_use_scheduler_arm_and_always_on_reference():
    common = {
        "panel_id": "humanitarian:1",
        "encoding": "uint8_simplex",
        "fully_counted_messages": 20,
        "fully_counted_bytes": 200,
        "normalized_time_integrated_estimation_error": 0.1,
        "primary_distributed_state_error": 0.1,
        "primary_distributed_state_error_p95": 0.2,
        "pointwise_estimation_mae_p95": 0.2,
        "disagreement_time_integrated_error": 0.1,
        "mean_detection_delay_steps": 2.0,
        "service_loss": 10.0,
        "autonomous_harmful_actions": 1,
        "normalized_autonomous_reward": 0.8,
    }
    frame = pd.DataFrame([
        {
            **common,
            "candidate_name": "always_u8",
            "scheduler": "always_on",
            "sketch_on_wire_bytes": 100,
            "transmitted_sketch_messages": 10,
        },
        {
            **common,
            "candidate_name": "generalized_0125_u8",
            "scheduler": "generalized_information",
            "sketch_on_wire_bytes": 50,
            "transmitted_sketch_messages": 5,
        },
    ])
    paired = _paired_comparisons(frame)
    row = paired.loc[paired.scheduler.eq("generalized_information")].iloc[0]
    assert np.isclose(row.log_sketch_message_ratio, np.log(6.0 / 11.0))
    assert np.isclose(row.log_wire_byte_ratio, np.log(51.0 / 101.0))
