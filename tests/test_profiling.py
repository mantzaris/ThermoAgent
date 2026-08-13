import gzip
import json

import pytest

from thermoagent.profiling import (
    _ordinary_message_activation_summary,
    _scale_factor,
    project_matrix,
)


def _summary(wall: float, calls: float, prompt: float, generated: float):
    return {
        "wall_clock_seconds_mean": wall,
        "wall_clock_seconds_p90": 2 * wall,
        "llm_calls_mean": calls,
        "llm_calls_p90": 2 * calls,
        "prompt_tokens_mean": prompt,
        "prompt_tokens_p90": 2 * prompt,
        "generated_tokens_mean": generated,
        "generated_tokens_p90": 2 * generated,
    }


def test_projection_scales_agentic_work_by_agents_and_epochs():
    assert _scale_factor("thermoagent", 12, 24, 4) == pytest.approx(1.8)
    assert _scale_factor("centralized_llm", 12, 24, 4) == pytest.approx(3.6)
    assert _scale_factor("centralized_lookahead", 12, 24, 4) == pytest.approx(7.2)


def test_projection_maps_parameter_matched_ablation_to_thermo_profile():
    config = {
        "stage": "ablations",
        "decision_interval": 4,
        "applications": {"commercial": {"n_agents": 8}},
        "methods": ["thermoagent", "no_episodic_memory"],
        "seeds": [1],
        "scenarios": {"stress": {"horizon": 20}},
    }
    projected = project_matrix(
        config,
        {"thermoagent": _summary(10.0, 20.0, 1000.0, 100.0)},
    )
    assert projected["episodes"] == 2
    assert projected["expected_wall_clock_seconds"] == 20.0
    assert projected["upper_generated_tokens"] == 400.0
    assert projected["profiling_analogue_episode_counts"] == {"thermoagent": 2}


def test_random_gate_calibration_uses_message_active_epochs_not_fanout(tmp_path):
    run_id = "pilot-commercial-thermoagent-n08-s1-paired_nominal_v8"
    path = tmp_path / "raw" / "pilot" / run_id / "events.jsonl.gz"
    path.parent.mkdir(parents=True)
    events = [
        {"kind": "llm_structured_response", "actor": "a", "step": 0},
        {"kind": "message", "actor": "a", "step": 0, "payload": {"kind": "coalition_proposal"}},
        {"kind": "message", "actor": "a", "step": 0, "payload": {"kind": "coalition_proposal"}},
        {"kind": "llm_structured_response", "actor": "b", "step": 0},
        {"kind": "message", "actor": "b", "step": 0, "payload": {"kind": "commitment_breach"}},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    summary = _ordinary_message_activation_summary(tmp_path, [run_id])
    assert summary["decision_epochs"] == 2
    assert summary["message_active_epochs"] == 1
    assert summary["ordinary_validated_messages"] == 2
    assert summary["probability"] == pytest.approx(0.5)
