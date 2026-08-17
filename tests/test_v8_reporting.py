from pathlib import Path

import pandas as pd

from thermoagent.v8_reporting import _stage_tables, build_v8_reporting


def test_v8_reporting_does_not_invent_unrun_stages(tmp_path: Path):
    root = tmp_path / "repo"
    results = root / "results" / "entropy_triggered_belief_monitoring_v8"
    results.mkdir(parents=True)
    report = build_v8_reporting(root, results)
    assert report["highest_primary_stage"] is None
    readme = (results / "README.md").read_text(encoding="utf-8")
    assert "not evaluated" in readme
    assert "real-human evidence" in readme


def test_v8_reporting_accepts_retained_pre_alias_pilot_schema(tmp_path: Path):
    results = tmp_path / "results"
    pilot = results / "pilots"
    pilot.mkdir(parents=True)
    pd.DataFrame([{
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "application": "humanitarian", "environment_seed": 1,
        "action_policy_id": "rule", "scheduler": "always_on",
        "autonomous_beneficial_actions": 1,
        "autonomous_neutral_actions": 0,
        "autonomous_harmful_actions": 0,
        "attempted_sketch_messages": 1,
        "transmitted_sketch_messages": 1,
        "delivered_sketch_messages": 1,
        "dropped_sketch_messages": 0,
        "forwarded_sketch_messages": 0,
        "sketch_on_wire_bytes": 34,
        "operational_messages": 0, "operational_bytes": 0,
        "fully_counted_messages": 1, "fully_counted_bytes": 34,
        "trigger_activation_rate": 1.0,
        "normalized_time_integrated_estimation_error": 0.125,
        "accepted_physical_actions_v8": 1,
        "service_loss": 1.0, "net_causal_utility": 0.5,
        "normalized_autonomous_reward": 0.5,
        "operator_escalation_requests": 0,
        "complexity": "small", "topology_family": "grid",
        "agent_count": 12, "horizon": 30,
    }]).to_csv(pilot / "episode_summary.csv", index=False)

    tables = _stage_tables(results)

    assert tables["communication_rows"][0][
        "mean_primary_distributed_state_error"
    ] == 0.125
