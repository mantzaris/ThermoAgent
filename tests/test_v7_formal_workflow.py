import json

from thermoagent.v7_formal_workflow import evaluate_formal_development_gates


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _formal_inputs(root, replay_status="pass"):
    _write_json(
        root / "statistics" / "dynamic_primary_analysis.json",
        {"H1_pass": True, "H2_pass": True},
    )
    _write_json(
        root / "statistics" / "communication_primary_analysis.json",
        {"H3_pass": True},
    )
    _write_json(
        root / "reproducibility" / "replay" / "replay_summary.json",
        {
            "status": replay_status,
            "replay_mismatches": 0,
            "privacy_failures": 0,
            "maximum_conservation_residual": 0.0,
        },
    )


def test_v7_formal_progression_requires_primary_results_and_integrity(tmp_path):
    _formal_inputs(tmp_path)
    report = evaluate_formal_development_gates(tmp_path)
    assert report["engineering_integrity_pass"] is True
    assert report["formal_development_primary_pass"] is True
    assert report["RL_training_unlocked"] is True


def test_v7_formal_progression_refuses_favorable_results_with_replay_failure(tmp_path):
    _formal_inputs(tmp_path, replay_status="fail")
    report = evaluate_formal_development_gates(tmp_path)
    assert report["engineering_integrity_pass"] is False
    assert report["formal_development_primary_pass"] is False
    assert report["RL_training_unlocked"] is False


def test_v7_formal_progression_retains_and_counts_episode_failures(tmp_path):
    _formal_inputs(tmp_path)
    failure = tmp_path / "negative_results" / "formal_dynamic_failures.csv"
    failure.parent.mkdir(parents=True, exist_ok=True)
    failure.write_text(
        "status,failure_type\nfailed,RuntimeError\n", encoding="utf-8",
    )
    report = evaluate_formal_development_gates(tmp_path)
    assert report["formal_episode_failures"] == 1
    assert report["engineering_integrity_pass"] is False
    assert report["formal_development_primary_pass"] is False
