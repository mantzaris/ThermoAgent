from pathlib import Path

from thermoagent.v6_experiments import run_episode
from thermoagent.v6_replay import replay_episode


def test_v6_ledger_replay_regenerates_metrics_conservation_and_privacy(tmp_path):
    root = tmp_path / "results"
    run_episode(
        Path.cwd(), root, "test_replay", "humanitarian", "compound",
        "private_fragmented", 66920, "action_value_margin", 0.5,
        "event_triggered", resume=False,
    )
    episode = next((root / "raw" / "test_replay").glob("*/episode.json*"))
    result = replay_episode(root, episode)
    assert result["status"] == "pass"
    assert result["metric_regeneration_match"]
    assert result["privacy_boundary_pass"]
    assert result["maximum_reconstructed_conservation_residual"] <= 1e-9
