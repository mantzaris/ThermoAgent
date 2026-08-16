from pathlib import Path

from thermoagent.dashboard.v6 import V6DashboardReplay, frame_svg_v6
from thermoagent.v6_experiments import run_episode
from thermoagent.v6_policies import SelectiveController


def test_v6_dashboard_uses_hashed_authorized_view_without_evaluator_state(tmp_path):
    root = tmp_path / "results"
    run_episode(
        Path.cwd(), root, "dashboard_test", "utility_restoration", "compound",
        "private_fragmented", 66930, "jensen_shannon", 0.5,
        "event_triggered", resume=False,
        controller_override=SelectiveController(
            "jensen_shannon", 0.5, 1, escalation_risk_threshold=0.0,
        ),
    )
    episode = next((root / "raw" / "dashboard_test").glob("*/episode.json*"))
    replay = V6DashboardReplay(episode)
    assert replay.frames
    populated = [value for value in replay.frames if value.view_hashes]
    assert populated
    serialized = repr(populated[-1].as_dict()).lower()
    assert "true_mode" not in serialized
    assert "evaluator_distributed_error" not in serialized
    svg = frame_svg_v6(populated[-1])
    assert svg.startswith("<svg") and "simulated operator" in svg.lower()
