from pathlib import Path

from thermoagent.dashboard.app import HTML
from thermoagent.dashboard.replay import DashboardReplay, frame_svg
from thermoagent.human_environment import HumanScenarioConfig
from thermoagent.human_operator import EscalationConfig
from thermoagent.human_runner import HumanOperatorEpisodeRunner, write_human_episode


def _dashboard_episode(tmp_path: Path) -> Path:
    config = HumanScenarioConfig(
        application="commercial",
        seed=12101,
        horizon=10,
        n_agents=8,
        topology="human_v3_development",
        disruption="moderate",
        decision_interval=2,
        communication_budget=60,
        operator_seed=22101,
    )
    runner = HumanOperatorEpisodeRunner(
        config,
        "thermohitl_rule",
        escalation_config=EscalationConfig(
            tau_on=0.25, tau_off=0.10, minimum_dwell=1, cooldown=2
        ),
        enable_counterfactual_probes=False,
    )
    result = runner.run("dashboard-test")
    write_human_episode(result, runner.env.ledger, tmp_path)
    return tmp_path / "episode.json"


def test_dashboard_replay_is_deterministic_and_gpu_free(tmp_path):
    episode = _dashboard_episode(tmp_path)
    first = DashboardReplay(episode)
    second = DashboardReplay(episode)
    assert first.digest() == second.digest()
    assert first.metadata()["gpu_required"] is False
    assert len(first.frames) == 10


def test_dashboard_frame_contains_all_execution_panels_without_private_state(tmp_path):
    replay = DashboardReplay(_dashboard_episode(tmp_path))
    frame = replay.frame(5).as_dict()
    assert set(frame) >= {
        "network", "thermodynamics", "alert_queue", "interventions",
        "workload", "explanation", "material_progress", "view_hashes",
    }
    serialized = str(frame)
    assert "private_cost" not in serialized
    assert "rng_state" not in serialized
    assert "future_disruption" not in serialized


def test_dashboard_svg_export_is_vector_and_has_readable_labels(tmp_path):
    replay = DashboardReplay(_dashboard_episode(tmp_path))
    svg = frame_svg(replay.frame(5))
    assert svg.startswith("<svg")
    assert "ThermoHITL operator replay" in svg
    assert "Energy–entropy phase plane" in svg
    assert "<circle" in svg and "<text" in svg
    assert "<image" not in svg


def test_dashboard_client_exposes_required_replay_controls_and_panels():
    for text in (
        "Play", "Step", "Jump to alert", "Export SVG",
        "Network and autonomy", "Thermodynamic system view",
        "Energy–entropy phase plane", "Alert queue", "Operator workload",
        "Explanation and bounded intervention", "Evaluator analysis",
        "Privileged analysis",
    ):
        assert text in HTML
