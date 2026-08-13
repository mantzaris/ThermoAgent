import json
from dataclasses import asdict
from pathlib import Path

from thermoagent.environment import ScenarioConfig
from thermoagent.replay import replay_episode, replay_results
from thermoagent.runner import EpisodeRunner, write_episode


def test_quantitative_episode_replays_from_recorded_tool_calls(tmp_path: Path):
    scenario = ScenarioConfig(
        application="commercial", seed=71, horizon=7, n_agents=8,
        disruption="moderate", decision_interval=2,
    )
    runner = EpisodeRunner(scenario, "scripted_independent")
    result = runner.run("replay-fixture")
    run_dir = tmp_path / "raw" / "main" / result.run_id
    write_episode(result, runner.env.ledger, run_dir)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"configuration": asdict(scenario)}), encoding="utf-8")
    report = replay_episode(run_dir / "episode.json", manifest)
    assert report["replay_passed"]
    assert report["metric_mismatches"] == []
    assert report["tool_result_mismatches"] == []


def test_replay_results_can_select_a_prospectively_named_run_family(tmp_path: Path):
    scenario = ScenarioConfig(
        application="commercial", seed=72, horizon=4, n_agents=8,
        disruption="nominal", decision_interval=2,
    )
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for run_id in ("pilot-paired_nominal_v6-keep", "pilot-old-diagnostic-skip"):
        runner = EpisodeRunner(scenario, "scripted_independent")
        result = runner.run(run_id)
        run_dir = tmp_path / "raw" / "pilot" / run_id
        write_episode(result, runner.env.ledger, run_dir)
        (manifests / (run_id + ".json")).write_text(
            json.dumps({"configuration": asdict(scenario)}), encoding="utf-8"
        )
    report = replay_results(
        tmp_path, ["pilot"], ["paired_nominal_v6"], "pilot-v6.json"
    )
    assert report["episodes_checked"] == 1
    assert report["episodes_passed"] == 1
    assert report["records"][0]["run_id"] == "pilot-paired_nominal_v6-keep"
