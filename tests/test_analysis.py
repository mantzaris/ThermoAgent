from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermoagent.analysis import (
    all_method_paired_statistics,
    collect_results,
    detection_episode_summary,
    estimator_comparison_statistics,
    failure_aware_paired_statistics,
    hierarchical_paired_bootstrap,
    holm_adjust,
    localization_episode_summary,
    sign_flip_pvalue,
)
from thermoagent.experiments import (
    _published_output_matches,
    _recover_published_staging,
    _resumed_manifest_matches_execution,
    expand_matrix,
    freeze_protocol,
    run_matrix,
    verify_protocol,
)
from thermoagent.environment import ScenarioConfig
from thermoagent.doet_analysis import _rl_option_selection
from thermoagent.types import CoordinationOption


def test_holm_adjust_is_monotone_in_sorted_p_values():
    raw = [0.03, 0.001, 0.02]
    adjusted = holm_adjust(raw)
    ordered = [adjusted[index] for index in np.argsort(raw)]
    assert ordered == sorted(ordered)
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_hierarchical_bootstrap_clusters_repeated_scenarios_by_seed():
    paired = pd.DataFrame({
        "seed": [1, 1, 2, 2],
        "improvement": [1.0, 1.0, 3.0, 3.0],
    })
    low, high = hierarchical_paired_bootstrap(paired, "improvement", draws=1000)
    assert low <= 2.0 <= high
    assert sign_flip_pvalue(np.array([1.0, 1.0, 1.0])) < 0.5


def test_monitoring_timing_and_localization_summaries_use_episode_rows():
    detections = pd.DataFrame([
        {
            "application": "commercial", "method": "thermoagent",
            "signal": "exact_entropy", "detection_step": 4,
            "detection_delay": 1, "visible_collapse_step": 6,
            "detected_before_collapse": True,
        },
        {
            "application": "commercial", "method": "thermoagent",
            "signal": "exact_entropy", "detection_step": np.nan,
            "detection_delay": np.nan, "visible_collapse_step": 5,
            "detected_before_collapse": False,
        },
    ])
    summary = detection_episode_summary(detections).iloc[0]
    assert summary["n_disrupted_episodes"] == 2
    assert summary["detection_rate"] == 0.5
    assert summary["proportion_before_visible_collapse"] == 0.5

    localization = pd.DataFrame([
        {
            "application": "commercial", "scenario_name": "compound",
            "top1_localization_correct": True,
            "top3_localization_correct": True,
        },
        {
            "application": "commercial", "scenario_name": "compound",
            "top1_localization_correct": False,
            "top3_localization_correct": True,
        },
    ])
    location_summary = localization_episode_summary(localization).iloc[0]
    assert location_summary["top1_localization_accuracy"] == 0.5
    assert location_summary["top3_localization_accuracy"] == 1.0


def test_collect_results_retains_failed_completion_rows(tmp_path: Path):
    stage = tmp_path / "main"
    stage.mkdir(parents=True)
    pd.DataFrame([{
        "run_id": "failed-run", "application": "commercial",
        "method": "thermoagent", "scenario_name": "compound",
        "seed": 7, "n_agents": 8, "status": "failed",
        "error": "RuntimeError: retained failure",
    }]).to_csv(stage / "episodes.csv", index=False)
    episodes, time_series, agent_metrics = collect_results(tmp_path)
    assert len(episodes) == 1
    assert episodes.iloc[0]["completion_status"] == "failed"
    assert "retained failure" in episodes.iloc[0]["failure_reason"]
    assert time_series.empty and agent_metrics.empty


def test_failed_episode_manifest_retains_complete_reproducibility_fields(
    tmp_path: Path, monkeypatch,
):
    import json
    import thermoagent.experiments as experiments

    class RaisingPlanner:
        revision = "intentional-failure-fixture"

        @staticmethod
        def plan_batch(_requests):
            raise RuntimeError("intentional planner failure")

    monkeypatch.setattr(
        experiments,
        "_planner_for_method",
        lambda _method, _shared: RaisingPlanner(),
    )
    config = tmp_path / "failure.yaml"
    config.write_text(
        """stage: main
planner_backend: mock
prompt_template_revision: fixture-v1
llm_seed: 7
rl_seed: 8
applications:
  commercial: {n_agents: 8}
methods: [centralized_llm]
seeds: [9]
scenarios:
  failure_cell:
    horizon: 1
    private_information: 0.0
    objective_misalignment: 0.5
    communication: reliable
    disruption: nominal
"""
    )
    results = tmp_path / "results"
    rows = run_matrix(config, tmp_path, results)
    assert rows[0]["status"] == "failed"
    manifest_path = next((results / "manifests").glob("main-*.json"))
    manifest = json.loads(manifest_path.read_text())
    required = {
        "configuration", "experiment_configuration", "model_identifier",
        "model_revision", "prompt_template_revision", "decoding",
        "environment_rng_streams", "topology_checksum", "dependencies",
        "hardware", "start_timestamp", "end_timestamp", "wall_clock_seconds",
        "environment_steps", "llm_calls", "prompt_tokens", "generated_tokens",
        "tool_calls", "messages", "checkpoint_selection",
        "checkpoint_sha256", "output_checksums", "completion_status", "error",
    }
    assert required <= set(manifest)
    assert manifest["completion_status"] == "failed"
    assert "intentional planner failure" in manifest["error"]


def test_collect_results_marks_prospectively_excluded_rows(tmp_path: Path):
    import json

    raw = tmp_path / "raw" / "pilot" / "commercial-thermo-final_nominal_v3-1"
    raw.mkdir(parents=True)
    (tmp_path / "reproducibility").mkdir()
    (tmp_path / "reproducibility" / "excluded_runs.json").write_text(
        json.dumps({"rules": [{
            "run_id_contains": "-final_nominal_v3",
            "reason": "known pre-freeze pairing defect",
        }]})
    )
    (raw / "episode.json").write_text(json.dumps({
        "run_id": "commercial-thermo-final_nominal_v3-1",
        "application": "commercial",
        "method": "thermoagent",
        "scenario": "nominal",
        "seed": 1,
        "completion_status": "complete",
        "wall_clock_seconds": 1.0,
        "metrics": {"primary_outcome": 1.0},
        "planner_metrics": {},
        "agent_metrics": {},
        "time_series": [{"step": 0}],
    }))
    episodes, time_series, agent_metrics = collect_results(tmp_path)
    assert not bool(episodes.iloc[0]["analysis_valid"])
    assert episodes.iloc[0]["exclusion_reason"] == "known pre-freeze pairing defect"
    assert not bool(time_series.iloc[0]["analysis_valid"])
    assert not bool(agent_metrics.iloc[0]["analysis_valid"])


def test_failure_aware_pairing_counts_asymmetric_failure_as_ranked_loss():
    rows = pd.DataFrame([
        {
            "stage": "main", "application": "commercial",
            "method": "thermoagent", "scenario_name": "cell", "seed": 1,
            "n_agents": 8, "completion_status": "failed",
            "primary_outcome": np.nan, "analysis_valid": True,
        },
        {
            "stage": "main", "application": "commercial",
            "method": "learned_no_entropy", "scenario_name": "cell", "seed": 1,
            "n_agents": 8, "completion_status": "complete",
            "primary_outcome": 4.0, "analysis_valid": True,
        },
        {
            "stage": "main", "application": "commercial",
            "method": "thermoagent", "scenario_name": "cell", "seed": 2,
            "n_agents": 8, "completion_status": "complete",
            "primary_outcome": 2.0, "analysis_valid": True,
        },
        {
            "stage": "main", "application": "commercial",
            "method": "learned_no_entropy", "scenario_name": "cell", "seed": 2,
            "n_agents": 8, "completion_status": "complete",
            "primary_outcome": 3.0, "analysis_valid": True,
        },
    ])
    result = failure_aware_paired_statistics(rows)
    row = result[result["comparator"] == "learned_no_entropy"].iloc[0]
    assert row["n_planned_pairs"] == 2
    assert row["n_both_complete"] == 1
    assert row["thermoagent_only_failures"] == 1
    assert row["failure_aware_win_rate"] == 0.5
    assert row["complete_case_mean_improvement"] == 1.0


def test_checksum_backed_staging_publish_recovers_without_rerun(tmp_path: Path):
    import hashlib

    run_id = "main-commercial-thermoagent-n08-s001"
    staging_root = tmp_path / ".staging" / "main"
    candidate = staging_root / (run_id + ".partial-fixture")
    candidate.mkdir(parents=True)
    episode = candidate / "episode.json"
    events = candidate / "events.jsonl.gz"
    episode.write_bytes(b"episode")
    events.write_bytes(b"events")
    manifest = {"output_checksums": {
        "episode.json": hashlib.sha256(b"episode").hexdigest(),
        "events.jsonl.gz": hashlib.sha256(b"events").hexdigest(),
    }}
    output = tmp_path / "raw" / "main" / run_id
    output.parent.mkdir(parents=True)
    assert _recover_published_staging(staging_root, output, run_id, manifest)
    assert output.joinpath("episode.json").read_bytes() == b"episode"
    assert not candidate.exists()


def test_resumed_published_episode_requires_both_manifest_checksums(tmp_path: Path):
    import hashlib

    output = tmp_path / "episode"
    output.mkdir()
    (output / "episode.json").write_bytes(b"episode")
    (output / "events.jsonl.gz").write_bytes(b"events")
    manifest = {"output_checksums": {
        "episode.json": hashlib.sha256(b"episode").hexdigest(),
        "events.jsonl.gz": hashlib.sha256(b"events").hexdigest(),
    }}
    assert _published_output_matches(output, manifest)
    (output / "events.jsonl.gz").write_bytes(b"tampered")
    assert not _published_output_matches(output, manifest)
    assert not _published_output_matches(output, {"output_checksums": {}})


def test_resumed_manifest_must_match_the_frozen_execution_contract():
    scenario = ScenarioConfig(
        application="commercial", seed=8101, horizon=16, n_agents=10,
        communication="partition", disruption="moderate",
        topology="tri_region_bridge_v2",
    )
    run_config = {
        "llm_seed": 9101,
        "protocol_checksum": "freeze",
        "model": {"identifier": "model", "revision": "revision"},
        "resolved_trigger": {"parameters": {"direction": "low"}},
    }
    manifest = {
        "completion_status": "complete",
        "source": {"checksum": "source"},
        "application": "commercial",
        "method": "doet_rule",
        "configuration": asdict(scenario),
        "environment_seed": 8101,
        "llm_seed": 9101,
        "rl_seed": 0,
        "topology_identifier": "tri_region_bridge_v2",
        "model_identifier": "model",
        "model_revision": "revision",
        "protocol_checksum": "freeze",
        "trigger_parameters": {"direction": "low"},
    }
    assert _resumed_manifest_matches_execution(
        manifest, "source", scenario, "doet_rule", 0, run_config
    )
    manifest["configuration"]["horizon"] = 17
    assert not _resumed_manifest_matches_execution(
        manifest, "source", scenario, "doet_rule", 0, run_config
    )


def test_rl_option_selection_reports_matched_total_variation(tmp_path: Path):
    root = tmp_path / "results"
    raw = root / "raw" / "holdout_locked"
    summaries = []
    for method, counts, rl_seed in (
        ("learned_no_entropy", {"0": 2, "1": 0}, 7301),
        ("doet_rl", {"0": 1, "1": 1}, 7302),
    ):
        run_id = "run-" + method
        output = raw / run_id
        output.mkdir(parents=True)
        (output / "episode.json").write_text(
            json.dumps({"agent_metrics": {"option_counts": counts}}),
            encoding="utf-8",
        )
        summaries.append({
            "run_id": run_id,
            "application": "commercial",
            "scenario_name": "isolated",
            "seed": 8101,
            "n_agents": 10,
            "method": method,
            "rl_training_seed": rl_seed,
        })
    options, differences = _rl_option_selection(
        root, pd.DataFrame(summaries)
    )
    assert len(options) == 2 * len(CoordinationOption)
    assert differences[
        "panel_option_total_variation_distance"
    ].nunique() == 1
    assert differences[
        "panel_option_total_variation_distance"
    ].iloc[0] == pytest.approx(0.5)
    request = differences[
        differences["option_name"] == "request_info"
    ].iloc[0]
    assert request[
        "option_proportion_difference_doet_minus_nonentropy"
    ] == pytest.approx(0.5)


def test_protocol_verification_fails_closed_after_frozen_file_changes(tmp_path: Path):
    import pytest

    config = tmp_path / "config.yaml"
    config.write_text("stage: main\n")
    freeze_path = tmp_path / "protocol_freeze.json"
    freeze_protocol(tmp_path, freeze_path, [config])
    verified = verify_protocol(tmp_path, freeze_path)
    assert verified["status"] == "verified"
    assert verified["files_verified"] == 1

    config.write_text("stage: changed\n")
    with pytest.raises(RuntimeError, match="verification failed"):
        verify_protocol(tmp_path, freeze_path)


def test_matrix_can_target_a_second_size_to_selected_scenarios():
    config = {
        "applications": {"commercial": {"n_agents": 11, "agent_counts": [8, 11]}},
        "methods": ["thermoagent"],
        "seeds": [1],
        "scenarios": {
            "all_sizes": {},
            "large_only": {"agent_counts_by_application": {"commercial": [11]}},
        },
    }
    matrix = expand_matrix(config)
    assert [(row[1], row[4]["name"]) for row in matrix] == [
        (8, "all_sizes"), (11, "all_sizes"), (11, "large_only")
    ]


def test_frozen_main_matrix_has_high_information_design():
    import yaml

    config = yaml.safe_load(Path("configs/main.yaml").read_text())
    matrix = expand_matrix(config)
    assert len(matrix) == 944
    factor_cells = {
        (row[4]["private_information"], row[4]["objective_misalignment"])
        for row in matrix if row[4]["name"].startswith("factor_")
    }
    assert factor_cells == {(0.0, 0.0), (0.5, 0.5), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    assert {row[1] for row in matrix} == {8, 10, 11}


def test_prefreeze_qualification_and_ablation_matrices_are_matched():
    import yaml

    pilot = yaml.safe_load(Path("configs/pilot.yaml").read_text())
    pilot_matrix = expand_matrix(pilot)
    assert len(pilot_matrix) == 174
    final_rows = [row for row in pilot_matrix if row[4]["name"].endswith("_v8")]
    assert len(final_rows) == 84
    assert {row[3] for row in final_rows} == set(pilot["scenarios"]["paired_nominal_v8"]["methods"])

    ablations = yaml.safe_load(Path("configs/ablations.yaml").read_text())
    ablation_matrix = expand_matrix(ablations)
    assert len(ablation_matrix) == 72
    assert {row[3] for row in ablation_matrix} == set(ablations["methods"])


def test_all_method_pairing_and_estimator_comparison_use_episode_units():
    rows = []
    for seed in (1, 2):
        for method, outcome in (("thermoagent", 2.0), ("random_gate", 3.0)):
            rows.append({
                "stage": "ablations",
                "completion_status": "complete",
                "application": "commercial",
                "method": method,
                "scenario_name": "compound",
                "seed": seed,
                "n_agents": 8,
                "primary_outcome": outcome + seed * 0.1,
            })
    paired = all_method_paired_statistics(pd.DataFrame(rows), "ablations")
    assert paired.iloc[0]["n_pairs"] == 2
    assert paired.iloc[0]["mean_improvement"] == 1.0

    time_rows = pd.DataFrame([{
        "stage": "main",
        "application": "commercial",
        "scenario": "reliable-moderate-p0.5-o0.5",
        "scenario_name": "cell",
        "exact_entropy": 0.5,
        "exact_free_energy": 0.2,
        "distributed_entropy_mean": 0.45,
        "distributed_free_energy_mean": 0.18,
        "delayed_entropy_mean": 0.4,
        "delayed_free_energy_mean": 0.16,
        "noisy_entropy_mean": 0.48,
        "noisy_free_energy_mean": 0.19,
    }])
    estimators = estimator_comparison_statistics(time_rows)
    assert set(estimators["estimator"]) == {
        "exact_evaluator_only",
        "agent_local_distributed",
        "one_period_delayed",
        "noisy_distributed_sigma_0.01",
        "no_entropy_estimate",
    }
