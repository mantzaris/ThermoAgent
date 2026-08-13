from dataclasses import replace
import hashlib
import sys
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import yaml

from thermoagent.doet import (
    CommunicationMode,
    DistributedEntropyTrigger,
    TriggerConfig,
)
from thermoagent.environment import ScenarioConfig
from thermoagent.planners import (
    MockPlanner,
    PlannerRequest,
    TransformersPlanner,
    validate_request_plan,
)
from thermoagent.runner import EpisodeRunner
from thermoagent.policy import CoordinationPolicy
from thermoagent.doet_analysis import _comparison_rows, _frontier_rows
from thermoagent.doet_ablations import run as design_doet_ablations
from thermoagent.doet_holdout import run as design_doet_holdout
from thermoagent.experiments import capture_source_provenance, expand_matrix
from thermoagent.types import CoordinationOption


def _config(**values):
    base = TriggerConfig(
        nominal_center=0.5,
        nominal_scale=0.1,
        rho=0.5,
        kappa=0.0,
        tau_on=2.0,
        tau_off=0.5,
        tau_crisis=4.0,
        minimum_dwell=2,
        cooldown=1,
        crisis_surprisal=99.0,
        quiet_gossip_rounds=1,
        targeted_gossip_rounds=2,
        crisis_gossip_rounds=3,
    )
    return replace(base, **values)


def test_trigger_configuration_rejects_invalid_hysteresis():
    with pytest.raises(ValueError, match="thresholds"):
        TriggerConfig(tau_off=2.0, tau_on=1.0)
    with pytest.raises(ValueError, match="direction"):
        TriggerConfig(direction="oracle_label")
    with pytest.raises(ValueError, match="signal_noise_std"):
        TriggerConfig(signal_noise_std=-0.1)
    with pytest.raises(ValueError, match="signal_scale"):
        TriggerConfig(signal_scale=0.0)


def test_agents_have_independent_trigger_state_and_no_global_input():
    trigger = DistributedEntropyTrigger(["a", "b"], _config(direction="high"))
    decision_a = trigger.update("a", 0, 0.8, 0.0, 0.0, 1.0)
    decision_b = trigger.update("b", 0, 0.5, 0.0, 0.0, 1.0)
    assert decision_a.mode == int(CommunicationMode.TARGETED)
    assert decision_b.mode == int(CommunicationMode.QUIET)
    assert trigger.states["a"] is not trigger.states["b"]
    # The API deliberately offers no true label or global entropy argument.
    with pytest.raises(TypeError):
        trigger.update(
            "b", 1, 0.5, 0.0, 0.0, 1.0,
            global_entropy=1.0,
        )


def test_cusum_hysteresis_enforces_dwell_and_deactivation_threshold():
    trigger = DistributedEntropyTrigger(["a"], _config(direction="high"))
    first = trigger.update("a", 0, 0.8, 0.0, 0.0, 1.0)
    assert first.activated
    assert first.mode == int(CommunicationMode.TARGETED)
    # Below tau_off immediately after activation, but minimum dwell is two.
    second = trigger.update("a", 1, 0.5, 0.0, 0.0, 1.0)
    assert second.mode == int(CommunicationMode.TARGETED)
    third = trigger.update("a", 2, 0.5, 0.0, 0.0, 1.0)
    assert third.mode == int(CommunicationMode.TARGETED)
    fourth = trigger.update("a", 3, 0.5, 0.0, 0.0, 1.0)
    assert fourth.deactivated
    assert fourth.mode == int(CommunicationMode.QUIET)


def test_absolute_and_low_direction_detect_opposite_entropy_changes():
    low = DistributedEntropyTrigger(["a"], _config(direction="low"))
    absolute = DistributedEntropyTrigger(["a"], _config(direction="absolute"))
    assert low.update("a", 0, 0.2, 0.0, 0.0, 1.0).activated
    assert absolute.update("a", 0, 0.8, 0.0, 0.0, 1.0).activated


def test_neighbor_alert_is_bounded_evidence_not_central_activation():
    config = _config(
        direction="high",
        tau_on=3.0,
        tau_crisis=5.0,
        alert_weight=0.5,
        propagation="neighbor",
    )
    trigger = DistributedEntropyTrigger(["a"], config)
    # Fifty alerts are capped at one alert's evidence and cannot force mode.
    decision = trigger.update("a", 0, 0.5, 0.0, 0.0, 1.0, delivered_alerts=50)
    assert decision.trigger_residual == pytest.approx(0.5)
    assert decision.mode == int(CommunicationMode.QUIET)


def test_consensus_disagreement_reduces_untrusted_local_level_evidence():
    config = _config(direction="high", tau_on=2.0, tau_crisis=4.0)
    confident = DistributedEntropyTrigger(["a"], config)
    uncertain = DistributedEntropyTrigger(["a"], config)
    assert confident.update("a", 0, 0.8, 0.0, 0.0, 1.0).activated
    assert not uncertain.update("a", 0, 0.8, 0.0, 0.9, 1.0).activated


def test_mode_controls_local_gossip_and_planning_cadence():
    trigger = DistributedEntropyTrigger(["a"], _config(direction="high"))
    assert trigger.gossip_rounds("a") == 1
    assert trigger.decision_interval("a") == 8
    trigger.update("a", 0, 0.8, 0.0, 0.0, 1.0)
    assert trigger.mode("a") == CommunicationMode.TARGETED
    assert trigger.gossip_rounds("a") == 2
    assert trigger.decision_interval("a") == 4
    trigger.update("a", 1, 1.0, 0.0, 0.0, 1.0)
    assert trigger.mode("a") == CommunicationMode.CRISIS
    assert trigger.gossip_rounds("a") == 3
    assert trigger.decision_interval("a") == 2


def test_trigger_replay_is_deterministic_and_steps_are_monotonic():
    values = [0.5, 0.7, 0.9, 0.4, 0.5]
    outputs = []
    for _ in range(2):
        trigger = DistributedEntropyTrigger(["a"], _config(direction="absolute"))
        outputs.append([
            trigger.update("a", step, value, 0.2, 0.05, 0.8).as_dict()
            for step, value in enumerate(values)
        ])
    assert outputs[0] == outputs[1]
    trigger = DistributedEntropyTrigger(["a"], _config())
    trigger.update("a", 0, 0.5, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="increase monotonically"):
        trigger.update("a", 0, 0.5, 0.0, 0.0, 1.0)


def test_public_route_affordance_prevents_arbitrary_material_target():
    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial",
            seed=77,
            n_agents=11,
            horizon=8,
            topology="ring_plus_hubs",
        ),
        "scripted_independent",
    )
    runner.env.transition()
    runner.env.deliver_observations()
    runner._update_monitor()
    source = next(
        agent_id for agent_id in runner.env.agent_ids
        if runner._material_action_guidance(agent_id)["eligible_offer_target_ids"]
    )
    agent = runner.env.agents[source]
    context = agent.retrieve_context(0, runner.env.ledger)
    context["material_action_guidance"] = runner._material_action_guidance(source)
    request = PlannerRequest(
        source,
        agent.identity.role,
        "commercial",
        int(CoordinationOption.NEGOTIATE),
        context,
        runner.env.public_identities(),
    )
    plan = MockPlanner().plan(request)
    assert plan.arguments["target"] in context["material_action_guidance"][
        "eligible_offer_target_ids"
    ]
    assert validate_request_plan(request, plan) is None
    public_guidance = context["material_action_guidance"]
    assert "inventory" not in public_guidance
    assert "private_cost" not in public_guidance


def test_doet_runner_counts_sparse_sketches_and_explicit_alerts_only():
    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial",
            seed=91,
            n_agents=8,
            horizon=10,
            disruption="correlated",
            communication_budget=120,
        ),
        "doet_rule",
        trigger_config={
            "nominal_center": 0.9,
            "nominal_scale": 0.05,
            "direction": "low",
            "tau_on": 1.0,
            "tau_off": 0.2,
            "tau_crisis": 2.0,
            "propagation": "neighbor",
        },
    )
    result = runner.run("doet-integration")
    trigger_events = [
        event for event in runner.env.ledger.events
        if event.kind == "coordination_trigger"
    ]
    assert trigger_events
    assert all(
        event.payload["signal_source"] == "distributed_operational_entropy"
        for event in trigger_events
    )
    assert all("global_entropy" not in event.payload for event in trigger_events)
    alert_messages = [
        event for event in runner.env.ledger.events
        if event.kind == "message" and event.payload.get("kind") == "entropy_alert"
    ]
    assert len(alert_messages) == result.metrics["trigger_alert_successes"]
    assert all(
        set(event.payload["payload"]) == {
            "recommended_mode", "anomaly_level", "protocol"
        }
        for event in alert_messages
    )
    assert result.metrics["total_communication_messages"] == (
        result.metrics["messages"] + result.metrics["monitor_sketch_messages"]
    )
    assert abs(result.metrics["conservation_error"]) < 1e-8


def test_fixed_always_on_is_a_strong_counted_status_broadcast_control():
    config = ScenarioConfig(
        application="commercial",
        seed=92,
        n_agents=8,
        horizon=10,
        disruption="moderate",
        communication_budget=200,
    )
    fixed_runner = EpisodeRunner(config, "fixed_always_on")
    fixed = fixed_runner.run("fixed-integration")
    periodic = EpisodeRunner(config, "periodic_communication").run(
        "periodic-integration"
    )
    fixed_packets = [
        event for event in fixed_runner.env.ledger.events
        if event.kind == "message" and event.payload.get("kind") == "fixed_status"
    ]
    assert fixed_packets
    assert all(
        set(event.payload["payload"]) == {
            "pressure", "capacity", "commitment_strain", "protocol"
        }
        for event in fixed_packets
    )
    assert fixed.metrics["messages"] > periodic.metrics["messages"]
    assert fixed.metrics["crisis_mode_fraction"] == pytest.approx(1.0)


def test_zero_rate_random_control_retains_quiet_local_planning():
    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=192, n_agents=8, horizon=10,
            disruption="moderate", random_gate_probability=0.0,
        ),
        "random_budget_matched",
    )
    result = runner.run("random-zero-rate")
    assert result.metrics["quiet_mode_fraction"] == pytest.approx(1.0)
    assert result.agent_metrics["tool_proposals"] >= 2 * 8
    assert result.metrics["communication_active_decision_epochs"] == 0


def test_doet_rl_actor_receives_only_24_local_features_and_trigger_mask():
    policy = CoordinationPolicy(seed=13)
    runner = EpisodeRunner(
        ScenarioConfig(
            application="humanitarian",
            seed=93,
            n_agents=8,
            horizon=8,
            communication_budget=100,
        ),
        "doet_rl",
        policy=policy,
        trigger_config={
            "nominal_center": 0.5,
            "nominal_scale": 0.1,
            "tau_on": 2.0,
            "tau_off": 0.5,
            "tau_crisis": 4.0,
        },
    )
    result = runner.run("doet-rl-local-input")
    assert result.trajectory
    assert all(len(row["observation"]) == 24 for row in result.trajectory)
    assert all(len(row["action_mask"]) == 9 for row in result.trajectory)
    assert abs(result.metrics["conservation_error"]) < 1e-8


def test_v2_holdout_topology_is_connected_and_distinct_from_prior_graphs():
    import networkx as nx

    config = ScenarioConfig(
        application="commercial",
        seed=94,
        n_agents=10,
        topology="tri_region_bridge_v2",
    )
    runner = EpisodeRunner(config, "scripted_independent")
    graph = nx.Graph()
    graph.add_nodes_from(runner.env.agent_ids)
    graph.add_edges_from(runner.env.initial_communication_edges)
    assert nx.is_connected(graph)
    assert runner.env.initial_communication_edges != set(
        EpisodeRunner(
            ScenarioConfig(
                application="commercial",
                seed=94,
                n_agents=10,
                topology="holdout_nine_agent",
            ),
            "scripted_independent",
        ).env.initial_communication_edges
    )


def test_balanced_rl_assignment_uses_every_seed_with_at_most_one_count_gap():
    config = {
        "applications": {"commercial": {"n_agents": 8}},
        "methods": ["doet_rl", "fixed_always_on"],
        "seeds": list(range(100, 125)),
        "rl_seeds": [1, 2, 3, 4, 5],
        "balanced_rl_assignment": True,
        "scenarios": {
            "one": {"communication": "reliable", "disruption": "moderate"},
            "two": {"communication": "partition", "disruption": "compound"},
        },
    }
    matrix = expand_matrix(config)
    counts = {seed: 0 for seed in config["rl_seeds"]}
    fixed = 0
    for _, _, _, method, scenario in matrix:
        if method == "doet_rl":
            counts[scenario["_rl_seed"]] += 1
        else:
            fixed += 1
    assert set(counts.values()) == {10}
    assert fixed == 50


def _analysis_fixture_frame():
    rows = []
    methods = [
        "fixed_always_on", "periodic_communication",
        "random_budget_matched", "learned_no_entropy",
        "kpi_cusum_trigger", "doet_rule", "doet_rl",
    ]
    scenarios = [
        "nominal", "isolated", "communication_partition",
        "correlated", "compound_ood",
    ]
    for application in ("commercial", "humanitarian"):
        for scenario_index, scenario in enumerate(scenarios):
            for seed in range(1, 6):
                for method_index, method in enumerate(methods):
                    fixed_loss = 100.0 + scenario_index + seed / 10.0
                    loss = fixed_loss + {
                        "fixed_always_on": 0.0,
                        "periodic_communication": 2.0,
                        "random_budget_matched": 2.5,
                        "learned_no_entropy": 1.5,
                        "kpi_cusum_trigger": 1.0,
                        "doet_rule": 0.5,
                        "doet_rl": 0.7,
                    }[method]
                    messages = {
                        "fixed_always_on": 100.0,
                        "periodic_communication": 65.0,
                        "random_budget_matched": 60.0,
                        "learned_no_entropy": 75.0,
                        "kpi_cusum_trigger": 58.0,
                        "doet_rule": 40.0,
                        "doet_rl": 45.0,
                    }[method]
                    rows.append({
                        "application": application,
                        "scenario_name": scenario,
                        "seed": seed,
                        "n_agents": 10,
                        "method": method,
                        "rl_training_seed": (
                            7301 + (seed - 1)
                            if method in (
                                "learned_no_entropy", "doet_rl"
                            ) else 0
                        ),
                        "primary_outcome": loss,
                        "total_communication_messages": messages,
                        "total_communication_bytes": messages * 100,
                        "prompt_tokens": messages * 20,
                        "generated_tokens": messages * 2,
                        "llm_calls": messages / 2,
                        "llm_latency_seconds": messages / 4,
                        "wall_clock_seconds": messages / 3,
                        "mean_consensus_rmse": 0.1,
                        "trigger_activations": 2,
                        "quiet_mode_fraction": 0.5,
                        "communication_active_decision_epochs": 10,
                        "tool_proposals": 20,
                    })
    return pd.DataFrame(rows)


def test_locked_analysis_uses_paired_panels_and_all_frontier_costs(monkeypatch):
    monkeypatch.setattr("thermoagent.doet_analysis.BOOTSTRAP_REPLICATES", 200)
    frame = _analysis_fixture_frame()
    comparisons, bootstrap = _comparison_rows(frame)
    primary = [
        row for row in comparisons
        if row["method"] == "doet_rule"
        and row["scenario"] == "all_non_nominal"
    ]
    assert len(primary) == 2
    assert all(row["noninferior"] for row in primary)
    assert all(row["communication_target_20_percent"] for row in primary)
    assert len(bootstrap["primary_holm_tests"]) == 4
    frontiers = _frontier_rows(frame)
    assert len(frontiers) == 8
    assert all(row["doet_improves_frontier"] for row in frontiers)


def test_locked_analysis_retains_failed_pair_count_without_imputation(monkeypatch):
    monkeypatch.setattr("thermoagent.doet_analysis.BOOTSTRAP_REPLICATES", 50)
    frame = _analysis_fixture_frame()
    frame["status"] = "complete"
    failed = (
        (frame["application"] == "commercial")
        & (frame["scenario_name"] == "isolated")
        & (frame["seed"] == 1)
        & (frame["method"] == "doet_rule")
    )
    frame.loc[failed, "status"] = "failed"
    comparisons, _ = _comparison_rows(frame)
    isolated = next(
        row for row in comparisons
        if row["method"] == "doet_rule"
        and row["application"] == "commercial"
        and row["scenario"] == "isolated"
    )
    aggregate = next(
        row for row in comparisons
        if row["method"] == "doet_rule"
        and row["application"] == "commercial"
        and row["scenario"] == "all_non_nominal"
    )
    assert isolated["paired_episodes"] == 4
    assert isolated["failed_pairs"] == 1
    assert aggregate["failed_pairs"] == 1


def test_filtered_deployment_provenance_records_branch_and_source(tmp_path):
    output = tmp_path / "execution_source.json"
    record = capture_source_provenance(Path.cwd(), output)
    assert output.is_file()
    assert record["branch"]
    assert len(record["source_checksum"]) == 64
    assert "credentials" in record["security_note"]


def test_real_llm_profile_is_small_and_seed_separated():
    config = yaml.safe_load(
        Path("configs/entropy_trigger_profile.yaml").read_text(
            encoding="utf-8"
        )
    )
    matrix = expand_matrix(config)
    assert len(matrix) == 8
    assert {row[2] for row in matrix} == {6001}
    assert {row[3] for row in matrix} == {
        "fixed_always_on", "doet_rule",
    }


def test_transformers_planner_applies_declared_llm_seed(monkeypatch):
    calls = []
    torch_module = ModuleType("torch")
    torch_module.manual_seed = lambda value: calls.append(("torch", value))
    torch_module.cuda = SimpleNamespace(
        manual_seed_all=lambda value: calls.append(("cuda", value))
    )
    torch_module.bfloat16 = "bfloat16"

    class FakeTokenizer:
        pad_token_id = 1
        eos_token = "eos"

    class FakeModel:
        def eval(self):
            return self

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )
    transformers_module.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeModel()
    )
    transformers_module.BitsAndBytesConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    planner = TransformersPlanner(
        "model", "revision", load_in_4bit=False, seed=9101
    )
    assert planner.seed == 9101
    assert calls == [("torch", 9101), ("cuda", 9101)]


def test_holdout_generator_requires_and_balances_all_five_training_seeds(tmp_path):
    root = tmp_path / "results"
    for directory in (
        "validation", "manifests", "training", "checkpoints", "protocol",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    trigger = {
        "parameters": {
            "trigger_type": "cusum", "direction": "low", "rho": 0.8,
            "kappa": 0.15, "tau_on": 1.5, "tau_off": 0.5,
            "tau_crisis": 3.0, "minimum_dwell": 2, "cooldown": 2,
            "propagation": "local",
        }
    }
    (root / "validation" / "trigger_selection.json").write_text(
        json.dumps({
            "selected_method_variant": "test", "selected_trigger": trigger,
        }), encoding="utf-8",
    )
    (root / "validation" / "budget_matched_controls.json").write_text(
        json.dumps({
            "periodic_interval": 8,
            "random_gate_probability": 0.5,
            "kpi_trigger": {
                "normalizers_path": (
                    "results/entropy_triggered_v2/calibration/"
                    "trigger_nominal_calibration.json"
                ),
                "normalizers_key": "kpi_normalizers",
                "parameters": {
                    "trigger_type": "cusum", "direction": "high",
                    "signal_scale": 1.0,
                },
            },
        }),
        encoding="utf-8",
    )
    pairs = pd.DataFrame([
        {"application": application, "scenario_name": scenario,
         "relative_degradation": 0.0}
        for application in ("commercial", "humanitarian")
        for scenario in ("isolated", "correlated", "compound_partition")
        for _ in range(4)
    ])
    pairs.to_csv(root / "validation" / "selected_trigger_pairs.csv", index=False)
    (root / "manifests" / "validation_sweep.json").write_text(
        json.dumps({
            "planner_backend": "transformers", "episodes_failed": 0,
            "episodes_complete": 288,
            "wall_clock_seconds_including_model_load": 3600.0,
            "cumulative_episode_single_gpu_hours": 1.0,
        }), encoding="utf-8",
    )
    (root / "manifests" / "profile_v2_sweep.json").write_text(
        json.dumps({
            "planner_backend": "transformers", "episodes_failed": 0,
            "episodes_complete": 8,
            "wall_clock_seconds_including_model_load": 600.0,
            "cumulative_episode_single_gpu_hours": 0.1,
        }), encoding="utf-8",
    )
    setup_dir = root / "logs" / "setup"
    setup_dir.mkdir(parents=True)
    (setup_dir / "model_smoke.json").write_text(
        json.dumps({
            "status": "complete", "load_seconds": 60.0,
            "batched_inference_seconds": 2.0,
        }), encoding="utf-8",
    )
    seed_rows = []
    for variant in ("no_entropy", "thermo", "doet_rl"):
        for seed in (7301, 7302, 7303, 7304, 7305):
            relative = Path("checkpoints") / f"{variant}_{seed}.pt"
            checkpoint = root / relative
            checkpoint.write_bytes(f"{variant}:{seed}".encode())
            seed_rows.append({
                "variant": variant, "rl_training_seed": seed,
                "status": "complete", "failure_reason": "",
                "checkpoint": str(relative),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            })
    pd.DataFrame(seed_rows).to_csv(
        root / "training" / "seed_manifest.csv", index=False
    )
    (root / "training" / "training_manifest.json").write_text(
        json.dumps({
            "status": "complete", "completed_trainings": 15,
            "single_gpu_hours_reserved": 1.5,
        }), encoding="utf-8",
    )
    config_path = tmp_path / "holdout.yaml"
    record = design_doet_holdout(root, config_path)
    assert record["episode_count"] == 1296
    assert record["base_scenario_panels"] == 144
    for counts in record["learned_assignment_counts"].values():
        assert set(counts) == {7301, 7302, 7303, 7304, 7305}
        assert max(counts.values()) - min(counts.values()) <= 1
    (root / "manifests" / "holdout_locked_sweep.json").write_text(
        json.dumps({
            "wall_clock_seconds_including_model_load": 7200.0,
            "cumulative_episode_single_gpu_hours": 2.0,
        }), encoding="utf-8",
    )
    (root / "reproducibility").mkdir()
    (root / "reproducibility" / "execution_source.json").write_text(
        json.dumps({"source_checksum": "test"}), encoding="utf-8"
    )
    ablation = design_doet_ablations(
        root, root / "protocol" / "extended_ablation_config.yaml"
    )
    assert ablation["episode_count"] == 96
    assert ablation["authorized"]
