from dataclasses import replace

import pytest

from thermoagent.doet import (
    CommunicationMode,
    DistributedEntropyTrigger,
    TriggerConfig,
)
from thermoagent.environment import ScenarioConfig
from thermoagent.planners import MockPlanner, PlannerRequest, validate_request_plan
from thermoagent.runner import EpisodeRunner
from thermoagent.policy import CoordinationPolicy
from thermoagent.experiments import expand_matrix
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
