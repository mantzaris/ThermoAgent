from dataclasses import replace

import pytest

from thermoagent.agents import PrivacyViolation
from thermoagent.v7_experiments import evaluator_counterfactual, make_environment, run_episode
from thermoagent.v7_policies import V7SelectiveController
from thermoagent.v7_types import V7StructuredDecision


@pytest.mark.parametrize(
    "application,topology",
    [("humanitarian", "small_world"), ("utility_restoration", "grid")],
)
def test_v7_persistent_agents_control_multiple_assets_and_keep_private_vaults(
    application, topology,
):
    environment = make_environment(
        application, "small", "medium", "high", "medium", topology, 77101,
    )
    assert len(environment.agents) == 12
    assert all(len(agent.identity.asset_scope) >= 2 for agent in environment.agents.values())
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    first, second = sorted(environment.agents)[:2]
    assert environment.agents[first].vault is not environment.agents[second].vault
    asset = environment.agents[first].identity.asset_scope[0]
    with pytest.raises(PrivacyViolation):
        environment.agents[first].vault.observation(second, asset)


@pytest.mark.parametrize(
    "application,topology,resource",
    [
        ("humanitarian", "random_geometric", "water"),
        ("utility_restoration", "scale_free", "spares"),
    ],
)
def test_v7_real_conservation_and_deliberate_fault_detection(
    application, topology, resource,
):
    environment = make_environment(
        application, "small", "high", "high", "high", topology, 77102,
    )
    for step in range(5):
        environment.advance_domain(step)
    assert environment.conservation_report()["feasible"]
    environment.inject_conservation_fault_for_test(resource, 0.25)
    report = environment.conservation_report()
    assert not report["feasible"]
    assert report["maximum_residual"] == pytest.approx(0.25)


def test_v7_action_schema_separates_physical_information_communication_delegation():
    environment = make_environment(
        "humanitarian", "small", "high", "high", "medium",
        "modular", 77103,
    )
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    agent = environment.agents[sorted(environment.agents)[0]]
    asset = agent.identity.asset_scope[0]
    decision = agent.propose(asset)
    assert decision.proposal.proposed_operational_action
    assert decision.information_action
    assert decision.communication_action
    assert decision.delegation_action
    if decision.information_action != "no_information_action":
        assert decision.proposal.proposed_operational_action != decision.information_action
    context = environment.risk_context(decision, 0).deployable()
    assert "evaluator_distributed_estimation_error" not in repr(context)
    assert environment.evaluator_estimation_errors


def test_v7_commitments_are_independently_accepted_countered_or_rejected():
    environment = make_environment(
        "humanitarian", "small", "high", "high", "medium", "modular", 77113,
    )
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    environment.process_commitments(0)
    statuses = {value.status for value in environment.commitments.values()}
    assert statuses.issubset({"accepted", "countered", "rejected"})
    assert statuses
    first = next(iter(environment.commitments.values()))
    assert environment.agents[first.proposer].commitments is not environment.agents[first.recipient].commitments


def test_v7_partition_blocks_delivery_and_counts_drop():
    environment = make_environment(
        "utility_restoration", "small", "high", "high", "high",
        "modular", 77104,
    )
    first_node, second_node = next(iter(environment.communication_graph.edges()))
    first = environment.node_agents[first_node]
    second = environment.node_agents[second_node]
    environment.communication_graph.edges[first_node, second_node]["available"] = False
    delivered = environment.send_message(first, second, "test", {"value": 1}, 0)
    assert not delivered
    assert environment.dropped_messages == 1
    assert not environment.agents[second].inbox


def test_v7_only_explicitly_delivered_evidence_changes_peer_belief():
    environment = make_environment(
        "humanitarian", "small", "high", "high", "low", "small_world", 77114,
    )
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    pair = None
    for first_node, second_node in environment.communication_graph.edges():
        first = environment.node_agents[first_node]
        second = environment.node_agents[second_node]
        shared = sorted(
            set(environment.agents[first].identity.asset_scope)
            & set(environment.agents[second].identity.asset_scope)
        )
        if shared:
            pair = (first, second, shared[0])
            break
    assert pair is not None
    first, second, asset = pair
    before = environment.agents[second].private_beliefs[asset]
    environment.send_message(
        first, second, "send_targeted_summary",
        {
            "target": asset,
            "belief_distribution": list(environment.agents[first].private_beliefs[asset]),
        },
        0,
    )
    delivery_step = min(environment.pending_messages)
    environment.deliver_messages(delivery_step)
    after = environment.agents[second].private_beliefs[asset]
    assert before != after


def test_v7_bounded_gossip_increases_distributed_contributors_and_counts_hops():
    environment = make_environment(
        "utility_restoration", "small", "high", "high", "low", "modular", 77115,
    )
    for step in range(8):
        environment.deliver_messages(step)
        environment.advance_domain(step)
        if step in environment.spec.decision_steps:
            environment.deliver_private_observations(step)
            environment.exchange_entropy_sketches(step)
    environment.deliver_messages(8)
    maximum = 1
    for agent_id, agent in environment.agents.items():
        for asset in agent.identity.asset_scope:
            state = environment.distributed_state(agent_id, asset, 8)
            maximum = max(maximum, len(state.contributors))
    assert maximum >= 2
    assert environment.sketch_messages > 0
    assert any(
        event.kind == "v7_entropy_sketch" and event.payload.get("hop_count", 0) > 0
        for event in environment.ledger.events
    )


def test_v7_dynamic_counterfactual_uses_identical_stochastic_tape():
    environment = make_environment(
        "utility_restoration", "small", "high", "high", "medium",
        "grid", 77105,
    )
    step = environment.disruption_step
    for current in range(step + 1):
        environment.deliver_messages(current)
        environment.advance_domain(current)
    environment.deliver_private_observations(step)
    agent = environment.agents[sorted(environment.agents)[0]]
    asset = agent.identity.asset_scope[0]
    decision = replace(agent.propose(asset), delegation_action="execute_autonomously")
    result = evaluator_counterfactual(environment, decision, step, 6)
    assert result["stochastic_tape_digest_action"] == result["stochastic_tape_digest_no_action"]
    assert isinstance(result["causal_utility"], float)


@pytest.mark.parametrize(
    "application,topology",
    [("humanitarian", "modular"), ("utility_restoration", "grid")],
)
def test_v7_end_to_end_episode_replays_dynamic_transitions(application, topology):
    output = run_episode(
        application, "small", "high", "high", "medium", topology, 77106,
        V7SelectiveController("combined_generalized_entropic", 0.60),
        counterfactual_limit_per_epoch=1,
    )
    summary = output["summary"]
    assert summary["horizon"] == 30
    assert summary["agent_count"] == 12
    assert summary["event_count"] > 100
    assert summary["maximum_conservation_residual"] <= 1e-9
    assert summary["privacy_boundary_pass"]
    assert summary["total_messages"] >= summary["sketch_messages"]
    assert output["event_ledger_digest"]


def test_v7_stored_episode_replays_and_reconstructs_conservation(tmp_path):
    from thermoagent.v7_replay import replay_all

    root = tmp_path / "v7"
    run_episode(
        "humanitarian", "small", "high", "high", "medium", "modular", 77116,
        V7SelectiveController("kpi_confidence", 0.60),
        results_root=root, stage="pilot", counterfactual_limit_per_epoch=1,
    )
    report = replay_all(root)
    assert report["episodes_replayed"] == 1
    assert report["replay_mismatches"] == 0
    assert report["maximum_conservation_residual"] <= 1e-9


def test_v7_structurally_held_out_topologies_are_executable():
    humanitarian = make_environment(
        "humanitarian", "small", "high", "high", "high", "chain", 77117,
    )
    utility = make_environment(
        "utility_restoration", "small", "high", "high", "high",
        "small_world", 77118,
    )
    assert humanitarian.operational_graph.number_of_edges() == 7
    assert utility.physical_graph.number_of_edges() > 7


def test_v7_utility_roles_have_bounded_domain_authority():
    environment = make_environment(
        "utility_restoration", "small", "high", "high", "high",
        "modular", 77119,
    )
    by_role = {
        agent.identity.role: set(agent.identity.physical_authority)
        for agent in environment.agents.values()
    }
    assert by_role["communications"] == {
        "no_operational_action", "reconfigure_service_edge",
        "restore_communication_relay",
    }
    assert "isolate_component" not in by_role["resource_allocation"]
    assert "dispatch_repair_crew" not in by_role["zone_operator"]
    assert "deploy_mobile_generation" in by_role["critical_load"]


def test_v7_defensive_isolation_expires_after_bounded_dwell():
    environment = make_environment(
        "utility_restoration", "small", "high", "high", "high",
        "modular", 77120,
    )
    target = sorted(environment.node_role)[-1]
    environment.isolated[target] = True
    environment.isolation_release_step[target] = 4
    for step in range(5):
        environment.advance_domain(step)
    assert not environment.isolated[target]
    assert target not in environment.isolation_release_step
    assert any(
        event.kind == "v7_domain_transition"
        and event.payload.get("kind") == "bounded_defensive_isolation_expired"
        and event.payload.get("asset") == target
        for event in environment.ledger.events
    )


def test_v7_operational_and_thermodynamic_communication_are_separate():
    environment = make_environment(
        "humanitarian", "small", "high", "high", "medium", "modular",
        77121, sketch_policy="always_on",
        operational_communication_policy="none",
    )
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    environment.exchange_entropy_sketches(0)
    agent = environment.agents[sorted(environment.agents)[0]]
    asset = agent.identity.asset_scope[0]
    decision = agent.propose(asset)
    environment.validate_and_schedule(decision, 0)
    assert environment.sketch_messages > 0
    assert environment.operational_messages == 0


def test_v7_kpi_communication_trigger_uses_local_change_not_period_zero():
    environment = make_environment(
        "utility_restoration", "small", "high", "high", "medium", "grid",
        77122, operational_communication_policy="kpi_event_triggered",
    )
    agent_id = sorted(environment.agents)[0]
    asset = environment.agents[agent_id].identity.asset_scope[0]
    first = environment.operational_communication_action(
        "send_targeted_summary", agent_id, asset, 0.20, 0,
    )
    stable = environment.operational_communication_action(
        "send_targeted_summary", agent_id, asset, 0.25, 3,
    )
    changed = environment.operational_communication_action(
        "send_targeted_summary", agent_id, asset, 0.50, 6,
    )
    assert first == "no_communication_action"
    assert stable == "no_communication_action"
    assert changed == "send_targeted_summary"
