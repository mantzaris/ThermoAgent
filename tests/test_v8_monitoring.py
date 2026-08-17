from dataclasses import replace

from thermoagent.v7_experiments import make_environment
from thermoagent.v7_agents import HUMANITARIAN_MODE_ACTION
from thermoagent.v8_experiments import run_v8_episode
from thermoagent.v8_experiments import authorized_distributed_estimate
from thermoagent.v8_monitoring import V8BeliefNetwork
from thermoagent.v8_trigger import TriggerConfig


def _prepared_environment(application="humanitarian", topology="small_world", seed=88001):
    environment = make_environment(
        application, "small", "high", "high", "low", topology, seed,
        sketch_policy="none", operational_communication_policy="none",
    )
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    return environment


def test_v8_network_uses_exact_wire_bytes_and_updates_only_delivered_recipient():
    environment = _prepared_environment()
    network = V8BeliefNetwork(
        environment, TriggerConfig(method="always_on"), encoding="fp16",
        maximum_hops=0,
    )
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
    observation = environment.agents[first].vault.observation(first, asset)
    transmitted = network._send(
        origin=first, transmitter=first, recipient=second, asset=asset,
        belief=environment.agents[first].private_beliefs[asset],
        confidence=observation.telemetry_confidence,
        original_sent_step=0, transmission_step=0, hop_count=0,
    )
    assert transmitted
    assert network.total_on_wire_bytes == network.header_bytes + network.payload_bytes + network.integrity_bytes
    delivery_step = min(network.pending)
    network.deliver(delivery_step)
    after = environment.agents[second].private_beliefs[asset]
    assert before != after
    assert network.delivered_messages == 1
    assert network.action_belief_updates == 1


def test_v8_partition_blocks_wire_transmission_and_recovery_forces_refresh():
    environment = _prepared_environment("utility_restoration", "modular", 88002)
    network = V8BeliefNetwork(
        environment,
        TriggerConfig(method="generalized_information", maximum_silence_steps=8),
        maximum_hops=0,
    )
    agent_id = sorted(environment.agents)[0]
    node = environment.agent_nodes[agent_id]
    neighbors = list(environment.communication_graph.neighbors(node))
    assert neighbors
    network.exchange(0)
    focal_transmissions_before = sum(
        count for (sender, _), count in network.edge_transmissions.items()
        if sender == agent_id
    )
    assert focal_transmissions_before > 0
    for neighbor in neighbors:
        environment.communication_graph.edges[node, neighbor]["available"] = False
    network.exchange(environment.spec.decision_interval)
    assert network.attempted_messages > 0
    assert sum(
        count for (sender, _), count in network.edge_transmissions.items()
        if sender == agent_id
    ) == focal_transmissions_before
    for neighbor in neighbors:
        environment.communication_graph.edges[node, neighbor]["available"] = True
    # Other connected agents can transmit too; the focal agent's logged reason
    # proves that the sender-local recovery behavior was exercised.
    network.exchange(2 * environment.spec.decision_interval)
    rows = [row for row in network.trigger_rows if row["sender"] == agent_id]
    assert rows[-1]["reason"] == "partition_recovery"
    assert rows[-1]["transmit"]


def test_v8_distributed_estimator_uses_delivered_cache_not_global_vaults():
    environment = _prepared_environment(seed=88003)
    network = V8BeliefNetwork(
        environment, TriggerConfig(method="always_on"), maximum_hops=1,
    )
    agent_id = sorted(environment.agents)[0]
    asset = sorted(environment.agents[agent_id].private_beliefs)[0]
    before = network.distributed_estimate(agent_id, asset, 0)
    assert before["contributors"] == 1
    network.exchange(0)
    for step in range(1, 6):
        network.deliver(step)
    after = network.distributed_estimate(agent_id, asset, 6)
    assert after["contributors"] >= 1
    assert after["contributors"] <= after["scoped_agents"] + len(environment.agents)
    assert 0.0 <= after["belief_mae"] <= 1.0
    assert all(event.private_to is not None for event in environment.ledger.events if event.kind == "v8_distributed_estimate")


def test_v8_policy_boundary_strips_every_evaluator_scoring_field():
    estimate = {
        "step": 3, "recipient": "a", "asset": "x", "contributors": 2,
        "scoped_agents": 3, "missing_agents": 1, "maximum_age": 4,
        "mean_age": 2.0, "distributed_pooled_belief": (0.6, 0.4),
        "distributed_disagreement": 0.1,
        "distributed_disrupted_probability": 0.4,
        "evaluator_global_pooled_belief": (0.1, 0.9),
        "evaluator_global_disagreement": 0.9,
        "evaluator_true_mode": "hidden",
        "belief_mae": 0.5, "disagreement_absolute_error": 0.8,
    }
    authorized = authorized_distributed_estimate(estimate)
    assert set(authorized) == {
        "step", "recipient", "asset", "contributors", "scoped_agents",
        "missing_agents", "maximum_age", "mean_age",
        "distributed_pooled_belief", "distributed_disagreement",
        "distributed_disrupted_probability",
    }
    assert not any("evaluator" in key for key in authorized)
    assert "belief_mae" not in authorized


def test_v8_full_dynamic_episode_preserves_tape_conservation_and_privacy(tmp_path):
    output = run_v8_episode(
        application="humanitarian",
        complexity="small",
        coupling="high",
        fragmentation="high",
        network_disruption="medium",
        topology_family="small_world",
        environment_seed=88004,
        trigger_config=TriggerConfig(
            method="generalized_information", maximum_silence_steps=6,
        ),
        results_root=tmp_path / "v8",
        stage="smoke",
    )
    summary = output["summary"]
    assert summary["horizon"] == 30
    assert summary["conservation_feasible"]
    assert summary["maximum_conservation_residual"] <= 1e-9
    assert summary["privacy_boundary_pass"]
    assert summary["transmitted_sketch_messages"] > 0
    assert summary["sketch_on_wire_bytes"] == (
        summary["sketch_header_bytes"]
        + summary["sketch_payload_bytes"]
        + summary["sketch_integrity_bytes"]
    )
    assert summary["event_count"] > 100
    assert output["event_ledger_sha256"]


def test_delivered_sketch_can_change_proposal_and_reach_later_domain_state():
    environment = _prepared_environment(seed=88005)
    network = V8BeliefNetwork(
        environment, TriggerConfig(method="always_on"), encoding="uint8_simplex",
        maximum_hops=0,
    )
    selected = None
    for first_node, second_node in environment.communication_graph.edges():
        sender = environment.node_agents[first_node]
        recipient = environment.node_agents[second_node]
        shared = sorted(
            set(environment.agents[sender].identity.asset_scope)
            & set(environment.agents[recipient].identity.asset_scope)
        )
        for asset in shared:
            agent = environment.agents[recipient]
            observation = agent.vault.observation(recipient, asset)
            feasible = set(observation.feasible_physical_actions) & set(
                agent.identity.physical_authority
            )
            modes = tuple(HUMANITARIAN_MODE_ACTION)
            alternatives = [
                index for index, mode in enumerate(modes)
                if HUMANITARIAN_MODE_ACTION[mode] in feasible
                and HUMANITARIAN_MODE_ACTION[mode] != "no_operational_action"
            ]
            if alternatives:
                selected = (sender, recipient, asset, alternatives[0], modes)
                break
        if selected is not None:
            break
    assert selected is not None
    sender, recipient, asset, action_index, modes = selected
    receiver = environment.agents[recipient]
    nominal_index = modes.index("nominal")
    baseline = [0.15] * len(modes)
    baseline[nominal_index] += 1.0 - sum(baseline)
    receiver.private_beliefs[asset] = tuple(baseline)
    before = receiver.propose(asset)
    evidence = [0.0] * len(modes)
    evidence[action_index] = 1.0
    observation = environment.agents[sender].vault.observation(sender, asset)
    assert network._send(
        origin=sender, transmitter=sender, recipient=recipient, asset=asset,
        belief=evidence, confidence=observation.telemetry_confidence,
        original_sent_step=0, transmission_step=0, hop_count=0,
    )
    network.deliver(min(network.pending))
    after = receiver.propose(asset)
    assert before.proposal.proposed_operational_action != after.proposal.proposed_operational_action
    assert after.proposal.is_physical
    result = environment.validate_and_schedule(
        replace(after, delegation_action="execute_autonomously"), step=1,
    )
    assert result["accepted_physical_action"]
    chain_id = result["causal_chain_id"]
    for step in range(2, environment.spec.horizon + 1):
        environment.advance_domain(step)
    completed = [
        value for value in environment.completed_actions
        if value.get("chain_id") == chain_id
    ]
    assert completed
    assert completed[0]["causal_effect"] != 0.0
