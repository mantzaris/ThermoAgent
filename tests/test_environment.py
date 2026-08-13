from pathlib import Path

import numpy as np
import pytest

from thermoagent.environment import LogisticsEnvironment, ScenarioConfig
from thermoagent.events import EventLedger
from thermoagent.types import PlanOutput


@pytest.mark.parametrize("application,n_agents", [("commercial", 8), ("humanitarian", 8)])
def test_deterministic_resource_conservation(application, n_agents):
    config = ScenarioConfig(application=application, seed=11, horizon=12, n_agents=n_agents, disruption="compound")
    env = LogisticsEnvironment(config)
    for _ in range(config.horizon):
        env.transition()
        env.deliver_observations()
        env.advance()
        assert abs(env.conservation_error()) < 1e-8


def test_shipments_validate_inventory_route_and_arrival():
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=1, n_agents=8))
    source, target = next(iter(env.physical_edges))
    too_much = env.states[source].inventory + 1.0
    result = env.execute_tool(source, "schedule_shipment", {"target": target, "quantity": too_much, "arrival_step": 2})
    assert not result.ok and result.code == "insufficient_inventory"
    quantity = min(2.0, env.states[source].inventory)
    before = env.total_material()
    result = env.execute_tool(source, "schedule_shipment", {"target": target, "quantity": quantity, "arrival_step": 2})
    assert result.ok
    assert np.isclose(env.total_material(), before)


def test_dispatch_respects_period_handling_capacity():
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=2, n_agents=8))
    source, target = next(iter(env.physical_edges))
    quantity = min(env.states[source].inventory, env.states[source].capacity + 0.5)
    if quantity <= env.states[source].capacity:
        env.states[source].inventory = env.states[source].capacity + 1.0
        quantity = env.states[source].capacity + 0.5
    result = env.execute_tool(source, "schedule_shipment", {
        "target": target, "quantity": float(quantity), "arrival_step": 2,
    })
    assert result.code == "handling_capacity_exceeded"


@pytest.mark.parametrize(
    "application,transport_role,demand_roles,tool",
    [
        ("commercial", "carrier", {"retailer"}, "schedule_shipment"),
        ("humanitarian", "transport", {"clinic", "community"}, "transfer_resource"),
    ],
)
def test_transport_organizations_have_executable_outbound_routes(
    application, transport_role, demand_roles, tool
):
    env = LogisticsEnvironment(ScenarioConfig(application=application, seed=5, n_agents=8))
    transport = next(
        agent_id for agent_id, agent in env.agents.items()
        if agent.identity.role == transport_role
    )
    destinations = [
        target for source, target in env.physical_edges
        if source == transport and env.agents[target].identity.role in demand_roles
    ]
    assert destinations
    result = env.execute_tool(transport, tool, {
        "target": destinations[0],
        "quantity": min(1.0, env.states[transport].inventory, env.states[transport].capacity),
        "arrival_step": 1,
    })
    assert result.ok


def test_dynamic_deadline_and_self_target_validation():
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=31, n_agents=8))
    source = env.agent_ids[0]
    self_quote = env.execute_tool(source, "request_quote", {"target": source, "quantity": 2.0, "due_step": 2})
    assert not self_quote.ok and self_quote.code == "self_target"
    target = env.agent_ids[1]
    late = env.execute_tool(source, "request_quote", {"target": target, "quantity": 2.0, "due_step": 500})
    assert not late.ok and late.code == "invalid_deadline"


def test_offer_rejects_unknown_recipient_without_creating_commitment():
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=32, n_agents=8))
    source = env.agent_ids[0]
    before = set(env.commitments)
    result = env.execute_tool(source, "submit_offer", {
        "target": "unknown_agent", "quantity": 1.0,
        "unit_price": 1.0, "due_step": env.step_index + 3,
    })
    assert not result.ok and result.code == "invalid_recipient"
    assert set(env.commitments) == before


def test_completed_delivery_can_only_be_verified_by_shipment_parties():
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=33, n_agents=8))
    carrier = next(
        agent_id for agent_id, agent in env.agents.items()
        if agent.identity.role == "carrier"
    )
    target = next(
        destination for source, destination in env.physical_edges
        if source == carrier
    )
    scheduled = env.execute_tool(carrier, "schedule_shipment", {
        "target": target, "quantity": 1.0, "arrival_step": 1,
    })
    assert scheduled.ok
    shipment_id = scheduled.data["shipment_id"]
    env.advance()
    env.transition()
    assert shipment_id in env.completed_shipments
    verified = env.execute_tool(target, "verify_delivery", {
        "shipment_id": shipment_id,
    })
    assert verified.ok and verified.data["delivered"]
    outsider = next(
        agent_id for agent_id in env.agent_ids
        if agent_id not in (carrier, target)
    )
    denied = env.execute_tool(outsider, "verify_delivery", {
        "shipment_id": shipment_id,
    })
    assert denied.code == "shipment_privacy"
    alter = env.execute_tool(carrier, "expedite_shipment", {
        "shipment_id": shipment_id,
    })
    assert alter.code == "shipment_completed"


def test_only_dispatching_transport_can_reroute_over_a_valid_route():
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=34, n_agents=8))
    carriers = [
        agent_id for agent_id, agent in env.agents.items()
        if agent.identity.role == "carrier"
    ]
    sender = carriers[0]
    targets = [
        destination for source, destination in env.physical_edges
        if source == sender
    ]
    scheduled = env.execute_tool(sender, "schedule_shipment", {
        "target": targets[0], "quantity": 1.0, "arrival_step": 2,
    })
    shipment_id = scheduled.data["shipment_id"]
    denied = env.execute_tool(carriers[1], "reroute_shipment", {
        "shipment_id": shipment_id, "new_target": targets[0],
    })
    assert denied.code == "shipment_authority"
    invalid_target = next(
        agent_id for agent_id in env.agent_ids
        if (sender, agent_id) not in env.physical_edges and agent_id != sender
    )
    no_route = env.execute_tool(sender, "reroute_shipment", {
        "shipment_id": shipment_id, "new_target": invalid_target,
    })
    assert no_route.code == "no_route"


def test_deterministic_replay_digest_for_same_seed():
    digests = []
    states = []
    for _ in range(2):
        env = LogisticsEnvironment(ScenarioConfig(application="humanitarian", seed=22, horizon=8, n_agents=8))
        for __ in range(8):
            env.transition()
            env.deliver_observations()
            env.advance()
        digests.append(env.ledger.digest())
        states.append(env.full_state_for_evaluator())
    assert digests[0] == digests[1]
    assert states[0] == states[1]


def test_event_ledger_roundtrip(tmp_path: Path):
    ledger = EventLedger()
    ledger.append(0, "disruption", "simulator", {"severity": 0.5})
    ledger.append(1, "metric", "evaluator", {"service": 0.9}, private_to="agent_a")
    path = tmp_path / "events.jsonl"
    ledger.write_jsonl(path)
    restored = EventLedger.read_jsonl(path)
    assert restored.digest() == ledger.digest()
    assert len(restored.visible_to("agent_b")) == 1


def test_compressed_event_ledger_roundtrip(tmp_path: Path):
    ledger = EventLedger()
    ledger.append(0, "message", "agent_a", {"recipient": "agent_b"}, private_to="agent_b")
    path = tmp_path / "events.jsonl.gz"
    ledger.write_jsonl(path)
    assert EventLedger.read_jsonl(path).digest() == ledger.digest()


def test_environment_never_invents_domain_action():
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=3, n_agents=8))
    before = env.full_state_for_evaluator()
    # Observation delivery and free-form prose have no mutation entry point.
    env.deliver_observations()
    after = env.full_state_for_evaluator()
    assert before["states"] == after["states"]
    tool_events = [e for e in env.ledger.events if e.kind == "tool_call"]
    assert tool_events == []


def test_holdout_topology_changes_real_connectivity_and_keeps_demands_reachable():
    default = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=4, n_agents=9))
    holdout = LogisticsEnvironment(ScenarioConfig(
        application="commercial", seed=4, n_agents=9, topology="holdout_nine_agent"
    ))
    assert default.initial_communication_edges != holdout.initial_communication_edges
    assert default.initial_physical_edges != holdout.initial_physical_edges
    demands = [
        agent_id for agent_id, agent in holdout.agents.items()
        if agent.identity.role == "retailer"
    ]
    assert all(any(target == demand for _, target in holdout.physical_edges) for demand in demands)


def test_compound_shock_closes_routes_and_enforces_observed_lead_time():
    config = ScenarioConfig(application="commercial", seed=8, horizon=9, n_agents=9, disruption="compound")
    env = LogisticsEnvironment(config)
    initial_edges = len(env.physical_edges)
    for _ in range(4):
        env.transition()
        env.deliver_observations()
        env.advance()
    assert env._disruption_applied
    assert env.closed_physical_edges
    assert len(env.physical_edges) < initial_edges
    assert env.route_lead_time_penalty == 2
    source, target = next(iter(env.physical_edges))
    result = env.execute_tool(source, "schedule_shipment", {
        "target": target,
        "quantity": min(1.0, env.states[source].inventory),
        "arrival_step": env.step_index + 1,
    })
    assert result.code == "lead_time_infeasible"


def test_humanitarian_compound_shock_removes_coordinator_and_depot_capacity():
    env = LogisticsEnvironment(ScenarioConfig(
        application="humanitarian", seed=9, horizon=9, n_agents=9,
        disruption="compound",
    ))
    for _ in range(4):
        env.transition()
        env.deliver_observations()
        env.advance()
    agency = next(agent_id for agent_id, agent in env.agents.items() if agent.identity.role == "agency")
    depot = next(agent_id for agent_id, agent in env.agents.items() if agent.identity.role == "depot")
    assert env.states[agency].capacity == 0.0
    assert env.states[depot].capacity == 0.0
    assert all(agency not in edge for edge in env.communication_edges)


def test_action_dependent_message_draws_do_not_shift_exogenous_trajectories():
    config = ScenarioConfig(
        application="commercial",
        seed=27,
        horizon=8,
        n_agents=8,
        communication="intermittent",
        disruption="moderate",
    )
    messaging = LogisticsEnvironment(config)
    quiet = LogisticsEnvironment(config)
    sender, recipient = messaging.agent_ids[:2]
    for step in range(config.horizon):
        if step < 3:
            messaging._send(sender, recipient, "information_request", {"topic": "capacity"})
            messaging._send(recipient, sender, "information_request", {"topic": "need"})
        messaging.transition()
        quiet.transition()
        messaging.deliver_observations()
        quiet.deliver_observations()
        for agent_id in messaging.agent_ids:
            left = messaging.states[agent_id]
            right = quiet.states[agent_id]
            assert left.demand == right.demand
            assert left.cumulative_demand == right.cumulative_demand
            # Messages do not move material, so production and fulfillment are
            # also paired when no operational tool is called.
            assert left.inventory == right.inventory
        messaging.advance()
        quiet.advance()


def test_partition_switches_message_and_sketch_graph_at_same_onset():
    env = LogisticsEnvironment(ScenarioConfig(
        application="commercial", seed=44, horizon=12, n_agents=8,
        communication="partition",
    ))
    first, last = env.agent_ids[0], env.agent_ids[-1]
    cross_edge = tuple(sorted((first, last)))
    env.communication_edges.add(cross_edge)
    onset = max(2, env.config.horizon // 3)
    env.step_index = onset - 1
    assert env._communication_probability(first, last) == 0.98
    assert cross_edge in env.active_communication_edges()
    env.step_index = onset
    assert env._communication_probability(first, last) == 0.0
    assert cross_edge not in env.active_communication_edges()
    same_side = tuple(sorted((env.agent_ids[0], env.agent_ids[1])))
    env.communication_edges.add(same_side)
    assert env._communication_probability(*same_side) == 0.85
    assert same_side in env.active_communication_edges()


def test_privacy_factor_changes_visibility_not_underlying_state_or_local_quality():
    shared = LogisticsEnvironment(ScenarioConfig(
        application="commercial", seed=53, horizon=8, n_agents=8,
        private_information=0.0,
    ))
    private = LogisticsEnvironment(ScenarioConfig(
        application="commercial", seed=53, horizon=8, n_agents=8,
        private_information=1.0,
    ))
    for agent_id in shared.agent_ids:
        assert vars(shared.states[agent_id]) == vars(private.states[agent_id])
        assert shared.agents[agent_id].utility == private.agents[agent_id].utility
        assert shared.private_observation(agent_id) == private.private_observation(agent_id)
    assert all("shared_operational_state" in row for row in shared.public_identities())
    assert all("shared_operational_state" not in row for row in private.public_identities())
