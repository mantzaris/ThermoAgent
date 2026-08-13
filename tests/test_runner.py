import pytest

from thermoagent.environment import ScenarioConfig
from thermoagent.planners import PlannerResponse
from thermoagent.policy import CoordinationPolicy
from thermoagent.runner import EpisodeRunner
from thermoagent.types import PlanOutput


@pytest.mark.parametrize("application,n_agents", [("commercial", 8), ("humanitarian", 8)])
def test_short_episode_runs_and_conserves(application, n_agents):
    config = ScenarioConfig(application=application, seed=5, horizon=8, n_agents=n_agents, disruption="moderate", decision_interval=2)
    result = EpisodeRunner(config, "scripted_independent").run()
    assert result.completion_status == "complete"
    assert len(result.time_series) == 8
    assert abs(result.metrics["conservation_error"]) < 1e-8
    assert result.metrics["primary_outcome"] >= 0
    assert all("exact_energy_sensitivity_balanced" in row for row in result.time_series)
    assert all("exact_free_energy_sensitivity_delay_commitment_heavy" in row for row in result.time_series)
    assert all("delayed_entropy_mean" in row for row in result.time_series)
    assert all("noisy_free_energy_mean" in row for row in result.time_series)
    assert all(
        len(row["surprisal_ranked_agents"].split(";")) == n_agents
        for row in result.time_series
    )
    assert 0.0 <= result.metrics["message_delivery_rate"] <= 1.0
    assert 0.0 <= result.metrics["inventory_efficiency"] <= 1.0 + 1e-9
    assert 0.0 <= result.agent_metrics["valid_tool_call_rate"] <= 1.0
    assert "minimum_agent_utility" in result.metrics
    if application == "humanitarian":
        assert result.metrics["cumulative_unmet_weighted_need"] >= sum(
            row["backlog"] for row in result.time_series
        )


def test_learned_policy_never_receives_evaluator_state():
    config = ScenarioConfig(application="commercial", seed=6, horizon=5, n_agents=8, decision_interval=2)
    policy = CoordinationPolicy(seed=6)
    runner = EpisodeRunner(config, "thermoagent", policy=policy)
    result = runner.run()
    assert result.trajectory
    assert all(len(row["observation"]) == 24 for row in result.trajectory)
    assert all("states" not in row for row in result.trajectory)
    assert all("action_mask" in row for row in result.trajectory)
    sketches = [
        event for event in runner.env.ledger.events
        if event.kind == "macrostate_sketch"
    ]
    assert sketches
    assert all(event.private_to == event.actor for event in sketches)
    assert all("exact_entropy" not in event.payload for event in sketches)
    assert all("global" not in event.payload for event in sketches)


@pytest.mark.parametrize(
    "method",
    [
        "learned_no_entropy", "autonomous_no_comm",
        "autonomous_fixed_comm", "random_gate",
    ],
)
def test_no_monitor_controls_receive_zero_monitor_features(method):
    config = ScenarioConfig(
        application="commercial", seed=61, horizon=3, n_agents=8,
        decision_interval=1, random_gate_probability=0.0,
    )
    policy = CoordinationPolicy(seed=61) if method == "learned_no_entropy" else None
    runner = EpisodeRunner(config, method, policy=policy)
    result = runner.run()
    if result.trajectory:
        assert all(all(value == 0.0 for value in row["observation"][16:22]) for row in result.trajectory)
    assert all(
        agent.entropy.local_entropy == 0.0
        and agent.entropy.local_free_energy == 0.0
        and agent.entropy.interaction_entropy == 0.0
        and agent.entropy.consensus_error == 0.0
        for agent in runner.env.agents.values()
    )
    assert result.metrics["monitor_sketch_messages"] == 0
    assert result.metrics["monitor_sketch_bytes"] == 0


def test_distributed_monitor_traffic_is_counted_in_communication_cost():
    config = ScenarioConfig(
        application="commercial", seed=611, horizon=3, n_agents=8,
        decision_interval=1,
    )
    result = EpisodeRunner(
        config, "thermoagent", policy=CoordinationPolicy(seed=611)
    ).run()
    assert result.metrics["monitor_sketch_messages"] > 0
    assert result.metrics["monitor_sketch_bytes"] > 0
    assert result.metrics["total_communication_messages"] == (
        result.metrics["messages"] + result.metrics["monitor_sketch_messages"]
    )
    assert result.metrics["total_communication_bytes"] == (
        result.metrics["message_bytes"] + result.metrics["monitor_sketch_bytes"]
    )


def test_actor_consensus_feature_is_neighbor_residual_not_evaluator_error():
    config = ScenarioConfig(
        application="commercial", seed=62, horizon=6, n_agents=8,
        communication="partition", decision_interval=2,
    )
    runner = EpisodeRunner(config, "thermoagent", policy=CoordinationPolicy(seed=62))
    result = runner.run()
    assert all("mean_local_consensus_residual" in row for row in result.time_series)
    # The evaluator RMSE compares with inaccessible global occupancy, whereas
    # actors get link-local residuals. They are not aliases.
    assert any(
        abs(row["mean_local_consensus_residual"] - row["consensus_rmse"]) > 1e-9
        for row in result.time_series
    )


def test_removing_agent_changes_event_ledger():
    full = EpisodeRunner(ScenarioConfig(application="commercial", seed=7, horizon=4, n_agents=8, decision_interval=2), "scripted_independent")
    small = EpisodeRunner(ScenarioConfig(application="commercial", seed=7, horizon=4, n_agents=7, decision_interval=2), "scripted_independent")
    full.run()
    small.run()
    assert full.env.ledger.digest() != small.env.ledger.digest()
    assert len([e for e in full.env.ledger.events if e.kind == "llm_request"]) != len([e for e in small.env.ledger.events if e.kind == "llm_request"])


def test_coordination_reward_is_credited_after_action_and_done_per_agent():
    config = ScenarioConfig(application="commercial", seed=13, horizon=7, n_agents=8, decision_interval=3)
    policy = CoordinationPolicy(seed=13)
    result = EpisodeRunner(config, "thermoagent", policy=policy, deterministic_policy=False).run()
    assert any(abs(row["reward"]) > 0 for row in result.trajectory)
    for agent_id in {row["agent_id"] for row in result.trajectory}:
        rows = [row for row in result.trajectory if row["agent_id"] == agent_id]
        assert sum(bool(row["done"]) for row in rows) == 1
        assert rows[-1]["done"]


def test_shuffled_monitor_is_causal_and_breaks_current_alignment():
    config = ScenarioConfig(application="commercial", seed=14, horizon=5, n_agents=8, decision_interval=2)
    thermo = EpisodeRunner(config, "thermoagent", policy=CoordinationPolicy(seed=14)).run()
    shuffled = EpisodeRunner(config, "shuffled_entropy", policy=CoordinationPolicy(seed=14)).run()
    # The first shuffled decision can only use the initialized prior-period
    # buffer, whereas ThermoAgent receives the current distributed estimate.
    assert all(row["observation"][17] == 0.0 for row in shuffled.trajectory[:8])
    assert any(row["observation"][17] != 0.0 for row in thermo.trajectory[:8])


def test_random_gate_probability_zero_is_a_silent_matched_control():
    config = ScenarioConfig(
        application="commercial", seed=15, horizon=4, n_agents=8,
        decision_interval=2, random_gate_probability=0.0,
    )
    result = EpisodeRunner(config, "random_gate").run()
    assert result.agent_metrics["option_counts"][8] > 0
    assert sum(value for key, value in result.agent_metrics["option_counts"].items() if key != 8) == 0


def test_random_gate_preserves_mandatory_private_offer_authority():
    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=151, horizon=2, n_agents=8,
            random_gate_probability=0.0,
        ),
        "random_gate",
    )
    runner.env.transition()
    runner.env.deliver_observations()
    seller = next(a for a in runner.env.agent_ids if runner.env.agents[a].identity.role == "supplier")
    buyer = next(a for a in runner.env.agent_ids if runner.env.agents[a].identity.role == "retailer")
    result = runner.env.execute_tool(seller, "submit_offer", {
        "target": buyer, "quantity": 1.0, "unit_price": 1.0,
        "due_step": runner.env.step_index + 3,
    })
    commitment = runner.env.commitments[result.data["commitment_id"]]
    runner.env.agents[buyer].commitments[commitment.commitment_id] = commitment
    assert runner._option(buyer)[0] == 4


def test_offer_response_option_is_masked_without_private_pending_offer():
    runner = EpisodeRunner(
        ScenarioConfig(application="commercial", seed=16, horizon=4, n_agents=8),
        "thermoagent", policy=CoordinationPolicy(seed=16),
    )
    runner.env.transition()
    runner.env.deliver_observations()
    agent_id = runner.env.agent_ids[0]
    mask = runner._local_option_mask(agent_id)
    assert not mask[4]


def test_no_episodic_memory_ablation_preserves_private_state_but_skips_retrieval():
    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=17, horizon=4, n_agents=8,
            decision_interval=2,
        ),
        "no_episodic_memory",
        policy=CoordinationPolicy(seed=17),
    )
    runner.run()
    retrievals = [
        event for event in runner.env.ledger.events
        if event.kind == "memory_retrieval"
    ]
    assert retrievals
    assert all(event.payload["count"] == 0 for event in retrievals)
    assert all(not event.payload["episodic_memory_enabled"] for event in retrievals)
    assert all(event.private_to == event.actor for event in retrievals)


def test_central_llm_cannot_infer_unreported_strongly_private_state():
    private = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=19, horizon=4, n_agents=8,
            private_information=1.0, decision_interval=2,
        ),
        "centralized_llm",
    )
    private.run()
    requests = [
        event for event in private.env.ledger.events
        if event.kind == "llm_request" and event.actor == "central_coordinator"
    ]
    assert requests
    assert all(event.payload["reported_agents"] == 0 for event in requests)
    assert not any(
        event.kind == "tool_call"
        and event.payload.get("tool") in ("schedule_shipment", "transfer_resource")
        for event in private.env.ledger.events
    )


def test_central_llm_fails_closed_on_blind_dispatch_without_public_demand():
    class BlindDispatchPlanner:
        revision = "adversarial-blind-dispatch"

        @staticmethod
        def plan_batch(requests):
            responses = []
            for request in requests:
                source = next(
                    row["agent_id"] for row in request.candidate_agents
                    if row["role"] in {"supplier", "manufacturer", "carrier", "warehouse"}
                )
                target = next(
                    row["agent_id"] for row in request.candidate_agents
                    if row["role"] == "retailer"
                )
                output = PlanOutput(
                    "Guess despite absent reports.",
                    "central_dispatch",
                    {
                        "source": source,
                        "target": target,
                        "quantity": 1.0,
                        "arrival_step": int(request.context["observation"]["step"]) + 2,
                    },
                    "Blind guess.",
                )
                responses.append(PlannerResponse(output, True, raw_text="{}"))
            return responses

    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=191, horizon=1, n_agents=8,
            private_information=1.0, decision_interval=1,
        ),
        "centralized_llm",
        planner=BlindDispatchPlanner(),
    )
    runner.run()
    coordinator_results = [
        event for event in runner.env.ledger.events
        if event.kind == "tool_result" and event.actor == "central_coordinator"
    ]
    assert coordinator_results
    assert coordinator_results[-1].payload["code"] == "coordinator_no_public_demand"
    assert not any(event.kind == "tool_call" for event in runner.env.ledger.events)


def test_central_llm_can_use_explicitly_shared_reports():
    shared = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=20, horizon=4, n_agents=8,
            private_information=0.0, decision_interval=2,
        ),
        "centralized_llm",
    )
    shared.run()
    requests = [
        event for event in shared.env.ledger.events
        if event.kind == "llm_request" and event.actor == "central_coordinator"
    ]
    assert requests
    assert all(event.payload["reported_agents"] == 8 for event in requests)


def test_legal_central_llm_gets_one_typed_slot_per_reported_demand():
    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=201, horizon=1, n_agents=11,
            private_information=0.0, decision_interval=1,
        ),
        "centralized_llm",
    )
    runner.run()
    demands = sum(
        agent.identity.role in {"retailer", "clinic", "community"}
        for agent in runner.env.agents.values()
    )
    requests = [
        event for event in runner.env.ledger.events
        if event.kind == "llm_request" and event.actor == "central_coordinator"
    ]
    assert len(requests) == demands
    assert {event.payload["assigned_target"] for event in requests} == {
        agent_id for agent_id, agent in runner.env.agents.items()
        if agent.identity.role in {"retailer", "clinic", "community"}
    }
    assert all(event.payload["eligible_source_ids"] for event in requests)
    for event in requests:
        assert all(
            (source, event.payload["assigned_target"]) in runner.env.physical_edges
            for source in event.payload["eligible_source_ids"]
        )


def test_central_llm_rejects_source_outside_public_route_assignment():
    class WrongRoutePlanner:
        revision = "adversarial-wrong-public-route"

        def __init__(self):
            self.runner = None

        def plan_batch(self, requests):
            responses = []
            for request in requests:
                target = request.context["coordinator_assignment"]["target"]
                eligible = set(request.context["coordinator_assignment"]["eligible_source_ids"])
                source = next(
                    agent_id for agent_id in self.runner.env.agent_ids
                    if agent_id not in eligible and agent_id != target
                )
                responses.append(PlannerResponse(PlanOutput(
                    "Use an invalid public route.", "central_dispatch",
                    {"source": source, "target": target, "quantity": 1.0, "arrival_step": 2},
                    "Adversarial route choice.",
                ), True, raw_text="{}"))
            return responses

    planner = WrongRoutePlanner()
    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=202, horizon=1, n_agents=11,
            private_information=0.0, decision_interval=1,
        ),
        "centralized_llm",
        planner=planner,
    )
    planner.runner = runner
    runner.run()
    results = [
        event for event in runner.env.ledger.events
        if event.kind == "tool_result" and event.actor == "central_coordinator"
    ]
    assert results
    assert all(event.payload["code"] == "coordinator_source_route" for event in results)
    assert not any(event.kind == "tool_call" for event in runner.env.ledger.events)


def test_full_information_upper_bound_replans_every_period():
    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=202, horizon=4, n_agents=11,
            private_information=1.0, decision_interval=4,
        ),
        "centralized_lookahead",
    )
    runner.run()
    dispatch_steps = {
        event.step for event in runner.env.ledger.events
        if event.kind == "tool_call"
        and event.payload.get("tool") == "schedule_shipment"
    }
    assert len(dispatch_steps) >= 3
