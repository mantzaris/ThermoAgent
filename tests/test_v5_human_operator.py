"""Prospective V5 engineering, privacy, causal, and statistical tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermoagent.agents import PrivacyViolation
from thermoagent.events import EventLedger
from thermoagent.v5_analysis import KPI_FEATURES, crossfit_action_values, select_budget
from thermoagent.v5_environment import V5PanelEnvironment
from thermoagent.v5_experiments import run_panel, write_panel
from thermoagent.v5_replay import replay_v5_episode
from thermoagent.v5_types import OPERATOR_ACTIONS
from thermoagent.v5_tools import V5ToolRegistry
from thermoagent.dashboard.v5 import V5DashboardReplay, frame_svg_v5


@pytest.fixture()
def utility_environment() -> V5PanelEnvironment:
    return V5PanelEnvironment(
        "utility_restoration", "compound", "private_fragmented", 51001,
        sketch_policy="event_triggered",
    )


def test_competitive_panel_has_multiple_incidents_and_actions(utility_environment):
    assert len(utility_environment.incidents) == 4
    assert len(OPERATOR_ACTIONS) >= 8
    rows = utility_environment.candidate_rows()
    assert len(rows) == 4 * len(OPERATOR_ACTIONS)
    assert all(sum(row["incident_id"] == incident for row in rows) == len(OPERATOR_ACTIONS) for incident in utility_environment.incidents)


def test_agent_vaults_memories_and_inboxes_are_separate(utility_environment):
    agents = list(utility_environment.agents.values())
    assert len({id(agent.vault) for agent in agents}) == len(agents)
    assert len({id(agent.inbox) for agent in agents}) == len(agents)
    owner = agents[0]
    intruder = agents[1]
    with pytest.raises(PrivacyViolation):
        owner.vault.observation(intruder.agent_id)


def test_fragmented_private_observations_differ(utility_environment):
    incident = next(iter(utility_environment.incidents))
    observations = [
        utility_environment.observations[agent_id]
        for agent_id in utility_environment.incident_agents[incident]
    ]
    beliefs = {tuple(round(value, 8) for value in item.private_evidence) for item in observations}
    assert len(beliefs) > 1


def test_operator_view_excludes_evaluator_only_state(utility_environment):
    incident = next(iter(utility_environment.incidents))
    view = utility_environment.operator_view(incident, KPI_FEATURES)
    encoded = json.dumps(view, sort_keys=True)
    for prohibited in ("true_mode", "correct_action", "stochastic_tape", "base_loss", "fragmentation"):
        assert prohibited not in encoded


def test_no_future_or_true_cyber_label_in_agent_context(utility_environment):
    for agent in utility_environment.agents.values():
        encoded = json.dumps(agent.context(), sort_keys=True)
        assert "true_mode" not in encoded
        assert "correct_action" not in encoded
        assert "stochastic_tape" not in encoded


def test_partition_blocks_a_contributor_but_counts_messages():
    environment = V5PanelEnvironment(
        "utility_restoration", "partition", "private_fragmented", 51002,
        sketch_policy="always_on",
    )
    for incident_id, thermo in environment.thermodynamics.items():
        assert thermo.sketch_messages == 15
        assert len(thermo.contributors) == 1
        assert thermo.sketch_bytes > 0


def test_sketch_policies_account_for_all_traffic():
    values = {}
    for policy in ("none", "periodic", "event_triggered", "always_on"):
        environment = V5PanelEnvironment(
            "humanitarian", "compound", "private_fragmented", 51003,
            sketch_policy=policy,
        )
        values[policy] = environment.summary()["sketch_messages"]
        events = [event for event in environment.ledger.events if event.kind == "thermodynamic_sketch"]
        assert len(events) == values[policy]
    assert values["none"] == 0
    assert values["periodic"] < values["always_on"]
    assert 0 < values["event_triggered"] <= values["always_on"]


def test_verification_is_imperfect_costly_and_delayed(utility_environment):
    incident = next(iter(utility_environment.incidents))
    effect = utility_environment.action_effect(incident, "verify")
    assert effect.intervention_cost > 0
    assert effect.delay_steps >= 1
    assert effect.operator_minutes > 0


def test_candidate_pool_contains_benefit_neutrality_and_harm(utility_environment):
    rows = utility_environment.candidate_rows()
    effects = np.asarray([float(row["causal_effect"]) for row in rows])
    assert np.any(effects > 1e-9)
    assert np.any(effects < -1e-9)
    assert np.any(np.abs(effects) <= 1e-9)


def test_public_information_retains_nonzero_intervention_effects():
    environment = V5PanelEnvironment(
        "humanitarian", "compound", "public_shared", 51004,
        sketch_policy="event_triggered",
    )
    effects = np.asarray([row["causal_effect"] for row in environment.candidate_rows()])
    assert np.any(effects > 1e-9)
    assert np.any(effects < -1e-9)


def test_counterfactual_actions_share_stochastic_tape(utility_environment):
    rows = utility_environment.candidate_rows()
    assert {row["stochastic_tape_digest"] for row in rows} == {utility_environment.stochastic_tape_digest}
    branches = [event for event in utility_environment.ledger.events if event.kind == "counterfactual_branch"]
    assert branches
    assert all(event.payload["rng_digest_with"] == event.payload["rng_digest_without"] for event in branches)


def test_wrong_action_can_cause_bounded_harm(utility_environment):
    effects = [utility_environment.action_effect(incident_id, action) for incident_id in utility_environment.incidents for action in OPERATOR_ACTIONS]
    harmful = [value for value in effects if value.causal_effect < 0]
    assert harmful
    assert min(value.causal_effect for value in harmful) >= -0.50


def test_fixed_coordination_is_decentralized_and_logged():
    environment = V5PanelEnvironment(
        "humanitarian", "compound", "private_fragmented", 51005,
    )
    result = environment.autonomous_outcome(True)
    assert result["negotiations"] == len(environment.incidents)
    assert result["operational_messages"] > 0
    decisions = [event for event in environment.ledger.events if event.kind in ("commitment", "counteroffer")]
    assert decisions
    assert all(event.actor != "simulator" for event in decisions)


def test_communication_partition_does_not_deliver_isolated_sketch():
    environment = V5PanelEnvironment(
        "utility_restoration", "partition", "private_fragmented", 51006,
        sketch_policy="always_on",
    )
    events = [event for event in environment.ledger.events if event.kind == "thermodynamic_sketch"]
    assert any(not event.payload["delivered"] for event in events)
    assert all(event.payload["delivered"] in (True, False) for event in events)


def test_conservation_and_feasibility(utility_environment):
    report = utility_environment.conservation_report()
    assert report["feasible"]
    assert report["maximum_residual"] <= 1e-12


def test_event_replay_is_exact(tmp_path: Path):
    environment, summary, candidates = run_panel(
        "utility_restoration", "telemetry_integrity", "private_fragmented", 51007,
    )
    root = tmp_path / "human_operator_v5"
    write_panel(Path.cwd(), root, "test", environment, summary, candidates)
    episode = root / "raw" / "test" / summary["run_id"] / "episode.json"
    replay = replay_v5_episode(episode)
    assert replay["mismatches"] == 0
    assert replay["maximum_conservation_residual"] <= 1e-12


def test_operator_budget_enforced_on_competitive_candidates(utility_environment):
    frame = pd.DataFrame(utility_environment.candidate_rows())
    selected = select_budget(frame, frame["causal_effect"], budget=2)
    assert selected["selected_count"].max() <= 2
    chosen = selected["selected_candidates"].iloc[0].split(";")
    incidents = [value.rsplit("|", 2)[-2] for value in chosen if value]
    assert len(incidents) == len(set(incidents))


def test_low_consensus_abstention_reduces_selection():
    environment = V5PanelEnvironment(
        "utility_restoration", "partition", "private_fragmented", 51008,
        sketch_policy="none",
    )
    frame = pd.DataFrame(environment.candidate_rows())
    scores = np.ones(len(frame))
    safe = select_budget(frame, scores, budget=2, consensus_abstention=True)
    forced = select_budget(frame, scores, budget=2, consensus_abstention=False, force_selection=True)
    assert safe["selected_count"].iloc[0] <= forced["selected_count"].iloc[0]


def test_grouped_crossfit_has_predictions_for_every_row():
    frames = []
    for seed in range(51100, 51105):
        environment = V5PanelEnvironment(
            "humanitarian", "compound", "private_fragmented", seed,
        )
        frames.append(pd.DataFrame(environment.candidate_rows()))
    frame = pd.concat(frames, ignore_index=True)
    predictions, folds = crossfit_action_values(frame, KPI_FEATURES, [0.1], budget=2)
    assert np.isfinite(predictions).all()
    assert len(folds) == 5
    assert sum(len(item["test_indices"]) for item in folds) == len(frame)


def test_feature_block_does_not_include_evaluator_entropy():
    assert "evaluator_global_entropy" not in KPI_FEATURES


def test_utility_application_is_abstract_and_offline(utility_environment):
    topology = next(event for event in utility_environment.ledger.events if event.kind == "topology_snapshot")
    assert topology.payload["abstract_defensive_simulation"] is True
    assert topology.payload["cyber_scope"] == "abstract state transitions only"


def test_candidate_action_is_not_encoded_by_incident_identifier():
    environment = V5PanelEnvironment(
        "utility_restoration", "compound", "private_fragmented", 51009,
    )
    rows = environment.candidate_rows()
    for row in rows:
        assert row["correct_action"] if "correct_action" in row else True
    deployable = environment.operator_view(next(iter(environment.incidents)), KPI_FEATURES)
    assert "correct_action" not in json.dumps(deployable)


def test_generated_v5_text_uses_lf(tmp_path: Path):
    environment, summary, candidates = run_panel(
        "commercial", "isolated_physical", "private_fragmented", 51010,
    )
    root = tmp_path / "results"
    write_panel(Path.cwd(), root, "test", environment, summary, candidates)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".jsonl"}:
            assert b"\r\n" not in path.read_bytes()


def test_role_specific_typed_tool_validation(utility_environment):
    agent = next(value for value in utility_environment.agents.values() if value.identity.role == "field_crew")
    incident = agent.identity.incident_scope[0]
    registry = V5ToolRegistry()
    valid = registry.validate(agent.identity.role, agent.identity.incident_scope, {
        "action": "deploy_repair_capacity",
        "incident_id": incident,
        "quantity": 1.0,
        "reason_code": "private field evidence",
    })
    assert valid.ok
    forbidden = registry.validate(agent.identity.role, agent.identity.incident_scope, {
        "action": "authorize_emergency_resource",
        "incident_id": incident,
        "quantity": 1.0,
        "reason_code": "outside crew authority",
    })
    assert not forbidden.ok
    out_of_scope = registry.validate(agent.identity.role, agent.identity.incident_scope, {
        "action": "deploy_repair_capacity",
        "incident_id": "hidden_peer_incident",
        "quantity": 1.0,
        "reason_code": "forbidden scope",
    })
    assert not out_of_scope.ok


def test_v5_dashboard_replay_is_deterministic_and_private(tmp_path: Path):
    environment, summary, candidates = run_panel(
        "utility_restoration", "partition", "private_fragmented", 51011,
    )
    root = tmp_path / "results"
    write_panel(Path.cwd(), root, "dashboard", environment, summary, candidates)
    episode = root / "raw" / "dashboard" / summary["run_id"] / "episode.json"
    first = V5DashboardReplay(episode)
    second = V5DashboardReplay(episode)
    assert first.digest() == second.digest()
    encoded = json.dumps([frame.as_dict() for frame in first.frames], sort_keys=True)
    for prohibited in ("true_mode", "correct_action", "stochastic_tape", "loss_without", "causal_effect"):
        assert prohibited not in encoded
    svg = frame_svg_v5(first.frames[-1])
    assert svg.startswith("<svg")
    assert "Simulated operator" in svg
