from pathlib import Path

import pandas as pd

from thermoagent.v6_artifacts import build_index, verify_artifacts
from thermoagent.v6_figures import FIGURE_DATA_SOURCES
from thermoagent.v6_reporting import _claims, _compute_table, _qwen_table


def test_artifact_index_detects_mutation_and_crlf(tmp_path):
    root = tmp_path / "results"
    (root / "development").mkdir(parents=True)
    target = root / "development" / "value.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    build_index(root)
    assert verify_artifacts(root)["passed"]
    target.write_bytes(b"a,b\r\n1,3\r\n")
    report = verify_artifacts(root)
    assert not report["passed"]
    assert {value["reason"] for value in report["failure_details"]} == {
        "checksum_mismatch", "crlf_text",
    }


def test_every_v6_publication_figure_declares_source_data():
    required = {
        "generalized_entropic_architecture", "entropy_family_curves",
        "risk_coverage", "harm_coverage", "utility_coverage",
        "entropy_family_effect_forest", "fragmented_public_interaction",
        "sequential_rl_learning_curves", "qwen_agent_evaluation",
        "utility_cyber_physical_network", "operator_dashboard",
        "matched_operator_dashboard", "causal_chain_funnel",
    }
    assert required.issubset(FIGURE_DATA_SOURCES)
    assert all(value.startswith("figures/data/") and value.endswith(".csv") for value in FIGURE_DATA_SOURCES.values())


def test_claims_matrix_does_not_use_selective_safety_gate_for_entropy_family(tmp_path):
    root = tmp_path / "results"
    permutation = root / "development" / "permutation"
    permutation.mkdir(parents=True)
    pd.DataFrame([{
        "application": "humanitarian",
        "observed_harm_rate_reduction": 0.01,
        "holm_adjusted_p": 0.20,
    }]).to_csv(permutation / "refit_permutation_family_test.csv", index=False)
    consensus = root / "development" / "sketch_reference"
    consensus.mkdir(parents=True)
    pd.DataFrame([
        {"sketch_policy": "event_triggered", "regime": "nominal", "evaluator_distributed_error": 0.03},
        {"sketch_policy": "event_triggered", "regime": "partition", "evaluator_distributed_error": 0.08},
    ]).to_csv(consensus / "distributed_consensus.csv", index=False)
    gates = {"gates": [
        {"gate": 4, "passed": True}, {"gate": 5, "passed": True},
        {"gate": 6, "passed": True}, {"gate": 7, "passed": True},
        {"gate": 8, "passed": True}, {"gate": 9, "passed": False},
        {"gate": 10, "passed": True},
    ]}
    frame = _claims(root, gates).set_index("hypothesis")
    assert frame.loc["H1", "status"] == "supported_in_development"
    assert frame.loc["H5", "status"] == "unsupported_or_mixed"
    assert frame.loc["H6", "status"] == "supported_in_development"
    assert frame.loc["H8", "status"] == "unsupported_or_mixed"


def test_compute_accounting_retains_qwen_and_rl_communication(tmp_path):
    root = tmp_path / "results"
    (root / "training" / "evaluation").mkdir(parents=True)
    (root / "qwen").mkdir(parents=True)
    pd.DataFrame([{
        "training_episodes": 2, "evaluation_episodes": 1,
        "wall_seconds": 12.0, "device": "cuda",
    }]).to_csv(root / "training" / "seed_manifest.csv", index=False)
    pd.DataFrame([{
        "total_messages": 7, "total_bytes": 701,
    }]).to_csv(root / "training" / "evaluation" / "seed.csv", index=False)
    (root / "qwen" / "qualification_summary.json").write_text(
        '{"episodes":1,"llm_calls":2,"prompt_tokens":30,'
        '"generated_tokens":4,"wall_seconds_including_model_load":5}',
        encoding="utf-8",
    )
    pd.DataFrame([{
        "total_messages": 11, "sketch_messages": 8, "total_bytes": 1201,
    }]).to_csv(root / "qwen" / "episode_summary.csv", index=False)

    totals = _compute_table(root)
    accounting = pd.read_csv(
        root / "tables" / "compute_token_communication_accounting.csv"
    ).set_index("workflow")
    assert accounting.loc["sequential_decentralized_ppo", "total_messages"] == 7
    assert pd.isna(accounting.loc["sequential_decentralized_ppo", "operational_messages"])
    assert accounting.loc["real_qwen_qualification", "operational_messages"] == 3
    assert accounting.loc["real_qwen_qualification", "thermodynamic_sketch_messages"] == 8
    assert totals["total_messages"] == 18
    assert totals["total_bytes"] == 1902


def test_qwen_reporting_uses_identifiable_no_action_regret_and_calibration(tmp_path):
    root = tmp_path / "results"
    (root / "qwen").mkdir(parents=True)
    (root / "qwen" / "qualification_summary.json").write_text(
        '{"applications":{"humanitarian":{"episodes":1}}}',
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "application": "humanitarian", "selected_action": "deploy_resource",
            "causal_effect": -0.2, "confidence": 0.8, "beneficial": False,
        },
        {
            "application": "humanitarian", "selected_action": "no_action",
            "causal_effect": 0.0, "confidence": 0.2, "beneficial": False,
        },
        {
            "application": "humanitarian", "selected_action": "reroute",
            "causal_effect": 0.4, "confidence": 0.7, "beneficial": True,
        },
    ]).to_csv(root / "qwen" / "decision_epochs.csv", index=False)

    _qwen_table(root)
    row = pd.read_csv(root / "tables" / "qwen_agent_qualification.csv").iloc[0]
    assert row["no_action_frequency"] == 1 / 3
    assert abs(row["mean_regret_relative_to_no_action"] - (0.2 / 3)) < 1e-12
    assert abs(row["benefit_confidence_brier"] - ((0.8**2 + 0.2**2 + 0.3**2) / 3)) < 1e-12
    assert "not prospectively defined" in row["best_authorized_action_regret_status"]
