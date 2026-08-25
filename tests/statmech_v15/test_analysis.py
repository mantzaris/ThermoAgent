import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v14.observables import irreversibility_sensitivity
from thermoagent.statmech_llm_v15.analysis import (
    _cluster_bootstrap_summary,
    _strict_cluster_values,
    V15_Z_FEATURES,
    cluster_seed_audit,
    fit_nominal_distances,
    irreversibility_sensitivity_with_floor_uncertainty,
    memory_prompt_balance,
    model_stratified_sensitivity,
    primary_hypotheses,
    quench_summaries,
    raw_generation_accounting_audit,
)
from thermoagent.statmech_llm_v15.experiment import formal_panel_design
from thermoagent.statmech_llm_v15.workflow import load_yaml
from thermoagent.statmech_llm_v15.reporting import (
    _unrecorded_infrastructure_accounting,
)


ROOT = Path(__file__).resolve().parents[2]


def test_shuffle_floor_uncertainty_audit_preserves_frozen_adjusted_estimator():
    states = np.asarray([0, 1, 3, 2, 0, 1, 2, 3] * 12, dtype=int)
    frozen = irreversibility_sensitivity(states, [2, 3], [0.1, 0.5], 40, 15159901)
    audited = irreversibility_sensitivity_with_floor_uncertainty(
        states, [2, 3], [0.1, 0.5], 40, 15159901
    )
    for old, new in zip(frozen, audited):
        for key in (
            "raw_block_divergence_nats_per_update",
            "shuffle_floor_nats_per_update",
            "adjusted_irreversibility_nats_per_update",
        ):
            assert np.isclose(old[key], new[key], rtol=0.0, atol=1.0e-15)
        assert new["shuffle_replicates"] == 40
        assert new["shuffle_floor_monte_carlo_se"] >= 0.0
        assert (
            new["shuffle_floor_mean_mc_ci_low"]
            <= new["shuffle_floor_nats_per_update"]
            <= new["shuffle_floor_mean_mc_ci_high"]
        )


def test_descriptive_cluster_summary_retains_undefined_units_as_json_null():
    summary = _cluster_bootstrap_summary(
        [1.0, float("nan"), 3.0], seed=15159999, replicates=100
    )
    assert summary["estimate"] == 2.0
    assert summary["independent_clusters"] == 2.0
    assert summary["clusters_total"] == 3.0
    assert summary["undefined_clusters"] == 1.0
    encoded = _strict_cluster_values(
        {"cluster_a": 1.0, "cluster_b": float("nan")}
    )
    assert json.loads(encoded) == {"cluster_a": 1.0, "cluster_b": None}
    assert "NaN" not in encoded


def _raw_record(path: Path, model: str, call_index: int) -> str:
    payload = {
        "call_index": call_index,
        "model_key": model,
        "model_calls": 2 if call_index == 2 else 1,
        "prompt_tokens": 100 + call_index,
        "generated_tokens": 10 + call_index,
        "latency_seconds": 0.5 + call_index,
        "first_pass_valid": call_index != 2,
        "repaired": call_index == 2,
        "valid": True,
    }
    serialized = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    destination = path / ("call_%08d_%s.json" % (call_index, digest[:12]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialized)
    return digest


def test_raw_generation_accounting_separates_interrupted_orphans(tmp_path):
    panel_root = tmp_path / "formal/panels"
    raw_root = tmp_path / "raw/formal"
    panel_root.mkdir(parents=True)
    retained = _raw_record(raw_root / "qwen", "qwen", 1)
    _raw_record(raw_root / "qwen", "qwen", 2)
    pd.DataFrame(
        [{"model_key": "qwen", "raw_artifact_sha256": retained}]
    ).to_csv(panel_root / "panel.csv", index=False)
    audit = raw_generation_accounting_audit(panel_root, raw_root)
    retained_row = audit[
        (audit["model_key"] == "qwen")
        & (audit["accounting_scope"] == "retained_panel")
    ].iloc[0]
    orphan_row = audit[
        (audit["model_key"] == "qwen")
        & (audit["accounting_scope"] == "orphan_interrupted_attempt")
    ].iloc[0]
    assert int(retained_row.record_count) == 1
    assert int(retained_row.model_calls) == 1
    assert int(orphan_row.record_count) == 1
    assert int(orphan_row.model_calls) == 2
    assert int(orphan_row.repair_attempted) == 1
    assert set(audit["status"]) == {"passed"}


def test_unrecorded_quota_failure_makes_measured_accounting_a_lower_bound(
    tmp_path, monkeypatch
):
    artifacts = tmp_path / "artifacts"
    incident = artifacts / "invalidated/quota_failure/accounting.json"
    incident.parent.mkdir(parents=True)
    incident.write_text(
        json.dumps(
            {
                "classification": (
                    "external_artifact_disk_quota_exceeded_during_atomic_raw_record"
                ),
                "model_key": "granite",
                "panel_id": "test_panel",
                "scientific_panel_completed": False,
                "generated_call_tokens_and_latency": (
                    "unavailable_because_atomic_record_was_not_written"
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMO_V15_ARTIFACT_ROOT", str(artifacts))
    accounting = _unrecorded_infrastructure_accounting()
    assert accounting["decision_requests"] == 1
    assert accounting["model_calls"] == 1
    assert accounting["calls_with_unknown_prompt_tokens"] == 1
    assert accounting["calls_with_unknown_generated_tokens"] == 1
    assert accounting["calls_with_unknown_latency"] == 1
    assert accounting["measured_generation_accounting_is_lower_bound"] is True


def _protocol():
    return load_yaml(ROOT / "configs/statmech_v15/protocol_template.yaml")


def _macro_fixture():
    rng = np.random.default_rng(15150001)
    rows = []
    conditions = (
        "nominal_markovized",
        "field_markovized",
        "field_persistent",
        "field_scrambled",
    )
    for model in ("qwen", "granite"):
        for cluster_index in range(6):
            cluster = "%s_c%d" % (model, cluster_index)
            for condition in conditions:
                for sweep in range(1, 46):
                    phase = "baseline" if sweep <= 15 else ("disruption" if sweep <= 30 else "recovery")
                    shift = 3.0 if condition != "nominal_markovized" and phase == "disruption" else 0.0
                    if condition != "nominal_markovized" and phase == "recovery":
                        shift = 2.0 - 0.12 * (sweep - 31)
                    row = {
                        "model_key": model,
                        "cluster_id": cluster,
                        "panel_id": "%s_%s" % (cluster, condition),
                        "condition": condition,
                        "memory_mode": condition.rsplit("_", 1)[-1],
                        "disruption": "nominal" if condition == "nominal_markovized" else "field_reversal",
                        "sweep": sweep,
                        "phase": phase,
                        "window_sweeps": 5,
                    }
                    for feature_index, feature in enumerate(V15_Z_FEATURES):
                        row[feature] = float(
                            0.1 * feature_index
                            + 0.02 * cluster_index
                            + shift
                            + rng.normal(scale=0.05)
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def test_nominal_fit_is_model_stratified_and_excludes_held_out_cluster():
    frame = _macro_fixture()
    corrected, diagnostics, thresholds = fit_nominal_distances(frame, _protocol())
    assert not corrected["macrostate_distance"].isna().any()
    assert len(thresholds) == 12
    assert set(diagnostics["held_out_cluster_excluded"]) == {True}
    for row in diagnostics.itertuples():
        assert str(row.held_out_cluster) not in json.loads(row.training_clusters)
    changed = frame.copy()
    selected = changed["cluster_id"] == "qwen_c0"
    changed.loc[selected, list(V15_Z_FEATURES)] += 10000.0
    _, _, changed_thresholds = fit_nominal_distances(changed, _protocol())
    assert np.isclose(thresholds["qwen:qwen_c0:w5"], changed_thresholds["qwen:qwen_c0:w5"])


def test_fixed_recovery_estimand_can_be_positive_zero_or_negative():
    frame = _macro_fixture()
    corrected, _, thresholds = fit_nominal_distances(frame, _protocol())
    summaries = quench_summaries(corrected, _protocol(), thresholds)
    assert len(summaries) == 48
    first_panel = corrected["panel_id"].iloc[0]
    modified = corrected.copy()
    early = (modified["panel_id"] == first_panel) & modified["sweep"].between(31, 35)
    late = (modified["panel_id"] == first_panel) & modified["sweep"].between(41, 45)
    modified.loc[early, "macrostate_distance"] = 1.0
    modified.loc[late, "macrostate_distance"] = 2.0
    altered = quench_summaries(modified, _protocol(), thresholds)
    value = float(
        altered.loc[
            altered["panel_id"] == first_panel,
            "fixed_early_minus_late_recovery_distance",
        ].iloc[0]
    )
    assert value < 0.0


def _panel_and_quench_effect_fixtures():
    panels = []
    quench = []
    for model in ("qwen", "granite"):
        for index in range(6):
            cluster = "%s_c%d" % (model, index)
            for condition, value in (
                ("field_markovized", 0.01),
                ("field_persistent", 0.05),
                ("field_scrambled", 0.02),
                ("nominal_markovized", 0.0),
            ):
                panels.append(
                    {
                        "model_key": model,
                        "cluster_id": cluster,
                        "condition": condition,
                        "adjusted_pathwise_irreversibility_nats_per_update": value,
                        "mean_prompt_tokens": 500.0 + (5.0 if "persistent" in condition else 0.0),
                        "mean_prompt_memory_entries": 2.5 if condition in ("field_persistent", "field_scrambled") else 0.0,
                    }
                )
                quench.append(
                    {
                        "model_key": model,
                        "cluster_id": cluster,
                        "condition": condition,
                        "maximum_post_quench_distance": 10.0 if condition == "field_markovized" else 1.0,
                        "fixed_early_minus_late_recovery_distance": 4.0,
                    }
                )
    return pd.DataFrame(panels), pd.DataFrame(quench)


def test_primary_inference_uses_graph_model_units_and_frozen_multiplicity():
    panels, quench = _panel_and_quench_effect_fixtures()
    effects, dispositions = primary_hypotheses(panels, quench, _protocol())
    assert set(effects["hypothesis"]) == {"H1", "H2", "H3", "H4"}
    assert int(effects.loc[effects["hypothesis"] == "H1", "independent_clusters"].iloc[0]) == 6
    assert set(effects.loc[effects["hypothesis"] != "H1", "independent_clusters"]) == {12}
    assert all(float(value) > 0.0 for value in effects["estimate"])
    assert float(effects.loc[effects["hypothesis"] == "H1", "allocated_alpha"].iloc[0]) == 0.02
    assert all(float(value) == 0.03 for value in effects.loc[effects["hypothesis"] != "H1", "allocated_alpha"])
    assert all(bool(dispositions[key]["supported"]) for key in dispositions)


def test_prompt_balance_pairs_only_within_model_cluster():
    panels, _ = _panel_and_quench_effect_fixtures()
    balance = memory_prompt_balance(panels)
    assert len(balance) == 12
    assert set(balance["persistent_minus_scrambled_mean_prompt_tokens"]) == {5.0}


def test_model_stratified_sensitivity_does_not_relabel_confirmation():
    panels, quench = _panel_and_quench_effect_fixtures()
    sensitivity = model_stratified_sensitivity(panels, quench)
    assert len(sensitivity) == 8
    assert set(sensitivity["model_key"]) == {"qwen", "granite"}
    assert set(sensitivity["independent_clusters"]) == {6}
    assert set(sensitivity["confirmatory_disposition_assigned"]) == {False}
    assert set(sensitivity["positive_clusters"]) == {6}


def test_model_cluster_seed_namespaces_are_disjoint_and_arms_are_matched():
    audit = cluster_seed_audit(formal_panel_design(_protocol()))
    assert len(audit) == 12
    assert set(audit["model_key"]) == {"qwen", "granite"}
    boolean_columns = [
        "matched_seed_within_cluster",
        "complete_four_arm_panel",
        "panel_seed_globally_unique",
        "graph_seed_globally_unique",
        "control_seed_globally_unique",
        "model_seed_namespaces_disjoint",
    ]
    assert audit[boolean_columns].to_numpy(bool).all()
