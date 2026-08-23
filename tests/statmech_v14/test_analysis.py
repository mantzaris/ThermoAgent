from pathlib import Path

import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v14.analysis import (
    OBSERVABLE_FAMILIES,
    PRIMARY_Z_FEATURES,
    cluster_preserving_permutation_analysis,
    disruption_summaries,
    exact_sign_flip_p,
    holm_adjust,
    nominal_distance_analysis,
    primary_hypotheses,
    representation_cv,
)
from thermoagent.statmech_llm_v14.workflow import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def _feature_fixture():
    labels = ("nominal", "field_reversal", "network_partition", "message_corruption")
    variables = sorted({value for family in OBSERVABLE_FAMILIES.values() for value in family})
    rows = []
    for cluster_index in range(6):
        for label_index, label in enumerate(labels):
            row = {"panel_id": f"c{cluster_index}_{label}", "cluster_id": f"c{cluster_index}", "label": label}
            for feature_index, variable in enumerate(variables):
                row[f"delta_{variable}"] = label_index + 0.01 * cluster_index + 0.001 * feature_index
                row[f"recovery_{variable}"] = -label_index + 0.01 * cluster_index
                row[f"peak_{variable}"] = abs(label_index - 1.5) + 0.001 * feature_index
            rows.append(row)
    return pd.DataFrame(rows)


def test_leave_one_cluster_out_never_splits_a_trajectory_cluster():
    protocol = load_yaml(ROOT / "configs/statmech_v14/protocol_template.yaml")
    predictions, folds, coefficients = representation_cv(_feature_fixture(), protocol)
    assert len(folds) == 18
    assert len(predictions) == 72
    assert not coefficients.empty
    assert set(folds["test_panels"]) == {4}
    for row in predictions.itertuples():
        assert str(row.panel_id).startswith(str(row.held_out_cluster) + "_")


def test_exact_sign_flip_resolution_and_holm_are_reproducible():
    assert np.isclose(exact_sign_flip_p([1.0] * 6), 1.0 / 64.0)
    adjusted = holm_adjust([1.0 / 64.0, 1.0 / 64.0, 1.0 / 64.0])
    assert np.isclose(adjusted[0], 3.0 / 64.0)
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_no_test_cluster_statistics_enter_training_preprocessing(monkeypatch):
    protocol = load_yaml(ROOT / "configs/statmech_v14/protocol_template.yaml")
    data = _feature_fixture()
    original = data.copy()
    held = data["cluster_id"] == "c0"
    columns = [column for column in data if column.startswith(("delta_", "recovery_", "peak_"))]
    data.loc[held, columns] += 10000.0
    predictions, folds, _ = representation_cv(data, protocol)
    assert len(predictions[predictions["held_out_cluster"] == "c0"]) == 12
    assert len(folds[folds["held_out_cluster"] == "c0"]) == 3
    assert original[~held][columns].equals(data[~held][columns])


def _macro_fixture():
    rng = np.random.default_rng(1491)
    rows = []
    for cluster_index in range(6):
        for disruption in ("nominal", "field_reversal", "network_partition", "message_corruption"):
            for sweep in range(1, 16):
                row = {
                    "cluster_id": f"c{cluster_index}",
                    "panel_id": f"c{cluster_index}_{disruption}",
                    "disruption": disruption,
                    "sweep": sweep,
                    "phase": "baseline" if sweep <= 5 else ("disruption" if sweep <= 10 else "recovery"),
                    "window_sweeps": 5,
                    "configuration_entropy": float(rng.normal()),
                    "reference_energy_per_agent": float(rng.normal()),
                    "belief_magnetization": float(rng.normal()),
                    "action_magnetization": float(rng.normal()),
                }
                for feature in PRIMARY_Z_FEATURES:
                    row.setdefault(feature, float(rng.normal()))
                rows.append(row)
    return pd.DataFrame(rows)


def _minimal_nominal_protocol():
    protocol = load_yaml(ROOT / "configs/statmech_v14/protocol_template.yaml")
    protocol["analysis"]["nominal_distance"]["estimators"] = ["shrinkage"]
    protocol["analysis"]["nominal_distance"]["ridge_fractions"] = [0.1]
    protocol["analysis"]["nominal_fit_windows"] = ["all_nominal"]
    return protocol


def test_nominal_threshold_is_fit_without_held_out_cluster_and_passed_explicitly():
    protocol = _minimal_nominal_protocol()
    original = _macro_fixture()
    corrected, robustness, thresholds = nominal_distance_analysis(
        original, protocol, return_thresholds=True, include_single_observable_ablations=True
    )
    changed = original.copy()
    changed.loc[changed["cluster_id"] == "c0", list(PRIMARY_Z_FEATURES)] += 100000.0
    _, _, changed_thresholds = nominal_distance_analysis(
        changed, protocol, return_thresholds=True, include_single_observable_ablations=True
    )
    assert np.isclose(thresholds["c0"], changed_thresholds["c0"])
    assert set(robustness["held_out_cluster_excluded"]) == {True}
    for row in robustness.itertuples():
        assert row.cluster_id not in row.training_clusters
    summaries = disruption_summaries(corrected, protocol, thresholds)
    assert set(summaries["threshold_source"]) == {"leave_one_cluster_out_training_nominal"}
    assert np.allclose(
        summaries["baseline_threshold_95"],
        summaries["cluster_id"].map(thresholds),
    )


def test_single_observable_ablation_removes_exactly_one_feature():
    _, robustness, _ = nominal_distance_analysis(
        _macro_fixture(),
        _minimal_nominal_protocol(),
        return_thresholds=True,
        include_single_observable_ablations=True,
    )
    deleted = robustness[robustness["deleted_observable"] != "none"]
    assert set(deleted["deleted_observable"]) == set(PRIMARY_Z_FEATURES)
    assert set(deleted["feature_count"]) == {len(PRIMARY_Z_FEATURES) - 1}


def test_h3_structurally_invalid_directional_estimand_cannot_be_supported():
    rows = []
    folds = []
    for index in range(6):
        cluster = f"c{index}"
        rows.extend(
            [
                {
                    "cluster_id": cluster,
                    "disruption": "field_reversal",
                    "maximum_post_quench_distance": 10.0,
                    "recovery_drop_estimand": 4.0,
                },
                {
                    "cluster_id": cluster,
                    "disruption": "nominal",
                    "maximum_post_quench_distance": 1.0,
                    "recovery_drop_estimand": 0.0,
                },
            ]
        )
        folds.extend(
            [
                {"held_out_cluster": cluster, "representation": "full_statmech", "balanced_accuracy": 1.0},
                {"held_out_cluster": cluster, "representation": "order_only", "balanced_accuracy": 0.0},
            ]
        )
    effects, dispositions = primary_hypotheses(pd.DataFrame(rows), pd.DataFrame(folds))
    assert bool(effects.loc[effects["hypothesis"] == "H3", "frozen_numerical_criterion_met"].iloc[0])
    assert not dispositions["H3"]["valid_directional_test"]
    assert not dispositions["H3"]["inferential_support"]
    assert not dispositions["H3"]["supported"]
    assert dispositions["H3"]["trajectory_evidence_consistent_with_recovery"]


def test_cluster_preserving_permutation_is_deterministic():
    protocol = load_yaml(ROOT / "configs/statmech_v14/protocol_template.yaml")
    first_null, first_summary = cluster_preserving_permutation_analysis(
        _feature_fixture(), protocol, replicates=8, seed=1492, workers=1
    )
    second_null, second_summary = cluster_preserving_permutation_analysis(
        _feature_fixture(), protocol, replicates=8, seed=1492, workers=1
    )
    pd.testing.assert_frame_equal(first_null, second_null)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert len(first_null) == 8
    assert set(first_summary["replicates"]) == {8}
