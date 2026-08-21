from pathlib import Path

import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v14.analysis import (
    OBSERVABLE_FAMILIES,
    exact_sign_flip_p,
    holm_adjust,
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

