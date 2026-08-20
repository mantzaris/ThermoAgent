import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v13.analysis import REPRESENTATIONS, _representation_cv


def test_representation_cross_validation_holds_out_whole_trajectory_clusters():
    labels = ("nominal", "field_reversal", "network_partition", "message_corruption")
    rows = []
    for cluster_index in range(4):
        for label_index, label in enumerate(labels):
            row = {
                "panel_id": f"c{cluster_index}_{label}",
                "cluster_id": f"c{cluster_index}",
                "label": label,
            }
            variables = sorted({item for values in REPRESENTATIONS.values() for item in values})
            for variable_index, variable in enumerate(variables):
                row[f"delta_{variable}"] = float(label_index + 0.01 * cluster_index + 0.001 * variable_index)
                row[f"recovery_{variable}"] = float(-label_index + 0.01 * cluster_index - 0.001 * variable_index)
            rows.append(row)
    predictions, folds = _representation_cv(pd.DataFrame(rows))
    assert len(folds) == 12
    assert len(predictions) == 48
    assert all(item["test_panels"] == 4 for item in folds)
    assert all(np.isfinite(item["multiclass_log_loss"]) for item in folds)
    for item in predictions:
        assert item["panel_id"].startswith(item["held_out_cluster"] + "_")
