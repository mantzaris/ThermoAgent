import numpy as np
import pandas as pd

from thermoagent.v7_formal_analysis import (
    BASE_NUMERIC, ENTROPIC_NUMERIC, FEATURE_BLOCKS, crossfit_feature_block,
    prepare_candidates,
)


def _frame():
    rows = []
    actions = ["allocate_shipment", "redirect_vehicle"]
    topologies = ["modular", "small_world", "random_geometric"]
    for panel in range(10):
        for decision in range(8):
            harmful = int((panel + decision) % 3 == 0)
            row = {
                "run_id": "panel-%02d" % panel,
                "application": "humanitarian",
                "environment_seed": 78000 + panel,
                "topology_family": topologies[panel % len(topologies)],
                "complexity": ("small", "medium", "large")[panel % 3],
                "coupling": ("low", "medium", "high")[panel % 3],
                "fragmentation": ("high", "low", "medium")[panel % 3],
                "network_disruption": "medium",
                "information_condition": "private_fragmented",
                "counterfactual_evaluated": True,
                "counterfactual_action_accepted": True,
                "proposed_operational_action": actions[decision % 2],
                "counterfactual_harmful": harmful,
                "counterfactual_beneficial": 1 - harmful,
                "counterfactual_causal_utility": 0.3 - 0.6 * harmful,
                "step": decision,
                "agent_id": "a-%d" % decision,
                "target": "x-%d" % decision,
            }
            for index, name in enumerate(BASE_NUMERIC + ENTROPIC_NUMERIC):
                row[name] = float((panel * 0.07 + decision * 0.11 + index * 0.03) % 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_v7_feature_blocks_do_not_leak_entropic_values_into_kpi_baseline():
    assert not set(ENTROPIC_NUMERIC).intersection(FEATURE_BLOCKS["strongest_nonentropic"])
    assert set(BASE_NUMERIC).issubset(FEATURE_BLOCKS["generalized_entropic"])


def test_v7_grouped_crossfit_keeps_entire_panel_out_of_training_fold():
    frame = prepare_candidates(_frame())
    predictions, audit = crossfit_feature_block(frame, "strongest_nonentropic")
    assert len(predictions) == len(frame)
    assert np.isfinite(predictions).all()
    assert all(row["panel_disjoint"] for row in audit)
    assert all(row["environment_seed_disjoint"] for row in audit)


def test_v7_prepare_candidates_uses_panel_not_action_as_independent_unit():
    frame = prepare_candidates(_frame())
    assert len(frame) == 80
    assert frame.panel_id.nunique() == 10
    assert frame.cluster_id.nunique() == 10


def test_v7_matched_policy_variants_share_one_independent_panel():
    original = _frame()
    duplicate = original.copy()
    duplicate["run_id"] = duplicate.run_id + "-entropic-controller"
    combined = prepare_candidates(pd.concat([original, duplicate], ignore_index=True))
    assert combined.run_id.nunique() == 20
    assert combined.panel_id.nunique() == 10
