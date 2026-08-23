import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v15.analysis import (
    V15_Z_FEATURES,
    fit_nominal_distances,
    memory_prompt_balance,
    primary_hypotheses,
    quench_summaries,
)
from thermoagent.statmech_llm_v15.workflow import load_yaml


ROOT = Path(__file__).resolve().parents[2]


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

