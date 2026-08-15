"""Statistical-unit and feature-boundary tests for v4 analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thermoagent.v4_analysis import (
    ENERGY_FEATURES,
    ENTROPY_DISAGREEMENT_FEATURES,
    FEATURE_BLOCKS,
    LOCAL_KPI_FEATURES,
    _budget_selection,
    _paired_bootstrap,
    crossfit_scores,
)


def _synthetic_candidates() -> pd.DataFrame:
    rows = []
    for seed in range(8):
        for incident in range(3):
            fragmented = float((seed + incident) % 3) / 2.0
            beneficial = int(fragmented > 0.6)
            row = {
                "cluster_id": "utility|compound|%d" % seed,
                "environment_seed": seed,
                "application": "utility_restoration",
                "regime": "compound",
                "information_condition": "private_fragmented",
                "incident_id": "incident_%d" % incident,
                "causal_utility": 0.8 if beneficial else -0.02,
                "intervention_effect": 0.82 if beneficial else 0.0,
                "loss_with_intervention": 1.0 if beneficial else 1.82,
                "operator_minutes": 8.0,
                "harmful": 0,
                "beneficial": beneficial,
            }
            for feature in LOCAL_KPI_FEATURES:
                row[feature] = 0.5
            row.update({
                "operational_energy": 0.5,
                "standardized_energy": 0.0,
                "distributed_entropy": fragmented,
                "entropy_anomaly": fragmented,
                "entropy_slope": 0.0,
                "belief_disagreement": fragmented,
                "consensus_confidence": 1.0 - 0.5 * fragmented,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_v4_feature_blocks_keep_kpis_and_thermodynamics_explicit() -> None:
    assert not set(LOCAL_KPI_FEATURES) & set(ENERGY_FEATURES)
    assert not set(LOCAL_KPI_FEATURES) & set(ENTROPY_DISAGREEMENT_FEATURES)
    assert set(FEATURE_BLOCKS["local_kpi_only"]) == set(LOCAL_KPI_FEATURES)
    assert set(ENTROPY_DISAGREEMENT_FEATURES).issubset(
        FEATURE_BLOCKS["kpi_plus_entropy_disagreement"]
    )


def test_v4_budget_selection_never_exceeds_panel_budget() -> None:
    frame = _synthetic_candidates()
    frame["score"] = frame["belief_disagreement"]
    selected = _budget_selection(frame, "score", budget=1)
    assert len(selected) == frame.cluster_id.nunique()
    assert (selected.selected_interventions == 1).all()


def test_v4_crossfit_separates_environment_seed_groups() -> None:
    frame = _synthetic_candidates()
    scores, folds = crossfit_scores(
        frame,
        FEATURE_BLOCKS["kpi_plus_entropy_disagreement"],
        c_grid=(0.05, 0.2),
        budget=1,
    )
    assert len(scores) == len(frame)
    assert np.isfinite(scores).all()
    assert len(folds) == 4
    combined = np.concatenate([fold.indices for fold in folds])
    assert sorted(combined.tolist()) == list(range(len(frame)))


def test_v4_bootstrap_uses_one_value_per_cluster() -> None:
    differences = np.asarray([0.1, 0.2, 0.3, 0.4])
    result = _paired_bootstrap(differences, replicates=10_000, seed=44041)
    assert result["mean"] == np.mean(differences)
    assert result["ci95_low"] > 0.0
    assert result["bootstrap_replicates"] == 10_000


def test_v4_thermodynamic_crossfit_ranks_fragmented_cases() -> None:
    frame = _synthetic_candidates()
    scores, _ = crossfit_scores(
        frame,
        FEATURE_BLOCKS["kpi_plus_entropy_disagreement"],
        c_grid=(0.05, 0.2, 1.0),
        budget=1,
    )
    frame["score"] = scores
    selected = _budget_selection(frame, "score", budget=1)
    assert selected.causal_utility.mean() > 0.5
