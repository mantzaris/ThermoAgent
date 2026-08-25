from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd

from thermoagent.statmech_llm_v15.figures import (
    _collective_correlations,
    _dynamical_persistence_shape,
    _surrogate,
    _style,
    _v14_audit,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v14_audit_figure_uses_existing_total_correlation_metric_names(tmp_path):
    result = tmp_path / "results"
    (result / "figures/pdf").mkdir(parents=True)
    (result / "figures/source_data").mkdir(parents=True)
    _style()
    catalog = []
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        _v14_audit(ROOT, result, catalog)
    assert not any(
        "No artists with labels" in str(item.message) for item in observed
    )
    source = pd.read_csv(
        result / "figures/source_data/figure05_v14_delayed_audit.csv"
    )
    dependence = source[source["panel"] == "dependence"]
    assert set(dependence["metric"]) == {
        "total_correlation_raw",
        "total_correlation_null_mean",
        "total_correlation_bias_adjusted",
    }
    assert len(catalog) == 1


def _write_extension_tables(result: Path) -> None:
    tables = result / "tables"
    tables.mkdir(parents=True)
    matrix_rows = []
    profile_rows = []
    curve_rows = []
    integrated_rows = []
    integrated_summary_rows = []
    binder_rows = []
    binder_summary_rows = []
    binder_window_rows = []
    binder_pooling_rows = []
    distribution_rows = []
    phases = ("baseline", "disruption", "recovery")
    conditions = (
        "nominal_markovized",
        "field_markovized",
        "field_persistent",
        "field_scrambled",
    )
    for model_index, model in enumerate(("qwen", "granite")):
        for phase_index, phase in enumerate(phases):
            for left in range(16):
                for right in range(16):
                    matrix_rows.append(
                        {
                            "model_key": model,
                            "condition": "field_persistent",
                            "phase": phase,
                            "agent_i": left,
                            "agent_j": right,
                            "community_i": left // 8,
                            "community_j": right // 8,
                            "connected_correlation": float(
                                np.exp(-abs(left - right) / 4.0)
                                * (1.0 if left // 8 == right // 8 else -0.25)
                            ),
                            "between_cluster_sd": 0.05,
                            "independent_clusters": 6,
                        }
                    )
            for distance in range(1, 5):
                estimate = 0.35 * np.exp(-distance / 2.0) - 0.02 * phase_index
                profile_rows.append(
                    {
                        "model_key": model,
                        "condition": "field_persistent",
                        "phase": phase,
                        "graph_distance": distance,
                        "metric": "connected_correlation",
                        "estimate": estimate,
                        "ci_low": estimate - 0.05,
                        "ci_high": estimate + 0.05,
                        "independent_clusters": 6,
                    }
                )
        for condition_index, condition in enumerate(conditions):
            for lag in range(33):
                estimate = float(np.exp(-lag / (8.0 + 2.0 * condition_index)))
                curve_rows.append(
                    {
                        "model_key": model,
                        "condition": condition,
                        "phase": "recovery",
                        "lag_updates": lag,
                        "lag_sweeps": lag / 16.0,
                        "metric": "autocorrelation",
                        "estimate": estimate,
                        "ci_low": estimate - 0.04,
                        "ci_high": estimate + 0.04,
                        "independent_clusters": 6,
                    }
                )
            for cluster in range(6):
                integrated_rows.append(
                    {
                        "model_key": model,
                        "cluster_id": "%s_%d" % (model, cluster),
                        "condition": condition,
                        "phase": "recovery",
                        "lag_truncation_sweeps": 2,
                        "lag_truncation_updates": 32,
                        "is_primary": True,
                        "integrated_autocorrelation_time_updates": 5.0
                        + condition_index
                        + 0.1 * cluster,
                    }
                )
            estimate = 5.0 + condition_index + 0.25
            integrated_summary_rows.append(
                {
                    "model_key": model,
                    "condition": condition,
                    "phase": "recovery",
                    "lag_truncation_sweeps": 2,
                    "lag_truncation_updates": 32,
                    "is_primary": True,
                    "metric": "integrated_autocorrelation_time_updates",
                    "estimate": estimate,
                    "ci_low": estimate - 0.2,
                    "ci_high": estimate + 0.2,
                    "independent_clusters": 6,
                }
            )
        for phase_index, phase in enumerate(phases):
            for cluster in range(6):
                binder_rows.append(
                    {
                        "model_key": model,
                        "cluster_id": "%s_%d" % (model, cluster),
                        "condition": "field_markovized",
                        "phase": phase,
                        "binder_cumulant": 0.2 + 0.1 * phase_index + 0.01 * cluster,
                    }
                )
                for window_index, window in enumerate(
                    ("full_phase", "early_half", "late_half")
                ):
                    binder_window_rows.append(
                        {
                            "model_key": model,
                            "cluster_id": "%s_%d" % (model, cluster),
                            "condition": "field_markovized",
                            "phase": phase,
                            "temporal_window": window,
                            "binder_cumulant": 0.2
                            + 0.1 * phase_index
                            + 0.01 * cluster
                            + 0.005 * window_index,
                            "magnetization_second_moment": 0.5,
                            "magnetization_fourth_moment": 0.3,
                            "near_zero_denominator": False,
                            "window_updates": 240 if window == "full_phase" else 120,
                        }
                    )
            binder_estimate = 0.2 + 0.1 * phase_index + 0.025
            binder_summary_rows.append(
                {
                    "model_key": model,
                    "condition": "field_markovized",
                    "phase": phase,
                    "metric": "binder_cumulant",
                    "estimate": binder_estimate,
                    "ci_low": binder_estimate - 0.02,
                    "ci_high": binder_estimate + 0.02,
                    "independent_clusters": 6,
                }
            )
            for window_index, window in enumerate(
                ("full_phase", "early_half", "late_half")
            ):
                for rule in (
                    "mean_of_cluster_cumulants",
                    "pooled_moments_across_clusters",
                ):
                    estimate = binder_estimate + 0.005 * window_index
                    binder_pooling_rows.append(
                        {
                            "model_key": model,
                            "condition": "field_markovized",
                            "phase": phase,
                            "temporal_window": window,
                            "pooling_rule": rule,
                            "estimate": estimate,
                            "ci_low": estimate - 0.02,
                            "ci_high": estimate + 0.02,
                            "independent_clusters": 6,
                        }
                    )
            for magnetization in np.linspace(-1.0, 1.0, 17):
                probability = float(
                    np.exp(-((magnetization - 0.15 * (phase_index - 1)) ** 2) / 0.2)
                )
                distribution_rows.append(
                    {
                        "model_key": model,
                        "condition": "field_persistent",
                        "phase": phase,
                        "belief_magnetization": magnetization,
                        "metric": "probability",
                        "estimate": probability,
                        "ci_low": max(0.0, probability - 0.03),
                        "ci_high": probability + 0.03,
                        "independent_clusters": 6,
                    }
                )
    for name, rows in (
        ("connected_correlation_matrix_means.csv", matrix_rows),
        ("connected_correlation_profile_summary.csv", profile_rows),
        ("autocorrelation_curve_summary.csv", curve_rows),
        ("integrated_autocorrelation.csv", integrated_rows),
        ("integrated_autocorrelation_summary.csv", integrated_summary_rows),
        ("binder_cumulants.csv", binder_rows),
        ("binder_cumulant_summary.csv", binder_summary_rows),
        ("binder_cumulant_sensitivity.csv", binder_window_rows),
        ("binder_cumulant_pooling_sensitivity.csv", binder_pooling_rows),
        ("magnetization_distribution_summary.csv", distribution_rows),
    ):
        pd.DataFrame(rows).to_csv(tables / name, index=False)


def test_collective_extension_figures_are_vector_pdfs_with_source_data(tmp_path):
    result = tmp_path / "results"
    (result / "figures/pdf").mkdir(parents=True)
    (result / "figures/source_data").mkdir(parents=True)
    _write_extension_tables(result)
    _style()
    catalog = []
    _collective_correlations(result, catalog)
    _dynamical_persistence_shape(result, catalog)
    assert len(catalog) == 2
    for name in (
        "figure13_graph_distance_correlations",
        "figure14_persistence_and_binder",
    ):
        pdf = result / "figures/pdf" / (name + ".pdf")
        source = result / "figures/source_data" / (name + ".csv")
        assert pdf.read_bytes().startswith(b"%PDF")
        assert len(pd.read_csv(source)) > 0
    persistence_source = pd.read_csv(
        result / "figures/source_data/figure14_persistence_and_binder.csv"
    )
    assert {
        "autocorrelation",
        "integrated_autocorrelation",
        "integrated_autocorrelation_summary",
        "binder",
        "binder_summary",
        "binder_window_sensitivity",
        "binder_pooling_sensitivity",
        "magnetization_distribution",
    } == set(persistence_source["source_type"])


def test_surrogate_quench_figure_includes_all_shared_observables_and_intervals(
    tmp_path,
):
    result = tmp_path / "results"
    (result / "tables").mkdir(parents=True)
    (result / "figures/pdf").mkdir(parents=True)
    (result / "figures/source_data").mkdir(parents=True)
    shutil.copyfile(
        ROOT
        / "results/collective_agent_statmech_v15/tables"
        / "v14_direct_surrogate_quench_trajectories.csv",
        result / "tables/v14_direct_surrogate_quench_trajectories.csv",
    )
    _style()
    catalog = []
    _surrogate(result, catalog)
    assert len(catalog) == 1
    pdf = result / "figures/pdf/figure10_direct_surrogate_quench.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    source = pd.read_csv(
        result / "figures/source_data/figure10_direct_surrogate_quench.csv"
    )
    assert len(source) == 90
    assert set(source["independent_clusters"]) == {6}
    for metric in (
        "belief",
        "action",
        "overlap",
        "energy",
        "entropy",
        "susceptibility",
        "correlation_time",
        "response",
    ):
        assert {metric, metric + "_ci_low", metric + "_ci_high"} <= set(
            source.columns
        )
