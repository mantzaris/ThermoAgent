"""Bias-qualified analysis for the controlled and persistent Qwen experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from thermoagent.statmech.exact import decode_state, encode_state, entropy_production_rate, stationary_distribution

from .estimators import block_time_reversal_kl, conditional_mutual_information_markov, fit_logistic_response
from .workflow import _atomic_csv, _atomic_json, artifact_root, load_yaml, stage_lock, utc_now


def heldout_paraphrase_logit(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, float]]:
    """Fit on two prompt paraphrases and score the third without leakage."""

    frame = pd.DataFrame(rows).copy()
    output: List[Dict[str, float]] = []
    for heldout in sorted(int(value) for value in frame["paraphrase"].unique()):
        train = frame[frame["paraphrase"] != heldout]
        test = frame[frame["paraphrase"] == heldout]
        fit = fit_logistic_response(
            train["local_field"].to_numpy(float),
            train["belief_spin"].to_numpy(int),
            train["previous_belief"].to_numpy(int),
            np.where(train["option_order_right_first"].to_numpy(int) > 0, 1, -1),
        )
        design = np.column_stack(
            [
                np.ones(len(test)),
                test["local_field"].to_numpy(float),
                test["previous_belief"].to_numpy(float),
                np.where(test["option_order_right_first"].to_numpy(int) > 0, 1.0, -1.0),
            ]
        )
        coefficients = np.asarray(
            [fit["intercept"], fit["field_slope"], fit["hysteresis_slope"], fit["option_order_slope"]],
            dtype=float,
        )
        linear = np.clip(design.dot(coefficients), -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        target = (test["belief_spin"].to_numpy(int) > 0).astype(float)
        nll = float(np.sum(np.logaddexp(0.0, linear) - target * linear))
        train_mean = float(np.clip(np.mean(train["belief_spin"].to_numpy(int) > 0), 1e-8, 1.0 - 1e-8))
        intercept_nll = float(-np.sum(target * np.log(train_mean) + (1.0 - target) * np.log(1.0 - train_mean)))
        output.append(
            {
                "heldout_paraphrase": float(heldout),
                "train_n": float(len(train)),
                "test_n": float(len(test)),
                "heldout_brier": float(np.mean((probability - target) ** 2)),
                "heldout_negative_log_likelihood": nll,
                "intercept_negative_log_likelihood": intercept_nll,
                "relative_nll_improvement": float((intercept_nll - nll) / intercept_nll),
                **fit,
            }
        )
    return output


def _smoothed_kernel_from_destinations(
    destinations: np.ndarray,
    sampled_replicates: np.ndarray,
    n_agents: int,
    pseudocount: float,
) -> np.ndarray:
    n_states, variable_count, _replicates = destinations.shape
    counts = np.zeros((n_states, n_states), dtype=float)
    for source in range(n_states):
        for variable in range(variable_count):
            selected = destinations[source, variable, sampled_replicates]
            counts[source] += np.bincount(selected, minlength=n_states)
    for source in range(n_states):
        state = decode_state(source, n_agents)
        for variable in range(2 * n_agents):
            for value in (-1, 1):
                destination = state.copy()
                values = destination.beliefs if variable < n_agents else destination.actions
                values[variable if variable < n_agents else variable - n_agents] = value
                counts[source, encode_state(destination)] += float(pseudocount)
    return counts / counts.sum(axis=1, keepdims=True)


def bootstrap_empirical_kernel_epr(
    rows: Sequence[Mapping[str, object]],
    n_agents: int,
    alphas: Sequence[float],
    replicates: int,
    seed: int,
    pseudocount: float = 0.5,
) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    """Stratified paired bootstrap of empirical-kernel irreversibility.

    Replicate indices are resampled jointly across alpha while state and
    scheduled-variable strata remain fixed.  Individual transitions are not
    presented as independent graph-level replications.
    """

    frame = pd.DataFrame(rows)
    alpha_values = [float(value) for value in alphas]
    n_states = 1 << (2 * int(n_agents))
    variable_count = 2 * int(n_agents)
    repeat_count = int(frame["replicate"].max()) + 1
    arrays: Dict[float, np.ndarray] = {}
    for alpha in alpha_values:
        array = np.full((n_states, variable_count, repeat_count), -1, dtype=int)
        subset = frame[np.isclose(frame["alpha"].to_numpy(float), alpha)]
        for row in subset.to_dict(orient="records"):
            array[int(row["state_index"]), int(row["variable"]), int(row["replicate"])] = int(
                row["destination_state"]
            )
        if np.any(array < 0):
            raise ValueError("controlled-kernel design has a missing state/variable/replicate cell")
        arrays[alpha] = array
    full_indices = np.arange(repeat_count, dtype=int)
    observed: Dict[float, float] = {}
    for alpha in alpha_values:
        kernel = _smoothed_kernel_from_destinations(arrays[alpha], full_indices, n_agents, pseudocount)
        stationary = stationary_distribution(kernel)
        observed[alpha] = entropy_production_rate(stationary, kernel)
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(replicates), len(alpha_values)), dtype=float)
    for bootstrap_index in range(int(replicates)):
        selected = rng.integers(0, repeat_count, repeat_count)
        for alpha_index, alpha in enumerate(alpha_values):
            kernel = _smoothed_kernel_from_destinations(arrays[alpha], selected, n_agents, pseudocount)
            samples[bootstrap_index, alpha_index] = entropy_production_rate(stationary_distribution(kernel), kernel)
    baseline_index = int(np.argmin(np.abs(np.asarray(alpha_values))))
    deltas = samples - samples[:, [baseline_index]]
    observed_baseline = observed[alpha_values[baseline_index]]
    summary: List[Dict[str, float]] = []
    for alpha_index, alpha in enumerate(alpha_values):
        summary.append(
            {
                "alpha": alpha,
                "empirical_kernel_epr_per_controlled_update": observed[alpha],
                "reciprocal_floor": observed_baseline,
                "bias_floor_adjusted_difference": observed[alpha] - observed_baseline,
                "difference_ci_low": float(np.quantile(deltas[:, alpha_index], 0.025)),
                "difference_ci_high": float(np.quantile(deltas[:, alpha_index], 0.975)),
                "bootstrap_probability_positive": float(np.mean(deltas[:, alpha_index] > 0.0)),
                "bootstrap_replicates": float(replicates),
            }
        )
    nonzero = np.asarray(alpha_values) > 0.0
    x = np.square(np.asarray(alpha_values)[nonzero])
    coefficient_samples = (deltas[:, nonzero].dot(x)) / float(x.dot(x))
    bootstrap_frame = pd.DataFrame(samples, columns=["epr_alpha_%s" % str(value).replace(".", "p") for value in alpha_values])
    bootstrap_frame["quadratic_delta_coefficient"] = coefficient_samples
    return summary, bootstrap_frame


def curve_model_comparison(alpha: Sequence[float], values: Sequence[float]) -> List[Dict[str, float]]:
    """Compare the three prespecified low-order response forms."""

    x = np.asarray(alpha, dtype=float)
    y = np.asarray(values, dtype=float)
    designs = {
        "intercept_plus_linear": np.column_stack([np.ones(x.size), x]),
        "intercept_plus_quadratic": np.column_stack([np.ones(x.size), x ** 2]),
        "intercept_plus_linear_plus_quadratic": np.column_stack([np.ones(x.size), x, x ** 2]),
    }
    rows: List[Dict[str, float]] = []
    for name, design in designs.items():
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design.dot(coefficients)
        rss = float(np.sum(residual ** 2))
        loocv = []
        for heldout in range(x.size):
            retained = np.arange(x.size) != heldout
            fitted = np.linalg.lstsq(design[retained], y[retained], rcond=None)[0]
            loocv.append(float((y[heldout] - design[heldout].dot(fitted)) ** 2))
        row: Dict[str, float] = {
            "model": name,
            "residual_sum_squares": rss,
            "leave_one_alpha_out_mse": float(np.mean(loocv)),
            "parameter_count": float(design.shape[1]),
        }
        for index, coefficient in enumerate(coefficients):
            row["coefficient_%d" % index] = float(coefficient)
        rows.append(row)
    return rows


def dynamic_irreversibility_panels(
    rows: Sequence[Mapping[str, object]],
    shuffle_replicates: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Compute panel-level coarse irreversibility and time-shuffle bias floors."""

    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(int(seed))
    output: List[Dict[str, object]] = []
    keys = ["application", "n_agents", "panel", "alpha"]
    for key, group in frame.groupby(keys, sort=True):
        ordered = group.sort_values("turn")
        states = ordered["coarse_macrostate"].to_numpy(int)
        observed = block_time_reversal_kl(states, 3, 0.5)
        null = np.asarray(
            [block_time_reversal_kl(states[rng.permutation(states.size)], 3, 0.5) for _ in range(int(shuffle_replicates))],
            dtype=float,
        )
        cmi = conditional_mutual_information_markov(states, 0.1)
        output.append(
            {
                "application": str(key[0]),
                "n_agents": int(key[1]),
                "panel": int(key[2]),
                "alpha": float(key[3]),
                "turns": len(ordered),
                "coarse_block_irreversibility": observed,
                "time_shuffle_null_mean": float(np.mean(null)),
                "time_shuffle_null_95": float(np.quantile(null, 0.95)),
                "bias_floor_adjusted_irreversibility": observed - float(np.mean(null)),
                "markov_cmi": cmi,
                "messages": int(ordered["messages_sent"].sum()),
                "message_wire_bytes": int(ordered["message_wire_bytes"].sum()),
                "prompt_tokens": int(ordered["prompt_tokens"].sum()),
                "generated_tokens": int(ordered["generated_tokens"].sum()),
                "mean_service_after": float(ordered["service_after"].mean()),
                "final_service_after": float(ordered["service_after"].iloc[-1]),
                "beneficial_tool_actions": int(np.sum(ordered["causal_service_change"].to_numpy(float) < 0.0)),
                "neutral_tool_actions": int(np.sum(np.isclose(ordered["causal_service_change"].to_numpy(float), 0.0))),
                "harmful_tool_actions": int(np.sum(ordered["causal_service_change"].to_numpy(float) > 0.0)),
            }
        )
    return output


def analyze_qwen_formal(repository: Path) -> Dict[str, object]:
    """Build external aggregate LLM results from frozen formal rows."""

    root = artifact_root()
    formal = root / "qwen/formal"
    if not (formal / "summary.json").exists():
        raise RuntimeError("formal Qwen summary is absent")
    amendment = load_yaml(repository / "configs/statmech_v10/llm_pilot_amendment.yaml")
    settings = amendment["llm_analysis"]
    output = root / "qwen/analysis"
    completion = output / "summary.json"
    if completion.exists():
        return json.loads(completion.read_text(encoding="utf-8"))
    with stage_lock("qwen_analysis"):
        calibration = pd.read_csv(formal / "calibration.csv")
        kernel = pd.read_csv(formal / "controlled_kernel.csv")
        dynamic = pd.read_csv(formal / "dynamic_trajectories.csv")
        heldout = heldout_paraphrase_logit(calibration.to_dict(orient="records"))
        _atomic_csv(heldout, output / "calibration_heldout_paraphrase.csv")
        alpha_values = sorted(float(value) for value in kernel["alpha"].unique())
        kernel_summary, bootstrap = bootstrap_empirical_kernel_epr(
            kernel.to_dict(orient="records"),
            2,
            alpha_values,
            int(settings["kernel_bootstrap_replicates"]),
            int(settings["kernel_bootstrap_seed"]),
            float(settings["empirical_kernel_pseudocount"]),
        )
        _atomic_csv(kernel_summary, output / "controlled_kernel_effects.csv")
        _atomic_csv(bootstrap.to_dict(orient="records"), output / "controlled_kernel_bootstrap.csv")
        curve_rows = curve_model_comparison(
            [row["alpha"] for row in kernel_summary],
            [row["empirical_kernel_epr_per_controlled_update"] for row in kernel_summary],
        )
        _atomic_csv(curve_rows, output / "controlled_kernel_curve_models.csv")
        panel_rows = dynamic_irreversibility_panels(
            dynamic.to_dict(orient="records"),
            int(settings["dynamic_shuffle_replicates_per_panel"]),
            int(settings["dynamic_shuffle_seed"]),
        )
        _atomic_csv(panel_rows, output / "dynamic_panel_irreversibility.csv")
        heldout_brier = float(np.mean([row["heldout_brier"] for row in heldout]))
        heldout_nll_gain = float(np.mean([row["relative_nll_improvement"] for row in heldout]))
        coefficient = bootstrap["quadratic_delta_coefficient"].to_numpy(float)
        global_fit = fit_logistic_response(
            calibration["local_field"].to_numpy(float),
            calibration["belief_spin"].to_numpy(int),
            calibration["previous_belief"].to_numpy(int),
            np.where(calibration["option_order_right_first"].to_numpy(int) > 0, 1, -1),
        )
        h5 = bool(
            heldout_brier <= float(settings["h5_maximum_heldout_brier"])
            and heldout_nll_gain >= float(settings["h5_minimum_relative_nll_improvement_over_intercept"])
            and global_fit["field_slope"] > 0.0
            and abs(global_fit["option_order_slope"]) <= float(settings["h5_maximum_absolute_option_order_slope"])
        )
        h6 = bool(np.quantile(coefficient, 0.025) > 0.0)
        panel_frame = pd.DataFrame(panel_rows)
        comparison_alpha = float(settings["h7_comparison_alpha"])
        positive_by_size: Dict[str, int] = {}
        for n_agents, part in panel_frame.groupby("n_agents"):
            pivot = part.pivot_table(index=["application", "panel"], columns="alpha", values="bias_floor_adjusted_irreversibility")
            if 0.0 not in pivot or comparison_alpha not in pivot:
                positive_by_size[str(int(n_agents))] = 0
            else:
                positive_by_size[str(int(n_agents))] = int(np.sum((pivot[comparison_alpha] - pivot[0.0]) > 0.0))
        h7 = bool(
            positive_by_size
            and all(
                count >= int(settings["h7_minimum_positive_panels_per_six_for_each_size"])
                for count in positive_by_size.values()
            )
        )
        summary = {
            "completed_at": utc_now(),
            "controlled_kernel_decisions": len(kernel),
            "dynamic_decisions": len(dynamic),
            "calibration_decisions": len(calibration),
            "heldout_paraphrase_brier": heldout_brier,
            "heldout_relative_nll_improvement": heldout_nll_gain,
            "global_local_policy_fit": global_fit,
            "quadratic_delta_coefficient_mean": float(np.mean(coefficient)),
            "quadratic_delta_coefficient_ci_low": float(np.quantile(coefficient, 0.025)),
            "quadratic_delta_coefficient_ci_high": float(np.quantile(coefficient, 0.975)),
            "positive_dynamic_panels_by_size": positive_by_size,
            "H5_logit_correspondence_passed": h5,
            "H6_llm_nonreciprocity_passed": h6,
            "H7_scale_replication_passed": h7,
            "dynamic_metric_qualification": "coarse-grained block time-reversal irreversibility; not full hidden-state entropy production",
        }
        _atomic_json(summary, completion)
        return summary
