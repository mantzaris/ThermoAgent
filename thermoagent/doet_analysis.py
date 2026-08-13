"""Locked, episode-level statistical and mechanistic analysis for DOET v2."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import yaml

from .events import sha256_file


PRIMARY_METHOD = "doet_rule"
FIXED_METHOD = "fixed_always_on"
PRIMARY_MARGIN = 0.02
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260813
COMMUNICATION_METRICS = (
    "total_communication_messages",
    "total_communication_bytes",
    "prompt_tokens",
    "generated_tokens",
    "llm_calls",
    "llm_latency_seconds",
    "wall_clock_seconds",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _paired(
    frame: pd.DataFrame,
    method: str,
    application: Optional[str] = None,
    scenario: Optional[str] = None,
) -> pd.DataFrame:
    keys = ["application", "scenario_name", "seed", "n_agents"]
    metrics = [
        "primary_outcome", "total_communication_messages",
        "total_communication_bytes", "prompt_tokens", "generated_tokens",
        "llm_calls", "llm_latency_seconds", "wall_clock_seconds",
        "mean_consensus_rmse", "trigger_activations", "quiet_mode_fraction",
        "communication_active_decision_epochs", "tool_proposals",
    ]
    subset = frame
    if application is not None:
        subset = subset[subset["application"] == application]
    if scenario is not None:
        subset = subset[subset["scenario_name"] == scenario]
    planned_left = subset[subset["method"] == method].copy()
    planned_right = subset[subset["method"] == FIXED_METHOD].copy()
    if "status" in subset:
        left_source = planned_left[planned_left["status"] == "complete"]
        right_source = planned_right[planned_right["status"] == "complete"]
    else:
        left_source = planned_left
        right_source = planned_right
    left = left_source[keys + ["rl_training_seed", *metrics]].copy()
    right = right_source[keys + metrics].copy()
    paired = left.merge(right, on=keys, suffixes=("_method", "_fixed"), validate="one_to_one")
    if "status" not in subset and (
        len(paired) != len(left) or len(paired) != len(right)
    ):
        raise ValueError(
            "incomplete matched panel for %s/%s/%s: method=%d fixed=%d paired=%d"
            % (method, application, scenario, len(left), len(right), len(paired))
        )
    planned_keys = {
        tuple(row) for row in pd.concat([
            planned_left[keys], planned_right[keys]
        ], ignore_index=True).drop_duplicates().itertuples(index=False, name=None)
    }
    matched_keys = {
        tuple(row) for row in paired[keys].itertuples(index=False, name=None)
    }
    paired.attrs["planned_pairs"] = len(planned_keys)
    paired.attrs["failed_pairs"] = len(planned_keys - matched_keys)
    if paired.empty:
        raise ValueError(
            "no complete matched panels for %s/%s/%s"
            % (method, application, scenario)
        )
    paired["loss_difference"] = (
        paired["primary_outcome_method"] - paired["primary_outcome_fixed"]
    )
    paired["relative_degradation"] = paired["loss_difference"] / np.maximum(
        np.abs(paired["primary_outcome_fixed"]), 1e-9
    )
    for metric in COMMUNICATION_METRICS:
        paired[metric + "_difference"] = (
            paired[metric + "_method"] - paired[metric + "_fixed"]
        )
        paired[metric + "_reduction"] = 1.0 - (
            paired[metric + "_method"]
            / np.maximum(paired[metric + "_fixed"], 1e-9)
        )
    return paired


def _bootstrap_means(
    paired: pd.DataFrame,
    column: str,
    rng: np.random.RandomState,
    replicates: Optional[int] = None,
) -> np.ndarray:
    if paired.empty:
        raise ValueError("bootstrap requires episode pairs")
    if replicates is None:
        replicates = BOOTSTRAP_REPLICATES
    training_seeds = sorted(set(
        int(value) for value in paired["rl_training_seed"].dropna()
        if int(value) != 0
    ))
    values = np.empty(replicates, dtype=float)
    if len(training_seeds) <= 1:
        raw = paired[column].to_numpy(dtype=float)
        for index in range(replicates):
            values[index] = float(np.mean(
                raw[rng.randint(0, len(raw), size=len(raw))]
            ))
        return values
    groups = {
        seed: paired[paired["rl_training_seed"] == seed][column].to_numpy(dtype=float)
        for seed in training_seeds
    }
    for index in range(replicates):
        sampled_seeds = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        sampled: List[float] = []
        for seed in sampled_seeds:
            raw = groups[int(seed)]
            sampled.extend(raw[rng.randint(0, len(raw), size=len(raw))])
        values[index] = float(np.mean(sampled))
    return values


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def _one_sided_noninferiority_p(values: np.ndarray, margin: float) -> float:
    if len(values) < 2:
        return 1.0
    mean = float(np.mean(values))
    standard_error = float(stats.sem(values))
    if standard_error <= 0:
        return 0.0 if mean < margin else 1.0
    statistic = (mean - margin) / standard_error
    return float(stats.t.cdf(statistic, df=len(values) - 1))


def _one_sided_superiority_p(reductions: np.ndarray) -> float:
    if len(reductions) < 2:
        return 1.0
    mean = float(np.mean(reductions))
    standard_error = float(stats.sem(reductions))
    if standard_error <= 0:
        return 0.0 if mean > 0 else 1.0
    statistic = mean / standard_error
    return float(1.0 - stats.t.cdf(statistic, df=len(reductions) - 1))


def _holm(rows: List[Dict[str, Any]], p_key: str, output_key: str) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1][p_key]))
    running = 0.0
    count = len(rows)
    adjusted: Dict[int, float] = {}
    for rank, (index, row) in enumerate(ordered):
        value = min(1.0, (count - rank) * float(row[p_key]))
        running = max(running, value)
        adjusted[index] = running
    for index, row in enumerate(rows):
        row[output_key] = adjusted[index]


def _comparison_rows(frame: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    rows: List[Dict[str, Any]] = []
    primary_tests: List[Dict[str, Any]] = []
    methods = (PRIMARY_METHOD, "doet_rl")
    scenarios = [None] + sorted(
        value for value in frame["scenario_name"].unique()
        if value != "nominal"
    )
    bootstrap_record: Dict[str, Any] = {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling": (
            "paired episodes; learned methods resample RL training seeds then "
            "matched panels within seed"
        ),
        "comparisons": {},
    }
    for method in methods:
        for application in ("commercial", "humanitarian"):
            for scenario in scenarios:
                if scenario is None:
                    analysis_frame = frame[
                        frame["scenario_name"] != "nominal"
                    ]
                    paired = _paired(
                        analysis_frame, method, application, None
                    )
                else:
                    paired = _paired(frame, method, application, scenario)
                degradation_boot = _bootstrap_means(
                    paired, "relative_degradation", rng
                )
                difference_boot = _bootstrap_means(
                    paired, "loss_difference", rng
                )
                message_boot = _bootstrap_means(
                    paired, "total_communication_messages_reduction", rng
                )
                degradation_ci = _mean_ci(degradation_boot)
                difference_ci = _mean_ci(difference_boot)
                message_ci = _mean_ci(message_boot)
                difference = paired["loss_difference"].to_numpy(dtype=float)
                standard_deviation = float(np.std(difference, ddof=1))
                row = {
                    "method": method,
                    "benchmark": FIXED_METHOD,
                    "application": application,
                    "scenario": scenario or "all_non_nominal",
                    "paired_episodes": int(len(paired)),
                    "rl_training_seeds": int(paired["rl_training_seed"].nunique()),
                    "mean_loss_difference": float(difference.mean()),
                    "loss_difference_ci95_low": difference_ci[0],
                    "loss_difference_ci95_high": difference_ci[1],
                    "mean_relative_degradation": float(paired["relative_degradation"].mean()),
                    "relative_degradation_ci95_low": degradation_ci[0],
                    "relative_degradation_ci95_high": degradation_ci[1],
                    "noninferiority_upper_one_sided_95": float(np.quantile(degradation_boot, 0.95)),
                    "noninferiority_margin": PRIMARY_MARGIN,
                    "noninferior": bool(np.quantile(degradation_boot, 0.95) < PRIMARY_MARGIN),
                    "paired_win_rate": float((difference < 0).mean()),
                    "probability_of_superiority": float(
                        ((difference < 0).sum() + 0.5 * (difference == 0).sum())
                        / len(difference)
                    ),
                    "standardized_paired_effect": (
                        float(difference.mean() / standard_deviation)
                        if standard_deviation > 0 else 0.0
                    ),
                    "mean_message_reduction": float(
                        paired["total_communication_messages_reduction"].mean()
                    ),
                    "message_reduction_ci95_low": message_ci[0],
                    "message_reduction_ci95_high": message_ci[1],
                    "communication_superior": bool(message_ci[0] > 0),
                    "communication_target_20_percent": bool(
                        message_ci[0] > 0
                        and paired["total_communication_messages_reduction"].mean() >= 0.20
                    ),
                    "failed_pairs": int(paired.attrs.get("failed_pairs", 0)),
                }
                for metric in COMMUNICATION_METRICS:
                    reduction_column = metric + "_reduction"
                    reduction_boot = _bootstrap_means(
                        paired, reduction_column, rng
                    )
                    reduction_ci = _mean_ci(reduction_boot)
                    row["mean_" + reduction_column] = float(
                        paired[reduction_column].mean()
                    )
                    row[reduction_column + "_ci95_low"] = reduction_ci[0]
                    row[reduction_column + "_ci95_high"] = reduction_ci[1]
                rows.append(row)
                key = "%s|%s|%s" % (
                    method, application, scenario or "all_non_nominal"
                )
                bootstrap_record["comparisons"][key] = {
                    "relative_degradation_quantiles": {
                        "0.025": degradation_ci[0],
                        "0.95": float(np.quantile(degradation_boot, 0.95)),
                        "0.975": degradation_ci[1],
                    },
                    "message_reduction_quantiles": {
                        "0.025": message_ci[0], "0.975": message_ci[1]
                    },
                }
                if method == PRIMARY_METHOD and scenario is None:
                    primary_tests.extend([
                        {
                            "hypothesis": "H1_noninferiority",
                            "application": application,
                            "raw_p": _one_sided_noninferiority_p(
                                paired["relative_degradation"].to_numpy(dtype=float),
                                PRIMARY_MARGIN,
                            ),
                        },
                        {
                            "hypothesis": "H2_communication_superiority",
                            "application": application,
                            "raw_p": _one_sided_superiority_p(
                                paired["total_communication_messages_reduction"].to_numpy(dtype=float)
                            ),
                        },
                    ])
    _holm(primary_tests, "raw_p", "holm_adjusted_p")
    return rows, {**bootstrap_record, "primary_holm_tests": primary_tests}


def _nondominated(grouped: pd.DataFrame, cost: str) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for _, value in grouped.iterrows():
        dominated_by = []
        for _, other in grouped.iterrows():
            if other["method"] == value["method"]:
                continue
            if (
                other["primary_outcome"] <= value["primary_outcome"]
                and other[cost] <= value[cost]
                and (
                    other["primary_outcome"] < value["primary_outcome"]
                    or other[cost] < value[cost]
                )
            ):
                dominated_by.append(str(other["method"]))
        output[str(value["method"])] = sorted(dominated_by)
    return output


def _common_panel_subset(
    frame: pd.DataFrame,
    methods: set[str],
) -> pd.DataFrame:
    """Restrict unequal compute-capped methods to identical scenario panels."""

    keys = ["scenario_name", "seed", "n_agents"]
    available = set(frame["method"].astype(str))
    if not methods.issubset(available):
        raise ValueError(
            "matched-panel comparison is missing methods: %s"
            % sorted(methods - available)
        )
    panel_sets = []
    for method in sorted(methods):
        values = frame[frame["method"] == method]
        panel_sets.append(set(map(tuple, values[keys].itertuples(
            index=False, name=None
        ))))
    common = set.intersection(*panel_sets)
    if not common:
        raise ValueError("matched-panel comparison has no common panels")
    selected = frame[frame["method"].isin(methods)].copy()
    mask = [
        tuple(value) in common
        for value in selected[keys].itertuples(index=False, name=None)
    ]
    selected = selected.loc[mask]
    expected = len(common)
    counts = selected.groupby("method").size().to_dict()
    if any(int(counts.get(method, 0)) != expected for method in methods):
        raise ValueError(
            "matched-panel comparison contains duplicate or missing rows: %s"
            % counts
        )
    return selected


def _pareto_rows(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    non_nominal = frame[frame["scenario_name"] != "nominal"].copy()
    metrics = [
        "primary_outcome", "total_communication_messages", "prompt_tokens",
        "generated_tokens", "llm_calls", "llm_latency_seconds",
        "wall_clock_seconds",
    ]
    rows: List[Dict[str, Any]] = []
    for application, app in non_nominal.groupby("application"):
        methods = set(app["method"].astype(str))
        matched = _common_panel_subset(app, methods)
        grouped = matched.groupby("method")[metrics].mean().reset_index()
        dominance = {
            cost: _nondominated(grouped, cost)
            for cost in (
                "total_communication_messages", "prompt_tokens",
                "llm_calls", "llm_latency_seconds",
            )
        }
        for _, value in grouped.iterrows():
            method = str(value["method"])
            rows.append({
                "application": application,
                "method": method,
                **{"mean_" + metric: float(value[metric]) for metric in metrics},
                **{
                    "pareto_nondominated_loss_" + cost.replace(
                        "total_communication_", ""
                    ).replace("llm_", ""): not dominance[cost][method]
                    for cost in dominance
                },
                "dominated_by": ";".join(
                    dominance["total_communication_messages"][method]
                ),
            })
    return rows


def _hypervolume(points: np.ndarray) -> float:
    """Normalized 2-D minimization hypervolume against reference (1.05, 1.05)."""

    reference = 1.05
    if not len(points):
        return 0.0
    ordered = points[np.argsort(points[:, 0])]
    best_y = reference
    area = 0.0
    for x_value, y_value in ordered:
        x_value = min(reference, max(0.0, float(x_value)))
        y_value = min(reference, max(0.0, float(y_value)))
        if y_value < best_y:
            area += (reference - x_value) * (best_y - y_value)
            best_y = y_value
    return float(area)


def _frontier_rows(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    comparators = {
        "periodic_communication", "random_budget_matched",
        "learned_no_entropy", "kpi_cusum_trigger",
    }
    non_nominal = frame[frame["scenario_name"] != "nominal"]
    rows: List[Dict[str, Any]] = []
    for application, app in non_nominal.groupby("application"):
        matched = _common_panel_subset(
            app, comparators | {PRIMARY_METHOD}
        )
        grouped = matched.groupby("method").agg(
            primary_outcome=("primary_outcome", "mean"),
            total_communication_messages=("total_communication_messages", "mean"),
            prompt_tokens=("prompt_tokens", "mean"),
            llm_calls=("llm_calls", "mean"),
            llm_latency_seconds=("llm_latency_seconds", "mean"),
        ).reset_index()
        eligible = grouped[grouped["method"].isin(comparators | {PRIMARY_METHOD})]
        if set(eligible["method"]) != comparators | {PRIMARY_METHOD}:
            raise ValueError("Pareto frontier is missing a preregistered comparator")
        loss_min = float(eligible["primary_outcome"].min())
        loss_span = max(float(eligible["primary_outcome"].max()) - loss_min, 1e-12)
        for cost in (
            "total_communication_messages", "prompt_tokens",
            "llm_calls", "llm_latency_seconds",
        ):
            cost_min = float(eligible[cost].min())
            cost_span = max(float(eligible[cost].max()) - cost_min, 1e-12)
            normalized = eligible.assign(
                loss_norm=(eligible["primary_outcome"] - loss_min) / loss_span,
                cost_norm=(eligible[cost] - cost_min) / cost_span,
            )
            base = normalized[normalized["method"].isin(comparators)]
            with_doet = normalized[
                normalized["method"].isin(comparators | {PRIMARY_METHOD})
            ]
            base_hv = _hypervolume(
                base[["cost_norm", "loss_norm"]].to_numpy(dtype=float)
            )
            doet_hv = _hypervolume(
                with_doet[["cost_norm", "loss_norm"]].to_numpy(dtype=float)
            )
            rows.append({
                "application": application,
                "cost_metric": cost,
                "comparator_hypervolume": base_hv,
                "hypervolume_with_doet": doet_hv,
                "doet_hypervolume_gain": doet_hv - base_hv,
                "doet_improves_frontier": bool(doet_hv > base_hv + 1e-12),
                "normalization": "application-wise min-max; reference=(1.05,1.05)",
                "comparators": ";".join(sorted(comparators)),
            })
    return rows


def _read_events(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mechanistic(
    results_root: Path,
    frame: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
    episode_lookup = {
        path.parent.name: path
        for path in (results_root / "raw" / "holdout_locked").glob("*/episode.json")
    }
    rows: List[Dict[str, Any]] = []
    message_rows: List[Dict[str, Any]] = []
    case_candidates: Dict[str, List[Tuple[float, pd.DataFrame]]] = defaultdict(list)
    for _, summary in frame[frame["method"].isin([PRIMARY_METHOD, "doet_rl"])].iterrows():
        episode_path = episode_lookup[str(summary["run_id"])]
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        events = _read_events(episode_path.parent / "events.jsonl.gz")
        disruption_events = [
            event for event in events if event["kind"] == "disruption"
        ]
        disruption_step = min(
            (int(event["step"]) for event in disruption_events),
            default=None,
        )
        triggers = [event for event in events if event["kind"] == "coordination_trigger"]
        activations = [event for event in triggers if event["payload"].get("activated")]
        deactivations = [
            event for event in triggers if event["payload"].get("deactivated")
        ]
        first_activation = min(
            (int(event["step"]) for event in activations), default=None
        )
        first_deactivation = min(
            (
                int(event["step"]) for event in deactivations
                if first_activation is not None
                and int(event["step"]) >= first_activation
            ),
            default=None,
        )
        series = pd.DataFrame(episode["time_series"])
        pre = (
            series[series["step"] < disruption_step]["service_loss"]
            if disruption_step is not None else series["service_loss"]
        )
        collapse_threshold = (
            float(pre.mean() + 0.10) if len(pre) else float("inf")
        )
        collapse_steps = (
            series[
                (series["step"] >= disruption_step)
                & (series["service_loss"] > collapse_threshold)
            ]["step"]
            if disruption_step is not None else series.iloc[0:0]["step"]
        )
        collapse_step = int(collapse_steps.iloc[0]) if len(collapse_steps) else None
        messages = [event for event in events if event["kind"] == "message"]
        kinds = Counter(str(event["payload"].get("kind")) for event in messages)
        for kind, count in sorted(kinds.items()):
            message_rows.append({
                "run_id": summary["run_id"],
                "application": summary["application"],
                "scenario": summary["scenario_name"],
                "method": summary["method"],
                "message_type": kind,
                "count": count,
            })
        sketch_count = int(summary.get("monitor_sketch_messages", 0) or 0)
        if sketch_count:
            message_rows.append({
                "run_id": summary["run_id"],
                "application": summary["application"],
                "scenario": summary["scenario_name"],
                "method": summary["method"],
                "message_type": "entropy_sketch",
                "count": sketch_count,
            })
        affected_agents = set()
        for event in disruption_events:
            payload = event["payload"]
            affected_agents.update(str(value) for value in payload.get("affected", []))
            affected_agents.update(
                str(value) for value in payload.get("facility_outages", [])
            )
            coordinator = payload.get("coordinator_loss")
            if coordinator:
                affected_agents.add(str(coordinator))
            for edge in payload.get("route_closures", []):
                affected_agents.update(str(value) for value in edge)
        contact_start = first_activation if first_activation is not None else disruption_step
        contact_end = contact_start + 4 if contact_start is not None else None
        coordination_kinds = {
            "information_request", "disclosure", "offer", "counteroffer",
            "coalition_proposal", "entropy_alert", "fixed_status",
        }
        contacted_agents = {
            str(event["payload"].get("recipient"))
            for event in messages
            if contact_start is not None
            and contact_start <= int(event["step"]) <= int(contact_end)
            and str(event["payload"].get("kind")) in coordination_kinds
        }
        trigger_at_activation = [
            event for event in triggers
            if first_activation is not None
            and int(event["step"]) == first_activation
        ]
        high_surprisal = {
            str(event["actor"])
            for event in sorted(
                trigger_at_activation,
                key=lambda event: float(
                    event["payload"].get("local_surprisal", 0.0)
                ),
                reverse=True,
            )[:3]
        }
        row = {
            "run_id": summary["run_id"],
            "application": summary["application"],
            "scenario": summary["scenario_name"],
            "method": summary["method"],
            "environment_seed": int(summary["seed"]),
            "rl_training_seed": int(summary["rl_training_seed"]),
            "disruption_step": disruption_step,
            "first_activation_step": first_activation,
            "first_deactivation_step": first_deactivation,
            "activation_delay_from_disruption": (
                first_activation - disruption_step
                if first_activation is not None and disruption_step is not None
                else None
            ),
            "service_collapse_step": collapse_step,
            "activation_before_collapse": bool(
                first_activation is not None
                and (collapse_step is None or first_activation < collapse_step)
            ),
            "nominal_false_activation": bool(
                summary["scenario_name"] == "nominal" and first_activation is not None
            ),
            "trigger_activations": int(summary["trigger_activations"]),
            "quiet_mode_fraction": float(summary["quiet_mode_fraction"]),
            "targeted_mode_fraction": float(summary["targeted_mode_fraction"]),
            "crisis_mode_fraction": float(summary["crisis_mode_fraction"]),
            "mean_consensus_rmse": float(summary["mean_consensus_rmse"]),
            "entropy_alert_messages": int(kinds.get("entropy_alert", 0)),
            "offer_messages": int(kinds.get("offer", 0)),
            "counteroffer_messages": int(kinds.get("counteroffer", 0)),
            "coalition_messages": int(sum(
                count for kind, count in kinds.items()
                if "coalition" in kind
            )),
            "affected_agents": ";".join(sorted(affected_agents)),
            "contacted_agents_after_activation": ";".join(
                sorted(contacted_agents)
            ),
            "affected_contact_precision": float(
                len(contacted_agents & affected_agents)
                / max(len(contacted_agents), 1)
            ),
            "affected_contact_recall": float(
                len(contacted_agents & affected_agents)
                / max(len(affected_agents), 1)
            ),
            "top3_local_surprisal_contact_rate": float(
                len(contacted_agents & high_surprisal)
                / max(len(high_surprisal), 1)
            ),
            "useful_coalition_precision": float(
                summary.get("useful_coalition_precision", 0.0) or 0.0
            ),
            "coalition_recall_when_required": summary.get(
                "coalition_recall_when_required"
            ),
            "agreement_rate": float(summary.get("agreement_rate", 0.0) or 0.0),
            "counteroffers": int(summary.get("counteroffers", 0) or 0),
            "commitment_breaches": int(
                summary.get("commitment_breaches", 0) or 0
            ),
        }
        rows.append(row)
        if summary["method"] == PRIMARY_METHOD and summary["scenario_name"] != "nominal":
            trigger_by_step: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
            message_by_step = Counter()
            for event in triggers:
                trigger_by_step[int(event["step"])].append(event["payload"])
            for event in messages:
                message_by_step[int(event["step"])] += 1
            case = series.copy()
            if "mean_trigger_statistic" in case:
                case["mean_trigger_statistic_agents"] = case[
                    "mean_trigger_statistic"
                ]
            else:
                case["mean_trigger_statistic_agents"] = [
                    float(np.mean([
                        value["cumulative_statistic"]
                        for value in trigger_by_step.get(int(step), [])
                    ])) if trigger_by_step.get(int(step)) else 0.0
                    for step in case["step"]
                ]
            cumulative_communication = (
                case["messages"].fillna(0)
                + case["monitor_sketch_messages"].fillna(0)
            )
            case["operational_messages_this_step"] = (
                cumulative_communication.diff()
                .fillna(cumulative_communication.iloc[0])
                .clip(lower=0)
                .astype(int)
            )
            case["disruption_step"] = disruption_step
            case["first_activation_step"] = first_activation
            case["run_id"] = summary["run_id"]
            case_candidates[str(summary["application"])].append((
                abs(float(summary["primary_outcome"]) - float(
                    frame[
                        (frame["application"] == summary["application"])
                        & (frame["scenario_name"] == summary["scenario_name"])
                        & (frame["method"] == PRIMARY_METHOD)
                    ]["primary_outcome"].median()
                )),
                case,
            ))
    cases = {
        application: sorted(values, key=lambda item: item[0])[0][1]
        for application, values in case_candidates.items()
    }
    return rows, message_rows, cases


def _write_latex(path: Path, frame: pd.DataFrame, caption: str) -> None:
    path.write_text(
        frame.to_latex(index=False, escape=True, caption=caption, float_format="%.4g"),
        encoding="utf-8",
    )


def _partition_consensus_rows(
    frame: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> List[Dict[str, Any]]:
    partition_names = {"communication_partition", "compound_ood"}
    doet = frame[
        (frame["method"] == PRIMARY_METHOD)
        & frame["scenario_name"].isin(partition_names)
    ].copy()
    fixed = frame[
        (frame["method"] == FIXED_METHOD)
        & frame["scenario_name"].isin(partition_names)
    ][["application", "scenario_name", "seed", "n_agents", "primary_outcome"]]
    fixed = fixed.rename(columns={"primary_outcome": "fixed_primary_outcome"})
    joined = doet.merge(
        fixed,
        on=["application", "scenario_name", "seed", "n_agents"],
        validate="one_to_one",
    )
    joined["relative_degradation"] = (
        joined["primary_outcome"] - joined["fixed_primary_outcome"]
    ) / joined["fixed_primary_outcome"].abs().clip(lower=1e-9)
    rows: List[Dict[str, Any]] = []
    for application, values in joined.groupby("application"):
        x_values = values["mean_consensus_rmse"].to_numpy(dtype=float)
        for outcome in ("relative_degradation", "trigger_activations"):
            y_values = values[outcome].to_numpy(dtype=float)
            if len(values) >= 3 and np.std(x_values) > 0 and np.std(y_values) > 0:
                slope, intercept, correlation, p_value, standard_error = stats.linregress(
                    x_values, y_values
                )
            else:
                slope = intercept = correlation = p_value = standard_error = float("nan")
            rows.append({
                "application": application,
                "outcome": outcome,
                "episodes": int(len(values)),
                "consensus_rmse_mean": float(np.mean(x_values)),
                "slope_per_rmse": float(slope),
                "intercept": float(intercept),
                "pearson_r": float(correlation),
                "p_value_exploratory": float(p_value),
                "slope_standard_error": float(standard_error),
                "interpretation": "mechanistic exploratory; not a primary causal test",
            })
    return rows


def _flatten_trigger_parameters(results_root: Path) -> pd.DataFrame:
    selection = json.loads(
        (results_root / "validation" / "trigger_selection.json").read_text(
            encoding="utf-8"
        )
    )
    trigger = selection["selected_trigger"]
    parameters = trigger.get("parameters", {})
    return pd.DataFrame([
        {
            "selected_method_variant": selection["selected_method_variant"],
            "parameter": key,
            "value": value,
            "source": "validation only",
        }
        for key, value in sorted(parameters.items())
    ])


def _communication_budget_table(results_root: Path) -> pd.DataFrame:
    design = json.loads(
        (results_root / "protocol" / "holdout_design_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = Path(str(design["config_path"]))
    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    rows = []
    calibration = config.get("budget_match_calibration", {})
    for method in config["methods"]:
        rows.append({
            "method": method,
            "hard_operational_message_budget": config["communication_budget"],
            "fixed_broadcast_fanout": (
                config["fixed_broadcast_fanout"]
                if method == FIXED_METHOD else None
            ),
            "periodic_interval": (
                config["periodic_interval"]
                if method == "periodic_communication" else None
            ),
            "random_gate_probability": (
                config["random_gate_probability"]
                if method == "random_budget_matched" else None
            ),
            "entropy_sketches_counted": method in (
                "thermoagent", "doet_rule", "doet_rl"
            ),
            "accounting": (
                "all sketches, messages, bytes, prompt/generated tokens, "
                "LLM calls, and latency"
            ),
            "validation_target_counted_messages_per_episode": calibration.get(
                "target_counted_messages_per_episode"
            ),
            "validation_predicted_messages_per_episode": (
                calibration.get("predicted_periodic_messages_per_episode")
                if method == "periodic_communication" else
                calibration.get("predicted_random_messages_per_episode")
                if method == "random_budget_matched" else None
            ),
        })
    return pd.DataFrame(rows)


def run(results_root: Path) -> Dict[str, Any]:
    summary_path = results_root / "holdout_locked" / "episodes.csv"
    design_manifest_path = results_root / "protocol" / "holdout_design_manifest.json"
    freeze_path = results_root / "protocol" / "holdout_freeze.json"
    for path in (summary_path, design_manifest_path, freeze_path):
        if not path.exists():
            raise FileNotFoundError(path)
    frame = pd.read_csv(summary_path)
    design = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    if len(frame) != int(design["episode_count"]):
        raise ValueError("holdout episode count does not match frozen design")
    failed = frame[frame["status"] != "complete"].copy()
    if len(failed):
        # Failures remain in the public table, but primary estimates use no
        # silent imputation. The run therefore cannot be declared confirmatory.
        analysis_status = "completed_with_failed_runs"
    else:
        analysis_status = "complete"
    _numeric(frame, [
        "primary_outcome", "total_communication_messages",
        "total_communication_bytes", "prompt_tokens", "generated_tokens",
        "llm_calls", "llm_latency_seconds", "wall_clock_seconds",
        "mean_consensus_rmse", "trigger_activations", "quiet_mode_fraction",
        "targeted_mode_fraction", "crisis_mode_fraction", "tool_proposals",
        "communication_active_decision_epochs", "rl_training_seed",
    ])
    completed = frame[frame["status"] == "complete"].copy()
    comparison_rows, bootstrap = _comparison_rows(frame)
    comparisons = pd.DataFrame(comparison_rows)
    pareto = pd.DataFrame(_pareto_rows(completed))
    frontier = pd.DataFrame(_frontier_rows(completed))
    mechanistic_rows, message_rows, cases = _mechanistic(results_root, completed)
    mechanistic = pd.DataFrame(mechanistic_rows)
    message_types = pd.DataFrame(message_rows)
    partition_consensus = pd.DataFrame(
        _partition_consensus_rows(completed, comparisons)
    )
    training_variability = completed[
        completed["method"].isin(["learned_no_entropy", "thermoagent", "doet_rl"])
    ].groupby(["application", "method", "rl_training_seed"]).agg(
        episodes=("run_id", "count"),
        mean_primary_outcome=("primary_outcome", "mean"),
        mean_total_messages=("total_communication_messages", "mean"),
        mean_llm_calls=("llm_calls", "mean"),
    ).reset_index()

    primary = comparisons[
        (comparisons["method"] == PRIMARY_METHOD)
        & (comparisons["scenario"] == "all_non_nominal")
    ]
    holm = pd.DataFrame(bootstrap["primary_holm_tests"])
    confirmatory_complete = analysis_status == "complete"
    h1 = bool(
        confirmatory_complete
        and
        len(primary) == 2
        and primary["noninferior"].all()
        and holm[holm["hypothesis"] == "H1_noninferiority"]["holm_adjusted_p"].lt(0.05).all()
    )
    h2 = bool(
        confirmatory_complete
        and
        len(primary) == 2
        and primary["communication_target_20_percent"].all()
        and holm[holm["hypothesis"] == "H2_communication_superiority"]["holm_adjusted_p"].lt(0.05).all()
    )
    primary_pareto = pareto[pareto["method"] == PRIMARY_METHOD]
    h3 = bool(
        confirmatory_complete
        and
        len(primary_pareto) == 2
        and primary_pareto["pareto_nondominated_loss_messages"].all()
        and len(frontier) == 8
        and frontier["doet_improves_frontier"].all()
    )
    non_nominal_mechanistic = mechanistic[
        (mechanistic["method"] == PRIMARY_METHOD)
        & (mechanistic["scenario"] != "nominal")
    ]
    nominal_mechanistic = mechanistic[
        (mechanistic["method"] == PRIMARY_METHOD)
        & (mechanistic["scenario"] == "nominal")
    ]
    h4 = bool(
        confirmatory_complete
        and
        len(non_nominal_mechanistic)
        and non_nominal_mechanistic["activation_before_collapse"].mean() >= 0.75
        and len(nominal_mechanistic)
        and nominal_mechanistic["nominal_false_activation"].mean() <= 0.10
    )
    partition_primary = comparisons[
        (comparisons["method"] == PRIMARY_METHOD)
        & comparisons["scenario"].isin(["communication_partition", "compound_ood"])
    ]
    consensus_loss = partition_consensus[
        partition_consensus["outcome"] == "relative_degradation"
    ]
    predictable_consensus_relation = bool(
        len(consensus_loss) == 2
        and np.isfinite(consensus_loss["slope_per_rmse"]).all()
        and np.isfinite(consensus_loss["pearson_r"]).all()
        and consensus_loss["slope_per_rmse"].gt(0).all()
        and consensus_loss["pearson_r"].ge(0.20).all()
    )
    h5 = bool(
        confirmatory_complete
        and len(partition_primary) == 4
        and partition_primary["noninferior"].all()
        and predictable_consensus_relation
    )
    h6 = bool(h1 and h2)
    hypotheses = pd.DataFrame([
        {"hypothesis": "H1", "outcome": "supported" if h1 else "unsupported", "criterion": "DOET-rule non-inferior to fixed in both applications after Holm correction"},
        {"hypothesis": "H2", "outcome": "supported" if h2 else "unsupported", "criterion": "message reduction CI excludes zero and mean reduction >=20% in both applications after Holm correction"},
        {"hypothesis": "H3", "outcome": "supported" if h3 else "unsupported", "criterion": "DOET-rule is loss-message nondominated and strictly increases the frozen normalized frontier hypervolume for messages, prompt tokens, calls, and latency in both applications"},
        {"hypothesis": "H4", "outcome": "supported" if h4 else "unsupported", "criterion": ">=75% activation before collapse and <=10% nominal episode false activation"},
        {"hypothesis": "H5", "outcome": "supported" if h5 else "unsupported", "criterion": "non-inferior in partition and compound-partition regimes in both applications, with positive consensus-RMSE/degradation slope and Pearson r >=0.20 in each application"},
        {"hypothesis": "H6", "outcome": "supported" if h6 else "unsupported", "criterion": "H1 and H2 supported in both applications"},
    ])

    processed = results_root / "processed"
    statistics = results_root / "statistics"
    tables = results_root / "tables"
    for directory in (processed, statistics, tables):
        directory.mkdir(parents=True, exist_ok=True)
    trigger_parameters = _flatten_trigger_parameters(results_root)
    communication_budgets = _communication_budget_table(results_root)
    design_table = pd.read_csv(
        results_root / "protocol" / "holdout_design.csv"
    )
    design_summary = design_table.groupby(
        ["application", "scenario", "method", "topology", "communication", "disruption"],
        dropna=False,
    ).agg(
        episodes=("environment_seed", "count"),
        unique_environment_seeds=("environment_seed", "nunique"),
        unique_rl_training_seeds=("rl_training_seed", "nunique"),
    ).reset_index()
    holdout_summary = completed.groupby(
        ["application", "scenario_name", "method"], dropna=False
    ).agg(
        episodes=("run_id", "count"),
        primary_outcome_mean=("primary_outcome", "mean"),
        primary_outcome_sd=("primary_outcome", "std"),
        total_messages_mean=("total_communication_messages", "mean"),
        prompt_tokens_mean=("prompt_tokens", "mean"),
        generated_tokens_mean=("generated_tokens", "mean"),
        llm_calls_mean=("llm_calls", "mean"),
        latency_seconds_mean=("llm_latency_seconds", "mean"),
    ).reset_index()
    budget_methods = {
        "periodic_communication", "random_budget_matched",
        "kpi_cusum_trigger", PRIMARY_METHOD,
    }
    matched_budget_parts = []
    for _, app in completed[
        completed["scenario_name"] != "nominal"
    ].groupby("application"):
        matched_budget_parts.append(_common_panel_subset(app, budget_methods))
    matched_budget = pd.concat(matched_budget_parts, ignore_index=True)
    non_nominal_budget = matched_budget.groupby(
        ["application", "method"], dropna=False
    ).agg(
        mean_counted_messages=("total_communication_messages", "mean"),
        mean_counted_bytes=("total_communication_bytes", "mean"),
        mean_prompt_tokens=("prompt_tokens", "mean"),
        mean_llm_calls=("llm_calls", "mean"),
    ).reset_index()
    budget_target = non_nominal_budget[
        non_nominal_budget["method"] == PRIMARY_METHOD
    ][["application", "mean_counted_messages"]].rename(
        columns={"mean_counted_messages": "doet_target_messages"}
    )
    achieved_budget_match = non_nominal_budget[
        non_nominal_budget["method"].isin(budget_methods)
    ].merge(budget_target, on="application", validate="many_to_one")
    achieved_budget_match["message_budget_relative_mismatch"] = (
        achieved_budget_match["mean_counted_messages"]
        - achieved_budget_match["doet_target_messages"]
    ) / achieved_budget_match["doet_target_messages"].clip(lower=1e-9)
    validation_ablation = pd.read_csv(
        results_root / "validation" / "trigger_candidate_comparison.csv"
    )
    locked_ablation_path = results_root / "ablations" / "episodes.csv"
    if locked_ablation_path.exists():
        locked_ablation = pd.read_csv(locked_ablation_path)
        locked_ablation_summary = locked_ablation.groupby(
            ["application", "scenario_name", "method", "method_variant"],
            dropna=False,
        ).agg(
            episodes=("run_id", "count"),
            complete=("status", lambda value: int((value == "complete").sum())),
            primary_outcome_mean=("primary_outcome", "mean"),
            total_messages_mean=("total_communication_messages", "mean"),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            llm_calls_mean=("llm_calls", "mean"),
        ).reset_index()
    else:
        locked_ablation_summary = pd.DataFrame(columns=[
            "application", "scenario_name", "method", "method_variant",
            "episodes", "complete", "primary_outcome_mean",
            "total_messages_mean", "prompt_tokens_mean", "llm_calls_mean",
        ])
    output_frames = {
        processed / "holdout_results.csv": completed,
        processed / "mechanistic_events.csv": mechanistic,
        processed / "message_type_counts.csv": message_types,
        statistics / "main_paired_comparisons.csv": comparisons,
        statistics / "pareto_points.csv": pareto,
        statistics / "pareto_frontier_hypervolume.csv": frontier,
        statistics / "training_seed_variability.csv": training_variability,
        statistics / "partition_consensus_relationship.csv": partition_consensus,
        tables / "experimental_design.csv": design_summary,
        tables / "trigger_parameters.csv": trigger_parameters,
        tables / "communication_budgets.csv": communication_budgets,
        tables / "achieved_budget_match.csv": achieved_budget_match,
        tables / "holdout_results.csv": holdout_summary,
        tables / "main_paired_comparisons.csv": comparisons,
        tables / "trigger_ablation_results.csv": validation_ablation,
        tables / "extended_ablation_results.csv": locked_ablation_summary,
        tables / "noninferiority_analysis.csv": comparisons[
            [column for column in comparisons if "noninfer" in column or column in (
                "method", "application", "scenario", "paired_episodes",
                "mean_relative_degradation", "relative_degradation_ci95_low",
                "relative_degradation_ci95_high",
            )]
        ],
        tables / "communication_reductions.csv": comparisons[
            [column for column in comparisons if "reduction" in column or column in (
                "method", "application", "scenario", "paired_episodes",
                "communication_superior", "communication_target_20_percent",
            )]
        ],
        tables / "pareto_operating_points.csv": pareto,
        tables / "rl_training_seed_results.csv": training_variability,
        tables / "failed_runs.csv": failed,
        tables / "hypothesis_outcomes.csv": hypotheses,
    }
    for path, value in output_frames.items():
        value.to_csv(path, index=False)
    monitoring_source = results_root / "monitoring" / "monitoring_baselines.csv"
    if monitoring_source.exists():
        monitoring_table = pd.read_csv(monitoring_source)
        monitoring_table.to_csv(
            tables / "monitoring_comparison.csv", index=False
        )
        output_frames[tables / "monitoring_comparison.csv"] = monitoring_table
    for application, case in cases.items():
        case.to_csv(processed / (application + "_event_case_study.csv"), index=False)
    bootstrap_path = statistics / "hierarchical_bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_latex(
        tables / "noninferiority_analysis.tex",
        output_frames[tables / "noninferiority_analysis.csv"],
        "DOET non-inferiority to always-on fixed communication.",
    )
    _write_latex(
        tables / "hypothesis_outcomes.tex",
        hypotheses,
        "Preregistered DOET hypothesis outcomes.",
    )
    _write_latex(
        tables / "experimental_design.tex",
        design_summary,
        "Locked DOET experimental design.",
    )
    compute = {
        "episodes": int(len(frame)),
        "failed_episodes": int(len(failed)),
        "wall_clock_hours_sum": float(completed["wall_clock_seconds"].sum() / 3600.0),
        "llm_calls": int(completed["llm_calls"].sum()),
        "prompt_tokens": int(completed["prompt_tokens"].sum()),
        "generated_tokens": int(completed["generated_tokens"].sum()),
        "messages_including_sketches": int(completed["total_communication_messages"].sum()),
        "structured_bytes": int(completed["total_communication_bytes"].sum()),
        "approximate_gpu_cost_usd_at_0_34_per_hour": float(
            completed["wall_clock_seconds"].sum() / 3600.0 * 0.34
        ),
    }
    compute_path = tables / "compute_token_accounting.json"
    compute_path.write_text(
        json.dumps(compute, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compute_by_method = completed.groupby(
        ["application", "method"], dropna=False
    ).agg(
        episodes=("run_id", "count"),
        wall_clock_hours=("wall_clock_seconds", lambda value: value.sum() / 3600.0),
        llm_calls=("llm_calls", "sum"),
        prompt_tokens=("prompt_tokens", "sum"),
        generated_tokens=("generated_tokens", "sum"),
        messages_including_sketches=("total_communication_messages", "sum"),
        structured_bytes=("total_communication_bytes", "sum"),
    ).reset_index()
    compute_by_method["approximate_gpu_cost_usd_at_0_34_per_hour"] = (
        compute_by_method["wall_clock_hours"] * 0.34
    )
    compute_by_method_path = tables / "compute_token_accounting.csv"
    compute_by_method.to_csv(compute_by_method_path, index=False)
    output_frames[compute_by_method_path] = compute_by_method
    stage_compute_rows = []
    for stage in ("profile_v2", "validation", "holdout_locked", "ablations"):
        candidates = sorted((results_root / "manifests").glob(stage + "_sweep*.json"))
        if not candidates:
            continue
        # The canonical first manifest is the full initial sweep. Resume
        # manifests are retained separately but must not be double-counted.
        canonical = results_root / "manifests" / (stage + "_sweep.json")
        selected_manifest = canonical if canonical.exists() else candidates[0]
        record = json.loads(selected_manifest.read_text(encoding="utf-8"))
        stage_gpu_hours = max(
            float(record.get(
                "wall_clock_seconds_including_model_load", 0.0
            )) / 3600.0,
            float(record.get("cumulative_episode_single_gpu_hours", 0.0)),
        )
        stage_compute_rows.append({
            "stage": stage,
            "manifest": str(selected_manifest.relative_to(results_root)),
            "episodes_planned": int(record.get("episodes_planned", 0)),
            "episodes_complete": int(record.get("episodes_complete", 0)),
            "episodes_failed": int(record.get("episodes_failed", 0)),
            "single_gpu_hours_including_model_load": stage_gpu_hours,
            "llm_calls": int(record.get("llm_calls", 0)),
            "prompt_tokens": int(record.get("prompt_tokens", 0)),
            "generated_tokens": int(record.get("generated_tokens", 0)),
        })
    training_manifest_path = results_root / "training" / "training_manifest.json"
    if training_manifest_path.is_file():
        training_record = json.loads(
            training_manifest_path.read_text(encoding="utf-8")
        )
        training_hours = float(training_record.get(
            "single_gpu_hours_reserved",
            float(training_record.get("wall_clock_seconds", 0.0) or 0.0)
            / 3600.0,
        ))
        stage_compute_rows.append({
            "stage": "training",
            "manifest": str(training_manifest_path.relative_to(results_root)),
            "episodes_planned": int(training_record.get(
                "planned_trainings", 0
            )),
            "episodes_complete": int(training_record.get(
                "completed_trainings", 0
            )),
            "episodes_failed": int(training_record.get(
                "failed_trainings", 0
            )),
            "single_gpu_hours_including_model_load": training_hours,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
        })
    model_smoke_path = results_root / "logs" / "setup" / "model_smoke.json"
    if model_smoke_path.is_file():
        smoke = json.loads(model_smoke_path.read_text(encoding="utf-8"))
        smoke_hours = (
            float(smoke.get("load_seconds", 0.0) or 0.0)
            + float(smoke.get("batched_inference_seconds", 0.0) or 0.0)
        ) / 3600.0
        stage_compute_rows.append({
            "stage": "model_smoke",
            "manifest": str(model_smoke_path.relative_to(results_root)),
            "episodes_planned": 1,
            "episodes_complete": int(smoke.get("status") == "complete"),
            "episodes_failed": int(smoke.get("status") != "complete"),
            "single_gpu_hours_including_model_load": smoke_hours,
            "llm_calls": int(smoke.get("batch_size", 0)),
            "prompt_tokens": int(smoke.get("prompt_tokens", 0)),
            "generated_tokens": int(smoke.get("generated_tokens", 0)),
        })
    stage_compute = pd.DataFrame(stage_compute_rows)
    stage_compute_path = tables / "total_compute_accounting.csv"
    stage_compute.to_csv(stage_compute_path, index=False)
    total_gpu_hours = float(
        stage_compute["single_gpu_hours_including_model_load"].sum()
        if len(stage_compute) else 0.0
    )
    total_compute = {
        "included_gpu_stages": stage_compute["stage"].tolist(),
        "additional_single_gpu_hours_including_model_load": total_gpu_hours,
        "approximate_gpu_cost_usd_at_0_34_per_hour": total_gpu_hours * 0.34,
        "llm_calls": int(stage_compute["llm_calls"].sum()) if len(stage_compute) else 0,
        "prompt_tokens": int(stage_compute["prompt_tokens"].sum()) if len(stage_compute) else 0,
        "generated_tokens": int(stage_compute["generated_tokens"].sum()) if len(stage_compute) else 0,
        "training_note": (
            "staged PPO used the deterministic mock planner and is CPU-bound, "
            "but its elapsed time on the paid single-GPU Pod is included in "
            "resource and cost accounting"
        ),
        "budget_limit_single_gpu_hours": 35.0,
    }
    total_compute_path = tables / "total_compute_accounting.json"
    total_compute_path.write_text(
        json.dumps(total_compute, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_frames[stage_compute_path] = stage_compute
    artifacts = sorted(
        list(output_frames) + [
            bootstrap_path, compute_path, compute_by_method_path,
            stage_compute_path, total_compute_path,
            *(processed / (app + "_event_case_study.csv") for app in cases),
        ]
    )
    manifest = {
        "status": analysis_status,
        "generated_at": _utc_now(),
        "primary_method": PRIMARY_METHOD,
        "primary_benchmark": FIXED_METHOD,
        "noninferiority_margin": PRIMARY_MARGIN,
        "primary_experimental_unit": "complete multi-agent episode",
        "holdout_input_checksum": sha256_file(summary_path),
        "freeze_checksum": sha256_file(freeze_path),
        "episode_count": int(len(frame)),
        "failed_episode_count": int(len(failed)),
        "hypotheses": hypotheses.to_dict(orient="records"),
        "outputs": {
            str(path.relative_to(results_root)): sha256_file(path)
            for path in artifacts if path.exists()
        },
    }
    manifest_path = statistics / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
