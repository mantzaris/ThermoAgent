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

from .events import sha256_file


PRIMARY_METHOD = "doet_rule"
FIXED_METHOD = "fixed_always_on"
PRIMARY_MARGIN = 0.02
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260813


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
    left = subset[subset["method"] == method][keys + [
        "rl_training_seed", *metrics
    ]].copy()
    right = subset[subset["method"] == FIXED_METHOD][keys + metrics].copy()
    paired = left.merge(right, on=keys, suffixes=("_method", "_fixed"), validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(
            "incomplete matched panel for %s/%s/%s: method=%d fixed=%d paired=%d"
            % (method, application, scenario, len(left), len(right), len(paired))
        )
    paired["loss_difference"] = (
        paired["primary_outcome_method"] - paired["primary_outcome_fixed"]
    )
    paired["relative_degradation"] = paired["loss_difference"] / np.maximum(
        np.abs(paired["primary_outcome_fixed"]), 1e-9
    )
    for metric in (
        "total_communication_messages", "total_communication_bytes",
        "prompt_tokens", "generated_tokens", "llm_calls",
        "llm_latency_seconds", "wall_clock_seconds",
    ):
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
    replicates: int = BOOTSTRAP_REPLICATES,
) -> np.ndarray:
    if paired.empty:
        raise ValueError("bootstrap requires episode pairs")
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
                paired = _paired(frame, method, application, scenario)
                if scenario is None:
                    paired = paired[paired["scenario_name"] != "nominal"].copy()
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
                    "failed_pairs": 0,
                }
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


def _pareto_rows(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    non_nominal = frame[frame["scenario_name"] != "nominal"].copy()
    metrics = [
        "primary_outcome", "total_communication_messages", "prompt_tokens",
        "generated_tokens", "llm_calls", "llm_latency_seconds",
        "wall_clock_seconds",
    ]
    rows: List[Dict[str, Any]] = []
    for application, app in non_nominal.groupby("application"):
        grouped = app.groupby("method")[metrics].mean().reset_index()
        for _, value in grouped.iterrows():
            dominated_by = []
            for _, other in grouped.iterrows():
                if other["method"] == value["method"]:
                    continue
                if (
                    other["primary_outcome"] <= value["primary_outcome"]
                    and other["total_communication_messages"] <= value["total_communication_messages"]
                    and (
                        other["primary_outcome"] < value["primary_outcome"]
                        or other["total_communication_messages"] < value["total_communication_messages"]
                    )
                ):
                    dominated_by.append(str(other["method"]))
            rows.append({
                "application": application,
                "method": value["method"],
                **{"mean_" + metric: float(value[metric]) for metric in metrics},
                "pareto_nondominated_loss_messages": not dominated_by,
                "dominated_by": ";".join(sorted(dominated_by)),
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
        disruption_step = max(2, int(episode["time_series"][-1]["step"] + 1) // 3)
        triggers = [event for event in events if event["kind"] == "coordination_trigger"]
        activations = [event for event in triggers if event["payload"].get("activated")]
        first_activation = min(
            (int(event["step"]) for event in activations), default=None
        )
        series = pd.DataFrame(episode["time_series"])
        pre = series[series["step"] < disruption_step]["service_loss"]
        collapse_threshold = float(pre.mean() + 0.10) if len(pre) else 1.0
        collapse_steps = series[
            (series["step"] >= disruption_step)
            & (series["service_loss"] >= collapse_threshold)
        ]["step"]
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
        row = {
            "run_id": summary["run_id"],
            "application": summary["application"],
            "scenario": summary["scenario_name"],
            "method": summary["method"],
            "environment_seed": int(summary["seed"]),
            "rl_training_seed": int(summary["rl_training_seed"]),
            "disruption_step": disruption_step,
            "first_activation_step": first_activation,
            "activation_delay_from_disruption": (
                first_activation - disruption_step
                if first_activation is not None else None
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
            case["mean_trigger_statistic_agents"] = [
                float(np.mean([
                    value["cumulative_statistic"]
                    for value in trigger_by_step.get(int(step), [])
                ])) if trigger_by_step.get(int(step)) else 0.0
                for step in case["step"]
            ]
            case["activated_agents"] = [
                sum(bool(value.get("activated")) for value in trigger_by_step.get(int(step), []))
                for step in case["step"]
            ]
            case["operational_messages_this_step"] = [
                int(message_by_step[int(step)]) for step in case["step"]
            ]
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
    comparison_rows, bootstrap = _comparison_rows(completed)
    comparisons = pd.DataFrame(comparison_rows)
    pareto = pd.DataFrame(_pareto_rows(completed))
    mechanistic_rows, message_rows, cases = _mechanistic(results_root, completed)
    mechanistic = pd.DataFrame(mechanistic_rows)
    message_types = pd.DataFrame(message_rows)
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
    h1 = bool(
        len(primary) == 2
        and primary["noninferior"].all()
        and holm[holm["hypothesis"] == "H1_noninferiority"]["holm_adjusted_p"].lt(0.05).all()
    )
    h2 = bool(
        len(primary) == 2
        and primary["communication_target_20_percent"].all()
        and holm[holm["hypothesis"] == "H2_communication_superiority"]["holm_adjusted_p"].lt(0.05).all()
    )
    primary_pareto = pareto[pareto["method"] == PRIMARY_METHOD]
    h3 = bool(
        len(primary_pareto) == 2
        and primary_pareto["pareto_nondominated_loss_messages"].all()
    )
    non_nominal_mechanistic = mechanistic[
        mechanistic["scenario"] != "nominal"
    ]
    h4 = bool(
        len(non_nominal_mechanistic)
        and non_nominal_mechanistic["activation_before_collapse"].mean() >= 0.75
        and mechanistic[mechanistic["scenario"] == "nominal"]["nominal_false_activation"].mean() <= 0.10
    )
    partition_primary = comparisons[
        (comparisons["method"] == PRIMARY_METHOD)
        & comparisons["scenario"].isin(["communication_partition", "compound_ood"])
    ]
    h5 = bool(len(partition_primary) == 4 and partition_primary["noninferior"].all())
    h6 = bool(h1 and h2)
    hypotheses = pd.DataFrame([
        {"hypothesis": "H1", "outcome": "supported" if h1 else "unsupported", "criterion": "DOET-rule non-inferior to fixed in both applications after Holm correction"},
        {"hypothesis": "H2", "outcome": "supported" if h2 else "unsupported", "criterion": "message reduction CI excludes zero and mean reduction >=20% in both applications after Holm correction"},
        {"hypothesis": "H3", "outcome": "supported" if h3 else "unsupported", "criterion": "DOET-rule is loss-message Pareto nondominated in both applications"},
        {"hypothesis": "H4", "outcome": "supported" if h4 else "unsupported", "criterion": ">=75% activation before collapse and <=10% nominal episode false activation"},
        {"hypothesis": "H5", "outcome": "supported" if h5 else "unsupported", "criterion": "non-inferior in partition and compound-partition regimes in both applications"},
        {"hypothesis": "H6", "outcome": "supported" if h6 else "unsupported", "criterion": "H1 and H2 supported in both applications"},
    ])

    processed = results_root / "processed"
    statistics = results_root / "statistics"
    tables = results_root / "tables"
    for directory in (processed, statistics, tables):
        directory.mkdir(parents=True, exist_ok=True)
    output_frames = {
        processed / "holdout_results.csv": completed,
        processed / "mechanistic_events.csv": mechanistic,
        processed / "message_type_counts.csv": message_types,
        statistics / "main_paired_comparisons.csv": comparisons,
        statistics / "pareto_points.csv": pareto,
        statistics / "training_seed_variability.csv": training_variability,
        tables / "noninferiority_analysis.csv": comparisons[
            [column for column in comparisons if "noninfer" in column or column in (
                "method", "application", "scenario", "paired_episodes",
                "mean_relative_degradation", "relative_degradation_ci95_low",
                "relative_degradation_ci95_high",
            )]
        ],
        tables / "communication_reductions.csv": comparisons[
            [column for column in comparisons if "message" in column or column in (
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
    compute = {
        "episodes": int(len(frame)),
        "failed_episodes": int(len(failed)),
        "wall_clock_hours_sum": float(completed["wall_clock_seconds"].sum() / 3600.0),
        "llm_calls": int(completed["llm_calls"].sum()),
        "prompt_tokens": int(completed["prompt_tokens"].sum()),
        "generated_tokens": int(completed["generated_tokens"].sum()),
        "messages_including_sketches": int(completed["total_communication_messages"].sum()),
        "structured_bytes": int(completed["total_communication_bytes"].sum()),
    }
    compute_path = tables / "compute_token_accounting.json"
    compute_path.write_text(
        json.dumps(compute, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = sorted(
        list(output_frames) + [
            bootstrap_path, compute_path,
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
