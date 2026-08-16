"""Development-only precision and compute planning for the frozen V6 design."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .v5_experiments import atomic_json, write_csv


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def run_power_and_compute_plan(repository: Path, results_root: Path) -> Dict[str, Any]:
    pilot = pd.read_csv(
        results_root / "pilots" / "pilot_v11_timing_final_analysis" / "risk_coverage_panel_results.csv"
    )
    pilot = pilot[
        (pilot.coverage_target == 0.5)
        & (pilot.information_condition == "private_fragmented")
        & pilot.application.isin(["humanitarian", "utility_restoration"])
    ]
    baseline = str(json.loads(
        (results_root / "pilots" / "pilot_v11_timing_final_analysis" / "risk_analysis.json").read_text(encoding="utf-8")
    )["selected_strongest_nonentropic_baseline"])
    rows: List[Dict[str, Any]] = []
    planned = {"development": 210, "validation": 120, "holdout": 144}
    for application in ("humanitarian", "utility_restoration"):
        base = pilot[(pilot.application == application) & (pilot.feature_block == baseline)]
        method = pilot[(pilot.application == application) & (pilot.feature_block == "combined_generalized_entropic")]
        paired = base[["cluster_id", "harmful_action_rate"]].merge(
            method[["cluster_id", "harmful_action_rate"]], on="cluster_id",
            suffixes=("_baseline", "_combined"), validate="one_to_one",
        )
        differences = paired.harmful_action_rate_baseline - paired.harmful_action_rate_combined
        observed_sd = float(differences.std(ddof=1))
        conservative_sd = max(observed_sd, 0.10)
        for stage, panels in planned.items():
            standard_error = conservative_sd / math.sqrt(panels)
            # Probability that a two-sided 95% normal interval excludes zero
            # when the true practical effect is the frozen 0.03 threshold.
            power = _normal_cdf(0.03 / standard_error - 1.959963984540054)
            rows.append({
                "application": application, "stage": stage,
                "independent_panels_per_information_condition": panels,
                "pilot_panels": len(paired), "pilot_mean_difference": float(differences.mean()),
                "pilot_sd": observed_sd, "planning_sd_floor": conservative_sd,
                "practical_effect": 0.03, "approximate_power": power,
                "approximate_95pct_half_width": 1.959963984540054 * standard_error,
                "method": "normal approximation from paired pilot SD with 0.10 floor",
            })
    write_csv(results_root / "protocol" / "development_power_precision_plan.csv", rows)
    # The GPU estimate is deliberately conservative. The prior pinned-Qwen
    # throughput was 108 calls / 109.34 wall seconds including model load.
    projection_rows = [
        {"component": "model_load_and_smoke", "gpu_hours": 0.35, "llm_calls": 30, "prompt_tokens": 30000, "generated_tokens": 3000, "storage_gib": 0.1},
        {"component": "five_method_x_five_seed_sequential_PPO", "gpu_hours": 8.0, "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "storage_gib": 0.5},
        {"component": "150_episode_real_Qwen_qualification", "gpu_hours": 1.25, "llm_calls": 2700, "prompt_tokens": 2700000, "generated_tokens": 250000, "storage_gib": 0.4},
        {"component": "validation_if_unlocked", "gpu_hours": 0.0, "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "storage_gib": 1.2},
        {"component": "sealed_holdout_if_unlocked", "gpu_hours": 0.0, "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "storage_gib": 1.4},
        {"component": "profiling_analysis_rendering", "gpu_hours": 0.4, "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "storage_gib": 0.3},
    ]
    subtotal = sum(value["gpu_hours"] for value in projection_rows)
    reserve = 0.15 * subtotal
    projection_rows.append({"component": "15_percent_safety_reserve", "gpu_hours": reserve, "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "storage_gib": 0.4})
    write_csv(results_root / "protocol" / "compute_projection.csv", projection_rows)
    report = {
        "planning_evidence": "V6 pilot only; calculated before formal development outcomes",
        "planned_independent_panels_per_application_information_condition": planned,
        "power_rows": rows,
        "projected_gpu_hours_including_reserve": subtotal + reserve,
        "projected_cost_usd_at_0_34_per_hour": (subtotal + reserve) * 0.34,
        "projected_llm_calls": sum(value["llm_calls"] for value in projection_rows),
        "projected_prompt_tokens": sum(value["prompt_tokens"] for value in projection_rows),
        "projected_generated_tokens": sum(value["generated_tokens"] for value in projection_rows),
        "projected_storage_gib": sum(value["storage_gib"] for value in projection_rows),
        "gpu_hour_cap": 50.0, "cost_cap_usd": 40.0,
        "within_caps": bool(subtotal + reserve <= 50.0 and (subtotal + reserve) * 0.34 <= 40.0),
    }
    atomic_json(results_root / "protocol" / "power_and_compute_plan.json", report)
    return report
