"""Compact repository-facing aggregation; raw LLM records remain external."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .statistics import safe_logit
from .workflow import artifact_root, atomic_csv, atomic_json, sha256_file, utc_now


def _directory_digest(root: Path) -> Dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    if not root.exists():
        return {"relative_path": root.name, "file_count": 0, "bytes": 0, "sha256_tree": None}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        digest.update(relative.encode("utf-8") + b"\0" + file_digest.encode("ascii") + b"\0")
        count += 1
        size += path.stat().st_size
    return {"relative_path": root.name, "file_count": count, "bytes": size, "sha256_tree": digest.hexdigest()}


def _qualification_cluster_rows(frame: pd.DataFrame) -> List[Dict[str, object]]:
    valid = frame[frame["valid_after_repair"] == 1].copy()
    baseline = valid[valid["condition"] == "no_message"][
        ["cluster_id", "probability_right", "belief_right", "action_choice", "commitment_status"]
    ].rename(
        columns={
            "probability_right": "baseline_probability_right",
            "belief_right": "baseline_belief_right",
            "action_choice": "baseline_action_choice",
            "commitment_status": "baseline_commitment_status",
        }
    )
    values = valid.merge(baseline, on="cluster_id", how="inner", validate="many_to_one")
    values["logit_change"] = safe_logit(values["probability_right"].to_numpy(float)) - safe_logit(
        values["baseline_probability_right"].to_numpy(float)
    )
    values["signed_logit_change"] = values["logit_change"] * values["expected_direction"]
    values["absolute_probability_change"] = np.abs(
        values["probability_right"].to_numpy(float) - values["baseline_probability_right"].to_numpy(float)
    )
    values["belief_switch"] = (values["belief_right"] != values["baseline_belief_right"]).astype(int)
    values["action_switch"] = (values["action_choice"] != values["baseline_action_choice"]).astype(int)
    values["commitment_change"] = (values["commitment_status"] != values["baseline_commitment_status"]).astype(int)
    columns = [
        "cluster_id",
        "domain",
        "paraphrase",
        "option_order_right_first",
        "private_observation",
        "condition",
        "expected_direction",
        "nominal_reliability",
        "probability_right",
        "baseline_probability_right",
        "logit_change",
        "signed_logit_change",
        "absolute_probability_change",
        "belief_switch",
        "action_switch",
        "commitment_change",
    ]
    return values[columns].to_dict(orient="records")


def _qualification_exploratory_rows(compact: pd.DataFrame) -> List[Dict[str, object]]:
    """Post-qualification diagnostics; these rows never participate in the frozen gate."""

    selected = compact[
        compact["condition"].str.startswith("single_") & (compact["expected_direction"] != 0)
    ].copy()
    rows: List[Dict[str, object]] = []

    def add(scope: str, group: pd.DataFrame) -> None:
        rows.extend(
            [
                {
                    "scope": scope,
                    "metric": "signed_logit_change",
                    "estimate": float(group["signed_logit_change"].mean()),
                    "rows": int(len(group)),
                    "independent_clusters": int(group["cluster_id"].nunique()),
                },
                {
                    "scope": scope,
                    "metric": "belief_switch_fraction",
                    "estimate": float(group["belief_switch"].mean()),
                    "rows": int(len(group)),
                    "independent_clusters": int(group["cluster_id"].nunique()),
                },
                {
                    "scope": scope,
                    "metric": "action_switch_fraction",
                    "estimate": float(group["action_switch"].mean()),
                    "rows": int(len(group)),
                    "independent_clusters": int(group["cluster_id"].nunique()),
                },
                {
                    "scope": scope,
                    "metric": "commitment_change_fraction",
                    "estimate": float(group["commitment_change"].mean()),
                    "rows": int(len(group)),
                    "independent_clusters": int(group["cluster_id"].nunique()),
                },
            ]
        )

    add("all_single_evidence", selected)
    add("left_supporting_evidence", selected[selected["expected_direction"] == -1])
    add("right_supporting_evidence", selected[selected["expected_direction"] == 1])
    for domain, group in selected.groupby("domain", sort=True):
        add("domain:" + str(domain), group)
    return rows


def build_repository_aggregates(repository: Path) -> Dict[str, object]:
    repository = Path(repository)
    external = artifact_root()
    results = repository / "results/llm_agent_entropy_v11"
    tables = results / "tables"
    source = results / "figures/source_data"
    reproducibility = results / "reproducibility"
    tables.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    reproducibility.mkdir(parents=True, exist_ok=True)

    stage_status: Dict[str, object] = {}
    for stage in ("pilot", "qualification", "formal"):
        analysis = external / stage / "analysis.json"
        run_summary = external / stage / "run_summary.json"
        if analysis.exists():
            shutil.copyfile(analysis, tables / (stage + "_analysis.json"))
            stage_status[stage] = json.loads(analysis.read_text(encoding="utf-8"))
        elif run_summary.exists():
            stage_status[stage] = json.loads(run_summary.read_text(encoding="utf-8"))
        else:
            stage_status[stage] = {"status": "not_run"}

    qualification_rows = external / "qualification/decisions.csv"
    if qualification_rows.exists():
        qualification_frame = pd.read_csv(qualification_rows)
        compact = _qualification_cluster_rows(qualification_frame)
        atomic_csv(compact, tables / "qualification_cluster_effects.csv")
        atomic_csv(compact, source / "belief_response.csv")
        exploratory = _qualification_exploratory_rows(pd.DataFrame(compact))
        atomic_csv(exploratory, tables / "qualification_exploratory_diagnostics.csv")
        calibration = (
            qualification_frame[qualification_frame["valid_after_repair"] == 1]
            .groupby(
                ["domain", "paraphrase", "option_order_right_first", "private_observation", "condition"],
                sort=True,
            )
            .agg(
                mean_reported_probability=("probability_right", "mean"),
                empirical_right_frequency=("belief_right", "mean"),
                repeated_samples=("belief_right", "size"),
            )
            .reset_index()
        )
        atomic_csv(calibration.to_dict(orient="records"), source / "reported_vs_empirical_calibration.csv")
    formal_panel = external / "formal/panel_metrics.csv"
    if formal_panel.exists():
        frame = pd.read_csv(formal_panel)
        atomic_csv(frame.to_dict(orient="records"), tables / "formal_panel_metrics.csv")
        atomic_csv(frame.to_dict(orient="records"), source / "formal_panel_metrics.csv")
    formal_rows = external / "formal/trajectory_rows.csv"
    if formal_rows.exists():
        trajectories = pd.read_csv(formal_rows)
        trajectories = trajectories[(trajectories["valid_after_repair"] == 1) & (trajectories["panel_family"] == "primary")]
        current_rows: List[Dict[str, object]] = []
        convergence_rows: List[Dict[str, object]] = []
        from .statistics import block_time_reversal_kl

        for panel_id, group in trajectories.groupby("panel_id", sort=True):
            ordered = group.sort_values("turn")
            states = ordered["coarse_macrostate"].to_numpy(int)
            for source_state, destination_state in zip(states[:-1], states[1:]):
                current_rows.append(
                    {
                        "panel_id": panel_id,
                        "application": ordered["application"].iloc[0],
                        "alpha": float(ordered["alpha"].iloc[0]),
                        "source_state": int(source_state),
                        "destination_state": int(destination_state),
                        "transition_count": 1,
                    }
                )
            for prefix in (8, 16, 24, 32):
                if len(states) >= prefix:
                    convergence_rows.append(
                        {
                            "panel_id": panel_id,
                            "application": ordered["application"].iloc[0],
                            "alpha": float(ordered["alpha"].iloc[0]),
                            "prefix_turns": prefix,
                            "block_time_reversal_kl": block_time_reversal_kl(states[:prefix], 3, 0.5),
                        }
                    )
        if current_rows:
            current = (
                pd.DataFrame(current_rows)
                .groupby(["application", "alpha", "source_state", "destination_state"], sort=True)["transition_count"]
                .sum()
                .reset_index()
            )
            atomic_csv(current.to_dict(orient="records"), source / "coarse_transition_currents.csv")
        if convergence_rows:
            atomic_csv(convergence_rows, source / "estimator_convergence.csv")
    paired = external / "formal/paired_primary_effects.csv"
    if paired.exists():
        shutil.copyfile(paired, tables / "formal_paired_primary_effects.csv")

    manifests = []
    for child in sorted(path for path in external.iterdir() if path.is_dir() and path.name != "locks"):
        manifests.append(_directory_digest(child))
    atomic_csv(manifests, reproducibility / "external_artifact_directory_checksums.csv")
    summary = {
        "generated_at": utc_now(),
        "starting_v10_commit": "4d372f00837bf75f90882392a92feac87dbc84b2",
        "branch": "evidence-grounded-llm-entropy-v11",
        "external_artifact_root": str(external),
        "external_file_count": int(sum(int(row["file_count"]) for row in manifests)),
        "external_bytes": int(sum(int(row["bytes"]) for row in manifests)),
        "stages": stage_status,
    }
    atomic_json(summary, reproducibility / "summary.json")
    return summary
