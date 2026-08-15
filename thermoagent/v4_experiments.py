"""Restartable v4 experiment orchestration and provenance manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .events import sha256_file
from .v4_runner import V4EpisodeConfig, V4EpisodeResult, V4EpisodeRunner


MODEL_IDENTIFIER = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PROMPT_REVISION = "thermohitl-v4-utility-json-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_checksum(repository: Path) -> str:
    digest = hashlib.sha256()
    for root in (repository / "thermoagent", repository / "configs", repository / "scripts"):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh", ".toml"}:
                digest.update(str(path.relative_to(repository)).encode("utf-8"))
                digest.update(path.read_bytes())
    return digest.hexdigest()


def git_metadata(repository: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=str(repository), capture_output=True,
            text=True, check=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def protocol_checksum(repository: Path) -> str:
    return sha256_file(repository / "configs" / "human_operator_v4_development.yaml")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_episode(
    repository: Path,
    results_root: Path,
    stage: str,
    result: V4EpisodeResult,
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    run_root = results_root / "raw" / stage / result.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    episode_path = run_root / "episode.json"
    ledger_path = run_root / "events.jsonl.gz"
    manifest_path = results_root / "manifests" / stage / (result.run_id + ".json")
    if episode_path.exists() or ledger_path.exists() or manifest_path.exists():
        raise FileExistsError("v4 run output already exists: %s" % result.run_id)
    ledger_sha = result.ledger.write_jsonl(ledger_path)
    episode = result.episode_dict()
    _atomic_json(episode_path, episode)
    git = git_metadata(repository)
    manifest: Dict[str, Any] = {
        **result.manifest_fields,
        "source_checksum": source_checksum(repository),
        "protocol_checksum": protocol_checksum(repository),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "dirty_tree": git["dirty"],
        "model_identifier": result.manifest_fields.get("model_identifier"),
        "model_revision": result.manifest_fields.get("model_revision"),
        "prompt_revision": PROMPT_REVISION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "generated_at": utc_now(),
        "episode_sha256": sha256_file(episode_path),
        "ledger_sha256": ledger_sha,
        "output_checksums": {
            str(episode_path.relative_to(repository)): sha256_file(episode_path),
            str(ledger_path.relative_to(repository)): ledger_sha,
        },
    }
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    _atomic_json(manifest_path, manifest)
    return manifest


def _summary_row(result: V4EpisodeResult) -> Dict[str, Any]:
    return {
        "run_id": result.run_id,
        "application": result.application,
        "regime": result.regime,
        "information_condition": result.information_condition,
        "method": result.method,
        "environment_seed": result.environment_seed,
        "operator_seed": result.operator_seed,
        "rl_seed": result.rl_seed,
        "status": result.status,
        **result.metrics,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_matrix(
    repository: Path,
    results_root: Path,
    stage: str,
    applications: Sequence[str],
    regimes: Sequence[str],
    information_conditions: Sequence[str],
    methods: Sequence[str],
    environment_seeds: Sequence[int],
    counterfactual_probes: bool = False,
    dense_candidates: bool = False,
    operator_profile: str = "high_accuracy_bounded",
    operator_budget: int = 2,
    resume: bool = True,
) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    counterfactual_rows: List[Dict[str, Any]] = []
    completed = 0
    skipped = 0
    failures: List[Dict[str, Any]] = []
    for application in applications:
        for regime in regimes:
            for information_condition in information_conditions:
                for method in methods:
                    for seed in environment_seeds:
                        config = V4EpisodeConfig(
                            application=application,
                            regime=regime,
                            information_condition=information_condition,
                            method=method,
                            environment_seed=int(seed),
                            operator_seed=int(seed) + 100_000,
                            horizon=20,
                            disruption_step=6,
                            operator_profile=operator_profile,
                            operator_budget=operator_budget,
                            counterfactual_probes=counterfactual_probes,
                            dense_candidates=dense_candidates,
                            stage=stage,
                        )
                        run_root = results_root / "raw" / stage / config.run_id
                        episode_path = run_root / "episode.json"
                        if resume and episode_path.exists():
                            episode = json.loads(episode_path.read_text(encoding="utf-8"))
                            summaries.append({
                                "run_id": episode["run_id"],
                                "application": episode["application"],
                                "regime": episode["regime"],
                                "information_condition": episode["information_condition"],
                                "method": episode["method"],
                                "environment_seed": episode["environment_seed"],
                                "operator_seed": episode["operator_seed"],
                                "rl_seed": episode.get("rl_seed"),
                                "status": episode["status"],
                                **episode["metrics"],
                            })
                            candidate_rows.extend(episode.get("candidate_interventions", []))
                            counterfactual_rows.extend(episode.get("counterfactuals", []))
                            skipped += 1
                            continue
                        try:
                            result = V4EpisodeRunner(config).run()
                            write_episode(repository, results_root, stage, result)
                            summaries.append(_summary_row(result))
                            candidate_rows.extend(result.candidate_interventions)
                            counterfactual_rows.extend({"run_id": result.run_id, **row} for row in result.counterfactuals)
                            completed += 1
                        except Exception as error:
                            failure = {
                                "run_id": config.run_id,
                                "application": application,
                                "regime": regime,
                                "information_condition": information_condition,
                                "method": method,
                                "environment_seed": seed,
                                "status": "failed",
                                "failure_type": type(error).__name__,
                                "failure_reason": str(error),
                            }
                            failures.append(failure)
                            _atomic_json(
                                results_root / "manifests" / stage / (config.run_id + ".failed.json"),
                                failure,
                            )
    write_csv(results_root / "development" / stage / "episode_summary.csv", summaries)
    if candidate_rows:
        write_csv(results_root / "development" / stage / "candidate_interventions.csv", candidate_rows)
    if counterfactual_rows:
        write_csv(results_root / "counterfactuals" / (stage + ".csv"), counterfactual_rows)
    if failures:
        write_csv(results_root / "negative_results" / (stage + "_failed_runs.csv"), failures)
    report = {
        "stage": stage,
        "completed_now": completed,
        "resumed_existing": skipped,
        "episodes": len(summaries),
        "failures": len(failures),
        "candidate_interventions": len(candidate_rows),
        "counterfactuals": len(counterfactual_rows),
        "finished_at": utc_now(),
    }
    _atomic_json(results_root / "development" / stage / "run_report.json", report)
    return report


def initialize_v4_results(results_root: Path) -> None:
    directories = (
        "protocol", "development", "validation", "training", "holdout", "raw",
        "statistics", "tables", "figures/pdf", "figures/previews",
        "dashboard_exports", "ablations", "counterfactuals", "monitoring", "logs",
        "manifests", "reproducibility/pdf_qa", "negative_results", "superseded",
    )
    for relative in directories:
        (results_root / relative).mkdir(parents=True, exist_ok=True)
