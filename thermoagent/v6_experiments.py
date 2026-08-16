"""Restartable V6 dynamic-panel execution and immutable provenance."""

from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .events import sha256_file
from .v5_experiments import atomic_json, source_checksum, write_csv
from .v6_environment import V6PanelEnvironment
from .v6_policies import NeverActController, SelectiveController


MODEL_IDENTIFIER = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PROMPT_REVISION = "generalized-entropic-consensus-v6-independent-json-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_metadata(repository: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=str(repository), check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit": os.environ.get("THERMO_SOURCE_COMMIT", "filtered_source_bundle"),
            "branch": os.environ.get("THERMO_SOURCE_BRANCH", "filtered_source_bundle"),
            "dirty": False,
            "git_metadata_filtered": True,
        }


def run_id(
    stage: str, application: str, regime: str, information_condition: str,
    seed: int, method: str, coverage: float, sketch_policy: str,
) -> str:
    return "v6-%s-%s-%s-%s-e%d-%s-c%03d-%s" % (
        stage, application, regime, information_condition, int(seed), method,
        int(round(100 * float(coverage))), sketch_policy,
    )


def _controller(
    method: str, coverage: float, escalation_slots: int,
    escalation_risk_threshold: float,
):
    if method == "never_act":
        return NeverActController()
    return SelectiveController(
        method, coverage, escalation_slots, escalation_risk_threshold,
    )


def _atomic_episode(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(encoded)
    temporary.replace(path)


def write_csv_gzip(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic LF-terminated CSV compressed with gzip mtime zero."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    source = list(rows)
    for row in source:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    if not fields:
        fields = ["status"]
        source = [{"status": "no_rows"}]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            import io
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(source)
    temporary.replace(path)


def read_episode_json(path: Path) -> Dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return dict(json.load(handle))
    return dict(json.loads(path.read_text(encoding="utf-8")))


def run_episode(
    repository: Path,
    results_root: Path,
    stage: str,
    application: str,
    regime: str,
    information_condition: str,
    seed: int,
    method: str,
    coverage: float,
    sketch_policy: str,
    escalation_slots: int = 1,
    escalation_risk_threshold: float = 0.80,
    resume: bool = True,
    controller_override: Optional[Any] = None,
    extra_configuration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    identifier = run_id(
        stage, application, regime, information_condition, seed, method,
        coverage, sketch_policy,
    )
    episode_root = results_root / "raw" / stage / identifier
    episode_path = episode_root / "episode.json.gz"
    legacy_episode_path = episode_root / "episode.json"
    existing_path = episode_path if episode_path.exists() else legacy_episode_path
    if resume and existing_path.exists():
        existing = read_episode_json(existing_path)
        if existing.get("status") == "complete":
            return dict(existing["summary"])
    started = utc_now()
    timer = time.perf_counter()
    environment = V6PanelEnvironment(
        application, regime, information_condition, int(seed), sketch_policy,
    )
    controller = controller_override or _controller(
        method, coverage, escalation_slots, escalation_risk_threshold,
    )
    summary = environment.run(controller, method)
    ledger_path = episode_root / "events.jsonl.gz"
    ledger_sha = environment.ledger.write_jsonl(ledger_path)
    summary.update({
        "run_id": identifier,
        "stage": stage,
        "status": "complete" if summary["conservation_feasible"] else "failed",
        "coverage_target": float(coverage),
        "escalation_slots_per_epoch": int(escalation_slots),
        "escalation_risk_threshold": float(escalation_risk_threshold),
        "wall_seconds": float(time.perf_counter() - timer),
        "event_ledger_path": str(ledger_path.relative_to(results_root)),
        "event_ledger_sha256": ledger_sha,
    })
    payload = {
        "study": "Generalized Entropic Consensus V6",
        "evidence_stage": stage,
        "status": summary["status"],
        "started_at": started,
        "ended_at": utc_now(),
        "configuration": {
            "application": application,
            "regime": regime,
            "information_condition": information_condition,
            "environment_seed": int(seed),
            "method": method,
            "coverage": float(coverage),
            "sketch_policy": sketch_policy,
            "escalation_slots_per_epoch": int(escalation_slots),
            "escalation_risk_threshold": float(escalation_risk_threshold),
            **dict(extra_configuration or {}),
        },
        "provenance": {
            **git_metadata(repository),
            "source_checksum": source_checksum(repository),
            "model_identifier": MODEL_IDENTIFIER,
            "model_revision": MODEL_REVISION,
            "prompt_revision": PROMPT_REVISION,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "summary": summary,
        "candidate_records": environment.candidate_records,
        "delegation_records": environment.delegation_records,
        "action_records": environment.action_records,
        "consensus_records": environment.consensus_records,
        "resource_accounting": environment.conservation_report(),
        "evaluator_boundary": {
            "true_state_used_by_controller": False,
            "future_tape_used_by_controller": False,
            "counterfactual_effect_used_by_controller": False,
            "simulated_operator": bool(summary["escalations"]),
            "real_human_participants": False,
        },
    }
    _atomic_episode(episode_path, payload)
    return summary


def _read_episode_rows(results_root: Path, stage: str) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {
        "summaries": [], "candidates": [], "delegations": [], "actions": [],
        "consensus": [],
    }
    paths = list((results_root / "raw" / stage).glob("*/episode.json"))
    paths.extend((results_root / "raw" / stage).glob("*/episode.json.gz"))
    for path in sorted(paths):
        payload = read_episode_json(path)
        if payload.get("status") != "complete":
            continue
        summary = dict(payload["summary"])
        output["summaries"].append(summary)
        prefix = {
            "run_id": summary["run_id"], "controller": summary["controller"],
            "coverage_target": summary["coverage_target"],
            "sketch_policy": summary["sketch_policy"],
        }
        for source, target in (
            ("candidate_records", "candidates"),
            ("delegation_records", "delegations"),
            ("action_records", "actions"),
            ("consensus_records", "consensus"),
        ):
            output[target].extend([{**prefix, **row} for row in payload.get(source, [])])
    return output


def aggregate_stage(
    results_root: Path, stage: str, include_candidate_records: bool = True,
) -> Dict[str, Any]:
    rows = _read_episode_rows(results_root, stage)
    if stage.startswith("pilot"):
        destination = results_root / "pilots" / stage
    elif stage.startswith("development_"):
        destination = results_root / "development" / stage[len("development_"):]
    else:
        destination = results_root / stage
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "episode_summary.csv", rows["summaries"])
    if include_candidate_records:
        write_csv(destination / "candidate_decisions.csv", rows["candidates"])
    else:
        atomic_json(destination / "candidate_decisions_aggregation.json", {
            "status": "omitted_from_flat_aggregate",
            "reason": "authoritative candidate rows remain in compressed per-episode ledgers; omission keeps each Git artifact below 50 MiB",
            "candidate_decisions": len(rows["candidates"]),
        })
    write_csv(destination / "delegation_decisions.csv", rows["delegations"])
    write_csv(destination / "completed_actions.csv", rows["actions"])
    write_csv(destination / "distributed_consensus.csv", rows["consensus"])
    report = {
        "stage": stage,
        "episodes": len(rows["summaries"]),
        "candidate_decisions": len(rows["candidates"]),
        "delegation_decisions": len(rows["delegations"]),
        "completed_actions": len(rows["actions"]),
        "consensus_estimates": len(rows["consensus"]),
        "failed_episodes": sum(row.get("status") != "complete" for row in rows["summaries"]),
    }
    atomic_json(destination / "stage_manifest.json", report)
    return report


def run_matrix(
    repository: Path,
    results_root: Path,
    stage: str,
    applications: Sequence[str],
    regimes: Sequence[str],
    information_conditions: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    coverages: Sequence[float],
    sketch_policies: Sequence[str],
    escalation_slots: int = 1,
    resume: bool = True,
) -> Dict[str, Any]:
    lock_path = results_root / "logs" / (stage + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        total = (
            len(applications) * len(regimes) * len(information_conditions)
            * len(seeds) * len(methods) * len(coverages) * len(sketch_policies)
        )
        completed = 0
        status_path = results_root / "logs" / "supervisor_status.json"
        for application in applications:
            for regime in regimes:
                for condition in information_conditions:
                    for seed in seeds:
                        for method in methods:
                            for coverage in coverages:
                                for sketch_policy in sketch_policies:
                                    run_episode(
                                        repository, results_root, stage, application,
                                        regime, condition, int(seed), method,
                                        float(coverage), sketch_policy,
                                        escalation_slots=escalation_slots,
                                        resume=resume,
                                    )
                                    completed += 1
                                    if completed % 10 == 0 or completed == total:
                                        atomic_json(status_path, {
                                            "stage": stage, "status": "running" if completed < total else "complete",
                                            "completed": completed, "total": total,
                                            "updated_at": utc_now(),
                                        })
        return aggregate_stage(results_root, stage)
