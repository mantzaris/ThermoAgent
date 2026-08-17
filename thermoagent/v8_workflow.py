"""Guarded, restartable V8 pilot and formal-stage orchestration."""

from __future__ import annotations

import fcntl
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml

from .v5_experiments import atomic_json, write_csv
from .v8_experiments import aggregate_v8_stage, run_v8_episode
from .v8_trigger import TriggerConfig


def load_configuration(repository: Path, filename: str) -> Dict[str, Any]:
    value = dict(yaml.safe_load(
        (repository / "configs" / filename).read_text(encoding="utf-8")
    ))
    base_name = value.pop("base_configuration", None)
    if base_name:
        base = load_configuration(repository, str(base_name))
        base.update(value)
        value = base
    return value


def trigger_from_row(row: Mapping[str, Any]) -> TriggerConfig:
    fields = set(TriggerConfig.__dataclass_fields__)
    return TriggerConfig(**{key: value for key, value in row.items() if key in fields})


def _worker(task: Mapping[str, Any]) -> Dict[str, Any]:
    panel = dict(task["panel"])
    candidate = dict(task["candidate"])
    action_policy = None
    checkpoint = task.get("policy_checkpoint")
    if checkpoint:
        from .v8_training import V8RoleIPPOPolicy
        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(str(task["repository"])) / checkpoint_path
        action_policy = V8RoleIPPOPolicy.load(checkpoint_path, stochastic=False)
    output = run_v8_episode(
        application=str(panel["application"]),
        complexity=str(panel["complexity"]),
        coupling=str(panel["coupling"]),
        fragmentation=str(panel["fragmentation"]),
        network_disruption=str(panel["network_disruption"]),
        topology_family=str(panel["topology_family"]),
        environment_seed=int(panel["environment_seed"]),
        information_condition=str(task["information_condition"]),
        trigger_config=trigger_from_row(candidate),
        encoding=str(candidate.get("encoding", "fp16")),
        maximum_hops=int(task["maximum_hops"]),
        operational_communication_policy=str(task["operational_communication_policy"]),
        action_policy=action_policy,
        results_root=Path(str(task["results_root"])),
        stage=str(task["stage"]),
        ledger_scope=str(task.get("ledger_scope", "full")),
        resume=True,
    )
    return {"candidate_name": candidate["name"], "summary": dict(output["summary"])}


def run_configured_stage(
    repository: Path,
    results_root: Path,
    *,
    configuration_filename: str,
    stage: str,
    include_encoding_checks: bool = False,
) -> Dict[str, Any]:
    configuration = load_configuration(repository, configuration_filename)
    lock_path = results_root / "logs" / (stage + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another V8 writer holds the %s stage lock" % stage) from error
    lock.seek(0)
    lock.truncate()
    lock.write("pid=%d\n" % os.getpid())
    lock.flush()
    panels = list(configuration["panels"])
    candidates = list(configuration["candidates"])
    tasks: List[Dict[str, Any]] = []
    policy_checkpoints = list(configuration.get("policy_checkpoints", [])) or [None]
    panel_seed_offset = int(configuration.get("panel_seed_offset", 0))
    for source_panel in panels:
        panel = dict(source_panel)
        panel["environment_seed"] = int(panel["environment_seed"]) + panel_seed_offset
        for candidate in candidates:
            for checkpoint in policy_checkpoints:
                tasks.append({
                    "panel": panel, "candidate": candidate,
                    "policy_checkpoint": checkpoint,
                    "repository": str(repository),
                    "information_condition": configuration.get(
                        "information_condition", "private_fragmented",
                    ),
                    "maximum_hops": configuration.get("maximum_hops", 1),
                    "operational_communication_policy": configuration.get(
                        "operational_communication_policy", "agent_event_triggered",
                    ),
                    "results_root": str(results_root), "stage": stage,
                    "ledger_scope": configuration.get("ledger_scope", "full"),
                })
    if include_encoding_checks:
        representative = {}
        for panel in panels:
            representative.setdefault(str(panel["application"]), panel)
        for panel in representative.values():
            for candidate in configuration.get("encoding_checks", []):
                tasks.append({
                    "panel": panel, "candidate": candidate,
                    "policy_checkpoint": None,
                    "repository": str(repository),
                    "information_condition": configuration.get(
                        "information_condition", "private_fragmented",
                    ),
                    "maximum_hops": configuration.get("maximum_hops", 1),
                    "operational_communication_policy": configuration.get(
                        "operational_communication_policy", "agent_event_triggered",
                    ),
                    "results_root": str(results_root), "stage": stage,
                    "ledger_scope": configuration.get("ledger_scope", "full"),
                })
    registry = []
    for candidate in candidates + list(configuration.get("encoding_checks", [])):
        trigger = trigger_from_row(candidate)
        import hashlib
        blob = json.dumps(asdict(trigger), sort_keys=True, separators=(",", ":"))
        registry.append({
            "candidate_name": candidate["name"],
            "method": trigger.method,
            "encoding": candidate.get("encoding", "fp16"),
            "configuration_digest": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12],
            "configuration_json": blob,
        })
    write_csv(results_root / stage / "candidate_registry.csv", registry)
    status = {
        "stage": stage, "status": "running", "tasks_total": len(tasks),
        "tasks_completed": 0, "tasks_failed": 0,
        "configuration": configuration_filename,
    }
    atomic_json(results_root / "logs" / (stage + "_supervisor_status.json"), status)
    failures: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []
    workers = max(1, int(configuration.get("workers", 1)))
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_tasks = {executor.submit(_worker, task): task for task in tasks}
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    completed.append(future.result())
                except Exception as error:
                    failures.append({
                        **dict(task["panel"]),
                        "candidate_name": task["candidate"]["name"],
                        "failure_type": type(error).__name__,
                        "failure_reason": str(error),
                    })
                status.update({
                    "tasks_completed": len(completed),
                    "tasks_failed": len(failures),
                })
                atomic_json(
                    results_root / "logs" / (stage + "_supervisor_status.json"),
                    status,
                )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    if failures:
        write_csv(results_root / "negative_results" / (stage + "_failures.csv"), failures)
    execution = aggregate_v8_stage(results_root, stage)
    status.update({
        "status": "complete" if not failures else "complete_with_failures",
        "execution": execution,
    })
    atomic_json(results_root / "logs" / (stage + "_supervisor_status.json"), status)
    return status
