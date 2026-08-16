"""Prospective V7 protocol freeze and untouched stage manifests."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from .events import sha256_file
from .v5_experiments import atomic_json, source_checksum, utc_now, write_csv


APPLICATION_TOPOLOGIES = {
    "humanitarian": ("random_geometric", "small_world", "modular"),
    "utility_restoration": ("grid", "scale_free", "modular"),
}


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=str(repository), check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _panel(
    application: str, complexity: str, coupling: str, fragmentation: str,
    network_disruption: str, information_condition: str, seed: int,
    topology: str, purpose: str,
) -> Dict[str, Any]:
    return {
        "panel_id": "v7-%s-%s-%d" % (purpose, application, seed),
        "application": application,
        "complexity": complexity,
        "coupling": coupling,
        "fragmentation": fragmentation,
        "network_disruption": network_disruption,
        "information_condition": information_condition,
        "topology_family": topology,
        "environment_seed": int(seed),
        "purpose": purpose,
        "reference_controller": "always_act",
        "reference_sketch_policy": "event_triggered",
        "operational_communication_policy": "agent_event_triggered",
        "counterfactual_limit_per_epoch": 6,
    }


def development_manifest() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seed = 787000
    for application in ("humanitarian", "utility_restoration"):
        topologies = APPLICATION_TOPOLOGIES[application]
        # Medium response surface: three graph instances per factor cell.
        for coupling_index, coupling in enumerate(("low", "medium", "high")):
            for fragmentation_index, fragmentation in enumerate(("low", "medium", "high")):
                disruption = (
                    "high" if coupling == "high" or fragmentation == "high"
                    else ("medium" if coupling == "medium" or fragmentation == "medium" else "low")
                )
                for replicate in range(3):
                    seed += 1
                    rows.append(_panel(
                        application, "medium", coupling, fragmentation,
                        disruption, "private_fragmented", seed,
                        topologies[(coupling_index + fragmentation_index + replicate) % len(topologies)],
                        "response_surface",
                    ))
        # The response surface already has three high/high panels; add nine to
        # reach the prospectively powered total of twelve.
        for replicate in range(9):
            seed += 1
            rows.append(_panel(
                application, "medium", "high", "high", "high",
                "private_fragmented", seed,
                topologies[replicate % len(topologies)], "high_complexity",
            ))
        for complexity in ("small", "large"):
            for replicate in range(4):
                seed += 1
                rows.append(_panel(
                    application, complexity, "high", "high", "high",
                    "private_fragmented", seed,
                    topologies[replicate % len(topologies)], "scale",
                ))
        for replicate in range(6):
            seed += 1
            rows.append(_panel(
                application, "medium", "high", "high", "high",
                "public_shared", seed,
                topologies[replicate % len(topologies)], "public_control",
            ))
    return rows


def sealed_manifest(stage: str, seed_start: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    count = 12 if stage == "validation" else 16
    for application in ("humanitarian", "utility_restoration"):
        held_out = "chain" if application == "humanitarian" else "small_world"
        for replicate in range(count):
            seed = seed_start + (0 if application == "humanitarian" else 1000) + replicate
            rows.append(_panel(
                application, "medium" if replicate < count - 4 else "large",
                "high", "high", "high", "private_fragmented", seed,
                held_out, "%s_high_complexity_ood" % stage,
            ))
        for replicate in range(4):
            seed = seed_start + (200 if application == "humanitarian" else 1200) + replicate
            rows.append(_panel(
                application, "medium", "high", "high", "high",
                "public_shared", seed, held_out,
                "%s_public_control" % stage,
            ))
    return rows


def freeze_protocol(repository: Path, results_root: Path) -> Dict[str, Any]:
    if (results_root / "protocol" / "freeze_manifest.json").exists():
        raise RuntimeError("V7 protocol is already frozen; refusing to overwrite it")
    gate_path = results_root / "development" / "gate_feasibility" / "gate_summary.json"
    if not gate_path.exists():
        raise RuntimeError("V7 feasibility gates have not been evaluated")
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gates.get("formal_development_unlocked")):
        raise RuntimeError("V7 feasibility no-go: formal development cannot be frozen")
    for stage in (
        "development_formal_reference", "development_formal_dynamic",
        "development_formal_communication", "training", "qwen",
        "validation", "holdout",
    ):
        if (results_root / "raw" / stage).exists():
            raise RuntimeError("cannot freeze after %s execution" % stage)
    dirty = _git(repository, "status", "--porcelain")
    if dirty:
        raise RuntimeError("V7 protocol freeze requires a clean committed source tree")
    source_commit = _git(repository, "rev-parse", "HEAD")
    source_branch = _git(repository, "branch", "--show-current")
    if source_branch != "complexity-entropic-coordination-v7":
        raise RuntimeError("V7 freeze attempted from wrong branch")
    candidate_path = repository / "configs" / "v7_protocol_candidate.yaml"
    config = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    config["study"]["status"] = "frozen_before_formal_development"
    protocol_root = results_root / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    frozen_path = protocol_root / "frozen_protocol.yaml"
    temporary = frozen_path.with_suffix(".yaml.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    temporary.replace(frozen_path)
    development = development_manifest()
    validation = sealed_manifest("validation", 797000)
    holdout = sealed_manifest("holdout", 807000)
    manifests = results_root / "manifests"
    write_csv(manifests / "development_inputs.csv", development)
    write_csv(manifests / "validation_inputs_sealed.csv", validation)
    write_csv(manifests / "holdout_inputs_sealed.csv", holdout)
    report = {
        "protocol_version": config["study"]["version"],
        "protocol_path": str(frozen_path.relative_to(results_root)),
        "protocol_sha256": sha256_file(frozen_path),
        "candidate_protocol_sha256": sha256_file(candidate_path),
        "source_commit": source_commit,
        "source_branch": source_branch,
        "source_checksum": source_checksum(repository),
        "development_panels": len(development),
        "validation_panels_sealed": len(validation),
        "holdout_panels_sealed": len(holdout),
        "validation_inputs_sha256": sha256_file(manifests / "validation_inputs_sealed.csv"),
        "holdout_inputs_sha256": sha256_file(manifests / "holdout_inputs_sealed.csv"),
        "frozen_at": utc_now(),
        "worktree_clean_before_freeze_write": True,
    }
    atomic_json(protocol_root / "freeze_manifest.json", report)
    write_csv(protocol_root / "exclusion_ledger.csv", [{"status": "none_at_freeze"}])
    write_csv(protocol_root / "failed_run_registry.csv", [{"status": "none_at_freeze"}])
    write_csv(protocol_root / "protocol_deviation_ledger.csv", [{"status": "none_at_freeze"}])
    return report


def assert_stage_unlocked(results_root: Path, stage: str) -> Mapping[str, Any]:
    path = results_root / "manifests" / "stage_disposition.json"
    if not path.exists():
        raise RuntimeError("V7 stage disposition does not exist")
    disposition = json.loads(path.read_text(encoding="utf-8"))
    if not bool(disposition.get("%s_unlocked" % stage, False)):
        raise RuntimeError("V7 %s is prospectively locked" % stage)
    return disposition
