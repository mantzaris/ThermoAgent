"""Prospective V6 seed manifests, protocol freeze, and stage disposition."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from .events import sha256_file
from .v5_experiments import atomic_json, source_checksum, utc_now, write_csv
from .v6_experiments import MODEL_IDENTIFIER, MODEL_REVISION, PROMPT_REVISION, git_metadata


VALIDATION_REGIMES = (
    "nominal", "isolated_physical", "telemetry_integrity",
    "partition", "compound", "ood",
)
HOLDOUT_REGIMES = VALIDATION_REGIMES


def _stage_rows(stage: str, seeds: Sequence[int], regimes: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian", "utility_restoration"):
        for regime in regimes:
            for condition in ("private_fragmented", "public_shared"):
                for seed in seeds:
                    rows.append({
                        "stage": stage,
                        "application": application,
                        "regime": regime,
                        "information_condition": condition,
                        "environment_seed": int(seed),
                        "topology_family": "%s_v6_topology_%d" % (application, int(seed) % 5),
                        "scenario_family": "%s_%s_family_%d" % (application, regime, int(seed) % 5),
                        "operator_seed": int(seed) + 100000,
                        "llm_seed": int(seed) + 200000,
                        "status": "sealed_not_run",
                    })
    return rows


def freeze_protocol(repository: Path, results_root: Path) -> Dict[str, Any]:
    config_path = repository / "configs" / "generalized_entropic_consensus_v6.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["study"]["status"] != "frozen_before_formal_development":
        raise RuntimeError("V6 config must be explicitly marked frozen before freeze")
    for stage in ("validation", "holdout"):
        if (results_root / stage).exists():
            raise RuntimeError("cannot freeze after %s outputs exist" % stage)
    protocol_root = results_root / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    frozen = protocol_root / "frozen_protocol.yaml"
    if frozen.exists() and frozen.read_bytes() != config_path.read_bytes():
        raise RuntimeError("frozen protocol already exists with different bytes")
    shutil.copyfile(config_path, frozen)
    validation_rows = _stage_rows(
        "validation", tuple(range(67101, 67121)), VALIDATION_REGIMES,
    )
    holdout_rows = _stage_rows(
        "holdout", tuple(range(68101, 68125)), HOLDOUT_REGIMES,
    )
    manifests = results_root / "manifests"
    write_csv(manifests / "validation_inputs_sealed.csv", validation_rows)
    write_csv(manifests / "holdout_inputs_sealed.csv", holdout_rows)
    git = git_metadata(repository)
    report = {
        "study": "Generalized Entropic Consensus V6",
        "protocol_version": config["study"]["protocol_version"],
        "frozen_before_formal_development": True,
        "frozen_at": utc_now(),
        "source_commit": git["commit"],
        "source_branch": git["branch"],
        "worktree_clean_before_freeze_write": not git["dirty"],
        "source_checksum": source_checksum(repository),
        "protocol_path": str(frozen.relative_to(results_root)),
        "protocol_sha256": sha256_file(frozen),
        "validation_manifest": str((manifests / "validation_inputs_sealed.csv").relative_to(results_root)),
        "validation_manifest_sha256": sha256_file(manifests / "validation_inputs_sealed.csv"),
        "validation_panels": len(validation_rows),
        "holdout_manifest": str((manifests / "holdout_inputs_sealed.csv").relative_to(results_root)),
        "holdout_manifest_sha256": sha256_file(manifests / "holdout_inputs_sealed.csv"),
        "holdout_panels": len(holdout_rows),
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "python": platform.python_version(),
        "validation_outputs_absent": True,
        "holdout_outputs_absent": True,
    }
    atomic_json(protocol_root / "freeze_manifest.json", report)
    for name, rows in (
        ("protocol_deviations.csv", [{"status": "none_at_freeze"}]),
        ("exclusion_ledger.csv", [{"status": "none_at_freeze"}]),
        ("failed_run_registry.csv", [{"status": "none_at_freeze"}]),
    ):
        write_csv(results_root / "reproducibility" / name, rows)
    return report


def write_stage_disposition(results_root: Path, gate_report: Mapping[str, Any]) -> None:
    unlocked = bool(gate_report["validation_unlocked"])
    if unlocked:
        return
    status = {
        "status": "prospectively_not_run",
        "reason": "one or more required V6 development gates failed",
        "gate_report": "development/gate_status.json",
        "thresholds_changed_after_outcomes": False,
        "validation_unlocked": False,
        "holdout_unlocked": False,
    }
    atomic_json(results_root / "reproducibility" / "formal_stage_disposition.json", status)
    (results_root / "reproducibility" / "formal_stage_disposition.md").write_text(
        "# Formal-stage disposition\n\n"
        "V6 validation and sealed holdout were prospectively not run because "
        "one or more frozen development gates failed. See "
        "`development/gate_status.json`. No threshold was weakened and no "
        "validation or holdout outcome was inspected.\n",
        encoding="utf-8",
    )
