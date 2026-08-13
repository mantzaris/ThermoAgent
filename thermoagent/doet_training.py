"""Restartable, non-selective multi-seed coordination-policy training."""

from __future__ import annotations

import csv
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .events import sha256_file
from .experiments import dependency_versions, hardware_summary, train_policy
from .policy import checkpoint_metadata


VARIANTS = ("no_entropy", "thermo", "doet_rl")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _existing_valid(
    checkpoint: Path,
    variant: str,
    seed: int,
    episodes: int,
) -> Dict[str, Any]:
    metadata = checkpoint_metadata(checkpoint)
    expected = {
        "variant": variant,
        "training_seed": int(seed),
        "episodes": int(episodes),
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "refusing to overwrite mismatched checkpoint %s: %s"
            % (checkpoint, json.dumps(mismatches, sort_keys=True))
        )
    return metadata


def _learning_curves(log_root: Path, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in seeds:
            path = log_root / ("%s_seed%d.jsonl" % (variant, seed))
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rows.append({
                    "variant": variant,
                    "rl_training_seed": int(seed),
                    **json.loads(line),
                })
    return rows


def run(
    results_root: Path,
    seeds: Iterable[int],
    episodes: int,
    calibration_path: Path,
    trigger_config_path: Path,
) -> Dict[str, Any]:
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 5:
        raise ValueError("the DOET protocol requires at least five RL training seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("RL training seeds must be unique")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    training_root = results_root / "training"
    checkpoint_root = results_root / "checkpoints"
    log_root = results_root / "logs" / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = training_root / "seed_manifest.csv"
    prior_rows: Dict[tuple, Dict[str, Any]] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                prior_rows[(row["variant"], int(row["rl_training_seed"]))] = row
    rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in seeds:
            checkpoint = checkpoint_root / (
                "coordination_%s_seed%d.pt" % (variant, seed)
            )
            log_path = log_root / ("%s_seed%d.jsonl" % (variant, seed))
            started = _utc_now()
            row: Dict[str, Any] = {
                "variant": variant,
                "rl_training_seed": seed,
                "episodes": episodes,
                "checkpoint": str(checkpoint.relative_to(results_root)),
                "log": str(log_path.relative_to(results_root)),
                "started_at": started,
                "status": "running",
                "failure_reason": "",
                "checkpoint_selection_rule": (
                    "final checkpoint after the identical fixed episode budget; "
                    "no outcome-based seed or checkpoint selection"
                ),
            }
            try:
                if checkpoint.exists():
                    metadata = _existing_valid(
                        checkpoint, variant, seed, episodes
                    )
                    row["resumed_existing_checkpoint"] = True
                else:
                    metadata = train_policy(
                        output_path=checkpoint,
                        variant=variant,
                        episodes=episodes,
                        seed=seed,
                        calibration_path=calibration_path,
                        log_path=log_path,
                        trigger_config_path=(
                            trigger_config_path if variant == "doet_rl" else None
                        ),
                    )
                    row["resumed_existing_checkpoint"] = False
                row.update({
                    "status": "complete",
                    "ended_at": _utc_now(),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "wall_clock_seconds": metadata.get("wall_clock_seconds"),
                    "final_window_primary_mean": metadata.get(
                        "final_window_primary_mean"
                    ),
                    "planner": metadata.get("planner"),
                    "training_method": metadata.get("training_method"),
                })
            except Exception as error:
                row.update({
                    "status": "failed",
                    "ended_at": _utc_now(),
                    "failure_reason": "%s: %s" % (
                        type(error).__name__, str(error)
                    ),
                    "traceback_sha256": __import__("hashlib").sha256(
                        traceback.format_exc().encode("utf-8")
                    ).hexdigest(),
                })
            rows.append(row)
            _write_csv(manifest_path, rows)

    curves = _learning_curves(log_root, seeds)
    curves_path = training_root / "learning_curves.csv"
    _write_csv(curves_path, curves)
    selection_rows = [{
        "variant": row["variant"],
        "rl_training_seed": row["rl_training_seed"],
        "checkpoint": row["checkpoint"],
        "checkpoint_sha256": row.get("checkpoint_sha256", ""),
        "status": row["status"],
        "selection_rule": row["checkpoint_selection_rule"],
    } for row in rows]
    selection_path = training_root / "checkpoint_selection.csv"
    _write_csv(selection_path, selection_rows)
    complete = [row for row in rows if row["status"] == "complete"]
    failed = [row for row in rows if row["status"] == "failed"]
    record = {
        "status": "complete" if not failed else "completed_with_failures",
        "generated_at": _utc_now(),
        "variants": list(VARIANTS),
        "rl_training_seeds": list(seeds),
        "seeds_per_variant": len(seeds),
        "episodes_per_seed": episodes,
        "planned_trainings": len(rows),
        "completed_trainings": len(complete),
        "failed_trainings": len(failed),
        "failed_seed_records": [
            {
                "variant": row["variant"],
                "seed": row["rl_training_seed"],
                "reason": row["failure_reason"],
            }
            for row in failed
        ],
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "trigger_config_path": str(trigger_config_path),
        "trigger_config_sha256": sha256_file(trigger_config_path),
        "dependencies": dependency_versions(),
        "hardware": hardware_summary(),
        "outputs": {
            "seed_manifest.csv": sha256_file(manifest_path),
            "checkpoint_selection.csv": sha256_file(selection_path),
            "learning_curves.csv": sha256_file(curves_path),
        },
    }
    record_path = training_root / "training_manifest.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record
