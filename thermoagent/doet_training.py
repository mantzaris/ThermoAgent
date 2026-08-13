"""Restartable, non-selective multi-seed coordination-policy training."""

from __future__ import annotations

import csv
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .events import sha256_file
from .experiments import (
    dependency_versions,
    hardware_summary,
    train_policy,
)
from .policy import checkpoint_metadata


VARIANTS = ("no_entropy", "thermo", "doet_rl")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
    attempts_path = training_root / "training_attempts.csv"
    provenance_path = results_root / "reproducibility" / "execution_source.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "multi-seed training requires captured execution provenance: %s"
            % provenance_path
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    prior_rows: Dict[tuple, Dict[str, Any]] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            for prior in csv.DictReader(handle):
                prior_rows[(
                    prior["variant"], int(prior["rl_training_seed"])
                )] = dict(prior)
    # ``seed_manifest.csv`` is a current-state view used by checkpoint
    # selection. Keep a separate append-only attempt ledger so a failed or
    # interrupted attempt cannot disappear when the restartable command is
    # invoked again.
    attempt_rows: List[Dict[str, Any]] = []
    if attempts_path.exists():
        with attempts_path.open(encoding="utf-8", newline="") as handle:
            attempt_rows.extend(dict(row) for row in csv.DictReader(handle))
    elif prior_rows:
        # Migrate manifests produced before the attempt ledger existed without
        # erasing their original execution-source fields. The imported record
        # is terminal-only because no synthetic start event should be invented.
        for key in sorted(prior_rows):
            prior = dict(prior_rows[key])
            prior.update({
                "attempt_number": 1,
                "attempt_id": "%s-seed%d-attempt1" % key,
                "attempt_event": "terminal",
                "attempt_imported_from_seed_manifest": True,
            })
            attempt_rows.append(prior)
        _write_csv(attempts_path, attempt_rows)
    attempt_counts: Dict[tuple, int] = {}
    for prior in attempt_rows:
        key = (prior.get("variant"), int(prior.get("rl_training_seed", -1)))
        attempt_counts[key] = max(
            attempt_counts.get(key, 0), int(prior.get("attempt_number", 0))
        )
    rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in seeds:
            attempt_key = (variant, seed)
            attempt_number = attempt_counts.get(attempt_key, 0) + 1
            attempt_counts[attempt_key] = attempt_number
            checkpoint = checkpoint_root / (
                "coordination_%s_seed%d.pt" % (variant, seed)
            )
            log_path = log_root / ("%s_seed%d.jsonl" % (variant, seed))
            started = _utc_now()
            prior = prior_rows.get(attempt_key, {})
            existing_checkpoint = checkpoint.exists()
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
                "source_commit": provenance.get("commit", "unknown"),
                "source_branch": provenance.get("branch", "unknown"),
                "source_dirty": provenance.get("dirty"),
                "source_checksum": provenance.get("source_checksum"),
                "checkpoint_origin_source_commit": (
                    prior.get("checkpoint_origin_source_commit")
                    or prior.get("source_commit")
                    if existing_checkpoint else provenance.get("commit", "unknown")
                ),
                "checkpoint_origin_source_branch": (
                    prior.get("checkpoint_origin_source_branch")
                    or prior.get("source_branch")
                    if existing_checkpoint else provenance.get("branch", "unknown")
                ),
                "checkpoint_origin_source_checksum": (
                    prior.get("checkpoint_origin_source_checksum")
                    or prior.get("source_checksum")
                    if existing_checkpoint else provenance.get("source_checksum")
                ),
                "protocol_checksum": sha256_file(trigger_config_path),
                "environment_seed_rule": (
                    "independent RandomState(training seed); fixed 192-episode "
                    "distribution when defaults are used"
                ),
                "llm_calls": 0,
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "planner": "deterministic mock planner during staged PPO",
                "attempt_number": attempt_number,
                "attempt_id": "%s-seed%d-attempt%d" % (
                    variant, seed, attempt_number
                ),
            }
            attempt_rows.append({**row, "attempt_event": "started"})
            _write_csv(attempts_path, attempt_rows)
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
                    # PPO is CPU-bound in this implementation, but it runs on
                    # the paid single-GPU Pod. Count the reserved Pod interval
                    # against the user's additional single-GPU-hour cap.
                    "single_gpu_hours_reserved": (
                        float(metadata.get("wall_clock_seconds", 0.0) or 0.0)
                        / 3600.0
                    ),
                    "final_window_primary_mean": metadata.get(
                        "final_window_primary_mean"
                    ),
                    "planner": metadata.get("planner"),
                    "training_method": metadata.get("training_method"),
                    "checkpoint_generated_at": metadata.get("generated_at"),
                    "trigger_normalizer_path": (
                        (metadata.get("trigger_normalizer_source") or {}).get(
                            "path", ""
                        )
                    ),
                    "trigger_normalizer_sha256": (
                        (metadata.get("trigger_normalizer_source") or {}).get(
                            "sha256", ""
                        )
                    ),
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
            attempt_rows.append({**row, "attempt_event": "terminal"})
            _write_csv(attempts_path, attempt_rows)
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
        "trigger_normalizer_sha256": row.get(
            "trigger_normalizer_sha256", ""
        ),
        "status": row["status"],
        "selection_rule": row["checkpoint_selection_rule"],
    } for row in rows]
    selection_path = training_root / "checkpoint_selection.csv"
    _write_csv(selection_path, selection_rows)
    complete = [row for row in rows if row["status"] == "complete"]
    failed = [row for row in rows if row["status"] == "failed"]
    training_wall_clock_seconds = float(sum(
        float(row.get("wall_clock_seconds", 0.0) or 0.0)
        for row in rows
    ))
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
        "training_attempt_events": len(attempt_rows),
        "nonterminal_attempts_retained": len({
            row.get("attempt_id") for row in attempt_rows
            if row.get("attempt_event") == "started"
        } - {
            row.get("attempt_id") for row in attempt_rows
            if row.get("attempt_event") == "terminal"
        }),
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "trigger_config_path": str(trigger_config_path),
        "trigger_config_sha256": sha256_file(trigger_config_path),
        "source_provenance": provenance,
        "source_provenance_sha256": sha256_file(provenance_path),
        "model_identifier_during_training": "none; frozen LLM coupled only in validation/holdout evaluation",
        "evaluation_model_identifier": "Qwen/Qwen2.5-7B-Instruct",
        "evaluation_model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "prompt_revision_during_training": "deterministic-mock-v2",
        "environment_steps_per_seed": int(episodes * 16),
        "llm_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "wall_clock_seconds": training_wall_clock_seconds,
        "single_gpu_hours_reserved": training_wall_clock_seconds / 3600.0,
        "compute_accounting_note": (
            "coordination PPO is CPU-bound but the existing single-GPU Pod is "
            "reserved during training, so elapsed training time counts against "
            "the 35-hour additional resource cap"
        ),
        "dependencies": dependency_versions(),
        "hardware": hardware_summary(),
        "outputs": {
            "seed_manifest.csv": sha256_file(manifest_path),
            "training_attempts.csv": sha256_file(attempts_path),
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
