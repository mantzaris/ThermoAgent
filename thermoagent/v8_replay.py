"""Integrity replay for unpacked and losslessly packed V8 event ledgers."""

from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import json
import lzma
import os
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from .events import sha256_file
from .v5_experiments import atomic_json, write_csv


PRIVATE_KINDS = {
    "v7_private_observation", "v7_belief_update",
    "v8_trigger_evaluated", "v8_distributed_estimate",
}
EVALUATOR_KINDS = {
    "v8_estimation_error", "v8_conservation_audit", "v8_privacy_audit",
}


def _ledger_audit(
    ledger_payload: bytes, episode: Mapping[str, Any], *,
    stored_sha256: str, logical_sha256: str, locator: str,
) -> Dict[str, Any]:
    logical_match = hashlib.sha256(ledger_payload).hexdigest() == logical_sha256
    contiguous = True
    event_count = 0
    digest = hashlib.sha256()
    metrics: List[Mapping[str, Any]] = []
    privacy_failures = 0
    for line in ledger_payload.decode("utf-8").splitlines():
        event_count += 1
        event = json.loads(line)
        contiguous = contiguous and (
            event.get("event_id") == "E%08d" % event_count
        )
        if event_count > 1:
            digest.update(b"\n")
        digest.update(json.dumps(event, sort_keys=True).encode("utf-8"))
        if event["kind"] == "metric":
            metrics.append(event["payload"])
        if event["kind"] in PRIVATE_KINDS:
            privacy_failures += int(event.get("private_to") is None)
        if event["kind"] in EVALUATOR_KINDS:
            privacy_failures += int(event.get("private_to") != "evaluator")
    digest_match = digest.hexdigest() == episode["event_ledger_digest"]
    stored_match = str(episode["event_ledger_sha256"]) == str(stored_sha256)
    summary = dict(episode["summary"])
    metric_match = len(metrics) == 1
    if metrics:
        for key in (
            "service_loss", "net_causal_utility", "fully_counted_messages",
            "fully_counted_bytes", "sketch_on_wire_bytes",
            "primary_distributed_state_error", "maximum_conservation_residual",
        ):
            # Early retained pilot schema versions predate the primary-error
            # alias.  Absence from both stored metric and episode summary is a
            # versioned omission, not a mismatch; one-sided absence still
            # fails regeneration.
            if key not in metrics[-1] and key not in summary:
                continue
            metric_match = (
                metric_match and key in metrics[-1] and key in summary
                and bool(np.isclose(
                    float(metrics[-1][key]), float(summary[key]), atol=1e-12,
                ))
            )
    finite = all(
        np.isfinite(float(value))
        for value in summary.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    residual = abs(float(summary.get("maximum_conservation_residual", np.inf)))
    reasons = []
    if not logical_match:
        reasons.append("logical_sha256")
    if not stored_match:
        reasons.append("stored_sha256")
    if not contiguous:
        reasons.append("event_sequence")
    if not digest_match:
        reasons.append("event_digest")
    if not metric_match:
        reasons.append("metric_regeneration")
    if privacy_failures:
        reasons.append("privacy")
    if not finite:
        reasons.append("nonfinite")
    if residual > 1e-9:
        reasons.append("conservation")
    return {
        "run_id": summary["run_id"], "stage": summary["stage"],
        "application": summary["application"], "ledger_locator": locator,
        "events": event_count, "logical_sha256_match": logical_match,
        "stored_sha256_match": stored_match, "event_sequence_contiguous": contiguous,
        "event_digest_match": digest_match, "metric_regeneration_match": metric_match,
        "privacy_failures": privacy_failures, "finite_metrics": finite,
        "maximum_conservation_residual": residual,
        "status": "pass" if not reasons else "mismatch",
        "mismatch_reasons": ";".join(reasons),
    }


def _packed_rows(archive_path: Path) -> List[Dict[str, Any]]:
    manifest_path = archive_path.with_suffix(".manifest.csv")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    by_member = {value["archive_member"]: value for value in manifest}
    prefix_members: Dict[str, set[str]] = {}
    for member_name in by_member:
        prefix, name = member_name.rsplit("/", 1)
        prefix_members.setdefault(prefix, set()).add(name)
    complete_prefixes = {
        prefix for prefix, names in prefix_members.items()
        if "episode.json" in names and "events.jsonl" in names
    }
    # A retained invalidated run may have a partial event stream but no atomic
    # episode payload.  Such a run was never eligible for replay before packing
    # (unpacked discovery is episode.json-based), so preserve it without
    # promoting it to a completed ledger.  Conversely, an episode payload with
    # no ledger is an integrity failure rather than a permissible partial run.
    missing_ledgers = [
        prefix for prefix, names in prefix_members.items()
        if "episode.json" in names and "events.jsonl" not in names
    ]
    if missing_ledgers:
        raise RuntimeError(
            "packed episode lacks an event ledger: %s" % sorted(missing_ledgers)[0]
        )
    episodes: Dict[str, Mapping[str, Any]] = {}
    pending_ledgers: Dict[str, tuple[bytes, Mapping[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    # Stream compressed archives once.  Random access into an xz-compressed tar
    # can repeatedly decompress the prefix for thousands of members, turning a
    # complete replay into an hours-long integrity check without changing its
    # semantics.
    with tarfile.open(archive_path, mode="r|xz") as archive:
        for member in archive:
            name = member.name.rsplit("/", 1)[-1]
            if name not in ("episode.json", "events.jsonl"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError("packed replay member is not a file")
            prefix = member.name.rsplit("/", 1)[0]
            payload = stream.read()
            if name == "episode.json":
                episodes[prefix] = json.loads(payload.decode("utf-8"))
                if prefix in pending_ledgers:
                    ledger_payload, ledger_manifest = pending_ledgers.pop(prefix)
                    rows.append(_ledger_audit(
                        ledger_payload, episodes.pop(prefix),
                        stored_sha256=ledger_manifest["stored_sha256"],
                        logical_sha256=ledger_manifest["logical_sha256"],
                        locator=str(archive_path) + "::" + prefix + "/events.jsonl",
                    ))
                continue
            if prefix not in complete_prefixes:
                continue
            ledger_manifest = by_member[member.name]
            if prefix not in episodes:
                pending_ledgers[prefix] = (payload, ledger_manifest)
                continue
            rows.append(_ledger_audit(
                payload, episodes.pop(prefix),
                stored_sha256=ledger_manifest["stored_sha256"],
                logical_sha256=ledger_manifest["logical_sha256"],
                locator=str(archive_path) + "::" + member.name,
            ))
    if episodes or pending_ledgers:
        raise RuntimeError(
            "packed complete run could not be paired: %s"
            % sorted(set(episodes) | set(pending_ledgers))[0]
        )
    if len(rows) != len(complete_prefixes):
        raise RuntimeError("packed complete-run count does not match replay rows")
    return rows


def _unpacked_episode(episode_path: Path) -> Dict[str, Any]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    ledger_path = episode_path.parent / "events.jsonl.xz"
    with lzma.open(ledger_path, "rb") as stream:
        payload = stream.read()
    return _ledger_audit(
        payload, episode,
        stored_sha256=sha256_file(ledger_path),
        logical_sha256=hashlib.sha256(payload).hexdigest(),
        locator=str(ledger_path),
    )


def _replay_job(job: Tuple[str, str]) -> List[Dict[str, Any]]:
    kind, raw_path = job
    path = Path(raw_path)
    if kind == "packed":
        return _packed_rows(path)
    if kind == "unpacked":
        return [_unpacked_episode(path)]
    raise ValueError("unknown V8 replay job: %s" % kind)


def replay_v8_results(results_root: Path) -> Dict[str, Any]:
    jobs: List[Tuple[str, str]] = []
    packed = results_root / "raw" / "packed"
    if packed.exists():
        for archive in sorted(packed.glob("*/*.tar.xz")):
            jobs.append(("packed", str(archive)))
    raw = results_root / "raw"
    if raw.exists():
        for stage_dir in sorted(value for value in raw.iterdir() if value.is_dir()):
            if stage_dir.name == "packed":
                continue
            for episode_path in sorted(stage_dir.glob("*/episode.json")):
                jobs.append(("unpacked", str(episode_path)))
    rows: List[Dict[str, Any]] = []
    if len(jobs) < 8:
        for job in jobs:
            rows.extend(_replay_job(job))
    else:
        workers = min(4, max(1, int(os.cpu_count() or 1)))
        # Process isolation bounds per-ledger memory and parallelizes xz/JSON
        # verification without changing ordered result assembly.
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for values in executor.map(_replay_job, jobs, chunksize=1):
                rows.extend(values)
    destination = results_root / "reproducibility" / "replay"
    write_csv(destination / "ledger_replay_audit.csv", rows)
    partial_runs = 0
    compaction = results_root / "reproducibility" / "compaction"
    if compaction.exists():
        for path in compaction.glob("*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            partial_runs += int(report.get("partial_run_directories", 0))
    report = {
        "episodes_replayed": len(rows),
        "partial_run_directories_retained_not_replayed": partial_runs,
        "replay_mismatches": sum(value["status"] != "pass" for value in rows),
        "events_replayed": int(sum(value["events"] for value in rows)),
        "privacy_failures": int(sum(value["privacy_failures"] for value in rows)),
        "maximum_conservation_residual": max([
            value["maximum_conservation_residual"] for value in rows
        ] or [0.0]),
        "status": "pass" if rows and all(value["status"] == "pass" for value in rows) else "fail",
    }
    atomic_json(destination / "replay_summary.json", report)
    return report
