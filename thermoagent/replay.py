"""Quantitative replay from recorded tool calls after LLM generation."""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .environment import LogisticsEnvironment, ScenarioConfig
from .events import Event, EventLedger


def _tool_pairs(events: Sequence[Event]) -> List[Tuple[Event, Event]]:
    pairs: List[Tuple[Event, Event]] = []
    pending: List[Event] = []
    for event in events:
        if event.kind == "tool_call":
            pending.append(event)
        elif event.kind == "tool_result" and pending:
            # Simulator-executed calls are serialized. Validation failures have
            # a result but no preceding tool_call and are intentionally skipped.
            call = pending.pop(0)
            pairs.append((call, event))
    if pending:
        raise ValueError("event ledger ends with an unmatched simulator tool call")
    return pairs


def _compare_public_metrics(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> List[str]:
    mismatches: List[str] = []
    for key, actual_value in actual.items():
        if key not in expected:
            continue
        expected_value = expected[key]
        if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
            if not math.isclose(float(actual_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-9):
                mismatches.append("%s expected=%r actual=%r" % (key, expected_value, actual_value))
        elif expected_value != actual_value:
            mismatches.append("%s expected=%r actual=%r" % (key, expected_value, actual_value))
    return mismatches


def replay_episode(episode_path: Path, manifest_path: Path) -> Dict[str, Any]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(ScenarioConfig)}
    configuration = {key: value for key, value in manifest["configuration"].items() if key in allowed}
    scenario = ScenarioConfig(**configuration)
    event_paths = sorted(episode_path.parent.glob("events.jsonl*"))
    if len(event_paths) != 1:
        raise FileNotFoundError("expected exactly one event ledger beside %s" % episode_path)
    recorded = EventLedger.read_jsonl(event_paths[0])
    pairs = _tool_pairs(recorded.events)
    pairs_by_step: Dict[int, List[Tuple[Event, Event]]] = {}
    result_by_call_id = {
        pair[0].event_id: pair[1] for pair in pairs
    }
    for pair in pairs:
        pairs_by_step.setdefault(pair[0].step, []).append(pair)
    replay_actions: Dict[int, List[Tuple[str, Event]]] = {}
    for event in recorded.events:
        if event.kind == "tool_call":
            replay_actions.setdefault(event.step, []).append(("tool", event))
        elif (
            event.kind == "message"
            and event.payload.get("kind") in ("fixed_status", "entropy_alert")
        ):
            replay_actions.setdefault(event.step, []).append(("protocol", event))

    env = LogisticsEnvironment(scenario)
    metric_mismatches: List[str] = []
    tool_mismatches: List[str] = []
    expected_series = {int(row["step"]): row for row in episode["time_series"]}
    for _ in range(scenario.horizon):
        env.transition()
        env.deliver_observations()
        expected = expected_series.get(env.step_index, {})
        metric_mismatches.extend(
            "step %d: %s" % (env.step_index, mismatch)
            for mismatch in _compare_public_metrics(expected, env.public_metrics())
        )
        for action_kind, event in replay_actions.get(env.step_index, []):
            if action_kind == "tool":
                payload = event.payload
                result = env.execute_tool(
                    event.actor,
                    str(payload["tool"]),
                    dict(payload["arguments"]),
                )
                recorded_result = result_by_call_id[event.event_id].payload
                if (
                    result.ok != bool(recorded_result.get("ok"))
                    or result.code != recorded_result.get("code")
                ):
                    tool_mismatches.append(
                        "event %s: expected %s/%s, replayed %s/%s" % (
                            event.event_id,
                            recorded_result.get("ok"), recorded_result.get("code"),
                            result.ok, result.code,
                        )
                    )
                continue
            payload = event.payload
            message_payload = payload.get("payload", {})
            if payload.get("kind") == "fixed_status":
                result = env.send_fixed_status_summary(
                    event.actor,
                    str(payload["recipient"]),
                    str(message_payload["pressure"]),
                    str(message_payload["capacity"]),
                    str(message_payload["commitment_strain"]),
                )
            else:
                result = env.send_entropy_alert(
                    event.actor,
                    str(payload["recipient"]),
                    int(message_payload["recommended_mode"]),
                    str(message_payload["anomaly_level"]),
                )
            expected_code = "packet_dropped" if payload.get("dropped") else "sent"
            if not result.ok or result.code != expected_code:
                tool_mismatches.append(
                    "event %s protocol message expected %s, replayed %s/%s"
                    % (event.event_id, expected_code, result.ok, result.code)
                )
        env.advance()
    return {
        "run_id": episode["run_id"],
        "episode_path": str(episode_path),
        "manifest_path": str(manifest_path),
        "recorded_tool_calls": len(pairs),
        "metric_mismatches": metric_mismatches,
        "tool_result_mismatches": tool_mismatches,
        "conservation_error": env.conservation_error(),
        "replay_passed": not metric_mismatches and not tool_mismatches and abs(env.conservation_error()) < 1e-8,
    }


def replay_results(
    results_root: Path,
    stages: Sequence[str],
    run_id_contains: Sequence[str] = (),
    report_name: str = "replay_report.json",
) -> Dict[str, Any]:
    if Path(report_name).name != report_name:
        raise ValueError("report_name must be a filename, not a path")
    records: List[Dict[str, Any]] = []
    for stage in stages:
        for episode_path in sorted((results_root / "raw" / stage).glob("*/episode.json")):
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            if run_id_contains and not any(value in episode["run_id"] for value in run_id_contains):
                continue
            manifest = results_root / "manifests" / (episode["run_id"] + ".json")
            if not manifest.exists():
                records.append({
                    "run_id": episode["run_id"], "replay_passed": False,
                    "error": "manifest missing",
                })
                continue
            try:
                records.append(replay_episode(episode_path, manifest))
            except Exception as error:
                records.append({
                    "run_id": episode["run_id"], "replay_passed": False,
                    "error": "%s: %s" % (type(error).__name__, error),
                })
    report = {
        "stages": list(stages),
        "run_id_contains": list(run_id_contains),
        "episodes_checked": len(records),
        "episodes_passed": sum(record.get("replay_passed", False) for record in records),
        "records": records,
    }
    output = results_root / "reproducibility" / report_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if records and report["episodes_passed"] != len(records):
        raise RuntimeError("one or more quantitative replays failed; see %s" % output)
    return report
