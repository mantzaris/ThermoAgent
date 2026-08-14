"""Exact v3 quantitative replay from recorded actions and interventions."""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .events import Event, EventLedger
from .human_environment import HumanOversightEnvironment, HumanScenarioConfig
from .human_operator import OperatorIntervention, OperatorView, validate_operator_view


def _compare(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> List[str]:
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


def _nonfinite_count(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(not math.isfinite(float(value)))
    if isinstance(value, Mapping):
        return sum(_nonfinite_count(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nonfinite_count(child) for child in value)
    return 0


def _tool_results(events: Sequence[Event]) -> Dict[str, Event]:
    output: Dict[str, Event] = {}
    pending: List[Event] = []
    for event in events:
        if event.kind == "tool_call":
            pending.append(event)
        elif event.kind == "tool_result" and pending:
            call = pending.pop(0)
            output[call.event_id] = event
    if pending:
        raise ValueError("v3 ledger ends with an unmatched tool call")
    return output


def _replay_event(
    env: HumanOversightEnvironment,
    event: Event,
    expected_results: Mapping[str, Event],
    mismatches: List[str],
) -> None:
    if event.kind == "tool_call":
        result = env.execute_tool(
            event.actor,
            str(event.payload["tool"]),
            dict(event.payload["arguments"]),
        )
        expected = expected_results[event.event_id].payload
        if result.ok != bool(expected.get("ok")) or result.code != expected.get("code"):
            mismatches.append(
                "%s expected %s/%s replayed %s/%s" % (
                    event.event_id,
                    expected.get("ok"), expected.get("code"),
                    result.ok, result.code,
                )
            )
    elif event.kind == "operator_action":
        intervention = OperatorIntervention(**event.payload)
        env.execute_human_intervention(intervention)
    elif (
        event.kind == "human_directive"
        and event.actor in env.agents
        and event.payload.get("response") in ("accepted", "rejected")
    ):
        accepted, _ = env.directive_response(event.actor)
        if accepted != (event.payload["response"] == "accepted"):
            mismatches.append(
                "%s directive response expected %s replayed %s" % (
                    event.event_id, event.payload["response"], accepted,
                )
            )
    elif event.kind == "message" and event.payload.get("kind") == "fixed_status":
        payload = event.payload["payload"]
        result = env.send_fixed_status_summary(
            event.actor,
            str(event.payload["recipient"]),
            str(payload["pressure"]),
            str(payload["capacity"]),
            str(payload["commitment_strain"]),
        )
        expected_code = "packet_dropped" if event.payload.get("dropped") else "sent"
        if result.code != expected_code:
            mismatches.append(
                "%s fixed status expected %s replayed %s" % (
                    event.event_id, expected_code, result.code,
                )
            )


def replay_human_episode(episode_path: Path, manifest_path: Path) -> Dict[str, Any]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(HumanScenarioConfig)}
    config = HumanScenarioConfig(**{
        key: value for key, value in manifest["configuration"].items()
        if key in allowed
    })
    event_paths = sorted(episode_path.parent.glob("events.jsonl*"))
    if len(event_paths) != 1:
        raise FileNotFoundError("expected exactly one v3 ledger beside %s" % episode_path)
    recorded = EventLedger.read_jsonl(event_paths[0])
    privacy_failures: List[str] = []
    for event in recorded.events:
        if event.kind != "operator_view":
            continue
        try:
            validate_operator_view(OperatorView(**event.payload))
        except Exception as error:
            privacy_failures.append(
                "%s: %s: %s" % (event.event_id, type(error).__name__, error)
            )
    nonfinite_values = _nonfinite_count(episode) + sum(
        _nonfinite_count(event.payload) for event in recorded.events
    )
    expected_results = _tool_results(recorded.events)
    by_step: Dict[int, List[Event]] = {}
    for event in recorded.events:
        by_step.setdefault(event.step, []).append(event)
    expected_series = {int(row["step"]): row for row in episode["time_series"]}
    env = HumanOversightEnvironment(config)
    metric_mismatches: List[str] = []
    action_mismatches: List[str] = []
    replay_kinds = {"tool_call", "operator_action", "human_directive", "message"}
    for _ in range(config.horizon):
        env.transition()
        env.deliver_observations()
        step_events = by_step.get(env.step_index, [])
        metric_position = next((
            index for index, event in enumerate(step_events)
            if event.kind == "metric" and event.actor == "v3_evaluator"
        ), len(step_events))
        for event in step_events[:metric_position]:
            if event.kind in replay_kinds:
                _replay_event(env, event, expected_results, action_mismatches)
        expected = expected_series[env.step_index]
        metric_mismatches.extend(
            "step %d: %s" % (env.step_index, mismatch)
            for mismatch in _compare(expected, env.public_metrics())
        )
        metric_mismatches.extend(
            "step %d v3: %s" % (env.step_index, mismatch)
            for mismatch in _compare(expected, env.v3_metrics())
        )
        for event in step_events[metric_position + 1 :]:
            if event.kind in replay_kinds:
                _replay_event(env, event, expected_results, action_mismatches)
        env.advance()
    passed = (
        not metric_mismatches
        and not action_mismatches
        and not privacy_failures
        and nonfinite_values == 0
        and abs(env.conservation_error()) < 1e-8
    )
    return {
        "run_id": episode["run_id"],
        "episode_path": str(episode_path),
        "manifest_path": str(manifest_path),
        "recorded_tool_calls": len(expected_results),
        "metric_mismatches": metric_mismatches,
        "tool_or_intervention_mismatches": action_mismatches,
        "operator_view_privacy_failures": privacy_failures,
        "nonfinite_values": nonfinite_values,
        "conservation_error": env.conservation_error(),
        "replay_passed": passed,
    }


def replay_human_results(
    results_root: Path,
    stages: Sequence[str],
    report_name: str = "human_replay_report.json",
) -> Dict[str, Any]:
    if Path(report_name).name != report_name:
        raise ValueError("report name must be a filename")
    records: List[Dict[str, Any]] = []
    for stage in stages:
        for episode_path in sorted((results_root / "raw" / stage).glob("*/episode.json")):
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            manifest = results_root / "manifests" / (episode["run_id"] + ".json")
            if not manifest.is_file():
                records.append({
                    "run_id": episode["run_id"],
                    "replay_passed": False,
                    "error": "manifest missing",
                })
                continue
            try:
                records.append(replay_human_episode(episode_path, manifest))
            except Exception as error:
                records.append({
                    "run_id": episode["run_id"],
                    "replay_passed": False,
                    "error": "%s: %s" % (type(error).__name__, error),
                })
    report = {
        "created_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "stages": list(stages),
        "episodes_checked": len(records),
        "episodes_passed": sum(bool(row.get("replay_passed")) for row in records),
        "mismatches": sum(not bool(row.get("replay_passed")) for row in records),
        "operator_view_privacy_failures": sum(
            len(row.get("operator_view_privacy_failures", [])) for row in records
        ),
        "nonfinite_values": sum(
            int(row.get("nonfinite_values", 0)) for row in records
        ),
        "maximum_absolute_conservation_residual": max(
            [abs(float(row.get("conservation_error", 0.0))) for row in records]
            or [0.0]
        ),
        "records": records,
    }
    output = results_root / "reproducibility" / report_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if records and report["episodes_passed"] != len(records):
        raise RuntimeError("one or more v3 replays failed; see %s" % output)
    return report
