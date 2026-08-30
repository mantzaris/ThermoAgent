"""Prospective V14 pilot, quench-panel design, and resumable Qwen execution."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from thermoagent.statmech_llm.discovery.provider import (
    MODEL_ID,
    MODEL_REVISION,
    QwenStatmechProvider,
    schema_checksum,
)
from thermoagent.statmech_llm.discovery.replay import RecordedDecisionProvider, RecordedDecisionStore

from .simulation import build_reciprocal_graph, run_corrected_quench_trajectory
from .workflow import (
    artifact_root,
    atomic_csv,
    atomic_json,
    execution_source_checksum,
    load_yaml,
    sha256_file,
    stage_lock,
    utc_now,
)


def _require_qwen_opt_in() -> None:
    if os.environ.get("THERMOAGENT_CORRECTED_QUENCH_ENABLE_QWEN") != "1":
        raise RuntimeError("V14 Qwen execution requires THERMOAGENT_CORRECTED_QUENCH_ENABLE_QWEN=1 on the authorized Pod")


def _provider(repository: Path, stage: str, settings: Mapping[str, object]) -> QwenStatmechProvider:
    return QwenStatmechProvider(
        artifact_root() / "raw" / stage,
        repository,
        inference_temperature=float(settings.get("inference_sampling_temperature", 0.5)),
        top_p=float(settings["top_p"]),
        maximum_new_tokens=int(settings["maximum_new_tokens"]),
    )


def _stable_seed(token: str, offset: int = 0) -> int:
    digest = hashlib.sha256(str(token).encode("utf-8")).digest()
    return int(14140000 + int(offset) + int.from_bytes(digest[:4], "big") % 400000)


def formal_panel_design(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    network = protocol["network"]  # type: ignore[index]
    quench = protocol["quench"]  # type: ignore[index]
    output: List[Dict[str, object]] = []
    clusters = list(network["cluster_ids"])  # type: ignore[index]
    if len(clusters) != int(network["graph_environment_clusters"]):  # type: ignore[index]
        raise ValueError("cluster manifest does not match declared count")
    for replicate, cluster in enumerate(clusters):
        for condition in quench["conditions"]:  # type: ignore[index]
            output.append(
                {
                    "family": "V14_quench_replication",
                    "subset": "prospective_formal",
                    "cluster_id": str(cluster),
                    "panel_id": "%s_%s" % (cluster, condition),
                    "n_agents": int(network["agent_count"]),  # type: ignore[index]
                    "topology": str(network["topology"]),  # type: ignore[index]
                    "replicate": int(replicate),
                    "coupling_strength": float(network["coupling_strength"]),  # type: ignore[index]
                    "sampling_temperature": float(protocol["model"]["inference_sampling_temperature"]),  # type: ignore[index]
                    "regime": "markovized",
                    "initial_condition": str(quench["initial_condition"]),  # type: ignore[index]
                    "sweeps": int(quench["sweeps"]),  # type: ignore[index]
                    "burn_in_sweeps": 0,
                    "disruption": str(condition),
                    "periods_sweeps": list(quench["periods_sweeps"]),  # type: ignore[index]
                }
            )
    return output


def panel_seed(panel: Mapping[str, object]) -> int:
    """Return a cluster-level tape seed shared by all four matched arms."""

    return _stable_seed("V14|%s" % panel["cluster_id"])


def graph_for_panel(panel: Mapping[str, object]):
    seed = panel_seed(panel)
    return build_reciprocal_graph(int(panel["n_agents"]), str(panel["topology"]), seed + 17)


def expected_decisions(protocol: Mapping[str, object]) -> int:
    return int(sum(int(panel["n_agents"]) * int(panel["sweeps"]) for panel in formal_panel_design(protocol)))


def _pilot_panels(settings: Mapping[str, object]) -> List[Dict[str, object]]:
    return [
        {
            "family": "V14_engineering_pilot",
            "subset": "engineering_only",
            "cluster_id": "V14_pilot_g0",
            "panel_id": "V14_pilot_%s" % condition,
            "n_agents": int(settings["n_agents"]),
            "topology": str(settings["topology"]),
            "replicate": 0,
            "coupling_strength": float(settings["coupling_strength"]),
            "sampling_temperature": float(settings["inference_sampling_temperature"]),
            "regime": "markovized",
            "initial_condition": "disordered",
            "sweeps": int(settings["sweeps_per_condition"]),
            "burn_in_sweeps": 0,
            "disruption": str(condition),
            "periods_sweeps": list(settings["periods_sweeps"]),
        }
        for condition in settings["conditions"]  # type: ignore[index]
    ]


class _RecordedPilotProvider(RecordedDecisionProvider):
    """Replay a completed pilot whose post-generation summary was interrupted."""

    def __init__(self, store: RecordedDecisionStore, digests: Sequence[str], records: Sequence[Mapping[str, object]]) -> None:
        super().__init__(store, digests)
        self.records = list(records)

    def environment_manifest(self) -> Dict[str, object]:
        first = self.records[0]
        return {
            "model_id": first["model_id"],
            "model_revision": first["model_revision"],
            "quantization": "NF4 double quantization; BF16 computation",
            "backend": "Transformers AutoModelForCausalLM; deterministic recorded-decision replay",
            "inference_sampling_temperature": first["inference_sampling_temperature"],
            "top_p": first["top_p"],
            "maximum_new_tokens": 96,
            "schema_sha256": first["schema_sha256"],
            "recovered_from_retained_raw_records": True,
            "accounting": {
                "decision_requests": len(self.records),
                "model_calls": int(sum(int(item["model_calls"]) for item in self.records)),
                "prompt_tokens": int(sum(int(item["prompt_tokens"]) for item in self.records)),
                "generated_tokens": int(sum(int(item["generated_tokens"]) for item in self.records)),
                "latency_seconds": float(sum(float(item["latency_seconds"]) for item in self.records)),
                "first_pass_valid": int(sum(bool(item["first_pass_valid"]) for item in self.records)),
                "repaired_valid": int(sum(bool(item["repaired"]) and bool(item["valid"]) for item in self.records)),
                "invalid_after_repair": int(sum(not bool(item["valid"]) for item in self.records)),
                "model_loading_seconds": None,
            },
        }


def _retained_pilot_provider(repository: Path, settings: Mapping[str, object]):
    raw_root = artifact_root() / "raw" / "pilot" / str(settings["attempt_id"])
    paths = sorted(raw_root.glob("call_*.json"))
    if len(paths) != int(settings["expected_decisions"]):
        return _provider(
            repository,
            "pilot/%s" % settings["attempt_id"],
            {"inference_sampling_temperature": 0.5, "top_p": 0.9, "maximum_new_tokens": 96},
        )
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(not isinstance(item, dict) for item in records):
        raise ValueError("retained pilot model record is not an object")
    digests = [sha256_file(path) for path in paths]
    return _RecordedPilotProvider(RecordedDecisionStore(raw_root), digests, records)


def _pilot_summary(
    rows: Sequence[Mapping[str, object]],
    provider,
    settings: Mapping[str, object],
) -> Dict[str, object]:
    frame = pd.DataFrame(rows)
    valid = frame[frame["valid_after_repair"] == 1]
    minus_plus = int(np.sum((valid["belief_before"] == -1) & (valid["belief_after"] == 1)))
    plus_minus = int(np.sum((valid["belief_before"] == 1) & (valid["belief_after"] == -1)))
    occupancy = float(np.mean(valid["belief_after"] == 1)) if len(valid) else float("nan")
    seconds_per_decision = float(frame["latency_seconds"].sum() / max(len(frame), 1))
    projected_hours = seconds_per_decision * 17280.0 / 3600.0
    projected_prompt = int(round(float(frame["prompt_tokens"].mean()) * 17280.0))
    interval = [float(item) for item in settings["latent_plus_occupancy_interval"]]  # type: ignore[index]
    field = frame[frame["disruption"] == "field_reversal"]
    partition = frame[frame["disruption"] == "network_partition"]
    corruption = frame[frame["disruption"] == "message_corruption"]
    checks = {
        "decision_limit": len(frame) <= int(settings["decision_limit"]),
        "declared_decision_count": len(frame) == int(settings["expected_decisions"]),
        "first_pass_valid": float(frame["first_pass_valid"].mean()) >= float(settings["first_pass_valid_minimum"]),
        "after_repair_valid": float(frame["valid_after_repair"].mean()) >= float(settings["after_repair_valid_minimum"]),
        "balanced_occupancy": interval[0] <= occupancy <= interval[1],
        "minus_to_plus": minus_plus >= int(settings["minimum_transitions_each_direction"]),
        "plus_to_minus": plus_minus >= int(settings["minimum_transitions_each_direction"]),
        "privacy": int(frame["unrelated_peer_private_mutations"].sum()) == 0,
        "message_delivery": int(frame["messages_delivered"].sum()) > 0,
        "field_schedule": bool(field["field_reversed"].sum() > 0 and field.iloc[-1]["field_reversed"] == 0),
        "partition_schedule": bool(partition["partition_active"].sum() > 0 and partition.iloc[-1]["partition_active"] == 0),
        "corruption_schedule": bool(corruption["message_corrupted"].sum() > 0),
        "projected_generation_hours": projected_hours <= float(settings["maximum_projected_formal_gpu_hours"]),
        "projected_prompt_tokens": projected_prompt <= int(settings["maximum_projected_formal_prompt_tokens"]),
    }
    return {
        "generated_at": utc_now(),
        "inspection_boundary": list(settings["inspection_boundary"]),
        "scientific_nonreciprocity_or_quench_effects_inspected": False,
        "decision_requests": int(len(frame)),
        "first_pass_valid_fraction": float(frame["first_pass_valid"].mean()),
        "valid_after_repair_fraction": float(frame["valid_after_repair"].mean()),
        "latent_plus_occupancy": occupancy,
        "belief_minus_to_plus": minus_plus,
        "belief_plus_to_minus": plus_minus,
        "mean_latency_seconds_per_decision": seconds_per_decision,
        "mean_prompt_tokens": float(frame["prompt_tokens"].mean()),
        "mean_generated_tokens": float(frame["generated_tokens"].mean()),
        "projected_formal_generation_hours": projected_hours,
        "projected_formal_prompt_tokens": projected_prompt,
        "checks": checks,
        "engineering_passed": bool(all(checks.values())),
        "provider_environment": provider.environment_manifest(),
    }


def run_engineering_pilot(repository: Path) -> Dict[str, object]:
    _require_qwen_opt_in()
    repository = Path(repository).resolve()
    configuration = load_yaml(repository / "configs/statmech_llm/corrected_quench/engineering.yaml")
    settings = configuration["pilot"]  # type: ignore[index]
    provider = _retained_pilot_provider(repository, settings)
    rows: List[Dict[str, object]] = []
    with stage_lock("pilot_%s" % settings["attempt_id"]):  # type: ignore[index]
        for panel in _pilot_panels(settings):
            rows.extend(
                run_corrected_quench_trajectory(
                    provider,
                    graph_for_panel(panel),
                    panel_seed(panel),
                    int(panel["sweeps"]),
                    float(panel["coupling_strength"]),
                    float(panel["sampling_temperature"]),
                    str(panel["disruption"]),
                    panel["periods_sweeps"],  # type: ignore[arg-type]
                    metadata={key: panel[key] for key in ("family", "subset", "cluster_id", "panel_id", "burn_in_sweeps")},
                )
            )
        if isinstance(provider, _RecordedPilotProvider):
            provider.assert_consumed()
        destination = artifact_root() / "pilot" / str(settings["attempt_id"])
        atomic_csv(rows, destination / "transitions.csv")
        summary = _pilot_summary(rows, provider, settings)
        atomic_json(summary, destination / "summary.json")
    return summary


def _retain_incomplete(path: Path, reason: str) -> None:
    destination = path.with_name(path.stem + ".retained_incomplete_%s_%s.csv" % (reason, int(time.time())))
    os.replace(str(path), str(destination))


def _formal_frames(root: Path) -> List[pd.DataFrame]:
    return [
        pd.read_csv(path)
        for path in sorted((root / "panels").glob("*.csv"))
        if ".retained_incomplete_" not in path.name
    ]


def _raw_resource_totals(stage: str) -> Dict[str, float]:
    records = list((artifact_root() / "raw" / stage).glob("call_*.json"))
    if not records:
        return {
            "attempted_decisions": 0.0,
            "model_calls": 0.0,
            "prompt_tokens": 0.0,
            "generated_tokens": 0.0,
            "latency_seconds": 0.0,
        }
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in records]
    return {
        "attempted_decisions": float(len(payloads)),
        "model_calls": float(sum(int(item["model_calls"]) for item in payloads)),
        "prompt_tokens": float(sum(int(item["prompt_tokens"]) for item in payloads)),
        "generated_tokens": float(sum(int(item["generated_tokens"]) for item in payloads)),
        "latency_seconds": float(sum(float(item["latency_seconds"]) for item in payloads)),
    }


def _enforce_resource_ceiling(protocol: Mapping[str, object]) -> None:
    totals = _raw_resource_totals("formal")
    compute = protocol["compute"]  # type: ignore[index]
    if totals["latency_seconds"] >= 3600.0 * float(compute["hard_generation_gpu_hours"]):  # type: ignore[index]
        raise RuntimeError("V14 generation-hour ceiling reached before the next atomic trajectory")
    if totals["prompt_tokens"] >= float(compute["maximum_formal_prompt_tokens"]):  # type: ignore[index]
        raise RuntimeError("V14 prompt-token ceiling reached before the next atomic trajectory")


def run_formal_experiment(repository: Path) -> Dict[str, object]:
    _require_qwen_opt_in()
    repository = Path(repository).resolve()
    frozen_path = repository / "configs/statmech_llm/corrected_quench/protocol.yaml"
    if not frozen_path.exists():
        raise RuntimeError("V14 protocol has not been frozen")
    protocol = load_yaml(frozen_path)
    current_source = execution_source_checksum(repository)
    frozen_source = str(protocol["provenance"]["execution_source_sha256"])  # type: ignore[index]
    if current_source != frozen_source:
        raise RuntimeError("V14 execution source changed after protocol freeze")
    expected = expected_decisions(protocol)
    if expected != int(protocol["compute"]["expected_formal_decisions"]):  # type: ignore[index]
        raise RuntimeError("V14 frozen decision accounting is inconsistent")
    if expected > int(protocol["compute"]["maximum_formal_decisions"]):  # type: ignore[index]
        raise RuntimeError("V14 formal plan exceeds its frozen decision ceiling")
    provider = _provider(repository, "formal", protocol["model"])  # type: ignore[arg-type,index]
    root = artifact_root() / "formal"
    panels_root = root / "panels"
    panels_root.mkdir(parents=True, exist_ok=True)
    completed: List[Dict[str, object]] = []
    invocation_started = time.perf_counter()
    with stage_lock("formal"):
        state_path = root / "run_state.json"
        if not state_path.exists():
            atomic_json(
                {
                    "started_at": utc_now(),
                    "planned_decisions": expected,
                    "planned_trajectories": len(formal_panel_design(protocol)),
                    "protocol_sha256": sha256_file(frozen_path),
                    "execution_source_sha256": current_source,
                },
                state_path,
            )
        for panel in formal_panel_design(protocol):
            path = panels_root / (str(panel["panel_id"]) + ".csv")
            expected_rows = int(panel["n_agents"]) * int(panel["sweeps"])
            if path.exists() and len(pd.read_csv(path)) == expected_rows:
                completed.append({"unit": panel["panel_id"], "rows": expected_rows})
                continue
            _enforce_resource_ceiling(protocol)
            if path.exists():
                _retain_incomplete(path, "row_count")
            rows = run_corrected_quench_trajectory(
                provider,
                graph_for_panel(panel),
                panel_seed(panel),
                int(panel["sweeps"]),
                float(panel["coupling_strength"]),
                float(panel["sampling_temperature"]),
                str(panel["disruption"]),
                panel["periods_sweeps"],  # type: ignore[arg-type]
                metadata={key: panel[key] for key in ("family", "subset", "cluster_id", "panel_id", "burn_in_sweeps")},
            )
            atomic_csv(rows, path)
            completed.append({"unit": panel["panel_id"], "rows": len(rows)})
            atomic_json(
                {
                    "updated_at": utc_now(),
                    "completed_trajectories": len(completed),
                    "planned_trajectories": len(formal_panel_design(protocol)),
                    "completed_rows": int(sum(int(item["rows"]) for item in completed)),
                    "planned_decisions": expected,
                    "last_unit": panel["panel_id"],
                },
                root / "supervisor_status.json",
            )
        frames = _formal_frames(root)
        observed = int(sum(len(frame) for frame in frames))
        if observed != expected:
            raise RuntimeError("formal rows %d differ from frozen %d" % (observed, expected))
        totals = {
            "model_calls": int(sum(frame["model_calls"].sum() for frame in frames)),
            "prompt_tokens": int(sum(frame["prompt_tokens"].sum() for frame in frames)),
            "generated_tokens": int(sum(frame["generated_tokens"].sum() for frame in frames)),
            "invalid_after_repair": int(sum((frame["valid_after_repair"] == 0).sum() for frame in frames)),
            "generation_latency_seconds": float(sum(frame["latency_seconds"].sum() for frame in frames)),
        }
        raw = _raw_resource_totals("formal")
        completion: Dict[str, object] = {
            "status": "complete",
            "completed_at": utc_now(),
            "protocol_sha256": sha256_file(frozen_path),
            "execution_source_sha256": current_source,
            "planned_decisions": expected,
            "observed_decision_rows": observed,
            "dynamic_trajectories": len(completed),
            **totals,
            "generation_gpu_hours": totals["generation_latency_seconds"] / 3600.0,
            "current_invocation_wall_seconds": time.perf_counter() - invocation_started,
            "all_formal_attempted_decisions_including_invalidated": int(raw["attempted_decisions"]),
            "all_formal_model_calls_including_invalidated": int(raw["model_calls"]),
            "all_formal_prompt_tokens_including_invalidated": int(raw["prompt_tokens"]),
            "all_formal_generated_tokens_including_invalidated": int(raw["generated_tokens"]),
            "all_formal_generation_latency_seconds_including_invalidated": float(raw["latency_seconds"]),
            "all_formal_generation_gpu_hours_including_invalidated": float(raw["latency_seconds"] / 3600.0),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "schema_sha256": schema_checksum(),
            "provider_environment_current_invocation": provider.environment_manifest(),
        }
        atomic_json(completion, root / "completion.json")
        atomic_json({**completion, "last_unit": "complete"}, root / "supervisor_status.json")
    return completion


__all__ = [
    "expected_decisions",
    "formal_panel_design",
    "graph_for_panel",
    "panel_seed",
    "run_engineering_pilot",
    "run_formal_experiment",
]
