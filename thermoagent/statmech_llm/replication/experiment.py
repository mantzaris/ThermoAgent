"""Prospective V13 pilot, frozen panel design, and resumable Qwen execution."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from thermoagent.statmech_llm.discovery.core import (
    AgentDecision,
    IndependentStatmechAgent,
    LatentMapping,
    SignalPacket,
    build_agent_prompt,
)
from thermoagent.statmech_llm.discovery.provider import (
    MODEL_ID,
    MODEL_REVISION,
    InvalidStructuredDecision,
    QwenStatmechProvider,
    schema_checksum,
)

from .simulation import build_reciprocal_graph, run_replication_trajectory
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
    if os.environ.get("THERMOAGENT_REPLICATION_ENABLE_QWEN") != "1":
        raise RuntimeError("V13 Qwen execution is locked to the existing authorized RunPod")


def _provider(repository: Path, stage: str, settings: Mapping[str, object]) -> QwenStatmechProvider:
    return QwenStatmechProvider(
        artifact_root() / "raw" / stage,
        repository,
        inference_temperature=float(settings.get("inference_sampling_temperature", 0.50)),
        top_p=float(settings["top_p"]),
        maximum_new_tokens=int(settings["maximum_new_tokens"]),
    )


def _stable_seed(token: str, offset: int = 0) -> int:
    digest = hashlib.sha256(str(token).encode("utf-8")).digest()
    return int(13130000 + int(offset) + int.from_bytes(digest[:4], "big") % 500000)


def formal_panel_design(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    package_a = protocol["work_package_a"]  # type: ignore[index]
    for label, topology in (("modular_primary", "modular"), ("ring_replication", "ring")):
        settings = package_a[label]  # type: ignore[index]
        for n_agents in settings["agent_counts"]:  # type: ignore[index]
            for replicate in range(int(settings["graph_environment_clusters"])):
                cluster = "A_%s_n%d_g%d" % ("mod" if topology == "modular" else "ring", int(n_agents), replicate)
                for coupling in settings["coupling_strengths"]:  # type: ignore[index]
                    for temperature in settings["inference_sampling_temperatures"]:  # type: ignore[index]
                        output.append(
                            {
                                "family": "A_order_fluctuation",
                                "subset": label,
                                "cluster_id": cluster,
                                "panel_id": "%s_J%.2f_T%.2f" % (cluster, float(coupling), float(temperature)),
                                "n_agents": int(n_agents),
                                "topology": topology,
                                "replicate": replicate,
                                "coupling_strength": float(coupling),
                                "sampling_temperature": float(temperature),
                                "regime": "markovized",
                                "initial_condition": str(settings["initial_condition"]),
                                "sweeps": int(settings["sweeps"]),
                                "burn_in_sweeps": int(settings["burn_in_sweeps"]),
                                "disruption": "nominal",
                                "periods_sweeps": None,
                            }
                        )
    relaxation = package_a["ordered_relaxation"]  # type: ignore[index]
    for replicate in range(int(relaxation["graph_environment_clusters"])):
        cluster = "A_relax_n16_g%d" % replicate
        for coupling in relaxation["coupling_strengths"]:  # type: ignore[index]
            for temperature in relaxation["inference_sampling_temperatures"]:  # type: ignore[index]
                output.append(
                    {
                        "family": "A_ordered_relaxation",
                        "subset": "ordered_relaxation",
                        "cluster_id": cluster,
                        "panel_id": "%s_J%.2f_T%.2f" % (cluster, float(coupling), float(temperature)),
                        "n_agents": int(relaxation["agent_count"]),
                        "topology": str(relaxation["topology"]),
                        "replicate": replicate,
                        "coupling_strength": float(coupling),
                        "sampling_temperature": float(temperature),
                        "regime": "markovized",
                        "initial_condition": str(relaxation["initial_condition"]),
                        "sweeps": int(relaxation["sweeps"]),
                        "burn_in_sweeps": 0,
                        "disruption": "nominal",
                        "periods_sweeps": None,
                    }
                )
    memory = protocol["work_package_b"]  # type: ignore[index]
    for replicate in range(int(memory["graph_environment_clusters"])):
        cluster = "B_memory_n16_g%d" % replicate
        for regime in memory["regimes"]:  # type: ignore[index]
            output.append(
                {
                    "family": "B_memory_quench",
                    "subset": "memory_confirmation",
                    "cluster_id": cluster,
                    "panel_id": "%s_%s" % (cluster, regime),
                    "n_agents": int(memory["agent_count"]),
                    "topology": str(memory["topology"]),
                    "replicate": replicate,
                    "coupling_strength": float(memory["coupling_strength"]),
                    "sampling_temperature": float(memory["inference_sampling_temperature"]),
                    "regime": str(regime),
                    "initial_condition": str(memory["initial_condition"]),
                    "sweeps": int(memory["sweeps"]),
                    "burn_in_sweeps": 0,
                    "disruption": str(memory["disruption"]),
                    "periods_sweeps": list(memory["periods_sweeps"]),
                }
            )
    disruptions = protocol["work_package_c"]  # type: ignore[index]
    for replicate in range(int(disruptions["graph_environment_clusters"])):
        cluster = "C_quench_n16_g%d" % replicate
        for condition in disruptions["conditions"]:  # type: ignore[index]
            output.append(
                {
                    "family": "C_disruption_recovery",
                    "subset": "controlled_quench",
                    "cluster_id": cluster,
                    "panel_id": "%s_%s" % (cluster, condition),
                    "n_agents": int(disruptions["agent_count"]),
                    "topology": str(disruptions["topology"]),
                    "replicate": replicate,
                    "coupling_strength": float(disruptions["coupling_strength"]),
                    "sampling_temperature": float(disruptions["inference_sampling_temperature"]),
                    "regime": "markovized",
                    "initial_condition": str(disruptions["initial_condition"]),
                    "sweeps": int(disruptions["sweeps"]),
                    "burn_in_sweeps": 0,
                    "disruption": str(condition),
                    "periods_sweeps": list(disruptions["periods_sweeps"]),
                }
            )
    return output


def panel_seed(panel: Mapping[str, object]) -> int:
    # Matched factor/regime/disruption arms share graph, fields, schedule,
    # counterbalancing, and inference seeds by excluding the arm from this key.
    return _stable_seed("%s|%s" % (panel["family"], panel["cluster_id"]))


def graph_for_panel(panel: Mapping[str, object]):
    seed = panel_seed(panel)
    return build_reciprocal_graph(int(panel["n_agents"]), str(panel["topology"]), seed + 17)


def expected_decisions(protocol: Mapping[str, object]) -> int:
    dynamic = sum(int(panel["n_agents"]) * int(panel["sweeps"]) for panel in formal_panel_design(protocol))
    micro = int(protocol["microscopic_response"]["expected_decisions"])  # type: ignore[index]
    return int(dynamic + micro)


def microscopic_response_rows(provider, protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    settings = protocol["microscopic_response"]  # type: ignore[index]
    rows: List[Dict[str, object]] = []
    cell_index = 0
    for private in settings["private_fields"]:  # type: ignore[index]
        for neighbor in settings["neighbor_fields"]:  # type: ignore[index]
            for belief in settings["current_beliefs"]:  # type: ignore[index]
                for action in settings["current_actions"]:  # type: ignore[index]
                    for coupling in settings["coupling_strengths"]:  # type: ignore[index]
                        for temperature in settings["inference_sampling_temperatures"]:  # type: ignore[index]
                            for replicate in range(int(settings["independent_replicates_per_cell"])):
                                draw_index = cell_index * int(settings["independent_replicates_per_cell"]) + replicate
                                seed = 13200000 + cell_index * 10 + replicate
                                plus = "amber" if draw_index % 2 == 0 else "cobalt"
                                order = ("amber", "cobalt") if (draw_index // 2) % 2 == 0 else ("cobalt", "amber")
                                mapping = LatentMapping(plus, order)
                                agent = IndependentStatmechAgent(0, "isolated_response_agent", int(private), int(belief), int(action))
                                signals = [int(neighbor), int(neighbor)] if int(neighbor) else [-1, 1]
                                for sender, signal in enumerate(signals, start=1):
                                    agent.receive(SignalPacket(sender, 0, signal, signal, signal, 0.7, 1, 0))
                                prompt = build_agent_prompt(agent, mapping, 0, "markovized", float(coupling), cell_index % 2)
                                invalid = False
                                try:
                                    result = provider.decide(prompt, seed, float(temperature))
                                    decision = AgentDecision.from_mapping(result.payload)
                                    after_belief = mapping.spin(decision.belief_choice)
                                    after_action = mapping.spin(decision.action_choice)
                                except InvalidStructuredDecision as error:
                                    result = error.result
                                    invalid = True
                                    after_belief = int(belief)
                                    after_action = int(action)
                                rows.append(
                                    {
                                        "cell_id": cell_index,
                                        "replicate": replicate,
                                        "private_field": int(private),
                                        "neighbor_field": int(neighbor),
                                        "current_belief": int(belief),
                                        "current_action": int(action),
                                        "coupling_strength": float(coupling),
                                        "sampling_temperature": float(temperature),
                                        "belief_after": int(after_belief),
                                        "action_after": int(after_action),
                                        "belief_switched": int(after_belief != int(belief)),
                                        "action_switched": int(after_action != int(action)),
                                        "latent_plus_label": plus,
                                        "display_order": ";".join(order),
                                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                                        "raw_artifact_sha256": "" if result is None else result.raw_artifact_sha256,
                                        "valid_after_repair": int(not invalid),
                                        "first_pass_valid": int(result.first_pass_valid) if result is not None else 0,
                                        "repaired": int(result.repaired) if result is not None else 0,
                                        "model_calls": int(1 + (not result.first_pass_valid)) if result is not None else 0,
                                        "prompt_tokens": int(result.prompt_tokens) if result is not None else 0,
                                        "generated_tokens": int(result.generated_tokens) if result is not None else 0,
                                        "latency_seconds": float(result.latency_seconds) if result is not None else 0.0,
                                    }
                                )
                            cell_index += 1
    return rows


def _pilot_panels(settings: Mapping[str, object]) -> List[Dict[str, object]]:
    definitions = [
        ("nominal", "markovized", 0.50),
        ("nominal", "markovized", 0.85),
        ("nominal", "persistent_memory", 0.50),
        ("field_reversal", "markovized", 0.50),
        ("network_partition", "markovized", 0.50),
        ("message_corruption", "markovized", 0.50),
    ]
    return [
        {
            "family": "engineering_pilot",
            "subset": "engineering_only",
            "cluster_id": "pilot_g0",
            "panel_id": "pilot_%s_%s_T%.2f" % (disruption, regime, temperature),
            "n_agents": int(settings["n_agents"]),
            "topology": "modular",
            "replicate": 0,
            "coupling_strength": float(settings["coupling_strength"]),
            "sampling_temperature": float(temperature),
            "regime": regime,
            "initial_condition": "disordered",
            "sweeps": int(settings["sweeps_per_condition"]),
            "burn_in_sweeps": 0,
            "disruption": disruption,
            "periods_sweeps": [1, 2, 1],
        }
        for disruption, regime, temperature in definitions
    ]


def _pilot_summary(rows: Sequence[Mapping[str, object]], provider: QwenStatmechProvider, settings: Mapping[str, object]) -> Dict[str, object]:
    frame = pd.DataFrame(rows)
    valid = frame[frame["valid_after_repair"] == 1]
    transitions_minus_plus = int(np.sum((valid["belief_before"] == -1) & (valid["belief_after"] == 1)))
    transitions_plus_minus = int(np.sum((valid["belief_before"] == 1) & (valid["belief_after"] == -1)))
    occupancy = float(np.mean(valid["belief_after"] == 1)) if len(valid) else float("nan")
    latency_per_decision = float(valid["latency_seconds"].sum() / max(len(frame), 1))
    projected = latency_per_decision * 32672.0 / 3600.0
    projected_prompt_tokens = int(round(float(frame["prompt_tokens"].mean()) * 32672))
    interval = [float(value) for value in settings["latent_plus_occupancy_interval"]]  # type: ignore[index]
    checks = {
        "decision_limit": len(frame) <= int(settings["decision_limit"]),
        "first_pass_valid": float(frame["first_pass_valid"].mean()) >= float(settings["first_pass_valid_minimum"]),
        "after_repair_valid": float(frame["valid_after_repair"].mean()) >= float(settings["after_repair_valid_minimum"]),
        "occupancy": interval[0] <= occupancy <= interval[1],
        "minus_to_plus": transitions_minus_plus >= int(settings["minimum_transitions_each_direction"]),
        "plus_to_minus": transitions_plus_minus >= int(settings["minimum_transitions_each_direction"]),
        "privacy": int(frame["unrelated_peer_private_mutations"].sum()) == 0,
        "delivery": int(frame["messages_delivered"].sum()) > 0,
        "field_schedule": bool(frame[frame["disruption"] == "field_reversal"]["field_reversed"].sum() > 0),
        "partition_schedule": bool(frame[frame["disruption"] == "network_partition"]["partition_active"].sum() > 0),
        "corruption_schedule": bool(frame[frame["disruption"] == "message_corruption"]["message_corrupted"].sum() > 0),
        "projected_gpu_hours": projected <= float(settings["maximum_projected_formal_gpu_hours"]),
        "projected_prompt_tokens": projected_prompt_tokens <= int(settings["maximum_projected_prompt_tokens"]),
    }
    return {
        "generated_at": utc_now(),
        "inspection_boundary": list(settings["inspection_boundary"]),
        "scientific_outcomes_inspected": False,
        "decision_requests": int(len(frame)),
        "first_pass_valid_fraction": float(frame["first_pass_valid"].mean()),
        "valid_after_repair_fraction": float(frame["valid_after_repair"].mean()),
        "latent_plus_occupancy": occupancy,
        "belief_minus_to_plus": transitions_minus_plus,
        "belief_plus_to_minus": transitions_plus_minus,
        "mean_latency_seconds_per_decision": latency_per_decision,
        "mean_prompt_tokens": float(frame["prompt_tokens"].mean()),
        "mean_generated_tokens": float(frame["generated_tokens"].mean()),
        "projected_formal_gpu_hours": projected,
        "projected_formal_prompt_tokens": projected_prompt_tokens,
        "checks": checks,
        "engineering_passed": bool(all(checks.values())),
        "provider_environment": provider.environment_manifest(),
    }


def run_engineering_pilot(repository: Path) -> Dict[str, object]:
    _require_qwen_opt_in()
    repository = Path(repository).resolve()
    configuration = load_yaml(repository / "configs/statmech_llm/replication/engineering.yaml")
    settings = configuration["pilot"]  # type: ignore[index]
    provider = _provider(
        repository,
        "pilot/%s" % settings["attempt_id"],  # type: ignore[index]
        {
            "inference_sampling_temperature": 0.50,
            "top_p": 0.90,
            "maximum_new_tokens": 96,
        },
    )
    rows: List[Dict[str, object]] = []
    with stage_lock("pilot_%s" % settings["attempt_id"]):  # type: ignore[index]
        for panel in _pilot_panels(settings):
            seed = panel_seed(panel)
            rows.extend(
                run_replication_trajectory(
                    provider,
                    graph_for_panel(panel),
                    seed,
                    int(panel["sweeps"]),
                    str(panel["regime"]),
                    float(panel["coupling_strength"]),
                    float(panel["sampling_temperature"]),
                    str(panel["initial_condition"]),
                    str(panel["disruption"]),
                    panel["periods_sweeps"],  # type: ignore[arg-type]
                    metadata={key: panel[key] for key in ("family", "subset", "cluster_id", "panel_id", "burn_in_sweeps")},
                )
            )
        summary = _pilot_summary(rows, provider, settings)
        destination = artifact_root() / "pilot" / str(settings["attempt_id"])
        atomic_csv(rows, destination / "transitions.csv")
        atomic_json(summary, destination / "summary.json")
    return summary


def _retain_incomplete(path: Path, reason: str) -> None:
    destination = path.with_name(path.stem + ".retained_incomplete_%s_%s.csv" % (reason, int(time.time())))
    os.replace(str(path), str(destination))


def _all_formal_frames(root: Path) -> List[pd.DataFrame]:
    paths = [root / "microscopic_response.csv"] + sorted((root / "panels").glob("*.csv"))
    return [pd.read_csv(path) for path in paths if path.exists() and ".retained_incomplete_" not in path.name]


def _formal_resources(root: Path) -> Dict[str, float]:
    frames = _all_formal_frames(root)
    totals = {
        "rows": float(sum(len(frame) for frame in frames)),
        "prompt_tokens": float(sum(frame["prompt_tokens"].sum() for frame in frames)),
        "generated_tokens": float(sum(frame["generated_tokens"].sum() for frame in frames)),
        "latency_seconds": float(sum(frame["latency_seconds"].sum() for frame in frames)),
    }
    raw_root = artifact_root() / "raw/formal"
    records = list(raw_root.glob("call_*.json"))
    if records:
        raw = [json.loads(path.read_text(encoding="utf-8")) for path in records]
        totals.update(
            {
                "all_attempted_decisions": float(len(raw)),
                "all_model_calls": float(sum(int(item["model_calls"]) for item in raw)),
                "all_prompt_tokens": float(sum(int(item["prompt_tokens"]) for item in raw)),
                "all_generated_tokens": float(sum(int(item["generated_tokens"]) for item in raw)),
                "all_latency_seconds": float(sum(float(item["latency_seconds"]) for item in raw)),
            }
        )
    else:
        totals.update(
            {
                "all_attempted_decisions": totals["rows"],
                "all_model_calls": totals["rows"],
                "all_prompt_tokens": totals["prompt_tokens"],
                "all_generated_tokens": totals["generated_tokens"],
                "all_latency_seconds": totals["latency_seconds"],
            }
        )
    return totals


def _enforce_formal_resource_ceiling(root: Path, protocol: Mapping[str, object]) -> None:
    totals = _formal_resources(root)
    compute = protocol["compute"]  # type: ignore[index]
    if totals["all_latency_seconds"] >= 3600.0 * float(compute["hard_generation_gpu_hours"]):  # type: ignore[index]
        raise RuntimeError("V13 hard generation-hour ceiling reached before the next atomic unit")
    if totals["all_prompt_tokens"] >= float(compute["maximum_formal_raw_prompt_tokens"]):  # type: ignore[index]
        raise RuntimeError("V13 prompt-token ceiling reached before the next atomic unit")


def run_formal_experiment(repository: Path) -> Dict[str, object]:
    _require_qwen_opt_in()
    repository = Path(repository).resolve()
    frozen_path = repository / "configs/statmech_llm/replication/protocol.yaml"
    if not frozen_path.exists():
        raise RuntimeError("V13 protocol is not frozen")
    protocol = load_yaml(frozen_path)
    current_source = execution_source_checksum(repository)
    frozen_source = str(protocol["provenance"]["execution_source_sha256"])  # type: ignore[index]
    if current_source != frozen_source:
        raise RuntimeError("V13 execution source changed after protocol freeze")
    expected = expected_decisions(protocol)
    declared = int(protocol["compute"]["expected_formal_decisions"])  # type: ignore[index]
    if expected != declared or expected > int(protocol["compute"]["maximum_formal_decisions"]):  # type: ignore[index]
        raise RuntimeError("frozen formal decision accounting is inconsistent")
    provider = _provider(repository, "formal", {**protocol["model"], "inference_sampling_temperature": 0.50})  # type: ignore[arg-type,index]
    root = artifact_root() / "formal"
    panel_root = root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    completed: List[Dict[str, object]] = []
    started = time.perf_counter()
    with stage_lock("formal"):
        state_path = root / "run_state.json"
        if not state_path.exists():
            atomic_json(
                {
                    "started_at": utc_now(),
                    "planned_decisions": expected,
                    "protocol_sha256": sha256_file(frozen_path),
                    "execution_source_sha256": current_source,
                },
                state_path,
            )
        micro_path = root / "microscopic_response.csv"
        micro_expected = int(protocol["microscopic_response"]["expected_decisions"])  # type: ignore[index]
        if micro_path.exists() and len(pd.read_csv(micro_path)) != micro_expected:
            _retain_incomplete(micro_path, "row_count")
        if not micro_path.exists():
            atomic_csv(microscopic_response_rows(provider, protocol), micro_path)
        completed.append({"unit": "microscopic_response", "rows": len(pd.read_csv(micro_path))})
        for panel in formal_panel_design(protocol):
            path = panel_root / (str(panel["panel_id"]) + ".csv")
            expected_rows = int(panel["n_agents"]) * int(panel["sweeps"])
            if path.exists() and len(pd.read_csv(path)) == expected_rows:
                completed.append({"unit": panel["panel_id"], "rows": expected_rows})
                continue
            _enforce_formal_resource_ceiling(root, protocol)
            if path.exists():
                _retain_incomplete(path, "row_count")
            seed = panel_seed(panel)
            rows = run_replication_trajectory(
                provider,
                graph_for_panel(panel),
                seed,
                int(panel["sweeps"]),
                str(panel["regime"]),
                float(panel["coupling_strength"]),
                float(panel["sampling_temperature"]),
                str(panel["initial_condition"]),
                str(panel["disruption"]),
                panel["periods_sweeps"],  # type: ignore[arg-type]
                metadata={key: panel[key] for key in ("family", "subset", "cluster_id", "panel_id", "burn_in_sweeps")},
            )
            atomic_csv(rows, path)
            completed.append({"unit": panel["panel_id"], "rows": len(rows)})
            atomic_json(
                {
                    "updated_at": utc_now(),
                    "completed_units": len(completed),
                    "planned_units": len(formal_panel_design(protocol)) + 1,
                    "completed_rows": int(sum(int(item["rows"]) for item in completed)),
                    "planned_decisions": expected,
                    "last_unit": panel["panel_id"],
                },
                root / "supervisor_status.json",
            )
        frames = _all_formal_frames(root)
        observed = int(sum(len(frame) for frame in frames))
        if observed != expected:
            raise RuntimeError("formal rows %d differ from frozen %d" % (observed, expected))
        calls = int(sum(frame["model_calls"].sum() for frame in frames))
        prompt_tokens = int(sum(frame["prompt_tokens"].sum() for frame in frames))
        generated_tokens = int(sum(frame["generated_tokens"].sum() for frame in frames))
        latency = float(sum(frame["latency_seconds"].sum() for frame in frames))
        invalid = int(sum((frame["valid_after_repair"] == 0).sum() for frame in frames))
        completion: Dict[str, object] = {
            "status": "complete",
            "completed_at": utc_now(),
            "protocol_sha256": sha256_file(frozen_path),
            "execution_source_sha256": current_source,
            "planned_decisions": expected,
            "observed_decision_rows": observed,
            "formal_units": len(completed),
            "dynamic_trajectories": len(completed) - 1,
            "model_calls": calls,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "invalid_after_repair": invalid,
            "generation_latency_seconds": latency,
            "generation_gpu_hours": latency / 3600.0,
            "current_invocation_wall_seconds": time.perf_counter() - started,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "schema_sha256": schema_checksum(),
            "provider_environment_current_invocation": provider.environment_manifest(),
        }
        all_resources = _formal_resources(root)
        completion.update(
            {
                "all_formal_attempted_decisions_including_invalidated": int(all_resources["all_attempted_decisions"]),
                "all_formal_model_calls_including_invalidated": int(all_resources["all_model_calls"]),
                "all_formal_prompt_tokens_including_invalidated": int(all_resources["all_prompt_tokens"]),
                "all_formal_generated_tokens_including_invalidated": int(all_resources["all_generated_tokens"]),
                "all_formal_generation_latency_seconds_including_invalidated": float(all_resources["all_latency_seconds"]),
                "all_formal_generation_gpu_hours_including_invalidated": float(all_resources["all_latency_seconds"] / 3600.0),
            }
        )
        atomic_json(completion, root / "completion.json")
        atomic_json({**completion, "last_unit": "complete"}, root / "supervisor_status.json")
    return completion
