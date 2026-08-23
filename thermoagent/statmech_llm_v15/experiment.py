"""Prospective V15 design, engineering pilot, freeze, and resumable execution."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from thermoagent.statmech_llm_v12.core import LatentMapping
from thermoagent.statmech_llm_v13.simulation import build_reciprocal_graph

from .provider import MODEL_SPECS, TransformersStatmechProvider, schema_checksum
from .simulation import CONDITIONS, condition_specification, memory_control_tape, run_v15_trajectory
from .workflow import (
    PARENT_COMMIT,
    artifact_root,
    atomic_bytes,
    atomic_csv,
    atomic_json,
    ensure_external_layout,
    execution_source_checksum,
    load_yaml,
    sha256_file,
    sha256_json,
    stage_lock,
    utc_now,
)


def formal_panel_design(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    clusters = int(protocol["network"]["clusters_per_model"])  # type: ignore[index]
    for model_key in ("qwen", "granite"):
        namespace = int(protocol["network"]["cluster_seed_namespaces"][model_key])  # type: ignore[index]
        for cluster_index in range(clusters):
            cluster_seed = namespace + 1000 * cluster_index
            cluster_id = "V15%s_g%d" % ("Q" if model_key == "qwen" else "G", cluster_index)
            for condition in protocol["conditions"]:  # type: ignore[index]
                disruption, memory_mode = condition_specification(str(condition))
                rows.append(
                    {
                        "model_key": model_key,
                        "model_id": protocol["models"][model_key]["identifier"],  # type: ignore[index]
                        "model_revision": protocol["models"][model_key]["revision"],  # type: ignore[index]
                        "cluster_index": cluster_index,
                        "cluster_id": cluster_id,
                        "panel_id": "%s_%s" % (cluster_id, condition),
                        "condition": str(condition),
                        "disruption": disruption,
                        "memory_mode": memory_mode,
                        "n_agents": int(protocol["network"]["n_agents"]),  # type: ignore[index]
                        "topology": str(protocol["network"]["topology"]),  # type: ignore[index]
                        "coupling_strength": float(protocol["network"]["coupling_strength"]),  # type: ignore[index]
                        "sampling_temperature": float(protocol["inference"]["decoding_temperature"]),  # type: ignore[index]
                        "sweeps": int(protocol["trajectory"]["sweeps"]),  # type: ignore[index]
                        "periods_sweeps": list(protocol["trajectory"]["periods_sweeps"]),  # type: ignore[index]
                        "panel_seed": cluster_seed + 17,
                        "graph_seed": cluster_seed + 101,
                        "control_seed": cluster_seed + 501,
                    }
                )
    return rows


def graph_for_panel(panel: Mapping[str, object]):
    return build_reciprocal_graph(
        int(panel["n_agents"]), str(panel["topology"]), int(panel["graph_seed"])
    )


def expected_decisions(protocol: Mapping[str, object], model_key: Optional[str] = None) -> int:
    selected = [
        panel
        for panel in formal_panel_design(protocol)
        if model_key is None or panel["model_key"] == model_key
    ]
    return int(sum(int(panel["n_agents"]) * int(panel["sweeps"]) for panel in selected))


def _provider(repository: Path, model_key: str, stage: str) -> TransformersStatmechProvider:
    specification = MODEL_SPECS[str(model_key)]
    return TransformersStatmechProvider(
        specification,
        artifact_root() / "raw" / stage / str(model_key),
        repository,
        inference_temperature=0.5,
        top_p=0.9,
        maximum_new_tokens=96,
    )


def _record_pilot_failure(model_key: str, provider: TransformersStatmechProvider, error: Exception) -> None:
    """Retain infrastructure and invalid engineering attempts without raw prompts."""

    destination = artifact_root() / "pilot" / (str(model_key) + "_failures.json")
    failures: List[Dict[str, object]] = []
    if destination.exists():
        value = json.loads(destination.read_text(encoding="utf-8"))
        if isinstance(value, list):
            failures = list(value)
    message = str(error)[:800]
    classification = (
        "missing_optional_huggingface_transfer_backend"
        if "hf_transfer" in message
        else "engineering_pilot_exception"
    )
    failures.append(
        {
            "attempt": len(failures) + 1,
            "model_key": str(model_key),
            "stage": "engineering_pilot",
            "classification": classification,
            "exception_type": type(error).__name__,
            "message": message,
            "decision_requests": int(provider.accounting["decision_requests"]),
            "model_calls": int(provider.accounting["model_calls"]),
            "prompt_tokens": int(provider.accounting["prompt_tokens"]),
            "generated_tokens": int(provider.accounting["generated_tokens"]),
            "latency_seconds": float(provider.accounting["latency_seconds"]),
            "scientific_contrasts_inspected": False,
            "recorded_at": utc_now(),
        }
    )
    atomic_json(failures, destination)


def run_engineering_pilot(repository: Path, model_key: str) -> Dict[str, object]:
    repository = Path(repository).resolve()
    if model_key not in MODEL_SPECS:
        raise ValueError("unknown V15 model key")
    ensure_external_layout()
    destination = artifact_root() / "pilot" / (str(model_key) + "_summary.json")
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    protocol = load_yaml(repository / "configs/statmech_v15/protocol_template.yaml")
    n_agents = int(protocol["network"]["n_agents"])  # type: ignore[index]
    decisions = int(protocol["engineering_pilot"]["decisions_per_model"])  # type: ignore[index]
    pilot_conditions = [
        str(value) for value in protocol["engineering_pilot"]["conditions"]  # type: ignore[index]
    ]
    if decisions % (n_agents * len(pilot_conditions)):
        raise ValueError("pilot decisions must divide into whole condition-specific sweeps")
    namespace = int(protocol["network"]["cluster_seed_namespaces"][model_key])  # type: ignore[index]
    seed = namespace + 900001
    graph = build_reciprocal_graph(n_agents, "modular", seed + 101)
    provider = _provider(repository, model_key, "pilot")
    rows: List[Dict[str, object]] = []
    sweeps_per_condition = decisions // (n_agents * len(pilot_conditions))
    if sweeps_per_condition < 3:
        raise ValueError("pilot requires baseline, disruption, and recovery periods")
    baseline = max(1, sweeps_per_condition // 4)
    disruption = max(1, sweeps_per_condition // 4)
    recovery = sweeps_per_condition - baseline - disruption
    try:
        with stage_lock("pilot_%s" % model_key):
            for condition in pilot_conditions:
                rows.extend(
                    run_v15_trajectory(
                        provider,
                        graph,
                        seed,
                        sweeps_per_condition,
                        condition,
                        0.8,
                        0.5,
                        [baseline, disruption, recovery],
                        metadata={
                            "stage": "engineering_pilot",
                            "model_key": model_key,
                            "pilot_condition": condition,
                        },
                        control_seed=seed + 501,
                    )
                )
    except Exception as error:
        _record_pilot_failure(model_key, provider, error)
        provider.unload()
        raise
    frame = pd.DataFrame(rows)
    valid = float(frame["valid_after_repair"].mean())
    plus = float((frame["belief_after"] > 0).mean())
    minus_to_plus = int(np.sum((frame["belief_before"] < 0) & (frame["belief_after"] > 0)))
    plus_to_minus = int(np.sum((frame["belief_before"] > 0) & (frame["belief_after"] < 0)))
    prompt_by_condition = frame.groupby("condition")["prompt_tokens"].mean().to_dict()
    persistent_prompt = float(prompt_by_condition["field_persistent"])
    scrambled_prompt = float(prompt_by_condition["field_scrambled"])
    prompt_difference_fraction = float(
        abs(persistent_prompt - scrambled_prompt) / max(persistent_prompt, 1.0)
    )
    checks = {
        "declared_decision_count": len(frame) == decisions,
        "valid_after_repair": valid >= 0.99,
        "nondegenerate_occupancy": 0.1 <= plus <= 0.9,
        "minus_to_plus": minus_to_plus > 0,
        "plus_to_minus": plus_to_minus > 0,
        "field_schedule": set(frame["phase"]) == {"baseline", "disruption", "recovery"},
        "privacy": int(frame["unrelated_peer_private_mutations"].sum()) == 0,
        "message_delivery": int(frame["messages_delivered"].sum()) == int(frame["valid_after_repair"].sum()),
        "memory_control_exercised": int(frame["prompt_memory_entry_count"].max()) == 3,
        "memory_prompt_token_distribution_matched": prompt_difference_fraction
        <= float(
            protocol["engineering_pilot"]  # type: ignore[index]
            ["maximum_persistent_scrambled_mean_prompt_token_difference_fraction"]
        ),
    }
    summary: Dict[str, object] = {
        "stage": "engineering_pilot",
        "model_key": model_key,
        "generated_at": utc_now(),
        "inspection_boundary": protocol["engineering_pilot"]["inspection_boundary"],  # type: ignore[index]
        "scientific_contrasts_inspected": False,
        "decision_requests": int(len(frame)),
        "valid_after_repair_fraction": valid,
        "latent_plus_occupancy": plus,
        "belief_minus_to_plus": minus_to_plus,
        "belief_plus_to_minus": plus_to_minus,
        "mean_prompt_tokens": float(frame["prompt_tokens"].mean()),
        "mean_generated_tokens": float(frame["generated_tokens"].mean()),
        "mean_latency_seconds_per_decision": float(frame["latency_seconds"].mean()),
        "mean_prompt_tokens_by_condition": prompt_by_condition,
        "persistent_scrambled_mean_prompt_token_difference_fraction": prompt_difference_fraction,
        "projected_model_formal_generation_hours": float(
            frame["latency_seconds"].mean() * expected_decisions(protocol, model_key) / 3600.0
        ),
        "projected_model_formal_prompt_tokens": int(
            np.ceil(frame["prompt_tokens"].mean() * expected_decisions(protocol, model_key))
        ),
        "checks": checks,
        "engineering_passed": bool(all(checks.values())),
        "provider_environment": provider.environment_manifest(),
    }
    atomic_csv(frame, artifact_root() / "pilot" / (str(model_key) + "_trajectory.csv"))
    atomic_json(summary, destination)
    provider.unload()
    return summary


def _write_frozen_manifests(repository: Path, protocol: Mapping[str, object]) -> Dict[str, str]:
    panels = formal_panel_design(protocol)
    seed_rows = [
        {
            key: panel[key]
            for key in (
                "model_key",
                "cluster_id",
                "panel_id",
                "condition",
                "panel_seed",
                "graph_seed",
                "control_seed",
            )
        }
        for panel in panels
    ]
    seed_path = repository / "configs/statmech_v15/seed_manifest.csv"
    atomic_csv(seed_rows, seed_path)
    control_rows: List[Dict[str, object]] = []
    seen = set()
    for panel in panels:
        key = (str(panel["model_key"]), str(panel["cluster_id"]))
        if key in seen:
            continue
        seen.add(key)
        mapping = LatentMapping.balanced(int(panel["panel_seed"]) + 17011)
        tape = memory_control_tape(
            int(panel["n_agents"]),
            int(panel["n_agents"]) * int(panel["sweeps"]),
            int(panel["panel_seed"]),
            int(panel["control_seed"]),
            mapping,
        )
        control_rows.append(
            {
                "model_key": panel["model_key"],
                "cluster_id": panel["cluster_id"],
                "panel_seed": panel["panel_seed"],
                "control_seed": panel["control_seed"],
                "updates": len(tape),
                "tape_sha256": sha256_json(tape),
                "future_information": False,
                "peer_private_state": False,
            }
        )
    control_path = repository / "configs/statmech_v15/memory_control_manifest.json"
    atomic_json(
        {
            "algorithm": "own-agent past opportunity timestamps plus deterministic state placebo",
            "entries": control_rows,
            "aggregate_sha256": sha256_json(control_rows),
        },
        control_path,
    )
    return {
        "seed_manifest_sha256": sha256_file(seed_path),
        "memory_control_manifest_sha256": sha256_file(control_path),
    }


def freeze_protocol(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    frozen_path = repository / "configs/statmech_v15/protocol_frozen.yaml"
    if frozen_path.exists():
        value = load_yaml(frozen_path)
        return {
            "status": "already_frozen",
            "protocol_sha256": sha256_file(frozen_path),
            "execution_source_sha256": value["provenance"]["execution_source_sha256"],  # type: ignore[index]
        }
    pilots = {
        key: json.loads(
            (artifact_root() / "pilot" / (key + "_summary.json")).read_text(encoding="utf-8")
        )
        for key in ("qwen", "granite")
    }
    if not all(bool(value.get("engineering_passed")) for value in pilots.values()):
        raise RuntimeError("both model engineering pilots must pass before freeze")
    protocol = load_yaml(repository / "configs/statmech_v15/protocol_template.yaml")
    projected_hours = float(
        sum(float(value["projected_model_formal_generation_hours"]) for value in pilots.values())
    )
    projected_tokens = int(
        sum(int(value["projected_model_formal_prompt_tokens"]) for value in pilots.values())
    )
    if projected_hours > float(protocol["compute"]["hard_generation_gpu_hours"]):  # type: ignore[index]
        raise RuntimeError("pilot projects V15 beyond the frozen GPU-hour ceiling")
    if projected_tokens > int(protocol["compute"]["maximum_prompt_tokens"]):  # type: ignore[index]
        raise RuntimeError("pilot projects V15 beyond the frozen prompt-token ceiling")
    manifests = _write_frozen_manifests(repository, protocol)
    protocol["status"] = "frozen_before_v15_formal_outcomes"
    protocol["provenance"]["execution_source_sha256"] = execution_source_checksum(repository)  # type: ignore[index]
    protocol["provenance"]["schema_sha256"] = schema_checksum()  # type: ignore[index]
    protocol["provenance"].update(manifests)  # type: ignore[union-attr]
    protocol["compute"]["projected_generation_gpu_hours"] = projected_hours  # type: ignore[index]
    protocol["compute"]["projected_prompt_tokens"] = projected_tokens  # type: ignore[index]
    protocol["engineering_pilot_results"] = pilots
    protocol["frozen_at_utc"] = utc_now()
    atomic_bytes(yaml.safe_dump(protocol, sort_keys=False).encode("utf-8"), frozen_path)
    destination = repository / "results/collective_agent_statmech_v15/protocol/protocol_frozen.yaml"
    atomic_bytes(frozen_path.read_bytes(), destination)
    summary = {
        "status": "frozen",
        "protocol_sha256": sha256_file(frozen_path),
        "execution_source_sha256": protocol["provenance"]["execution_source_sha256"],  # type: ignore[index]
        "schema_sha256": protocol["provenance"]["schema_sha256"],  # type: ignore[index]
        **manifests,
        "projected_generation_gpu_hours": projected_hours,
        "projected_prompt_tokens": projected_tokens,
    }
    atomic_json(summary, repository / "results/collective_agent_statmech_v15/protocol/freeze_summary.json")
    return summary


def _completed_accounting(panel_root: Path, model_key: str) -> Dict[str, float]:
    totals = {
        "observed_decision_rows": 0.0,
        "model_calls": 0.0,
        "prompt_tokens": 0.0,
        "generated_tokens": 0.0,
        "latency_seconds": 0.0,
        "invalid_after_repair": 0.0,
    }
    for path in sorted(panel_root.glob("%s_*.csv" % ("V15Q" if model_key == "qwen" else "V15G"))):
        frame = pd.read_csv(path)
        totals["observed_decision_rows"] += len(frame)
        totals["model_calls"] += float(frame["model_calls"].sum())
        totals["prompt_tokens"] += float(frame["prompt_tokens"].sum())
        totals["generated_tokens"] += float(frame["generated_tokens"].sum())
        totals["latency_seconds"] += float(frame["latency_seconds"].sum())
        totals["invalid_after_repair"] += float(np.sum(frame["valid_after_repair"] == 0))
    return totals


def _assert_next_panel_within_compute_budget(
    panel_root: Path,
    panel: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    """Refuse a new atomic panel when its pilot-based projection crosses a ceiling."""

    current = {
        key: sum(_completed_accounting(panel_root, model)[key] for model in ("qwen", "granite"))
        for key in ("observed_decision_rows", "prompt_tokens", "latency_seconds")
    }
    model_key = str(panel["model_key"])
    pilot = protocol["engineering_pilot_results"][model_key]  # type: ignore[index]
    panel_decisions = int(panel["n_agents"]) * int(panel["sweeps"])
    projected_prompt_tokens = float(pilot["mean_prompt_tokens"]) * panel_decisions  # type: ignore[index]
    projected_latency_seconds = (
        float(pilot["mean_latency_seconds_per_decision"]) * panel_decisions  # type: ignore[index]
    )
    if (
        current["observed_decision_rows"] + panel_decisions
        > float(protocol["compute"]["maximum_total_decisions"])  # type: ignore[index]
    ):
        raise RuntimeError("V15 decision ceiling would be crossed by the next atomic trajectory")
    if (
        current["prompt_tokens"] + projected_prompt_tokens
        > float(protocol["compute"]["maximum_prompt_tokens"])  # type: ignore[index]
    ):
        raise RuntimeError("V15 prompt-token ceiling would be crossed by the next atomic trajectory")
    if (
        current["latency_seconds"] + projected_latency_seconds
        > 3600.0 * float(protocol["compute"]["hard_generation_gpu_hours"])  # type: ignore[index]
    ):
        raise RuntimeError("V15 GPU-hour ceiling would be crossed by the next atomic trajectory")


def _validate_existing_panel(path: Path, panel: Mapping[str, object], protocol_sha: str) -> None:
    frame = pd.read_csv(path)
    expected = int(panel["n_agents"]) * int(panel["sweeps"])
    if len(frame) != expected:
        raise RuntimeError("existing V15 panel has an invalid row count: %s" % path)
    for key in ("panel_id", "cluster_id", "condition", "model_key"):
        if set(frame[key].astype(str)) != {str(panel[key])}:
            raise RuntimeError("existing V15 panel metadata mismatch: %s" % path)
    if "protocol_sha256" not in frame or set(frame["protocol_sha256"].astype(str)) != {protocol_sha}:
        raise RuntimeError("existing V15 panel protocol mismatch: %s" % path)


def run_formal_model(repository: Path, model_key: str) -> Dict[str, object]:
    repository = Path(repository).resolve()
    ensure_external_layout()
    protocol_path = repository / "configs/statmech_v15/protocol_frozen.yaml"
    protocol = load_yaml(protocol_path)
    source = execution_source_checksum(repository)
    if source != str(protocol["provenance"]["execution_source_sha256"]):  # type: ignore[index]
        raise RuntimeError("V15 execution source differs from the frozen checksum")
    protocol_sha = sha256_file(protocol_path)
    panels = [panel for panel in formal_panel_design(protocol) if panel["model_key"] == model_key]
    panel_root = artifact_root() / "formal/panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    provider = _provider(repository, model_key, "formal")
    started = time.perf_counter()
    with stage_lock("formal_%s" % model_key):
        for panel in panels:
            destination = panel_root / (str(panel["panel_id"]) + ".csv")
            if destination.exists():
                _validate_existing_panel(destination, panel, protocol_sha)
                continue
            _assert_next_panel_within_compute_budget(panel_root, panel, protocol)
            metadata = {
                "stage": "formal",
                "model_key": model_key,
                "model_id": panel["model_id"],
                "model_revision": panel["model_revision"],
                "cluster_id": panel["cluster_id"],
                "panel_id": panel["panel_id"],
                "protocol_sha256": protocol_sha,
                "execution_source_sha256": source,
            }
            rows = run_v15_trajectory(
                provider,
                graph_for_panel(panel),
                int(panel["panel_seed"]),
                int(panel["sweeps"]),
                str(panel["condition"]),
                float(panel["coupling_strength"]),
                float(panel["sampling_temperature"]),
                panel["periods_sweeps"],
                metadata=metadata,
                mapping_override=LatentMapping.balanced(int(panel["panel_seed"]) + 17011),
                control_seed=int(panel["control_seed"]),
            )
            frame = pd.DataFrame(rows)
            if len(frame) != int(panel["n_agents"]) * int(panel["sweeps"]):
                raise RuntimeError("new V15 panel row mismatch")
            if int(frame["unrelated_peer_private_mutations"].sum()) != 0:
                raise RuntimeError("V15 privacy boundary failed")
            atomic_csv(frame, destination)
            atomic_json(
                {
                    "panel_id": panel["panel_id"],
                    "rows": len(frame),
                    "sha256": sha256_file(destination),
                    "completed_at": utc_now(),
                },
                artifact_root() / "formal" / (str(panel["panel_id"]) + "_manifest.json"),
            )
    accounting = _completed_accounting(panel_root, model_key)
    invalid_fraction = float(
        accounting["invalid_after_repair"] / max(accounting["observed_decision_rows"], 1.0)
    )
    if invalid_fraction > float(protocol["compute"]["maximum_invalid_after_repair_fraction"]):  # type: ignore[index]
        raise RuntimeError("systematic invalid V15 output exceeds the frozen one-percent ceiling")
    completion: Dict[str, object] = {
        "status": "complete"
        if int(accounting["observed_decision_rows"]) == expected_decisions(protocol, model_key)
        else "partial",
        "model_key": model_key,
        "model_id": MODEL_SPECS[model_key].identifier,
        "model_revision": MODEL_SPECS[model_key].revision,
        "protocol_sha256": protocol_sha,
        "execution_source_sha256": source,
        "planned_trajectories": len(panels),
        "completed_trajectories": int(
            sum((panel_root / (str(panel["panel_id"]) + ".csv")).exists() for panel in panels)
        ),
        "planned_decisions": expected_decisions(protocol, model_key),
        **{key: int(value) if key != "latency_seconds" else float(value) for key, value in accounting.items()},
        "generation_gpu_hours": float(accounting["latency_seconds"] / 3600.0),
        "current_invocation_wall_seconds": float(time.perf_counter() - started),
        "invalid_after_repair_fraction": invalid_fraction,
        "provider_environment_current_invocation": provider.environment_manifest(),
        "completed_at": utc_now(),
    }
    atomic_json(completion, artifact_root() / "formal" / ("completion_%s.json" % model_key))
    provider.unload()
    completion_paths = [artifact_root() / "formal" / ("completion_%s.json" % key) for key in ("qwen", "granite")]
    if all(path.exists() for path in completion_paths):
        values = [json.loads(path.read_text(encoding="utf-8")) for path in completion_paths]
        if all(value["status"] == "complete" for value in values):
            atomic_json(
                {
                    "status": "complete",
                    "models": values,
                    "dynamic_trajectories": int(sum(value["completed_trajectories"] for value in values)),
                    "observed_decision_rows": int(sum(value["observed_decision_rows"] for value in values)),
                    "model_calls": int(sum(value["model_calls"] for value in values)),
                    "prompt_tokens": int(sum(value["prompt_tokens"] for value in values)),
                    "generated_tokens": int(sum(value["generated_tokens"] for value in values)),
                    "generation_gpu_hours": float(sum(value["generation_gpu_hours"] for value in values)),
                    "completed_at": utc_now(),
                    "protocol_sha256": protocol_sha,
                    "execution_source_sha256": source,
                },
                artifact_root() / "formal/completion.json",
            )
    return completion


__all__ = [
    "expected_decisions",
    "formal_panel_design",
    "freeze_protocol",
    "graph_for_panel",
    "run_engineering_pilot",
    "run_formal_model",
]
