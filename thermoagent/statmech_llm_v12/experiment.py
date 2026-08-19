"""Prospective pilot, formal panel design, and resumable Qwen execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import (
    AgentDecision,
    IndependentStatmechAgent,
    LatentMapping,
    SignalPacket,
    StructuredProvider,
    build_agent_prompt,
)
from .graphs import build_delivery_graph
from .provider import InvalidStructuredDecision, QwenStatmechProvider
from .simulation import (
    DecentralizedStatmechNetwork,
    generate_update_tape,
    make_agents,
    run_trajectory,
)
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
    if os.environ.get("THERMO_V12_ENABLE_QWEN") != "1":
        raise RuntimeError("V12 Qwen execution is locked to the existing authorized RunPod")


def _provider(repository: Path, stage: str, settings: Mapping[str, object]) -> QwenStatmechProvider:
    return QwenStatmechProvider(
        artifact_root() / "raw" / stage,
        repository,
        inference_temperature=float(settings["inference_sampling_temperature"]),
        top_p=float(settings["top_p"]),
        maximum_new_tokens=int(settings["maximum_new_tokens"]),
    )


def pilot_summary(rows: Sequence[Mapping[str, object]], provider: QwenStatmechProvider) -> Dict[str, object]:
    frame = pd.DataFrame(rows)
    valid = frame[frame["valid_after_repair"] == 1]
    belief_minus_plus = int(np.sum((valid["belief_before"] == -1) & (valid["belief_after"] == 1)))
    belief_plus_minus = int(np.sum((valid["belief_before"] == 1) & (valid["belief_after"] == -1)))
    action_minus_plus = int(np.sum((valid["action_before"] == -1) & (valid["action_after"] == 1)))
    action_plus_minus = int(np.sum((valid["action_before"] == 1) & (valid["action_after"] == -1)))
    total = max(len(frame), 1)
    output: Dict[str, object] = {
        "generated_at": utc_now(),
        "inspection_boundary": "validity_occupancy_transition_counts_runtime_only",
        "decision_requests": int(len(frame)),
        "valid_after_repair": int(len(valid)),
        "first_pass_validity": float(frame["first_pass_valid"].sum() / total),
        "after_repair_validity": float(frame["valid_after_repair"].sum() / total),
        "latent_plus_belief_occupancy": float(np.mean(valid["belief_after"] == 1)),
        "latent_plus_action_occupancy": float(np.mean(valid["action_after"] == 1)),
        "belief_minus_to_plus": belief_minus_plus,
        "belief_plus_to_minus": belief_plus_minus,
        "action_minus_to_plus": action_minus_plus,
        "action_plus_to_minus": action_plus_minus,
        "delivered_messages": int(frame["messages_delivered"].sum()),
        "privacy_mutations": int(frame["unrelated_peer_private_mutations"].sum()),
        "prompt_tokens": int(provider.accounting["prompt_tokens"]),
        "generated_tokens": int(provider.accounting["generated_tokens"]),
        "latency_seconds": float(provider.accounting["latency_seconds"]),
        "provider_environment": provider.environment_manifest(),
        "prohibited_metrics_not_computed": [
            "entropy_production",
            "time_reversal_divergence",
            "nonreciprocity_effect",
            "probability_currents",
        ],
    }
    return output


def evaluate_pilot_targets(summary: Mapping[str, object], targets: Mapping[str, object]) -> Dict[str, object]:
    lower, upper = [float(value) for value in targets["latent_plus_occupancy_interval"]]  # type: ignore[index]
    checks = {
        "first_pass_validity": float(summary["first_pass_validity"]) >= float(targets["minimum_first_pass_validity"]),
        "after_repair_validity": float(summary["after_repair_validity"]) >= float(targets["minimum_after_repair_validity"]),
        "belief_occupancy": lower <= float(summary["latent_plus_belief_occupancy"]) <= upper,
        "action_occupancy": lower <= float(summary["latent_plus_action_occupancy"]) <= upper,
        "belief_minus_to_plus": int(summary["belief_minus_to_plus"]) >= int(targets["minimum_each_belief_transition_direction"]),
        "belief_plus_to_minus": int(summary["belief_plus_to_minus"]) >= int(targets["minimum_each_belief_transition_direction"]),
        "action_minus_to_plus": int(summary["action_minus_to_plus"]) >= int(targets["minimum_each_action_transition_direction"]),
        "action_plus_to_minus": int(summary["action_plus_to_minus"]) >= int(targets["minimum_each_action_transition_direction"]),
        "privacy": int(summary["privacy_mutations"]) <= int(targets["maximum_privacy_mutations"]),
        "delivery": int(summary["delivered_messages"]) >= int(targets["minimum_delivered_messages"]),
    }
    return {"checks": checks, "estimability_passed": bool(all(checks.values()))}


def run_engineering_pilot(repository: Path) -> Dict[str, object]:
    _require_qwen_opt_in()
    repository = Path(repository).resolve()
    config = load_yaml(repository / "configs/statmech_v12/engineering.yaml")
    model = config["model"]  # type: ignore[index]
    settings = config["pilot"]  # type: ignore[index]
    attempt_id = str(settings.get("attempt_id", "pilot_attempt1"))
    root = artifact_root() / "pilot" / attempt_id
    summary_path = root / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    provider = _provider(repository, attempt_id, model)
    rows: List[Dict[str, object]] = []
    with stage_lock("pilot"):
        for panel_index in range(int(settings["panels"])):
            seed = int(settings["panel_seed_base"]) + panel_index
            graph = build_delivery_graph(
                int(settings["n_agents"]),
                str(settings["topology"]),
                seed,
                seed + 31,
                0.0,
            )
            plus_label = "amber" if panel_index % 2 == 0 else "cobalt"
            display_order = ("amber", "cobalt") if (panel_index // 2) % 2 == 0 else ("cobalt", "amber")
            panel_rows = run_trajectory(
                provider,
                graph,
                seed,
                int(settings["sweeps_per_panel"]),
                str(settings["regime"]),
                float(settings["coupling_strength"]),
                float(model["inference_sampling_temperature"]),
                str(settings["initial_condition"]),
                metadata={"stage": "pilot", "panel_id": "pilot_%02d" % panel_index},
                mapping_override=LatentMapping(plus_label, display_order),
            )
            rows.extend(panel_rows)
            atomic_json(
                {"stage": "pilot", "completed_panels": panel_index + 1, "decision_rows": len(rows), "updated": utc_now()},
                artifact_root() / "supervisor_status.json",
            )
        atomic_csv(rows, root / "transitions.csv")
        summary = pilot_summary(rows, provider)
        summary["attempt_id"] = attempt_id
        summary["estimability"] = evaluate_pilot_targets(summary, settings["estimability_targets"])
        atomic_json(summary, summary_path)
    return summary


def _alpha_arms(levels: Sequence[float]) -> List[Tuple[float, str, bool]]:
    output: List[Tuple[float, str, bool]] = []
    for alpha in [float(value) for value in levels]:
        if np.isclose(alpha, 0.0):
            output.append((0.0, "reciprocal", False))
        else:
            output.append((alpha, "forward", False))
            output.append((alpha, "transpose", True))
    return output


def formal_panel_design(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    network = protocol["network"]  # type: ignore[index]
    levels = network["nonreciprocity_levels"]  # type: ignore[index]
    arms = _alpha_arms(levels)
    output: List[Dict[str, object]] = []
    small = protocol["small_network"]  # type: ignore[index]
    for n_agents in small["agent_counts"]:  # type: ignore[index]
        for replicate in range(int(small["graph_environment_replicates"])):
            cluster = "small_n%d_g%d" % (int(n_agents), replicate)
            for alpha, orientation, reverse in arms:
                output.append(
                    {
                        "family": "small_network",
                        "cluster_id": cluster,
                        "panel_id": "%s_a%.2f_%s" % (cluster, alpha, orientation),
                        "n_agents": int(n_agents),
                        "topology": str(small["topology"]),
                        "replicate": replicate,
                        "alpha": alpha,
                        "orientation": orientation,
                        "reverse_orientation": reverse,
                        "sweeps": int(small["sweeps"]),
                        "burn_in_sweeps": int(small["burn_in_sweeps"]),
                        "coupling_strength": float(small["coupling_strength"]),
                        "sampling_temperature": float(small["inference_sampling_temperature"]),
                        "regime": str(small["regime"]),
                        "initial_condition": str(small["initial_condition"]),
                        "control": "unaltered",
                    }
                )
    collective = protocol["collective_network"]  # type: ignore[index]
    for n_agents in collective["agent_counts"]:  # type: ignore[index]
        for topology in collective["topologies"]:  # type: ignore[index]
            for coupling in collective["coupling_strengths"]:  # type: ignore[index]
                for temperature in collective["inference_sampling_temperatures"]:  # type: ignore[index]
                    for replicate in range(int(collective["graph_environment_replicates"])):
                        cluster = "collective_n%d_%s_k%.2f_t%.2f_g%d" % (
                            int(n_agents), topology, float(coupling), float(temperature), replicate
                        )
                        for alpha, orientation, reverse in arms:
                            output.append(
                                {
                                    "family": "collective_network",
                                    "cluster_id": cluster,
                                    "panel_id": "%s_a%.2f_%s" % (cluster, alpha, orientation),
                                    "n_agents": int(n_agents),
                                    "topology": str(topology),
                                    "replicate": replicate,
                                    "alpha": alpha,
                                    "orientation": orientation,
                                    "reverse_orientation": reverse,
                                    "sweeps": int(collective["sweeps"]),
                                    "burn_in_sweeps": int(collective["burn_in_sweeps"]),
                                    "coupling_strength": float(coupling),
                                    "sampling_temperature": float(temperature),
                                    "regime": str(collective["regime"]),
                                    "initial_condition": str(collective["initial_condition"]),
                                    "control": "unaltered",
                                }
                            )
    memory = protocol["persistent_memory"]  # type: ignore[index]
    memory_arms = _alpha_arms(memory["nonreciprocity_levels"])  # type: ignore[index]
    for n_agents in memory["agent_counts"]:  # type: ignore[index]
        for topology in memory["topologies"]:  # type: ignore[index]
            for replicate in range(int(memory["graph_environment_replicates"])):
                cluster = "memory_n%d_%s_g%d" % (int(n_agents), topology, replicate)
                for regime in memory["regimes"]:  # type: ignore[index]
                    for alpha, orientation, reverse in memory_arms:
                        output.append(
                            {
                                "family": "persistent_memory",
                                "cluster_id": cluster,
                                "panel_id": "%s_%s_a%.2f_%s" % (cluster, regime, alpha, orientation),
                                "n_agents": int(n_agents),
                                "topology": str(topology),
                                "replicate": replicate,
                                "alpha": alpha,
                                "orientation": orientation,
                                "reverse_orientation": reverse,
                                "sweeps": int(memory["sweeps"]),
                                "burn_in_sweeps": int(memory["burn_in_sweeps"]),
                                "coupling_strength": float(memory["coupling_strength"]),
                                "sampling_temperature": float(memory["inference_sampling_temperature"]),
                                "regime": str(regime),
                                "initial_condition": "disordered",
                                "control": "unaltered",
                            }
                        )
    relaxation = protocol["relaxation"]  # type: ignore[index]
    relaxation_arms = _alpha_arms(relaxation["nonreciprocity_levels"])  # type: ignore[index]
    for n_agents in relaxation["agent_counts"]:  # type: ignore[index]
        for temperature in relaxation["inference_sampling_temperatures"]:  # type: ignore[index]
            for initial in relaxation["initial_conditions"]:  # type: ignore[index]
                for replicate in range(int(relaxation["graph_environment_replicates"])):
                    cluster = "relax_n%d_t%.2f_%s_g%d" % (int(n_agents), float(temperature), initial, replicate)
                    for alpha, orientation, reverse in relaxation_arms:
                        output.append(
                            {
                                "family": "relaxation",
                                "cluster_id": cluster,
                                "panel_id": "%s_a%.2f_%s" % (cluster, alpha, orientation),
                                "n_agents": int(n_agents),
                                "topology": str(relaxation["topology"]),
                                "replicate": replicate,
                                "alpha": alpha,
                                "orientation": orientation,
                                "reverse_orientation": reverse,
                                "sweeps": int(relaxation["sweeps"]),
                                "burn_in_sweeps": 0,
                                "coupling_strength": 0.70,
                                "sampling_temperature": float(temperature),
                                "regime": "markovized",
                                "initial_condition": str(initial),
                                "control": "unaltered",
                            }
                        )
    controls = protocol["controls"]  # type: ignore[index]
    for replicate in range(int(controls["graph_environment_replicates"])):
        cluster = "control_g%d" % replicate
        for control in controls["arms"]:  # type: ignore[index]
            output.append(
                {
                    "family": "controls",
                    "cluster_id": cluster,
                    "panel_id": "%s_%s" % (cluster, control),
                    "n_agents": int(controls["n_agents"]),
                    "topology": str(controls["topology"]),
                    "replicate": replicate,
                    "alpha": float(controls["alpha"]),
                    "orientation": str(controls["orientation"]),
                    "reverse_orientation": False,
                    "sweeps": int(controls["sweeps"]),
                    "burn_in_sweeps": 1,
                    "coupling_strength": 0.70,
                    "sampling_temperature": float(controls["inference_sampling_temperature"]),
                    "regime": "markovized",
                    "initial_condition": "disordered",
                    "control": str(control),
                }
            )
    return output


def _panel_seed(panel: Mapping[str, object]) -> int:
    # Orientation and alpha deliberately do not enter: matched arms share the
    # same update, display-order, inference-seed, and recipient-uniform tapes.
    family_offsets = {
        "small_network": 100000,
        "collective_network": 300000,
        "persistent_memory": 600000,
        "relaxation": 700000,
        "controls": 800000,
    }
    family = str(panel["family"])
    token = "%s|%s" % (family, panel["cluster_id"])
    stable = int.from_bytes(token.encode("utf-8"), "little") % 100000
    return 12012000 + family_offsets[family] + stable


def _graph_for_panel(panel: Mapping[str, object]) -> object:
    seed = _panel_seed(panel)
    return build_delivery_graph(
        int(panel["n_agents"]),
        str(panel["topology"]),
        seed + 17,
        seed + 37,
        float(panel["alpha"]),
        bool(panel["reverse_orientation"]),
    )


def _micro_response_rows(provider: StructuredProvider, protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    settings = protocol["microscopic_response"]  # type: ignore[index]
    rows: List[Dict[str, object]] = []
    cell_index = 0
    for private in settings["private_fields"]:  # type: ignore[index]
        for neighbor in settings["neighbor_fields"]:  # type: ignore[index]
            for belief in settings["current_beliefs"]:  # type: ignore[index]
                for action in settings["current_actions"]:  # type: ignore[index]
                    for coupling in settings["coupling_strengths"]:  # type: ignore[index]
                        for temperature in settings["inference_sampling_temperatures"]:  # type: ignore[index]
                            for regime in settings["regimes"]:  # type: ignore[index]
                                for replicate in range(int(settings["independent_replicates_per_cell"])):
                                    seed = 12200000 + cell_index * 10 + replicate
                                    plus = "amber" if cell_index % 2 == 0 else "cobalt"
                                    order = ("amber", "cobalt") if (cell_index // 2) % 2 == 0 else ("cobalt", "amber")
                                    mapping = LatentMapping(plus, order)
                                    agent = IndependentStatmechAgent(
                                        0,
                                        "isolated_response_agent",
                                        int(private),
                                        int(belief),
                                        int(action),
                                        _memory_history=["bounded prior local state"] if regime == "persistent_memory" else [],
                                    )
                                    signals = [int(neighbor), int(neighbor)] if int(neighbor) else [-1, 1]
                                    for sender, signal in enumerate(signals, start=1):
                                        agent.receive(SignalPacket(sender, 0, signal, signal, signal, 0.7, 1, 0))
                                    prompt = build_agent_prompt(
                                        agent, mapping, 0, str(regime), float(coupling), cell_index % 2
                                    )
                                    valid = 1
                                    try:
                                        result = provider.decide(prompt, seed, float(temperature))
                                        decision = AgentDecision.from_mapping(result.payload)
                                    except InvalidStructuredDecision as error:
                                        valid = 0
                                        result = error.result
                                        decision = None
                                    rows.append(
                                        {
                                            "cell_id": "micro_%04d" % cell_index,
                                            "information_state_id": "p%d_n%d_b%d_a%d_k%.2f_t%.2f_%s"
                                            % (
                                                int(private),
                                                int(neighbor),
                                                int(belief),
                                                int(action),
                                                float(coupling),
                                                float(temperature),
                                                str(regime),
                                            ),
                                            "replicate": replicate,
                                            "private_field": int(private),
                                            "neighbor_field": int(neighbor),
                                            "current_belief": int(belief),
                                            "current_action": int(action),
                                            "coupling_strength": float(coupling),
                                            "sampling_temperature": float(temperature),
                                            "regime": str(regime),
                                            "latent_plus_label": plus,
                                            "amber_first": int(order[0] == "amber"),
                                            "paraphrase": cell_index % 2,
                                            "belief_after": 0 if decision is None else mapping.spin(decision.belief_choice),
                                            "action_after": 0
                                            if decision is None
                                            else mapping.spin(decision.action_choice),
                                            "confidence": float("nan") if decision is None else float(decision.confidence),
                                            "valid_after_repair": valid,
                                            "first_pass_valid": 0 if result is None else int(result.first_pass_valid),
                                            "repaired": 0 if result is None else int(result.repaired),
                                            "repair_attempted": 0 if result is None else int(not result.first_pass_valid),
                                            "model_calls": 0 if result is None else int(1 + (not result.first_pass_valid)),
                                            "prompt_tokens": 0 if result is None else int(result.prompt_tokens),
                                            "generated_tokens": 0 if result is None else int(result.generated_tokens),
                                            "latency_seconds": 0.0 if result is None else float(result.latency_seconds),
                                            "raw_artifact_sha256": "" if result is None else result.raw_artifact_sha256,
                                        }
                                    )
                                    cell_index += 1
    return rows


def _hysteresis_panel_rows(
    provider: StructuredProvider,
    settings: Mapping[str, object],
    replicate: int,
    alpha: float,
    orientation: str,
    reverse: bool,
) -> Tuple[str, List[Dict[str, object]]]:
    """Generate one deterministic hysteresis panel from recorded local decisions."""

    cluster = "hysteresis_g%d" % int(replicate)
    seed = 12900000 + int(replicate)
    panel_id = "%s_a%.2f_%s" % (cluster, float(alpha), str(orientation))
    graph = build_delivery_graph(
        int(settings["n_agents"]),
        str(settings["topology"]),
        seed + 17,
        seed + 37,
        float(alpha),
        bool(reverse),
    )
    mapping = LatentMapping("amber" if int(replicate) % 2 == 0 else "cobalt", ("amber", "cobalt"))
    network = DecentralizedStatmechNetwork(
        make_agents(graph.n_agents, seed, "ordered"),
        graph,
        mapping,
        "markovized",
        float(settings["coupling_strength"]),
    )
    fields = [int(value) for value in settings["external_field_sweep"]]  # type: ignore[index]
    updates_per_field = int(settings["updates_per_field"])
    tape = generate_update_tape(graph.n_agents, len(fields) * updates_per_field, seed + 101)
    rows: List[Dict[str, object]] = []
    update_index = 0
    for segment, field in enumerate(fields):
        for agent in network.agents:
            agent.private_field = field
        for _ in range(updates_per_field):
            row = network.offered_update(
                provider,
                tape[update_index],
                update_index,
                float(settings["inference_sampling_temperature"]),
            )
            row.update(
                {
                    "family": "hysteresis",
                    "cluster_id": cluster,
                    "panel_id": panel_id,
                    "n_agents": graph.n_agents,
                    "topology": graph.topology,
                    "alpha": float(alpha),
                    "orientation": str(orientation),
                    "external_field": field,
                    "field_segment": segment,
                    "regime": "markovized",
                    "control": "unaltered",
                }
            )
            rows.append(row)
            update_index += 1
    return panel_id, rows


def _run_hysteresis_panels(
    provider: StructuredProvider, protocol: Mapping[str, object]
) -> List[Tuple[str, List[Dict[str, object]]]]:
    settings = protocol["hysteresis"]  # type: ignore[index]
    levels = settings["nonreciprocity_levels"]  # type: ignore[index]
    arms = _alpha_arms(levels)
    output: List[Tuple[str, List[Dict[str, object]]]] = []
    destination_root = artifact_root() / "formal" / "hysteresis"
    expected_rows = len(settings["external_field_sweep"]) * int(settings["updates_per_field"])  # type: ignore[index]
    for replicate in range(int(settings["graph_environment_replicates"])):
        for alpha, orientation, reverse in arms:
            panel_id = "hysteresis_g%d_a%.2f_%s" % (replicate, alpha, orientation)
            existing = destination_root / (panel_id + ".csv")
            if existing.exists() and int(pd.read_csv(existing).shape[0]) == expected_rows:
                output.append((panel_id, []))
                continue
            output.append(
                _hysteresis_panel_rows(provider, settings, replicate, alpha, orientation, reverse)
            )
    return output


def _retain_incomplete(path: Path, reason: str) -> None:
    """Move an invalid external panel into the retained failure registry."""

    if not path.exists():
        return
    failures = artifact_root() / "failures" / "formal_incomplete"
    failures.mkdir(parents=True, exist_ok=True)
    token = utc_now().replace(":", "-").replace("+", "_")
    destination = failures / (path.stem + "__" + reason + "__" + token + path.suffix)
    os.replace(str(path), str(destination))


def expected_decisions(protocol: Mapping[str, object]) -> int:
    dynamic = sum(int(panel["n_agents"]) * int(panel["sweeps"]) for panel in formal_panel_design(protocol))
    micro = int(protocol["microscopic_response"]["expected_decisions"])  # type: ignore[index]
    hysteresis = int(protocol["hysteresis"]["expected_decisions"])  # type: ignore[index]
    return int(dynamic + micro + hysteresis)


def run_formal_experiment(repository: Path) -> Dict[str, object]:
    _require_qwen_opt_in()
    repository = Path(repository).resolve()
    frozen_path = repository / "configs/statmech_v12/protocol_frozen.yaml"
    if not frozen_path.exists():
        raise RuntimeError("V12 protocol is not frozen")
    protocol = load_yaml(frozen_path)
    frozen_source = str(protocol["provenance"]["execution_source_sha256"])  # type: ignore[index]
    current_source = execution_source_checksum(repository)
    if frozen_source != current_source:
        raise RuntimeError("V12 execution source changed after protocol freeze")
    expected = expected_decisions(protocol)
    declared = int(protocol["compute"]["expected_primary_decisions"])  # type: ignore[index]
    if expected != declared:
        raise RuntimeError("formal design count %d differs from frozen declaration %d" % (expected, declared))
    model_settings = dict(protocol["model"])  # type: ignore[arg-type]
    model_settings["inference_sampling_temperature"] = 0.72
    provider = _provider(repository, "formal", model_settings)
    root = artifact_root() / "formal"
    panel_root = root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    completed: List[Dict[str, object]] = []
    with stage_lock("formal"):
        run_state_path = root / "run_state.json"
        if not run_state_path.exists():
            atomic_json(
                {
                    "started_at": utc_now(),
                    "planned_decisions": expected,
                    "protocol_sha256": sha256_file(frozen_path),
                    "execution_source_sha256": current_source,
                },
                run_state_path,
            )
        micro_path = root / "microscopic_response.csv"
        micro_expected = int(protocol["microscopic_response"]["expected_decisions"])  # type: ignore[index]
        if micro_path.exists() and int(pd.read_csv(micro_path).shape[0]) != micro_expected:
            _retain_incomplete(micro_path, "row_count")
        if not micro_path.exists():
            atomic_csv(_micro_response_rows(provider, protocol), micro_path)
        completed.append({"panel_id": "microscopic_response", "rows": int(pd.read_csv(micro_path).shape[0])})
        panels = formal_panel_design(protocol)
        for panel_index, panel in enumerate(panels):
            destination = panel_root / (str(panel["panel_id"]) + ".csv")
            expected_rows = int(panel["n_agents"]) * int(panel["sweeps"])
            if destination.exists() and int(pd.read_csv(destination).shape[0]) == expected_rows:
                completed.append({"panel_id": panel["panel_id"], "rows": expected_rows})
                continue
            if destination.exists():
                _retain_incomplete(destination, "row_count")
            graph = _graph_for_panel(panel)
            rows = run_trajectory(
                provider,
                graph,  # type: ignore[arg-type]
                _panel_seed(panel),
                int(panel["sweeps"]),
                str(panel["regime"]),
                float(panel["coupling_strength"]),
                float(panel["sampling_temperature"]),
                str(panel["initial_condition"]),
                control=str(panel["control"]),
                metadata={
                    "family": panel["family"],
                    "cluster_id": panel["cluster_id"],
                    "panel_id": panel["panel_id"],
                    "orientation": panel["orientation"],
                    "burn_in_sweeps": panel["burn_in_sweeps"],
                },
            )
            atomic_csv(rows, destination)
            completed.append({"panel_id": panel["panel_id"], "rows": len(rows)})
            atomic_json(
                {
                    "stage": "formal",
                    "completed_units": len(completed),
                    "planned_dynamic_panels": len(panels),
                    "last_panel": panel["panel_id"],
                    "provider_accounting_this_process": provider.accounting,
                    "updated": utc_now(),
                },
                artifact_root() / "supervisor_status.json",
            )
        hysteresis_root = root / "hysteresis"
        hysteresis_root.mkdir(parents=True, exist_ok=True)
        for panel_id, rows in _run_hysteresis_panels(provider, protocol):
            destination = hysteresis_root / (panel_id + ".csv")
            if destination.exists() and int(pd.read_csv(destination).shape[0]) != len(rows) and rows:
                _retain_incomplete(destination, "row_count")
            if not destination.exists() and rows:
                atomic_csv(rows, destination)
            completed.append({"panel_id": panel_id, "rows": int(pd.read_csv(destination).shape[0])})
        all_csv = [micro_path] + sorted(panel_root.glob("*.csv")) + sorted(hysteresis_root.glob("*.csv"))
        total_rows = int(sum(pd.read_csv(path, usecols=[0]).shape[0] for path in all_csv))
        tokens = {
            "decision_requests": 0,
            "model_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "latency_seconds": 0.0,
            "repair_attempts": 0,
            "repaired_valid": 0,
            "invalid": 0,
        }
        for path in all_csv:
            frame = pd.read_csv(path)
            tokens["decision_requests"] += int(frame.shape[0])
            tokens["model_calls"] += int(frame.get("model_calls", pd.Series(dtype=float)).fillna(0).sum())
            tokens["prompt_tokens"] += int(frame.get("prompt_tokens", pd.Series(dtype=float)).fillna(0).sum())
            tokens["generated_tokens"] += int(frame.get("generated_tokens", pd.Series(dtype=float)).fillna(0).sum())
            tokens["latency_seconds"] += float(frame.get("latency_seconds", pd.Series(dtype=float)).fillna(0).sum())
            tokens["repair_attempts"] += int(frame.get("repair_attempted", pd.Series(dtype=float)).fillna(0).sum())
            tokens["repaired_valid"] += int(frame.get("repaired", pd.Series(dtype=float)).fillna(0).sum())
            tokens["invalid"] += int((frame.get("valid_after_repair", pd.Series(dtype=float)).fillna(0) == 0).sum())
        summary: Dict[str, object] = {
            "started_at": json.loads(run_state_path.read_text(encoding="utf-8"))["started_at"],
            "completed_at": utc_now(),
            "protocol_sha256": sha256_file(frozen_path),
            "execution_source_sha256": current_source,
            "planned_decisions": expected,
            "observed_decision_rows": total_rows,
            "completed_units": len(completed),
            "dynamic_panel_count": len(panels),
            "tokens_and_latency": tokens,
            "generation_gpu_hours": float(tokens["latency_seconds"] / 3600.0),
            "generation_plus_current_model_load_gpu_hours": float(
                (tokens["latency_seconds"] + provider.accounting["model_loading_seconds"]) / 3600.0
            ),
            "provider_environment": provider.environment_manifest(),
            "status": "complete" if total_rows == expected else "incomplete",
        }
        atomic_json(summary, root / "completion.json")
        if total_rows != expected:
            raise RuntimeError("formal row accounting mismatch")
    return summary
