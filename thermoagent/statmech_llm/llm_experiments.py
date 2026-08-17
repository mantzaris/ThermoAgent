"""Controlled and dynamic Qwen-agent experiments for V10.

All response-level records are external.  The module is importable without GPU
dependencies; Qwen is loaded only after an explicit execution gate.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from thermoagent.statmech.exact import decode_state, encode_state, entropy_production_rate, stationary_distribution
from thermoagent.statmech.model import Microstate, topology_adjacency

from .agents import (
    ACTION_CHOICES,
    BELIEF_CHOICES,
    DecentralizedLLMNetwork,
    DeliveredMessage,
    IndependentLLMAgent,
    StructuredDecision,
    make_agents,
)
from .applications import (
    DefensiveUtilityMapping,
    HumanitarianCoordinationMapping,
    application_roles,
    private_evidence_text,
)
from .estimators import block_time_reversal_kl, conditional_mutual_information_markov, fit_logistic_response
from .qwen import MODEL_ID, MODEL_REVISION, QwenStructuredProvider
from .theory import directed_family
from .workflow import (
    _atomic_csv,
    _atomic_json,
    artifact_root,
    load_yaml,
    sha256_file,
    source_checksum,
    stage_lock,
    utc_now,
)


def _require_execution_opt_in() -> None:
    if os.environ.get("THERMO_V10_ENABLE_QWEN") != "1":
        raise RuntimeError(
            "Qwen execution is locked. Set THERMO_V10_ENABLE_QWEN=1 only on the existing authorized RunPod."
        )


def _fresh_agent(
    identifier: int,
    role: str,
    belief: int,
    action: int,
    observation: str,
) -> IndependentLLMAgent:
    return IndependentLLMAgent(
        identifier=int(identifier),
        role=str(role),
        private_observation=str(observation),
        _memory=["current belief and commitment are provisional and may be revised when authorized evidence changes"],
        _belief=int(belief),
        _action=int(action),
        _belief_confidence=0.5,
        _commitment="retain",
        _authorized_tools=(
            "no_tool",
            "commit_plan_left",
            "commit_plan_right",
            "request_information",
            "defer_action",
        ),
    )


def _plan_order(seed: int) -> Tuple[str, str]:
    return ("plan_left", "plan_right") if int(seed) % 2 == 0 else ("plan_right", "plan_left")


def _pilot_design(call_count: int) -> List[Tuple[float, Tuple[str, str], int, int]]:
    """Return a fully crossed evidence-by-order pilot design.

    The original pilot assigned option order from call parity while cycling six
    evidence fields.  That aliased each field with one order.  This retained
    pilot amendment crosses every field with both orders and changes neither
    the prompt nor the prespecified execution gate.
    """

    fields = (-1.0, -0.55, -0.15, 0.15, 0.55, 1.0)
    cell_count = len(fields) * 2
    if int(call_count) <= 0 or int(call_count) % cell_count != 0:
        raise ValueError(f"pilot call_count must be a positive multiple of {cell_count}")
    replicates = int(call_count) // cell_count
    design: List[Tuple[float, Tuple[str, str], int, int]] = []
    for field in fields:
        for order in (("plan_left", "plan_right"), ("plan_right", "plan_left")):
            for replicate in range(replicates):
                design.append((field, order, replicate % 3, replicate))
    return design


def _state_agents(
    state_index: int,
    n_agents: int,
    application: str,
    communication: np.ndarray,
    private_fields: np.ndarray,
    paraphrase: int,
) -> List[IndependentLLMAgent]:
    state = decode_state(int(state_index), int(n_agents))
    roles = application_roles(application, n_agents)
    agents = [
        _fresh_agent(
            index,
            roles[index],
            int(state.beliefs[index]),
            int(state.actions[index]),
            private_evidence_text(application, float(private_fields[index]), paraphrase),
        )
        for index in range(n_agents)
    ]
    for recipient in range(n_agents):
        for sender in range(n_agents):
            weight = float(communication[recipient, sender])
            if sender == recipient or weight <= 0.0:
                continue
            agents[recipient].receive(
                DeliveredMessage(
                    sender=sender,
                    recipient=recipient,
                    time_step=0,
                    outgoing_signal="support_right" if state.beliefs[sender] > 0 else "support_left",
                    outgoing_message="My local evidence and commitment currently favor %s."
                    % ("plan_right" if state.beliefs[sender] > 0 else "plan_left"),
                    influence_weight=weight,
                )
            )
    return agents


def _pilot_rows(
    provider: QwenStructuredProvider,
    call_count: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for call, (field, option_order, paraphrase, replicate) in enumerate(_pilot_design(call_count)):
        application = "humanitarian" if call % 2 == 0 else "utility"
        seed = 51000 + call
        agent = _fresh_agent(
            0,
            application_roles(application, 1)[0],
            -1 if replicate % 3 == 0 else 1,
            -1 if replicate % 4 == 0 else 1,
            private_evidence_text(application, field, paraphrase),
        )
        network = DecentralizedLLMNetwork([agent], np.zeros((1, 1)))
        decision = network.offered_update(
            0,
            provider,
            seed,
            None,
            option_order,
            paraphrase,
        )
        record = network.decision_ledger[-1]
        rows.append(
            {
                "call": call,
                "application": application,
                "local_field": field,
                "option_order_right_first": int(option_order[0] == "plan_right"),
                "paraphrase": paraphrase,
                "replicate": replicate,
                "belief_spin": decision.belief_spin,
                "action_spin": decision.action_spin,
                "belief_confidence": decision.belief_confidence,
                "tool_action": decision.tool_action,
                "outgoing_signal": decision.outgoing_signal,
                "first_pass_valid": int(record["first_pass_valid"]),
                "repaired": int(record["repaired"]),
                "prompt_tokens": int(record["prompt_tokens"]),
                "generated_tokens": int(record["generated_tokens"]),
                "latency_seconds": float(record["latency_seconds"]),
                "raw_artifact_sha256": str(record["raw_artifact_sha256"]),
            }
        )
    return rows


def run_qwen_pilot(repository: Path) -> Dict[str, object]:
    _require_execution_opt_in()
    protocol = load_yaml(repository / "configs/statmech_v10/protocol.yaml")
    settings = protocol["llm_execution_design"]
    output = artifact_root() / "qwen/pilot"
    completion = output / "summary.json"
    if completion.exists():
        return json.loads(completion.read_text(encoding="utf-8"))
    provider = QwenStructuredProvider(
        artifact_root() / "qwen/raw/pilot",
        repository,
        float(settings["inference_sampling_temperature"]),
        float(settings["top_p"]),
        int(settings["maximum_new_tokens"]),
    )
    with stage_lock("qwen_pilot"):
        rows = _pilot_rows(provider, int(settings["pilot_calls"]))
        _atomic_csv(rows, output / "decisions.csv")
        frame = np.asarray([row["belief_spin"] for row in rows], dtype=int)
        first_valid = float(np.mean([row["first_pass_valid"] for row in rows]))
        after_repair = 1.0 - provider.accounting["invalid_after_repair"] / float(len(rows))
        nontrivial = float(np.mean([row["tool_action"] not in ("no_tool", "defer_action") for row in rows]))
        positive = np.asarray([row["belief_spin"] > 0 for row in rows], dtype=float)
        fields = np.asarray([row["local_field"] for row in rows], dtype=float)
        response_direction = float(np.mean(positive[fields > 0.0]) - np.mean(positive[fields < 0.0]))
        fit = fit_logistic_response(
            fields,
            frame,
            option_order=np.asarray([1 if row["option_order_right_first"] else -1 for row in rows]),
        )
        gates = settings["execution_gate"]
        passed = bool(
            first_valid >= float(gates["minimum_first_pass_validity"])
            and after_repair >= float(gates["minimum_after_repair_validity"])
            and nontrivial >= float(gates["minimum_nontrivial_action_fraction"])
            and abs(fit["option_order_slope"]) <= float(gates["maximum_absolute_option_order_effect"])
            and response_direction > 0.20
        )
        summary = {
            "completed_at": utc_now(),
            "calls": len(rows),
            "first_pass_validity": first_valid,
            "after_repair_validity": after_repair,
            "nontrivial_tool_action_fraction": nontrivial,
            "private_evidence_response_difference": response_direction,
            "action_diversity": int(len(set(frame.tolist()))),
            "logistic_fit": fit,
            "provider_accounting": provider.accounting,
            "pilot_gate_passed": passed,
            "formal_unlocked": passed,
        }
        _atomic_json(summary, completion)
        return summary


def _message_pilot_design(pair_count: int) -> List[Tuple[str, int, Tuple[str, str], int]]:
    """Return the frozen, fully crossed matched-message pilot design."""

    applications = ("humanitarian", "utility")
    orders = (("plan_left", "plan_right"), ("plan_right", "plan_left"))
    base_cells = len(applications) * 3 * len(orders)
    if int(pair_count) <= 0 or int(pair_count) % base_cells != 0:
        raise ValueError(f"message pair_count must be a positive multiple of {base_cells}")
    replicates = int(pair_count) // base_cells
    return [
        (application, paraphrase, order, replicate)
        for application in applications
        for paraphrase in range(3)
        for order in orders
        for replicate in range(replicates)
    ]


def _message_counterfactual_rows(
    provider: QwenStructuredProvider,
    pair_count: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pair, (application, paraphrase, option_order, replicate) in enumerate(
        _message_pilot_design(pair_count)
    ):
        initial_belief = -1 if replicate == 0 else 1
        initial_action = 1 if replicate == 0 else -1
        for message_spin in (-1, 1):
            agent = _fresh_agent(
                0,
                application_roles(application, 1)[0],
                initial_belief,
                initial_action,
                "Your private evidence is balanced and inconclusive between the two plans.",
            )
            agent.receive(
                DeliveredMessage(
                    sender=1,
                    recipient=0,
                    time_step=0,
                    outgoing_signal="support_right" if message_spin > 0 else "support_left",
                    outgoing_message=(
                        "My independently observed local evidence supports plan_right."
                        if message_spin > 0
                        else "My independently observed local evidence supports plan_left."
                    ),
                    influence_weight=1.0,
                )
            )
            network = DecentralizedLLMNetwork([agent], np.zeros((1, 1)))
            seed = 515000 + pair
            decision = network.offered_update(
                0,
                provider,
                seed,
                "belief",
                option_order,
                paraphrase,
            )
            record = network.decision_ledger[-1]
            rows.append(
                {
                    "pair": pair,
                    "application": application,
                    "paraphrase": paraphrase,
                    "replicate": replicate,
                    "option_order_right_first": int(option_order[0] == "plan_right"),
                    "previous_belief": initial_belief,
                    "previous_action": initial_action,
                    "message_spin": message_spin,
                    "belief_spin": decision.belief_spin,
                    "belief_confidence": decision.belief_confidence,
                    "reason_code": decision.reason_code,
                    "first_pass_valid": int(record["first_pass_valid"]),
                    "repaired": int(record["repaired"]),
                    "prompt_tokens": int(record["prompt_tokens"]),
                    "generated_tokens": int(record["generated_tokens"]),
                    "latency_seconds": float(record["latency_seconds"]),
                    "raw_artifact_sha256": str(record["raw_artifact_sha256"]),
                }
            )
    return rows


def run_qwen_message_pilot(repository: Path) -> Dict[str, object]:
    """Exercise the prespecified delivered-message counterfactual gate."""

    _require_execution_opt_in()
    evidence_path = artifact_root() / "qwen/pilot/summary.json"
    if not evidence_path.exists() or not json.loads(evidence_path.read_text(encoding="utf-8"))["pilot_gate_passed"]:
        raise RuntimeError("message pilot remains locked until the corrected evidence pilot passes")
    amendment_path = repository / "configs/statmech_v10/llm_pilot_amendment.yaml"
    amendment = load_yaml(amendment_path)
    settings = amendment["message_counterfactual_pilot"]
    output = artifact_root() / "qwen/message_pilot"
    completion = output / "summary.json"
    if completion.exists():
        return json.loads(completion.read_text(encoding="utf-8"))
    base = load_yaml(repository / "configs/statmech_v10/protocol.yaml")["llm_execution_design"]
    provider = QwenStructuredProvider(
        artifact_root() / "qwen/raw/message_pilot",
        repository,
        float(base["inference_sampling_temperature"]),
        float(base["top_p"]),
        int(base["maximum_new_tokens"]),
    )
    with stage_lock("qwen_message_pilot"):
        rows = _message_counterfactual_rows(provider, int(settings["matched_pairs"]))
        _atomic_csv(rows, output / "decisions.csv")
        by_pair: Dict[int, Dict[int, int]] = {}
        for row in rows:
            by_pair.setdefault(int(row["pair"]), {})[int(row["message_spin"])] = int(row["belief_spin"])
        right = np.asarray([row["belief_spin"] > 0 for row in rows if int(row["message_spin"]) > 0], dtype=float)
        left = np.asarray([row["belief_spin"] > 0 for row in rows if int(row["message_spin"]) < 0], dtype=float)
        response_difference = float(np.mean(right) - np.mean(left))
        switch_fraction = float(np.mean([values[-1] != values[1] for values in by_pair.values()]))
        directional_fraction = float(np.mean([values[-1] < values[1] for values in by_pair.values()]))
        first_validity = float(np.mean([row["first_pass_valid"] for row in rows]))
        after_repair_validity = 1.0 - provider.accounting["invalid_after_repair"] / float(len(rows))
        passed = bool(
            first_validity >= float(settings["minimum_first_pass_validity"])
            and after_repair_validity >= float(settings["minimum_after_repair_validity"])
            and response_difference >= float(settings["minimum_right_minus_left_message_response"])
            and switch_fraction >= float(settings["minimum_paired_choice_switch_fraction"])
        )
        summary = {
            "completed_at": utc_now(),
            "protocol_version": amendment["protocol_version"],
            "amendment_sha256": sha256_file(amendment_path),
            "matched_pairs": len(by_pair),
            "decisions": len(rows),
            "first_pass_validity": first_validity,
            "after_repair_validity": after_repair_validity,
            "right_minus_left_message_response": response_difference,
            "paired_choice_switch_fraction": switch_fraction,
            "directional_pair_fraction": directional_fraction,
            "provider_accounting": provider.accounting,
            "environment": provider.environment_manifest(),
            "message_pilot_gate_passed": passed,
            "formal_unlocked_pending_source_freeze": passed,
        }
        _atomic_json(summary, completion)
        return summary


def freeze_qwen_protocol(repository: Path) -> Dict[str, object]:
    """Freeze the post-pilot LLM source without overwriting the CPU freeze."""

    base_path = repository / "configs/statmech_v10/protocol.yaml"
    amendment_path = repository / "configs/statmech_v10/llm_pilot_amendment.yaml"
    evidence_path = artifact_root() / "qwen/pilot/summary.json"
    message_path = artifact_root() / "qwen/message_pilot/summary.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    message = json.loads(message_path.read_text(encoding="utf-8"))
    if not evidence["pilot_gate_passed"] or not message["message_pilot_gate_passed"]:
        raise RuntimeError("LLM source cannot freeze because a required pilot gate failed")
    manifest = {
        "frozen_at": utc_now(),
        "protocol_version": load_yaml(amendment_path)["protocol_version"],
        "base_protocol_sha256": sha256_file(base_path),
        "llm_amendment_sha256": sha256_file(amendment_path),
        "scientific_source_sha256": source_checksum(repository),
        "evidence_pilot_summary_sha256": sha256_file(evidence_path),
        "message_pilot_summary_sha256": sha256_file(message_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "git_commit": None,
        "reason_git_commit_absent": "User required all V10 work to remain uncommitted.",
        "formal_llm_unlocked": True,
    }
    destination = artifact_root() / "qwen/formal_freeze_manifest.json"
    _atomic_json(manifest, destination)
    return manifest


def _calibration_rows(provider: QwenStructuredProvider, settings: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    calibration = settings["calibration"]
    call = 0
    for field in calibration["local_fields"]:
        for paraphrase in range(int(settings["prompt_paraphrases"])):
            for order_index in range(int(settings["option_orders"])):
                for replicate in range(int(calibration["inference_seeds_per_field_paraphrase_order"])):
                    seed = 520000 + call
                    agent = _fresh_agent(
                        0,
                        "local_coordinator",
                        -1 if replicate % 2 == 0 else 1,
                        -1 if replicate % 3 == 0 else 1,
                        private_evidence_text("humanitarian", float(field), paraphrase),
                    )
                    network = DecentralizedLLMNetwork([agent], np.zeros((1, 1)))
                    order = ("plan_left", "plan_right") if order_index == 0 else ("plan_right", "plan_left")
                    decision = network.offered_update(0, provider, seed, "belief", order, paraphrase)
                    record = network.decision_ledger[-1]
                    rows.append(
                        {
                            "call": call,
                            "seed": seed,
                            "local_field": float(field),
                            "paraphrase": paraphrase,
                            "option_order_right_first": int(order[0] == "plan_right"),
                            "previous_belief": -1 if replicate % 2 == 0 else 1,
                            "belief_spin": decision.belief_spin,
                            "belief_confidence": decision.belief_confidence,
                            "first_pass_valid": int(record["first_pass_valid"]),
                            "repaired": int(record["repaired"]),
                            "prompt_tokens": int(record["prompt_tokens"]),
                            "generated_tokens": int(record["generated_tokens"]),
                            "latency_seconds": float(record["latency_seconds"]),
                            "raw_artifact_sha256": str(record["raw_artifact_sha256"]),
                        }
                    )
                    call += 1
    return rows


def _kernel_rows(
    provider: QwenStructuredProvider,
    settings: Mapping[str, object],
    selected_alpha: Optional[float] = None,
) -> List[Dict[str, object]]:
    kernel_settings = settings["small_kernel"]
    n_agents = int(kernel_settings["n_agents"])
    base = topology_adjacency(n_agents, "ring", 530001)
    family = directed_family(base, 530002)
    private_fields = np.linspace(0.15, -0.10, n_agents)
    rows: List[Dict[str, object]] = []
    replicates = int(kernel_settings["repeats_per_state_variable_condition"])
    calls_per_alpha = (1 << (2 * n_agents)) * (2 * n_agents) * replicates
    for alpha_index, alpha in enumerate(kernel_settings["alphas"]):
        if selected_alpha is not None and not np.isclose(float(alpha), float(selected_alpha)):
            continue
        communication = family.at(float(alpha))
        for state_index in range(1 << (2 * n_agents)):
            for variable in range(2 * n_agents):
                layer = "belief" if variable < n_agents else "action"
                agent_index = variable if layer == "belief" else variable - n_agents
                for replicate in range(replicates):
                    call = (
                        alpha_index * calls_per_alpha
                        + state_index * (2 * n_agents) * replicates
                        + variable * replicates
                        + replicate
                    )
                    matched_call = (
                        state_index * (2 * n_agents) * replicates
                        + variable * replicates
                        + replicate
                    )
                    seed = 5300000 + matched_call
                    agents = _state_agents(
                        state_index,
                        n_agents,
                        "humanitarian",
                        communication,
                        private_fields,
                        replicate % int(settings["prompt_paraphrases"]),
                    )
                    network = DecentralizedLLMNetwork(agents, communication)
                    decision = network.offered_update(
                        agent_index,
                        provider,
                        seed,
                        layer,
                        _plan_order(seed),
                        replicate % int(settings["prompt_paraphrases"]),
                    )
                    state = decode_state(state_index, n_agents)
                    if layer == "belief":
                        state.beliefs[agent_index] = decision.belief_spin
                    else:
                        state.actions[agent_index] = decision.action_spin
                    record = network.decision_ledger[-1]
                    rows.append(
                        {
                            "call": call,
                            "seed": seed,
                            "alpha": float(alpha),
                            "state_index": state_index,
                            "variable": variable,
                            "layer": layer,
                            "agent": agent_index,
                            "replicate": replicate,
                            "destination_state": encode_state(state),
                            "belief_confidence": decision.belief_confidence,
                            "outgoing_signal": decision.outgoing_signal,
                            "tool_action": decision.tool_action,
                            "option_order_right_first": int(_plan_order(seed)[0] == "plan_right"),
                            "paraphrase": replicate % int(settings["prompt_paraphrases"]),
                            "first_pass_valid": int(record["first_pass_valid"]),
                            "repaired": int(record["repaired"]),
                            "prompt_tokens": int(record["prompt_tokens"]),
                            "generated_tokens": int(record["generated_tokens"]),
                            "latency_seconds": float(record["latency_seconds"]),
                            "raw_artifact_sha256": str(record["raw_artifact_sha256"]),
                        }
                    )
    return rows


def empirical_kernel_from_rows(rows: Sequence[Mapping[str, object]], n_agents: int, alpha: float, pseudocount: float = 0.5) -> np.ndarray:
    n_states = 1 << (2 * int(n_agents))
    counts = np.zeros((n_states, n_states), dtype=float)
    relevant = [row for row in rows if np.isclose(float(row["alpha"]), float(alpha))]
    for row in relevant:
        counts[int(row["state_index"]), int(row["destination_state"])] += 1.0
    # Jeffreys smoothing only on the two allowed destinations per scheduled variable.
    for source in range(n_states):
        state = decode_state(source, n_agents)
        for variable in range(2 * n_agents):
            for value in (-1, 1):
                destination = state.copy()
                values = destination.beliefs if variable < n_agents else destination.actions
                values[variable if variable < n_agents else variable - n_agents] = value
                counts[source, encode_state(destination)] += float(pseudocount)
    row_sums = counts.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise RuntimeError("empirical kernel contains an empty row")
    return counts / row_sums[:, None]


def _dynamic_condition_rows(
    provider: QwenStructuredProvider,
    settings: Mapping[str, object],
    n_index: int,
    panel: int,
    alpha_index: int,
) -> List[Dict[str, object]]:
    dynamic = settings["dynamic_networks"]
    n_agents = int(dynamic["agent_counts"][int(n_index)])
    alpha = float(dynamic["alphas"][int(alpha_index)])
    turns = int(dynamic["turns_per_panel"])
    panels = int(dynamic["independent_panels"])
    alpha_count = len(dynamic["alphas"])
    base = topology_adjacency(n_agents, "ring" if n_agents == 4 else "small_world", 540000 + panel + n_agents)
    family = directed_family(base, 541000 + panel + n_agents)
    initial_agents = make_agents(n_agents, 542000 + panel, application_roles("utility", n_agents))
    network = DecentralizedLLMNetwork([agent.clone() for agent in initial_agents], family.at(alpha))
    environment = DefensiveUtilityMapping() if panel % 2 else HumanitarianCoordinationMapping()
    application = "utility" if panel % 2 else "humanitarian"
    rows: List[Dict[str, object]] = []
    state_sequence: List[int] = []
    for turn in range(turns):
        call = (((int(n_index) * panels + int(panel)) * alpha_count + int(alpha_index)) * turns + turn)
        agent_index = turn % n_agents
        matched_call = ((int(n_index) * panels + int(panel)) * turns + turn)
        seed = 5400000 + matched_call
        messages_before = len(network.message_ledger)
        bytes_before = network.message_wire_bytes
        decision = network.offered_update(
            agent_index,
            provider,
            seed,
            None,
            _plan_order(seed),
            turn % int(settings["prompt_paraphrases"]),
        )
        consequence = environment.apply(decision)
        network.private_agent_for_test(agent_index).append_memory(
            "validated local tool outcome service_after=%.3f" % float(consequence["service_after"])
        )
        beliefs = np.asarray([network.private_agent_for_test(i)._belief for i in range(n_agents)])
        actions = np.asarray([network.private_agent_for_test(i)._action for i in range(n_agents)])
        # Coarse macrostate retains signed belief and action counts.
        macrostate = int((beliefs > 0).sum() * (n_agents + 1) + (actions > 0).sum())
        state_sequence.append(macrostate)
        record = network.decision_ledger[-1]
        rows.append(
            {
                "call": call,
                "seed": seed,
                "application": application,
                "n_agents": n_agents,
                "panel": int(panel),
                "alpha": alpha,
                "turn": turn,
                "agent": agent_index,
                "belief_spin": decision.belief_spin,
                "action_spin": decision.action_spin,
                "belief_confidence": decision.belief_confidence,
                "belief_magnetization": float(np.mean(beliefs)),
                "action_magnetization": float(np.mean(actions)),
                "coarse_macrostate": macrostate,
                "messages_sent": len(network.message_ledger) - messages_before,
                "message_wire_bytes": network.message_wire_bytes - bytes_before,
                "message_count_cumulative": len(network.message_ledger),
                "message_wire_bytes_cumulative": network.message_wire_bytes,
                "outgoing_signal": decision.outgoing_signal,
                "tool_action": decision.tool_action,
                "commitment_status": decision.commitment_status,
                "service_before": float(consequence["service_before"]),
                "service_after": float(consequence["service_after"]),
                "causal_service_change": float(consequence["causal_service_change"]),
                "first_pass_valid": int(record["first_pass_valid"]),
                "repaired": int(record["repaired"]),
                "prompt_tokens": int(record["prompt_tokens"]),
                "generated_tokens": int(record["generated_tokens"]),
                "latency_seconds": float(record["latency_seconds"]),
                "raw_artifact_sha256": str(record["raw_artifact_sha256"]),
            }
        )
    irreversibility = block_time_reversal_kl(state_sequence, 3, 0.5)
    markov_cmi = conditional_mutual_information_markov(state_sequence, 0.1)
    for row in rows:
        row["trajectory_block_irreversibility"] = irreversibility
        row["markov_cmi"] = markov_cmi
    return rows


def _dynamic_rows(provider: QwenStructuredProvider, settings: Mapping[str, object]) -> List[Dict[str, object]]:
    dynamic = settings["dynamic_networks"]
    rows: List[Dict[str, object]] = []
    for n_index, _n_agents in enumerate(dynamic["agent_counts"]):
        for panel in range(int(dynamic["independent_panels"])):
            for alpha_index, _alpha in enumerate(dynamic["alphas"]):
                rows.extend(_dynamic_condition_rows(provider, settings, n_index, panel, alpha_index))
    return rows


def _read_rows(path: Path) -> List[Dict[str, object]]:
    return pd.read_csv(path).to_dict(orient="records")


def _aggregate_accounting(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    repaired = int(sum(int(row["repaired"]) for row in rows))
    return {
        "decisions": len(rows),
        "model_calls": len(rows) + repaired,
        "first_pass_valid": int(sum(int(row["first_pass_valid"]) for row in rows)),
        "repaired_valid": repaired,
        "invalid_after_repair": 0,
        "prompt_tokens": int(sum(int(row["prompt_tokens"]) for row in rows)),
        "generated_tokens": int(sum(int(row["generated_tokens"]) for row in rows)),
        "latency_seconds": float(sum(float(row["latency_seconds"]) for row in rows)),
    }


def run_qwen_formal(repository: Path) -> Dict[str, object]:
    _require_execution_opt_in()
    pilot_path = artifact_root() / "qwen/pilot/summary.json"
    if not pilot_path.exists() or not json.loads(pilot_path.read_text(encoding="utf-8"))["pilot_gate_passed"]:
        raise RuntimeError("Qwen formal study remains locked because the pilot gate has not passed")
    message_path = artifact_root() / "qwen/message_pilot/summary.json"
    if not message_path.exists() or not json.loads(message_path.read_text(encoding="utf-8"))["message_pilot_gate_passed"]:
        raise RuntimeError("Qwen formal study remains locked because the message counterfactual gate has not passed")
    freeze_path = artifact_root() / "qwen/formal_freeze_manifest.json"
    if not freeze_path.exists():
        raise RuntimeError("Qwen formal study remains locked because the LLM source has not been frozen")
    freeze_manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze_manifest["scientific_source_sha256"] != source_checksum(repository):
        raise RuntimeError("Qwen scientific source changed after the LLM freeze")
    if freeze_manifest["base_protocol_sha256"] != sha256_file(repository / "configs/statmech_v10/protocol.yaml"):
        raise RuntimeError("Qwen base protocol changed after the LLM freeze")
    if freeze_manifest["llm_amendment_sha256"] != sha256_file(repository / "configs/statmech_v10/llm_pilot_amendment.yaml"):
        raise RuntimeError("Qwen pilot amendment changed after the LLM freeze")
    protocol = load_yaml(repository / "configs/statmech_v10/protocol.yaml")
    settings = copy.deepcopy(protocol["llm_execution_design"])
    amendment = load_yaml(repository / "configs/statmech_v10/llm_pilot_amendment.yaml")
    overrides = amendment["formal_design_overrides"]
    settings["dynamic_networks"]["independent_panels"] = int(overrides["dynamic_independent_panels"])
    planned_decisions = (
        int(settings["calibration"]["planned_calls"])
        + int(settings["small_kernel"]["planned_calls"])
        + len(settings["dynamic_networks"]["agent_counts"])
        * int(settings["dynamic_networks"]["independent_panels"])
        * len(settings["dynamic_networks"]["alphas"])
        * int(settings["dynamic_networks"]["turns_per_panel"])
    )
    if planned_decisions != int(overrides["planned_primary_decisions"]):
        raise RuntimeError("formal Qwen decision count does not match the frozen amendment")
    output = artifact_root() / "qwen/formal"
    completion = output / "summary.json"
    if completion.exists():
        return json.loads(completion.read_text(encoding="utf-8"))
    provider = QwenStructuredProvider(
        artifact_root() / "qwen/raw/formal",
        repository,
        float(settings["inference_sampling_temperature"]),
        float(settings["top_p"]),
        int(settings["maximum_new_tokens"]),
    )
    began = time.perf_counter()
    with stage_lock("qwen_formal"):
        checkpoints = output / "checkpoints"
        calibration_checkpoint = checkpoints / "calibration.csv"
        if calibration_checkpoint.exists():
            calibration = _read_rows(calibration_checkpoint)
        else:
            calibration = _calibration_rows(provider, settings)
            _atomic_csv(calibration, calibration_checkpoint)
        _atomic_csv(calibration, output / "calibration.csv")
        kernel_rows: List[Dict[str, object]] = []
        for alpha in settings["small_kernel"]["alphas"]:
            checkpoint = checkpoints / ("kernel_alpha_%s.csv" % str(alpha).replace(".", "p"))
            if checkpoint.exists():
                batch = _read_rows(checkpoint)
            else:
                batch = _kernel_rows(provider, settings, float(alpha))
                _atomic_csv(batch, checkpoint)
            kernel_rows.extend(batch)
        _atomic_csv(kernel_rows, output / "controlled_kernel.csv")
        dynamic_rows: List[Dict[str, object]] = []
        dynamic = settings["dynamic_networks"]
        for n_index, n_agents in enumerate(dynamic["agent_counts"]):
            for panel in range(int(dynamic["independent_panels"])):
                for alpha_index, alpha in enumerate(dynamic["alphas"]):
                    checkpoint = checkpoints / (
                        "dynamic_n%d_panel%d_alpha_%s.csv"
                        % (int(n_agents), panel, str(alpha).replace(".", "p"))
                    )
                    if checkpoint.exists():
                        batch = _read_rows(checkpoint)
                    else:
                        batch = _dynamic_condition_rows(provider, settings, n_index, panel, alpha_index)
                        _atomic_csv(batch, checkpoint)
                    dynamic_rows.extend(batch)
        _atomic_csv(dynamic_rows, output / "dynamic_trajectories.csv")
        kernel_results = []
        n_agents = int(settings["small_kernel"]["n_agents"])
        for alpha in settings["small_kernel"]["alphas"]:
            kernel = empirical_kernel_from_rows(kernel_rows, n_agents, float(alpha), 0.5)
            stationary = stationary_distribution(kernel)
            kernel_results.append(
                {
                    "alpha": float(alpha),
                    "empirical_kernel_epr_per_controlled_update": entropy_production_rate(stationary, kernel),
                    "stationarity_residual": float(np.max(np.abs(stationary.dot(kernel) - stationary))),
                }
            )
        _atomic_csv(kernel_results, output / "kernel_summary.csv")
        frame_fields = [float(row["local_field"]) for row in calibration]
        fit = fit_logistic_response(
            frame_fields,
            [int(row["belief_spin"]) for row in calibration],
            [int(row["previous_belief"]) for row in calibration],
            [1 if int(row["option_order_right_first"]) else -1 for row in calibration],
        )
        summary = {
            "completed_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - began,
            "calibration_decisions": len(calibration),
            "controlled_kernel_decisions": len(kernel_rows),
            "dynamic_decisions": len(dynamic_rows),
            "provider_accounting_current_process": provider.accounting,
            "aggregate_accounting": _aggregate_accounting(calibration + kernel_rows + dynamic_rows),
            "environment": provider.environment_manifest(),
            "resumability": "atomic calibration, per-alpha kernel, and per-panel-alpha dynamic checkpoints",
            "local_policy_fit": fit,
            "kernel_results": kernel_results,
            "raw_artifact_root": str(artifact_root() / "qwen/raw/formal"),
        }
        _atomic_json(summary, completion)
        return summary
