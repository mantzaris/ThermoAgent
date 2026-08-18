"""Prospective matched evidence-use qualification for V11."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import (
    EvidenceGroundedDecision,
    EvidencePacket,
    IndependentEvidenceAgent,
    build_evidence_prompt,
    serialize_evidence_packet,
)
from .qwen import QwenEvidenceProvider
from .statistics import calibration_summary, fit_reliability_response, paired_cluster_bootstrap, safe_logit
from .workflow import artifact_root, atomic_csv, atomic_json, load_yaml, stage_lock, utc_now


@dataclass(frozen=True)
class EvidenceCondition:
    name: str
    packets: Tuple[EvidencePacket, ...]
    expected_direction: int
    nominal_reliability: float
    delivery_mode: str


def _packet(source: str, observation: str, reliability: float, freshness: float, domain: str) -> EvidencePacket:
    return EvidencePacket(
        source_id=source,
        observation=observation,
        reliability=float(reliability),
        observation_time=0,
        delivery_time=0 if freshness >= 0.999 else 4,
        freshness=float(freshness),
        evidence_domain=domain,
        explanation="A conditionally independent observation from the shared signal model.",
    )


def qualification_conditions(domain: str, reliabilities: Sequence[float], include_extended: bool) -> List[EvidenceCondition]:
    conditions = [EvidenceCondition("no_message", tuple(), 0, 0.5, "none")]
    conditions.append(
        EvidenceCondition(
            "placebo",
            (
                EvidencePacket(
                    source_id="peer_placebo",
                    observation="unknown",
                    reliability=0.5,
                    observation_time=0,
                    delivery_time=0,
                    freshness=1.0,
                    evidence_domain=domain,
                    explanation="No directional observation was available.",
                    packet_kind="placebo",
                ),
            ),
            0,
            0.5,
            "one_way",
        )
    )
    for reliability in reliabilities:
        for direction, observation in ((-1, "left"), (1, "right")):
            conditions.append(
                EvidenceCondition(
                    "single_%s_r%.2f" % (observation, reliability),
                    (_packet("peer_1", observation, reliability, 1.0, domain),),
                    direction,
                    float(reliability),
                    "one_way",
                )
            )
    if include_extended:
        for direction, observation in ((-1, "left"), (1, "right")):
            opposite = "right" if observation == "left" else "left"
            conditions.extend(
                [
                    EvidenceCondition(
                        "stale_%s_r0.85" % observation,
                        (_packet("peer_1", observation, 0.85, 0.25, domain),),
                        direction,
                        0.85,
                        "one_way",
                    ),
                    EvidenceCondition(
                        "agreeing_%s_two_r0.65" % observation,
                        (
                            _packet("peer_1", observation, 0.65, 1.0, domain),
                            _packet("peer_2", observation, 0.65, 1.0, domain),
                        ),
                        direction,
                        0.65,
                        "one_way",
                    ),
                    EvidenceCondition(
                        "conflict_net_%s" % observation,
                        (
                            _packet("peer_strong", observation, 0.85, 1.0, domain),
                            _packet("peer_weak", opposite, 0.65, 1.0, domain),
                        ),
                        direction,
                        0.85,
                        "one_way",
                    ),
                    EvidenceCondition(
                        "reciprocal_%s_r0.75" % observation,
                        (_packet("peer_1", observation, 0.75, 1.0, domain),),
                        direction,
                        0.75,
                        "reciprocal",
                    ),
                ]
            )
    return conditions


def _prompt_with_delivery_mode(
    agent: IndependentEvidenceAgent,
    mode: str,
    order: Tuple[str, str],
    paraphrase: int,
    time_step: int,
    delivery_mode: str,
) -> str:
    prompt = build_evidence_prompt(agent, mode, order, paraphrase, time_step)
    return prompt + "\nLOCAL_DELIVERY_MODE=" + json.dumps(delivery_mode)


def prompt_pair_fingerprint(prompt: str) -> str:
    """Fingerprint a prompt after removing only delivered packets and delivery mode."""

    prefix, serialized = prompt.split("\nCONTROLLED_TASK=", 1)
    envelope_text, _mode = serialized.rsplit("\nLOCAL_DELIVERY_MODE=", 1)
    envelope = json.loads(envelope_text)
    envelope["authorized_local_view"]["delivered_evidence"] = "<TREATMENT>"
    return hashlib.sha256((prefix + json.dumps(envelope, sort_keys=True)).encode("utf-8")).hexdigest()


def _stage_design(settings: Mapping[str, object], include_extended: bool) -> List[Dict[str, object]]:
    reliabilities = [float(value) for value in settings["evidence_reliabilities"]]  # type: ignore[index]
    replicates = int(settings["inference_replicates"])
    rows: List[Dict[str, object]] = []
    cluster = 0
    for domain in ("route_viability", "repair_hypothesis"):
        conditions = qualification_conditions(domain, reliabilities, include_extended)
        for paraphrase in range(int(settings["prompt_paraphrases"])):
            for order_index, order in enumerate((("left", "right"), ("right", "left"))):
                for private_observation in ("left", "right"):
                    for replicate in range(replicates):
                        cluster_id = "c%04d" % cluster
                        cluster += 1
                        for condition in conditions:
                            rows.append(
                                {
                                    "cluster_id": cluster_id,
                                    "domain": domain,
                                    "paraphrase": paraphrase,
                                    "option_order": order,
                                    "option_order_right_first": int(order[0] == "right"),
                                    "private_observation": private_observation,
                                    "replicate": replicate,
                                    "condition": condition,
                                }
                            )
    return rows


def expected_decision_requests(settings: Mapping[str, object], include_extended: bool) -> int:
    return len(_stage_design(settings, include_extended))


def _require_qwen_opt_in() -> None:
    if os.environ.get("THERMO_V11_ENABLE_QWEN") != "1":
        raise RuntimeError("Qwen execution is locked; opt in only on the existing authorized RunPod")


def run_qualification_stage(repository: Path, stage: str) -> Dict[str, object]:
    _require_qwen_opt_in()
    if stage not in ("pilot", "qualification"):
        raise ValueError("stage must be pilot or qualification")
    config_name = "engineering.yaml" if stage == "pilot" else "qualification_frozen.yaml"
    config = load_yaml(Path(repository) / "configs/statmech_v11" / config_name)
    settings = config[stage]  # type: ignore[index]
    include_extended = bool(settings["include_extended_conditions"])
    output = artifact_root() / stage
    decision_path = output / "decisions.csv"
    summary_path = output / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, object]] = []
    completed = set()
    if decision_path.exists():
        old = pd.read_csv(decision_path)
        existing = old.to_dict(orient="records")
        completed = set(str(value) for value in old["request_id"])
    provider = QwenEvidenceProvider(
        artifact_root() / "raw" / stage,
        repository,
        float(settings["inference_sampling_temperature"]),
        float(settings["top_p"]),
        int(settings["maximum_new_tokens"]),
    )
    design = _stage_design(settings, include_extended)
    rows = list(existing)
    with stage_lock(stage):
        for index, cell in enumerate(design):
            condition = cell["condition"]
            request_id = "%s_%06d" % (stage, index)
            if request_id in completed:
                continue
            private = _packet(
                "private_agent",
                str(cell["private_observation"]),
                float(settings["private_signal_reliability"]),
                1.0,
                str(cell["domain"]),
            )
            agent = IndependentEvidenceAgent(int(str(cell["cluster_id"])[1:]), "local_coordinator", private)
            for packet in condition.packets:
                agent.receive(packet)
            prompt = _prompt_with_delivery_mode(
                agent,
                "qualification_unanchored",
                cell["option_order"],  # type: ignore[arg-type]
                int(cell["paraphrase"]),
                0,
                condition.delivery_mode,
            )
            seed = int(settings["seed_base"]) + int(str(cell["cluster_id"])[1:])
            row: Dict[str, object] = {
                "request_id": request_id,
                "cluster_id": cell["cluster_id"],
                "domain": cell["domain"],
                "paraphrase": cell["paraphrase"],
                "option_order_right_first": cell["option_order_right_first"],
                "private_observation": cell["private_observation"],
                "replicate": cell["replicate"],
                "condition": condition.name,
                "expected_direction": condition.expected_direction,
                "nominal_reliability": condition.nominal_reliability,
                "delivery_mode": condition.delivery_mode,
                "delivered_packet_count": len(condition.packets),
                "delivered_wire_bytes": sum(len(serialize_evidence_packet(packet)) for packet in condition.packets),
                "paired_prompt_fingerprint": prompt_pair_fingerprint(prompt),
                "inference_seed": seed,
                "valid_after_repair": 0,
            }
            try:
                result = provider.decide(prompt, seed)
                decision = EvidenceGroundedDecision.from_mapping(result.payload)
                outgoing_accepted = agent.apply_decision(decision, 0)
                row.update(
                    {
                        "valid_after_repair": 1,
                        "first_pass_valid": int(result.first_pass_valid),
                        "repaired": int(result.repaired),
                        "probability_right": decision.probability_right,
                        "belief_right": int(decision.belief_choice == "right"),
                        "action_choice": decision.action_choice,
                        "commitment_status": decision.commitment_status,
                        "outgoing_packet_present": int(decision.outgoing_evidence_action == "send_private_evidence"),
                        "outgoing_packet_accepted": int(outgoing_accepted),
                        "reason_code": decision.reason_code,
                        "prompt_tokens": result.prompt_tokens,
                        "generated_tokens": result.generated_tokens,
                        "latency_seconds": result.latency_seconds,
                        "raw_artifact_sha256": result.raw_artifact_sha256,
                    }
                )
            except ValueError:
                row.update({"first_pass_valid": 0, "repaired": 0})
            rows.append(row)
            atomic_csv(rows, decision_path)
        summary = {
            "stage": stage,
            "completed_at": utc_now(),
            "decision_requests": len(design),
            "completed_rows": len(rows),
            "provider_accounting": provider.accounting,
            "environment": provider.environment_manifest(),
        }
        atomic_json(summary, summary_path)
    return summary


def _effect_rows(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["valid_after_repair"] == 1].copy()
    baseline = valid[valid["condition"] == "no_message"][["cluster_id", "probability_right"]].rename(
        columns={"probability_right": "baseline_probability_right"}
    )
    merged = valid.merge(baseline, on="cluster_id", how="inner", validate="many_to_one")
    merged["logit_change"] = safe_logit(merged["probability_right"].to_numpy(float)) - safe_logit(
        merged["baseline_probability_right"].to_numpy(float)
    )
    merged["signed_logit_change"] = merged["logit_change"] * merged["expected_direction"]
    return merged


def analyze_qualification_stage(repository: Path, stage: str) -> Dict[str, object]:
    if stage not in ("pilot", "qualification"):
        raise ValueError("invalid stage")
    output = artifact_root() / stage
    frame = pd.read_csv(output / "decisions.csv")
    effects = _effect_rows(frame)
    primary = effects[
        effects["condition"].str.startswith("single_") & (effects["expected_direction"] != 0)
    ].copy()
    clusters = {
        str(key): group["signed_logit_change"].to_numpy(float)
        for key, group in primary.groupby("cluster_id", sort=True)
    }
    bootstrap = paired_cluster_bootstrap(clusters, 10000, 11701 if stage == "pilot" else 11702)
    placebo_probability = effects[effects["condition"] == "placebo"][["cluster_id", "probability_right"]].rename(
        columns={"probability_right": "placebo_probability_right"}
    )
    placebo_adjusted = primary.merge(placebo_probability, on="cluster_id", how="inner", validate="many_to_one")
    placebo_adjusted["placebo_adjusted_signed_logit_change"] = placebo_adjusted["expected_direction"] * (
        safe_logit(placebo_adjusted["probability_right"].to_numpy(float))
        - safe_logit(placebo_adjusted["placebo_probability_right"].to_numpy(float))
    )
    placebo_adjusted_bootstrap = paired_cluster_bootstrap(
        {
            str(key): group["placebo_adjusted_signed_logit_change"].to_numpy(float)
            for key, group in placebo_adjusted.groupby("cluster_id", sort=True)
        },
        10000,
        11711 if stage == "pilot" else 11712,
    )
    reliability = fit_reliability_response(
        primary["nominal_reliability"].to_numpy(float), primary["signed_logit_change"].to_numpy(float)
    )
    placebo = effects[effects["condition"] == "placebo"]
    placebo_abs = float(np.mean(np.abs(placebo["logit_change"].to_numpy(float)))) if len(placebo) else float("nan")
    calibration_cells = (
        frame[frame["valid_after_repair"] == 1]
        .groupby(
            ["domain", "paraphrase", "option_order_right_first", "private_observation", "condition"],
            sort=True,
        )
        .agg(mean_reported_probability=("probability_right", "mean"), empirical_right_frequency=("belief_right", "mean"))
        .reset_index()
    )
    calibration = calibration_summary(
        calibration_cells["mean_reported_probability"].to_numpy(float),
        calibration_cells["empirical_right_frequency"].to_numpy(float),
    )
    by_reliability = [
        {
            "nominal_reliability": float(key),
            "mean_signed_logit_change": float(group["signed_logit_change"].mean()),
            "count": int(len(group)),
        }
        for key, group in primary.groupby("nominal_reliability", sort=True)
    ]
    by_domain: Dict[str, Dict[str, float]] = {}
    for domain, group in primary.groupby("domain", sort=True):
        by_domain[str(domain)] = paired_cluster_bootstrap(
            {str(key): part["signed_logit_change"].to_numpy(float) for key, part in group.groupby("cluster_id")},
            10000,
            11800 + len(by_domain),
        )
    order_means = primary.groupby("option_order_right_first")["signed_logit_change"].mean()
    option_order_effect = float(order_means.get(1, np.nan) - order_means.get(0, np.nan))
    paraphrase_means = primary.groupby("paraphrase")["signed_logit_change"].mean().to_numpy(float)
    paraphrase_range = float(np.max(paraphrase_means) - np.min(paraphrase_means))
    mechanism: Dict[str, float] = {}
    if stage == "qualification":
        condition_means = effects.groupby("condition")["signed_logit_change"].mean()
        for direction in ("left", "right"):
            single_65 = float(condition_means.get("single_%s_r0.65" % direction, np.nan))
            single_75 = float(condition_means.get("single_%s_r0.75" % direction, np.nan))
            single_85 = float(condition_means.get("single_%s_r0.85" % direction, np.nan))
            mechanism["agreeing_minus_single_r0.65_%s" % direction] = float(
                condition_means.get("agreeing_%s_two_r0.65" % direction, np.nan) - single_65
            )
            mechanism["fresh_minus_stale_r0.85_%s" % direction] = float(
                single_85 - condition_means.get("stale_%s_r0.85" % direction, np.nan)
            )
            mechanism["reciprocal_minus_oneway_r0.75_%s" % direction] = float(
                condition_means.get("reciprocal_%s_r0.75" % direction, np.nan) - single_75
            )
    first_pass = float(frame["first_pass_valid"].fillna(0).mean())
    after_repair = float(frame["valid_after_repair"].mean())
    report: Dict[str, object] = {
        "stage": stage,
        "analyzed_at": utc_now(),
        "decision_requests": int(len(frame)),
        "independent_matched_clusters": int(frame["cluster_id"].nunique()),
        "first_pass_validity": first_pass,
        "after_repair_validity": after_repair,
        "primary_signed_logit_effect": bootstrap,
        "reliability_response": reliability,
        "by_reliability": by_reliability,
        "by_domain": by_domain,
        "placebo_mean_absolute_logit_shift": placebo_abs,
        "placebo_adjusted_directional_effect": placebo_adjusted_bootstrap,
        "absolute_option_order_effect": abs(option_order_effect),
        "paraphrase_mean_range": paraphrase_range,
        "calibration": calibration,
        "mechanism_checks": mechanism,
        "probability_standard_deviation": float(frame.loc[frame["valid_after_repair"] == 1, "probability_right"].std()),
        "right_choice_fraction": float(frame.loc[frame["valid_after_repair"] == 1, "belief_right"].mean()),
        "outgoing_packet_acceptance_fraction": float(
            frame.loc[frame["valid_after_repair"] == 1, "outgoing_packet_accepted"].mean()
        ),
        "paired_prompt_identity_passed": bool(frame.groupby("cluster_id")["paired_prompt_fingerprint"].nunique().max() == 1),
    }
    if stage == "qualification":
        config = load_yaml(Path(repository) / "configs/statmech_v11/qualification_frozen.yaml")
        thresholds = config["qualification"]["gates"]  # type: ignore[index]
        domain_pass = all(float(value["ci_low"]) > 0.0 for value in by_domain.values())
        monotone_values = [row["mean_signed_logit_change"] for row in by_reliability]
        monotone = all(b >= a - float(thresholds["monotonic_tolerance"]) for a, b in zip(monotone_values, monotone_values[1:]))
        report["gate_components"] = {
            "validity": first_pass >= float(thresholds["minimum_first_pass_validity"]) and after_repair >= float(thresholds["minimum_after_repair_validity"]),
            "directional_effect": float(bootstrap["ci_low"]) > 0.0 and float(bootstrap["estimate"]) >= float(thresholds["minimum_signed_logit_effect"]),
            "monotonicity": monotone and float(reliability["normative_llr_slope"]) >= float(thresholds["minimum_llr_slope"]),
            "placebo_separation": float(placebo_adjusted_bootstrap["ci_low"]) > 0.0 and float(placebo_adjusted_bootstrap["estimate"]) >= float(thresholds["minimum_placebo_adjusted_effect"]),
            "order_and_paraphrase": abs(option_order_effect) <= float(thresholds["maximum_absolute_order_effect"]) and paraphrase_range <= float(thresholds["maximum_paraphrase_range"]),
            "semantic_replication": domain_pass,
            "prompt_isolation": bool(report["paired_prompt_identity_passed"]),
            "transition_diversity": float(report["probability_standard_deviation"]) >= float(thresholds["minimum_probability_sd"]) and float(thresholds["minimum_choice_fraction"]) <= float(report["right_choice_fraction"]) <= 1.0 - float(thresholds["minimum_choice_fraction"]),
            "message_actionability": float(report["outgoing_packet_acceptance_fraction"]) >= float(thresholds["minimum_outgoing_packet_acceptance"]),
        }
        report["qualification_gate_passed"] = bool(all(report["gate_components"].values()))  # type: ignore[union-attr]
        report["formal_network_unlocked"] = report["qualification_gate_passed"]
    atomic_json(report, output / "analysis.json")
    atomic_csv(by_reliability, output / "by_reliability.csv")
    return report
