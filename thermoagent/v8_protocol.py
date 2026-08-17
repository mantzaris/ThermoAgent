"""Prospective V8 protocol and sealed panel-manifest construction."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from .events import sha256_file
from .v5_experiments import atomic_json, source_checksum, write_csv


PRIMARY_SEEDS = (88201, 88202, 88203, 88204, 88205)
APPLICATIONS = ("humanitarian", "utility_restoration")


def _panels(stage: str, per_application: int) -> List[Dict[str, Any]]:
    if stage not in ("validation", "holdout"):
        raise ValueError("sealed V8 panels are validation or holdout only")
    base = 88300000 if stage == "validation" else 88400000
    rows: List[Dict[str, Any]] = []
    for application_index, application in enumerate(APPLICATIONS):
        familiar = (
            ("random_geometric", "small_world")
            if application == "humanitarian" else ("grid", "scale_free")
        )
        for index in range(int(per_application)):
            # Modular is structurally held out from formal development. In the
            # holdout it remains half the panels; the remaining half use fresh
            # graph instances in deliberately novel factor combinations.
            if stage == "validation" or index < per_application // 2:
                topology = "modular"
            else:
                topology = familiar[index % len(familiar)]
            rows.append({
                "application": application,
                "complexity": ("small", "medium", "large")[index % 3],
                "coupling": ("low", "medium", "high")[(index // 3) % 3],
                "fragmentation": ("high", "low", "medium")[(index // 2) % 3],
                "network_disruption": ("medium", "high", "low")[(index // 5) % 3],
                "topology_family": topology,
                "environment_seed": base + application_index * 10000 + index + 1,
            })
    return rows


def _registry(results_root: Path) -> Dict[str, Dict[str, Any]]:
    path = results_root / "development" / "candidate_registry.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for row in rows:
        configuration = json.loads(row["configuration_json"])
        configuration["name"] = row["candidate_name"]
        configuration["encoding"] = row["encoding"]
        values[row["candidate_name"]] = configuration
    return values


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(
            dict(value), sort_keys=False, default_flow_style=False,
        ))
    temporary.replace(path)


def freeze_v8_protocol(repository: Path, results_root: Path) -> Dict[str, Any]:
    """Freeze only after complete formal development and before training."""
    selection_path = results_root / "development_final" / "development_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    gates = dict(selection["development_feasibility"])
    if not gates.get("development_progression_feasible", False):
        raise RuntimeError("V8 development feasibility gate did not unlock protocol freeze")
    generalized = str(selection["selected_generalized_trigger"])
    comparator = str(selection["selected_strongest_nonentropic_comparator"])
    registry_path = results_root / "development_final" / "candidate_registry.csv"
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    registry = {}
    for row in registry_rows:
        configuration = json.loads(row["configuration_json"])
        configuration["name"] = row["candidate_name"]
        configuration["encoding"] = row["encoding"]
        registry[row["candidate_name"]] = configuration
    broad_registry_path = results_root / "development" / "candidate_registry.csv"
    broad_registry = {}
    with broad_registry_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            configuration = json.loads(row["configuration_json"])
            configuration["name"] = row["candidate_name"]
            configuration["encoding"] = row["encoding"]
            broad_registry[row["candidate_name"]] = configuration
    candidates = [
        registry["always_on_u8"], registry["none_u8"],
        registry[generalized], registry[comparator],
    ]
    ensemble_path = (
        "results/entropy_triggered_belief_monitoring_v8/training/checkpoints/"
        "v8-ippo-five-seed-ensemble.json.gz"
    )
    validation_panels = _panels("validation", 30)
    holdout_panels = _panels("holdout", 40)
    common = {
        "version": "v8-frozen-1.0",
        "results_namespace": "results/entropy_triggered_belief_monitoring_v8",
        "maximum_hops": 2,
        "operational_communication_policy": "agent_event_triggered",
        "information_condition": "private_fragmented",
        "ledger_scope": "dynamic_delta",
        "workers": 4,
        "candidates": candidates,
        "policy_checkpoints": [ensemble_path],
    }
    development_source = yaml.safe_load(
        (repository / "configs" / "v8_development.yaml").read_text(encoding="utf-8")
    )
    development_agent_config = {
        **common,
        "stage": "development_agent",
        "panels": list(development_source["panels"]),
    }
    member_paths = [
        "results/entropy_triggered_belief_monitoring_v8/training/checkpoints/"
        "v8-ippo-seed-%d.json.gz" % value for value in PRIMARY_SEEDS
    ]
    stability_panels = []
    for application in APPLICATIONS:
        available = [
            value for value in development_source["panels"]
            if value["application"] == application
        ]
        for complexity in ("small", "medium", "large"):
            stability_panels.extend([
                value for value in available if value["complexity"] == complexity
            ][:2])
    seed_stability_config = {
        **common,
        "stage": "seed_stability",
        "panels": stability_panels,
        "policy_checkpoints": member_paths,
    }
    validation_config = {**common, "stage": "validation", "panels": validation_panels}
    holdout_config = {**common, "stage": "holdout", "panels": holdout_panels}
    ablation_panels = []
    for application in APPLICATIONS:
        available = [
            value for value in development_source["panels"]
            if value["application"] == application
        ]
        for complexity in ("small", "medium", "large"):
            ablation_panels.extend([
                value for value in available if value["complexity"] == complexity
            ][:4])
    ablation_candidates = list(broad_registry.values())
    for name in ("generalized_013_u8", "kpi_010_u8"):
        if name in registry and name not in broad_registry:
            ablation_candidates.append(registry[name])
    ablation_config = {
        **common, "stage": "ablations", "panels": ablation_panels,
        "candidates": ablation_candidates,
    }
    validation_path = repository / "configs" / "v8_validation_frozen.yaml"
    holdout_path = repository / "configs" / "v8_holdout_locked.yaml"
    development_agent_path = repository / "configs" / "v8_development_agent_frozen.yaml"
    seed_stability_path = repository / "configs" / "v8_seed_stability_frozen.yaml"
    ablation_path = repository / "configs" / "v8_ablations_frozen.yaml"
    _write_yaml(development_agent_path, development_agent_config)
    _write_yaml(seed_stability_path, seed_stability_config)
    _write_yaml(validation_path, validation_config)
    _write_yaml(holdout_path, holdout_config)
    _write_yaml(ablation_path, ablation_config)
    write_csv(results_root / "manifests" / "validation_panels.csv", validation_panels)
    write_csv(results_root / "manifests" / "holdout_panels.csv", holdout_panels)
    write_csv(results_root / "manifests" / "ablation_panels.csv", ablation_panels)
    write_csv(results_root / "manifests" / "rl_training_seeds.csv", [
        {"rl_seed": value, "status": "prospectively_selected"}
        for value in PRIMARY_SEEDS
    ])
    protocol = {
        "protocol_version": "v8-frozen-1.0",
        "study": "Entropy-triggered belief monitoring V8",
        "stage_at_freeze": "post-development_pre-training_pre-validation",
        "primary_trigger": generalized,
        "primary_trigger_configuration": registry[generalized],
        "primary_encoding": "uint8_simplex",
        "strongest_nonentropic_comparator": comparator,
        "strongest_nonentropic_comparator_configuration": registry[comparator],
        "wire_protocol": {
            "schema_version": 1,
            "framing": "TBV8 big-endian deterministic struct plus CRC32",
            "byte_accounting": "exact len(serialized frame), including header, payload, integrity, forwarding, and drops after transmission",
            "primary_payload": "uint8 simplex with largest-remainder mass summing to 255",
        },
        "action_policy": {
            "algorithm": "role-specific decentralized linear IPPO with per-agent terminal-aware GAE",
            "training_seeds": list(PRIMARY_SEEDS),
            "training_episodes_per_seed": 18,
            "training_scheduler_mixture": [
                "always_on", "generalized_information", "kpi_change", "periodic", "none",
            ],
            "frozen_evaluation_policy": "unweighted parameter mean over all five completed seeds",
            "best_seed_selection": False,
        },
        "primary_hypotheses": {
            "H1": "generalized trigger reduces belief-sketch messages and actual wire bytes versus always-on while preserving distributed estimation in both applications",
            "H2": "at matched actual bytes the generalized trigger improves distributed-state estimation versus the frozen strongest non-entropic scheduler in both applications",
            "H3": "the same frozen decentralized policy retains service, harmful-action, and reward performance under generalized versus always-on exchange",
        },
        "primary_margins": {
            "H1_message_reduction_lower_95": 0.25,
            "H1_wire_byte_reduction_lower_95": 0.25,
            "H1_primary_error_increase_upper_95": 0.02,
            "H1_primary_pointwise_p95_increase_upper_95": 0.01,
            "H1_detection_delay_increase_steps_upper_95": 5.0,
            "H2_primary_error_advantage_lower_95": 0.0,
            "H2_practical_primary_error_advantage": 0.001,
            "H3_relative_service_degradation_upper_95": 0.02,
            "H3_harmful_action_rate_degradation_upper_95": 0.02,
            "H3_reward_degradation_upper_95": 0.02,
        },
        "progression": {
            "H1_required_for_holdout": True,
            "H3_required_for_holdout": True,
            "H2_required_for_entropy_specific_claim_only": True,
            "all_five_training_seeds_must_complete": True,
            "no_collapsed_training_seed": True,
            "validation_run_once": True,
            "holdout_run_once_only_after_validation": True,
        },
        "statistics": {
            "independent_unit": "environment panel; policy seed is a crossed training factor",
            "bootstrap": "paired hierarchical cluster bootstrap, 10000 fixed-seed replicates",
            "secondary_multiplicity": "Holm familywise correction",
            "validation_panels_per_application": 30,
            "holdout_panels_per_application": 40,
        },
        "compute_projection": {
            "gpu_hours": 0.0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "estimated_cpu_core_hours": 12.0,
            "estimated_git_facing_storage_mib": 1200,
            "incremental_cloud_cost_usd": 0.0,
            "runpod_status": "unreachable; CPU NumPy IPPO is the prespecified implementation",
        },
        "prohibited_claims": [
            "real-human effectiveness", "literal thermodynamics",
            "entropy-specific superiority if H2 fails",
            "communication savings that exclude sketch or operational traffic",
            "confirmatory holdout evidence unless the locked stage actually runs",
        ],
        "development_selection_sha256": sha256_file(selection_path),
        "validation_configuration_sha256": sha256_file(validation_path),
        "holdout_configuration_sha256": sha256_file(holdout_path),
        "development_agent_configuration_sha256": sha256_file(development_agent_path),
        "seed_stability_configuration_sha256": sha256_file(seed_stability_path),
        "ablation_configuration_sha256": sha256_file(ablation_path),
        "source_checksum": source_checksum(repository),
    }
    protocol_path = results_root / "protocol" / "v8_frozen_protocol.json"
    atomic_json(protocol_path, protocol)
    protocol["protocol_sha256"] = sha256_file(protocol_path)
    atomic_json(results_root / "protocol" / "freeze_manifest.json", protocol)
    return protocol


def close_v8_development_no_go(
    repository: Path, results_root: Path,
) -> Dict[str, Any]:
    """Seal the pilot stop without pretending that a formal protocol froze.

    The eligibility rules and candidate set were written prospectively in the
    cited notes/configurations.  This function runs only after the pilot gate
    decision and therefore records provenance; it cannot unlock training,
    validation, or holdout and deliberately does not create
    ``v8_frozen_protocol.json``.
    """
    no_go_path = results_root / "negative_results" / "v8_stop_decision.json"
    no_go = json.loads(no_go_path.read_text(encoding="utf-8"))
    if bool(no_go.get("formal_development_unlocked")):
        raise RuntimeError("V8 no-go closure requires a failed formal-development gate")
    forbidden = (
        results_root / "validation" / "episode_summary.csv",
        results_root / "holdout" / "episode_summary.csv",
        results_root / "training" / "training_summary.json",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("a locked V8 stage contains outcome data")
    required_not_run = (
        results_root / "training" / "NOT_RUN.md",
        results_root / "validation" / "NOT_RUN.md",
        results_root / "holdout" / "NOT_RUN.md",
    )
    if not all(path.exists() for path in required_not_run):
        raise RuntimeError("locked-stage NOT_RUN records are incomplete")

    relative_evidence = (
        "configs/v8_hysteresis_repair_pilot.yaml",
        "configs/v8_hysteresis_repair_pilot_v2.yaml",
        "configs/v8_hysteresis_repair_pilot_v3.yaml",
        "notes/97_v8_hysteresis_repair_pilot_rule.md",
        "notes/98_v8_replacement_formal_development_rule.md",
        "notes/99_v8_hysteresis_repair_pilot_iteration_2.md",
        "notes/100_v8_hysteresis_state_machine_repair.md",
        "notes/101_v8_pilot_no_go_disposition.md",
        str(no_go_path.relative_to(repository)),
        str((results_root / "tables" / "trigger_feasibility.csv").relative_to(repository)),
    )
    evidence_hashes = {}
    for relative in relative_evidence:
        path = repository / relative
        if not path.exists():
            raise RuntimeError("missing V8 development-protocol evidence: %s" % relative)
        evidence_hashes[relative] = sha256_file(path)

    protocol = {
        "development_protocol_version": "v8-development-3.0-no-go",
        "record_type": (
            "post-pilot provenance seal of prospectively written development "
            "rules; not a confirmatory protocol freeze"
        ),
        "parent_v7_commit": "e46b6738231883e92b9b525ab1c3c190e38391e7",
        "results_namespace": "results/entropy_triggered_belief_monitoring_v8",
        "research_stage_closed": "pilot_trigger_feasibility",
        "prospective_candidates": {
            "generalized_011_u8": {
                "tau_on": 0.11, "tau_off": 0.04, "encoding": "uint8_simplex",
            },
            "generalized_0115_u8": {
                "tau_on": 0.115, "tau_off": 0.04, "encoding": "uint8_simplex",
            },
        },
        "shared_trigger_parameters": {
            "weights": {"js": 0.45, "entropy_spectrum": 0.25,
                        "confidence": 0.15, "age": 0.15},
            "q_family": [0.5, 1.0, 1.5, 2.0, 3.0],
            "cooldown_steps": 2,
            "maximum_silence_steps": 30,
            "partition_recovery_refresh": True,
            "maximum_hops": 2,
            "information_score_excludes_age_for_off_latch_release": True,
            "high_excursion_may_transmit_while_latch_active": True,
        },
        "prospective_gate": {
            "unit": "application-specific development panel",
            "minimum_information_score_fraction": 0.05,
            "maximum_pre_disruption_noninitial_transmission_rate": 0.10,
            "both_applications_required": True,
            "if_neither_candidate_passes": (
                "stop before replacement formal development, multi-seed "
                "training, validation, and holdout"
            ),
        },
        "observed_disposition": {
            "status": "fail_stop",
            "reason": no_go["stop_reason"],
            "formal_development_unlocked": False,
            "multi_seed_training_unlocked": False,
            "validation_unlocked": False,
            "holdout_unlocked": False,
            "confirmatory_claims_supported": False,
        },
        "primary_hypotheses": {
            "H1": "not formally tested",
            "H2": "not formally tested; no strongest comparator frozen",
            "H3": "not tested; five-seed training remained locked",
        },
        "wire_protocol": {
            "schema_version": 1,
            "framing": "TBV8 deterministic big-endian binary frame plus CRC32",
            "primary_pilot_encoding": "uint8 simplex, largest-remainder mass=255",
            "accounting": "exact serialized length including header and integrity bytes",
        },
        "compute_caps": {
            "maximum_gpu_hours": 50.0,
            "maximum_incremental_cloud_cost_usd": 40.0,
            "observed_gpu_hours": 0.0,
            "observed_llm_calls": 0,
            "observed_cloud_cost_usd": 0.0,
        },
        "source_checksum": source_checksum(repository),
        "evidence_sha256": evidence_hashes,
    }
    protocol_path = results_root / "protocol" / "v8_development_protocol_no_go.json"
    atomic_json(protocol_path, protocol)
    manifest = {
        "development_protocol_version": protocol["development_protocol_version"],
        "status": "closed_no_go_before_formal_freeze",
        "development_protocol_path": str(protocol_path.relative_to(repository)),
        "development_protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256_scope": "development_protocol_sha256 identifies the sealed protocol body",
        "source_checksum": protocol["source_checksum"],
        "parent_v7_commit": protocol["parent_v7_commit"],
        "formal_source_commit": "not created; progression stopped before formal freeze",
        "validation_manifest_created": False,
        "holdout_manifest_created": False,
    }
    # Keep this compatibility name for the reporting layer while making the
    # checksum scope explicit and non-self-referential.
    manifest["manifest_sha256"] = manifest["development_protocol_sha256"]
    atomic_json(results_root / "protocol" / "v8_development_stop_manifest.json", manifest)
    return manifest
