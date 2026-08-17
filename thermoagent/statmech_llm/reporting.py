"""Compact reproducibility summaries and lean JSTAT export for V10."""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from .workflow import (
    _atomic_csv,
    _atomic_json,
    artifact_root,
    clean_export_root,
    environment_manifest,
    sha256_file,
    source_checksum,
    utc_now,
)


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def external_manifest(repository: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    root = artifact_root()
    remote_manifest = root / "qwen/remote_external_artifacts.csv"
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".tmp" or path.name.startswith("."):
            continue
        if path == remote_manifest:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "storage_root": str(root),
                "availability": "local external artifact",
            }
        )
    if remote_manifest.exists():
        remote_rows = pd.read_csv(remote_manifest).to_dict(orient="records")
        known = {(str(row["relative_path"]), str(row["sha256"])) for row in rows}
        for row in remote_rows:
            key = (str(row["relative_path"]), str(row["sha256"]))
            if key in known:
                continue
            rows.append(
                {
                    "relative_path": str(row["relative_path"]),
                    "size_bytes": int(row["size_bytes"]),
                    "sha256": str(row["sha256"]),
                    "storage_root": str(row["storage_root"]),
                    "availability": "existing RunPod external artifact",
                }
            )
    rows.sort(key=lambda row: (str(row["storage_root"]), str(row["relative_path"])))
    destination = repository / "results/llm_agent_entropy_v10/reproducibility/external_artifacts.csv"
    _atomic_csv(rows, destination)
    return rows


def summarize_qwen_pilots(repository: Path) -> Dict[str, object]:
    """Build compact pilot evidence without copying raw prompts or completions."""

    root = artifact_root()
    attempts = [
        (
            "evidence_attempt_2",
            root / "qwen/pilot_history/attempt2_confounded_design/summary.json",
            "completed; evidence field aliased with display order; excluded from gate",
        ),
        (
            "evidence_attempt_3",
            root / "qwen/pilot/summary.json",
            "completed; fully crossed evidence by order; passed under the pre-message-clarification prompt",
        ),
        (
            "message_attempt_1",
            root / "qwen/message_pilot_history/attempt1_confounded_prior/summary.json",
            "completed; every prior was plan_left; excluded from gate",
        ),
        (
            "message_attempt_2",
            root / "qwen/message_pilot_history/attempt2_balanced_prior_prompt/summary.json",
            "completed; balanced priors; message gate failed",
        ),
        (
            "message_attempt_3",
            root / "qwen/message_pilot/summary.json",
            "completed after one prompt clarification; message gate failed",
        ),
    ]
    rows: List[Dict[str, object]] = [
        {
            "attempt": "evidence_attempt_1",
            "status": "failed before a model decision",
            "scientific_disposition": "retained technical failure: deterministic cuBLAS workspace variable absent",
            "decisions": 0,
            "model_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "generation_latency_seconds": 0.0,
            "first_pass_validity": "",
            "after_repair_validity": "",
            "gate_passed": False,
        }
    ]
    for name, path, disposition in attempts:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        accounting = payload["provider_accounting"]
        rows.append(
            {
                "attempt": name,
                "status": "completed",
                "scientific_disposition": disposition,
                "decisions": int(payload.get("calls", payload.get("decisions", 0))),
                "model_calls": int(accounting["model_calls"]),
                "prompt_tokens": int(accounting["prompt_tokens"]),
                "generated_tokens": int(accounting["generated_tokens"]),
                "generation_latency_seconds": float(accounting["latency_seconds"]),
                "first_pass_validity": float(payload["first_pass_validity"]),
                "after_repair_validity": float(payload["after_repair_validity"]),
                "gate_passed": bool(payload.get("pilot_gate_passed", payload.get("message_pilot_gate_passed", False))),
            }
        )
    _atomic_csv(rows, repository / "results/llm_agent_entropy_v10/tables/qwen_pilot_attempts.csv")

    evidence_path = root / "qwen/pilot/decisions.csv"
    message_path = root / "qwen/message_pilot/decisions.csv"
    figure_rows: List[Dict[str, object]] = []

    def append_figure_row(panel: str, x: float, condition: str, choices: pd.Series) -> None:
        count = int(len(choices))
        successes = int(np.sum(choices.to_numpy(int) > 0))
        probability = successes / float(count)
        z = 1.959963984540054
        denominator = 1.0 + z * z / count
        center = (probability + z * z / (2.0 * count)) / denominator
        radius = z * np.sqrt(probability * (1.0 - probability) / count + z * z / (4.0 * count * count)) / denominator
        figure_rows.append(
            {
                "panel": panel,
                "x": x,
                "condition": condition,
                "mean_right_choice": probability,
                "wilson_ci_low": min(probability, max(0.0, center - radius)),
                "wilson_ci_high": max(probability, min(1.0, center + radius)),
                "independent_decisions": count,
            }
        )

    if evidence_path.exists():
        evidence = pd.read_csv(evidence_path)
        for key, part in evidence.groupby(["local_field", "option_order_right_first"], sort=True):
            append_figure_row(
                "private_evidence",
                float(key[0]),
                "right option first" if int(key[1]) else "left option first",
                part["belief_spin"],
            )
    if message_path.exists():
        message = pd.read_csv(message_path)
        for key, part in message.groupby(["previous_belief", "message_spin"], sort=True):
            append_figure_row(
                "delivered_message",
                int(key[1]),
                "prior right" if int(key[0]) > 0 else "prior left",
                part["belief_spin"],
            )
    if figure_rows:
        _atomic_csv(figure_rows, repository / "results/llm_agent_entropy_v10/figures/source_data/figure_08_qwen_pilot.csv")

    total_decisions = sum(int(row["decisions"]) for row in rows)
    total_calls = sum(int(row["model_calls"]) for row in rows)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    generated_tokens = sum(int(row["generated_tokens"]) for row in rows)
    generation_seconds = sum(float(row["generation_latency_seconds"]) for row in rows)
    final_message = json.loads((root / "qwen/message_pilot/summary.json").read_text(encoding="utf-8"))
    payload = {
        "generated_at": utc_now(),
        "status": "pilot qualification stopped prospectively; formal LLM study not run",
        "completed_decisions": total_decisions,
        "model_calls_including_repairs": total_calls,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "measured_generation_latency_seconds": generation_seconds,
        "measured_generation_gpu_hours": generation_seconds / 3600.0,
        "estimated_allocated_gpu_hours_including_five_model_loads": 0.23,
        "estimated_cost_usd_at_0_34_per_hour": 0.23 * 0.34,
        "estimated_cost_usd_at_0_69_per_hour": 0.23 * 0.69,
        "final_message_response_difference": final_message["right_minus_left_message_response"],
        "final_message_switch_fraction": final_message["paired_choice_switch_fraction"],
        "required_message_response_difference": 0.20,
        "formal_qwen_decisions": 0,
        "formal_qwen_unlocked": False,
        "model": final_message["environment"],
    }
    _atomic_json(payload, repository / "results/llm_agent_entropy_v10/tables/qwen_pilot_summary.json")
    return payload


def build_summary(repository: Path) -> Dict[str, object]:
    results = repository / "results/llm_agent_entropy_v10"
    principal = json.loads((results / "tables/principal_results.json").read_text(encoding="utf-8"))
    freeze = json.loads((artifact_root() / "formal/freeze_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((artifact_root() / "formal/completion_manifest.json").read_text(encoding="utf-8"))
    qwen = summarize_qwen_pilots(repository)
    principal["llm_stage"] = {
        "status": "pilot_no_go_formal_not_unlocked",
        "reason": "the final balanced delivered-message counterfactual produced zero belief switches and failed the frozen gate",
        "completed_pilot_decisions": qwen["completed_decisions"],
        "model_calls_including_repairs": qwen["model_calls_including_repairs"],
        "prompt_tokens": qwen["prompt_tokens"],
        "generated_tokens": qwen["generated_tokens"],
        "measured_generation_gpu_hours": qwen["measured_generation_gpu_hours"],
        "estimated_allocated_gpu_hours": qwen["estimated_allocated_gpu_hours_including_five_model_loads"],
        "estimated_incremental_cost_usd_range": [
            qwen["estimated_cost_usd_at_0_34_per_hour"],
            qwen["estimated_cost_usd_at_0_69_per_hour"],
        ],
        "final_message_response_difference": qwen["final_message_response_difference"],
        "qualification_requirement": qwen["required_message_response_difference"],
        "formal_qwen_decisions": 0,
    }
    principal["claims"].update(
        {
            "H5_llm_local_policy": "pilot evidence only: controlled private evidence changed choices, but the final prompt was not formally calibrated",
            "H6_llm_time_reversal_asymmetry": "not tested: delivered-message qualification gate failed before formal execution",
            "H7_llm_replication": "not tested: formal graph, prompt, seed, and size study was prospectively locked",
        }
    )
    principal["llm_pilot_amendment_sha256_current"] = sha256_file(
        repository / "configs/statmech_v10/llm_pilot_amendment.yaml"
    )
    principal["llm_updated_at"] = utc_now()
    _atomic_json(principal, results / "tables/principal_results.json")
    artifacts = external_manifest(repository)
    pdf_qa_path = results / "reproducibility/pdf_qa.json"
    pdf_qa = json.loads(pdf_qa_path.read_text(encoding="utf-8")) if pdf_qa_path.exists() else None
    manuscript_qa_path = results / "reproducibility/manuscript_qa.json"
    manuscript_qa = (
        json.loads(manuscript_qa_path.read_text(encoding="utf-8"))
        if manuscript_qa_path.exists()
        else None
    )
    test_path = artifact_root() / "tests/test_summary.json"
    test_summary = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {
        "status": "summary not yet generated",
    }
    payload = {
        "generated_at": utc_now(),
        "provenance": {
            "v9_remote_commit": "8e8315d25684a1c582c6a7b46fbb5786bc3f0557",
            "v10_branch": "llm-agent-entropy-production-v10",
            "v10_committed": False,
            "v10_pushed": False,
        },
        "freeze": freeze,
        "formal_completion": completion,
        "principal_results": principal,
        "environment": environment_manifest(),
        "tests": test_summary,
        "pdf_qa": pdf_qa,
        "manuscript_qa": manuscript_qa,
        "external_artifact_root": str(artifact_root()),
        "external_artifact_count": len(artifacts),
        "external_artifact_bytes": sum(int(row["size_bytes"]) for row in artifacts),
        "repository_v10_bytes": _tree_size(results)
        + _tree_size(repository / "thermoagent/statmech_llm")
        + _tree_size(repository / "tests/statmech_v10")
        + _tree_size(repository / "configs/statmech_v10")
        + _tree_size(repository / "paper/jstat_v10"),
        "scientific_source_sha256_current": source_checksum(repository),
        "llm_execution": {
            "status": qwen["status"],
            "reason": "final delivered-message response was 0.00 against the frozen 0.20 qualification gate",
            "qwen_decisions": qwen["completed_decisions"],
            "qwen_calls": qwen["model_calls_including_repairs"],
            "prompt_tokens": qwen["prompt_tokens"],
            "generated_tokens": qwen["generated_tokens"],
            "measured_generation_gpu_hours": qwen["measured_generation_gpu_hours"],
            "estimated_allocated_gpu_hours": qwen["estimated_allocated_gpu_hours_including_five_model_loads"],
            "estimated_cost_usd_range": [
                qwen["estimated_cost_usd_at_0_34_per_hour"],
                qwen["estimated_cost_usd_at_0_69_per_hour"],
            ],
            "formal_qwen_decisions": 0,
        },
    }
    _atomic_json(payload, results / "reproducibility/summary.json")
    return payload


def record_test_summary(junit_path: Path) -> Dict[str, object]:
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    payload = {
        "generated_at": utc_now(),
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "time_seconds": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
        "junit_path_external": str(Path(junit_path).resolve()),
        "junit_sha256": sha256_file(Path(junit_path)),
    }
    _atomic_json(payload, artifact_root() / "tests/test_summary.json")
    return payload


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "*.aux",
                "*.bbl",
                "*.blg",
                "*.fdb_latexmk",
                "*.fls",
                "*.log",
                "*.out",
                "*.synctex.gz",
            ),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_clean_export(repository: Path) -> Dict[str, object]:
    destination = clean_export_root()
    repository = repository.resolve()
    if destination == repository or repository in destination.parents or destination == Path("/"):
        raise ValueError("unsafe export destination")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    included = [
        Path("LICENSE"),
        Path("pyproject.toml"),
        Path("thermoagent/__init__.py"),
        Path("thermoagent/statmech"),
        Path("thermoagent/statmech_llm"),
        Path("configs/statmech_v10"),
        Path("tests/statmech_v9"),
        Path("tests/statmech_v10"),
        Path("results/llm_agent_entropy_v10"),
        Path("paper/jstat_v10"),
        Path("notes/v10_research_log.md"),
    ]
    included.extend(sorted(path.relative_to(repository) for path in repository.glob("scripts/run-statmech-v10-*.sh")))
    for relative in included:
        source = repository / relative
        if not source.exists():
            raise FileNotFoundError("clean-export input missing: %s" % relative)
        _copy_path(source, destination / relative)
    files = [path for path in destination.rglob("*") if path.is_file()]
    inventory = [
        {
            "relative_path": str(path.relative_to(destination)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    _atomic_csv(inventory, destination / "EXPORT_INVENTORY.csv")
    readme = destination / "README.md"
    with readme.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "# ThermoAgent JSTAT V10 clean export\n\n"
            "This is a lean, non-Git publication export. It contains the V10 source, frozen "
            "configuration, focused V9/V10 tests, aggregate tables, vector figures, manuscript, "
            "and external-artifact checksum manifest. Raw Qwen transcripts and transition records "
            "are intentionally excluded and are reproducible through the documented commands.\n"
        )
    payload = {
        "path": str(destination),
        "created_at": utc_now(),
        "file_count": len(inventory) + 2,
        "size_bytes": _tree_size(destination),
        "largest_files": sorted(inventory, key=lambda row: int(row["size_bytes"]), reverse=True)[:10],
        "git_repository_initialized": (destination / ".git").exists(),
    }
    _atomic_json(payload, destination / "EXPORT_SUMMARY.json")
    return payload
