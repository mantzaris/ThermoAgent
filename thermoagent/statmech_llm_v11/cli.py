"""Command-line entry points for the staged V11 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .formal import analyze_formal_network, expected_formal_decisions, run_formal_network
from .qualification import analyze_qualification_stage, expected_decision_requests, run_qualification_stage
from .workflow import artifact_root, atomic_json, load_yaml, sha256_file, source_tree_checksum, utc_now


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def freeze_manifest(stage: str) -> dict:
    repository = repository_root()
    if stage == "qualification":
        protocol = repository / "configs/statmech_v11/qualification_frozen.yaml"
    elif stage == "formal":
        qualification_path = artifact_root() / "qualification/analysis.json"
        if not qualification_path.exists():
            raise RuntimeError("formal freeze is locked until qualification analysis exists")
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        if not qualification.get("formal_network_unlocked", False):
            raise RuntimeError("formal freeze is locked because qualification did not pass")
        protocol = repository / "configs/statmech_v11/formal_frozen.yaml"
    else:
        raise ValueError("stage must be qualification or formal")
    if not protocol.exists():
        raise RuntimeError("protocol file is absent: %s" % protocol)
    manifest = {
        "stage": stage,
        "frozen_at": utc_now(),
        "protocol_relative_path": protocol.relative_to(repository).as_posix(),
        "protocol_sha256": sha256_file(protocol),
        "source_tree_sha256": source_tree_checksum(repository),
        "starting_v10_commit": "4d372f00837bf75f90882392a92feac87dbc84b2",
    }
    atomic_json(manifest, artifact_root() / "manifests" / (stage + "_freeze.json"))
    return manifest


def estimate_calls(stage: str) -> dict:
    repository = repository_root()
    if stage == "pilot":
        values = load_yaml(repository / "configs/statmech_v11/engineering.yaml")["pilot"]
        decisions = expected_decision_requests(values, False)
    elif stage == "qualification":
        values = load_yaml(repository / "configs/statmech_v11/qualification_frozen.yaml")["qualification"]
        decisions = expected_decision_requests(values, bool(values["include_extended_conditions"]))
    elif stage == "formal":
        values = load_yaml(repository / "configs/statmech_v11/formal_frozen.yaml")["formal"]
        decisions = expected_formal_decisions(values)
    else:
        raise ValueError("unknown stage")
    return {"stage": stage, "decision_requests": decisions, "maximum_model_calls_with_one_repair_each": 2 * decisions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "estimate-pilot",
            "estimate-qualification",
            "estimate-formal",
            "run-pilot",
            "analyze-pilot",
            "freeze-qualification",
            "run-qualification",
            "analyze-qualification",
            "freeze-formal",
            "run-formal",
            "analyze-formal",
            "build-results",
            "generate-figures",
            "validate-pdfs",
        ),
    )
    args = parser.parse_args()
    repository = repository_root()
    if args.command.startswith("estimate-"):
        result = estimate_calls(args.command.split("-", 1)[1])
    elif args.command == "run-pilot":
        result = run_qualification_stage(repository, "pilot")
    elif args.command == "analyze-pilot":
        result = analyze_qualification_stage(repository, "pilot")
    elif args.command == "freeze-qualification":
        result = freeze_manifest("qualification")
    elif args.command == "run-qualification":
        result = run_qualification_stage(repository, "qualification")
    elif args.command == "analyze-qualification":
        result = analyze_qualification_stage(repository, "qualification")
    elif args.command == "freeze-formal":
        result = freeze_manifest("formal")
    elif args.command == "run-formal":
        result = run_formal_network(repository)
    elif args.command == "analyze-formal":
        result = analyze_formal_network(repository)
    elif args.command == "build-results":
        from .reporting import build_repository_aggregates

        result = build_repository_aggregates(repository)
    elif args.command == "generate-figures":
        from .figures import generate_figures

        result = {"generated_pdfs": generate_figures(repository)}
    elif args.command == "validate-pdfs":
        from .pdf_qa import validate_pdfs

        result = validate_pdfs(repository)
    else:
        raise AssertionError("unreachable")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
