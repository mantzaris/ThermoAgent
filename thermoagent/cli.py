"""Command-line entry point for tests, training, experiments, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermoagent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="fit macrostate thresholds on nominal training episodes")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--seeds", default="101,102,103,104,105")
    calibrate.add_argument("--horizon", type=int, default=24)

    monitor = subparsers.add_parser("select-monitor", help="pilot requested macrostate estimator formulations")
    monitor.add_argument("--calibration", type=Path, required=True)
    monitor.add_argument("--output", type=Path, required=True)
    monitor.add_argument("--seeds", default="501,502,503")
    monitor.add_argument("--horizon", type=int, default=18)

    train = subparsers.add_parser("train-policy", help="train the PPO coordination metapolicy")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--variant", choices=("thermo", "no_entropy"), required=True)
    train.add_argument("--episodes", type=int, default=96)
    train.add_argument("--seed", type=int, default=3001)
    train.add_argument("--calibration", type=Path)
    train.add_argument("--log", type=Path, required=True)

    sweep = subparsers.add_parser("sweep", help="run a restartable configured experiment matrix")
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--results", type=Path, default=Path("results"))
    sweep.add_argument("--root", type=Path, default=Path("."))
    sweep.add_argument("--limit", type=int)

    analyze = subparsers.add_parser("analyze", help="aggregate episodes and run episode-level statistics")
    analyze.add_argument("--results", type=Path, default=Path("results"))

    index = subparsers.add_parser("index", help="rebuild results/INDEX.csv with checksums")
    index.add_argument("--results", type=Path, default=Path("results"))

    figures = subparsers.add_parser("figures", help="generate all publication PDF figures and PNG previews")
    figures.add_argument("--results", type=Path, default=Path("results"))

    pdfs = subparsers.add_parser("validate-pdfs", help="open, inspect fonts, and render every PDF")
    pdfs.add_argument("--results", type=Path, default=Path("results"))

    visual = subparsers.add_parser("mark-visual-qa", help="record completed inspection of rendered PDF previews")
    visual.add_argument("--results", type=Path, default=Path("results"))
    visual.add_argument("--reviewer", required=True)
    visual.add_argument("--note", required=True)

    capture = subparsers.add_parser("capture-env", help="capture non-secret dependency, hardware, and source provenance")
    capture.add_argument("--results", type=Path, default=Path("results"))
    capture.add_argument("--root", type=Path, default=Path("."))

    replay = subparsers.add_parser("replay", help="replay quantitative state from recorded tool-call ledgers")
    replay.add_argument("--results", type=Path, default=Path("results"))
    replay.add_argument("--stages", nargs="+", default=["main", "ablations", "holdout"])
    replay.add_argument("--run-id-contains", action="append", default=[])
    replay.add_argument("--report-name", default="replay_report.json")

    model = subparsers.add_parser("model-smoke", help="load the frozen model and validate batched CUDA JSON planning")
    model.add_argument("--output", type=Path, default=Path("results/smoke/model_smoke.json"))
    model.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    model.add_argument("--revision", required=True)
    model.add_argument("--max-new-tokens", type=int, default=128)

    agentic = subparsers.add_parser("agentic-smoke", help="run Stage 1 real-LLM negotiation and both applications")
    agentic.add_argument("--output-dir", type=Path, default=Path("results/smoke/stage1"))
    agentic.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    agentic.add_argument("--revision", required=True)

    freeze = subparsers.add_parser("freeze-protocol", help="write a non-overwriting checksum lock before main evaluation")
    freeze.add_argument("--output", type=Path, default=Path("results/reproducibility/protocol_freeze.json"))
    freeze.add_argument("--root", type=Path, default=Path("."))
    freeze.add_argument("files", nargs="+", type=Path)

    verify = subparsers.add_parser("verify-protocol", help="verify every checksum in the frozen main protocol")
    verify.add_argument("--freeze", type=Path, default=Path("results/reproducibility/protocol_freeze.json"))
    verify.add_argument("--root", type=Path, default=Path("."))

    profile = subparsers.add_parser("profile-budget", help="project main compute and cost from paired-v5 pilot data")
    profile.add_argument("--results", type=Path, default=Path("results"))
    profile.add_argument("--output", type=Path, default=Path("results/reproducibility/prelaunch_budget.json"))
    profile.add_argument("--configs", nargs="+", type=Path, required=True)
    profile.add_argument("--hourly-rates", default="0.34,0.69")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        from .experiments import calibrate_nominal

        seeds = [int(value) for value in args.seeds.split(",") if value]
        value = calibrate_nominal(args.output, seeds, args.horizon)
    elif args.command == "select-monitor":
        from .experiments import select_monitor_formulation

        seeds = [int(value) for value in args.seeds.split(",") if value]
        value = select_monitor_formulation(args.calibration, args.output, seeds, args.horizon)
    elif args.command == "train-policy":
        from .experiments import train_policy

        value = train_policy(args.output, args.variant, args.episodes, args.seed, args.calibration, args.log)
    elif args.command == "sweep":
        from .experiments import run_matrix

        rows = run_matrix(args.config, args.root.resolve(), args.results, args.limit)
        value = {"episodes": len(rows), "complete": sum(row.get("status") == "complete" for row in rows), "failed": sum(row.get("status") == "failed" for row in rows)}
    elif args.command == "analyze":
        from .analysis import write_analysis

        value = write_analysis(args.results)
    elif args.command == "index":
        from .analysis import build_index

        value = {"indexed_artifacts": build_index(args.results)}
    elif args.command == "figures":
        from .figures import generate_all

        value = {"figures": generate_all(args.results)}
    elif args.command == "validate-pdfs":
        from .figures import validate_pdfs

        value = validate_pdfs(args.results)
    elif args.command == "mark-visual-qa":
        from .figures import mark_visual_qa

        value = mark_visual_qa(args.results, args.reviewer, args.note)
    elif args.command == "capture-env":
        from .experiments import capture_reproducibility

        value = capture_reproducibility(args.root.resolve(), args.results)
    elif args.command == "replay":
        from .replay import replay_results

        value = replay_results(
            args.results, args.stages, args.run_id_contains, args.report_name
        )
    elif args.command == "model-smoke":
        from .smoke import model_smoke

        value = model_smoke(args.output, args.model, args.revision, args.max_new_tokens)
    elif args.command == "agentic-smoke":
        from .smoke import agentic_smoke

        value = agentic_smoke(args.output_dir, args.model, args.revision)
    elif args.command == "freeze-protocol":
        from .experiments import freeze_protocol

        value = freeze_protocol(args.root.resolve(), args.output, args.files)
    elif args.command == "verify-protocol":
        from .experiments import verify_protocol

        value = verify_protocol(args.root.resolve(), args.freeze)
    elif args.command == "profile-budget":
        from .profiling import profile_budget

        rates = [float(item) for item in args.hourly_rates.split(",") if item]
        value = profile_budget(args.results, args.configs, args.output, rates)
    else:
        raise AssertionError(args.command)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
