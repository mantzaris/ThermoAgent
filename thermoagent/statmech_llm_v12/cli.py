"""Command-line entry points for the staged V12 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_formal
from .experiment import run_engineering_pilot, run_formal_experiment
from .workflow import artifact_root, freeze_protocol, repository_root


def main() -> None:
    parser = argparse.ArgumentParser(description="ThermoAgent V12 stochastic-thermodynamics workflow")
    parser.add_argument(
        "command",
        choices=("pilot", "freeze", "formal", "replay", "analyze", "figures", "report", "pdf-qa", "verify"),
    )
    args = parser.parse_args()
    repository = repository_root()
    if args.command == "pilot":
        result = run_engineering_pilot(repository)
    elif args.command == "freeze":
        engineering = repository / "configs/statmech_v12/engineering.yaml"
        from .workflow import load_yaml

        settings = load_yaml(engineering)["pilot"]
        summary_path = artifact_root() / "pilot" / str(settings.get("attempt_id", "pilot_attempt1")) / "summary.json"
        if not summary_path.exists():
            raise RuntimeError("engineering pilot has not completed")
        pilot = json.loads(summary_path.read_text(encoding="utf-8"))
        if not bool(pilot["estimability"]["estimability_passed"]):
            raise RuntimeError("pilot estimability targets did not pass; document an engineering amendment first")
        result = freeze_protocol(repository, pilot)
    elif args.command == "formal":
        result = run_formal_experiment(repository)
    elif args.command == "replay":
        from .replay import replay_formal

        result = replay_formal(repository)
    elif args.command == "analyze":
        result = analyze_formal(repository)
    elif args.command == "figures":
        from .figures import generate_figures

        result = generate_figures(repository)
    elif args.command == "report":
        from .reporting import build_results

        result = build_results(repository)
    elif args.command == "pdf-qa":
        from .reporting import validate_pdfs

        result = validate_pdfs(repository)
    else:
        from .reporting import verify_package

        result = verify_package(repository)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
