"""Command-line entry points for the staged V13 workflow."""

from __future__ import annotations

import argparse
import json

from .analysis import analyze_formal
from .experiment import run_engineering_pilot, run_formal_experiment
from .workflow import artifact_root, freeze_protocol, load_yaml, repository_root


def main() -> None:
    parser = argparse.ArgumentParser(description="ThermoAgent V13 collective statistical mechanics")
    parser.add_argument("command", choices=("pilot", "freeze", "formal", "replay", "analyze", "figures", "report", "pdf-qa", "verify"))
    args = parser.parse_args()
    repository = repository_root()
    if args.command == "pilot":
        result = run_engineering_pilot(repository)
    elif args.command == "freeze":
        settings = load_yaml(repository / "configs/statmech_v13/engineering.yaml")["pilot"]
        summary_path = artifact_root() / "pilot" / str(settings["attempt_id"]) / "summary.json"  # type: ignore[index]
        if not summary_path.exists():
            raise RuntimeError("V13 engineering pilot has not completed")
        pilot = json.loads(summary_path.read_text(encoding="utf-8"))
        if not bool(pilot["engineering_passed"]):
            raise RuntimeError("engineering estimability did not pass; a prospective amendment is required")
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
