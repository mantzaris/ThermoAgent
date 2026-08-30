"""Command-line interface for the final JSTAT study workflow."""

from __future__ import annotations

import argparse
import json

from .workflow import repository_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThermoAgent cross-model memory and quench study"
    )
    parser.add_argument(
        "command",
        choices=(
            "pilot",
            "freeze",
            "formal",
            "replay",
            "analyze",
            "surrogate",
            "figures",
            "report",
            "pdf-qa",
            "pdf-qa-record",
            "verify",
        ),
    )
    parser.add_argument("--model", choices=("qwen", "granite"))
    parser.add_argument("--manual-status", choices=("passed", "failed"))
    parser.add_argument("--manual-notes", default="")
    args = parser.parse_args()
    repository = repository_root()
    if args.command == "pilot":
        if args.model is None:
            raise SystemExit("pilot requires --model qwen or --model granite")
        from .experiment import run_engineering_pilot

        result = run_engineering_pilot(repository, args.model)
    elif args.command == "freeze":
        from .experiment import freeze_protocol

        result = freeze_protocol(repository)
    elif args.command == "formal":
        if args.model is None:
            raise SystemExit("formal requires --model qwen or --model granite")
        from .experiment import run_formal_model

        result = run_formal_model(repository, args.model)
    elif args.command == "replay":
        from .replay import replay_formal

        result = replay_formal(repository)
    elif args.command == "analyze":
        from .analysis import analyze_formal

        result = analyze_formal(repository)
    elif args.command == "surrogate":
        from .surrogate import dense_surrogate_size_quench, corrected_quench_out_of_sample_comparison

        result = {
            "corrected_quench_out_of_sample": corrected_quench_out_of_sample_comparison(
                repository
            ),
            "size_sensitivity": dense_surrogate_size_quench(repository),
        }
    elif args.command == "figures":
        from .figures import generate_figures

        result = generate_figures(repository)
    elif args.command == "report":
        from .reporting import build_results

        result = build_results(repository)
    elif args.command == "pdf-qa":
        from .reporting import validate_pdfs

        result = validate_pdfs(repository)
    elif args.command == "pdf-qa-record":
        if args.manual_status is None:
            raise SystemExit("pdf-qa-record requires --manual-status passed or failed")
        from .reporting import record_manual_pdf_qa

        result = record_manual_pdf_qa(
            repository, args.manual_status, args.manual_notes
        )
    else:
        from .reporting import verify_package

        result = verify_package(repository)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
