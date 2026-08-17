"""Command-line entry points for V10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .figures import generate_figures, validate_pdfs
from .llm_analysis import analyze_qwen_formal
from .llm_experiments import freeze_qwen_protocol, run_qwen_formal, run_qwen_message_pilot, run_qwen_pilot
from .reporting import build_clean_export, build_summary, record_test_summary
from .workflow import analyze, audit_v9, freeze, run_development, run_formal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("audit", "development", "freeze", "formal", "qwen-pilot", "qwen-message-pilot", "qwen-freeze", "qwen-formal", "qwen-analyze", "analyze", "figures", "qa", "qa-reviewed", "summary", "export", "test-summary"),
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    if args.command == "audit":
        result = {"rows": len(audit_v9(repository))}
    elif args.command == "development":
        result = run_development(repository)
    elif args.command == "freeze":
        result = freeze(repository)
    elif args.command == "formal":
        result = run_formal(repository)
    elif args.command == "qwen-pilot":
        result = run_qwen_pilot(repository)
    elif args.command == "qwen-message-pilot":
        result = run_qwen_message_pilot(repository)
    elif args.command == "qwen-freeze":
        result = freeze_qwen_protocol(repository)
    elif args.command == "qwen-formal":
        result = run_qwen_formal(repository)
    elif args.command == "qwen-analyze":
        result = analyze_qwen_formal(repository)
    elif args.command == "analyze":
        result = analyze(repository)
    elif args.command == "figures":
        result = {"figures": generate_figures(repository)}
    elif args.command == "qa":
        result = validate_pdfs(repository, manual_reviewed=False)
    elif args.command == "qa-reviewed":
        result = validate_pdfs(repository, manual_reviewed=True)
    elif args.command == "summary":
        result = build_summary(repository)
    elif args.command == "export":
        result = build_clean_export(repository)
    else:
        if args.junit is None:
            parser.error("--junit is required for test-summary")
        result = record_test_summary(args.junit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
