"""Command line for V9 statistical-mechanics workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import freeze_manifest, run_formal, run_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "freeze", "formal", "analyze", "figures", "qa", "export"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    if arguments.command == "pilot":
        result = run_pilot(repository)
    elif arguments.command == "freeze":
        result = freeze_manifest(repository)
    elif arguments.command == "formal":
        result = run_formal(repository)
    elif arguments.command == "analyze":
        from .reporting import analyze_formal

        result = analyze_formal(repository)
    elif arguments.command == "figures":
        from .figures import generate_figures

        result = generate_figures(repository)
    elif arguments.command == "qa":
        from .figures import validate_pdfs

        result = validate_pdfs(repository)
    else:
        from .reporting import build_clean_export

        result = build_clean_export(repository)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
