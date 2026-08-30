#!/usr/bin/env python3
"""Verify frozen protocol identity and the consolidated publication source."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from thermoagent.statmech_llm.workflow import verify_consolidated_source  # noqa: E402


def main() -> int:
    result = verify_consolidated_source(REPOSITORY)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
