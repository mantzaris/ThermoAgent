#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/human_operator_v4"
mkdir -p "$results_dir/logs" "$results_dir/reproducibility"
report_xml="$results_dir/reproducibility/pytest_v4.xml"
if python3 -c 'import pytest' >/dev/null 2>&1; then
  test_command=(python3 -m pytest)
elif command -v pytest >/dev/null 2>&1; then
  test_command=(pytest)
else
  echo "pytest is not available in the active project environment" >&2
  exit 2
fi
set +e
(cd "$repo_dir" && "${test_command[@]}" -q --junitxml="$report_xml") \
  2>&1 | tee "$results_dir/logs/tests.log"
test_status="${PIPESTATUS[0]}"
set -e
python3 - "$report_xml" "$results_dir/reproducibility/test_report.json" <<'PY'
import json, sys
from pathlib import Path
from xml.etree import ElementTree
source, target = map(Path, sys.argv[1:])
root = ElementTree.parse(source).getroot()
suite = root if root.tag == "testsuite" else root.find("testsuite")
record = {
    "tests": int(suite.attrib.get("tests", 0)),
    "failed": int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)),
    "skipped": int(suite.attrib.get("skipped", 0)),
    "seconds": float(suite.attrib.get("time", 0.0)),
    "scope": "complete repository test suite",
}
target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
exit "$test_status"
