#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/human_operator_v5"
mkdir -p "$results_dir/logs" "$results_dir/reproducibility"
cd "$repo_dir"
python3 -m pytest -q --junitxml="$results_dir/reproducibility/test_results.xml" \
  2>&1 | tee "$results_dir/logs/tests.log"
python3 -c '
from pathlib import Path
from xml.etree import ElementTree
from thermoagent.v5_experiments import atomic_json
root = ElementTree.parse(Path("results/human_operator_v5/reproducibility/test_results.xml")).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
total = sum(int(s.attrib.get("tests", 0)) for s in suites)
failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
atomic_json(Path("results/human_operator_v5/reproducibility/test_summary.json"), {
    "source": "reproducibility/test_results.xml", "total_tests": total,
    "passed_tests": total - failures - errors - skipped, "failed_tests": failures,
    "errors": errors, "skipped_tests": skipped,
    "passed": failures == 0 and errors == 0,
})
'
