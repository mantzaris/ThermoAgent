# V7 exact reproduction commands

Run from a fresh clone at the final V7 branch commit with the documented Python
environment:

```bash
./scripts/run-v7-tests.sh
./scripts/run-v7-complexity-audit.sh
./scripts/run-v7-environment-smoke.sh
./scripts/run-v7-pilots.sh
./scripts/run-v7-pilot-iteration2.sh
./scripts/run-v7-pilot-iteration3.sh
./scripts/replay-v7-results.sh
./scripts/evaluate-v7-gates.sh
./scripts/freeze-v7-protocol.sh
./scripts/run-v7-development.sh
./scripts/evaluate-v7-gates.sh
./scripts/compact-v7-artifacts.sh
./scripts/replay-v7-results.sh
./scripts/generate-v7-figures.sh
./scripts/validate-v7-pdfs.sh
./scripts/build-v7-report.sh
./scripts/index-v7-artifacts.sh
```

The following commands must refuse to run because the frozen formal primary
gate failed:

```bash
./scripts/train-v7-multiseed.sh
./scripts/run-v7-real-qwen.sh
./scripts/run-v7-validation.sh
./scripts/run-v7-holdout.sh
```

Verify compacted payloads, artifact checksums, and stage locks:

```bash
./scripts/compact-v7-artifacts.sh --verify-only
./scripts/replay-v7-results.sh
./scripts/index-v7-artifacts.sh --verify-only
```
