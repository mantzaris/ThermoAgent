# Exact v3 reproduction commands

Run from the repository root:

```bash
./scripts/run-human-operator-tests.sh
./scripts/run-human-operator-diagnostics.sh
./scripts/run-human-operator-development.sh
./scripts/run-human-operator-monitoring.sh
./scripts/replay-human-operator-results.sh
./scripts/analyze-human-operator-results.sh
./scripts/generate-human-operator-figures.sh
./scripts/validate-human-operator-pdfs.sh
```

The complete restartable wrapper is:

```bash
./scripts/rebuild-human-operator-results.sh
```

Remote real-Qwen qualification uses the existing Pod only:

```bash
./scripts/runpod-sync.sh
./scripts/runpod-sync-v3-controls.sh
./scripts/runpod-exec.sh ./scripts/setup-human-operator-runpod.sh
./scripts/runpod-exec.sh env \
  THERMO_HUMAN_REAL_STAGE=development_real_llm_actionability_retry1 \
  ./scripts/run-human-operator-real-llm-actionability.sh
./scripts/runpod-fetch-v3-results.sh
```

SSH endpoint/identity variables are local private operator configuration and
must not be written into Git-facing commands or artifacts.

The validation/training/holdout scripts are intentionally guarded and fail
while `development/gate_status.json` has `holdout_unlocked=false`.
