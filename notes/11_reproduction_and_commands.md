# Reproduction and commands

Run from `/workspace/ThermoAgent` on the execution pod unless stated otherwise.

```bash
./scripts/setup-runpod.sh
./scripts/run-tests.sh
./scripts/run-calibration.sh
./scripts/train-policies.sh
./scripts/run-sweep.sh configs/smoke_real.yaml
./scripts/run-sweep.sh configs/pilot.yaml
./scripts/run-sweep.sh configs/main.yaml
./scripts/run-sweep.sh configs/ablations.yaml
./scripts/run-sweep.sh configs/holdout.yaml
./scripts/replay-results.sh
./scripts/analyze-results.sh
./scripts/generate-figures.sh
./results/reproducibility/tools/polish-figures.sh
.venv/bin/python -m thermoagent mark-visual-qa \
  --results results \
  --reviewer "<reviewer>" \
  --note "All rendered previews inspected at final size"
./scripts/rebuild-results.sh
./results/reproducibility/tools/rebuild-final-results.sh
```

Set `HF_HOME=/workspace/.cache/huggingface` and
`THERMO_CACHE=/workspace/.cache/thermoagent`; the provided setup/sweep scripts
do so automatically. Remote source synchronization is non-deleting and
credential-filtered. The direct port changes when a pod mapping changes, so
operators may set `THERMO_REMOTE_HOST`, `THERMO_REMOTE_PORT`,
`THERMO_REMOTE_IDENTITY`, and `THERMO_REMOTE_KNOWN_HOSTS` locally.

`replay-results.sh` reconstructs the quantitative simulator state from each
recorded post-freeze event ledger before outcome analysis. `analyze-results.sh`
writes episode-level processed data, paired statistics, monitoring summaries,
tables, compute accounting, and `INDEX.csv`. `generate-figures.sh` creates all
paper-facing vector PDFs and PNG previews, mechanically validates that each PDF
opens, contains embedded font information, and renders, then rebuilds the
artifact index. The separate `mark-visual-qa` command records a human-readable
review only after every rendered preview has actually been inspected.

The complete rebuild entry point reruns replay, aggregation, figure generation,
mechanical PDF validation, and indexing from retained raw ledgers. It does not
rerun LLM episodes. All of these entry points passed an isolated 128-episode
pre-freeze rehearsal; final invocations use the immutable post-freeze results.

`results/reproducibility/tools/rebuild-final-results.sh` preserves the frozen
rebuild, then applies the
separately documented presentation-only polish required by final visual QA.
Manual inspection is intentionally never auto-certified: after examining all
ten previews, run `mark-visual-qa` and then rebuild the index with:

```bash
.venv/bin/python -m thermoagent index --results results
```

The combined final replay report is
`results/reproducibility/replay_report.json` (1,096/1,096 passed). The three
sweep summaries with measured including-load times and tokens are
`results/manifests/{main,ablations,holdout}_sweep.json`.
