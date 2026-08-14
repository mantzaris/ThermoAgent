# Reproduction and compute accounting

## Local deterministic sequence

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

Dashboard replay:

```bash
./scripts/run-human-operator-dashboard.sh --episode \
  results/human_operator_v3/raw/development_trigger_candidate_n10_v4/<run-id>/episode.json
```

## Existing RunPod workflow

Filtered source sync uses `scripts/runpod-sync.sh`; v3 calibration/provenance
uses `scripts/runpod-sync-v3-controls.sh`; v3-only retrieval uses
`scripts/runpod-fetch-v3-results.sh`. Remote execution uses
`scripts/runpod-exec.sh` in `/workspace/ThermoAgent`. SSH keys, Git metadata,
results, caches, virtual environments, environment files, and credentials are
excluded from the source sync.

The qualification command was:

```bash
THERMO_HUMAN_REAL_STAGE=development_real_llm_actionability_retry1 \
  ./scripts/run-human-operator-real-llm-actionability.sh
```

## Accounting

- completed v3 manifests/episodes: 817;
- Qwen calls: 589;
- prompt/generated tokens: 1,084,097 / 32,250;
- manifest episode GPU-hours: 0.1227669;
- conservative three-initialization overhead: 0.0666667 hours;
- estimated total: 0.1894335 single-GPU hours;
- estimated cost at $0.34/hour: $0.0644;
- training/validation/holdout GPU-hours: 0.

The initial projection was 0.172 hours per four-episode qualification including
a 15% reserve. Total use remained far below the 40-hour cap.

