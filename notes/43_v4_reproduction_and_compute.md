# V4 reproduction and compute accounting

## Derived-artifact rebuild

From the repository root:

```bash
./scripts/run-human-operator-v4-tests.sh
./scripts/replay-human-operator-v4-results.sh
./scripts/analyze-human-operator-v4-results.sh
./scripts/generate-human-operator-v4-figures.sh
./scripts/validate-human-operator-v4-pdfs.sh
./scripts/build-human-operator-v4-report.sh
```

The wrapper `./scripts/rebuild-human-operator-v4-results.sh` runs the first five
steps; reporting/index generation is run last so checksums describe final
artifacts. None of these commands reruns experimental episodes.

Formal development can be resumed with
`./scripts/run-human-operator-v4-development.sh`, but existing frozen evidence
should not be selectively rerun. The real model qualifier is
`./scripts/run-human-operator-v4-real-qwen.sh` on the established RunPod
environment. Validation/training/holdout scripts are intentionally absent or
guarded because the gate report forbids those stages.

## Compute

- Deterministic formal development: 1,584 CPU episodes, summed recorded runner
  time 331.73 seconds, no GPU/LLM calls.
- Real Qwen: six calls, 3,942 prompt tokens, 328 generated tokens, 6.879 seconds
  measured generation latency.
- Model load progress log: 29.44 seconds.
- Accounted Qwen GPU time: approximately 36.319 seconds = 0.01009 single-GPU
  hours; approximately USD 0.0034 at USD 0.34/hour.
- Validation, training, holdout: 0 GPU-hours and 0 calls because not run.
- No v4 package installation or model download was required.

The account is deliberately narrower than Pod rental time: setup/SSH checks are
not episode compute, and the RunPod platform did not expose a separate billing
record to the repository. `reproducibility/v4_actual_compute.json` records the
calculation and caveat.

## Communication and storage

Across four distinct deterministic development matrices: 588 agent messages
(131,544 bytes), 771,840 counted thermodynamic sketch messages (169,821,262
bytes), and 2,304 typed tool calls. These are summed accounting totals, not one
method comparison. Formal event ledgers are individually gzip-compressed; all
artifacts are below 50 MB. Pilot raw trees are deterministic stage archives.

## Provenance

Protocol checksum:
`8eb867207ef638aa4b4c774d99cd464a74a3b6b45a821e081bf7c22dcf68b234`.
Execution source checksum:
`de34b41d7beda8c54546f9a6d027652ff5f438f0d98789cc0dc30a965e3ae37a`.
All 1,590 manifests, raw ledger checksums, and output hashes are retained.
