# V5 reproduction and compute accounting

Canonical commands from the repository root:

```bash
./scripts/run-human-operator-v5-tests.sh
./scripts/run-human-operator-v5-development.sh
./scripts/run-human-operator-v5-sketch-ablation.sh
./scripts/analyze-human-operator-v5-results.sh
./scripts/run-human-operator-v5-real-qwen.sh
./scripts/run-human-operator-v5-training.sh
./scripts/replay-human-operator-v5-results.sh
./scripts/evaluate-human-operator-v5-gates.sh
./scripts/generate-human-operator-v5-figures.sh
./scripts/validate-human-operator-v5-pdfs.sh
./scripts/build-human-operator-v5-report.sh
./scripts/verify-human-operator-v5-artifacts.sh
```

The development command is restartable and will reuse completed atomic panel
artifacts. The analysis command includes both 199-replicate refit permutation
tests and therefore takes materially longer than the other CPU workflows.
Validation and holdout scripts fail loudly because those stages were never
unlocked.

Real-Qwen qualification used the existing RTX 4090 and pinned
`Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, Transformers 4.55.4,
bitsandbytes NF4, BF16 computation, and PyTorch 2.8.0+cu128. It made 108 calls,
consumed 104,836 prompt and 6,116 generated tokens, and took 109.34 seconds
including model load (0.0304 single-GPU hours; about USD 0.010 at the project's
illustrative USD 0.34/hour rate).

The ten RL runs used CPU, totaling 302,080 sampled training decision epochs,
21,600 development evaluation decisions, and 898.7 summed CPU seconds. No V5
validation or holdout compute was consumed.
