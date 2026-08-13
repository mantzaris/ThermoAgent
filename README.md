# ThermoAgent

Free-Energy-Guided Coordination of Independent Autonomous Logistics Agents

Local operators and future development sessions should consult
[`notes/README.md`](notes/README.md) for the research record and
[`results/README.md`](results/README.md) for the experiment-facing account of
the evidence. Both are Git-facing; credentials, local agent configuration,
virtual environments, model weights, and caches are excluded.

ThermoAgent is a reproducible research system for asking when genuinely
independent, tool-using organizational agents are justified relative to strong
centralized and scripted logistics controls. It provides two abstract,
material-conserving applications (commercial supply chain and humanitarian
coalition), private per-agent state and memory, validated role-specific tools,
explicit message and commitment ledgers, a frozen open-weight LLM planner, a
decentralized PPO coordination metapolicy, statistical-mechanics-inspired
operational monitoring, time-varying distributed gossip, event-sourced replay,
paired experiment matrices, statistical analysis, and vector figure generation.

The current primary planner is `Qwen/Qwen2.5-7B-Instruct` at immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`, served in-process with Transformers
4.55.4 and bitsandbytes NF4 on one RTX 4090. Model weights are never part of the
repository.

## Result in brief

The complete frozen design contains 944 main, 72 ablation, and 80 locked-
holdout episodes; all 1,096 replay exactly. Operational entropy detects
disruption well, but the calibrated free-energy gap does not. ThermoAgent has
small in-distribution advantages over its matched no-entropy actor that miss
Holm correction and disappear on holdout. Fixed, scripted, and legal
centralized controls generally match or beat it, and every privacy/misalignment
necessity-map cell is negative. The current evidence therefore supports the
engineering architecture and distributed monitor, not a claim that autonomous
agents improve logistics. See [`results/README.md`](results/README.md) for all
effects, uncertainty, negative findings, figures, tables, and limitations.

## Reproducing the study

Inside the RunPod execution copy at `/workspace/ThermoAgent`:

```bash
./scripts/setup-runpod.sh
./scripts/capture-reproducibility.sh
./scripts/run-tests.sh -q
./scripts/run-calibration.sh
./scripts/train-policies.sh
./scripts/run-agentic-smoke.sh
./scripts/run-pilot.sh
./scripts/freeze-protocol.sh
./scripts/run-main.sh
./scripts/run-ablations.sh
./scripts/run-holdout.sh
./scripts/replay-results.sh
./scripts/analyze-results.sh
./scripts/generate-figures.sh
./results/reproducibility/tools/polish-figures.sh
./scripts/validate-pdfs.sh
```

To rebuild all derived artifacts from the retained event ledgers without
rerunning LLM episodes, use
`./results/reproducibility/tools/rebuild-final-results.sh`. Manual PDF
preview inspection is deliberately recorded as a separate step; the exact
command is in the results README.

The frozen evaluation scripts verify
[`results/reproducibility/protocol_freeze.json`](results/reproducibility/protocol_freeze.json)
before executing. Run IDs are deterministic, completed rows resume safely, and
failed/timed-out rows are retained rather than silently rerun.

## RunPod execution environment

Development control, Git operations, Codex/OpenAI authentication, and SSH keys
remain on the local computer. The RunPod host is an execution target only; do
not install Codex there and do not copy local `.env`, credential, key, `.codex`,
or `.agents` files to it.

The default target is the SSH alias `runpod-thermo`, with an execution copy at
`/workspace/ThermoAgent`. Override these non-secret settings when necessary:

```bash
export THERMO_REMOTE_HOST=runpod-thermo
export THERMO_REMOTE_DIR=/workspace/ThermoAgent
# Direct mappings can also set THERMO_REMOTE_PORT, THERMO_REMOTE_IDENTITY, and
# optionally THERMO_REMOTE_KNOWN_HOSTS for an operator-managed host-key file.
```

### Operator workflow

```bash
# Deploy source without deleting remote files or uploading runtime results.
./scripts/runpod-sync.sh

# Verify SSH, the execution copy, CUDA-enabled PyTorch, and a GPU matrix multiply.
./scripts/runpod-smoke-test.sh

# Run any repository command on the RunPod.
./scripts/runpod-exec.sh python3 path/to/simulation.py

# Inspect GPU use, compute processes, and recent run artifacts.
./scripts/runpod-monitor.sh

# Copy all run artifacts back, or fetch one run by ID.
./scripts/runpod-fetch.sh
./scripts/runpod-fetch.sh RUN_ID

# Fetch Git-facing research artifacts generated remotely.
./scripts/runpod-fetch-results.sh
```

Research artifacts are written under `results/`; generic operator artifacts may
use `runs/<run-id>/`. The current implementation loads one in-process frozen
model and keeps every organization's planner context, private state, memories,
inbox, commitments, utility, and decision loop separate. Model weights, Python
environments, and caches belong under `/workspace` outside Git tracking so they
survive container-layer replacement. The isolated project environment is
`/workspace/ThermoAgent/.venv`, and Hugging Face caches live under
`/workspace/.cache`.
