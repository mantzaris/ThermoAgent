# ThermoAgent research notes and RunPod operations

Last verified: 2026-08-12 (America/New_York)

This is Git-facing research and operator documentation. It contains decisions,
failed runs, commands, and the evidence boundary for paper claims. Never place
credentials, current SSH endpoints, private keys, tokens, or private environment
configuration here.

## Authority and safety boundary

- The local repository at `/home/resort/Documents/repos/ThermoAgent` is the
  source of truth and the development/control plane.
- The RunPod is an execution target only; operators configure its SSH target
  locally through the supported `THERMO_REMOTE_*` variables.
- The remote execution copy is `/workspace/ThermoAgent`.
- Keep Codex, Git control, OpenAI authentication, SSH private keys, `.env`
  files, `.codex`, and `.agents` on the local computer.
- Never install Codex or copy OpenAI credentials to the RunPod.
- The remote copy intentionally has no `.git` directory.
- Deployment is non-deleting. Do not add `--delete` to rsync without an
  explicit review of remote-only environments, models, and run artifacts.

## Access

Run commands from the local repository:

```bash
cd /home/resort/Documents/repos/ThermoAgent

# Interactive remote shell, starting in the execution copy
./scripts/runpod-exec.sh

# Run a command in /workspace/ThermoAgent
./scripts/runpod-exec.sh python3 path/to/simulation.py
```

The SSH endpoint and identity remain in local operator configuration. Do not
copy that configuration or key into `/workspace` or these notes.

Optional non-secret overrides:

```bash
export THERMO_REMOTE_HOST=runpod-thermo
export THERMO_REMOTE_DIR=/workspace/ThermoAgent
```

## Deployment

```bash
./scripts/runpod-sync.sh
```

The sync script creates `/workspace/ThermoAgent` when needed and sends the
local working tree with rsync. It includes these sanitized research notes and
excludes Git metadata, credentials, keys, local environments/caches, and
run/result directories. It does not delete remote files. It uses
`--no-owner --no-group` because the RunPod network volume rejects ownership
preservation even for the container's root user.

After changing the sync filters, verify that no credential-like file can be
transferred before deploying.

## Remote environment observed on 2026-08-11

- Ubuntu 24.04 container
- NVIDIA GeForce RTX 4090, 24,564 MiB reported VRAM
- NVIDIA driver 570.195.03; driver CUDA capability 12.8
- 32 logical CPUs and 124 GiB RAM
- Persistent network-backed `/workspace`
- Python 3.12.3
- PyTorch 2.8.0+cu128 with CUDA available
- `uv` 0.9.0, Git 2.43.0, and rsync 3.2.7
- Transformers and Accelerate were installed into the isolated project
  environment after this base-image audit
- No Docker CLI, Conda, or `nvcc`; these are not required for the existing
  CUDA-enabled PyTorch runtime

Treat this inventory as a snapshot. Re-check it after a pod/image replacement.

## Verification

```bash
# Runs a CUDA matrix multiplication and writes runs/setup-smoke-<UTC>/result.json
./scripts/runpod-smoke-test.sh

# Shows GPU metrics, active CUDA processes, top processes, and recent artifacts
./scripts/runpod-monitor.sh

# Fetch all run artifacts, or one run by ID, into local ignored runs/
./scripts/runpod-fetch.sh
./scripts/runpod-fetch.sh RUN_ID
```

The initial end-to-end verification succeeded on 2026-08-11:

- SSH and `nvidia-smi` succeeded.
- CUDA-enabled PyTorch performed a 2048 by 2048 matrix multiplication.
- Remote artifact: `runs/setup-smoke-20260811T213641Z/result.json`.
- Monitoring found the artifact.
- Fetch copied it into the local ignored `runs/` directory.
- Local and remote source-file SHA-256 listings matched exactly.

## Implemented inference topology

The measured design uses one in-process, four-bit Transformers model instance
for efficient batched inference. Agents remain logically independent: each has
its own identity, observation, memory, utility, inbox, commitments, recurrent
policy state, and action authority. No paid API or OpenAI credential is used.

Store persistent remote state under `/workspace`:

```text
/workspace/ThermoAgent/       execution copy and run artifacts
/workspace/ThermoAgent/.venv/ isolated project environment (not Git-facing)
/workspace/.cache/huggingface model cache (not Git-facing)
/workspace/.cache/thermoagent other large runtime cache (not Git-facing)
```

## Normal operating sequence

1. Develop and authenticate locally.
2. Run `./scripts/runpod-sync.sh`.
3. Verify the isolated environment and cached immutable model revision.
4. Launch a uniquely named, detached simulation run through `start-job.sh`.
5. Observe it with `runpod-monitor.sh` (or SSH-based log tailing once logs exist).
6. Write outputs beneath `runs/<run-id>/`.
7. Retrieve outputs with `runpod-fetch.sh` and analyze them locally.
