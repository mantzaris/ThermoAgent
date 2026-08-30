# ThermoAgent: JSTAT publication repository

This repository contains the paper **Statistical-mechanical characterization of
memory and quench response in state-separated LLM-agent networks**, its compact
evidence package, and the code needed to validate or deliberately reproduce the
study.

The study treats locally informed, state-separated language-model agent
instances as an interacting finite stochastic system. It measures collective
order, effective reference energy, entropy, dependence, correlation, response
to a field quench, finite-horizon recovery, and coarse-grained path-reversal
asymmetry. Effective energy is not physical energy, decoding temperature is not
thermodynamic temperature, and path-reversal divergence is not claimed to be
exact thermodynamic entropy production.

## Repository map

- `paper/JSTAT/`: self-contained LaTeX manuscript, bibliography, 14 vector PDF
  figures, generated result macros, and the built paper.
- `thermoagent/statmech_llm/`: final implementation. Semantic subpackages retain
  the discovery, replication, and corrected-quench dependency closure.
- `configs/statmech_llm/`: immutable scientific configurations arranged by
  study role.
- `tests/statmech_llm/`: scientific, replay, reporting, and integrity tests.
- `results/JSTAT/`: compact aggregate tables, figure source data, correction
  records, protocol records, and reproducibility manifests.
- `docs/`: methodology, reproducibility, validation, and data dictionary.
- `scripts/`: the small supported workflow surface.

## Environment

For analysis-only use, create a Python environment and install the research
dependencies from `pyproject.toml`. The exact generation environment can be
created on a CUDA 12.8 system with:

```bash
scripts/setup-study-environment.sh
```

The pinned generation stack uses Python 3.12.3, PyTorch 2.8.0+cu128,
Transformers 4.55.4, bitsandbytes 0.47.0, NF4 double quantization, and BF16
computation. Model weights are not included. The two pinned models are
`Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28` and
`ibm-granite/granite-3.3-8b-instruct` at revision
`51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`.

## Safe validation and paper build

These commands do not call an LLM:

```bash
scripts/run-tests.sh
scripts/generate-figures.sh
scripts/build-jstat-paper.sh
scripts/verify-jstat-paper-assets.sh
scripts/verify-source-checksum.py
```

`scripts/generate-figures.sh` rebuilds into an external temporary directory by
default and requires all 14 regenerated source-data CSVs to match the retained
tables byte for byte. It does not overwrite the canonical PDFs. A deliberate
toolchain-specific PDF rebuild requires the explicit `--in-place` argument.

Replay and primary analysis require the non-distributed external trajectory
records:

```bash
export THERMOAGENT_ARTIFACT_ROOT=/workspace/ThermoAgent-JSTAT-artifacts
scripts/replay-results.sh
scripts/analyze-results.sh
```

The formal experiment is intentionally guarded and is never run by validation:

```bash
export THERMOAGENT_ENABLE_LLM=1
scripts/run-formal-experiment.sh qwen
scripts/run-formal-experiment.sh granite
```

## Data and compute

The repository distributes aggregate numerical tables and one source CSV for
each publication figure. It does not distribute raw prompts, model completions,
or complete trajectories, matching the manuscript's data-availability
statement. The original paper experiment used an NVIDIA GeForce RTX 4090 and
19.19 measured generation GPU-hours. A later from-scratch reconstruction used
at least 49.415 measured generation GPU-hours; its accounting and the original
accounting are retained separately. See [reproducibility.md](docs/reproducibility.md)
for the exact distinction.

The independent inferential unit is a complete graph/environment trajectory
cluster—not an agent, update, message, token, rolling window, or classifier
prediction.
