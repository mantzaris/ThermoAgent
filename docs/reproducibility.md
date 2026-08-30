# Reproducibility

## Frozen study identity

- Frozen protocol SHA-256:
  `863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9`
- Frozen execution-source SHA-256:
  `ec9f26223a335558b2789ebd59ee3c3fa0f9e7d1b815fd9b09a1e1960af55e78`
- Clean semantic source SHA-256 before publication consolidation:
  `f8d4fa546ba46a42cd4234dd8af6ad60309c231f2997e10d0d25830f6dddb2f2`
- Structured-response schema SHA-256:
  `c0382247001c9c586190b81ad4a83535ceb71b03dadfd81e80cac220e9580f0d`
- Seed manifest SHA-256:
  `d9850e5854af307364cd5504d75d0df449412817f3026b1149e8d7de6e8fdaf4`
- Memory-control manifest SHA-256:
  `24bae40fbc4026b66eef87768dd9ae4edbcc3e7b7a9b1f6a850aede21b32a9f4`

The frozen protocol itself is unchanged by publication consolidation. The
unversioned package is a mechanical semantic relocation of its complete active
dependency closure. A consolidation manifest records old-to-new paths and the
new source-tree checksum; future deliberate generation verifies both that
manifest and the unchanged frozen protocol.

## Models and environment

| Role | Model | Revision |
|---|---|---|
| original family | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| independent family | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |

Generation used Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
Transformers 4.55.4, bitsandbytes 0.47.0, NF4 double quantization, BF16
computation, decoding temperature 0.5, top-p 0.9, and at most 96 generated
tokens per decision. Package pins are in `requirements-runpod.txt`.

## Retained and unavailable data

The repository contains frozen protocols, seed/control manifests, aggregate
tables, source data, vector figures, paper sources, and compact verification
records. It intentionally excludes raw prompts, completions, model weights,
and full trajectory records. Exact replay and primary reanalysis therefore
require a separately retained external artifact tree. Set its location with:

```bash
export THERMOAGENT_ARTIFACT_ROOT=/workspace/ThermoAgent-JSTAT-artifacts
```

The original external tree was unavailable after an infrastructure reset. A
from-scratch reconstruction reproduced all 48 trajectories and 34,560 decisions
before this consolidation. Its content-addressed manifests remain in the
compact evidence package, but the reconstructed raw text is also not
distributed.

## Commands

Analysis-independent checks:

```bash
scripts/run-tests.sh
scripts/generate-figures.sh
scripts/build-jstat-paper.sh
scripts/verify-jstat-paper-assets.sh
scripts/verify-source-checksum.py
```

The default figure command writes to a newly created external `/tmp` directory,
compares all regenerated source tables byte for byte with the retained
canonical tables, and leaves the frozen PDF assets untouched. The explicit
`scripts/generate-figures.sh --in-place` mode is reserved for a deliberate PDF
rebuild under a recorded Matplotlib toolchain; PDF bytes can vary across
Matplotlib versions even when every numerical source table is identical.

Commands requiring the external trajectory tree:

```bash
scripts/replay-results.sh
scripts/analyze-results.sh
```

Model download and deliberate formal reproduction:

```bash
scripts/setup-study-environment.sh
.venv/bin/python scripts/prefetch-models.py
export THERMOAGENT_ENABLE_LLM=1
scripts/run-formal-experiment.sh qwen
scripts/run-formal-experiment.sh granite
```

No test, figure, paper-build, or verification command invokes an LLM.

## Compute accounting

The original experiment reported 19.193 measured generation GPU-hours on an
NVIDIA GeForce RTX 4090. The independent reconstruction recorded 48.737 hours
for complete formal panels, 0.349 hours for successful model pilots, and 0.328
hours in interrupted-panel records, for at least 49.415 measured generation
hours plus one post-generation infrastructure call whose tokens and latency
were not durably recorded. These are different executions and are never added
together as one scientific sample.
