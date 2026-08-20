# V13: collective statistical mechanics of decentralized LLM agents

## Supported scope

V13 prospectively tests whether statistical-mechanical observables provide a compact reduced description of actual independent LLM-agent networks. The binary beliefs and actions are Qwen choices, reference energy is an effective symmetric-layer observable, decoding temperature is a decision-noise control, and adjusted block reversal divergence is coarse-grained pathwise irreversibility. The study does not claim physical heat, exact LLM entropy production, a thermodynamic-limit transition, controller superiority, application benefit, or human evidence.

V12 is an immutable discovery study. V13 does not reopen its negative nonreciprocity endpoint. Directed communication alone remains a documented boundary: it did not reliably raise irreversibility under the V12 degree- and traffic-matched conditions.

## Model, agents, and update

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Runtime: NF4_double_quantization_BF16_compute; top-p 0.9; at most 96 generated tokens; no chain-of-thought request.
- Each agent separately owns belief, action, confidence, commitment, bounded memory, private field, workload, inbox, outbox, context, and typed authority.
- The scheduler uses a random permutation within each sweep, offers one local update, transports the model-selected packet, and never selects a scientific state or action. One sweep is `N` attempted updates.
- The graph/environment trajectory cluster is the independent inferential unit.

## Frozen experiment

Protocol `v13-collective-agent-statmech-1.2`; SHA-256 `a5259bbfd49da20b23a79646c248e0723a7fc382fa37b6165e3e936b4b669e3a`; execution-source SHA-256 `72c76946020e6ff7137848de25f384dd9cb25b5c3999754cac0b1b65ae7a4cc9`. Amendment 01 was made before any network panel existed because four paired clusters made the frozen sign-flip threshold mathematically unattainable; all 163 interrupted microscopic records were retained. Amendment 02 clarified before restart that the 18M-token ceiling includes pilot and interrupted calls; no scientific design changed. Work Package A uses `N={8,16}`, modular primary and ring replication graphs, coupling `{0.35,0.80}`, decoding noise `{0.50,0.85}`, and Markovized agents. Work Package B pairs Markovized and bounded-memory agents on six `N=16` modular field-quench clusters. Work Package C pairs nominal, field-reversal, inter-community partition, and 50% categorical packet-corruption trajectories on four `N=16` modular clusters.

Formal execution completed 32,672/32,672 analyzed decisions in 72 graph trajectories plus the microscopic grid. Including the retained pre-amendment records and pilot, the study used 33,027 calls, 18,387,880 prompt tokens, and 2,652,913 generated tokens. The prompt-token total exceeded the 18-million target by 387,880 (2.2%); this token-only projection was documented before completion, and no scientific panel, seed, contrast, or stopping rule changed. Total metered generation was 14.735 GPU-hours; estimated incremental cost is USD 5.01--10.17. Raw model records and transitions remain external at `/workspace/ThermoAgent-v13-artifacts`.

## Confirmatory V12-to-V13 effects

- H1 coupling: order -0.0006 (95% CI -0.0069 to 0.0057), susceptibility -0.0035 (95% CI -0.0134 to 0.0088), and integrated correlation time -3.5555 (95% CI -8.9104 to -0.2368).
- H2 decoding noise: order 0.0078 (95% CI 0.0013 to 0.0152), susceptibility 0.0160 (95% CI 0.0025 to 0.0331), and integrated correlation time 3.2319 (95% CI 0.7517 to 5.4844).
- H3 bounded memory: adjusted pathwise irreversibility 0.04030 (95% CI 0.02883 to 0.05856) nats per attempted update.

H1 and H2 did not replicate: estimates were null or opposite to the V12 discovery directions, with Holm-adjusted hypothesis-level p-values of 1.0. H3 replicated and passed Holm correction (adjusted p=0.047995). The H1--H3 family uses paired graph-cluster bootstrap intervals, intersection-union directional tests for the multi-endpoint H1/H2 claims, and Holm correction across the three frozen hypotheses. V12 discovery and V13 confirmation estimates are stored separately in `tables/v12_discovery_effects.csv` and `tables/hypothesis_effects.csv`.

## Disruptions and reduced representation

Controlled disruptions changed maximum distance from the leave-cluster-out nominal manifold by 14.5446 (95% CI 14.0547 to 14.8675) on average relative to the matched undisturbed trajectory. This aggregate H4 result is driven by field reversal: mean maximum distance was 45.055 for reversal, 2.315 for message corruption, 1.633 for partition, and 1.789 for nominal operation. Four-class disruption separation by the full statistical-mechanical representation achieved mean leave-cluster-out accuracy 0.500, versus 0.312 for simple aggregates and 0.250 for order-only features. H5 accuracy above chance was 0.2500 (95% CI 0.2500 to 0.2500); H6 full-minus-strongest-reduced accuracy was 0.1875 (95% CI 0.0625 to 0.2500). All four field reversals were classified correctly, but only four of twelve partition, corruption, and nominal panels were; with four clusters, H5/H6 are positive under their frozen interval criteria but preliminary.

Energy--entropy portraits and the fixed reduced vector trace baseline, quench, and recovery, but this is disruption response—not early warning. High entropy is not assigned a universal good/bad meaning. The fitted kinetic surrogate captured the noise direction but missed the coupling direction (surrogate/direct coupling effects +0.01497/-0.00059; noise +0.02955/+0.00775), so H7 was not supported and the surrogate is not substituted for direct LLM evidence.

## What ran and what did not

Ran: a 192-decision engineering pilot limited to estimability/runtime, the complete frozen formal grid, deterministic content-addressed replay, CPU surrogate map, all frozen analyses, 22 candidate vector figures, and manuscript/PDF QA. Did not run: a second LLM, new nonreciprocity search, application-performance trial, human study, validation/holdout reuse, thermodynamic-limit scaling claim, or outcome-dependent rerun.

## Files and reproduction

- Operative protocol: `protocol/protocol_frozen_v1.2.yaml`; invalidated pre-outcome freezes: `protocol/protocol_frozen_v1.0_invalidated.yaml` and `protocol/protocol_frozen_v1.1_invalidated.yaml`
- Primary results: `statistics/primary_results.json`
- Panel and trajectory tables: `tables/`
- Figures and exact sources: `figures/pdf/`, `figures/source_data/`, and `figures/figure_catalog.csv`
- Replay, checksums, compute, and PDF QA: `reproducibility/`
- Manuscript: `../../paper/jstat_v13/main.tex` and `main.pdf`

```bash
PYTHON_BIN=/workspace/ThermoAgent/.venv/bin/python THERMO_V13_ARTIFACT_ROOT=/workspace/ThermoAgent-v13-artifacts scripts/run-statmech-v13-tests.sh
THERMO_V13_ENABLE_QWEN=1 scripts/run-statmech-v13-pilot.sh
scripts/freeze-statmech-v13-protocol.sh
THERMO_V13_ENABLE_QWEN=1 scripts/run-statmech-v13-formal.sh
scripts/replay-statmech-v13.sh
scripts/analyze-statmech-v13.sh
scripts/generate-statmech-v13-figures.sh
MPLCONFIGDIR=/tmp/v13-mpl-cache .venv/bin/python paper/jstat_v13/refine_figures.py
scripts/build-statmech-v13-results.sh
scripts/build-statmech-v13-paper.sh
scripts/verify-statmech-v13.sh
```
