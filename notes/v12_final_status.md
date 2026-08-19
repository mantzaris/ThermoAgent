# V12 final status

Date: 2026-08-18 America/New_York / 2026-08-19 UTC.

## Provenance and Git disposition

- Branch: `llm-agent-stochastic-thermodynamics-v12`.
- Immutable parent: `0d73693160c25e251533d6f6720fdd78b349605e` on
  `evidence-grounded-llm-entropy-v11`; the fetched remote and local parent
  matched before branching.
- V1--V11 result namespaces were not modified.
- No V12 file was staged, committed, pushed, tagged, published, or submitted.
- The final worktree is intentionally uncommitted for manual review.

## Frozen protocol

- Version: `v12-llm-statmech-1.0`.
- Protocol SHA-256:
  `cee49ca7111b81ae9544a8d32a754ce4c9f15bff16333e98588147a7f7f665b6`.
- Frozen execution-source SHA-256:
  `0796286362ec4dde0eb4f2dc88ecea4c3bf53859e618aec1928ba8b9e8b0a154`.
- Post-freeze, pre-output bookkeeping repair SHA-256:
  `9072c6380c02eef00b1d25e23bd5da4e266aff4a6a9b8dc2b36185520d6c073c`.
  This repair copied the frozen replicate field into a derived summary row; it
  changed no model decision, panel, seed, estimand, test, or numerical input.

## Formal execution

- Primary model: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, Transformers 4.55.4,
  PyTorch 2.8.0+cu128, NF4 double quantization and BF16 computation on an RTX
  4090; top-p 0.9; maximum 144 new tokens.
- Formal units: 401; 400 were network/hysteresis trajectories and one was the
  576-decision microscopic response grid. The 400 network trajectories span
  69 graph/environment clusters; the microscopic grid spans 96 independent
  information-state clusters.
- Network sizes: N=3, 4, 8, and 16. Topologies: fixed-degree ring and modular.
  Nonreciprocity: alpha=0, 0.2, 0.5, and 0.8 with matched forward/transpose
  orientations. One sweep is N attempted random-sequential updates.
- Decisions: 44,352/44,352. Network updates: 43,776 across 6,082
  network-equivalent sweeps; isolated microscopic decisions: 576.
- Formal model calls: 44,354; prompt tokens: 24,230,610; generated tokens:
  3,543,392. Two first-pass responses used the frozen greedy repair; zero
  remained invalid.
- All pilots plus formal work: 44,864 decision requests, 44,882 calls,
  24,509,789 prompt tokens, and 3,585,830 generated tokens.
- Metered generation: 20.3182 single-GPU hours total; estimated incremental
  cost USD 6.91--14.02. Formal generation plus its current model load was
  20.0923 hours. Analysis used 76.54 CPU-seconds (80.36 wall-seconds).
- Communication: 34,788 model-selected packets, 626,184 complete binary wire
  bytes, and zero private-state mutations.

## Scientific disposition

- **H1 supported:** delivered neighbor field changed latent-plus categorical
  choice by +0.08333 per unit field (95% CI 0.04167--0.125; 96 clusters).
  The microscopic grid had exactly 0.5 latent-plus occupancy and 58 belief
  transitions in each direction.
- **H2 not supported:** strong-alpha adjusted block-KL contrast was -0.0002327
  nats/update (-0.001441--0.001013; 8 small clusters) and -0.0007589
  (-0.01115--0.009718; 32 collective clusters).
- **H3 not supported:** small and collective nonreciprocity-dose slopes were
  +0.0007068 (-0.0005011--0.002331) and -0.00005391
  (-0.01449--0.01456), respectively.
- **H4 not supported:** the small-system quadratic coefficient was positive,
  but held-cluster RMSE (0.001937) exceeded the linear model (0.001925); the
  collective quadratic interval crossed zero.
- **H5 supported as a finite-size effect:** coupling increased absolute belief
  order (+0.03298), susceptibility (+0.04550), and correlation time (+2.319
  updates); decoding noise reduced them by 0.05387, 0.07356, and 5.990 updates.
  The prespecified family survived Holm correction.
- **H6 not supported:** N=8 collective point estimates were positive and N=16
  point estimates negative; every collective orientation interval crossed
  zero.
- Small trajectories occupied 1.80 projected states and 3.20 transition pairs
  on average, with exact mean belief--action overlap 1.0. Collective
  trajectories occupied 5.26 states and 8.04 transition pairs, with mean
  overlap 0.904 and configuration entropy 1.132 nats.
- Reciprocal adjusted block-KL floors were -0.0002775 nats/update in small
  systems and -0.04176 collectively. Sparse support, nonzero conditional
  history information, and substantial collective early/late JS divergence
  preclude exact LLM Markov-entropy-production claims.
- Persistent memory increased adjusted block irreversibility by +0.01790
  nats/update (0.00357--0.03399; 24 clusters). No three-cluster content,
  temporal, sender, placebo, or no-message control survived Holm correction.

## Integrity and presentation

- Combined V10--V12 regression suite: 102 passed, 0 failed, 0 errors, 0
  skipped (36 V10, 33 V11, 33 V12), plus the V12 analysis-repair self-test.
- Content-addressed deterministic regeneration: 44,352 rows in 401 units;
  zero mismatches.
- Package verification passed formal accounting, frozen source, privacy,
  checksums, replay, PDF automation, manual visual QA, repository-size,
  individual-file-size, forbidden-artifact, and empty-index checks.
- Sixteen figure PDFs and the 18-page manuscript opened, had embedded fonts
  and extractable text, and rendered at 300 DPI. All 17 PDFs and all 34
  rendered pages were manually inspected at original rendered resolution.
- Repository-facing V12 files total approximately 3.0 MB. The largest files
  are `agent_statistics.csv` (654,247 bytes), `panel_statistics.csv` (627,898
  bytes), and `main.pdf` (459,711 bytes); no new file approaches 10 MiB.
- Raw artifacts remain external at `/workspace/ThermoAgent-v12-artifacts/`
  (221 MB on the final Pod check). Stage tree hashes are recorded in
  `results/llm_agent_statmech_v12/reproducibility/external_artifact_trees.csv`.
- No V12 tmux, Python experiment, Qwen, or CUDA compute process remained.
  The RTX 4090 reported 0% utilization and 1 MiB used. Only the Pod's standard
  Jupyter service remained; the Pod is idle and safe to stop, but not delete.

## Publication assessment

The package is paper-ready as an honest mixed/negative finite-size study and a
substantive empirical statistical-mechanics characterization of one open LLM.
It supports a causal microscopic neighbor response and collective
coupling/noise effects. It does not support a positive network-level
nonreciprocity or quadratic-entropy-production claim. JSTAT submission is
plausible after independent statistical-physics review, but stronger evidence
would require longer trajectories, better state support and stationarity,
larger sizes, and ideally a second pinned model.

## Operational security note

During the final read-only process inspection, one terminal diagnostic emitted
an existing Jupyter command-line authentication argument into the tool output.
It was not copied into the repository, artifacts, notes, or final numerical
reports and is not reproduced here. The Jupyter credential should be rotated
before the Pod is reused or shared.
