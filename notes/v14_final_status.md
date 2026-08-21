# V14 final status

## Provenance and scope

- Branch: `memory-quench-agent-statmech-v14`.
- Immutable parent: `20a9ca66041b1636bed15d5916aabcb605e6a063` on `origin/collective-agent-statmech-v13`.
- Protocol: `v14-memory-quench-agent-statmech-1.0`.
- Protocol SHA-256: `5d8440dedbf389c02f3b448f38abfd1a370b8f9c4fafdba4760afc706e0bcfdf`.
- Frozen execution-source SHA-256: `2b4276dd323bb048d8c98834ed0e9f8bfe5a0ed46e8735db3b471a6dc97e91ad`.
- V1--V13 namespaces were not modified or regenerated. V14 remains unstaged, uncommitted, and unpushed for human review.

## Formal design and accounting

- Six independent graph/environment clusters, four matched conditions, 24 trajectories, 16 agents, and 45 sweeps per trajectory.
- Conditions: nominal evolution, private-field reversal, temporary inter-community partition, and categorical message corruption for a preassigned 50% of senders.
- The complete graph/environment cluster is the inferential unit; agents, updates, windows, messages, and tokens are not independent replicates.
- Formal execution: 17,280 decisions and calls, 9,510,001 prompt tokens, 1,391,607 generated tokens, and 7.78995 generation GPU-hours.
- Engineering pilot: 256 decisions and calls, 138,700 prompt tokens, 20,340 generated tokens, and 0.11417 generation GPU-hours.
- Total: 17,536 decisions/calls, 9,648,701 prompt tokens, 1,411,947 generated tokens, and 7.90412 metered generation GPU-hours.
- Approximate incremental RTX 4090 cost: USD 2.69--5.45. Formal CPU analysis used 49.03 CPU-seconds; ancillary tests, plotting, LaTeX, transfer, and QA were local and are not included in that narrow analysis timer.
- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, Transformers 4.55.4, PyTorch 2.8.0+cu128, bitsandbytes 0.47.0, NF4 double quantization with BF16 computation, decoding temperature 0.5, top-p 0.9, maximum 96 generated tokens, no chain-of-thought request.

## Scientific disposition

- Memory: the immutable V12 discovery estimate is 0.01790 nats/update (95% interval 0.00357--0.03399; 24 clusters), and the immutable V13 prospective replication is 0.04030 (0.02883--0.05856; six clusters). Their descriptive fixed-effect synthesis is 0.02936 (0.01872--0.03999). V14 ran no new memory trajectories and does not relabel the synthesis as a V14 confirmation.
- Field quench: maximum post-quench macrostate departure minus matched nominal is 133.80477 regularized distance units (107.94944--184.07463; exact one-sided sign-flip `p=0.015625`; Holm-adjusted `p=0.046875`). All six paired cluster effects are positive.
- Counter-quench: restoration creates a second departure pulse. The frozen early-peak-minus-final-five contrast is 134.11032 (106.65373--184.19622), but its sign is structurally nonnegative, so its sign-flip result is not standalone evidence. Every trajectory crossed its held-out nominal threshold six sweeps after restoration, and the final-five mean distance is 1.560.
- Representation: full-minus-order-only leave-one-cluster-out balanced accuracy is 0.33333 (0.16667--0.45833; exact `p=0.03125`; Holm-adjusted `p=0.046875`). Absolute balanced accuracy is 0.125 for order-only, 0.458 for simple uncertainty, and 0.458 for the full representation. The full state adds information beyond order alone but does not outperform the smaller uncertainty representation overall.
- Entropy and dependence: field reversal raises disruption-period configuration entropy from 0.223 to 0.387 nats, entropy rate from 0.070 to 0.356 nats/sweep, total correlation from 0.274 to 4.909 nats, pairwise mutual information from 0.00020 to 0.10627 nats, and energy variance from 0.00955 to 1.16175, while absolute belief magnetization remains near zero. This is the clearest evidence that order averages alone discard collective structure.
- Robustness: 266 of 276 frozen macrostate-distance sensitivity cells retain a positive field-minus-nominal contrast and ten are exactly zero; none reverse the ordering. Removing entropy/dependence nearly eliminates the primary-geometry contrast. Single-observable deletion and nominal-distance recomputation for alternative rolling windows were not implemented in the frozen audit and were not reconstructed after outcome inspection.
- Boundaries: partition and corruption remain close to nominal; the representation does not provide omnibus disruption classification. V12's matched nonreciprocity result remains unsupported. V13's coupling/noise directions were not independently rerun in V14. The kinetic surrogate misses the direct LLM coupling direction and overestimates the noise contrast. No thermodynamic-limit transition, literal physical energy, physical free energy, exact LLM entropy production, universal model behavior, or performance benefit is claimed.

## Integrity, artifacts, and publication package

- All 17,280 formal rows across 24 trajectories replayed with zero mismatches. Privacy mutations, scheduler substitution, invalid final outputs, repairs, nonfinite primary values, message drops, and accounting mismatches are zero.
- The repository-wide suite collected and passed 599 tests, including 22 focused V14 tests and the relevant V10--V13 regressions.
- The paper-facing package contains 26 data-derived vector figure PDFs with source CSVs and a 17-page manuscript using eight main figures.
- Automated PDF QA passed for 27 PDFs and 43 pages. Manual inspection passed at original size and at 300 DPI; fonts are embedded, text is extractable, and no material clipping, overlap, or missing glyph remains.
- The repository-facing V14 addition is approximately 7.6 MB, well below the 15 MiB preference and 25 MiB ceiling. No individual V14 file exceeds 10 MiB; the largest is the 744,066-byte manuscript PDF.
- External raw artifacts: `/workspace/ThermoAgent-v14-artifacts/`, 17,572 files and 86,744,804 bytes in the compact manifest, tree SHA-256 `cc42761c0ce2b0873f72d571f74a04f5bcbf345e07b4651606a4398bb121dd05`. Raw artifacts are not repository-facing.
- The existing RTX 4090 Pod is idle at 1 MiB GPU memory and 0% utilization, with no active compute process or tmux session. It is safe to stop, but not delete.

## Readiness

The implementation, frozen execution, replay, statistical tables, source-data figures, and manuscript form a defensible single-model finite-system characterization. The central JSTAT-facing claim is supported: statistical-mechanical observables expose a reproducible memory-associated temporal asymmetry and a field-quench/counter-quench trajectory that magnetization alone does not describe. The package is ready for expert scientific review, not yet unconditional submission. Remaining work includes independent replication on another pinned model, more graph clusters and sizes, a new prospective memory experiment rather than synthesis alone, improved partition/corruption power, the missing single-observable and alternative-window distance sensitivities, and external scrutiny of the coarse-graining and literature positioning.
