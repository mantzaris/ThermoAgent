# V13 final status

V13 completed the frozen `v13-collective-agent-statmech-1.2` design from V12
parent `457f6d635b60292623c8d97aa3b0c60d8d0aac4e`. Its protocol SHA-256 is
`a5259bbfd49da20b23a79646c248e0723a7fc382fa37b6165e3e936b4b669e3a` and its
formal execution-source SHA-256 is
`72c76946020e6ff7137848de25f384dd9cb25b5c3999754cac0b1b65ae7a4cc9`.

## Disposition

- H1 coupling confirmation: not supported; order and susceptibility were null,
  and correlation time changed in the opposite direction.
- H2 decoding-noise confirmation: not supported; all three estimates changed in
  the opposite direction, with nonzero intervals for order, susceptibility, and
  correlation time.
- H3 bounded-memory confirmation: supported after frozen Holm correction;
  adjusted block irreversibility increased by 0.04030 nats per attempted update
  (95% CI 0.02883 to 0.05856).
- H4 macrostate departure: supported only under the frozen aggregate criterion
  and driven by the field reversal; communication partition and corruption did
  not separate materially from nominal operation.
- H5/H6 representation analyses: positive under their frozen interval criteria
  but preliminary. Full leave-cluster-out accuracy was 0.500 and its increment
  over the strongest reduced representation was 0.1875 (95% CI 0.0625 to
  0.2500); all field reversals but only 4/12 remaining panels were correct.
- H7 kinetic-surrogate trend capture: not supported. Only the noise direction
  was captured.

The supported paper claim is therefore narrow: an explicitly bounded memory
state reproducibly adds coarse-grained time asymmetry, and a fixed
statistical-mechanical representation resolves a strong environmental quench
better than reduced aggregates. V13 does not establish reproducible monotone
coupling/noise laws, generic communication-disruption detection, exact entropy
production, physical energy or temperature, a phase transition, or application
benefit.

## Execution and integrity

- Pilot: 192 decisions, used only for validity, occupancy, and runtime.
- Analyzed formal: 32,672 decisions/calls, including 288 microscopic decisions
  and 32,384 network updates in 72 trajectories (2,264 sweeps; 21 unique
  clusters).
- Retained invalidated pre-amendment attempts: 163.
- Total project calls including pilot and invalidated work: 33,027.
- Tokens: 18,387,880 prompt and 2,652,913 generated.
- Metered generation: 14.7351 GPU-hours; estimated incremental cost USD
  5.01--10.17.
- Replay: 32,672 rows in 73 units, zero mismatches.
- Final focused regression suite: 125 V10--V13 tests passed; zero failures,
  errors, or skips.
- PDF QA: 22 one-page vector figures plus the 21-page manuscript (23 PDFs,
  43 pages) opened, rendered at 300 DPI, exposed extractable text, used embedded
  fonts, and passed manual inspection at original rendered resolution with no
  observed clipping or overlap.
- External raw tree: `/workspace/ThermoAgent-v13-artifacts/`; checksums are in
  `results/collective_agent_statmech_v13/reproducibility/external_artifact_summary.json`.
- No V1--V12 artifact was modified. No V13 file was staged, committed, pushed,
  tagged, published, or submitted.

Final test, PDF, repository-size, checksum, process, and worktree audits are
recorded in the results reproducibility summary and result README.
