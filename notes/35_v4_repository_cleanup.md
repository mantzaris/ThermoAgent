# V4 repository cleanup

## Preservation boundary

- The immutable v3 scientific snapshot remains commit
  `3f844966930b1cfb5a43bdf3a4d3e744391d1018`.
- This work is on `thermodynamic-human-oversight-v4`, created directly from
  that commit.
- Frozen v1 and v2 result namespaces were not normalized or regenerated.

## Line endings

A scoped `.gitattributes` policy now forces LF for maintained source and text
artifact formats and marks binary/result formats as binary. Forty-seven v3
text artifacts with CRLF endings were normalized on the v4 branch. The
original and normalized SHA-256 values, CRLF counts, parser used, and semantic
equality result are recorded in
`results/human_operator_v4/reproducibility/v3_line_ending_normalization.json`.
All 47 parser-level comparisons passed. CSV writers now request `\n`
explicitly, and a regression test scans source, scripts, configs, v3, and v4
text outputs for CRLF. V1/v2 are deliberately outside that test boundary.

## Historical monitoring clarification

`results/human_operator_v3/monitoring/README.md` now labels its original
ranking analysis as a superseded intermediate development classifier. It links
to the authoritative final dense causal Gate 5 records. The historical result
was retained: both applications failed the earlier ranking test; the final
causal analysis passed humanitarian, failed commercial, and therefore failed
the cross-application v3 gate.

## Figure repair

- The energy/entropy phase plane uses nominal-calibrated standardized energy,
  standardized entropy anomaly, the exact two-variable projection of the
  frozen prospective rule, and actual disruption/intervention points.
- The intervention funnel now separates episode-wide autonomous material
  actions from paired counterfactual intervention probes.
- Operator-view incremental value now reports absolute utility, paired and
  relative effects, cluster-bootstrap intervals, regimes, and independent
  panel counts, with a prominent development-only boundary.
- The dashboard overview is populated from a schema-validated ledger frame;
  commercial and humanitarian SVG replay exports and hashes are retained.
- The RL-training `PROSPECTIVELY NOT RUN` placeholder moved out of the main
  publication figure set and remains in the reproducibility record.
- All 19 evidence-bearing PDFs opened, exposed fonts, rendered, and passed
  original-resolution visual inspection after layout fixes.

## Verification at cleanup checkpoint

- Full repository tests: 184 passed.
- `git diff --check`: clean.
- Diff check from the v2 base including the v3/v4 cleanup: clean.
- V3 quantitative evidence was not recomputed or reinterpreted; figure and
  documentation corrections only expose the existing evidence more clearly.
