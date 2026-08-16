# V6 statistical outputs

V6 stopped after prospective development gates failed. Consequently, every
inferential result in this package is development evidence; there are no
validation or holdout statistics.

Canonical machine-readable outputs are deliberately kept beside the stage that
generated them rather than duplicated here:

- paired dynamic effects and cluster-bootstrap intervals:
  `../development/dynamic/paired_dynamic_effects.csv`;
- matched risk/coverage effects:
  `../development/risk_analysis/primary_matched_effects.csv`;
- fragmentation interactions:
  `../development/dynamic/fragmentation_interaction.csv`;
- entropy-family comparisons:
  `../development/entropy_family/entropy_family_summary.csv`;
- full-refit permutation tests:
  `../development/permutation/refit_permutation_family_test.csv`;
- gate decisions: `../development/gate_status.json` and
  `../development/gate_checks.csv`.

Publication-facing copies and catalogs are under `../tables/`. The complete
episode panel, rather than an incident or action row, is the inferential unit.
