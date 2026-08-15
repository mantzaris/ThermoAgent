# Superseded intermediate v3 monitoring classifier

> **Historical development artifact — not the authoritative Gate 5 result.**
> This file records the earlier `monitoring_development` classifier, where
> both applications failed the ranking-only test. It preceded the dense paired
> counterfactual analysis and must not be cited as the final v3 Gate 5 result.

The final causal Gate 5 analysis used actual paired intervention effects and a
fixed attention budget. It found a commercial failure and a humanitarian pass,
then failed the required cross-application rule. The authoritative evidence is:

- [`../development/gate_status.json`](../development/gate_status.json) — final
  prospective six-gate decision;
- [`causal_value_summary.json`](causal_value_summary.json) — final dense causal
  summary;
- [`causal_incremental_value.csv`](causal_incremental_value.csv) — application
  results and thresholds; and
- [`../PAPER_SUMMARY.md`](../PAPER_SUMMARY.md) — paper-facing interpretation.

The JSON below is retained verbatim as a superseded intermediate diagnostic for
development-history transparency. Its row-level classification target and
ranking-only criterion differ from the final causal-utility Gate 5 analysis.

```json

{
  "applications": [
    "commercial",
    "humanitarian"
  ],
  "created_at": "2026-08-14T20:04:34.114333+00:00",
  "gate_5_rank_passed_both_applications": false,
  "incremental": [
    {
      "application": "commercial",
      "comparison": "same_information_local_KPI_plus_thermodynamics_minus_local_KPI",
      "delta_average_precision": -0.004047970121630495,
      "delta_brier_score": -0.0005564306090793218,
      "delta_roc_auc": -0.00021434400404929388,
      "gate_threshold": 0.05,
      "rank_gate_passed": false
    },
    {
      "application": "humanitarian",
      "comparison": "same_information_local_KPI_plus_thermodynamics_minus_local_KPI",
      "delta_average_precision": -0.01125012160750627,
      "delta_brier_score": -0.0006898207391500136,
      "delta_roc_auc": -0.00038938492063489427,
      "gate_threshold": 0.05,
      "rank_gate_passed": false
    }
  ],
  "interpretation": "Thermodynamic features are tested only against ordinary KPIs from the same private-local boundary. The intervention label is evaluator-only and used for development training/evaluation, never actor execution.",
  "rows": 17664,
  "stage": "monitoring_development"
}
```
