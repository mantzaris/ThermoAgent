# V3 monitoring and intervention-value analysis

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
