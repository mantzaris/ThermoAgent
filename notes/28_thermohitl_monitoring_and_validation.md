# Same-information monitoring and validation disposition

Monitoring rows were split on development environment seeds. Candidate
predictors used only fields available inside the assigned private-local view.
Ordinary KPI and KPI-plus-thermodynamic models were compared at matched
attention budgets using AP, ROC AUC, Brier score, and realized paired
counterfactual utility.

On test seeds 12405 and 12406:

- commercial: delta AP -0.01467, delta AUC -0.00664, delta Brier -0.00339,
  and relative budgeted utility +0.02864; gate fail;
- humanitarian: delta AP +0.06486, delta AUC +0.02494, delta Brier -0.06818,
  and relative budgeted utility +2.11842; gate pass.

The cross-application rule failed. No validation seed was generated or opened,
no operating point was selected on validation, and no margin was frozen. The
`validation/NOT_RUN.json` record is deliberate absence, not lost data.
