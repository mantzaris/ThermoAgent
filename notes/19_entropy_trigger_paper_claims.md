# DOET paper claims and evidence

No positive v2 claim is currently supported. Candidate claims remain hypotheses
until mapped to a frozen holdout statistic, table, and figure. The final mapping
will classify each as confirmed, suggestive, mixed, unsupported, or untested.

The unchanged v1 boundary result remains: operational entropy monitored
disruption well, but entropy-conditioned coordination was not confirmatorily
better and autonomous agents did not beat strong simpler controls.

| Candidate claim | Required evidence | Current status |
|---|---|---|
| Original ties were caused by identical policies | Diagnostics tables + tie PDF | **Unsupported wording**: policies diverged; ties arose because actions had no demand-reaching material consequence. |
| Entropy universally outpredicts ordinary KPIs | Monitoring tables + incremental-value PDF | **Unsupported**: full/global KPIs subsume entropy. |
| Entropy adds information under private local observability | `monitoring/incremental_value.csv`, entropy incremental-value PDF | **Suggestive development evidence**: about +0.10 AP and +0.17 AUC; seen holdout ranking gains are not robust. |
| DOET is non-inferior to fixed communication | Locked non-inferiority table/forest, H1 | **Untested**. |
| DOET reduces fully counted communication by at least 20% | Locked reduction table/figure, H2 | **Untested**. |
| DOET improves the loss-cost Pareto frontier | Frozen hypervolume table/Pareto PDF, H3 | **Untested**. |
| DOET activates before visible collapse with low nominal false activation | Mechanistic table/case figures, H4 | **Untested**. |
| DOET remains useful under partitions | Partition statistics/PDF, H5 | **Untested**. |
| Result generalizes across both applications | H1/H2 in both applications, H6 | **Untested**. |
| Independent autonomous agents are necessary | Fixed/central/scripted boundary comparison | **Unsupported by v1; not a permitted v2 claim without new direct evidence**. |

`python -m thermoagent report-doet` will populate the final README and paper
summary only from frozen analysis tables. It classifies the result as strong
AIJ direction, narrower publishable direction, or insufficient; it cannot
convert an unsupported hypothesis into prose by manual selection.
