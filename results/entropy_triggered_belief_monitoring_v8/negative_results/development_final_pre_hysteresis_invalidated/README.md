# Invalidated pre-freeze final-development attempt

The last atomic supervisor update recorded 106 of 240 originally scheduled
arms and zero episode failures. Disk inspection after cancellation found 108
complete atomic episode manifests plus two partial run directories; all 110
directories are retained. It is ineligible for V8 trigger selection because a
pre-outcome source audit established that a successful send did not enter the
hysteresis off state, so `tau_off` was not operational. The runner was stopped
before aggregate outcomes were generated or inspected.

The exact complete and partial raw directories, configuration, candidate
registry, and last supervisor status are preserved. No completed episode was deleted. The
replacement `development_final` stage reruns the complete candidate-by-panel
design with corrected Schmitt-trigger state, adds the predeclared KPI 0.10
budget interpolation arm, and uses losslessly replayable dynamic-delta ledgers
to avoid full-ledger compression dominating execution time.

This is an engineering invalidation before protocol freeze, not a selective
outcome rerun and not validation or holdout evidence.
