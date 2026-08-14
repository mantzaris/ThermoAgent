# Functional dashboard artifact

The implementation is in `thermoagent/dashboard/`; its launcher is
`scripts/run-human-operator-dashboard.sh`. Replay mode requires no GPU and reads
one complete v3 event ledger. Live mode runs a bounded deterministic mock
scenario. Both modes expose identical schema-validated panels and support SVG
export.

This dashboard was engineering-tested with simulated operators. It has not been
evaluated with human participants. See `../protocol/future_human_study_protocol.md`.

