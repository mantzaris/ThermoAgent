# V8 claims-to-evidence matrix

| Claim | Status | Evidence |
|---|---|---|
| Actual deterministic binary belief serialization was implemented and audited | Supported engineering claim | `thermoagent/v8_wire.py`; `tests/test_v8_wire.py` |
| The corrected trigger became information-score driven | Supported development-only mechanism observation | `tables/trigger_feasibility.csv` |
| The candidate satisfied nominal communication feasibility | Failed | `negative_results/v8_stop_decision.json` |
| H1 communication-efficient estimation | Untested formally | `statistics/v8_pilot_no_go_summary.json` |
| H2 generalized-information superiority | Untested formally | no frozen matched-byte comparator |
| H3 frozen learned-agent retention | Untested | `training/NOT_RUN.md` |
| Validation or locked-holdout replication | Untested | `validation/NOT_RUN.md`; `holdout/NOT_RUN.md` |

## Prohibited extensions

- Do not call pilot byte-reduction point estimates confirmatory H1 support.
- Do not claim generalized entropy beat a frozen non-entropic comparator.
- Do not claim multi-seed RL, Qwen, human-operator, or real-world evidence.
- Do not omit operational, forwarding, dropped, stale, header, or integrity traffic.
- Do not reinterpret the failed nominal-traffic gate after observing it.
