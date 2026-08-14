# Executed ThermoHITL development protocol

The protocol was recorded before v3 outcome experiments in
`notes/21_thermohitl_protocol.md`. This result-facing copy records the executed
decision boundary.

All six gates had to pass: engineering; actionability (90% first pass, 98%
after one repair, 70% accepted-to-next, 30% accepted-to-demand plus causal
effects); coordination necessity (at least 5% aggregate improvement in both
applications and two regimes each); bounded-human causal usefulness in both
applications/two regimes; same-information thermodynamic value (delta AP/AUC at
least 0.05 or utility at least 5% in both applications); and trigger feasibility
(nonzero/timely activation, at most 10% pre/nominal false activation, at least
one causal effect).

The trigger candidate was `tau_on=1.5`, `tau_off=0.6`,
`actionable_tau_on=1.1`, dwell 2, cooldown 3. Energy weights were fixed at
`.24/.22/.16/.14/.12/.12`. Qwen was fixed to
`Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`.

Final disposition: gates 1–4 and 6 passed; Gate 5 failed commercially. The
validation/holdout protocol was never frozen and no such outcome was opened.
