# V8 pilot iteration 1 and estimand repair

Pilot iteration 1 completed 196/196 scheduler arms with zero execution failures.
It is retained under `results/entropy_triggered_belief_monitoring_v8/pilots/`
and `raw/pilots/`. Its initial selector returned no eligible trigger.

Two implementation defects were found before any protocol freeze, validation,
or holdout execution:

1. The candidate registry joined on the trigger digest alone. Encoding is not a
   `TriggerConfig` field, so FP32, FP16, and uint8 always-on rows shared that
   digest and were duplicated by the analysis join. The raw episodes were
   unique and correct; the derived encoding summary was not.
2. The evaluator pooled target was calculated from agent beliefs after received
   messages had modified them. Thus each scheduler was scored against a target
   partly produced by that scheduler. This violated the common-estimand
   requirement. V8 now defines the evaluator target as the reliability-weighted
   pool of current independently delivered private evidence, before exchange.
   Agent operational beliefs may still change after explicit message delivery.

The first selector and feasibility files are preserved with an
`initial_invalidated` suffix. Pilot iteration 2 uses new seeds and deterministic
XZ ledgers. It does not overwrite iteration 1.

The absolute p95 error limit of 0.08 was also found to be below the always-on
topology/latency floor (approximately 0.11--0.14 in iteration 1), making it an
invalid absolute noninferiority criterion. Before iteration 2 outcomes, it was
replaced by two joint limits: absolute p95 no greater than 0.16 and paired p95
increase over always-on no greater than 0.01. The primary mean integrated-error
margin remains 0.02. This change is anchored to the always-on reference scale,
not to whether a generalized trigger passed.
