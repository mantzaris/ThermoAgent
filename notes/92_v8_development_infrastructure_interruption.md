# V8 development infrastructure interruption

The first formal-development invocation was interrupted after 261 of 624
scheduler arms completed. There were zero completed-run failures. Six workers
were writing temporary XZ ledgers when the interruption occurred; none had an
`episode.json` completion manifest.

The six incomplete directories, their partial bytes, and checksums were moved
intact to:

`results/entropy_triggered_belief_monitoring_v8/negative_results/interrupted_development/`

No completed episode was rerun or removed. The identical 624-task
configuration was resumed with its restart guard, which skipped the 261
completed run IDs and reran only the six incomplete attempts plus tasks that
had never started.

The cause was storage-only: eight concurrent preset-9 XZ encoders caused
memory pressure and low throughput on large panels. The repair changed the
worker count from eight to four and the per-run *lossless* XZ preset from nine
to three. The logical JSONL, event order, simulator state, seeds, trigger
configuration, policy, metrics, and scientific comparisons were unchanged.
Every completed ledger is subsequently decompressed, replayed, and packed
losslessly with member-level hashes. This infrastructure change was made
without inspecting comparative development outcomes.
