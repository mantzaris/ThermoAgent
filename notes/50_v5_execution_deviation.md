# V5 formal-development execution deviation

The first `development_primary` launch was interrupted at the client boundary
after 200 panels while its persistent terminal session continued remotely. A
second resume was started because the first session's process identifier was
misread as a system PID. The two writers raced: one completed panel outputs
while the other recorded `FileExistsError` failure manifests and both attempted
to replace the same aggregate CSV temporary path.

This is an infrastructure failure, not a scientific outcome. No comparative
development result was analyzed. The entire namespace is retained under
`results/human_operator_v5/raw/development_primary/`, including every failure
manifest and the incomplete aggregate report, and is ineligible for evidence.

Protocol 5.0.4 adds an advisory exclusive lock for each stage. The valid formal
stage is `development_primary_v2`, using untouched seeds 51101-51120. No
simulator mechanic, hypothesis, endpoint, model, gate, or threshold changed.
This response follows the prospectively specified invalid-stage policy.
