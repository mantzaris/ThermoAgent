# V13 methodology and immutable-parent audit

## Provenance

The fetched branch `origin/llm-agent-stochastic-thermodynamics-v12` and the
local V12 `HEAD` both resolved to
`457f6d635b60292623c8d97aa3b0c60d8d0aac4e` before V13 work began. The local
worktree, index, and `git diff --check` were clean. V13 was branched locally as
`collective-agent-statmech-v13`; no V13 staging, commit, push, tag, publication,
or pull request is authorized.

The existing RunPod path `/workspace/ThermoAgent` is a source snapshot rather
than a Git checkout. A file-level comparison found 100 overlapping V12 files,
no remote-only files, 12 later publication files present only in the pushed
commit, and 15 overlapping files whose RunPod copies predate the pushed
versions. The complete V12 configuration, source, and test roots matched the
pushed commit byte-for-byte. The older RunPod files and its 221 MiB external
V12 artifact tree remain untouched. V13 deployment is restricted to new V13
namespaces.

## Scientific reframe

V12 is the immutable discovery study. It established a causal local neighbor
response, coupling-associated increases in finite-size order, susceptibility,
and persistence, noise-associated decreases in the same observables, and a
positive bounded-memory contrast in coarse-grained path irreversibility. It did
not support nonreciprocity-induced irreversibility. V13 therefore treats
directed communication as a boundary finding and prospectively tests whether a
statistical-mechanical reduced description organizes collective LLM-agent
dynamics, controlled quenches, and recovery.

The primary claims are descriptive and explanatory. Categorical beliefs and
actions are realized Qwen choices. Reference energy is computed from the
symmetric influence layer and is not physical energy. Decoding temperature is
a decision-noise control, not thermodynamic temperature. Adjusted block
time-reversal divergence is coarse-grained pathwise irreversibility, not exact
total entropy production. Two finite sizes cannot establish a thermodynamic
phase transition.

## Prospective design and compute rationale

The operative amended formal design contains 32,672 LLM decisions: 288 isolated response draws,
12,800 Work Package A updates, 8,064 matched memory-quench updates, and 11,520
disruption/recovery updates. Work Package A uses 30-sweep modular trajectories,
20-sweep ring and ordered-relaxation subsets, rather than the suggested 35
sweeps, to retain three primary clusters per cell while fitting the stated
token and GPU budget. Work Packages B and C use 42- and 45-sweep trajectories.
At V12's metered generation rate this projects to roughly 14.6 GPU-hours before
the shorter V13 prompt/output savings. The engineering pilot is capped at 192
decisions and may inspect validity, occupancy, transition diversity, schedule
integrity, and runtime only.

The initial frozen protocol used four memory clusters. During the interrupted
microscopic-response block—and before any network panel existed—an audit found
that an exact one-sided sign-flip test with four pairs has minimum attainable
`p=1/16=0.0625`. The run was stopped under the protocol's mathematical-
inconsistency rule after 163 raw microscopic records. None were analyzed or
deleted. Amendment 01 increases the memory experiment to six clusters, for
minimum `p=1/64`, and offsets compute by narrowing only secondary ring and
ordered-start subsets. H1/H2 modular-primary and Work Package C designs are
unchanged. The operative design projects below 18 million prompt tokens.
The global ceiling counts the 104,455 pilot prompt tokens and all interrupted
records; Amendment 02 therefore limits formal-raw generation to 17,895,545
tokens. It was made before the amended run began and changes no scientific
design element.

During formal execution, and after two atomic network panels but before any
scientific outcome analysis, token-only accounting showed that the full
predefined design requires approximately 18.19--18.24 million total prompt
tokens. The engineering pilot had underestimated the network prompt length by
about one percent. V13 therefore invokes the prospectively allowed documented-
benchmark exception to the 18 million target. Scientific settings and sample
sizes remain frozen; the overage is bounded by atomic-panel granularity and is
reported as a transparent resource deviation. The 18-hour generation ceiling
is unchanged.

The halfway token-only audit increased the estimate to approximately 18.39
million after directly measuring the longer bounded-memory prompt. This
refinement did not use scientific outcomes. It is retained alongside the
earlier estimate so the evolution of the resource forecast is auditable.

Matched factor arms share graph, private fields, initialization, update
schedule, display counterbalancing, and inference seeds. The graph/environment
trajectory cluster—not an update, token, node, or window—is the independent
unit. H1--H3 are prospectively confirmatory; V12 estimates remain separately
labelled discovery evidence. H4--H7 are new V13 analyses.

## Macroscopic representation

The frozen reduced vector contains belief and action magnetization,
belief--action overlap, configuration entropy, entropy rate, total correlation,
reference energy per agent, energy variance, susceptibility, a spatial
correlation summary, and disagreement. Entropies use fixed coarse-graining and
are interpreted jointly with order, energy, and correlations. Nominal-manifold
distance is fit from nominal training trajectories only. Simple, order-only,
and full statistical-mechanical representations are compared with
leave-one-graph-cluster-out linear models; no test-trajectory feature selection
is permitted.

## Disruptions

The only formal disruptions are fixed before outcome inspection: a private
field sign reversal, removal and restoration of all inter-community delivery
paths, and independent flipping of 50% of delivered categorical packet fields
with packet count and byte length preserved. Each Work Package C trajectory is
15 sweeps baseline, 15 disruption, and 15 recovery. These abrupt quenches test
response and recovery, not early warning. Work Package B applies a matched
14/14/14 field quench to Markovized and bounded-memory agents.
