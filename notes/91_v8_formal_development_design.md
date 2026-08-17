# V8 formal development design

Recorded before execution of the 24-panel-per-application method-selection
batch: 2026-08-17T00:31:09-04:00.

This is development evidence, not validation or holdout evidence. The V8
validation and holdout seed namespaces have not been generated or executed.

## Independent panels

The design in `configs/v8_development.yaml` contains 24 independent panels for
each primary application (48 total). Within each application it contains eight
small, eight medium, and eight large V7 environments. Coupling,
fragmentation, and network disruption each take low, medium, and high values.
The development topology families are random-geometric and small-world for
humanitarian logistics, and grid and scale-free for utility restoration.
Modular graphs are reserved for fresh validation instances if progression is
unlocked.

The environment panel (application, graph instance, stochastic tape, and
seed), not a message, edge, agent, action, or time step, is the independent
unit. All scheduler arms within a panel use the same environment seed and
stochastic tape.

## Candidate set and selection

The batch includes always-on and no-exchange anchors; the calibrated full
generalized-information trigger; KPI-change, predictive-uncertainty-change,
L1 belief drift, periodic, random, and age-of-information non-entropic
schedulers; V7 Shannon-change, Jensen-Shannon, and Tsallis-spectrum ablations;
and a clearly nondeployable offline oracle.

The primary payload for every arm is the same deterministic uint8-simplex
wire representation. Only scheduling changes.

The primary generalized candidate was provisionally calibrated in pilot 3 at
`tau_on=0.125`, `tau_off=0.04`, cooldown 2, and maximum silence 30. Formal
development must confirm eligibility across both applications. The strongest
non-entropic comparator is selected lexicographically among methods within a
matched-byte neighborhood: absolute log-byte-budget distance first, then
distributed disagreement error, disruption delay, downstream loss, and a
stable method-name tie break. Selection uses development only and will be
frozen before validation.

## Pilot-derived precision plan

Pilot 3 contained only four panels per application and is not used for formal
inference. Its generalized-trigger standard deviations for sketch-byte
reduction were approximately 0.099 (humanitarian) and 0.061 (utility), while
the primary estimation-error increase was approximately 0.001 or less on
average. Twenty-four development panels per application reduce the nominal
standard error for byte reduction to roughly 0.020 and 0.013 before allowing
for topology/regime heterogeneity. Validation and holdout retain the requested
minimums of 30 and 40 independent panels per application; interval-width and
power checks after development may increase, but never reduce, those counts.

## Pre-execution checksums

- Development configuration: `db5996f594e1153c6437520d89f8f4b4c34e0e43ba58b832865df57705a24733`
- Trigger source: `62dd0482aef554cca895af35210771d7cafba89638269bed48df224279340aa2`
- Wire source: `8730ce8f5dbde857cc28a00d6911ef239375e4fd5eab61606b11ce105f82eee7`
- Monitoring source: `8fa60fab990064b5812e7251dedca219b180707ad78eb0f8007ff0d9366ec4dc`
- Episode source: `8106cad562c9da5a2b26364592353b7fa2fc06d8b837c750056ceed65de65000`
- Analysis source: `89b92d74849ca64f7e24cfc529ff2bdd688e05aa3d621992052b5195433fc798`
- Sequential-policy source: `4f5396eed43e0aacdd6c1e12c73f4251974e8e3e48c9f98f863c403714e3cea0`

No threshold or comparator may be revised in response to the results of this
batch without classifying the subsequent run as a new, retained development
iteration. No V8 validation or holdout outcome may be opened before a clean
protocol freeze.
