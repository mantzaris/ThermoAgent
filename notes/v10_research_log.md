# V10 research log: nonreciprocal LLM-agent entropy production

## 2026-08-17: provenance and scope

The pushed V9 reference was verified before any V10 edit. Both local `HEAD` and
`origin/statistical-mechanics-agentic-systems-v9` resolved to
`8e8315d25684a1c582c6a7b46fbb5786bc3f0557`, the worktree and index were clean,
and `git diff --check` passed. The local branch
`llm-agent-entropy-production-v10` was created from that commit. V9 contains
historical text saying it was uncommitted and unpushed; V9 was not edited. Its
correct committed and pushed provenance is recorded only here and in V10.

The user prohibited staging, committing, and pushing. No such operation is part
of the V10 scripts. Detailed artifacts use `/tmp/ThermoAgent-v10-artifacts`
because the local `/workspace` is not writable. A clean publication export uses
`/tmp/ThermoAgent-JSTAT-v10-clean-export` for the same reason. Both locations
are outside Git.

## V9 normalization audit

The V9 exact transition matrix chooses one of the `2N` belief/action variables
uniformly and retains self transitions. Its Schnakenberg expression is therefore
in nats per attempted variable update. It is neither per accepted flip nor per
agent nor per sweep. V10 uses these explicit conversions:

- one sweep: `2N` attempted updates;
- EPR per sweep: `2N` times EPR per update;
- EPR per agent per sweep: `2` times EPR per update.

The V9 large-system field named `entropy_production_per_update` is a sampled
mean local log-rate ratio. In a stationary, fully observed Markov chain its mean
equals total EPR because the mean system-entropy boundary term vanishes. It is
not the same computational object as the dense exact current sum. V10 calls it
*pathwise stationary irreversibility* and validates it against exact kernels.

V9's directed communication construction assigns each reciprocal edge pair
weights `1+alpha` and `1-alpha`. It preserves the support, two directed message
opportunities, pairwise total weight, and global total weight. It does not
preserve every node's in/out strength. V10 states this limitation and reports
the row-divergence and antisymmetric spectral diagnostics.

V9's N=3 table showed EPR divided by `alpha^2` near 0.11--0.12 across several
asymmetries. That was descriptive numerical evidence, not a derivation. V10 does
not alter it.

## Perturbative derivation fixed before the formal CPU run

For a row-stochastic discrete kernel

`W(alpha) = W0 + alpha V + O(alpha^2)`

with reversible stationary distribution `pi0`, write

`pi(alpha) = pi0 + alpha r + O(alpha^2)`.

Differentiating stationarity and normalization gives

`r (I-W0) = pi0 V`, and `r 1 = 0`.

V10 solves this constrained linear system directly. With reciprocal equilibrium
flux `q_xy = pi0_x W0_xy`, first-order flux

`f_xy = r_x W0_xy + pi0_x V_xy`,

and current derivative `j_xy = f_xy-f_yx`, both stationary current and affinity
vanish at `alpha=0` and begin at first order. Expanding the Schnakenberg sum
therefore gives

`sigma(alpha) = C alpha^2 + O(alpha^3)`,

where

`C = (1/2) sum_{x,y:q_xy>0} j_xy^2/q_xy >= 0`.

The linear physical term vanishes because it is a product of a first-order
current and a first-order affinity around a detailed-balance reference. This is
a discrete-time finite-Markov-chain result. The coefficient is decomposed by
whether an edge flips a belief or action variable. Action-layer currents can be
nonzero even though the perturbation enters the communication layer, because
the stationary response propagates through the belief--action coupling `K`.

An independent central difference of the full heat-bath kernel agreed with the
analytical derivative to `3.83e-11` in the retained engineering pilot. The
stationary-response equation closed below `1e-16`.

## LLM-agent boundary

The V10 LLM harness is not a text wrapper around an Ising-selected action. Each
agent owns an independent private observation, memory, belief, committed action,
inbox, outbox, role, and typed authority. The scheduler chooses only an update
opportunity. Qwen returns a schema-validated belief, action, commitment status,
outgoing signal/message, tool action, confidence, and reason code. The agent's
returned choice is applied without centralized substitution. Controlled
micro-update mode changes only the scheduled belief or action; full-turn mode
can change both and communicate.

`A[i,j]` means sender `j` influences recipient `i`. Each delivered natural-
language message carries locally visible influence metadata, encoded with a
deterministic binary header for real byte accounting. Reciprocity conditions
keep support and message opportunities fixed. Counterfactual tests mutate one
agent's private evidence and show that peer private vaults are unchanged.

The pinned execution design uses `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, NF4 quantization, BF16 computation,
inference sampling temperature 0.65, and top-p 0.90. Inference temperature is
explicitly distinct from the effective decision temperature fitted from the
local response curve.

## RunPod disposition and compute decision

The established alias still resolves to `213.173.109.33:19465`, but the
authorized SSH attempt returned `Connection refused`. No repository lifecycle
or API credential capable of starting that existing Pod was found. No new Pod
was created, purchased, resized, or deleted. Local hardware has no GPU and the
local environment intentionally lacks Torch/Transformers. Accordingly:

- CPU theory, exact calculations, trajectory estimators, tests, protocol,
  figures, and manuscript preparation proceed;
- Qwen pilot and formal stages remain not run;
- no Qwen result, local-policy fit, Markov adequacy result, or LLM entropy-
  production claim will be fabricated.

The prospective Qwen plan contains 12,568 primary calls and a one-repair reserve
up to 15,082 calls. It must first profile under 20 single-GPU hours on the
existing RTX 4090 and pass schema, action-diversity, option-bias, private-
evidence, message-use, and non-substitution gates.

## Development and freeze chronology

The retained development pilot ran 12 exact cells and six trajectory profiling
cells. It was used only for derivative verification and runtime sizing. One
combined V9/V10 test command initially failed during collection because both
directories contained a module named `test_agents.py`; no scientific test ran
under that failed collection. The command was corrected prospectively to use
pytest's `--import-mode=importlib`. The next combined run passed 54 tests.

After the engineering checks, protocol `v10.1.0-frozen-cpu` was frozen with:

- protocol SHA-256: `fe1b8a599f82da6ce7b69ece6479b90dc7ddcb923b90e8e5a9fd3249be8efbf5`;
- scientific-source SHA-256: `e2f566aa52d3728c12796663650b764445bff317bfa45fdf0af5711e024e1c03`.

The formal CPU run uses new V10 seeds and writes phase-atomic aggregate rows
outside Git. It does not touch V1--V9.

## Literature and novelty audit

The closest theoretical precedents materially narrow any novelty claim.
Glauber established local stochastic Ising dynamics; Schnakenberg formulated
Markov-network currents and entropy production; Blume connected logit revision
to statistical mechanics of strategic interaction. Asymmetric kinetic Ising
models are established, including modern mean-field work by Aguilera and
coauthors. Fruchart and coauthors developed nonreciprocal collective phase
transitions. Most importantly, Di Carlo (2025) explicitly reports quadratic
near-reciprocal entropy production in a nonreciprocal metric kinetic Ising
model. V10 therefore cannot claim the quadratic onset itself as new.

The narrower prospective addition is the exact finite-kernel stationary-
response coefficient for a coupled belief--action model, its layer
decomposition, and—only if executed and supported—the controlled realization
with independent LLM agents. Roldan and Parrondo justify trajectory time-
reversal KL as a lower bound under coarse observation; Kawaguchi and Nakayama
show why hidden entropy production matters under coarse graining. Recent 2026
preprints already study LLM populations through Ising-like binary alignment,
especially De Nobili's *Collective Alignment in LLM Multi-Agent Systems*.
That work makes an LLM/statistical-physics juxtaposition non-novel by itself.
V10 differs only if it sustains distinct belief/action variables, private
memory and messaging, operational typed actions, nonreciprocal currents, and
qualified irreversibility inference.

The official JSTAT site was checked on 2026-08-17. JSTAT accepts LaTeX, uses
numeric references, encourages persistent identifiers and data/code statements,
and directs authors to IOP preparation guidance. IOP currently accepts common
LaTeX variants and does not require journal-like typesetting at initial
submission. The V10 draft therefore uses a portable 12-point article layout;
the official `iopjournal` class can be adopted at submission without changing
the scientific content.

## Formal and final disposition

The phase-atomic formal CPU study completed without a rerun or checksum change:
96 primary exact cells, 336 coefficient-grid cells, 144 exact-size cells, 1,280
independent trajectory cells, and 104 synthetic-estimator cells. Its completion
manifest records 2,213.58 wall-clock seconds and 2,016.98 summed trajectory-cell
CPU seconds. A resumability check returned the same five artifact hashes, the
same protocol hash, and the same scientific-source hash.

The exact reciprocal null was at most `1.100864565477282e-31` nats per attempted
update. Across eight directed orientations, the mean perturbative coefficient
was `0.1108860741`, with a 10,000-replicate orientation-bootstrap interval of
`[0.0964557513, 0.1205295952]`. For `alpha <= 0.02`, the maximum relative error
between exact `sigma/alpha^2` and the predicted coefficient was `0.000562956`.
The coefficient varied substantially with topology, temperature, and `K`, but
its correlation with the selected antisymmetric spectral norm was only
`-0.221`; no simple spectral scaling law or phase transition is claimed.

The focused V9/V10 regression set passed 56 of 56 tests with no failures,
errors, or skips. The complete repository collection contained 470 tests: 456
passed, 12 failed because the intentionally lean local CPU environment does not
contain PyTorch, and two were skipped. All 12 failures occur while importing
`torch` from legacy PPO tests; the external JUnit report is retained. Installing
a large framework solely to mask this known environment boundary was rejected.

Seven paper-facing figure PDFs and the ten-page manuscript were opened,
font-checked, text-checked, rendered externally at 300 DPI, and manually
inspected at original resolution and 300 DPI. Initial presentation defects were
corrected by a post-freeze renderer operating only on unchanged source CSVs;
the frozen scientific checksum did not change. No final clipping, overlap, tiny
labels, missing uncertainty, rasterized text, or unembedded fonts was observed.

The authorized RunPod endpoint remained unavailable, so Qwen calls, prompt and
generated tokens, GPU hours, and model cost are all zero. H5--H7 therefore
remain untested. The defensible result is a theory and exact/numerical
stochastic-agent contribution: the finite-kernel quadratic coefficient and its
belief/action transition-layer decomposition. It is not evidence that actual
LLM-agent trajectories exhibit the predicted response. A JSTAT paper using the
requested LLM title remains blocked on the preregistered Qwen calibration,
Markov-state audit, reciprocal bias floor, nonreciprocal trajectory experiment,
and graph/prompt/seed replication.

The lean non-Git export contains 85 files and approximately 1.04 MB at
`/tmp/ThermoAgent-JSTAT-v10-clean-export`. The compact repository-facing V10
package is also approximately 1.04 MB. The external manifest indexes 33 raw,
test, completion, and QA artifacts (6,344,512 bytes); every recorded size and
SHA-256 was rechecked with zero mismatches. The final established-SSH retry was
again refused. No visible Python, pytest, Qwen, CUDA, or experiment process and
no tmux server remained. Nothing was staged, committed, pushed, tagged, or
released.

## 2026-08-17 continuation correction: RunPod proxy and Qwen pilot no-go

The preceding endpoint disposition is superseded by this timestamped
continuation, without rewriting the earlier observation. The local SSH alias
still targets the stale public mapping `213.173.109.33:19465`, which refuses the
TCP connection before an SSH handshake. The existing RunPod proxy identity at
`ssh.runpod.io`, using the established RSA key and a forced terminal, reaches
the same online Pod. On connection, host `2acac16f37c7`, the existing
`/workspace/ThermoAgent`, an idle NVIDIA RTX 4090, the pinned model cache, and a
listening internal SSH daemon were verified. Thus the Pod and configuration
were not generally unavailable; only the historical public IP/port forwarding
route was stale. RunPod's proxy SSH was used for command execution and
`runpodctl` relays for compact file transfer because proxy SSH does not provide
SCP/SFTP.

One diagnostic command printed the running Jupyter command line, including its
session token, into tool output. The token is intentionally not reproduced
here. It must be treated as exposed and regenerated if the diagnostic
transcript is shared. Jupyter was not stopped or reconfigured because that
would be an external lifecycle action beyond the research task.

The Qwen pilot chronology is retained rather than overwritten:

1. Evidence attempt 1 loaded the model but failed before a scientific decision
   because deterministic CUDA execution required
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Its log and exit status remain external.
2. Evidence attempt 2 completed 120 decisions but aliased the six evidence
   fields with the two display orders. Its apparent option-order effect was not
   identifiable and the attempt is classified as a confounded design.
3. Evidence attempt 3 fully crossed all six fields with both display orders at
   the unchanged 120-decision budget. It passed: 96.7% first-pass validity,
   100% after one allowed repair, private-evidence response difference 0.35,
   nontrivial tool-action fraction 0.40, field slope 1.2379, effective decision
   temperature 1.6157, and option-order slope -0.1163 versus the frozen
   absolute bound 0.15. This used the prompt immediately before the later
   message clarification and is development evidence, not final held-out
   calibration.
4. Message attempt 1 used 24 matched left/right-message pairs but a parity bug
   initialized every prior belief to `plan_left`; it is excluded and retained.
5. Message attempt 2 balanced priors inside application, paraphrase, and order
   cells. Qwen retained the prior in all 48 decisions, yielding zero message
   response against the predeclared 0.20 minimum.
6. Before any formal outcome, one prompt clarification identified inbox entries
   as new evidence and defined `influence_weight` as a locally visible
   reliability coefficient. Message attempt 3 again retained the prior in all
   48 decisions. Its response difference, paired switch fraction, and
   directional-pair fraction were all zero. The gate was not lowered.

The amendment `v10.1.3-llm-pilot-amendment` records those repairs, the unchanged
gate, matched-alpha inference seeds, atomic formal checkpoints, and the planned
13,728-decision formal design. Because the final delivered-message gate failed,
the LLM source was not formally frozen and the large empirical-kernel/dynamic
trajectory study was not started. H6 and H7 remain untested. This is the exact
prospective stopping behavior required by the protocol: nonreciprocity acts
through directed delivered evidence, so a model that does not use that evidence
cannot credibly test nonreciprocal LLM probability currents.

Across five completed pilot attempts, Qwen produced 384 decisions and 388 model
calls including four repairs, consuming 167,031 prompt tokens, 34,224 generated
tokens, and 677.31 seconds of measured generation latency (0.188 GPU-hours).
Including model loads, allocated GPU time is estimated at 0.23 hours, or about
USD 0.078--0.159 at the project's documented USD 0.34--0.69 hourly range. Raw
prompts and completions remain outside Git under
`/workspace/ThermoAgent-v10-artifacts`; the repository contains only aggregate
pilot tables, source data, and hashes.

## 2026-08-17 final verification

The final focused V9/V10 regression suite passed 63 of 63 tests with no
failures, errors, or skips. Eight canonical vector figure PDFs and the rebuilt
11-page manuscript passed opening, extractable-text, embedded-font, and render
checks. Every figure was inspected at native publication size and 300 DPI; the
manuscript was re-inspected after its final figure rebuild. The final manuscript
SHA-256 is
`46b32dcfbb36d839a9f3f74941d80dcc7be34bf426c213817383353ca0e4020b`.

The final lean export was regenerated from a fresh directory at
`/tmp/ThermoAgent-JSTAT-v10-clean-export`: 96 files, approximately 1.25 MB, and
no Git metadata. The repository-facing V10 package remained approximately
1.36 MB, well below the 25 MB constraint. Final Git and process checks are
reported in the handoff; V10 remained unstaged, uncommitted, and unpushed.
