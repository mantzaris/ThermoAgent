# V11 research log

## 2026-08-17: provenance and stage-zero audit

- Fetched `origin` without rewriting history.
- Verified local and remote V10 commit
  `4d372f00837bf75f90882392a92feac87dbc84b2`.
- Verified a clean V10 worktree before branching.
- Created local branch `evidence-grounded-llm-entropy-v11` from that commit.
- No V1--V10 source or result artifact was modified.
- The existing RunPod `2acac16f37c7` is reachable through the established
  RunPod SSH proxy. Its RTX 4090 was idle, the pinned Qwen cache was present,
  and no research Python, CUDA, or tmux process was active. The remote
  `/workspace/ThermoAgent` deployment is a source copy without Git metadata.
- V10's message null was audited as an interface-specific binary-switch null,
  not a calibrated absence-of-evidence-use result.
- The modular scaling anomaly was diagnosed before V11 outcome generation as
  increasing-degree, unnormalized-coupling freezing with metastability.

All V11 raw prompts, completions, per-decision rows, pilot logs, and rendered QA
images will remain outside Git under `/workspace/ThermoAgent-v11-artifacts/` on
the existing Pod or `/tmp/ThermoAgent-v11-artifacts/` locally. Repository-facing
files will contain only compact aggregate evidence and reproducibility metadata.

## 2026-08-17: implementation and retained pilot launch

V11 now uses typed likelihood-bearing evidence packets, deterministic binary
serialization, continuous probability elicitation, a derived 0.5-threshold
binary belief, separately typed action and commitment fields, isolated private
state, and external-only raw Qwen records. The retained engineering design has
128 decision requests crossing two semantic framings, two paraphrases, both
option orders, both private-signal directions, no-message and placebo controls,
and left/right packets at reliabilities 0.55, 0.70, and 0.85. Within a matched
cluster, the inference seed and all non-treatment prompt content are fixed.

Pilot attempt 1 failed before model loading or any decision because the Pod's
system Python lacks pandas. Its log and exit record are retained externally as
`logs/pilot_attempt_1_wrong_interpreter.*`. The existing project virtual
environment already contains the required scientific and model packages, so no
package was installed. Attempt 2 uses that environment, the existing RTX 4090,
the cached pinned Qwen revision, and `/workspace/ThermoAgent-v11-artifacts/`.

Attempt 2 was prospectively stopped after every completed request remained
invalid after repair. Inspection showed coherent probabilities and actions but
`outgoing_evidence: null` paired with `outgoing_abstention: false`: the model
interpreted the latter as abstention from the overall decision. Because the
nullable evidence field already expresses send-versus-abstain, the redundant
Boolean was removed. Abstention is now defined only as
`outgoing_evidence: null`. No valid scientific pilot row existed before this
amendment; all attempt-2 raw records and partial rows are retained externally
under an explicitly invalidated namespace.

Attempt 3 verified that the nullable-only schema works: its first 21 requests
were 100% valid after repair and exhibited continuous-probability variation.
However, first-pass validity was 85.7% because some outputs retained the now
forbidden Boolean, and no response supplied an accepted outgoing packet. The
attempt was therefore stopped before completion and retained as a second
invalidated engineering pilot. The final pilot prompt explicitly prohibits the
redundant key and defines the role responsibility to copy the agent's own valid
private packet unchanged, while preserving null as an available abstention for
malformed or unavailable evidence. This change precedes the decisive
qualification and does not alter any observed scientific threshold.

Attempt 4 made outgoing packets explicit but required the model to transcribe a
nested immutable packet. At 28 retained requests it had 100% validity after
repair but only 71.4% first-pass validity; accepted sends were 42.9%. The pilot
was stopped because transcription errors, rather than agent choice, dominated
the communication metric. V11 now exposes the simpler typed action
`send_private_evidence` versus `abstain`. Choosing the former invokes a
deterministic role-authorized tool that serializes the agent's own immutable
packet. The scheduler still cannot decide whether to send or alter the content.
This is the final interface amendment permitted before completing the retained
pilot and freezing the decisive qualification.

## 2026-08-17: retained pilot result and qualification freeze

The final 128-request pilot completed without selective reruns. First-pass
validity was 97.66%, validity after one repair was 99.22%, and accepted evidence
sending among valid responses was 98.43%. The primary signed evidence effect
was `0.1251` log-odds (95% cluster bootstrap CI `0.0570` to `0.2125`) across 16
matched clusters. It replicated in route viability (`0.1829`, CI `0.0601` to
`0.3334`) and repair hypotheses (`0.0673`, CI `0.0157` to `0.1289`). Mean
effects increased across reliabilities 0.55, 0.70, and 0.85: `0.0600`, `0.1449`,
and `0.1562`. Reported probabilities were poorly calibrated to thresholded
choice frequencies (ECE `0.338`), and placebo message presence caused a large
nondirectional/rightward shift. These are explicit risks, not discarded rows.

Before generating decisive qualification data, protocol
`v11-qualification-1.0` froze a directional placebo contrast, 48 independent
matched clusters, 864 requests, all treatment conditions, model settings, and
gates. The practical signed and placebo-adjusted log-odds minima are both
`0.10`; both also require positive 95% cluster intervals. The protocol SHA-256
is `7bd4082f9d085222e22d195e3ff603f0f76e27c36b347bfc4132fbb164a3d03f` and
the execution source-tree SHA-256 is
`7b7189aff31b88d437c6076db33686fd3b73efb2738a5728b073215afd31905e`.

Before any qualification call, cross-host verification showed that the first
source hash incorrectly included interpreter-specific `__pycache__` bytecode.
The checksum routine was corrected to exclude `.pyc`/`.pyo` files and was
re-run on both hosts. The protocol hash is unchanged; the authoritative
qualification execution-source hash is
`1f7bcd164e4f07f033def27a8236ed0995413b4c12dcd6203549ca08d48d395e`, identical
locally and on the Pod. The earlier source hash is retained above as a
superseded manifest diagnostic, not an execution provenance claim.

The decisive qualification was launched once in the persistent tmux session
`thermov11-qualification`. The RunPod SSH proxy requires an interactive PTY;
an attempted noninteractive status command produced the misleading message
that the client did not support PTY, even though the Pod itself remained online
and GPU-active. No Pod, package, or credential change was needed.

## 2026-08-17: qualification-time isolated reporting and formal-template work

While the frozen qualification source continued unchanged on the Pod, local
work was restricted to code that is not imported by the running qualification
process and to a prospective formal-stage template. These changes will not be
synced to the Pod unless the qualification gate passes, and would then receive
a separate formal execution checksum. They do not retroactively replace the
qualification source hash above.

- Invalid formal LLM responses are specified as failed update attempts with an
  explicit state-preserving self transition; no scheduler action is substituted.
- The formal estimator discards a declared 32-turn burn-in but retains invalid
  attempts in the path and in per-attempt-update normalization.
- Local signal reliability varies over the declared set 0.55, 0.70, and 0.85,
  making the cyclic reliability-label permutation a real evidence-destroying
  control rather than an identity transformation.
- The pre-outcome formal template uses 32 matched graph/environment clusters,
  two sizes (6 and 8 agents), ring and modular skeletons, four nonreciprocity
  values, 128 turns per primary panel, and 72 matched control panels. It projects
  20,992 decisions, approximately 10.3 GPU hours, and USD 3.50--7.11 at the
  repository's previously documented hourly range, below the 20-hour cap.
- Prespecified formal diagnostics now compare zero-intercept linear,
  quadratic, and mixed low-alpha response models by leave-one-cluster-out
  prediction, test application-specific paired effects, and quantify actual
  message and wire-byte imbalance across reciprocity arms.
- A scripted Bayesian engineering check verified causal communication use:
  delivered packets reached 76 of 128 eligible recipient turns, and changing
  only network orientation changed nine later local decisions in each semantic
  application while message counts remained exactly matched. This is an
  implementation check, not LLM evidence.

The local host's unversioned `python` command resolves to Python 2, so V11 shell
entry points now select `${PYTHON_BIN:-python3}`. This repair affects only
reproduction ergonomics. The qualification job already uses the existing
RunPod project virtual environment explicitly.

The current JSTAT and IOP instructions were checked before manuscript work.
They permit standard LaTeX, request reviewer-readable formatting (IOP recommends
at least 12-point body text), require embedded figures in the manuscript, and
request a data/software availability statement. The V11 draft will therefore
use a conservative standard article class with standard packages rather than
depending on an unverified local journal class. Current nearby primary
literature includes the asymmetric kinetic-Ising entropy-production study of
Di Carlo (2025), the 2026 statistical-physics study of LLM collective alignment,
and 2026 preprints on LLM-network conformity and observable belief revision.
These works narrow the novelty claim: V11 does not claim that quadratic onset
near detailed balance, LLM conformity, or confidence elicitation is new.

## 2026-08-17: decisive qualification and prospective stop

The one-shot 864-request qualification completed with no selective reruns.
There were 842 first-pass-valid responses, 21 responses valid after one repair,
and one response invalid after repair. The resulting validity rates were
97.45% and 99.88%. Provider accounting records 886 calls, 559,895 prompt
tokens, 76,620 generated tokens, and 1,536.064 seconds of generation latency.

The frozen pooled signed log-odds estimate was 0.1343 (95% cluster-bootstrap CI
0.0915--0.1811; 48 matched clusters); the placebo-adjusted estimate was 0.1343
(CI 0.0905--0.1816). Both semantic framings had positive intervals. However,
the reliability-LLR slope was 0.04965, just below the frozen 0.05 minimum, and
the mean at reliability 0.85 declined from the 0.75 level by more than the
allowed 0.03. The binary right-choice fraction was 0.9224, above the frozen
0.90 ceiling. Monotonicity and transition diversity therefore failed, so the
all-components progression rule locked the formal network experiment.

A clearly post-gate diagnostic was performed only to characterize the no-go.
Right-supporting evidence had a mean signed log-odds effect of 0.5574, whereas
left-supporting evidence had -0.2939, indicating a strong direction/prior bias.
Reported probabilities were also poorly calibrated to repeated binary choices
(ECE 0.2955). These diagnostics did not alter the gate and will not be used to
retune V11.

Only four paper-facing figures were generated: architecture, evidence process,
belief response, and calibration. No placeholder network, entropy-production,
control, or Markov figure was created. All four PDFs passed open, text, font,
and 300-DPI render checks and were manually inspected at native and 300-DPI
resolution. Raw prompts and completions remain external.

## 2026-08-17: accounting and preservation

The retained final pilot plus qualification used 992 requests, 1,017 calls,
639,368 prompt tokens, 87,726 generated tokens, and 1,757.620 seconds (0.4882
hours) of measured generation latency. Three invalidated interface pilots add
115 requests and 173 calls. Their early provider failed to persist token usage
for calls whose responses remained invalid after repair, so exact all-pilot
token accounting is not recoverable; recorded totals are disclosed as lower
bounds rather than reconstructed as exact.

The RunPod raw tree has 992 files (3,753,092 bytes; tree SHA-256
`8d4116171fdde105bda529465810faf50a9f2a15f1403bdaf2636a3df1f3e4d3`).
Invalidated pilots have 118 files (571,918 bytes; tree SHA-256
`37733b518368beb2e046c2e2e9eedbfe62b8984920acb2908765b286f6e96540`).
The compact checksum record is repository-facing; raw records remain under
`/workspace/ThermoAgent-v11-artifacts/`.
