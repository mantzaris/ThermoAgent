# V15 append-only research and execution log

## 2026-08-21: provenance and RunPod recovery

- Verified local and remote V14 parent commit `103e4c4598ecc26a98c37a8d03ee3663f9be1070`; created local branch `jstat-scientific-audit-v15` without modifying history.
- Confirmed the starting worktree was clean. V1-V13 namespaces remain outside the V15 edit set.
- Reconnected to the existing RunPod after its stop/start cycle. No new Pod was created. The RTX 4090 was visible and idle; the persistent `/workspace` volume, `/workspace/ThermoAgent`, and V12-V14 artifact roots remained present.
- Verified the retained V14 formal, raw-formal, and pilot trees against committed manifests. Verified all retained V12 and V13 stage-tree hashes. V14 replay covers 17,280 rows in 24 trajectories with zero mismatches.
- Verified the existing environment: Python 3.12.3; PyTorch 2.8.0+cu128; Transformers 4.55.4; bitsandbytes 0.47.0; CUDA 12.8 runtime; Qwen exact revision cached. A bounded NF4/BF16 Qwen structured-generation smoke test passed. TeX and Poppler tools are available.
- Resolved public Mistral revision `c170c708c41dac9275d15a8fff4eca08d52bab71` before pilot execution. The model is an independent family and does not require gated access.

## 2026-08-21: blinded V14 audit implementation

- Confirmed from source that the historical recovery threshold used the held-out field panel baseline rather than the cluster-excluded nominal-training distribution.
- Confirmed the V14 H3 sign statistic is structurally nonnegative and therefore cannot support directional relaxation. No V14 outcome was changed or reinterpreted as a new experiment.
- Implemented explicit LOCO threshold maps, structured H3 validity fields, the missing 10,000 full-pipeline cluster-preserving permutations, independent 3/5/7-window recomputation, all single-observable deletions, and marginal-preserving dependence bias floors.
- Added a versioned correction archive that preserves the committed V14 primary JSON, hypothesis table, recovery table, README, claims matrix, and result macros before replacement.
- Added focused synthetic and end-to-end tests. Initial focused V14 tests pass; full retained-data recomputation remains pending at this log point.

## 2026-08-21: V15 design before engineering outcomes

- Defined the two-model, six-cluster-per-model matched field-quench and memory-control study before any V15 model call.
- Defined a deterministic own-agent, past-only scrambled-history control with the same prompt section, entry count, and field format as genuine memory. It contains no future event and no peer-private state.
- Allocated H1 alpha 0.02 and an H2-H4 Holm family alpha 0.03 before formal execution. Defined H4 as fixed sweeps 31-35 minus 41-45 so its sign is not tautological.
- Set expected formal size to 48 trajectories and 34,560 decisions. Set a hard 25 generation-GPU-hour and 22-million-prompt-token ceiling. The engineering pilot will determine whether the full design fits; no cluster count may be reduced below six after outcomes.
- Restricted the 128-decision-per-model pilot to engineering observables. No primary scientific contrast is computed in pilot code.

## 2026-08-21: V14 retained-data correction completed

- Recomputed all 24 retained V14 trajectories (17,280 attempted updates) without altering any frozen raw outcome. Replay had zero mismatches and no privacy mutation.
- The corrected leave-one-cluster-out threshold fit excludes each held-out graph/environment cluster. All six field-reversal trajectories re-entered their training-only nominal threshold exactly six sweeps after restoration; mean final-five-sweep distance was 1.560.
- Preserved the historical H3 estimate (134.110324), interval (106.653730 to 184.196216), raw sign-flip value (0.015625), and Holm value (0.046875), but marked the structurally nonnegative directional test invalid and inferential support false.
- Completed the frozen 10,000-replicate cluster-preserving full-pipeline permutation analysis, independently recomputed three-, five-, and seven-sweep nominal geometries, deleted every macrostate coordinate in turn, and added marginal-preserving information-estimator bias floors.
- At the five-sweep window, raw field-minus-nominal total correlation was 4.635 nats, its mean circular-shift null floor was 4.058 nats, and the untruncated adjusted contrast was 0.576 nats. The normalized contrast was -0.043, demonstrating material scale sensitivity rather than a universal dependence magnitude.

## 2026-08-21: engineering model attempts and independent-family fallback

- The Qwen engineering pilot completed 128 decisions with 100% valid structured output after at most one repair, latent-plus occupancy 0.492, 29 transitions in each belief direction, and no privacy, timing, delivery, or memory-control failure. It projected 7.751 generation hours and 9,826,650 prompt tokens for its formal half.
- The preferred Mistral family had two zero-science infrastructure failures (optional downloader configuration, then missing tokenizer dependencies), both retained externally. After installing only pinned `sentencepiece==0.2.0` and `protobuf==5.29.5`, its completed 128-decision engineering pilot remained invalid: 47/128 outputs were valid after one repair, 81 remained invalid, and no plus-to-minus belief transition occurred. No pathwise, quench, or memory contrast was inspected.
- Rejected Mistral before protocol freeze on engineering grounds. Its exact revision remains recorded for provenance and all calls, tokens, failures, and latency remain in external accounting.
- Selected the pre-outcome, genuinely different, openly accessible fallback `ibm-granite/granite-3.3-8b-instruct` at exact revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` (Apache 2.0). Granite must independently pass the same bounded engineering pilot before the V15 protocol can freeze.

## 2026-08-22: Granite engineering qualification and formal freeze

- The first Granite attempt stopped before generation when the persistent Hugging Face cache reached its quota. The zero-generation failure was reconstructed into the external failure ledger (`pilot/granite_failures.json`, SHA-256 `4a83fbdd51415206e9883f4ece0f7976038a5a2faf9bd324082368f325b86e69`). To recover space without destroying evidence, the rejected Mistral cache was moved intact to ephemeral `/tmp/rejected-mistral-cache-v15`; no research artifact was deleted.
- Granite then completed the same 128-decision engineering pilot with 100% first-pass structured validity, latent-plus occupancy 0.65625, eight minus-to-plus and two plus-to-minus belief transitions, and no privacy, timing, delivery, or memory-control failure. Persistent-versus-scrambled mean prompt-token difference was 0.001190 relative; mean prompt and generated tokens were 630.094 and 87.063 per decision. Mean decision latency was 2.345 seconds.
- The blind throughput projection for the two formal halves was 7.751 Qwen generation hours plus 11.255 Granite generation hours, totaling 19.006 hours and 20,714,670 prompt tokens. This fit the prospectively declared ceilings of 25 generation GPU-hours, 22 million prompt tokens, and 34,560 formal decisions; no cluster or arm was removed.
- Froze V15 protocol SHA-256 `863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9`, execution-source SHA-256 `ec9f26223a335558b2789ebd59ee3c3fa0f9e7d1b815fd9b09a1e1960af55e78`, schema SHA-256 `c0382247001c9c586190b81ad4a83535ceb71b03dadfd81e80cac220e9580f0d`, seed-manifest SHA-256 `d9850e5854af307364cd5504d75d0df449412817f3026b1149e8d7de6e8fdaf4`, and memory-control-manifest SHA-256 `24bae40fbc4026b66eef87768dd9ae4edbcc3e7b7a9b1f6a850aede21b32a9f4` before any formal outcome existed.
- Started the complete frozen Qwen formal half in a monitored, resumable RunPod session. Scientific panel contents remain uninspected until the formal execution and accounting checks complete.

## 2026-08-22: derived-model completion and verification during sealed execution

- Completed the exploratory out-of-sample quench comparison using coefficients fitted only from immutable V13 microscopic-response data. No V14 or V15 quench outcome entered coefficient fitting. The direct and kinetic systems share the graph family, initial-condition generator, field schedule, coupling, and quench boundaries. The surrogate reproduces a field-responsive pulse in some clusters but generally overstates its amplitude and shifts peak timing toward the quench or counter-quench boundary; these failures are retained as evidence against a complete low-dimensional closure.
- Completed the prespecified CPU kinetic-surrogate size sensitivity at `N in {8,16,32,64}` with 32 replicates per size and condition (256 trajectories). This is an effective-model comparison, not direct-LLM finite-size scaling.
- Regenerated the corrected V14 paper and 28 vector candidate figures. Automated QA passed for 29 PDFs and 46 pages. Every figure and all 18 manuscript pages were rendered at 300 DPI and manually inspected at original resolution; no material clipping, overlap, missing glyph, unreadable legend, or incorrect panel lettering remained.
- Re-ran 52 focused V14/V15 tests and 125 relevant V10-V13 regressions successfully. A full 626-test repository suite also passed while formal V15 panel contents remained sealed.
- The local RunPod SSH alias remained stale after the Pod restart and the saved RunPod API credential did not authorize endpoint discovery. The existing authenticated terminal remains reliable for execution and monitoring. No credential, Pod identifier, authenticated URL, or environment-variable value was exposed.

## 2026-08-22: redacted operational correction

- The final sentence of the preceding entry described diagnostics up to that point. During a later proxy reconnection, the provider's standard terminal banner emitted a Pod identifier once in internal tool output. No credential, authentication token, complete authenticated URL, SSH private key, or environment-variable value was printed, and the identifier is not reproduced in repository-facing records. Subsequent checks use terminal echo suppression and marker-delimited, allowlisted status fields.

## 2026-08-22: complete formal execution, replay, and outcome release

- Completed all 48 frozen trajectories: 24 Qwen and 24 Granite, six graph/environment clusters per model and four matched arms per cluster. Formal execution produced 34,560 attempted decision rows, 34,565 model calls, 20,908,194 prompt tokens, 2,893,967 generated tokens, and 18.9366 metered generation hours. No privacy mutation occurred. Qwen had zero invalid outputs after repair; Granite retained two invalid decisions after the bounded repair policy without selective rerun.
- Replayed all 48 trajectories and 34,560 rows through content-addressed recorded decisions with zero mismatches. The external formal tree contains 99 files and 57,514,139 bytes; the external raw-formal tree contains 34,560 files and 125,893,817 bytes. Their tree hashes are retained in the repository manifest.
- Released the sealed aggregate outcomes only after both model halves and queued replay/analysis/reporting completed. H1-H4 all meet their prospectively frozen criteria. The pooled memory contrasts are positive but heterogeneous: the Qwen persistent-minus-scrambled decomposition is small and mixed (4/6 positive; exact model-specific sign-flip `p=0.21875`), while Granite is positive in all six clusters.
- The fixed-window recovery contrast is positive in all 12 model/cluster pairs. All Qwen field-Markovized paths re-enter training-nominal thresholds after six sweeps; only one Granite path re-enters within the 15-sweep recovery horizon. H4 is therefore reported as return toward the restored regime, not complete cross-model recovery.
- Generated 12 vector V15 figures and compact source CSVs. Built the synchronized manuscript with eight main figures. Manual visual inspection and final test reruns remain pending at this log point.
- First-pass visual QA identified crossed architecture labels, an empty delayed-audit panel caused by stale metric names, crowded multipanel labels, and unused panel space. Added a presentation-only renderer under the paper directory so these layout corrections do not modify the frozen execution-source tree or any numerical source table.

## 2026-08-22: final scientific and repository verification

- Re-rendered the corrected V15 publication figures and rebuilt the 16-page manuscript. Automated PDF QA passed for all 13 V15 PDFs (12 figures plus the manuscript), covering 28 pages, with embedded fonts and extractable text. Every figure and manuscript page was inspected manually at its original 300-DPI rendering; the final layouts have no material clipping, overlap, missing glyph, unreadable legend, or incorrect panel lettering.
- Re-inspected the corrected V14 caption and rebuilt page at 300 DPI. The V14 automated and manual QA records now cover all 29 PDFs and 46 pages, including the 18-page manuscript.
- The version-focused V10-V15 command passed all 177 collected tests. The complete historical repository suite then passed all 626 tests in 400.44 seconds using importlib test isolation and a temporary, untracked CPU-only PyTorch bootstrap; no tracked dependency or test configuration was changed. Scikit-learn emitted documented robust-covariance numerical warnings, but no test failed or skipped.
- Verified exact replay of all 48 V15 formal trajectories and 34,560 attempted updates with zero mismatches. The frozen protocol, execution-source, schema, seed, and memory-control hashes remain unchanged after post-formal reporting and presentation edits.

## 2026-08-22: completion-audit source-checksum correction

- Found that the frozen V15 source enumerator included an ignored root-level `__init__.pyc` because it excluded `__pycache__` directories but not standalone bytecode files. The cache was 224 bytes with SHA-256 `ab2d053b6168ee0b1e4766e84c276d52217aef11ee21330ed83277d26cf574fc`.
- Independently reconstructed the frozen legacy execution-source SHA-256 `ec9f26223a335558b2789ebd59ee3c3fa0f9e7d1b815fd9b09a1e1960af55e78` from the current semantic source manifest plus that recorded cache path and digest. The cache-free semantic-source checksum is `f8d4fa546ba46a42cd4234dd8af6ad60309c231f2997e10d0d25830f6dddb2f2`.
- Added a compact, clean-checkout provenance verifier under the V15 reproducibility directory. The ignored cache is excluded from repository indexes and remains unstaged. No protocol, formal result, model call, raw trajectory, or scientific disposition changed.

## 2026-08-22: final RunPod reachability recheck

- A read-only SSH recheck after all scientific work returned `connection refused`. The last successful remote audit had shown zero CUDA compute processes, zero V15 workers, and zero tmux sessions. The completed external-tree checksums and replay evidence remain valid, but current live Pod state could not be refreshed and is reported as unreachable rather than inferred.
