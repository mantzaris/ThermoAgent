# JSTAT V16 structure-only implementation report

**Implementation date:** 26 August 2026
**Starting branch:** `jstat-paper-structure-v16`
**Starting local commit:** `534efc0e83770757bcb1cd41183de25ac5f3fe85`
**Starting remote commit:** `origin/jstat-paper-structure-v16` at `534efc0e83770757bcb1cd41183de25ac5f3fe85`
**Authoritative specification:** `notes/jstat_v16_structure_audit.md` at blob `a1d67dd87a5cd538697337e544e235478f315157`
**Scope:** sectioning, atomic paragraph movement, figure placement, cross-reference repair, supplement integration, and structure-caused layout repair only

The branch was clean at the starting commit. The local audit blob matched the
pushed blob. No fast-forward was needed.

## 1. Pre-edit frozen-artifact record

| Artifact or canonical set | Starting SHA-256 |
|---|---|
| `paper/jstat_v15/main.tex` | `ecb7d528bd93d7bbc697fca70ced6c704bf91b66faf2d1734ee652a11e975ebe` |
| `paper/jstat_v15/supplement.tex` | `c9104f750cfbaa5cb3ff6726274308c5d5931c2c014421dcc3fb9981079b2609` |
| `paper/jstat_v15/results_macros.tex` | `cd95f8fe3e01157cc8cee733dc04e5b443fd693fe3aab39beb5ca488ae0d4679` |
| `paper/jstat_v15/references.bib` | `e1180c565538d4c1a07af5e38a15ec6d0d45a0bf7597351e1f4421e51e294538` |
| Canonical ordered digest of the 14 figure-PDF `sha256sum` records | `a8491ade93c526f1c89d2b3bec0eb568d89d09c802bdf8f4f534c67fa390214b` |
| Canonical ordered digest of the 14 figure-source-CSV `sha256sum` records | `15ea2cacf56194bc503e35518b16494f9861f500fbd9c687b77fd04a8561212e` |

The individual frozen figure hashes are recorded below so that the final
comparison does not rely only on aggregate manifests.

| Asset | Figure PDF SHA-256 | Source CSV SHA-256 |
|---|---|---|
| 1 | `9c839c9293a74afaec29e88c33f8c81a62b2efb165f5ec536cba774ef2f930a6` | `f8ba2c96b167e1d199f55566caf55d42ef76c9e1021e27bfc36c3fc4b705ff55` |
| 2 | `d9c64a4b6102abac3cd702f8367a71469bdeb563864ffc3acccf46cc134143e2` | `9746b9cab47c58950fac96da2f5cf1b6af29594d5a6cbd2a8795a972510164e0` |
| 3 | `307cdf8aa2cbd9e87f80494e224a30faad86ca97a188bc70905c5e7fb0fb25c7` | `0f7318f4641a1fa95d287ed3e178b67f11f1303148ca61e8841dc00a55f90d01` |
| 4 | `28b051e46b61bef8027a2b52b2bd7599304b11a7b1089e9c26707e352ae7e11a` | `9cce0bfd41c28b1d3e727a597202a76c0e046cf10cba4349350fa90f571d47b2` |
| 5 | `ab806f1aafa705e856af81417fbf4db925668218084ef81a9f877d00b56f7895` | `9020ecf9216c348cd21af6b4454963ca22f124808b8a979646faaf5172f32e0d` |
| 6 | `b7bd3b5367ebc9ce22271c262f40b86ad255921dd83401c1a9198b68cd1b662b` | `71b5d9b9b0c47a531270d8cf2a2782e61c64e07ecfb51e9856d1eaeee42c8537` |
| 7 | `9e09c0218736c903c8b6afb860db31ef81347590bcc94f72c6b6189bbefb47ed` | `fdd52dc2b59879645887087367577b7fc2292722aec68f9e29cbfd996cd62bdb` |
| 8 | `eeefbf62ff5e51b93d757e0112042246e5e834b8a9121184876d1d07920f2dd5` | `0cdd7b0f792e8df96604cc852b563b9ae5cf3cc906dc06e54783be94633c15a9` |
| 9 | `8975f000f27d181ba9c99141657f6abf16fae862dba9506e059b9abde4b7714d` | `dc10bd56902d03500a03c2d023ddc5b9067fe1c92bbf38b6476b4d1d308b12fd` |
| 10 | `22b9ac164083292048c9ccf2aed5e94fb3864583aab5103de61974b244f0cde8` | `0bd36695aaf56f0cb2eb9cc7b8c53ef8fd3e97c22910030f7ddde2aaa038c0ca` |
| 11 | `2df1742bab61f4076300d1433c3bb5a05a7c8540f099b8b211417632f509d7d4` | `9647715ab15aac6fdfd22c37ed9a0baf55aadb5f90cc8dca00af0225efa5e2c3` |
| 12 | `d34c5017af7eff7ea065f3cfe1730db9ffd6224c5c9b56694b8de5648c665285` | `888e807f356c822d7dcf55b919efb2ea4e90933c1d88cd67c2cccf855460bee9` |
| 13 | `6271b8b469a52caa8d263bc23995dccde5249b51bd0ac864395c9393c3df3f13` | `c3fd12ca890eb1f7f840b9227023f3d608a37a62066927d9acaacade232d7473` |
| 14 | `3e8f383bacc00c77221bf34ae90bd4ee5741cd60c3c1bef89cd5ad7c6bebcc01` | `25873ee443193347f6aba94badaeec038c3c1e9611b296bd141ba5538c7e8d46` |

## 2. Pre-edit movement ledger

Line spans below refer to the approved pre-movement `main.tex` at the starting
commit. A sentence-range notation records the only planned paragraph splits;
each split is at an existing sentence boundary. `E1`--`E15`, `F1`--`F14`, and
`C1`--`C15` refer to the inventories following the table.

| Current block | Source span | Approved destination | Equations | Figures | Results-macro uses | Claim/caveat lock | Planned structural handling and transition need |
|---|---:|---|---|---|---|---|---|
| Front matter and abstract | 1--75 | Front matter and abstract, unchanged | None | None | Abstract: `V15Trajectories`, `V15Decisions`, `V15HOne`, `V15HOneCI`, `V15HTwo`, `V15HTwoCI`, `V15HThree`, `V15HThreeCI`, `V15HFour`, `V15HFourCI` | Global literal/effective boundary | Retain verbatim; preserve digit-catcode loader. No transition. |
| Introduction | 76--131 | Section 1 | None | F1 at 125--130 | None | C1 | Retain paragraphs; move F1 to 2.1; add one roadmap sentence. |
| Stochastic process and information boundaries | 132--190 | Section 2 wrapper | E1--E3 | None after F1 relocation | None | C2 | Retitle and add section label; split only at existing subsection/paragraph boundaries. |
| Local agent state | 134--167 | 2.1 and 2.2 | E1--E2 | F1 inserted in 2.1 | None | C2 | Lines 136--157 to 2.1; lines 159--167 to 2.2. Add first citation for F1 and appendix-authority pointer. |
| Markovized, persistent, and scrambled history | 169--190 | 2.3 | E3 | F12 cited; float in B.3 | None | C2 and control portion of C7 | Retain text atomically; retitle; add section label and appendix figure pointer only. |
| Quench protocol and nominal geometry: protocol paragraph | 353--362 | 2.4 | None | None | None | C6 | Move complete paragraph verbatim. |
| Prospective V15 design: matched-design paragraph | 403--410 | 2.4 and 3.5 | None | None | None | C7 | Sentences 1--3 to 2.4; final sentence (inferential unit) to 3.5. Existing sentence-boundary split. |
| Collective observables | 191--312 | Section 3 wrapper | E4--E12 | None | None | C3--C4 and C13 | Retitle and add section label; move subsections as listed below. |
| Order and effective compatibility | 193--274 | 3.1 and 3.4 | E4--E9 | None | None | C3 | Lines 195--222 (through susceptibility window) to 3.1; lines 222--274 beginning “Three secondary observables” to 3.4. Claim/caveat paragraphs stay with equations. |
| Entropy and dependence | 275--301 | 3.2 | E10--E11 | None | None | C4 | Retain verbatim and merge with rolling macrostate. |
| Rolling macrostate | 302--312 | 3.2 | E12 | None | None | C13 | Retain verbatim immediately after entropy/dependence; coordinate order locked. |
| Pathwise temporal asymmetry | 313--352 | 3.3 and 4.1 | E13--E14 | F2 at 347--352 | None | C5 | Lines 315--346 to 3.3 verbatim; move F2 to 4.1 and add first citation. |
| Quench protocol and nominal geometry: geometry/estimand paragraphs | 363--378 | 3.5 | E15 | None | None | C6 | Move complete paragraphs verbatim after observable definitions. |
| Prospective V15 design: hypothesis/inference paragraph | 420--429 | 3.5 | None | None | None | C7 | Move complete paragraph verbatim; add Appendix A.3 pointer. |
| V14 correction and audit | 431--455 | 4.2, A.1, and A.2 | None | F3 and F4 imported from old Section 5; F5 in A.2 | None | C8 | Lines 433--447 remain 4.2; lines 449--455 move verbatim to A.2. Add only figure/appendix pointers. |
| Quench protocol figures | 380--390 | 4.2 and A.1 | None | F3 at 380--384; F4 at 386--390 | None | C6/C8 | F3 to 4.2; F4 to A.1; add explicit first citations before each float. |
| Prospective V15 design: model/software and pilot paragraphs | 392--401, 412--419 | 4.3 | None | F12 cited in 2.3/6.2; float in B.3 | `V15Trajectories`, `V15Decisions` at 416--417 | C7 | Move the two complete paragraphs verbatim. No scientific merge. |
| Results: field-quench replication | 457--495 | 5.1 and 5.2 | None | F6 at 473--477 | H1 effect/CI/disposition; H4 effect/CI/disposition | C9 | Lines 459--477 to 5.1; lines 479--495 to 5.2. Existing F6 citation retained before float. |
| Results: memory and temporal asymmetry | 496--549 | 6.1--6.3 | None | F7 at 527--531; F8 cited and placed B.2 | H2/H3 effects, CIs, dispositions | C10 | Split first paragraph only at existing sentence boundaries: H2 sentences to 6.1, H3 sentences to 6.2. Keep the negative adjusted-value caveat with both comparisons in 6.2. Heterogeneity and estimator sensitivity to 6.3; first-cite F7 and appendix-cite F8. |
| Spatial correlation and finite-window persistence | 551--609 | 5.3--5.4 and Appendix C | None (references E7--E9) | F13 at 574--583; F14 cited, float in C.3 | Correlation, persistence, and Binder macros | C11 | Lines 553--583 to 5.3; lines 585--609 to 5.4. Replace supplement-specific reference only; retain all caveats and result macros. |
| What the reduced representation contributes | 655--681 | 7.1 and 6.4 | None | F9 at 676--680 | None | C13 | Lines 657--674 to 7.1; F9 to 6.4 with explicit first citation. |
| Out-of-sample kinetic closure | 610--654 | 7.2--7.3 and B.4 | None | F10 at 622--627; F11 cited and placed B.4 | None | C12 | Definition, F10, and comparison to 7.2; failure modes and size-context paragraph to 7.3; add first citation for F10 and appendix pointer for F11. |
| Interpretation and limitations | 682--710 | 8.1, 8.3, and 8.4 | None | None | None | C14 | First paragraph to 8.1; complete six-item limitations paragraph to 8.3; negative-boundaries paragraph to 8.4. No internal paraphrase. |
| Conclusion | 711--733 | 8.2, 8.4, and Section 9 | None | None | None | C14 | Move the existing projected-history sentence to 8.2 and the existing final future-work sentence to 8.4; retain all other conclusion sentences in Section 9. Both splits are at existing sentence boundaries. |
| Reproducibility and data availability | 734--784 | Main unnumbered statement and D.2--D.3 | None | None | GPU-hour/accounting macros | C15 | Keep lines 736--764 and 781--783 in main; move the complete arm-independence paragraph 766--771 to D.2 and the complete digest paragraph 773--779 to D.3 verbatim. No condensed replacement. |
| AI-assisted preparation declaration | 785--790 | Main unnumbered statement | None | None | None | Administrative text lock | Retain wording verbatim. |
| Appendix A: Frozen inferential details | 792--807 | A.2 and A.3 | None | F5 in A.2 | None | C7/C8 | First paragraph 795--801 to A.3; second paragraph 803--806 to A.2. |
| Appendix B: Estimator sensitivity | 808--821 | B.1 and B.2 | None | F8 in B.2 | None | C4/C5/C10 | First paragraph 810--815 to B.1; second paragraph 817--821 to B.2. |
| Appendix C: Authority and privacy checks | 823--831 | D.1 | None | None | None | C2/C15 | Move complete paragraph verbatim; add section pointer from 2.2. |
| Bibliography and end document | 833--836 | After Appendix D | None | None | None | Citation lock | Retain bibliography style/file exactly; add a structural page flush only if required by float verification. |
| Supplement S1 scientific paragraph | `supplement.tex` 29--39 | C.1--C.2 | References E8--E9 | None | None | C11 | Split at existing sentence boundaries. Change only hard-coded equation numbers to label references. |
| Supplement S1 figure | `supplement.tex` 41--48 | C.3 | None | F14 | None | C11 | Preserve caption verbatim; add semantic label and explicit first prose citation. |

## 3. Equation inventory before movement

| ID | Starting printed number | Source span | Existing label | Destination |
|---|---:|---:|---|---|
| E1 | 1 | 142--144 | None | 2.1 |
| E2 | 2 | 148--150 | None | 2.1 |
| E3 | 3 | 176--178 | None | 2.3 |
| E4 | 4 | 196--200 | None | 3.1 |
| E5 | 5 | 203--208 | `eq:href` | 3.1 |
| E6 | 6 | 219--221 | None | 3.1 |
| E7 | 7 | 226--232 | `eq:connected-correlation` | 3.4 |
| E8 | 8 | 241--248 | `eq:autocorrelation` | 3.4 |
| E9 | 9 | 261--265 | `eq:binder` | 3.4 |
| E10 | 10 | 280--282 | None | 3.2 |
| E11 | 11 | 288--291 | None | 3.2 |
| E12 | 12 | 305--309 | None | 3.2 |
| E13 | 13 | 318--323 | `eq:pathkl` | 3.3 |
| E14 | 14 | 337--339 | None | 3.3 |
| E15 | 15 | 373--376 | None | 3.5 |

All equation environments will be moved as complete blocks. The five existing
labels are immutable, and the `Z_t` coordinate order in E12 is immutable.

## 4. Figure inventory before movement

| ID | Frozen asset | Original declaration | Approved destination | Final semantic label |
|---|---|---:|---|---|
| F1 | `figure01_augmented_state_architecture.pdf` | `main.tex` 125--130 | 2.1 | `fig:architecture` |
| F2 | `figure02_memory_discovery_replication.pdf` | `main.tex` 347--352 | 4.1 | `fig:memory` |
| F3 | `figure03_v14_quench_time_series.pdf` | `main.tex` 380--384 | 4.2 | `fig:v14series` |
| F4 | `figure04_v14_cluster_recovery.pdf` | `main.tex` 386--390 | A.1 | `fig:v14recovery` |
| F5 | `figure05_v14_delayed_audit.pdf` | Previously unembedded | A.2 | `fig:v14audit` |
| F6 | `figure06_cross_model_quench.pdf` | `main.tex` 473--477 | 5.1 | `fig:crossquench` |
| F7 | `figure07_v15_memory_controls.pdf` | `main.tex` 527--531 | 6.1--6.2 boundary | `fig:v15memory` |
| F8 | `figure08_path_reversal_sensitivity.pdf` | Previously unembedded | B.2 | `fig:path-sensitivity` |
| F9 | `figure09_confirmatory_effects.pdf` | `main.tex` 676--680 | 6.4 | `fig:effects` |
| F10 | `figure10_direct_surrogate_quench.pdf` | `main.tex` 622--627 | 7.2 | `fig:surrogate` |
| F11 | `figure11_surrogate_size_sensitivity.pdf` | Previously unembedded | B.4 | `fig:surrogate-size` |
| F12 | `figure12_memory_prompt_balance.pdf` | Previously unembedded | B.3 | `fig:prompt-balance` |
| F13 | `figure13_graph_distance_correlations.pdf` | `main.tex` 574--583 | 5.3 | `fig:correlations` |
| F14 | `figure14_persistence_and_binder.pdf` | `supplement.tex` 41--48 | C.3 | `fig:persistence-binder` |

## 5. Claim-and-caveat block register

| ID | Pre-edit source | Atomic scientific content and inseparable boundary |
|---|---|---|
| C1 | 76--124 | Descriptive/explanatory reduced-variable claim; no performance claim and no universality from two models. |
| C2 | 132--190 | Complete/observed state and hidden-history mechanism; not proof of physical dissipation. |
| C3 | 193--274 | Order, effective compatibility, fluctuations, spatial/persistence/shape observables; no literal energy, heat capacity, equilibrium correlation time, critical slowing, Binder crossing, or transition. |
| C4 | 275--301 | Entropy and dependence estimators; short-window floor retained, adjusted estimates untruncated and possibly negative. |
| C5 | 313--346 | Block-reversal KL; not exact entropy production, with projected non-Markov dynamics and distinct null dispersion/Monte Carlo-floor uncertainty. |
| C6 | 353--390 | Finite-system quench and held-out-cluster-excluded geometry; no quantum/thermodynamic-limit analogy and no cross-model absolute-distance pooling. |
| C7 | 392--430 | Frozen V15 design and cluster inference; agents/updates/tokens/windows are not replicates, sign symmetry is not treatment-label randomization, decoding temperature is not physical temperature. |
| C8 | 431--456 | V14 derived-analysis correction with raw trajectories unchanged; historical H3 invalid and delayed audits nonprospective. |
| C9 | 457--495 | H1/H4 response and restoration; within-model scales only, movement toward restoration not complete recovery, five Granite trajectories remain above threshold. |
| C10 | 496--549 | H2/H3 memory contrasts; not positive absolute entropy production, Qwen H3 decomposition not separately confirmatory, magnitude estimator-dependent. |
| C11 | 551--609 and supplement 29--48 | Post-reconstruction spatial/persistence/shape diagnostics; no extra hypothesis family, replication unit, correlation length, equilibrium time, critical slowing, crossing, or transition. |
| C12 | 610--654 | V13-only surrogate and closure failures; no V14/V15 refit, failure is a closure boundary, CPU sizes are not direct-LLM finite-size scaling. |
| C13 | 655--674 | Complementary information beyond order; no omnibus superiority claim. |
| C14 | 682--733 | Effective statistical-mechanical interpretation and limitations; no literal thermodynamic variables, universality, phase transition, exact entropy production, benefit, or performance claim. |
| C15 | 734--783 and old Appendix C | Reproducibility, authority, and provenance; lost original raw tree, unverifiable historical call-file digest, lower-bound accounting, and non-scientific cache correction remain explicit. |

## 6. Paragraphs requiring minimal structural transitions

Before editing, transitions are authorized only at these joins:

1. End of Section 1: roadmap from framing to system/formalism/evidence/results/closure/discussion.
2. Section 2.1: first prose citation for F1.
3. Section 2.2: pointer from authority/information boundaries to Appendix D.1.
4. Section 2.3: pointer from scrambled-history construction to the prompt-balance diagnostic in Appendix B.3.
5. Section 3.5: pointer from exact cluster inference to Appendix A.3.
6. Section 4.1: role sentence and first prose citation for F2.
7. Section 4.2: first prose citations for F3 and appendix F4; pointer to delayed audit in A.2.
8. Section 5.4: replace supplement-specific numbering with Appendix C/F14 label references.
9. Sections 6.1--6.2: grammatical joins created by splitting the H2/H3 paragraph, plus first citation for F7.
10. Section 6.2: pointer to prompt-balance diagnostic F12.
11. Section 6.3: pointer to estimator-sensitivity F8.
12. Section 6.4: synthesis-role sentence and first citation for F9.
13. Section 7.2: first prose citation for F10.
14. Section 7.3: pointer to effective-model size-context F11.
15. Section 8.2: cross-reference joining the moved projected-history summary sentence to Sections 2.3 and 6.
16. Appendices A.1, A.2, B.2, B.3, B.4, and C.3: explicit first citations for F4, F5, F8, F12, F11, and F14.
17. Appendix C.1: hard-coded supplement equation numbers converted to `eq:autocorrelation` and `eq:binder` references.

Every sentence actually added will be quoted verbatim in the final transition
ledger. No transition may carry a new result, citation, estimator,
interpretation, or generalized claim.

## 7. Approved final hierarchy

The implemented hierarchy matches the approved audit. Labels shown here are
new stable structural anchors; the two `\texorpdfstring` line breaks in 3 and
7.3 affect only typesetting of the approved long headings.

1. **Introduction** — `sec:introduction`
2. **State-separated LLM-agent networks and quench protocols** — `sec:system-protocol`
   - 2.1 Augmented, local, and observed state — `sec:state`
   - 2.2 Information boundaries and random-sequential update dynamics — `sec:information-updates`
   - 2.3 Markovized, persistent, and scrambled-history conditions — `sec:history-conditions`
   - 2.4 Field quench, restoration, and matched controls — `sec:quench-protocol`
3. **Statistical-mechanical observables and inferential design** — `sec:observables-inference`
   - 3.1 Order, overlap, effective compatibility, and fluctuations — `sec:order-compatibility`
   - 3.2 Entropy, dependence, and the rolling macrostate — `sec:entropy-macrostate`
   - 3.3 Coarse-grained pathwise temporal asymmetry — `sec:path-asymmetry`
   - 3.4 Spatial correlation and finite-window persistence diagnostics — `sec:spatial-persistence-observables`
   - 3.5 Nominal geometry, recovery estimand, and exact cluster inference — `sec:inferential-design`
4. **Evidence hierarchy, correction, and frozen study roles** — `sec:evidence-hierarchy`
   - 4.1 Memory discovery and prospective replication roles — `sec:memory-study-roles`
   - 4.2 Corrected V14 quench evidence and audit boundary — `sec:v14-correction`
   - 4.3 Frozen V15 cross-model confirmation and control logic — `sec:v15-frozen-design`
5. **Quench response, restoration, and finite-size organization** — `sec:quench-results`
   - 5.1 Cross-model field response — `sec:field-response`
   - 5.2 Restoration and threshold re-entry — `sec:restoration`
   - 5.3 Spatial reorganization under the quench — `sec:spatial-results`
   - 5.4 Finite-window persistence and order-parameter shape — `sec:persistence-results`
6. **Memory and projected temporal asymmetry** — `sec:memory-results`
   - 6.1 Persistent versus Markovized history — `sec:persistent-markovized`
   - 6.2 Genuine versus scrambled-history control — `sec:persistent-scrambled`
   - 6.3 Model heterogeneity and estimator dependence — `sec:memory-heterogeneity`
   - 6.4 Confirmatory synthesis across H1--H4 — `sec:confirmatory-synthesis`
7. **Reduced dynamical descriptions and closure limits** — `sec:closure`
   - 7.1 Information retained beyond order — `sec:representation`
   - 7.2 Out-of-sample kinetic surrogate — `sec:surrogate`
   - 7.3 Captured response, failure modes, and the size-context boundary — `sec:closure-limits`
8. **Discussion** — `sec:discussion`
   - 8.1 Literal stochastic process versus effective statistical mechanics — `sec:literal-effective`
   - 8.2 Projected memory as hidden history — `sec:hidden-history`
   - 8.3 Finite-size, model, topology, and estimator limitations — `sec:limitations`
   - 8.4 Negative results, scope, and future work — `sec:negative-scope`
9. **Conclusions** — `sec:conclusions`

The unnumbered Reproducibility and data availability and AI-assisted
preparation declarations remain between Conclusions and the appendices. The
bibliography remains after the appendices.

- **Appendix A. Study chronology, V14 correction, and frozen inferential details** — `app:study-inference`
  - A.1 Training-only threshold correction and historical H3 invalidation — `app:v14-correction`
  - A.2 Delayed V14 permutation and observable-deletion audit — `app:v14-audit`
  - A.3 Frozen H1--H4 tests, multiplicity, and cluster bootstrap — `app:frozen-inference`
- **Appendix B. Estimator, control, and closure sensitivities** — `app:sensitivities`
  - B.1 Rolling dependence and marginal-shift null construction — `app:dependence-sensitivity`
  - B.2 Block-reversal estimator and floor sensitivity — `app:path-sensitivity`
  - B.3 Scrambled-history prompt-balance control — `app:prompt-balance`
  - B.4 Reduced-model size context — `app:surrogate-size`
- **Appendix C. Finite-window persistence and Binder diagnostics** — `app:persistence-binder`
  - C.1 Autocorrelation construction and truncation — `app:autocorrelation`
  - C.2 Binder trajectory-first and pooled-moment sensitivity — `app:binder-sensitivity`
  - C.3 Persistence, Binder, and occupancy display — `app:persistence-display`
- **Appendix D. Implementation integrity, authority, privacy, and provenance** — `app:provenance`
  - D.1 Agent authority and information-boundary checks — `app:authority`
  - D.2 Arm independence, replay, and artifact integrity — `app:arm-integrity`
  - D.3 Reconstruction and technical provenance details — `app:technical-provenance`

## 8. Transition-sentence ledger

The following sentences are the complete set of newly written structural
transitions and explicit figure/appendix pointers. They should receive focused
review in the later language-editing pass.

1. “The sections below proceed from the system and its observables to the evidence hierarchy, direct results, reduced-description test, and interpretation.”
2. “Figure~`\ref{fig:architecture}` summarizes the information boundaries and the augmented-to-macroscopic map.”
3. “The corresponding authority and private-state checks are recorded in Appendix~`\ref{app:authority}`.”
4. “Prompt-balance diagnostics for the scrambled-history control are retained in Figure~`\ref{fig:prompt-balance}` of Appendix~`\ref{app:prompt-balance}`.”
5. “The exact-test, multiplicity, and cluster-bootstrap details are retained in Appendix~`\ref{app:frozen-inference}`.”
6. “Figure~`\ref{fig:memory}` keeps discovery, prospective replication, and the V15 cross-model extension in their distinct study roles.”
7. “Figure~`\ref{fig:v14series}` displays the corrected V14 quench time series.”
8. “Cluster-level recovery and delayed-audit material is retained in Appendices~`\ref{app:v14-correction}` and~`\ref{app:v14-audit}`, including Figures~`\ref{fig:v14recovery}` and~`\ref{fig:v14audit}`.”
9. “Figure~`\ref{fig:v15memory}` displays the paired cluster contrasts for the two pinned model families.”
10. “The prompt-balance diagnostic in Figure~`\ref{fig:prompt-balance}` records the scope of the format and approximate-length control.”
11. “The block-length, pseudocount, and shuffled-floor sensitivities are displayed in Figure~`\ref{fig:path-sensitivity}` of Appendix~`\ref{app:path-sensitivity}`.”
12. “Figure~`\ref{fig:effects}` places the four frozen effects and their cluster-bootstrap intervals in a single confirmatory synthesis.”
13. “Figure~`\ref{fig:surrogate}` compares the direct V14 trajectories with the out-of-sample kinetic surrogate.”
14. “The effective-model size context is displayed in Figure~`\ref{fig:surrogate-size}` of Appendix~`\ref{app:surrogate-size}`.”
15. “Figure~`\ref{fig:v14recovery}` retains the cluster-level recovery audit under training-only thresholds.”
16. “Figure~`\ref{fig:v14audit}` records the delayed dependence and full-pipeline permutation audit.”
17. “Figure~`\ref{fig:path-sensitivity}` displays the frozen estimator-sensitivity grid.”
18. “Figure~`\ref{fig:prompt-balance}` records the frozen prompt-balance diagnostic.”
19. “Figure~`\ref{fig:surrogate-size}` places the direct-system anchor beside the frozen effective-model size context.”
20. “Figure~`\ref{fig:persistence-binder}` displays the accompanying persistence, Binder, and occupancy diagnostics.”

No new sentence was needed in Discussion 8.2: the existing conclusion sentence
about augmented state, observable projection, rolling macrostate, and hidden
memory was moved there verbatim.

Two existing sentences received cross-reference-only repairs:

- “Figure~`\ref{fig:persistence-binder}` in Appendix~`\ref{app:persistence-binder}` combines the full autocorrelation curve with the fixed-lag integral in equation (`\ref{eq:autocorrelation}`), the Binder statistic in equation (`\ref{eq:binder}`), and empirical magnetization occupancy.” This replaces only “Supplementary figure S1”.
- “The diagnostics below accompany equations~(`\ref{eq:autocorrelation}`) and~(`\ref{eq:binder}`) of the main text.” This replaces only the unsafe hard-coded “equations (8)--(9)”.

Four captions were required for frozen assets that had not been embedded. The
sentences were assembled only from the verified frozen figure-catalog purpose,
estimand, supported-claim, and limitation fields:

- F5: “Delayed prespecified dependence and permutation audits. Raw and bias-adjusted dependence, window sensitivity, and cluster-preserving nulls are shown explicitly. These audits were completed after formal outcomes because of implementation omissions.”
- F8: “Block-length, pseudocount, and shuffled-bias-floor sensitivity for bias-adjusted block reversal divergence. Memory contrasts can be checked against estimator choices; observable coarse-graining and finite length remain limiting.”
- F11: “Direct $N=16$ anchors in a denser inexpensive effective-model size context. These are CPU kinetic-surrogate quench responses, not direct-LLM finite-size scaling.”
- F12: “Scrambled-history prompt-length control. Per-cluster mean token counts verify that prompt length and format are approximately matched; semantic content cannot be exactly token-matched turn by turn.”

## 9. Figure placement and labels

All 14 frozen assets occur exactly once. Every source-level first `\ref` occurs
before its `\resultfigure` declaration. The float macro changed only from
`[t]` to `[htbp]`; limited `\clearpage` barriers prevent carryover from Section
5 into Section 6, from main text into appendices, and from Appendix C into
Appendix D.

| Asset | Final scientific location | Final printed figure/page | Semantic label | Caption provenance |
|---|---|---|---|---|
| 1 | 2.1 | Figure 1, page 3 | `fig:architecture` | Existing caption unchanged |
| 2 | 4.1 | Figure 2, page 8 | `fig:memory` | Existing caption unchanged |
| 3 | 4.2 | Figure 3, page 9 | `fig:v14series` | Existing caption unchanged |
| 6 | 5.1 | Figure 4, page 10 | `fig:crossquench` | Existing caption unchanged |
| 13 | 5.3 | Figure 5, page 12 | `fig:correlations` | Existing caption unchanged |
| 7 | 6.1--6.2 boundary | Figure 6, page 13 | `fig:v15memory` | Existing caption unchanged |
| 9 | 6.4 | Figure 7, page 14 | `fig:effects` | Existing caption unchanged |
| 10 | 7.2 | Figure 8, page 16 | `fig:surrogate` | Existing caption unchanged |
| 4 | A.1 | Figure 9, page 19 | `fig:v14recovery` | Existing caption unchanged |
| 5 | A.2 | Figure 10, page 20 | `fig:v14audit` | Frozen figure catalog |
| 8 | B.2 | Figure 11, page 21 | `fig:path-sensitivity` | Frozen figure catalog |
| 12 | B.3 | Figure 12, page 22 | `fig:prompt-balance` | Frozen figure catalog |
| 11 | B.4 | Figure 13, page 22 | `fig:surrogate-size` | Frozen figure catalog |
| 14 | C.3 | Figure 14, page 24 | `fig:persistence-binder` | Standalone-supplement caption unchanged |

The nine original semantic figure labels are unchanged. The five approved
labels added are `fig:v14audit`, `fig:path-sensitivity`,
`fig:surrogate-size`, `fig:prompt-balance`, and
`fig:persistence-binder`. Printed numbers are automatic and no hard-coded
figure-number references remain.

## 10. Equation and cross-reference verification

All 15 original numbered display environments remain. A whitespace-normalized,
order-independent hash of the complete equation/align blocks is
`5d13a4b0f65cd3c091a5937cd47bdb44ad568862fabefa0b8af898bd51475930`
both before and after movement.

| Equation ID | Old printed number | New printed number | Final location | Label status |
|---|---:|---:|---|---|
| E1 | 1 | 1 | 2.1 | Unlabeled, unchanged |
| E2 | 2 | 2 | 2.1 | Unlabeled, unchanged |
| E3 | 3 | 3 | 2.3 | Unlabeled, unchanged |
| E4 | 4 | 4 | 3.1 | Unlabeled, unchanged |
| E5 | 5 | 5 | 3.1 | `eq:href` preserved |
| E6 | 6 | 6 | 3.1 | Unlabeled, unchanged |
| E10 | 10 | 7 | 3.2 | Unlabeled, unchanged |
| E11 | 11 | 8 | 3.2 | Unlabeled, unchanged |
| E12 | 12 | 9 | 3.2 | Unlabeled, coordinate order unchanged |
| E13 | 13 | 10 | 3.3 | `eq:pathkl` preserved |
| E14 | 14 | 11 | 3.3 | Unlabeled, unchanged |
| E7 | 7 | 12 | 3.4 | `eq:connected-correlation` preserved |
| E8 | 8 | 13 | 3.4 | `eq:autocorrelation` preserved |
| E9 | 9 | 14 | 3.4 | `eq:binder` preserved |
| E15 | 15 | 15 | 3.5 | Unlabeled, unchanged |

The 18 citation commands have the same pre/post multiset hash,
`f349e059165e53353e4329c63fc569bb22f1a70e38e2d8e36dbc724c333ab80f`.
All 24 cited bibliography keys therefore remain reachable. The 40 V15
results-macro invocations have the same pre/post multiset hash,
`3116c87d8d6b42ece61327f1162d7b6bed52133b5e37733c9318658ded29a7d0`.

The final LaTeX log contains no undefined reference, unresolved citation,
duplicate-label, overfull-box, underfull-box, or rerun warning. All new
`sec:` and `app:` labels are enumerated in Section 7 above. The only new
scientific-object labels are the five figure labels enumerated in Section 9;
all five pre-existing equation labels and nine pre-existing figure labels are
unchanged.

## 11. Supplement migration verification

The former supplement's full scientific paragraph matches Appendix C after
normalizing whitespace and making the required replacement of “equations
(8)--(9)” with references to `eq:autocorrelation` and `eq:binder`. The
distinctive supplement opening occurs once in `main.tex`; the text is not
duplicated elsewhere. The Figure 14 caption is whitespace-normalized identical
to the supplement caption.

Appendix C preserves:

- trajectory-first construction before six-cluster uncertainty summaries;
- complete model--graph--environment clusters as the uncertainty unit;
- the fixed two-sweep truncation and one-/three-sweep sensitivities;
- full-phase and early/late estimates;
- cluster-mean and pooled-moment constructions;
- occupancy alongside Binder values;
- undefined zero-variance handling in the main estimator definition;
- the non-equilibrium-correlation-time, no-critical-slowing, no-crossing, and
  no-phase-transition boundaries across Section 3.4 and Appendix C.

The integrated article compiled and was visually inspected before
`supplement.tex` and `supplement.pdf` were removed with `git rm`. The V15 build
script no longer contains the conditional standalone-supplement build. No
active file under `paper/jstat_v15` or `scripts` refers to those removed files
or to “Supplementary figure S1”.

## 12. Frozen-artifact hash comparison

| Frozen artifact or canonical set | Before | After | Result |
|---|---|---|---|
| `results_macros.tex` | `cd95f8fe3e01157cc8cee733dc04e5b443fd693fe3aab39beb5ca488ae0d4679` | `cd95f8fe3e01157cc8cee733dc04e5b443fd693fe3aab39beb5ca488ae0d4679` | Unchanged |
| `references.bib` | `e1180c565538d4c1a07af5e38a15ec6d0d45a0bf7597351e1f4421e51e294538` | `e1180c565538d4c1a07af5e38a15ec6d0d45a0bf7597351e1f4421e51e294538` | Unchanged |
| Ordered digest of 14 figure-PDF hash records | `a8491ade93c526f1c89d2b3bec0eb568d89d09c802bdf8f4f534c67fa390214b` | `a8491ade93c526f1c89d2b3bec0eb568d89d09c802bdf8f4f534c67fa390214b` | All unchanged |
| Ordered digest of 14 source-CSV hash records | `15ea2cacf56194bc503e35518b16494f9861f500fbd9c687b77fd04a8561212e` | `15ea2cacf56194bc503e35518b16494f9861f500fbd9c687b77fd04a8561212e` | All unchanged |

The individual final figure and source hashes also match every row in the
pre-edit table in Section 1. No simulation, statistics, analysis,
configuration, test, source-data, or figure-generation file appears in the
Git diff.

Final generated manuscript hashes before commit:

- `main.tex`: `2c1fba616d349a5ca1438ef75e1a984276a850bf009430ccec209d510245c8f0`
- `main.pdf`: `868fb3335f702240421171d1d97fc6e06e868872d6e5a7b3efe9c42bc067d0b8`

These two generated-file hashes are expected to differ from V15 because the
approved structure, cross-references, added frozen figures, and Appendix C are
now present. They will be recalculated if any final report-driven source check
requires another build.

## 13. Compilation and PDF inspection

Commands used:

```text
cd paper/jstat_v15
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
./scripts/build-statmech-v15-paper.sh
```

`latexmk` ran the required BibTeX and repeated pdfLaTeX passes. The established
build script was then verified after removal of its supplement branch. The
final article is **27 A4 pages**, contains printed Figures 1--14, and contains
all 15 numbered displays.

All 27 rendered pages were rasterized and visually inspected, both in contact
sheets and at larger scale for the title/abstract, long headings, main figures,
appendix transitions, and final persistence/Binder display. Findings:

- title, abstract, keywords, and Introduction render without clipping;
- the nine-section and four-appendix hierarchy is visible and correctly ordered;
- the two approved long headings use line-break-only typesetting repairs;
- equations fit without clipping or wrapping defects;
- every figure and caption is legible and unclipped;
- Section 5's spatial figure is flushed before Section 6;
- the Section 7 surrogate figure is flushed before Discussion content;
- Figure 14 is flushed before Appendix D;
- no unexpected blank page or isolated heading remains;
- Reproducibility, AI declaration, appendices, and references are in the approved order;
- references occupy the lower part of page 25 and pages 26--27; the partially
  filled final page reflects the unchanged 24-entry bibliography and no
  typography compression was applied.

## 14. Scientific-diff verification

The permitted-difference audit passed.

- Every normalized pre-movement scientific sentence from `main.tex` and the
  supplement is present in the new `main.tex`, after only the two approved
  semantic-reference substitutions recorded in Section 8.
- A token-multiset containment check found no missing pre-movement scientific
  token beyond the intentionally removed strings “Supplementary”, “figure”,
  “S1”, literal equation numbers 8/9, and the original `[t]` float specifier.
- No pre-movement numerical token is missing.
- The 15 complete equation blocks have identical normalized pre/post hashes.
- The 40 result-macro calls and 18 citation commands have identical pre/post
  multisets.
- All nine original main-article captions and the former supplement's Figure
  14 caption are unchanged after whitespace normalization.
- The four new captions use only frozen figure-catalog wording and preserve
  their delayed-audit, estimator-sensitivity, effective-model-size, and
  approximate-prompt-balance limitations.
- `results_macros.tex`, `references.bib`, all figure PDFs, and all figure source
  CSVs have identical hashes.
- Section splitting occurred only at existing sentence boundaries recorded in
  the pre-edit ledger.
- Claim/caveat blocks C1--C15 remain present. In particular, no literal-energy,
  exact-entropy-production, full-recovery, universality, phase-transition,
  direct-LLM-size-scaling, or performance claim was introduced.

## 15. Files changed, removed, and deferred matters

Expected final file set:

| Status | File | Reason |
|---|---|---|
| Modified | `paper/jstat_v15/main.tex` | Approved hierarchy, atomic movement, labels/references, 14-figure placement, Appendix C integration, and limited float barriers |
| Modified | `paper/jstat_v15/main.pdf` | Recompiled 27-page Version-of-Record candidate |
| Removed | `paper/jstat_v15/supplement.tex` | Complete scientific content migrated to Appendix C |
| Removed | `paper/jstat_v15/supplement.pdf` | Obsolete standalone rendering |
| Modified | `scripts/build-statmech-v15-paper.sh` | Removed obsolete conditional supplement compilation |
| Added | `notes/jstat_v16_structure_implementation.md` | Movement, transition, verification, and handoff ledger |

Administrative handling:

- The main data-availability paragraph and its reconstruction/accounting
  language remain unchanged.
- The complete arm-independence paragraph moved verbatim to Appendix D.2.
- The complete bytecode-cache/provenance paragraph moved verbatim to Appendix
  D.3.
- No condensed replacement statement was created.
- The AI-assisted preparation declaration remains verbatim.

Deferred to later authorized language or production work:

- review and possible polishing of the 20 transition sentences in Section 8;
- any shortening of the long main data/provenance statement;
- journal-class/template conversion;
- bibliography or literature-review revision;
- typography changes beyond the structural line breaks and float barriers;
- any figure redesign or scientific reanalysis.

The tracked directory `paper/jstat_v15 (copy)` was added in the author's prior
`copy` commit, is not referenced by any active build script, and was left
untouched. Its old standalone supplement is therefore a historical duplicate,
not an active submission/build target.

## 16. Commit and push

The verified structure implementation was committed locally as
`9d4a01a6eac66abd4a71061749af398ef0b8b409` with message
`Reorganize JSTAT manuscript structure`.

Immediately before the push, `git fetch origin` confirmed that the remote tip
remained the starting commit
`534efc0e83770757bcb1cd41183de25ac5f3fe85`; the branch had not advanced or
diverged. The normal HTTPS push could not authenticate in this shell and
stopped with `fatal: could not read Username for 'https://github.com': No such
device or address`. GitHub CLI authentication was absent, the configured Git
credential cache contained no repository credential, and the loaded SSH key
was not accepted by GitHub. No force push, rebase, merge, reset, credential
change, or remote-URL change was attempted.

This status paragraph is a documentation-only follow-up to the structural
commit. Because a Git commit cannot contain its own hash, the exact local tip
containing this finalized report is stated in the author handoff.
