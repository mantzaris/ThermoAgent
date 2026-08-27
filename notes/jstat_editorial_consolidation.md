# JSTAT editorial consolidation report

## 1. Starting state and scope

- Branch: `jstat-paper-structure-v16`
- Starting local commit: `48c0e66efe1cb07a081fcda766440d4997eb3816`
- Starting remote commit after `git fetch origin`: `48c0e66efe1cb07a081fcda766440d4997eb3816`
- Earlier structural commit retained in history: `9d4a01a6eac66abd4a71061749af398ef0b8b409`
- The worktree was clean, the expected starting commit was `HEAD`, and no unrelated commits were present before editing.
- No simulation, RunPod, analysis, configuration, test, or data workflow was run or modified. The document class remains `article`.

This pass consolidates the approved structure, expands the Introduction, removes repository-development language from manuscript prose, turns the Discussion into continuous prose, and simplifies the computational and availability material. Numerical results, inferential dispositions, equations, and frozen figure assets remain unchanged.

## 2. Final hierarchy

1. Introduction
2. State-separated LLM-agent networks and quench protocols
   1. Network state and update dynamics
   2. History conditions and matched controls
   3. Field quench and restoration protocol
3. Statistical-mechanical observables and inferential design
   1. Macroscopic and information-theoretic observables
   2. Temporal asymmetry, spatial correlation, and persistence diagnostics
   3. Recovery estimands and exact inference
4. Study design and evidential roles
   1. Exploratory discovery and corrected earlier evidence
   2. Cross-model confirmation and control logic
5. Quench response, restoration, and finite-size organization
   1. Cross-model response and recovery
   2. Spatial and finite-window organization
6. Memory and projected temporal asymmetry
   1. Persistent, Markovized, and scrambled histories
   2. Model heterogeneity and estimator dependence
7. Reduced dynamical descriptions and closure limits
   1. Information retained beyond mean order and the kinetic surrogate
   2. Captured response, failure modes, and finite-size context
8. Discussion
9. Conclusions
10. Data and code availability (unnumbered)
11. Acknowledgements (unnumbered)
12. Appendix A: Study chronology, correction, and inferential details
13. Appendix B: Estimator, control, and closure sensitivities
14. Appendix C: Finite-window persistence and Binder diagnostics
15. Appendix D: Computational implementation and validation
16. References

Discussion and Conclusions have no nested headings. Appendices A--D are continuous appendices without nested headings.

## 3. Section and subsection size audit

Word counts are TeXcount prose counts; paragraph counts include source paragraphs with at least 20 prose words and exclude equations and figure captions. Counts on a parent section aggregate its subsections. The minimum-size rule formally applies to subsections; every retained subsection independently exceeds 100 words and also has at least two substantive paragraphs.

| Remaining heading | Words | Substantive paragraphs | Rule result | Source subsections merged into it |
|---|---:|---:|---|---|
| 1. Introduction | 1,041 | 9 | No subsection; target met | Expanded existing Introduction |
| 2. State-separated LLM-agent networks and quench protocols | 558 | 7 | Parent section | 2.1--2.4 below |
| 2.1 Network state and update dynamics | 278 | 3 | Pass | Augmented, local, and observed state; Information boundaries and random-sequential update dynamics |
| 2.2 History conditions and matched controls | 157 | 2 | Pass | Markovized, persistent, and scrambled-history conditions |
| 2.3 Field quench and restoration protocol | 123 | 2 | Pass | Field quench, restoration, and matched controls |
| 3. Statistical-mechanical observables and inferential design | 891 | 12 | Parent section | 3.1--3.5 below |
| 3.1 Macroscopic and information-theoretic observables | 240 | 4 | Pass | Order, overlap, effective compatibility, and fluctuations; Entropy, dependence, and the rolling macrostate |
| 3.2 Temporal asymmetry, spatial correlation, and persistence diagnostics | 428 | 5 | Pass | Coarse-grained pathwise temporal asymmetry; Spatial correlation and finite-window persistence diagnostics |
| 3.3 Recovery estimands and exact inference | 223 | 3 | Pass | Nominal geometry, recovery estimand, and exact cluster inference |
| 4. Study design and evidential roles | 391 | 5 | Parent section | 4.1--4.3 below |
| 4.1 Exploratory discovery and corrected earlier evidence | 277 | 3 | Pass | Memory discovery and prospective replication roles; Corrected earlier quench evidence and audit boundary |
| 4.2 Cross-model confirmation and control logic | 114 | 2 | Pass | Frozen cross-model confirmation and control logic |
| 5. Quench response, restoration, and finite-size organization | 617 | 8 | Parent section | 5.1--5.4 below |
| 5.1 Cross-model response and recovery | 261 | 4 | Pass | Cross-model field response; Restoration and threshold re-entry |
| 5.2 Spatial and finite-window organization | 356 | 4 | Pass | Spatial reorganization under the quench; Finite-window persistence and order-parameter shape |
| 6. Memory and projected temporal asymmetry | 551 | 7 | Parent section | 6.1--6.4 below |
| 6.1 Persistent, Markovized, and scrambled histories | 235 | 4 | Pass | Persistent versus Markovized history; Genuine versus scrambled-history control |
| 6.2 Model heterogeneity and estimator dependence | 316 | 3 | Pass | Model heterogeneity and estimator dependence; H1--H4 synthesis paragraph |
| 7. Reduced dynamical descriptions and closure limits | 550 | 6 | Parent section | 7.1--7.3 below |
| 7.1 Information retained beyond mean order and the kinetic surrogate | 438 | 4 | Pass | Information retained beyond order; Out-of-sample kinetic surrogate |
| 7.2 Captured response, failure modes, and finite-size context | 112 | 2 | Pass | Captured response, failure modes, and size-context boundary |
| 8. Discussion | 717 | 8 | No subsections | Four former Discussion subsections |
| 9. Conclusions | 121 | 2 | No subsections | Existing conclusion content, with audience-facing chronology |
| Data and code availability | 67 | 1 | Unnumbered end matter | Condensed former reproducibility statement |
| Acknowledgements | 28 | 1 | Unnumbered end matter | Condensed former AI declaration |
| Appendix A. Study chronology, correction, and inferential details | 367 | 4 | No subsections | All former A.1--A.3 content |
| Appendix B. Estimator, control, and closure sensitivities | 273 | 4 | No subsections | All former B.1--B.4 content |
| Appendix C. Finite-window persistence and Binder diagnostics | 258 | 3 | No subsections | All former C.1--C.3 content |
| Appendix D. Computational implementation and validation | 248 | 3 | No subsections | Scientifically relevant portions of former D.1--D.3 plus concise validation context |

The baseline had 40 subsections and the consolidated manuscript has 14, a net removal of 26 subsection headings.

## 4. Complete subsection merge map

| Baseline subsection | Final destination |
|---|---|
| Augmented, local, and observed state | 2.1 Network state and update dynamics |
| Information boundaries and random-sequential update dynamics | 2.1 Network state and update dynamics |
| Markovized, persistent, and scrambled-history conditions | 2.2 History conditions and matched controls |
| Field quench, restoration, and matched controls | 2.3 Field quench and restoration protocol |
| Order, overlap, effective compatibility, and fluctuations | 3.1 Macroscopic and information-theoretic observables |
| Entropy, dependence, and the rolling macrostate | 3.1 Macroscopic and information-theoretic observables |
| Coarse-grained pathwise temporal asymmetry | 3.2 Temporal asymmetry, spatial correlation, and persistence diagnostics |
| Spatial correlation and finite-window persistence diagnostics | 3.2 Temporal asymmetry, spatial correlation, and persistence diagnostics |
| Nominal geometry, recovery estimand, and exact cluster inference | 3.3 Recovery estimands and exact inference |
| Memory discovery and prospective replication roles | 4.1 Exploratory discovery and corrected earlier evidence |
| Corrected V14 quench evidence and audit boundary | 4.1 Exploratory discovery and corrected earlier evidence |
| Frozen V15 cross-model confirmation and control logic | 4.2 Cross-model confirmation and control logic |
| Cross-model field response | 5.1 Cross-model response and recovery |
| Restoration and threshold re-entry | 5.1 Cross-model response and recovery |
| Spatial reorganization under the quench | 5.2 Spatial and finite-window organization |
| Finite-window persistence and order-parameter shape | 5.2 Spatial and finite-window organization |
| Persistent versus Markovized history | 6.1 Persistent, Markovized, and scrambled histories |
| Genuine versus scrambled-history control | 6.1 Persistent, Markovized, and scrambled histories |
| Model heterogeneity and estimator dependence | 6.2 Model heterogeneity and estimator dependence |
| Confirmatory synthesis across H1--H4 | Concluding paragraph of 6.2 |
| Information retained beyond order | 7.1 Information retained beyond mean order and the kinetic surrogate |
| Out-of-sample kinetic surrogate | 7.1 Information retained beyond mean order and the kinetic surrogate |
| Captured response, failure modes, and the size-context boundary | 7.2 Captured response, failure modes, and finite-size context |
| Literal stochastic process versus effective statistical mechanics | Continuous Section 8, principally paragraphs 1 and 6 |
| Projected memory as hidden history | Continuous Section 8, paragraph 3 |
| Finite-size, model, topology, and estimator limitations | Continuous Section 8, paragraph 7 |
| Negative results, scope, and future work | Continuous Section 8, paragraphs 4 and 8 |
| Training-only threshold correction and historical H3 invalidation | Continuous Appendix A |
| Delayed V14 permutation and observable-deletion audit | Continuous Appendix A |
| Frozen H1--H4 tests, multiplicity, and cluster bootstrap | Continuous Appendix A |
| Rolling dependence and marginal-shift null construction | Continuous Appendix B, paragraph 1 |
| Block-reversal estimator and floor sensitivity | Continuous Appendix B, paragraph 2 |
| Scrambled-history prompt-balance control | Continuous Appendix B, paragraph 3 |
| Reduced-model size context | Continuous Appendix B, paragraph 4 |
| Autocorrelation construction and truncation | Continuous Appendix C, paragraph 1 |
| Binder trajectory-first and pooled-moment sensitivity | Continuous Appendix C, paragraph 2 |
| Persistence, Binder, and occupancy display | Continuous Appendix C, paragraph 3 and figure 14 |
| Agent authority and information-boundary checks | Continuous Appendix D, paragraph 1; terminology softened to information available to each agent |
| Arm independence, replay, and artifact integrity | Continuous Appendix D, paragraphs 2--3 |
| Reconstruction and technical provenance details | Scientifically relevant validation retained in Appendix D; bytecode/cache discussion removed |

## 5. Audience-facing chronology and version-label map

| Internal label | Audience-facing replacement |
|---|---|
| V12 | Exploratory memory discovery / exploratory evidence |
| V13 | Prospective replication, or the earlier microscopic-response data used to fit the surrogate |
| V14 | Earlier quench experiment, corrected earlier analysis, or delayed audit, according to role |
| V15 | Prospective cross-model experiment, cross-model confirmation study, reported cross-model experiment, or prespecified effects |
| V16 | No manuscript equivalent; retained only in branch, path, and internal-note names |
| “Frozen V15” | Prespecified cross-model design or fixed prospective design |
| “V14 H3” | Invalidated historical recovery sign statistic / earlier hypothesis |

No internal version label remains in audience-facing LaTeX prose, headings, captions, availability text, or acknowledgement text. Historical macro identifiers, semantic labels, and asset filenames remain in the source because they are nonprinting implementation identifiers and the results-macro mechanism is frozen.

There is one unavoidable rendered-artwork exception: the unchanged frozen artwork for figure 2 contains `V12 discovery`, `V13 replication`, `V15 granite`, and `V15 qwen`, and figure 9 contains the trace identifiers `V14Q_g0` through `V14Q_g5`. PDF text extraction detects only those embedded artwork strings. Removing them would require altering frozen figure assets, which this pass expressly forbids. A later figure-edit authorization is required if the no-version-label rule is to include text embedded inside those two frozen plots.

## 6. Discussion consolidation

Section 8 now contains eight continuous paragraphs and no `\subsection`, `\subsubsection`, `\paragraph`, bold pseudo-heading, or unnumbered internal heading. Its progression is:

1. implemented stochastic system and state separation;
2. finite-horizon quench response and recovery;
3. hidden history under coarse-grained projection;
4. model heterogeneity and negative findings;
5. spatial, persistence, Binder, and reduced-closure evidence;
6. effective versus literal statistical-mechanical interpretation;
7. principal finite-size, topology, model, and estimator limitations;
8. larger systems, additional topologies and models, longer stationary trajectories, and richer observable states.

The exact-entropy-production, equilibrium-relaxation, criticality, universality, topology, scrambled-control, finite-window, and reduced-surrogate caveats remain explicit.

## 7. Introduction expansion

The Introduction is 1,041 words by TeXcount, comprises nine substantive paragraphs, has no subsections, and cites 25 distinct sources. Its paragraph sequence is:

1. interacting LLM agents as recurrent stochastic processes;
2. autonomous/generative architectures, memory, and interaction;
3. computational social simulation and its validity boundaries;
4. cooperation, conventions, disagreement, topology, and recent collective-behavior studies;
5. agent-based models, social dynamics, kinetic interaction models, and nonequilibrium networks;
6. trajectory asymmetry, hidden state, coarse-graining, and nonliteral thermodynamic language;
7. the qualified literature gap and controlled-system requirement;
8. why the combined observable, intervention, control, inference, and closure design is needed;
9. contribution, conservative scope, and final manuscript roadmap.

The gap statement is deliberately qualified because adjacent work already examines conventions, cooperation, opinion dynamics, network topology, finite-size ordering, and reduced regimes. The narrower distinction claimed here is the joint use of explicit state separation, field reversal/restoration, persistent and scrambled history interventions, matched cluster inference, and an out-of-sample closure test in one finite stochastic network study.

### New references added and verified

All new references are cited. Publisher or proceedings records and abstracts were inspected; published records were preferred to preprints.

| Key | Verified publication | DOI or verification URL |
|---|---|---|
| `wang2024agents` | Wang et al., *A Survey on Large Language Model Based Autonomous Agents*, Frontiers of Computer Science 18, 186345 (2024) | https://doi.org/10.1007/s11704-024-40231-1 |
| `guo2024multiagents` | Guo et al., *Large Language Model Based Multi-agents: A Survey of Progress and Challenges*, IJCAI 2024, 8048--8057 | https://doi.org/10.24963/ijcai.2024/890 |
| `li2023camel` | Li et al., *CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society*, NeurIPS 36, 51991--52008 (2023) | https://doi.org/10.52202/075280-2264 |
| `park2023generative` | Park et al., *Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023, 1--22 | https://doi.org/10.1145/3586183.3606763 |
| `argyle2023` | Argyle et al., *Out of One, Many: Using Language Models to Simulate Human Samples*, Political Analysis 31, 337--351 (2023) | https://doi.org/10.1017/pan.2023.2 |
| `aher2023` | Aher et al., *Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies*, PMLR 202, 337--371 (2023) | https://proceedings.mlr.press/v202/aher23a.html |
| `gao2024abm` | Gao et al., *Large Language Models Empowered Agent-Based Modeling and Simulation: A Survey and Perspectives*, Humanities and Social Sciences Communications 11, 1259 (2024) | https://doi.org/10.1057/s41599-024-03611-3 |
| `chuang2024` | Chuang et al., *Simulating Opinion Dynamics with Networks of LLM-based Agents*, Findings of NAACL 2024, 3326--3346 | https://doi.org/10.18653/v1/2024.findings-naacl.211 |
| `flintashery2025` | Ashery, Aiello, and Baronchelli, *Emergent Social Conventions and Collective Bias in LLM Populations*, Science Advances 11, eadu9368 (2025) | https://doi.org/10.1126/sciadv.adu9368 |
| `akata2025` | Akata et al., *Playing Repeated Games with Large Language Models*, Nature Human Behaviour 9, 1380--1390 (2025) | https://doi.org/10.1038/s41562-025-02172-y |
| `bonabeau2002` | Bonabeau, *Agent-Based Modeling: Methods and Techniques for Simulating Human Systems*, PNAS 99(suppl. 3), 7280--7287 (2002) | https://doi.org/10.1073/pnas.082080899 |
| `castellano2009` | Castellano, Fortunato, and Loreto, *Statistical Physics of Social Dynamics*, Reviews of Modern Physics 81, 591--646 (2009) | https://doi.org/10.1103/RevModPhys.81.591 |
| `baronchelli2018` | Baronchelli, *The Emergence of Consensus: A Primer*, Royal Society Open Science 5, 172189 (2018) | https://doi.org/10.1098/rsos.172189 |

No bibliography entry was deleted or bibliographic metadata corrected in this pass. Four preserved legacy entries (`schnakenberg1976`, `seifert2005`, `kutvonen2015`, and `zhang2023`) are no longer cited after the focused Introduction rewrite and therefore do not appear in the rendered `unsrtnat` bibliography. The database contains 37 unique keys, 35 unique DOI fields, and 37 unique normalized titles; no key, DOI, or normalized-title duplicate was found. All 33 currently cited keys resolve to bibliography entries.

## 8. Data and code availability

Final source wording (67 words; the macro renders as `19.19`):

> Code, configurations, figure source data, aggregate results, and reproduction scripts are available at https://github.com/mantzaris/ThermoAgent. The reported cross-model experiment ran on an NVIDIA GeForce RTX 4090 and used approximately 19.19 measured generation GPU-hours. Raw prompts, completions, and full trajectories are not distributed because of size and potential sensitivity, and the original external raw tree is no longer available; content-addressed manifests and replay and aggregate-validation artifacts document the reconstruction.

The repository records do not support approximately 30 GPU-hours as the original reported experiment total. They distinguish:

- 18.9366 metered generation GPU-hours for the 48 completed formal trajectories in the original run (`notes/v15_research_log.md`);
- 19.192943537685398 total metered generation GPU-hours for the original campaign (`historical_reference_total_metered_generation_gpu_hours`), rendered as 19.19 by the frozen macro;
- 48.737223 hours for the later 48-trajectory independent reconstruction;
- at least 49.414869 hours for the reconstruction plus retained pilots and interrupted-call accounting;
- 29.194502 hours for the Granite portion of the reconstruction, which is the only repository quantity close to the recalled 30 hours and is not the full reported experiment.

The concise statement therefore uses the documented original-campaign total and does not substitute the reconstruction total.

## 9. Acknowledgements and AI disclosure

The standalone AI declaration was removed. The disclosure is now a single sentence in an unnumbered `Acknowledgements` section, consistent with IOP's current generative-AI disclosure guidance (https://publishingsupport.iopscience.iop.org/questions/generative-ai-tools/):

> OpenAI Codex (GPT-5) was used for manuscript organization and language editing, code assistance, and literature discovery; the author reviewed all resulting material and takes responsibility for the manuscript.

No model is listed as an author and no undocumented model version was introduced.

## 10. Computational appendix disposition

Appendix D was retained because the information boundary, experimental-arm isolation, pinned model/configuration consistency, replay result, reconstruction tolerance, and link between repository artifacts and reported results are scientifically relevant. It was retitled `Computational implementation and validation`, reduced to three explanatory paragraphs, and stripped of nested headings and checklist tone.

The material actually removed from the former Appendix D is the ignored 224-byte bytecode-cache account, the cache-free versus legacy source-digest reconstruction, its rigid nested headings, and checklist-style terminology. The former long reproducibility statement additionally lost template/DPI discussion, Pod-replacement chronology, and detailed interrupted-call accounting. Other engineering details named in the author instruction—cache hygiene, Git housekeeping, test-suite counts, secret scans, and oversized-file scans—were not present in the editorial baseline and were not introduced. Rigid `authority`, `privacy`, and `provenance` framing was removed from visible prose. The remaining limitation—that the original external raw tree is unavailable and historical raw-file digest identity cannot be checked—is retained because it affects reproducibility interpretation.

## 11. Persistence and Binder reference repair

The main Results section explicitly directs the reader to Appendix C and figure `fig:persistence-binder`. Appendix C now introduces the float before it appears with the semantic sentence:

> The autocorrelation and Binder diagnostics defined in equations `\eqref{eq:autocorrelation}` and `\eqref{eq:binder}` are summarized in figure `\ref{fig:persistence-binder}`.

There is no hard-coded equation or figure number in the source and no “diagnostics below” pointer. Appendix C remains a single coherent appendix and preserves trajectory-first calculation, complete-cluster uncertainty, truncation, full/early/late windows, pooled-moment sensitivity, undefined zero-variance cases, occupancy, and the explicit nonclaims about equilibrium correlation time, critical slowing, Binder crossing, and phase transition.

## 12. Equations, figures, references, and frozen artifacts

- Displayed equations: 15 before and 15 after. Whitespace-normalized equation-block SHA-256 is identical before and after: `a94e50c13c3d5edeb33f376090133c7bbcd5f8c864c69a44710e6deabcef708f`.
- All semantic equation labels are unique and retained, including `eq:href`, `eq:connected-correlation`, `eq:autocorrelation`, `eq:binder`, and `eq:pathkl`.
- Figures: 14 `\resultfigure` declarations, 14 distinct frozen assets, and 14 unique semantic labels. Every figure has a prose `\ref` before its float. All assets appear exactly once.
- Labels: 33 total and no duplicates.
- Citations: 33 cited keys, no missing bibliography key, and no unresolved citation after the converged build.
- Result-macro file before/after: `cd95f8fe3e01157cc8cee733dc04e5b443fd693fe3aab39beb5ca488ae0d4679` (unchanged).
- Frozen figure-PDF manifest before/after: `a8491ade93c526f1c89d2b3bec0eb568d89d09c802bdf8f4f534c67fa390214b` (unchanged).
- Frozen figure source-CSV manifest before/after: `15ea2cacf56194bc503e35518b16494f9861f500fbd9c687b77fd04a8561212e` (unchanged).
- Bibliography before: `e1180c565538d4c1a07af5e38a15ec6d0d45a0bf7597351e1f4421e51e294538`; after verified additions: `f2e513e606520e97c20f246faa65dca0951b8acb808dd982b1a192bd292286c9`.
- Scientific result-macro uses are unchanged. Two trajectory/decision macros are additionally reused in Appendix D; four administrative accounting macros were removed with the long reproducibility statement. No macro definition or expansion changed.
- Git status contains no simulation, analysis, configuration, test, figure, source-data, or other unrelated file.

## 13. Compilation and inspection

Build command:

```text
./scripts/build-statmech-v15-paper.sh
```

The established `latexmk` process ran PDFLaTeX, BibTeX, and the necessary repeat pass to convergence and exited with status 0. A retained-log verification run found no undefined references, undefined citations, multiply defined labels, rerun requests, overfull boxes, or underfull boxes in the final pass. `git diff --check` passes.

The final PDF has 28 pages. All 28 pages were rendered at 110 DPI for whole-document inspection, with title/abstract, Introduction, Discussion, end matter, appendix transitions, figures, equations, captions, bibliography, and final page also inspected at larger scale where needed. Findings:

- Introduction flows across pages 1--3 without an isolated heading.
- The 14 figures and captions are legible and unclipped; no equation wraps outside the text area.
- Discussion is continuous across pages 18--19 with no pseudo-headings.
- Data availability and Acknowledgements flow into Appendix A without an unexpected blank page.
- Appendix C begins with a substantive paragraph before its figure and retains all required caveats.
- Appendix D and References transition cleanly.
- A bibliography-only `\small` scope was added after the initial inspection found a nearly empty 29th page containing one reference. The final 28-page bibliography is balanced across pages 26--28.
- There are no unexpected blank pages or materially isolated headings.

Final extracted PDF text contains the required `Data and code availability` and `Acknowledgements` headings and no standalone `AI-assisted preparation declaration`, no unresolved `??`, and no supplement-specific numbering. The frozen-artwork version-label exception is documented in Section 5 above.

## 14. Transition and synthesis sentences for later language review

The following targeted sentences deserve attention in any later prose-only pass; they currently perform structural functions without adding scientific claims:

- Roadmap: “Section 2 defines the network and interventions, Section 3 introduces the observables and inference, and Section 4 distinguishes exploratory, corrected earlier, and prospective cross-model evidence. Sections 5--6 present the quench and memory results, Section 7 tests reduced closure, and Sections 8--9 discuss the scope and conclusions.” (The source uses semantic references throughout.)
- Evidence roles: “The memory evidence is separated into exploratory discovery, a subsequent prospective replication, and the present cross-model extension with an additional history control.”
- Prompt-balance pointer: “The scrambled-history condition preserves prompt presence and approximate length while destroying temporal alignment; the corresponding prompt-balance diagnostic is retained in Appendix B.” (The source adds semantic appendix and figure references.)
- H1--H4 synthesis: “Taken together, H1--H4 address distinct claims rather than a single composite effect: H1 tests field-driven departure, H2 tests persistent history relative to current-state prompting, H3 tests temporally related content relative to a format-matched control, and H4 tests fixed-window restoration.”
- Reduced-description link: “The field-response comparison next asks whether information beyond mean order can be compressed into a low-dimensional local surrogate without fitting the quench trajectories themselves.”

## 15. Files changed and deferred items

Changed:

- `paper/jstat_v15/main.tex`
- `paper/jstat_v15/main.pdf`
- `paper/jstat_v15/references.bib`
- `notes/jstat_editorial_consolidation.md`

No files are removed in this pass. No administrative or language work remains necessary for this pass, apart from possible later stylistic refinement of the listed transition sentences. The only unresolved requirement is the immutable figure-artwork version text described in Section 5; changing it requires a future, explicit exception to the frozen-figure restriction.

## 16. Commit and push status

- Manuscript implementation commit: `e47c1b4c44670188779d93422185d0a6302fc7b7` (`Consolidate JSTAT narrative and literature framing`).
- Report/status commit: the subsequent commit containing this report; it does not alter the manuscript.
- Push: a normal, non-force push was attempted after the manuscript commit and failed because GitHub HTTPS credentials are unavailable in this environment: `fatal: could not read Username for 'https://github.com': No such device or address`. The remote therefore remains at `48c0e66efe1cb07a081fcda766440d4997eb3816`. After committing this report, the same normal push is attempted once more. If authentication remains unavailable, the author can publish both local commits with `git push origin jstat-paper-structure-v16`.
