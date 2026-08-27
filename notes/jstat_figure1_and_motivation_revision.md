# JSTAT Figure 1 and motivation revision

## 1. Starting branch and commit

- Branch: `jstat-paper-structure-v16`
- Starting local and remote commit: `3087aeb4a5103a50419bf19cc49617874bb13b39`
- The expected commit was present at `HEAD`, and the local and remote branches
  were aligned after `git fetch origin`.
- The tracked worktree was clean. The only pre-existing untracked material was
  the author-created backup directory `paper/jstat_v15 (another copy)/`; the
  author explicitly instructed that it be preserved unchanged and included in
  this commit.
- No RunPod connection, LLM simulation, data generation, or scientific
  analysis was performed.

## 2. Figure 1 defects observed before editing

The baseline Figure 1 was rendered independently at 300 dpi and its manuscript
page was rendered at 200 dpi. Both views showed the same defects:

- `Observable projection Y_t` extended through the right border of its box.
- `Rolling macrostate Z_t` extended outside its box, and the horizontal arrow
  crossed the label.
- The left-hand labels, especially `Private observation`, had inadequate
  horizontal padding.
- Fixed-coordinate arrowheads were partly hidden beneath target boxes because
  the arrows were drawn at a lower z-order than the boxes.
- Several arrow shafts entered their targets farther than necessary.
- The two evaluator-bound diagonals had crowded and fragile endpoints.
- The general architecture was incorrectly labeled `Local Qwen transition`,
  although it applies to both pinned model families.

The title, subtitle, semantic color groups, and scientific information-flow
relationships were otherwise useful and were retained.

## 3. Exact Figure 1 generator changes

The canonical `_architecture` implementation in
`thermoagent/statmech_llm_v15/figures.py` now:

- uses component-specific positions, widths, and heights recorded in the
  Figure 1 source CSV;
- wraps all labels deliberately and uses 9-point source text rather than
  shrinking the typography;
- widens the two evaluator boxes independently;
- changes `Local Qwen transition` to `Local LLM transition`;
- retains references to every `FancyBboxPatch`, label object, and box center;
- draws arrows with `FancyArrowPatch` against those patch objects;
- validates rendered label containment after the canvas is drawn; and
- preserves exactly the original ten directed information relationships.

A `generate_figure1()` entry point was added. It regenerates only the
architecture PDF and CSV, replaces only the Figure 1 row of the existing
catalog, and refreshes the catalog-dependent generation metadata. The paper
renderer exposes this path as `--figure 1`; its default remains the established
all-figure generation path.

Targeted command used:

```text
PYTHONPATH=/home/resort/Documents/repos/ThermoAgent python3 paper/jstat_v15/render_publication_figures.py --figure 1
```

## 4. Arrow-routing approach

Each source and target is identified by semantic component name. The arrow
uses the stored source and target centers as its nominal path and the actual
source and target boxes as `patchA` and `patchB`. Patch clipping plus
`shrinkA=2.0` and `shrinkB=1.5` stops the shaft at the visible box boundaries.
The complete arrow is drawn above the boxes and below the labels. A modest
opposite curvature for the two evaluator-bound diagonals keeps their paths and
arrowheads visually distinct. Horizontal and local-to-output links remain
straight.

## 5. Text-containment verification

After the Matplotlib canvas is drawn, `_validate_text_containment()` obtains
the renderer bounding box for each text object and its associated patch. The
generator raises an assertion unless every text extent lies within the box
extent with at least 2.5 points of rectangular padding. The targeted generation
completed with no assertion for all ten labels:

1. Private observation
2. Bounded memory
3. Delivered inbox
4. Local LLM transition
5. Belief/action packet
6. Typed local action
7. Delivery graph
8. Environment
9. Observable projection $Y_t$
10. Rolling macrostate $Z_t$

Visual inspection independently confirmed that no glyph or mathematical
subscript touches a border or the axes boundary.

## 6. Standalone and in-manuscript visual inspection

The regenerated standalone PDF was rasterized and inspected at 200 and 300
dpi. The manuscript was rebuilt, all 28 pages were rasterized and inspected,
and the Figure 1 page was inspected separately at 220 dpi. At the final
`0.96\textwidth` inclusion size on page 5:

- all labels are contained and readable;
- every arrowhead is fully visible;
- no arrow crosses a label;
- the two diagonal arrows remain distinct;
- the projection-to-macrostate link is unobstructed;
- line weights, mathematical subscripts, box spacing, and title/subtitle
  separation remain legible; and
- the information flow can be followed without relying on the caption.

No clipping, unexpected blank page, isolated heading, equation-wrap defect, or
caption collision was observed elsewhere in the manuscript.

## 7. Figure 1 files and metadata changed

- `thermoagent/statmech_llm_v15/figures.py`
- `paper/jstat_v15/render_publication_figures.py`
- `results/collective_agent_statmech_v15/figures/pdf/figure01_augmented_state_architecture.pdf`
- `results/collective_agent_statmech_v15/figures/source_data/figure01_augmented_state_architecture.csv`
- `results/collective_agent_statmech_v15/figures/figure_catalog.csv` (Figure 1
  hash fields only)
- `results/collective_agent_statmech_v15/reproducibility/figure_generation.json`
  (generation timestamp and catalog hash)
- `paper/jstat_v15/main.pdf`

Figure 1 changed from PDF hash
`9c839c9293a74afaec29e88c33f8c81a62b2efb165f5ec536cba774ef2f930a6`
to
`9cabba5504c80d0e37f975f8c32a7d918fc0a777dd7d4b522f330b90e7999171`.
Its layout CSV changed from
`f8ba2c96b167e1d199f55566caf55d42ef76c9e1021e27bfc36c3fc4b705ff55`
to
`c16d75a74125cd3ba74942f64dc543a132d2a355e60870799fd34e1632220e87`.

## 8. Hash confirmation for Figures 2--14

| Frozen set | Before | After | Result |
|---|---|---|---|
| Ordered hash manifest for Figure 2--14 PDFs | `ce6bc8beab17171e5f972eb7da11724e344c96c030b928304bf8be706791de8a` | `ce6bc8beab17171e5f972eb7da11724e344c96c030b928304bf8be706791de8a` | All unchanged |
| Ordered hash manifest for Figure 2--14 source CSVs | `f940589c861416bd6d3be8a6404572e1d67bbbdb58e2ad08b15bfdc9353eda16` | `f940589c861416bd6d3be8a6404572e1d67bbbdb58e2ad08b15bfdc9353eda16` | All unchanged |
| `results_macros.tex` | `cd95f8fe3e01157cc8cee733dc04e5b443fd693fe3aab39beb5ca488ae0d4679` | `cd95f8fe3e01157cc8cee733dc04e5b443fd693fe3aab39beb5ca488ae0d4679` | Unchanged |

The Git diff contains no simulation, analysis, configuration, test, or
quantitative source-data file.

## 9. Introduction word counts

- Previous Introduction: 1,041 TeXcount prose words.
- Revised Introduction: 1,211 TeXcount prose words.
- Net change: 170 prose words, achieved by restructuring and conceptual
  synthesis rather than by adding a results preview.

## 10. Introduction paragraph counts

- Previous Introduction: 9 substantive paragraphs.
- Revised Introduction: 11 substantive paragraphs.
- The Introduction has no subsection, subsubsection, paragraph heading, or
  pseudo-heading.

## 11. Paragraph-level motivation audit

| Paragraph | Main function | Scientific question motivated | Literature synthesized | Transition to next paragraph | Advances motivation? |
|---:|---|---|---|---|---|
| 1 | Establish interacting LLM populations as the scientific object | Why can group behavior not be inferred from one agent or benchmark? | Autonomous- and multi-agent surveys | Moves from collective importance to the recurrent component mechanism | Yes; states the problem and its methodological importance |
| 2 | Explain recurrent feedback and stochastic history dependence | Which properties belong to the interacting process rather than an isolated response? | Architectural concepts established by the surveys | Leads to concrete agent architectures and memory | Yes; identifies what component evaluation misses |
| 3 | Synthesize architectures, communication, and memory | What do interactive architectures establish, and what do their usual evaluations leave unresolved? | CAMEL and generative agents | Broadens from architecture to population experiments | Yes; links memory and communication to the measurement gap |
| 4 | Synthesize computational social simulation and collective outcomes | What group-level behavior is already reproducible, and why is semantic plausibility insufficient? | Survey simulation, behavioral replication, opinion dynamics, conventions, repeated games | Introduces the graph as an independent source of organization | Yes; separates observed group outcomes from dynamical explanation |
| 5 | Establish network-mediated organization | Why can means not reveal topology, influence routes, communities, or graph-distance structure? | Albert--Barabási, Boccaletti et al., Dorogovtsev et al. | Moves from network structure to the broader micro-to-macro framework | Yes; supplies the network-specific rationale for spatial observables |
| 6 | Explain statistical mechanics for finite interacting systems | Which collective coordinates and fluctuations remain measurable without literal physical equivalence? | Agent-based modeling, social statistical physics, stochastic local dynamics, strategic interaction, nonreciprocity | Leads from measurement to perturbation and response | Yes; gives the principal micro-to-macro rationale and finite-size boundary |
| 7 | Motivate field reversal and restoration | What stability, sensitivity, relaxation, and recovery properties do nominal averages hide? | Nonequilibrium logic grounded in the preceding stochastic-dynamics literature | Turns from external perturbation to internal retained history | Yes; explains why the quench is scientifically informative |
| 8 | Motivate history interventions and projected asymmetry | Is an observed trajectory non-Markov because measured state omits genuine history rather than merely more prompt material? | Stochastic thermodynamics, path reversal, hidden entropy production, coarse-graining | Leads to the precise unresolved joint problem | Yes; motivates all three history conditions and preserves the nonliteral interpretation |
| 9 | State the qualified literature gap | Which combination of state control, intervention, diagnostics, inference, and closure remains uncommon? | Recent group-size, topology, coupling, finite-size, and reduced-regime LLM studies | Leads directly to the integrated study design | Yes; distinguishes this study without claiming no adjacent work exists |
| 10 | Explain why the measurement suite is combined | What ambiguity does each observable, control, inferential unit, and closure test resolve? | Synthesizes the preceding statistical-mechanical and agent literature | Leads to contribution, scope, and roadmap | Yes; turns a list of methods into a design argument |
| 11 | State contribution, conservative scope, and roadmap | What is this paper claiming, and what is it explicitly not claiming? | No new literature; synthesizes the established gap | Hands off to the system and protocol section | Yes; fixes the contribution and interpretation boundaries |

No paragraph merely catalogs studies: each identifies an issue, synthesizes
relevant evidence, states the unresolved point, and advances the argument.

## 12. Statistical-mechanics motivation added

The revision adds connected explanations of six roles:

- **Micro to macro:** complicated local language-conditioned transitions can
  still be studied through population order, dependence, correlations,
  response, distributions, and variation across realizations.
- **Network organization:** topology, local information, communities,
  graph-distance structure, and influence paths can differ even when the
  population mean is the same.
- **Finite size and fluctuations:** fluctuations and cluster-level variation
  are informative in finite populations and finite windows without implying a
  thermodynamic limit or phase transition.
- **Perturbation and response:** reversal and restoration distinguish nominal
  similarity from dynamical robustness, delayed response, and incomplete
  recovery.
- **Memory and coarse-graining:** persistent, Markovized, and scrambled
  histories separate retained information, prompt quantity, and genuine
  temporal relation while motivating projected temporal diagnostics.
- **Reduced description:** the kinetic surrogate tests what a small
  macroscopic state can reproduce out of sample and where closure fails.

## 13. New references added and purpose

| BibTeX key | Published reference | Purpose in the Introduction |
|---|---|---|
| `albert2002` | R. Albert and A.-L. Barabási, *Statistical Mechanics of Complex Networks*, Reviews of Modern Physics 74, 47--97 (2002) | Supports the statistical-mechanical connection between network topology, dynamics, and robustness. |
| `boccaletti2006` | S. Boccaletti, V. Latora, Y. Moreno, M. Chavez, and D.-U. Hwang, *Complex Networks: Structure and Dynamics*, Physics Reports 424, 175--308 (2006) | Supports the role of topology, community structure, propagation, perturbation, and collective dynamics in interconnected units. |
| `dorogovtsev2008` | S. N. Dorogovtsev, A. V. Goltsev, and J. F. F. Mendes, *Critical Phenomena in Complex Networks*, Reviews of Modern Physics 80, 1275--1335 (2008) | Supports the narrower statement that topology and strong finite-size effects can alter observed cooperative-network regimes; it is not cited as evidence of a transition in the present system. |

## 14. DOI and authoritative verification sources

- `albert2002`: https://doi.org/10.1103/RevModPhys.74.47 and the APS article
  and BibTeX records at
  https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.74.47
- `boccaletti2006`: https://doi.org/10.1016/j.physrep.2005.10.009 and the
  publisher abstract and publication record at
  https://www.sciencedirect.com/science/article/pii/S037015730500462X
- `dorogovtsev2008`: https://doi.org/10.1103/RevModPhys.80.1275 and the APS
  article record at
  https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.80.1275

The publisher abstracts were inspected for claim support. The bibliography now
has 40 unique keys, 38 unique DOI fields, and 40 unique normalized titles; the
duplicate-key, duplicate-DOI, and duplicate-title checks found zero duplicate
groups. All cited keys resolve under BibTeX.

## 15. Scientific preservation

Only the Introduction differs within `main.tex`: the pre-Introduction prefix
and the complete text from Section 2 onward have byte-identical hashes against
commit `3087aeb4`. The 15 displayed equation/align blocks have the same
whitespace-normalized ordered hash before and after:
`a19441c1e51b423f3bf01cb0b6fd7ee49899b171783cd698931ebcb4d721d2c5`.
All 14 `\resultfigure` declarations remain, and the rendered PDF contains 14
numbered figure captions.

No numerical result, interval, hypothesis disposition, model qualification,
caveat, scientific conclusion, result macro, or quantitative figure changed.
The revision does not describe a phase transition or projected path divergence
as literal entropy production.

## 16. Compilation and reference checks

Commands used:

```text
./scripts/build-statmech-v15-paper.sh
SOURCE_DATE_EPOCH=1787443941 FORCE_SOURCE_DATE=1 TZ=UTC latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex
```

The established build completed successfully through the required BibTeX and
pdfLaTeX passes. The final LaTeX log has no undefined citation, undefined
reference, duplicate-label, overfull-box, underfull-box, or rerun warning. The
BibTeX log has no warning or error. `git diff --check` is clean.

## 17. Final PDF

- Page count: 28 A4 pages.
- Figures: 14, each embedded once.
- Displayed equations/align blocks: 15.
- Final PDF hash before commit:
  `3407969cd9b0ea7d2e17d57d1fbf248a4b3e0f43036e03bbfe2c2a3193ddff40`.
- Standalone Figure 1 and all manuscript pages were visually inspected.

## 18. Files changed

Scientific manuscript and Figure 1 work:

- `paper/jstat_v15/main.tex`
- `paper/jstat_v15/main.pdf`
- `paper/jstat_v15/references.bib`
- `paper/jstat_v15/render_publication_figures.py`
- `thermoagent/statmech_llm_v15/figures.py`
- `results/collective_agent_statmech_v15/figures/pdf/figure01_augmented_state_architecture.pdf`
- `results/collective_agent_statmech_v15/figures/source_data/figure01_augmented_state_architecture.csv`
- `results/collective_agent_statmech_v15/figures/figure_catalog.csv`
- `results/collective_agent_statmech_v15/reproducibility/figure_generation.json`
- `notes/jstat_figure1_and_motivation_revision.md`

The author-supplied backup directory `paper/jstat_v15 (another copy)/` is added
unchanged in the same commit at the author's explicit request. It was not used
as an editorial or scientific source for this pass.

## 19. Commit and push status

The report is part of the implementation commit, so that commit's
content-derived hash and the subsequent normal-push outcome cannot be embedded
self-referentially here. They are reported in the author-facing final response.
The intended commit message is:

```text
Improve Figure 1 and strengthen study motivation
```

The normal push target is `origin/jstat-paper-structure-v16`; no force push,
rebase, reset, or merge is authorized or used.
