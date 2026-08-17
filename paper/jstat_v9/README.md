# JSTAT V9 working paper

This is a compact, non-final manuscript prepared against the current IOP/JSTAT
author guidance checked on 2026-08-17. IOP accepts common LaTeX variants and
does not require its class file at initial submission, so the draft uses the
standard 12-point `article` class for reviewer readability.

Build after generating the V9 tables and figures:

```bash
cd paper/jstat_v9
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The numerical macros in `results_macros.tex` are generated from the compact
principal-results summary. All references in `references.bib` were checked
against publisher or DOI metadata; no citation was invented. The current file
is a working paper package, not a submission-ready claim of a thermodynamic-limit
transition.
