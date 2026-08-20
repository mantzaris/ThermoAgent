# JSTAT V13 working manuscript

The review manuscript uses standard 12-point LaTeX. Current IOP guidance,
checked on 19 August 2026, states that common TeX variants are accepted and the
IOP class is optional for initial review. It requests at least 12-point
reviewer-facing text, embedded figures, vector PDF or EPS where possible, and
roughly 8--12 point text at final figure size.

Primary guidance:

- https://publishingsupport.iopscience.iop.org/questions/article-format/
- https://publishingsupport.iopscience.iop.org/questions/latex-template/
- https://publishingsupport.iopscience.iop.org/questions/figures-journal-articles/
- https://publishingsupport.iopscience.iop.org/journals/journal-of-statistical-mechanics-theory-and-experiment/

Build with `scripts/build-statmech-v13-paper.sh`. Numerical macros are generated
from the compact V13 aggregate tables; no result is manually typed into a
figure. All bibliography entries were checked against publisher or arXiv
primary records. The recent related-work audit includes arXiv:2608.16578,
2608.02827, 2605.10528, 2601.05606, and 2510.22422.
