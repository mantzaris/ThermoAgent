# V4 maintenance after the immutable result snapshot

The scientific snapshot remains commit
`8ccd27df248940fc0cbb55c43a30949de3370533`. This maintenance changes no V4
episode, candidate, statistic, gate decision, or scientific interpretation.

## Classified changes

- **Line-ending only:** `results/smoke/episodes.csv` and
  `results/smoke/history/episodes-5fe54b2403ca.csv` are normalized from CRLF to
  LF. Parsed values and row order are identical. Original and normalized hashes
  are recorded in `results/human_operator_v4/reproducibility/`.
- **Presentational:** the two lowest utility-network labels move above their
  nodes so they do not overlap the legend; the source topology and values are
  unchanged.
- **Presentational:** dashboard cumulative operator minutes are labeled
  "Minutes before decision," and an action logged at the same step is labeled
  "applied after displayed view." This documents the existing event order.
- **Reproducibility:** the top-level and V4 indexes, PDF QA records, and
  maintenance verification records are rebuilt after these changes.

The V4 numerical gate report is compared byte-for-byte with the immutable
snapshot during final maintenance verification.
