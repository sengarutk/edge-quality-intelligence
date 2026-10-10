# Revision history of the paper results

## Revision 2 (focused revision after external review)

Changes to the method:
- **Rule (4) fix.** An uncorroborated HIGH now also requires the current smoothed visual score to
  reach tau_high. Before, the k-of-N window kept reporting high evidence for several frames after
  a glare burst ended while |v - s| had already fallen below tau_div, so decaying glare escalated to
  HIGH on nominal sensors. The change was made after inspecting the first ablation and is
  disclosed in the paper (Sections III-B and VI).
- **External baselines** (`src/runtime/alarm_baselines.py`): DELAY_TIMER, EMA_HYSTERESIS,
  DECISION_FUSION, defined from the existing thresholds before they were run.
- **Temporally correlated scores**: a second ablation with an AR(1) Gaussian copula, rho = 0.9
  (`results/ablation_rho0.9/`).
- **Statistics**: intervals in the paper now resample categories (cluster bootstrap); the
  ablation summary JSON keeps its unit-level fields unchanged.
- **Latency**: the benchmark is now paced at 30 FPS (it ran back-to-back before) and measures
  synchronous persistence and a writer thread (`src/runtime/async_writer.py`).
- **Trace replay**: reports when the first flag happens relative to the onset and how many alerts
  fall before and after it, instead of a recall/delay that counted an incident opened before onset.
- Experimental units are (category, replicate) with replicate 0-2; the former labels "seeds 11,
  23, 37" were never used as random seeds and have been removed.

## Values quoted by Paper B (revision 1 -> revision 2)

Means over 21 units; brackets are 95% intervals. Revision-2 intervals resample *categories*
and are therefore wider than the unit-level intervals of revision 1.

| Quantity | Revision 1 | Revision 2 |
|---|---|---|
| Nominal FA/h, BASELINE (B0) | 4,616 [2,690, 6,764] | 4,616 [1,557, 8,374] |
| Nominal FA/h, EMA_ONLY (B1) | 40.6 | 40.6 |
| Nominal FA/h, EMA_KOFN (B2), FULL_POLICY (B3) | 0.0 | 0.0 |
| Alerts/episode, B0 | 137.64 | 137.64 [125.15, 149.53] |
| Alerts/episode, B1 | 149.21 | 149.21 |
| Alerts/episode, B2 | 153.95 | 153.95 [149.39, 157.89] |
| Alerts/episode, B3 | 1.13 [1.00, 1.27] | **1.00 [1.00, 1.00]** (rule-(4) fix) |
| Mean delay (frames), B2 and B3 | 3.7 | 3.7 |
| Glare alerts/burst, B0 | 2.63 | 2.63 [2.24, 3.14] |
| Glare alerts/burst, B1 | 4.27 | 4.27 |
| Glare alerts/burst, B2 | 5.66 | 5.66 [3.64, 7.33] |
| Glare alerts/burst, B3 | 0.66 [0.55, 0.78] | 0.66 [0.44, 0.83] |
| Critical FA/h, B3 (multi-modal) | 28.9 | **4.0** (rule-(4) fix) |
| States FA/h, B3 | 15.3 | **1.3** (rule-(4) fix) |
| Mean GPU cycle (ms) | 24.4 (unpaced) | **27.3** synchronous, 26.8 writer thread (paced at 30 FPS) |
| Vision stage (ms) | 13.6 (unpaced) | **16.9** (paced) |
| GPU frame-budget misses | 1.62% (unpaced) | 2.46% synchronous, 1.00% writer thread |
| Operator-load crossover, B3 | ~35 episodes/h | **~40** episodes/h |

Paper B macros that quote revision-1 values (for example `\PaperALatency` = 24.4 and
`\PaperAVisionLatency` = 13.6) must be updated before Paper B is submitted.

## Short defects (unchanged by the rule fix; external baselines added)

Mean recall over 21 units, 20 defects of fixed length per 9,000-frame run:

| Length (frames) | 1 | 2 | 3 | 5 | 10 | min. length for 0.95 |
|---|---|---|---|---|---|---|
| B0 | 0.96 | 0.99 | 1.00 | 1.00 | 1.00 | 1 |
| B2, B3 | 0.10 | 0.43 | 0.70 | 0.91 | 0.99 | 8 |
| DELAY_TIMER, DECISION_FUSION | 0.00 | 0.01 | 0.07 | 0.80 | 0.93 | 15 |

## Revision 1

Rerun on the official MVTec AD release: `grid` and `leather` had been synthetic stand-ins and
`bottle` had an extra `test/defect` folder. All seven categories were linked to the official
archive (MD5 `eefca59f2cede9c3fc5b6befbfec275e`, checked by
`scripts/download_dataset.py --verify-only`) and every result was regenerated.
