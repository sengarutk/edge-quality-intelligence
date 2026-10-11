# Edge Inspection Runtime

Research artifact for the paper *Alert Policies for Industrial Visual Inspection at the Edge: An
Integrated Evaluation of Temporal Gating, Sensor Fusion and Durable Event Delivery* (`paper/main.pdf`).
The paper is a systems-integration and evaluation study; it does not claim a new alarm algorithm.

The runtime turns per-frame visual anomaly scores (PatchCore) and three physical sensor channels
into operator alerts, and delivers every event to an MQTT broker through a durable SQLite spool.
The repository also contains an MVTec AD benchmark of PatchCore, PaDiM and a convolutional
autoencoder (`scripts/run_benchmark.py`), which is not part of the paper.

## What is real and what is simulated

| Component | Status |
|---|---|
| Visual anomaly scores in the policy experiments | Real PatchCore scores on 7 MVTec AD categories (development) and 12 VisA categories (held-out, policy frozen), normalized with held-out training images only; drawn per frame from small pools, independently (main runs) or with an AR(1) copula correlation of 0.9 (robustness runs) |
| Comparison policies | 8 variants of our policy and 3 external alarm baselines (delay-timer, EMA with hysteresis, decision-level fusion) on identical inputs |
| Glare artifacts | Synthetic highlights added to real good images, then scored by PatchCore |
| Sensor channels (vibration, temperature, current) | Simulated (`src/runtime/sensor_simulator.py`) |
| "IMS / C-MAPSS" traces | Hand-written synthetic curves inspired by those datasets, not derived from them |
| Broker durability | Real Mosquitto broker, real disconnects, broker restarts, and SIGKILLs of the publisher, the subscriber and the inspection process |
| Latency | Measured per stage with a real PatchCore forward pass on an RTX 4050 Laptop GPU and on CPU, paced at 30 FPS, with database writes on the inspection thread or on a writer thread |
| Operator queue | Analytical M/M/1 model plus simulation; not observed with operators |

## Main results (from `paper/generated_metrics.tex`)

Means over 21 (category, replicate) units on MVTec AD; see the paper for category-level intervals
and exact sign-flip permutation tests over categories.

* Good parts only, independent frame scores: single-frame thresholding raised 4,616 false
  alerts/h, smoothing alone 40.6/h, the full policy none. With temporally correlated scores
  (rho = 0.9) the full policy raised 188.6/h, so the zero result depends on independent sampling.
* A sustained defect produced 1.00 alert with the full policy (137.6 with single-frame
  thresholding) at a mean delay of 3.7 frames; a 1-3 frame glare burst still produced 0.66 alerts.
* External baselines: a textbook delay-timer suppressed glare far better (0.03 alerts/burst) but
  needed 15-frame defects for 0.95 recall (full policy: 8 frames); decision-level fusion matched
  the full policy on sustained defects and multi-modal recall but raised 5.4 vs 4.0 false
  line-stops/h and 125.6 vs 1.3 false alerts/h around machine states.
* Divergence triage lowered false line-stop (HIGH) escalations from 117.0/h to 4.0/h; without
  sensor fusion, recall on mechanical faults dropped from 1.00 to 0.62.
* Held-out VisA run (12 categories, 36 units, policy and parameters frozen before scoring):
  state gating (2.9 vs 107.1-122.6 false alerts/h around machine states, better in 12/12
  categories), divergence triage (104.4 -> 5.5 false line-stops/h) and the delay-timer's glare
  advantage replicated. The line-stop advantage over decision fusion did not (5.5 vs 5.9/h,
  p = 1.000); the full policy raised 1.7 false alerts/h on good parts, 1.34 alerts per defect, and
  reached a sustained-defect recall of only 0.77 (single-frame thresholding 0.99), because many
  VisA defects score near or below tau_med.
* Broker durability: no event lost under link losses up to 120 s, a broker restart and a publisher
  SIGKILL; every loss in the overflow case is covered by a counted eviction (at-least-once only).
* Subscriber and process crashes: with a clean session, a 10 s subscriber crash lost 308 events
  (all 300 sent while it was down); with a persistent session (now the default config) it lost
  none, but a 60 s crash still lost 807 under Mosquitto's default 1,000-message client queue and
  none with the limit removed (`configs/mosquitto.conf` now raises it). 100 SIGKILLs of the
  inspection process lost no decision with synchronous persistence or with the writer thread.
  Power loss was not tested; `synchronous=FULL` raised the median spool insert from 0.02 to 4.19 ms.
* Latency (GPU, paced at 30 FPS): 2.46% of cycles missed the 33.3 ms budget with synchronous
  database writes and 1.00% with a writer thread; the worst cycle was not improved. CPU-only
  inference missed the budget in every cycle.

Known limitations are listed in Section VI of the paper and in `docs/design/failure-modes.md`.

### Note on the autoencoder baseline (MVTec benchmark)
On the official MVTec AD images the convolutional autoencoder scores near or below chance on
most categories (mean image AUROC 0.30 on metal_nut, 0.38 on carpet, 0.45 to 0.52 on cable, bottle
and leather) and is useful only on grid (0.86) and hazelnut (0.76); see
`results/benchmark_f1/mvtec_ad/tables/summary_multiseed.md`. Its bottleneck (128 x 16 x 16) is wide,
so it can reconstruct defects as well as good parts. Treat it as a weak lower reference, not as a
tuned competitor. The robustness (corruption) stress test was not run for the committed tables,
which therefore show "--" for MRD.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Experiments additionally need the MVTec AD categories bottle, cable, carpet, grid, hazelnut,
leather and metal_nut in `data/mvtec_ad/` (`python scripts/download_dataset.py`), the `mosquitto`
binary for the durability benchmark, and `pdflatex`/`bibtex` for the paper.

## Reproduce the paper

```bash
bash scripts/reproduce_all_paper_results.sh
```

This runs the tests, builds the PatchCore score banks, the policy ablation (1,386 runs with
independent and 1,386 with correlated scores on MVTec AD; 2,376 of each on VisA), the short-defect
study, the sensitivity sweep, the trace replay, the latency and broker benchmarks, regenerates
every macro, table and figure (`scripts/build_paper_assets.py`, which fails instead of guessing if an input is
missing) and compiles `paper/main.pdf`.

## Tests

```bash
python -m pytest tests/
```

`tests/test_rt_policy.py` states the policy rules as contract tests;
`tests/test_rt_mqtt_resilience.py` includes a round trip through a real broker when `mosquitto`
is installed.

## Layout

```
configs/            policy, sensor, MQTT and system configuration; workload definitions (scenarios/)
src/runtime/        inference, policy, sensors, spool, MQTT, audit log, fault injection
src/experiments/    workload generation and replay; MVTec benchmark experiments
src/metrics/        alert metrics, statistics, queue models, detection metrics
src/models/         PatchCore, PaDiM, autoencoder (src/methods is an alias)
scripts/            experiment drivers and asset builders
results/            JSON outputs used by the paper (score-bank models and score archives are not committed)
paper/              everything needed to publish: main.tex, references.bib, IEEEtran.bst,
                    generated_metrics.tex, tables/, figures/, main.pdf, compile_paper.sh
docs/design/        design notes (architecture, policy, event schema, failure modes, data card)
tests/              unit, contract and integration tests
```

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
