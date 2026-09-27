# Edge Inspection Runtime

Research artifact for the paper *Reducing Alert Fatigue in Industrial Edge Inspection Through
Multi-Modal Temporal Policy Gating and Durable Event Spooling* (`paper/main.pdf`).

The runtime turns per-frame visual anomaly scores (PatchCore) and three physical sensor channels
into operator alerts, and delivers every event to an MQTT broker through a durable SQLite spool.
The repository also contains an MVTec AD benchmark of PatchCore, PaDiM and a convolutional
autoencoder (`scripts/run_benchmark.py`), which is not part of the paper.

## What is real and what is simulated

| Component | Status |
|---|---|
| Visual anomaly scores in the policy experiments | Real PatchCore scores on 7 MVTec AD categories, normalized with held-out training images only |
| Glare artifacts | Synthetic highlights added to real good images, then scored by PatchCore |
| Sensor channels (vibration, temperature, current) | Simulated (`src/runtime/sensor_simulator.py`) |
| "IMS / C-MAPSS" traces | Hand-written synthetic curves inspired by those datasets, not derived from them |
| Broker durability | Real Mosquitto broker, real disconnects, broker restarts and process kills |
| Latency | Measured per stage with a real PatchCore forward pass on an RTX 4050 Laptop GPU and on CPU |
| Operator queue | Analytical M/M/1 model plus simulation; not observed with operators |

## Main results (from `paper/generated_metrics.tex`)

* Good parts only: single-frame thresholding raised 2,752 false alerts/h; every temporal policy raised 0-2.
* A sustained defect produced 1.08 alerts with the full policy (141.9 with the baseline) at a
  mean delay of 3.4 frames; a 1-3 frame glare burst still produced 0.61 alerts on average.
* Divergence triage lowered false line-stop (HIGH) escalations from 91.2/h to 17.4/h; without
  sensor fusion, recall on mechanical faults dropped from 1.00 to 0.62.
* Broker durability: no event lost under link losses up to 120 s, a broker restart and a publisher
  SIGKILL; losses in the overflow case equal the counted evictions.
* Latency (GPU): mean 21.3 ms per cycle, but 1.24% of cycles exceeded the 33.3 ms budget
  (SQLite checkpoint spikes); CPU-only inference missed the budget in every cycle.

Known limitations are listed in Section VI of the paper and in `docs/design/failure-modes.md`.

### Note on the autoencoder baseline (MVTec benchmark)
The convolutional autoencoder scores near or below chance on several object categories
(e.g. metal_nut image AUROC about 0.30) but perfectly on grid and leather. This is not a
pipeline error: the model reconstructs defects almost as well as good parts (reconstruction
MSE about 0.02 against an input variance of 1.4), and the remaining residual tracks image
brightness, which on metal_nut is lower for defective parts. Narrowing the bottleneck to 16 or
4 channels did not change this (0.30 and 0.32). Plain L2 autoencoders are known to be weak on
these objects; treat this baseline as a lower reference, not as a tuned competitor.

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

This runs the tests, builds the PatchCore score banks, the policy ablation (1,008 runs), the
sensitivity sweep, the trace replay, the latency and broker benchmarks, regenerates every macro,
table and figure (`scripts/build_paper_assets.py`, which fails instead of guessing if an input is
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
