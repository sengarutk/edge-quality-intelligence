# Portfolio Notes for This Project

Talking points, CV bullets and statement-of-purpose material for graduate applications.
Every number below comes from `paper/generated_metrics.tex` (regenerated from `results/`);
if the experiments are re-run, check the numbers again before using them.

---

## 1. Where this project fits

This repository is the edge-runtime project ("Flagship 4"). It also contains an MVTec AD
benchmark of PatchCore, PaDiM and an autoencoder ("Flagship 1"). The quantization/TensorRT
("Flagship 2") and custom-kernel ("Flagship 3") projects are **not** in this repository, so no
claim about them can be supported from here.

---

## 2. Statement-of-purpose material

### Reliable decisions from anomaly scores
> A visual anomaly detector gives a score per frame, but an inspection line needs alerts an
> operator can act on. I built an edge runtime that combines PatchCore scores with vibration,
> temperature and current telemetry through smoothing, k-of-N persistence, machine-state
> gating, cross-modal triage and an incident latch. I evaluated eight policy variants on six
> workloads driven by real PatchCore scores on seven MVTec AD categories (1,008 runs). On
> good parts, single-frame thresholding raised 4,616 false alerts per hour and my policy none;
> a sustained defect produced about one alert instead of 138, at a delay of 3.7 frames. The
> experiments also showed where the design falls short: short glare still produced 0.66 alerts
> per burst, defects visible in only one to three frames were often missed, and the blur check
> mistook some real defects for a defocused camera.

### Durable delivery
> I built a store-and-forward MQTT publisher that writes every event to a SQLite spool before
> sending it and deletes it only after the broker acknowledges it. Tested against a real
> Mosquitto broker, it lost no event through 120-second link outages, a broker restart and a
> killed publisher process, and every loss in an intentional overflow test was accounted for.
> Measuring end-to-end latency with a real PatchCore forward pass showed that 1.62% of
> cycles still missed a 33 ms frame budget, which pointed to moving decoding and database
> writes off the inspection thread as the next thing to fix.

---

## 3. CV bullets (all supported by the repository)

- **Edge inspection runtime (Python, PyTorch, SQLite, MQTT):** turned PatchCore anomaly scores
  and sensor telemetry into operator alerts; with real MVTec AD scores, reduced alerts per
  sustained defect from 138 to 1.1 and false alerts on good parts from 4,616/h to zero.
- **Ablation study:** 8 policy variants x 6 workloads x 21 paired units (1,008 runs), with
  bootstrap confidence intervals, Holm-corrected Wilcoxon tests and a sensitivity sweep.
- **Durable messaging:** acknowledgement-driven SQLite spool; zero lost events against a real
  broker under outages up to 120 s, broker restarts and process crashes.
- **Measured latency:** per-stage timing on an RTX 4050 Laptop GPU (mean 24.4 ms per cycle);
  identified and fixed a synchronous-fsync bottleneck in the audit log.
- **Testing:** 234 automated tests, including policy contract tests and a real-broker
  integration test.

---

## 4. Interview talking points

### Why not threshold each frame?
Scores of good parts regularly cross the review threshold, and one defect stays in view for
dozens of frames, so per-frame alerts flood the operator. Persistence alone is not enough:
it lets anything that lasts a few frames through as a stream of alerts. The incident latch,
which merges a sustained condition into one alert, gave the largest reduction.

### How is delivery made durable?
Events are written to the spool before any network I/O; one thread sends them in order and
deletes a row only in the broker-acknowledgement callback. Delivery is at-least-once, so
consumers de-duplicate on the event id. The spool evicts the oldest rows only when it is
full, and counts them.

### What does sensor fusion add?
Without sensors, recall on mechanical faults that are invisible to the camera dropped from
1.00 to 0.62. Divergence triage routes visual evidence that the sensors do not support to a
review instead of a line stop, which cut false line-stop alerts from 116 to 24 per hour.

### Honest limitations to mention
Sensors are simulated, glare is synthetic, workloads are denser than a real line, and the
evaluation compares only ablations of the same policy.
