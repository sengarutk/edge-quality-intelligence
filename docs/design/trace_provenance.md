# Provenance of Sensor Data

* **Online sensor simulator** (`src/runtime/sensor_simulator.py`): first-order thermal model,
  load-scaled vibration and current, Gaussian noise. Parameters are in
  `configs/sensor_config.yaml`. It is not fitted to any machine.
* **Synthetic run-to-failure proxies** (`src/runtime/trace_replay.py`,
  `generate_ims_bearing_trace` and `generate_cmapss_turbofan_trace`, written to
  `data/traces/bearing_proxy.csv` and `data/traces/thermal_creep_proxy.csv`): hand-written
  piecewise curves (healthy plateau, incipient drift, runaway) whose *shape* is inspired by
  the NASA IMS bearing and C-MAPSS turbofan datasets. No value is taken from or fitted to
  those datasets, and they must not be described as real or calibrated traces.
* **Replay scoring**: the replay calibrates channel means and standard deviations on the first
  60 samples and scores readings with the same fusion rule as the simulator
  (`composite_sensor_score`). The benchmark (`scripts/run_real_trace_benchmark.py`) excludes the
  calibration samples from scoring; the onset is step 420 (bearing) and 390 (thermal creep).
