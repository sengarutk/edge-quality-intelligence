# Experimental Environment

All results in `results/` and in the paper were produced on one machine:

| Item | Value |
|---|---|
| Host | Laptop, Linux under WSL2 (kernel 6.6, x86_64), 16 logical CPUs |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) |
| Python | 3.12.3 |
| MQTT broker | Mosquitto 2.0.18 (local instance started by the benchmark) |
| SQLite | 3.45.1 |

Exact package versions and the git commit are written to `results/environment_manifest.json`
by `scripts/write_environment_manifest.py`. No experiment was run on embedded hardware
(e.g. Jetson) or an industrial gateway; latency on such devices is unknown.
