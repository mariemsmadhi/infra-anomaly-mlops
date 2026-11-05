# Infra Anomaly MLOps

An end-to-end MLOps pipeline that learns what "normal" looks like for Linux servers and flags abnormal behavior (CPU spikes, memory pressure, filling disks). It is validated by injecting real failures into virtual machines and measuring how fast they are detected.

**Stack:** Vagrant (VMware Fusion), Prometheus, Grafana, Docker Compose, scikit-learn, MLflow, FastAPI, Bash, Python.

<!-- TODO: add a demo GIF here (failure injected -> ANOMALY in monitor -> alert Firing in Prometheus) -->

## Why this project

A model is only useful if it is deployed, monitored and kept up to date. This project applies DevOps practices to a small ML service that watches infrastructure I built myself:

- two real Linux servers (VMs) provisioned as code,
- a monitoring stack (Prometheus + Grafana) running in containers,
- an anomaly detector trained on their normal behavior, tracked with MLflow and served through an API,
- the API itself monitored like any production service,
- failure injection to measure detection quality instead of assuming it.

**How a verdict becomes an alert:**

1. `monitor.py` reads the latest metrics of each server from Prometheus every 15 seconds.
2. It sends them to the API (`POST /predict`), which answers `normal` or `anomaly` with a score.
3. The API publishes the verdict as a Prometheus gauge, `anomaly_flag{server="..."}` (1 = anomaly, 0 = normal).
4. Prometheus scrapes the API and evaluates the alert rules. If the flag stays at 1 for 30 seconds, the alert goes from *Pending* to *Firing*.

## Tech stack

| Layer | Technology | Role |
|---|---|---|
| Virtualization | Vagrant + VMware Fusion | Two Ubuntu 22.04 (arm64) VMs: `web01` (Nginx), `db01` (MariaDB) |
| Provisioning | Bash (`scripts/provision.sh`) | Installs Node Exporter, Nginx, MariaDB, stress-ng |
| Metrics | Node Exporter + Prometheus | CPU, memory, disk, load, network |
| Dashboards | Grafana (dashboard 1860) | Server and service visualization |
| ML | scikit-learn | Isolation Forest + z-score baseline |
| Experiment tracking | MLflow (SQLite backend) | Parameters, metrics, model versions |
| Serving | FastAPI + Uvicorn + Pydantic | `/predict`, `/health`, `/reload`, `/metrics` |
| Containers | Docker + Docker Compose | Prometheus, Grafana, API |
| Tests | pytest | API behavior with a small fake model |
| Failure injection | stress-ng, fallocate | CPU, memory and disk incidents |

## Repository structure

```
.
├── Vagrantfile                      # the two VMs
├── docker-compose.yml               # Prometheus, Grafana, API
├── Dockerfile                       # API image
├── api/
│   ├── main.py                      # FastAPI service + Prometheus metrics
│   └── requirements.txt             # pinned to the training environment
├── src/
│   ├── collect_metrics.py           # Prometheus -> data/metrics.csv
│   ├── train.py                     # training, MLflow tracking, model export
│   ├── monitor.py                   # live metrics -> /predict -> logs/monitor.csv
│   └── measure_detection.py         # failures vs verdicts -> results table
├── scripts/
│   ├── provision.sh                 # VM provisioning
│   ├── generate_load.sh             # light random activity (non-flat "normal")
│   ├── inject_failure.sh            # cpu | mem | disk failures
│   └── run_series.sh                # runs a series of failures with calm pauses
├── prometheus/
│   ├── prometheus.yml               # scrape jobs + rule files
│   └── alert.rules.yml              # alert rules
├── grafana/provisioning/datasources/prometheus.yml
├── tests/                           # pytest
└── models/current/                  # exported model (git-ignored)
```

## Data and model

**Features** (one row per server per minute, 8 columns):

| Feature | Meaning |
|---|---|
| `cpu_pct` | CPU busy percentage (2-minute average) |
| `mem_pct` | Memory used (based on `MemAvailable`) |
| `disk_pct` | Root filesystem used |
| `load1` | 1-minute load average |
| `net_rx_kbps`, `net_tx_kbps` | Network traffic (2-minute average) |
| `is_web01`, `is_db01` | One-hot encoding of the server |

**Training** (`src/train.py`):

- **Time-based split:** the first 80% of each server's data trains the model, the most recent 20% tests it (no shuffling, since the model must predict the future).
- **Baseline:** a z-score rule (flag a row if any metric is more than 4 standard deviations from the server's usual value).
- **Model:** `Pipeline(StandardScaler, IsolationForest(n_estimators=200, contamination=0.01))`. It learns only from normal data, with no labeled failures.
- **First evaluation:** synthetic anomalies are injected into the test set to compute recall, precision, F1, ROC-AUC and false-alarm rate. These are provisional; the real evaluation uses injected failures (below).
- **MLflow:** each run logs parameters and metrics, and registers the model as `infra-anomaly-detector` with a version number.
- **Export:** the model is saved to `models/current/` (`model.joblib` + `metadata.json`), which the API loads at startup or on `POST /reload`.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Verdict for one server's metrics: `{"server","anomaly","score","model_version"}` |
| `/health` | GET | Service status, whether a model is loaded, model version |
| `/reload` | POST | Load the newest model from disk without restarting |
| `/metrics` | GET | Prometheus metrics of the service itself |
| `/docs` | GET | Interactive documentation (Swagger UI) |

Invalid input returns `422` (for example CPU above 100 or an unknown server). No loaded model returns `503`.

**Metrics exposed by the API:**

| Metric | Type | Meaning |
|---|---|---|
| `anomaly_flag{server}` | Gauge | 1 if the latest verdict was an anomaly, else 0 |
| `anomaly_score{server}` | Gauge | Latest anomaly score (higher = more abnormal) |
| `anomaly_predictions_total{server,result}` | Counter | Predictions made |
| `anomaly_prediction_seconds` | Histogram | Prediction latency |
| `anomaly_model_loaded` | Gauge | 1 if a model is loaded |

## Quick start

**Prerequisites:** VMware Fusion, Vagrant with the `vagrant-vmware-desktop` plugin and the Vagrant VMware Utility, Docker Desktop, Python 3.11+. Developed on an Apple Silicon Mac (arm64).

```bash
# 1. Python environment for the scripts
python -m venv .venv && source .venv/bin/activate
pip install pandas requests scikit-learn mlflow joblib fastapi "uvicorn[standard]" prometheus-client httpx pytest

# 2. Servers
vagrant up --provider=vmware_desktop
curl -s http://192.168.56.11:9100/metrics | head -3          # Node Exporter answers

# 3. Monitoring stack and API
docker compose up -d --build
# Prometheus: http://localhost:9090   Grafana: http://localhost:3000 (admin/admin)

# 4. Collect normal data (let it run for 2-3 days for a meaningful model)
nohup bash scripts/generate_load.sh > /dev/null 2>&1 &
python src/collect_metrics.py --hours 72

# 5. Train, then load the model into the API
python src/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db            # http://localhost:5000
curl -X POST localhost:8000/reload
curl -s localhost:8000/health

# 6. Watch the servers live (keep this terminal open)
python src/monitor.py
```

Run the tests with `python -m pytest -q`.

> `api/requirements.txt` pins the library versions of the training environment. The model file is pickled, so the API container must use compatible versions.

## Failure injection and measurement

Detection quality is measured, not assumed. With `monitor.py` running in another terminal:

```bash
scripts/inject_failure.sh cpu web01 120      # also: mem, disk ; any VM ; duration in seconds
scripts/run_series.sh                         # several failures with calm pauses in between
python src/measure_detection.py               # results table
```

| Script | Role |
|---|---|
| `inject_failure.sh` | Creates a real incident inside a VM and records its start time in `logs/injections.csv` |
| `monitor.py` | Records every verdict in `logs/monitor.csv` |
| `measure_detection.py` | Compares the two files: detected or not, detection time, and the false-alarm rate outside failure windows |

The injector and the monitor are independent: the model is never told a failure is coming.

### Results

Measured on a small lab (two mostly idle 1-vCPU VMs, 15-second checks). Detection times have a resolution of about 15 seconds.

| Scenario | Server | Detected | Detection time | Runs |
|---|---|---|---|---|
| CPU saturation (95%) | web01 | yes | ~23 s | 1 |
| Memory pressure (70% of free RAM) | web01 | yes | 30 s | 1 |
| Disk fill (15% of the disk) | web01 | not measured yet | - | 0 |
| CPU saturation | db01 | not measured yet | - | 0 |

**False alarms during normal operation:** 18 alerts out of 2,820 checks (0.6%). A single anomalous check does not fire an alert, because the alert rule requires the flag to stay at 1 for 30 seconds.

<!-- TODO: update this table after the full failure series, and add the number of runs per scenario -->

## Alerting

Alert rules live in `prometheus/alert.rules.yml` and are evaluated by Prometheus. Their state is visible at `http://localhost:9090/alerts` (Inactive, Pending, Firing).

| Alert | Condition | Meaning |
|---|---|---|
| `ServerAnomalyDetected` | `anomaly_flag == 1` for 30 s | The model considers a server abnormal |
| `AnomalyModelNotLoaded` | `anomaly_model_loaded == 0` for 1 m | The API has no model: the detector is blind |
| `NoPredictionsReceived` | no predictions for 2 m | `monitor.py` is stopped: nothing is watching |
| `NodeExporterDown` | `up{job="node"} == 0` for 1 m | A server is unreachable |

**Current status:** alerts fire in Prometheus. **Notification delivery (Telegram or Slack through Alertmanager) is not configured yet**; it is on the roadmap.

## Known limitations

- **Small, very flat dataset.** The VMs are almost idle (CPU around 1%), so the model is sensitive and the training set is small. More days of data and more varied load would give a more robust model.
- **Few runs per scenario.** Detection times come from single runs and have about 15 s resolution. They are indicative, not statistics.
- **Isolation Forest scores saturate.** It tells you a point is outside the normal range, not by how much: a CPU at 95% scores about the same as one at 55%. A z-score rule is actually better on an isolated extreme spike; the Isolation Forest is more useful for unusual combinations of metrics.
- **Detection delay.** CPU and network metrics are 2-minute averages, so they rise gradually after a failure starts, and the verdict returns to `normal` 1-2 minutes after it ends. The 30 s alert delay adds to it.
- **Synthetic first evaluation.** The initial scores in MLflow use injected fake anomalies in the test set; the failure-injection results are the real measurement.
- **The flag keeps the latest verdict.** If `monitor.py` stops while `anomaly_flag` is 1, it stays at 1 until the next prediction; `NoPredictionsReceived` exists to catch this.
- **Single-node lab.** Two VMs on one laptop, not a production environment.

## Roadmap

- [x] VMs as code, monitoring stack, data export
- [x] Training with MLflow tracking and a baseline comparison
- [x] FastAPI service with Prometheus metrics, tests, Docker
- [x] Live monitor, failure injection, detection measurement
- [x] Alert rules evaluated in Prometheus
- [ ] Complete the failure series (disk, repeats, db01) and update the results table
- [ ] Notifications with Alertmanager (Telegram or Slack)
- [ ] Retrain on 2-3 days of data (model v2) and compare with v1 in MLflow
- [ ] Hybrid detector (z-score as a second opinion) if single-metric failures are missed
- [ ] Scheduled retraining (`scripts/retrain.sh`)
- [ ] CI with GitHub Actions (ruff, pytest, shellcheck, Docker build)

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `node_exporter` fails with "Start request repeated too quickly" | Wrong-architecture binary (amd64 on an arm64 VM) | Remove `/usr/local/bin/node_exporter`, `systemctl reset-failed node_exporter`, run `vagrant provision` |
| VMware error "Cannot open the disk ... Directory not empty" | Stale `.lck` lock after an unclean shutdown | `find .vagrant -name "*.lck" -exec rm -rf {} +`, then `vagrant up` |
| Prometheus target `api:8000` down ("no such host") | The `api` container is not running | `docker compose ps -a`, `docker compose logs api` |
| `/health` shows `model_loaded: false` | No exported model yet | `python src/train.py`, then `curl -X POST localhost:8000/reload` |
| `measure_detection.py` shows `-` for the score | The monitor was not running during the failure | Re-run the failure with `monitor.py` open |

## Author

Mariem Smadhi. IT support engineer moving toward systems administration and DevOps.
