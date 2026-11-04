"""Read live metrics from Prometheus, ask the API for a verdict, and log everything."""
import argparse
import csv
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from collect_metrics import QUERIES

PROM = os.environ.get("PROM_URL", "http://localhost:9090")
API = os.environ.get("API_URL", "http://localhost:8000")
COLUMNS = ["timestamp", "server", "anomaly", "score", *QUERIES.keys()]
PERCENT_FIELDS = ("cpu_pct", "mem_pct", "disk_pct")


def fetch_current():
    """Latest value of each metric per server: {"web01": {"cpu_pct": 8.1, ...}}."""
    data = {}
    for name, query in QUERIES.items():
        r = requests.get(f"{PROM}/api/v1/query", params={"query": query}, timeout=10)
        r.raise_for_status()
        for series in r.json()["data"]["result"]:
            server = series["metric"].get("instance_name")
            value = float(series["value"][1])
            if server and not math.isnan(value):
                data.setdefault(server, {})[name] = value
    # keep only servers for which we got every metric
    return {s: m for s, m in data.items() if len(m) == len(QUERIES)}


def clip(metrics):
    """Rates can come out slightly below 0 or above 100: keep values in the API's valid range."""
    return {k: (min(max(v, 0.0), 100.0) if k in PERCENT_FIELDS else max(v, 0.0))
            for k, v in metrics.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=15, help="seconds between checks")
    p.add_argument("--log", default="logs/monitor.csv")
    args = p.parse_args()

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log.exists()
    f = open(log, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    if new_file:
        writer.writeheader()

    print(f"Monitoring every {args.interval}s  (Prometheus: {PROM}, API: {API}). Ctrl+C to stop.")
    while True:
        started = time.time()
        try:
            for server, metrics in fetch_current().items():
                metrics = clip(metrics)
                r = requests.post(f"{API}/predict", json={"server": server, **metrics}, timeout=5)
                r.raise_for_status()
                res = r.json()
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                writer.writerow({"timestamp": ts, "server": server, "anomaly": res["anomaly"],
                                 "score": res["score"], **{k: round(v, 3) for k, v in metrics.items()}})
                f.flush()
                flag = "ANOMALY" if res["anomaly"] else "ok     "
                print(f"{ts} {server:6} {flag} score={res['score']:.3f} "
                      f"cpu={metrics['cpu_pct']:.1f} mem={metrics['mem_pct']:.1f}")
        except Exception as e:  # keep running if Prometheus or the API is briefly unavailable
            print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} error: {e}")
        time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
