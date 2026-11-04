"""Train an anomaly detector on server metrics and track it with MLflow."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "infra-anomaly"
MODEL_NAME = "infra-anomaly-detector"

METRICS = ["cpu_pct", "mem_pct", "disk_pct", "load1", "net_rx_kbps", "net_tx_kbps"]
SERVERS = ["web01", "db01"]
FEATURES = METRICS + [f"is_{s}" for s in SERVERS]


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    for s in SERVERS:
        df[f"is_{s}"] = (df["server"] == s).astype(int)
    return df


def time_split(df, train_frac=0.8):
    """Train on the past, test on the most recent data (no shuffling)."""
    train, test = [], []
    for _, g in df.groupby("server"):
        cut = int(len(g) * train_frac)
        train.append(g.iloc[:cut])
        test.append(g.iloc[cut:])
    return pd.concat(train), pd.concat(test)


def inject_anomalies(test, frac=0.05, seed=42):
    """Create fake incidents in the test set to measure detection quality.
    """
    rng = np.random.default_rng(seed)
    data = test.copy().reset_index(drop=True)
    labels = np.zeros(len(data), dtype=int)
    idx = rng.choice(len(data), size=max(1, int(len(data) * frac)), replace=False)
    for i in idx:
        kind = rng.choice(["cpu", "mem", "disk", "net"])
        if kind == "cpu":
            data.loc[i, "cpu_pct"] = min(100, data.loc[i, "cpu_pct"] + rng.uniform(40, 60))
            data.loc[i, "load1"] += rng.uniform(1, 4)
        elif kind == "mem":
            data.loc[i, "mem_pct"] = min(100, data.loc[i, "mem_pct"] + rng.uniform(25, 40))
        elif kind == "disk":
            data.loc[i, "disk_pct"] = min(100, data.loc[i, "disk_pct"] + rng.uniform(15, 30))
        else:
            data.loc[i, "net_rx_kbps"] += rng.uniform(500, 2000)
            data.loc[i, "net_tx_kbps"] += rng.uniform(500, 2000)
        labels[i] = 1
    return data, labels


def score_report(y_true, y_pred, scores):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def zscore_baseline(train, test, threshold):
    """Simple rule: flag a point if any metric is far from the server's usual value."""
    stats = train.groupby("server")[METRICS].agg(["mean", "std"])
    scores = np.zeros(len(test))
    for server in SERVERS:
        mask = (test["server"] == server).to_numpy()
        if not mask.any() or server not in stats.index:
            continue
        mean = stats.loc[server].xs("mean", level=1)
        std = stats.loc[server].xs("std", level=1).replace(0, 1e-6).fillna(1e-6)
        z = ((test.loc[mask, METRICS] - mean) / std).abs()
        scores[mask] = z.max(axis=1).to_numpy()
    return (scores > threshold).astype(int), scores


def export_model(pipe, metrics, run_id, version):
    """Save the model in a plain folder so the API can load it without MLflow."""
    out = Path("models/current")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    meta = {
        "features": FEATURES,
        "servers": SERVERS,
        "model_name": MODEL_NAME,
        "model_version": str(version) if version else None,
        "mlflow_run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Model exported to {out}/ (MLflow version {version})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/metrics.csv")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--contamination", type=float, default=0.01)
    p.add_argument("--z-threshold", type=float, default=4.0)
    args = p.parse_args()

    df = load_data(args.csv)
    if len(df) < 200:
        raise SystemExit(f"Only {len(df)} rows. Collect more data first.")
    train, test = time_split(df)
    test_inj, y_true = inject_anomalies(test)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # --- Baseline: z-score rule ---
    with mlflow.start_run(run_name="baseline-zscore"):
        mlflow.log_params({"model": "zscore", "z_threshold": args.z_threshold})
        pred, scores = zscore_baseline(train, test_inj, args.z_threshold)
        m = score_report(y_true, pred, scores)
        mlflow.log_metrics(m)
        print("Baseline z-score :", {k: round(v, 3) for k, v in m.items()})

    # --- Isolation Forest ---
    with mlflow.start_run(run_name="isolation-forest"):
        params = {"model": "isolation_forest", "n_estimators": args.n_estimators,
                  "contamination": args.contamination}
        mlflow.log_params(params)
        mlflow.log_params({"train_rows": len(train), "test_rows": len(test),
                           "features": ",".join(FEATURES)})

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("iforest", IsolationForest(n_estimators=args.n_estimators,
                                        contamination=args.contamination,
                                        random_state=42)),
        ])
        pipe.fit(train[FEATURES])

        # How often does it raise a false alarm on normal data?
        normal_pred = (pipe.predict(test[FEATURES]) == -1).astype(int)
        false_alarm_rate = normal_pred.mean()

        pred = (pipe.predict(test_inj[FEATURES]) == -1).astype(int)
        scores = -pipe.score_samples(test_inj[FEATURES])
        m = score_report(y_true, pred, scores)
        m["false_alarm_rate"] = float(false_alarm_rate)
        mlflow.log_metrics(m)
        print("Isolation Forest :", {k: round(v, 3) for k, v in m.items()})

        info = mlflow.sklearn.log_model(
            pipe, name="model", serialization_format="cloudpickle",
            input_example=train[FEATURES].head(5),
            registered_model_name=MODEL_NAME,
        )
        export_model(pipe, m, mlflow.active_run().info.run_id,
                     getattr(info, "registered_model_version", None))
    print(f"\nModel registered as '{MODEL_NAME}'. Open the UI with:\n"
          f"  mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main()
