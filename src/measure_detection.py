"""Compare injected failures with what the monitor detected, and print a README-ready table."""
import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", default="logs/monitor.csv")
    p.add_argument("--injections", default="logs/injections.csv")
    p.add_argument("--grace", type=int, default=60, help="seconds after the end of a failure still counted")
    p.add_argument("--cooldown", type=int, default=180, help="seconds after a failure ignored for false alarms")
    a = p.parse_args()

    mon = pd.read_csv(a.monitor, parse_dates=["timestamp"])
    inj = pd.read_csv(a.injections, parse_dates=["start_utc"])
    mon["anomaly"] = mon["anomaly"].astype(str).str.lower().eq("true")

    rows, busy = [], pd.Series(False, index=mon.index)
    for _, i in inj.iterrows():
        start = i["start_utc"]
        end = start + pd.Timedelta(seconds=int(i["duration_s"]))
        on_vm = mon["server"] == i["vm"]
        window = mon[on_vm & (mon["timestamp"] >= start) & (mon["timestamp"] <= end + pd.Timedelta(seconds=a.grace))]
        hits = window[window["anomaly"]]
        detected = not hits.empty
        delay = int((hits["timestamp"].iloc[0] - start).total_seconds()) if detected else None
        status = "yes" if detected else ("NO DATA (monitor off?)" if window.empty else "NO")
        rows.append((i["scenario"], i["vm"], status,
                     f"{delay}" if detected else "-", f"{window['score'].max():.3f}" if len(window) else "-"))
        busy |= (mon["timestamp"] >= start) & (mon["timestamp"] <= end + pd.Timedelta(seconds=a.cooldown))

    print("| Scenario | Server | Detected | Detection time (s) | Max score |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")

    calm = mon[~busy]
    if len(calm):
        n, fa = len(calm), int(calm["anomaly"].sum())
        print(f"\nFalse alarms outside failure windows: {fa} / {n} checks ({100 * fa / n:.1f} %)")


if __name__ == "__main__":
    main()
