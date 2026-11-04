#!/usr/bin/env bash
# Usage: scripts/inject_failure.sh <cpu|mem|disk> [vm] [duration_seconds]
# Creates a real incident on a VM and records its start time in logs/injections.csv.
set -euo pipefail

SCENARIO="${1:?usage: $0 <cpu|mem|disk> [vm] [duration_s]}"
VM="${2:-web01}"
DURATION="${3:-120}"
LOG="logs/injections.csv"

mkdir -p logs
[ -f "$LOG" ] || echo "start_utc,scenario,vm,duration_s" > "$LOG"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${SCENARIO},${VM},${DURATION}" >> "$LOG"

case "$SCENARIO" in
  cpu)
    CMD="stress-ng --cpu 1 --cpu-load 95 --timeout ${DURATION}s" ;;
  mem)
    CMD="stress-ng --vm 1 --vm-bytes 70% --vm-keep --timeout ${DURATION}s" ;;
  disk)
    CMD="trap 'sudo rm -f /var/tmp/fill.img' EXIT; size=\$(df --output=size -B1 / | tail -1); sudo fallocate -l \$((size * 15 / 100)) /var/tmp/fill.img; sleep ${DURATION}" ;;
  *)
    echo "Unknown scenario: $SCENARIO (use cpu, mem or disk)" >&2; exit 1 ;;
esac

echo "Injecting '${SCENARIO}' on ${VM} for ${DURATION}s..."
vagrant ssh "$VM" -c "$CMD"
echo "Done. Detection results: python src/measure_detection.py"
