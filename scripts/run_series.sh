#!/usr/bin/env bash
# Runs the remaining failure tests one after the other, with a calm pause between them.
# Usage: scripts/run_series.sh            (default pause: 240 s)
#        PAUSE=300 scripts/run_series.sh
# The monitor (python src/monitor.py) must already be running in another terminal.
set -euo pipefail

PAUSE="${PAUSE:-240}"

run() {
  echo
  echo "=== $(date '+%H:%M:%S')  scripts/inject_failure.sh $*"
  scripts/inject_failure.sh "$@"
  echo "Calm pause: ${PAUSE}s (do not touch anything)..."
  sleep "$PAUSE"
}

run mem  web01 120
run disk web01 120
run disk web01 120
run cpu  web01 120
run cpu  db01  120

echo
echo "Series finished. Now run: python src/measure_detection.py"
