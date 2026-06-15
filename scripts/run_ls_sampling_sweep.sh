#!/usr/bin/env bash
# Link-state arm of the sampling-interval sensitivity sweep (#7). The 1-min grid
# baseline already lives in ntn-paper-eval-data/*/shortest_path_link_state/grid/,
# so here we add 10s and 5min. Starlink-LS-10s is intentionally skipped (global
# shortest-path over 1584 sats x 2160 steps is prohibitive); the trend across the
# three lighter constellations plus all-four at 5min/1min carries the argument.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

OUT=${OUT:-paper_eval_outputs/sampling-sweep-6h}
PY=.venv/bin/python
GS=leopath/config/ground_stations_dense.yaml

run() {
  local const=$1 label=$2 tsm=$3
  local dir="$OUT/$const/link_state_$label"
  echo "=== $(date +%H:%M:%S) LS $const $label (tsm=$tsm) ==="
  $PY -m leopath.experiments.eval_harness \
    --config leopath/config/$const.yaml \
    --output-dir "$dir" \
    --isl-scenario grid \
    --algorithm shortest_path_link_state \
    --gs-config "$GS" \
    --end-time-hours 6.0 \
    --time-step-minutes "$tsm" 2>&1 | grep -v '%|' || true
}

# 5-min sweep, all four, light->heavy
for c in telesat oneweb kuiper starlink; do run "$c" "interval_5min" 5.0; done
# 10-s sweep, three lighter constellations only (skip starlink)
for c in telesat oneweb kuiper; do run "$c" "interval_10s" 0.16667; done

echo "=== $(date +%H:%M:%S) LS SWEEP DONE ==="
