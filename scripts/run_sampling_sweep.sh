#!/usr/bin/env bash
# Sampling-interval sensitivity sweep (#7): rerun the topological-forwarding pivot
# grid matrix at 10s and 5min sampling to compare against the 1-min baseline
# already present in eval-data-staging/topological-pivot-runs-6h/.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

OUT=${OUT:-paper_eval_outputs/sampling-sweep-6h}
PY=.venv/bin/python
GS=leopath/config/ground_stations_dense.yaml

run() {
  local const=$1 label=$2 tsm=$3
  local dir="$OUT/$const/$label"
  echo "=== $(date +%H:%M:%S) $const $label (tsm=$tsm) ==="
  $PY -m leopath.experiments.eval_harness \
    --config leopath/config/$const.yaml \
    --output-dir "$dir" \
    --isl-scenario grid \
    --algorithm topological_routing \
    --gs-config "$GS" \
    --end-time-hours 6.0 \
    --time-step-minutes "$tsm" \
    --distance-mode torus_weighted_pivot \
    --plane-weight 100.0 --sat-weight 1.0 --shell-weight 1000.0 \
    --segment-count 2 --segment-mode plane_then_inplane \
    --prediction-horizon-minutes 5 2>&1 | grep -v '%|' || true
}

# 5-min sweep first (fast, gives quick partial result), light->heavy
for c in telesat oneweb kuiper starlink; do run "$c" "interval_5min" 5.0; done
# 10-s sweep, light->heavy (heaviest = starlink last)
for c in telesat oneweb kuiper starlink; do run "$c" "interval_10s" 0.16667; done

echo "=== $(date +%H:%M:%S) SWEEP DONE ==="
