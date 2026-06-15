#!/usr/bin/env bash
# Regenerate the 1-min grid baselines fresh, under the current harness, so the
# sampling sweep (#7) compares like with like. The pre-existing 1-min exports
# (eval-data LS, staging pivot) predate the current update/renumber accounting
# and report zeros for those columns, which is not comparable to the fresh
# 10s/5min runs.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

OUT=${OUT:-paper_eval_outputs/sampling-sweep-6h}
PY=.venv/bin/python
GS=leopath/config/ground_stations_dense.yaml

run_top() {
  local const=$1
  echo "=== $(date +%H:%M:%S) TOP $const 1min ==="
  $PY -m leopath.experiments.eval_harness \
    --config leopath/config/$const.yaml --output-dir "$OUT/$const/interval_1min" \
    --isl-scenario grid --algorithm topological_routing --gs-config "$GS" \
    --end-time-hours 6.0 --time-step-minutes 1.0 \
    --distance-mode torus_weighted_pivot \
    --plane-weight 100.0 --sat-weight 1.0 --shell-weight 1000.0 \
    --segment-count 2 --segment-mode plane_then_inplane \
    --prediction-horizon-minutes 5 2>&1 | grep -v '%|' || true
}
run_ls() {
  local const=$1
  echo "=== $(date +%H:%M:%S) LS $const 1min ==="
  $PY -m leopath.experiments.eval_harness \
    --config leopath/config/$const.yaml --output-dir "$OUT/$const/link_state_interval_1min" \
    --isl-scenario grid --algorithm shortest_path_link_state --gs-config "$GS" \
    --end-time-hours 6.0 --time-step-minutes 1.0 2>&1 | grep -v '%|' || true
}

for c in telesat oneweb kuiper starlink; do run_top "$c"; done
for c in telesat oneweb kuiper starlink; do run_ls "$c"; done
echo "=== $(date +%H:%M:%S) 1MIN RERUN DONE ==="
