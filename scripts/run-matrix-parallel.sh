#!/usr/bin/env bash
# Run an evaluation matrix as parallel Docker jobs and record per-job wall-clock.
#
# Each job is a single-threaded Python process, so concurrency is bounded by
# JOBS rather than by cores. BLAS threading is pinned to 1 inside the container;
# without that, numpy oversubscribes and 24 jobs fight over the whole machine.
#
#   JOBS=24 ./scripts/run-matrix-parallel.sh /path/to/output
#
# Completed jobs are skipped on re-invocation, so an interrupted campaign can be
# resumed by running the same command again.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"

OUTPUT_BASE=${1:-"$ROOT_DIR/paper_eval_outputs"}
IMAGE=${IMAGE:-leopath:rev}
JOBS=${JOBS:-24}

CONFIGS=${CONFIGS:-"telesat oneweb kuiper starlink"}
ALGORITHMS=${ALGORITHMS:-"topological_routing dra_routing shortest_path_link_state explicit_path_routing"}
ISL_SCENARIOS=${ISL_SCENARIOS:-"ring grid"}

END_TIME_HOURS=${END_TIME_HOURS:-6}
TIME_STEP_MINUTES=${TIME_STEP_MINUTES:-1}
GS_CONFIG=${GS_CONFIG:-/app/leopath/config/ground_stations_dense.yaml}
TOPOLOGICAL_DISTANCE_MODE=${TOPOLOGICAL_DISTANCE_MODE:-torus_weighted_pivot}
EXPLICIT_PATH_REFRESH_INTERVAL_STEPS=${EXPLICIT_PATH_REFRESH_INTERVAL_STEPS:-1}
EXPLICIT_PATH_FINAL_EGRESS_MODE=${EXPLICIT_PATH_FINAL_EGRESS_MODE:-dynamic}

TIMING_CSV="$OUTPUT_BASE/job_timings.csv"
mkdir -p "$OUTPUT_BASE"
[ -f "$TIMING_CSV" ] || echo "config,algorithm,isl_scenario,seconds,exit_code,finished_at" > "$TIMING_CSV"

run_job() {
  local cfg=$1 alg=$2 isl=$3
  local out="$OUTPUT_BASE/${cfg}/${alg}/${isl}"

  if [ -f "$out/timestep_metrics.csv" ] && [ -f "$out/metadata.json" ]; then
    echo "[skip] $cfg/$alg/$isl already complete"
    return 0
  fi
  rm -rf "$out"; mkdir -p "$out"

  local extra=()
  case "$alg" in
    topological_routing)
      extra=(--distance-mode "$TOPOLOGICAL_DISTANCE_MODE") ;;
    explicit_path_routing)
      extra=(--segment-refresh-interval-steps "$EXPLICIT_PATH_REFRESH_INTERVAL_STEPS"
             --explicit-final-egress-mode "$EXPLICIT_PATH_FINAL_EGRESS_MODE") ;;
  esac

  local start
  start=$(date +%s)
  echo "[$(date +%H:%M:%S)] start $cfg/$alg/$isl"
  docker run --rm --entrypoint python \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    -v "$ROOT_DIR/leopath/config:/app/leopath/config:ro" \
    -v "$out:/app/output" \
    "$IMAGE" \
    -m leopath.experiments.eval_harness \
    --config "/app/leopath/config/${cfg}.yaml" \
    --output-dir /app/output \
    --isl-scenario "$isl" \
    --algorithm "$alg" \
    --gs-config "$GS_CONFIG" \
    --end-time-hours "$END_TIME_HOURS" \
    --time-step-minutes "$TIME_STEP_MINUTES" \
    "${extra[@]}" \
    > "$out/harness.log" 2>&1
  local rc=$? elapsed=$(( $(date +%s) - start ))

  echo "${cfg},${alg},${isl},${elapsed},${rc},$(date -Iseconds)" >> "$TIMING_CSV"
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] ok   $cfg/$alg/$isl (${elapsed}s)"
  else
    echo "[$(date +%H:%M:%S)] FAIL $cfg/$alg/$isl (rc=$rc, ${elapsed}s) - see $out/harness.log"
  fi
}

echo "=== $(date) matrix start: JOBS=$JOBS image=$IMAGE out=$OUTPUT_BASE ==="
echo "    ${END_TIME_HOURS}h at ${TIME_STEP_MINUTES}min steps"

# Heaviest constellations first so the long pole starts early rather than
# trailing behind a queue of small runs.
for cfg in $CONFIGS; do
  for alg in $ALGORITHMS; do
    for isl in $ISL_SCENARIOS; do
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
      run_job "$cfg" "$alg" "$isl" &
    done
  done
done
wait

echo "=== $(date) matrix done ==="
failed=$(awk -F, 'NR>1 && $5!=0' "$TIMING_CSV" | wc -l)
[ "$failed" -eq 0 ] && echo "all jobs exited 0" || echo "$failed job(s) failed; grep the timings CSV"
