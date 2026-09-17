#!/usr/bin/env bash
# Failure-injection sweep for the robustness evaluation.
#
# Every (constellation, condition, seed) cell runs each routing variant against
# the same failure pattern: the pattern depends on the seed and the failure
# parameters, never on the algorithm, so variants are compared paired. Jobs run
# as single-threaded Docker processes, as in run-matrix-parallel.sh, and
# completed jobs are skipped on re-invocation.
#
#   IMAGE=leopath:<tag> JOBS=40 ./scripts/run-failure-sweep.sh /path/to/output
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"

OUTPUT_BASE=${1:-"$ROOT_DIR/failure_sweep_outputs"}
IMAGE=${IMAGE:-leopath:rev}
JOBS=${JOBS:-40}

# Heaviest constellations first, so the longest jobs start early.
CONFIGS=${CONFIGS:-"starlink kuiper oneweb telesat"}
ISL_SCENARIO=${ISL_SCENARIO:-grid}
END_TIME_HOURS=${END_TIME_HOURS:-1}
TIME_STEP_MINUTES=${TIME_STEP_MINUTES:-1}
SEEDS=${SEEDS:-"1 2 3 4 5"}
GS_CONFIG=${GS_CONFIG:-/app/leopath/config/ground_stations_dense.yaml}
EXPLICIT_SLOW_REFRESH_STEPS=${EXPLICIT_SLOW_REFRESH_STEPS:-15}

# name|harness flags. Random conditions run once per seed.
RANDOM_CONDITIONS=(
  "isl_p0.01|--failure-type isl --failure-rate 0.01"
  "isl_p0.02|--failure-type isl --failure-rate 0.02"
  "isl_p0.05|--failure-type isl --failure-rate 0.05"
  "isl_p0.10|--failure-type isl --failure-rate 0.10"
  "isl_p0.20|--failure-type isl --failure-rate 0.20"
  "sat_p0.005|--failure-type satellite --failure-rate 0.005"
  "sat_p0.01|--failure-type satellite --failure-rate 0.01"
  "sat_p0.02|--failure-type satellite --failure-rate 0.02"
  "sat_p0.05|--failure-type satellite --failure-rate 0.05"
  "void_b2|--failure-type void --failure-void-size 2"
  "void_b4|--failure-type void --failure-void-size 4"
  "void_b8|--failure-type void --failure-void-size 8"
)
# Deterministic conditions run once, with the first seed.
FIXED_CONDITIONS=(
  "none|--failure-type none"
  "cut|--failure-type cut"
  "polar_lat75|--failure-type polar --failure-polar-latitude-deg 75"
  "polar_lat60|--failure-type polar --failure-polar-latitude-deg 60"
)
VARIANTS=(
  "link_state|--algorithm shortest_path_link_state"
  "explicit_r1|--algorithm explicit_path_routing --segment-refresh-interval-steps 1 --explicit-final-egress-mode dynamic --explicit-backup-adjacencies"
  "explicit_r${EXPLICIT_SLOW_REFRESH_STEPS}|--algorithm explicit_path_routing --segment-refresh-interval-steps ${EXPLICIT_SLOW_REFRESH_STEPS} --explicit-final-egress-mode dynamic --explicit-backup-adjacencies"
  "dra|--algorithm dra_routing"
  "topological_nominal|--algorithm topological_routing --distance-mode torus_weighted_pivot --geometry-source nominal"
  "topological_observed|--algorithm topological_routing --distance-mode torus_weighted_pivot --geometry-source observed"
  "topological_nominal_progress|--algorithm topological_routing --distance-mode torus_weighted_pivot --geometry-source nominal --forwarding-guard progress"
  "topological_observed_progress|--algorithm topological_routing --distance-mode torus_weighted_pivot --geometry-source observed --forwarding-guard progress"
  "topological_nominal_progress_repair|--algorithm topological_routing --distance-mode torus_weighted_pivot --geometry-source nominal --forwarding-guard progress --local-repair square"
  "topological_nominal_progress_exceptions|--algorithm topological_routing --distance-mode torus_weighted_pivot --geometry-source nominal --forwarding-guard progress --exception-policy grow"
  "topological_nominal_progress_repair_exceptions|--algorithm topological_routing --distance-mode torus_weighted_pivot --geometry-source nominal --forwarding-guard progress --local-repair square --exception-policy grow"
)

TIMING_CSV="$OUTPUT_BASE/job_timings.csv"
mkdir -p "$OUTPUT_BASE"
[ -f "$TIMING_CSV" ] || echo "config,condition,seed,variant,seconds,exit_code,finished_at" > "$TIMING_CSV"

run_job() {
  local cfg=$1 condition=$2 seed=$3 variant=$4 flags=$5
  local out="$OUTPUT_BASE/${cfg}/${condition}/seed${seed}/${variant}"

  if [ -f "$out/timestep_metrics.csv" ] && [ -f "$out/metadata.json" ]; then
    echo "[skip] $cfg/$condition/seed$seed/$variant already complete"
    return 0
  fi
  rm -rf "$out"; mkdir -p "$out"

  local args
  read -r -a args <<< "$flags"
  local start
  start=$(date +%s)
  echo "[$(date +%H:%M:%S)] start $cfg/$condition/seed$seed/$variant"
  docker run --rm --entrypoint python \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 -e LEOPATH_CODE_VERSION="$IMAGE" \
    -v "$ROOT_DIR/leopath/config:/app/leopath/config:ro" \
    -v "$out:/app/output" \
    "$IMAGE" \
    -m leopath.experiments.eval_harness \
    --config "/app/leopath/config/${cfg}.yaml" \
    --output-dir /app/output \
    --isl-scenario "$ISL_SCENARIO" \
    --gs-config "$GS_CONFIG" \
    --end-time-hours "$END_TIME_HOURS" \
    --time-step-minutes "$TIME_STEP_MINUTES" \
    --failure-seed "$seed" \
    "${args[@]}" \
    > "$out/harness.log" 2>&1
  local rc=$? elapsed=$(( $(date +%s) - start ))

  echo "${cfg},${condition},${seed},${variant},${elapsed},${rc},$(date -Iseconds)" >> "$TIMING_CSV"
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] ok   $cfg/$condition/seed$seed/$variant (${elapsed}s)"
  else
    echo "[$(date +%H:%M:%S)] FAIL $cfg/$condition/seed$seed/$variant (rc=$rc, ${elapsed}s) - see $out/harness.log"
  fi
}

# Queue every variant of one (constellation, condition, seed) cell.
launch_cell() {
  local cfg=$1 condition=$2 seed=$3 condition_flags=$4
  local variant_entry
  for variant_entry in "${VARIANTS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    run_job "$cfg" "$condition" "$seed" "${variant_entry%%|*}" \
      "$condition_flags ${variant_entry#*|}" &
  done
}

read -r -a seed_list <<< "$SEEDS"
echo "=== $(date) failure sweep start: JOBS=$JOBS image=$IMAGE out=$OUTPUT_BASE ==="
echo "    ${ISL_SCENARIO}, ${END_TIME_HOURS}h at ${TIME_STEP_MINUTES}min steps, seeds: $SEEDS"

for cfg in $CONFIGS; do
  for condition_entry in "${FIXED_CONDITIONS[@]}"; do
    launch_cell "$cfg" "${condition_entry%%|*}" "${seed_list[0]}" "${condition_entry#*|}"
  done
  for condition_entry in "${RANDOM_CONDITIONS[@]}"; do
    for seed in "${seed_list[@]}"; do
      launch_cell "$cfg" "${condition_entry%%|*}" "$seed" "${condition_entry#*|}"
    done
  done
done
wait

echo "=== $(date) failure sweep done ==="
failed=$(awk -F, 'NR>1 && $6!=0' "$TIMING_CSV" | wc -l)
[ "$failed" -eq 0 ] && echo "all jobs exited 0" || echo "$failed job(s) failed; grep the timings CSV"
