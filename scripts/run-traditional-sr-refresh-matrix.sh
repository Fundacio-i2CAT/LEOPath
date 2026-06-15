#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CONTAINER_ROOT="/app/output"

EVAL_USE_DOCKER=${EVAL_USE_DOCKER:-0}

to_runtime_path() {
  local path="$1"
  if [ "$EVAL_USE_DOCKER" = "1" ]; then
    if [[ "$path" == "$ROOT_DIR"* ]]; then
      echo "${path/$ROOT_DIR/$CONTAINER_ROOT}"
      return
    fi
  fi
  echo "$path"
}

run_eval_harness() {
  if [ "$EVAL_USE_DOCKER" = "1" ]; then
    docker compose run --rm --entrypoint "python" leo-routing-simu -m leopath.experiments.eval_harness "$@"
  else
    python -m leopath.experiments.eval_harness "$@"
  fi
}

OUTPUT_BASE=${1:-"$ROOT_DIR/paper_eval_outputs"}
CONFIGS=${CONFIGS:-"starlink kuiper oneweb telesat dense_synthetic"}
ISL_SCENARIOS=${ISL_SCENARIOS:-"ring grid"}
REFRESH_INTERVALS=${REFRESH_INTERVALS:-"2 6"}
PREDICTION_HORIZONS=${PREDICTION_HORIZONS:-"0 5"}
FORCE_RERUN=${FORCE_RERUN:-0}

END_TIME_HOURS=${END_TIME_HOURS:-6}
TIME_STEP_MINUTES=${TIME_STEP_MINUTES:-5}
GS_CONFIG=${GS_CONFIG:-"$ROOT_DIR/leopath/config/ground_stations_dense.yaml"}
SEGMENT_COUNT=${SEGMENT_COUNT:-2}

GS_CONFIG=$(to_runtime_path "$GS_CONFIG")

run_case() {
  local config_name="$1"
  local isl="$2"
  local refresh="$3"
  local horizon="$4"
  local suffix=""
  if [ "$horizon" != "0" ] && [ "$horizon" != "0.0" ]; then
    suffix="_h${horizon}"
  fi

  local out_dir="$OUTPUT_BASE/${config_name}/traditional_segment_routing/${isl}/segments_${SEGMENT_COUNT}_refresh_${refresh}${suffix}"
  if [ "$FORCE_RERUN" != "1" ] && [ -f "$out_dir/metadata.json" ] && [ -f "$out_dir/timestep_metrics.csv" ] && [ -f "$out_dir/delta_metrics.csv" ]; then
    printf 'SKIP %s %s refresh=%s horizon=%s\n' "$config_name" "$isl" "$refresh" "$horizon"
    return
  fi

  mkdir -p "$out_dir"
  local config_path="$ROOT_DIR/leopath/config/${config_name}.yaml"
  local runtime_config_path
  local runtime_out_dir
  runtime_config_path=$(to_runtime_path "$config_path")
  runtime_out_dir=$(to_runtime_path "$out_dir")

  printf 'RUN  %s %s refresh=%s horizon=%s\n' "$config_name" "$isl" "$refresh" "$horizon"
  if [ "$horizon" != "0" ] && [ "$horizon" != "0.0" ]; then
    run_eval_harness \
      --config "$runtime_config_path" \
      --output-dir "$runtime_out_dir" \
      --isl-scenario "$isl" \
      --algorithm traditional_segment_routing \
      --gs-config "$GS_CONFIG" \
      --end-time-hours "$END_TIME_HOURS" \
      --time-step-minutes "$TIME_STEP_MINUTES" \
      --segment-count "$SEGMENT_COUNT" \
      --segment-refresh-interval-steps "$refresh" \
      --prediction-horizon-minutes "$horizon"
  else
    run_eval_harness \
      --config "$runtime_config_path" \
      --output-dir "$runtime_out_dir" \
      --isl-scenario "$isl" \
      --algorithm traditional_segment_routing \
      --gs-config "$GS_CONFIG" \
      --end-time-hours "$END_TIME_HOURS" \
      --time-step-minutes "$TIME_STEP_MINUTES" \
      --segment-count "$SEGMENT_COUNT" \
      --segment-refresh-interval-steps "$refresh"
  fi
}

for config_name in $CONFIGS; do
  for isl in $ISL_SCENARIOS; do
    for refresh in $REFRESH_INTERVALS; do
      for horizon in $PREDICTION_HORIZONS; do
        run_case "$config_name" "$isl" "$refresh" "$horizon"
      done
    done
  done
done
