#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
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

OUTPUT_BASE=${1:-"$ROOT_DIR/paper_eval_outputs"}

ALGORITHMS=${ALGORITHMS:-"topological_routing shortest_path_link_state predictive_link_state"}
ISL_SCENARIOS=${ISL_SCENARIOS:-"ring grid"}
CONFIGS=${CONFIGS:-"starlink kuiper oneweb telesat dense_synthetic"}

END_TIME_HOURS=${END_TIME_HOURS:-6}
TIME_STEP_MINUTES=${TIME_STEP_MINUTES:-5}
GS_CONFIG=${GS_CONFIG:-"$ROOT_DIR/leopath/config/ground_stations_dense.yaml"}
GS_CONFIG=$(to_runtime_path "$GS_CONFIG")
PREDICTION_HORIZONS=${PREDICTION_HORIZONS:-"0 5 10"}
SEGMENT_COUNTS=${SEGMENT_COUNTS:-"2 3"}

run_eval_harness() {
  if [ "$EVAL_USE_DOCKER" = "1" ]; then
    docker compose run --rm --entrypoint "python" leo-routing-simu -m leopath.experiments.eval_harness "$@"
  else
    python -m leopath.experiments.eval_harness "$@"
  fi
}

for config_name in $CONFIGS; do
  config_path="$ROOT_DIR/leopath/config/${config_name}.yaml"
  config_path=$(to_runtime_path "$config_path")
  for algorithm in $ALGORITHMS; do
    for isl in $ISL_SCENARIOS; do
      if [ "$algorithm" = "predictive_link_state" ]; then
        for horizon in $PREDICTION_HORIZONS; do
          out_dir="$OUTPUT_BASE/${config_name}/${algorithm}/${isl}/horizon_${horizon}m"
          mkdir -p "$out_dir"
          runtime_out_dir=$(to_runtime_path "$out_dir")
          run_eval_harness \
            --config "$config_path" \
            --output-dir "$runtime_out_dir" \
            --isl-scenario "$isl" \
            --algorithm "$algorithm" \
            --gs-config "$GS_CONFIG" \
            --end-time-hours "$END_TIME_HOURS" \
            --time-step-minutes "$TIME_STEP_MINUTES" \
            --prediction-horizon-minutes "$horizon"
        done
      elif [ "$algorithm" = "traditional_segment_routing" ]; then
        for count in $SEGMENT_COUNTS; do
          out_dir="$OUTPUT_BASE/${config_name}/${algorithm}/${isl}/segments_${count}"
          mkdir -p "$out_dir"
          runtime_out_dir=$(to_runtime_path "$out_dir")
          run_eval_harness \
            --config "$config_path" \
            --output-dir "$runtime_out_dir" \
            --isl-scenario "$isl" \
            --algorithm "$algorithm" \
            --gs-config "$GS_CONFIG" \
            --end-time-hours "$END_TIME_HOURS" \
            --time-step-minutes "$TIME_STEP_MINUTES" \
            --segment-count "$count"
        done
      else
        out_dir="$OUTPUT_BASE/${config_name}/${algorithm}/${isl}"
        mkdir -p "$out_dir"
        runtime_out_dir=$(to_runtime_path "$out_dir")
        run_eval_harness \
          --config "$config_path" \
          --output-dir "$runtime_out_dir" \
          --isl-scenario "$isl" \
          --algorithm "$algorithm" \
          --gs-config "$GS_CONFIG" \
          --end-time-hours "$END_TIME_HOURS" \
          --time-step-minutes "$TIME_STEP_MINUTES"
      fi
    done
  done
done
