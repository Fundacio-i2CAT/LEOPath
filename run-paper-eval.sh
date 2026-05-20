#!/bin/sh
set -eu

ROOT_DIR=$(dirname "$0")
CONFIG_PATH="$ROOT_DIR/leopath/config/starlink.yaml"
GS_CONFIG="$ROOT_DIR/leopath/config/ground_stations_dense.yaml"
OUTPUT_BASE=${1:-"$ROOT_DIR/paper_eval_outputs"}

ALGORITHMS=${ALGORITHMS:-"shortest_path_link_state topological_routing"}
ISL_SCENARIOS="ring grid"
END_TIME_HOURS=${END_TIME_HOURS:-3}
TIME_STEP_MINUTES=${TIME_STEP_MINUTES:-1}
EXPLICIT_PATH_REFRESH_INTERVAL_STEPS=${EXPLICIT_PATH_REFRESH_INTERVAL_STEPS:-1}

for algorithm in $ALGORITHMS; do
  for isl in $ISL_SCENARIOS; do
    output_dir="$OUTPUT_BASE/${algorithm}/${isl}"
    if [ "$algorithm" = "explicit_path_routing" ]; then
      python -m leopath.experiments.eval_harness \
        --config "$CONFIG_PATH" \
        --gs-config "$GS_CONFIG" \
        --output-dir "$output_dir" \
        --isl-scenario "$isl" \
        --algorithm "$algorithm" \
        --end-time-hours "$END_TIME_HOURS" \
        --time-step-minutes "$TIME_STEP_MINUTES" \
        --segment-refresh-interval-steps "$EXPLICIT_PATH_REFRESH_INTERVAL_STEPS"
    else
      python -m leopath.experiments.eval_harness \
        --config "$CONFIG_PATH" \
        --gs-config "$GS_CONFIG" \
        --output-dir "$output_dir" \
        --isl-scenario "$isl" \
        --algorithm "$algorithm" \
        --end-time-hours "$END_TIME_HOURS" \
        --time-step-minutes "$TIME_STEP_MINUTES"
    fi
  done
done
