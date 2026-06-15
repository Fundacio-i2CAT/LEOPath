#!/bin/sh
set -eu

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CONFIG_PATH="$ROOT_DIR/leopath/config/starlink.yaml"
GS_CONFIG="$ROOT_DIR/leopath/config/ground_stations_quick_eval.yaml"
OUTPUT_BASE=${1:-"$ROOT_DIR/quick_eval_outputs"}

ALGORITHMS=${ALGORITHMS:-"shortest_path_link_state topological_routing"}
ISL_SCENARIOS="ring grid"

for algorithm in $ALGORITHMS; do
  for isl in $ISL_SCENARIOS; do
    output_dir="$OUTPUT_BASE/${algorithm}/${isl}"
    python -m leopath.experiments.eval_harness \
      --config "$CONFIG_PATH" \
      --gs-config "$GS_CONFIG" \
      --output-dir "$output_dir" \
      --isl-scenario "$isl" \
      --algorithm "$algorithm" \
      --end-time-hours 3 \
      --time-step-minutes 10
  done
done
