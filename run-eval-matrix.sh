#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

OUTPUT_BASE=${1:-"$ROOT_DIR/paper_eval_outputs"}

ALGORITHMS=${ALGORITHMS:-"topological_routing shortest_path_link_state predictive_link_state segment_routing"}
ISL_SCENARIOS=${ISL_SCENARIOS:-"ring grid"}
CONFIGS=${CONFIGS:-"starlink kuiper oneweb telesat dense_synthetic"}

END_TIME_HOURS=${END_TIME_HOURS:-6}
TIME_STEP_MINUTES=${TIME_STEP_MINUTES:-5}
GS_CONFIG=${GS_CONFIG:-"$ROOT_DIR/leopath/config/ground_stations_dense.yaml"}

for config_name in $CONFIGS; do
  config_path="$ROOT_DIR/leopath/config/${config_name}.yaml"
  for algorithm in $ALGORITHMS; do
    for isl in $ISL_SCENARIOS; do
      out_dir="$OUTPUT_BASE/${config_name}/${algorithm}/${isl}"
      mkdir -p "$out_dir"
      python -m leopath.experiments.eval_harness \
        --config "$config_path" \
        --output-dir "$out_dir" \
        --isl-scenario "$isl" \
        --algorithm "$algorithm" \
        --gs-config "$GS_CONFIG" \
        --end-time-hours "$END_TIME_HOURS" \
        --time-step-minutes "$TIME_STEP_MINUTES"
    done
  done
done
