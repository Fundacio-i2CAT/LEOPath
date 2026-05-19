# Experiments

## Evaluation matrix

Run the full cross-product of constellations, algorithms, and ISL scenarios.

## Run the evaluation matrix

```bash
./run-eval-matrix.sh
```

If local Python dependencies are not available, run via Docker:

```bash
EVAL_USE_DOCKER=1 ./run-eval-matrix.sh
```

Set variables to scope the runs:

```bash
ALGORITHMS="topological_routing shortest_path_link_state" \
CONFIGS="starlink kuiper" \
ISL_SCENARIOS="ring grid" \
./run-eval-matrix.sh
```

## Recommended protocol

- **Duration**: 6–24 hours simulated time per run.
- **Time step**: 5–10 minutes.
- **ISL scenarios**: `ring` and `grid` for each constellation.
- **Algorithms**: topological, link-state baseline, predictive link-state.

## Suggested parameter sweep

- **Predictive link-state**: `prediction_horizon_minutes` in {0, 5, 10}.
- **Explicit-path routing**: `segment_count` in {2, 3}; optionally sweep refresh cadence for alternative controller update policies.

## Design notes

- Predictive link-state runs compute paths over a future topology snapshot, so keep the horizon within 1–2 time steps.
- `explicit_path_routing` should be presented as the paper's explicit-path family example implementation.

## Constellation set

- `starlink`
- `kuiper`
- `oneweb`
- `telesat`
- `dense_synthetic`

## Output layout

Outputs are written under `paper_eval_outputs/` by default:

```
paper_eval_outputs/
  starlink/
    topological_routing/
      ring/
      grid/
    predictive_link_state/
      ring/
        horizon_0m/
        horizon_5m/
```

## Aggregate results

Create a single CSV with mean metrics per run:

```bash
python -m leopath.experiments.aggregate_eval \
  --input paper_eval_outputs \
  --output paper_eval_outputs/aggregate_summary.csv
```

## Time-series plots (paper style)

Generate the same plot style as the Jan 28 runs:

```bash
python -m leopath.experiments.plot_eval_timeseries \
  --input-dir paper_eval_outputs/ether_simple \
  --output-dir paper_eval_plots/ether_simple \
  --stretch-metric distance
```

Includes compute time per step when available.

## Notes

- For predictive link-state, use the same time step as other algorithms for fair churn comparison.
- Keep ground stations fixed across runs for comparability.
