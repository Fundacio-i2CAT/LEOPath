# Experiments

## Evaluation matrix

Run the full cross-product of constellations, algorithms, and ISL scenarios.

## Run the evaluation matrix

```bash
./run-eval-matrix.sh
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
- **Algorithms**: topological, link-state baseline, predictive link-state, segment routing.

## Suggested parameter sweep

- **Predictive link-state**: `prediction_horizon_minutes` in {0, 5, 10}.
- **Segment routing**: `segment_count` in {2, 3} and `segment_mode` = `plane_then_inplane`.

## Design notes

- Predictive link-state runs compute paths over a future topology snapshot, so keep the horizon within 1–2 time steps.
- Segment routing results are most interpretable when plane counts are large and uniform.

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
    segment_routing/
      grid/
        segments_2/
        segments_3/
```

## Notes

- For predictive link-state, use the same time step as other algorithms for fair churn comparison.
- Keep ground stations fixed across runs for comparability.
