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
```

## Notes

- For predictive link-state, use the same time step as other algorithms for fair churn comparison.
- Keep ground stations fixed across runs for comparability.
