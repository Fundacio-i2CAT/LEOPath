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

## Output layout

Outputs are written under `paper_eval_outputs/` by default:

```
paper_eval_outputs/
  starlink/
    topological_routing/
      ring/
      grid/
```
