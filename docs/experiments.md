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
- **Algorithms**: topological, link-state baseline, DRA, explicit-path.

## Suggested parameter sweep

- **Topological routing**: `distance_mode` in {`torus_unit`, `torus_weighted_lookahead`, `torus_weighted_pivot`}. Sweeping this against a fixed `dra_routing` run separates the effect of the distance metric from everything else.
- **Explicit-path routing**: `segment_refresh_interval_steps` in {1, 3} to compare controller update policies, and `explicit_final_egress_mode` in {`strict`, `dynamic`} to compare drop-on-stale-egress against local repair.

## Design notes

- `dra_routing` overrides `distance_mode` internally, so passing one has no effect. Use `topological_routing` with `distance_mode: torus_unit` if you want the hop-only metric under a name you control.
- Present `explicit_path_routing` as the paper's family-level explicit-path example, with SRv6-like local protection rather than transit shortest-path fallback.

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
    dra_routing/
      ring/
      grid/
    explicit_path_routing/
      ring/
      grid/
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

## Failure sweep

`scripts/run-failure-sweep.sh` runs the robustness sweep as parallel Docker jobs on +Grid, one hour at one-minute steps, covering every condition described under failure injection in `evaluation.md`, with seeds 1 to 5 for the random ones:

```bash
IMAGE=leopath:<tag> JOBS=40 ./scripts/run-failure-sweep.sh /path/to/sweep
```

Runs land in `<sweep>/<constellation>/<condition>/seed<k>/<variant>/`. The runner skips any job that already has output, so adding a variant to the script and rerunning with a newer image runs only that variant, against the same failures; each run's `metadata.json` keeps the image tag as `code_version`. Don't start a second runner on a directory while the first is still going, though, because it would restart the jobs that haven't finished.

```bash
python -m leopath.experiments.summarize_failure_sweep --input /path/to/sweep --output-dir failure_summary
```

writes one row per run, a per-cell summary with 95% intervals across seeds, and Markdown tables per constellation.

## Notes

- Use the same time step across every algorithm in a matrix, otherwise churn numbers are not comparable: a tighter sampling interval mechanically raises the link-state update rate while leaving topological forwarding untouched.
- Keep ground stations fixed across runs for comparability.
- Run parallel harness processes in Docker or in separate working directories. Each run writes its TLE file to the current directory, and runs sharing one overwrite each other's.
