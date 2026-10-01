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

`ISL_SCENARIO=brick_a` (or `brick_b`, `grid_seam`, `ring`) runs the whole sweep on another wiring, and `CONFIGS="shells/kuiper_610 shells/starlink_gen2_525"` on per-shell configs.

## Estimator cost per shell

```bash
python scripts/benchmark_pivot_estimators.py            # every config in leopath/config/shells
python scripts/benchmark_pivot_estimators.py kuiper_610 # one shell
```

For each shell it builds the pivot tables from derived ISL lengths, times a table lookup against a table-free `DerivedPivotEstimator` query, and checks the two agree on 200 random pairs. Timings are single-threaded CPython and only comparable with each other; the entries a satellite holds and the O(S) operations per query are what carry over. On the Gen2 shells the tables reach 497 280 entries and take over 20 s to build per snapshot on a busy server, against 7 constants for the derived estimator at about twice the lookup time.

## Real constellations

Two scripts compare a CelesTrak TLE snapshot with the grids the configs assume. Both read a TLE file and propagate every satellite to the same instant with SGP4.

```bash
curl -o starlink.tle "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
python scripts/celestrak_shell_geometry.py starlink.tle 43 480 --alt-tol 10
python scripts/celestrak_lattice_fit.py
```

`celestrak_shell_geometry.py` keeps one inclination and altitude window, groups planes by clustering node angles, and reports plane count, occupancy and how evenly planes and slots are spread. `celestrak_lattice_fit.py` asks the sharper question of whether satellites sit on a slot lattice with empty slots or scatter: per plane it fits the slot count and phase and measures the residual to the nearest slot. On the 29 September 2026 snapshot the residual was about 20 km, far below what scattered satellites would give, so the irregularity in flying shells comes mostly from empty slots. The fitted slot count is ambiguous up to multiples; the residual isn't.

## Ground-terminal population

The routing runs use a handful of ground stations. `scripts/terminal_population.py` asks what happens to addressing with many more terminals, without routing any packet: it places N terminals, attaches each to one satellite every snapshot, and reports how many terminals the busiest satellite carries, how many bits the endpoint index x then needs, how often terminals change address and the directory updates per second that implies.

```bash
python scripts/fetch_population_points.py data/population
python scripts/terminal_population.py leopath/config/starlink.yaml \
  --layout census uniform --population data/population/population_points.csv \
  --policy nearest stay_while_visible --speed-kmh 0 250 \
  --terminals 1000 10000 100000 1000000
```

`fetch_population_points.py` downloads census population for North America from the national statistics offices: US Census Bureau county estimates for 2024 placed at the 2024 Gazetteer county points, INEGI's 2020 census localities for Mexico, and Statistics Canada's 2021 dissemination blocks placed at their dissemination area's point. It writes one CSV of population points and records each source file's SHA-256 in `SOURCES.txt`; the totals match the three censuses exactly. `--layout census` draws terminals in proportion to that population, `uniform` spreads them evenly, the optimistic case for the busiest satellite. `nearest` moves a terminal to its nearest satellite every snapshot, `stay_while_visible` keeps its satellite until it sets. `--speed-kmh` moves every terminal on its own heading; at 250 km/h the address-change rate moves by at most about 1% either way, within run-to-run noise, since the satellites move a hundred times faster than any vehicle. At one-minute sampling the rate can't exceed one change per terminal-minute; Starlink's Gen2 525 km shell, whose satellites sit about 360 km apart along their orbits, reaches that ceiling under `nearest`, so its true rate may be higher.

## Notes

- Use the same time step across every algorithm in a matrix, otherwise churn numbers are not comparable: a tighter sampling interval mechanically raises the link-state update rate while leaving topological forwarding untouched.
- Keep ground stations fixed across runs for comparability.
- Run parallel harness processes in Docker or in separate working directories. Each run writes its TLE file to the current directory, and runs sharing one overwrite each other's.
