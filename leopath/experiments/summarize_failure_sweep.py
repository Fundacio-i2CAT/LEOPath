"""Summarise a failure-injection sweep into per-seed and per-condition tables.

Reads the layout written by ``scripts/run-failure-sweep.sh``,
``<input>/<constellation>/<condition>/seed<k>/<variant>/``. Each run is reduced
to one value per metric by pooling its snapshots, then seeds are combined into
a mean with a 95% confidence interval. Every variant faces the same failure
pattern for a given seed, so the delivery gap to link-state is taken per seed
before averaging, which gives a tighter interval than differencing the means.

    python -m leopath.experiments.summarize_failure_sweep --input SWEEP --output-dir OUT
"""

import argparse
import math
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .metrics import FORWARDING_FAILURE_CAUSES
from .run_pooling import column_mean, column_sum, pooled_delivery, ratio, read_rows, write_rows

CONSTELLATION_ORDER = ["telesat", "oneweb", "kuiper", "starlink"]
CONDITION_ORDER = [
    "none",
    "isl_p0.01",
    "isl_p0.02",
    "isl_p0.05",
    "isl_p0.10",
    "isl_p0.20",
    "sat_p0.005",
    "sat_p0.01",
    "sat_p0.02",
    "sat_p0.05",
    "void_b2",
    "void_b4",
    "void_b8",
    "cut",
    "polar_lat75",
    "polar_lat60",
]
VARIANT_ORDER = [
    "link_state",
    "explicit_r1",
    "explicit_r15",
    "dra",
    "topological_nominal",
    "topological_observed",
]
BASELINE_VARIANT = "link_state"
METRICS = (
    "deliverable_per_snapshot",
    "delivery_rate",
    "delivery_gap_vs_link_state",
    "stretch_dist_shared",
    "non_optimal_egress_rate",
    "isls_removed_per_snapshot",
    "satellites_down_per_snapshot",
    "fstate_updates_per_snapshot",
    "compute_time_ms",
    *(f"failure_share_{cause}" for cause in FORWARDING_FAILURE_CAUSES),
)
# Two-sided 95% Student t quantiles by degrees of freedom.
_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def summarize_run(run_dir: Path) -> dict[str, float | None]:
    """One pooled value per metric for a single sweep run."""
    rows = read_rows(run_dir / "timestep_metrics.csv")
    deltas = read_rows(run_dir / "delta_metrics.csv")
    failures = column_sum(rows, "delivery_forwarding_failure")
    summary = {
        **pooled_delivery(rows),
        "isls_removed_per_snapshot": column_mean(rows, "failure_isls_removed"),
        "satellites_down_per_snapshot": column_mean(rows, "failure_satellites_down"),
        "fstate_updates_per_snapshot": column_mean(deltas, "sat_fstate_updates_total_mean"),
        "compute_time_ms": column_mean(rows, "compute_time_ms"),
    }
    for cause in FORWARDING_FAILURE_CAUSES:
        summary[f"failure_share_{cause}"] = ratio(
            column_sum(rows, f"delivery_failure_{cause}"), failures
        )
    return summary


def discover_runs(input_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for metadata_path in sorted(input_dir.glob("*/*/seed*/*/metadata.json")):
        run_dir = metadata_path.parent
        constellation, condition, seed_dir, variant = run_dir.relative_to(input_dir).parts
        runs.append(
            {
                "constellation": constellation,
                "condition": condition,
                "seed": int(seed_dir.removeprefix("seed")),
                "variant": variant,
                **summarize_run(run_dir),
            }
        )
    return runs


def add_baseline_gaps(runs: list[dict[str, Any]]) -> None:
    """Delivery-rate gap to link-state, paired by constellation, condition and seed."""
    baseline = {
        (run["constellation"], run["condition"], run["seed"]): run["delivery_rate"]
        for run in runs
        if run["variant"] == BASELINE_VARIANT
    }
    for run in runs:
        reference = baseline.get((run["constellation"], run["condition"], run["seed"]))
        rate = run["delivery_rate"]
        run["delivery_gap_vs_link_state"] = (
            None if reference is None or rate is None else reference - rate
        )


def mean_ci95(values: list[float]) -> tuple[float | None, float | None]:
    """Mean and 95% confidence half-width; a single value has no interval."""
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    degrees = len(values) - 1
    t_quantile = _T_975[max(df for df in _T_975 if df <= degrees)]
    return mean, t_quantile * statistics.stdev(values) / math.sqrt(len(values))


def _rank(value: str, order: list[str]) -> tuple[int, str]:
    return (order.index(value), "") if value in order else (len(order), value)


def combine_seeds(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        cells[(run["constellation"], run["condition"], run["variant"])].append(run)

    def cell_order(cell: tuple[str, str, str]) -> tuple:
        return (
            _rank(cell[0], CONSTELLATION_ORDER),
            _rank(cell[1], CONDITION_ORDER),
            _rank(cell[2], VARIANT_ORDER),
        )

    summary = []
    for constellation, condition, variant in sorted(cells, key=cell_order):
        seeds = cells[(constellation, condition, variant)]
        row: dict[str, Any] = {
            "constellation": constellation,
            "condition": condition,
            "variant": variant,
            "seeds": len(seeds),
        }
        for metric in METRICS:
            mean, ci95 = mean_ci95([seed[metric] for seed in seeds if seed[metric] is not None])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci95
        summary.append(row)
    return summary


def _percent(mean: float | None, ci95: float | None, signed: bool = False) -> str:
    if mean is None:
        return "—"
    text = f"{100 * mean:+.1f}" if signed else f"{100 * mean:.1f}"
    return text if ci95 is None else f"{text} ± {100 * ci95:.1f}"


def _delivery_cell(row: dict[str, Any]) -> str:
    return _percent(row["delivery_rate_mean"], row["delivery_rate_ci95"])


def _gap_cell(row: dict[str, Any]) -> str:
    return _percent(
        row["delivery_gap_vs_link_state_mean"],
        row["delivery_gap_vs_link_state_ci95"],
        signed=True,
    )


def _stretch_cell(row: dict[str, Any]) -> str:
    value = row["stretch_dist_shared_mean"]
    return "—" if value is None else f"{value:.4f}"


def _cause_cell(row: dict[str, Any]) -> str:
    shares = {
        cause: row[f"failure_share_{cause}_mean"]
        for cause in FORWARDING_FAILURE_CAUSES
        if row[f"failure_share_{cause}_mean"]
    }
    if not shares:
        return "—"
    cause = max(shares, key=lambda key: shares[key])
    return f"{cause} {100 * shares[cause]:.0f}%"


TABLES: tuple[tuple[str, Callable[[dict[str, Any]], str]], ...] = (
    ("Delivery rate, % of deliverable pairs", _delivery_cell),
    ("Delivery gap to link-state, percentage points, paired by seed", _gap_cell),
    ("Distance stretch, shared basis", _stretch_cell),
    ("Dominant forwarding-failure cause, share of failures", _cause_cell),
)


def _table(
    title: str,
    conditions: list[str],
    variants: list[str],
    cells: dict[tuple[str, str], dict[str, Any]],
    render_cell: Callable[[dict[str, Any]], str],
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| condition | " + " | ".join(variants) + " |",
        "|---" * (len(variants) + 1) + "|",
    ]
    for condition in conditions:
        rendered = [
            render_cell(cells[(condition, variant)]) if (condition, variant) in cells else "—"
            for variant in variants
        ]
        lines.append(f"| {condition} | " + " | ".join(rendered) + " |")
    lines.append("")
    return lines


def render_markdown(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Failure sweep summary",
        "",
        "Each run is pooled over its snapshots, then seeds are averaged. `±` is the 95%",
        "confidence half-width across seeds; deterministic conditions ran a single seed.",
        "",
    ]
    constellations = sorted(
        {row["constellation"] for row in summary},
        key=lambda name: _rank(name, CONSTELLATION_ORDER),
    )
    for constellation in constellations:
        cells = {
            (row["condition"], row["variant"]): row
            for row in summary
            if row["constellation"] == constellation
        }
        conditions = sorted({c for c, _ in cells}, key=lambda c: _rank(c, CONDITION_ORDER))
        variants = sorted({v for _, v in cells}, key=lambda v: _rank(v, VARIANT_ORDER))
        lines += [f"## {constellation}", ""]
        for title, render_cell in TABLES:
            lines += _table(title, conditions, variants, cells, render_cell)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise a failure-injection sweep")
    parser.add_argument("--input", required=True, help="Sweep output directory")
    parser.add_argument("--output-dir", required=True, help="Where to write CSVs and tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(Path(args.input))
    if not runs:
        raise SystemExit(f"No sweep runs found under {args.input}")
    add_baseline_gaps(runs)
    summary = combine_seeds(runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "failure_sweep_seeds.csv", runs)
    write_rows(output_dir / "failure_sweep_summary.csv", summary)
    (output_dir / "failure_sweep_tables.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"{len(runs)} runs in {len(summary)} cells -> {output_dir}")


if __name__ == "__main__":
    main()
