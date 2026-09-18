"""Tables for the evaluation matrix: delivery, state by category and cost.

Reads ``<input>/<constellation>/<algorithm>/<isl>/`` as written by
``scripts/run-matrix-parallel.sh``, pools each run over its snapshots, and
writes one row per run plus Markdown tables grouped by ISL scenario.

    python -m leopath.experiments.state_accounting_tables --input MATRIX --output-dir OUT
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .run_pooling import column_max, column_mean, pooled_delivery, read_rows, write_rows

CONSTELLATION_ORDER = ["telesat", "oneweb", "kuiper", "starlink"]
ALGORITHM_ORDER = [
    "topological_routing",
    "dra_routing",
    "explicit_path_routing",
    "shortest_path_link_state",
]
ISL_ORDER = ["ring", "grid", "grid_seam"]
# Per-category totals reported alongside their parts.
DERIVED_SUMS = {
    "aux_geometry_entries": ("aux_geometry_row_edge_entries", "aux_geometry_plane_edge_entries"),
    "aux_path_cost_entries": ("aux_path_cost_row_entries", "aux_path_cost_plane_entries"),
    "aux_lsdb_entries": ("aux_lsdb_node_entries", "aux_lsdb_link_entries"),
}

Formatter = Callable[[float | None], str]


def _number(digits: int) -> Formatter:
    def render(value: float | None) -> str:
        return "—" if value is None else f"{value:,.{digits}f}"

    return render


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


TOPOLOGICAL_COLUMNS: list[tuple[str, str, Formatter]] = [
    ("installed FIB", "fstate_installed", _number(1)),
    ("neighbour entries", "fstate_neighbors", _number(1)),
    ("per-node cache, max", "aux_cache_pairs_per_sat_max", _number(0)),
    ("distance evals / sat / snapshot", "aux_distance_evals_per_sat_mean", _number(0)),
    ("decisions / sat / snapshot", "aux_decisions_per_sat_mean", _number(0)),
    ("geometry entries (derivable)", "aux_geometry_entries", _number(0)),
    ("path-cost entries (recomputable)", "aux_path_cost_entries", _number(0)),
    ("simulator pivot cache", "aux_pivot_cache_entries", _number(0)),
    ("geometry build, ms", "aux_geometry_build_ms", _number(0)),
]
LINK_STATE_COLUMNS: list[tuple[str, str, Formatter]] = [
    ("installed FIB", "fstate_installed", _number(1)),
    ("unreachable markers", "fstate_markers", _number(1)),
    ("LSDB nodes", "aux_lsdb_node_entries", _number(0)),
    ("LSDB links", "aux_lsdb_link_entries", _number(0)),
    ("SPF tree / sat", "aux_spf_tree_entries_per_sat", _number(0)),
    ("simulator all-pairs entries", "aux_spf_all_pairs_entries", _number(0)),
    ("simulator all-pairs build, ms", "aux_spf_all_pairs_build_ms", _number(0)),
]
MATRIX_TABLES: list[tuple[str, str, Formatter]] = [
    ("Deliverable pairs per snapshot", "deliverable_per_snapshot", _number(1)),
    ("Delivery rate, % of deliverable pairs", "delivery_rate", _percent),
    ("Distance stretch, shared basis", "stretch_dist_shared", _number(5)),
    ("Non-optimal egress, % of delivered pairs", "non_optimal_egress_rate", _percent),
    ("Installed forwarding entries per satellite", "fstate_installed", _number(1)),
    ("Unreachable-destination markers per satellite", "fstate_markers", _number(1)),
    ("Analytical forwarding-state proxy per satellite", "fstate_proxy", _number(1)),
    (
        "Forwarding-state updates per satellite per snapshot",
        "fstate_updates_per_snapshot",
        _number(2),
    ),
    ("Compute time per snapshot, ms (simulator, relative only)", "compute_time_ms", _number(0)),
]


def summarize_run(run_dir: Path) -> dict[str, float | None]:
    rows = read_rows(run_dir / "timestep_metrics.csv")
    deltas = read_rows(run_dir / "delta_metrics.csv")
    summary = {
        **pooled_delivery(rows),
        "fstate_proxy": column_mean(rows, "fstate_size_mean"),
        "fstate_installed": column_mean(rows, "fstate_installed_mean"),
        "fstate_markers": column_mean(rows, "fstate_markers_mean"),
        "fstate_neighbors": column_mean(rows, "fstate_neighbors_mean"),
        "fstate_updates_per_snapshot": column_mean(deltas, "sat_fstate_updates_total_mean"),
        "compute_time_ms": column_mean(rows, "compute_time_ms"),
    }
    # Sizes and build times are averaged over snapshots; per-satellite maxima
    # are kept as the maximum over the run.
    aux_keys = [key for key in (rows[0] if rows else {}) if key.startswith("aux_")]
    for key in aux_keys:
        summary[key] = column_max(rows, key) if key.endswith("_max") else column_mean(rows, key)
    for derived, parts in DERIVED_SUMS.items():
        values = [value for value in (summary.get(part) for part in parts) if value is not None]
        if len(values) == len(parts):
            summary[derived] = sum(values)
    return summary


def discover_runs(input_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for metadata_path in sorted(input_dir.glob("*/*/*/metadata.json")):
        run_dir = metadata_path.parent
        constellation, algorithm, isl = run_dir.relative_to(input_dir).parts
        runs.append(
            {
                "constellation": constellation,
                "algorithm": algorithm,
                "isl": isl,
                **summarize_run(run_dir),
            }
        )
    return runs


def _ordered(values: set[str], order: list[str]) -> list[str]:
    return sorted(values, key=lambda v: (order.index(v), "") if v in order else (len(order), v))


def _matrix(
    index: dict[tuple[str, str, str], dict[str, Any]],
    constellations: list[str],
    algorithms: list[str],
    isl: str,
    title: str,
    key: str,
    render: Formatter,
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| constellation | " + " | ".join(algorithms) + " |",
        "|---" * (len(algorithms) + 1) + "|",
    ]
    for constellation in constellations:
        cells = [
            (
                render(index[(constellation, algorithm, isl)].get(key))
                if (constellation, algorithm, isl) in index
                else "—"
            )
            for algorithm in algorithms
        ]
        lines.append(f"| {constellation} | " + " | ".join(cells) + " |")
    return lines + [""]


def _categories(
    index: dict[tuple[str, str, str], dict[str, Any]],
    constellations: list[str],
    algorithm: str,
    isl: str,
    title: str,
    columns: list[tuple[str, str, Formatter]],
) -> list[str]:
    present = [c for c in constellations if (c, algorithm, isl) in index]
    if not present:
        return []
    lines = [
        f"### {title}",
        "",
        "| constellation | " + " | ".join(label for label, _, _ in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for constellation in present:
        run = index[(constellation, algorithm, isl)]
        cells = [render(run.get(key)) for _, key, render in columns]
        lines.append(f"| {constellation} | " + " | ".join(cells) + " |")
    return lines + [""]


def render_markdown(runs: list[dict[str, Any]]) -> str:
    index = {(run["constellation"], run["algorithm"], run["isl"]): run for run in runs}
    constellations = _ordered({run["constellation"] for run in runs}, CONSTELLATION_ORDER)
    algorithms = _ordered({run["algorithm"] for run in runs}, ALGORITHM_ORDER)
    lines = [
        "# Evaluation matrix summary",
        "",
        "Rates are pooled over each run's snapshots and stretch is weighted by the pairs it",
        "was measured over. Sizes are entries per satellite unless marked as simulator-wide.",
        "",
    ]
    for isl in _ordered({run["isl"] for run in runs}, ISL_ORDER):
        lines += [f"## {isl}", ""]
        for title, key, render in MATRIX_TABLES:
            lines += _matrix(index, constellations, algorithms, isl, title, key, render)
        lines += _categories(
            index,
            constellations,
            "topological_routing",
            isl,
            "Topological routing: state by category",
            TOPOLOGICAL_COLUMNS,
        )
        lines += _categories(
            index,
            constellations,
            "shortest_path_link_state",
            isl,
            "Link-state: database and shortest-path state",
            LINK_STATE_COLUMNS,
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tables for an evaluation matrix")
    parser.add_argument("--input", required=True, help="Matrix output directory")
    parser.add_argument("--output-dir", required=True, help="Where to write CSVs and tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(Path(args.input))
    if not runs:
        raise SystemExit(f"No runs found under {args.input}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "matrix_runs.csv", runs)
    (output_dir / "matrix_tables.md").write_text(render_markdown(runs), encoding="utf-8")
    print(f"{len(runs)} runs -> {output_dir}")


if __name__ == "__main__":
    main()
