import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregate evaluation summaries")
    parser.add_argument("--summary", required=True, help="Path to summary.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for output plots")
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.4,
        }
    )


def algorithm_family(algorithm: str) -> str:
    if algorithm == "shortest_path_link_state":
        return "Link-state"
    if algorithm == "topological_routing":
        return "Topological"
    if algorithm == "predictive_link_state":
        return "Predictive LS"
    if algorithm == "segment_routing":
        return "Segment routing"
    if algorithm == "traditional_segment_routing":
        return "Traditional SR"
    return algorithm


def read_summary(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["family"] = algorithm_family(row["algorithm"])
            for key in (
                "timestep_mean_fstate_size_mean",
                "delta_mean_sat_gs_churn",
                "timestep_mean_stretch_dist_mean",
                "timestep_mean_compute_time_ms",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def mean_by_family_and_isl(rows: list[dict], metric: str) -> dict[tuple[str, str], float]:
    agg = defaultdict(list)
    for row in rows:
        agg[(row["family"], row["isl_scenario"])].append(row[metric])
    return {k: sum(v) / len(v) for k, v in agg.items()}


def plot_algorithm_comparison(rows: list[dict], output_dir: Path) -> None:
    families = [
        "Link-state",
        "Topological",
        "Predictive LS",
        "Traditional SR",
        "Segment routing",
    ]
    isls = ["grid", "ring"]
    colors = {"grid": "#1f77b4", "ring": "#ff7f0e"}
    metrics = [
        ("timestep_mean_fstate_size_mean", "Forwarding State", True),
        ("delta_mean_sat_gs_churn", "Sat→GS Churn", False),
        ("timestep_mean_stretch_dist_mean", "Distance Stretch", False),
        ("timestep_mean_compute_time_ms", "Compute Time (ms)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    axes = axes.flatten()

    x = list(range(len(families)))
    width = 0.36

    for ax, (metric, title, use_log) in zip(axes, metrics):
        means = mean_by_family_and_isl(rows, metric)
        for i, isl in enumerate(isls):
            values = [means.get((family, isl), 0.0) for family in families]
            offset = -width / 2 if i == 0 else width / 2
            ax.bar(
                [v + offset for v in x],
                values,
                width=width,
                label=isl.upper(),
                color=colors[isl],
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=12)
        if use_log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(frameon=False)

    fig.suptitle("Algorithm Comparison Across ISL Scenarios", y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / "algorithm_comparison_summary.png")
    plt.close(fig)


def plot_fstate_by_constellation(rows: list[dict], output_dir: Path) -> None:
    filtered = [
        row
        for row in rows
        if row["family"] in {"Link-state", "Topological"}
        and row["isl_scenario"] in {"grid", "ring"}
    ]

    constellations = sorted({row["constellation_name"] for row in filtered})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    isls = ["grid", "ring"]
    colors = {"Link-state": "#4c72b0", "Topological": "#55a868"}

    for ax, isl in zip(axes, isls):
        group = [row for row in filtered if row["isl_scenario"] == isl]
        values = defaultdict(dict)
        for row in group:
            values[row["constellation_name"]][row["family"]] = row[
                "timestep_mean_fstate_size_mean"
            ]

        x = list(range(len(constellations)))
        width = 0.36
        ls_vals = [values[c].get("Link-state", 0.0) for c in constellations]
        tr_vals = [values[c].get("Topological", 0.0) for c in constellations]

        ax.bar([v - width / 2 for v in x], ls_vals, width, label="Link-state", color=colors["Link-state"])
        ax.bar([v + width / 2 for v in x], tr_vals, width, label="Topological", color=colors["Topological"])
        ax.set_title(f"Forwarding State by Constellation ({isl.upper()})")
        ax.set_xticks(x)
        ax.set_xticklabels(constellations, rotation=18, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(frameon=False)

    axes[0].set_ylabel("State units (log scale)")
    fig.tight_layout()
    fig.savefig(output_dir / "fstate_constellation_bars.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_style()
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_summary(summary_path)
    if not rows:
        raise SystemExit("No rows found in summary file")

    plot_algorithm_comparison(rows, output_dir)
    plot_fstate_by_constellation(rows, output_dir)


if __name__ == "__main__":
    main()
