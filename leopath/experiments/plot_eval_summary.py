import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ALGORITHM_COLORS = {
    "Link-state": "#4c72b0",
    "Topological": "#55a868",
    "Explicit-path": "#8172b3",
}

CONSTELLATION_LABELS = {
    "Dense-LEO-synthetic": "Dense LEO",
    "Kuiper-synthetic": "Kuiper",
    "OneWeb-synthetic": "OneWeb",
    "Telesat-synthetic": "Telesat",
    "Starlink-550": "Starlink",
}


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
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2.4,
        }
    )


def algorithm_family(algorithm: str) -> str:
    if algorithm == "shortest_path_link_state":
        return "Link-state"
    if algorithm == "topological_routing":
        return "Topological"
    if algorithm == "explicit_path_routing":
        return "Explicit-path"
    return algorithm


def constellation_label(name: str) -> str:
    return CONSTELLATION_LABELS.get(name, name.replace("-synthetic", ""))


def read_summary(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            algorithm = row["algorithm"]
            if algorithm not in {
                "shortest_path_link_state",
                "topological_routing",
                "explicit_path_routing",
            }:
                continue

            row["family"] = algorithm_family(row["algorithm"])
            numeric_keys = (
                "timestep_mean_fstate_size_mean",
                "delta_mean_sat_gs_churn",
                "timestep_mean_stretch_dist_mean",
                "timestep_mean_compute_time_ms",
            )
            if any(row.get(key, "") in {"", None} for key in numeric_keys):
                continue
            for key in numeric_keys:
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
        "Explicit-path",
    ]
    isls = ["grid", "ring"]
    metrics = [
        ("timestep_mean_fstate_size_mean", "Forwarding Table Entries", True),
        ("delta_mean_sat_gs_churn", "Sat→GS Churn", "log"),
        ("timestep_mean_stretch_dist_mean", "Distance Stretch", False),
        ("timestep_mean_compute_time_ms", "Compute Time (ms)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8))
    axes = axes.flatten()

    x = list(range(len(families)))
    width = 0.36

    for ax, (metric, title, scale_mode) in zip(axes, metrics):
        means = mean_by_family_and_isl(rows, metric)
        for i, isl in enumerate(isls):
            offset = -width / 2 if i == 0 else width / 2
            for j, family in enumerate(families):
                value = means.get((family, isl), 0.0)
                if scale_mode == "log":
                    value = max(value, 1e-4)
                label = family if i == 0 and j == 0 else None
                ax.bar(
                    [x[j] + offset],
                    [value],
                    width=width,
                    color=ALGORITHM_COLORS[family],
                    alpha=0.72 if isl == "grid" else 0.42,
                    hatch=None if isl == "grid" else "//",
                    edgecolor=ALGORITHM_COLORS[family],
                    label=label,
                )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=10)
        if scale_mode is True or scale_mode == "log":
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(frameon=False)

    fig.suptitle("Algorithm Comparison Across ISL Scenarios", y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / "algorithm_comparison_summary.png")
    plt.close(fig)


def plot_fstate_by_constellation(rows: list[dict], output_dir: Path) -> None:
    families = [
        "Link-state",
        "Topological",
        "Explicit-path",
    ]
    filtered = [row for row in rows if row["family"] in set(families) and row["isl_scenario"] in {"grid", "ring"}]

    constellations = sorted({row["constellation_name"] for row in filtered})
    constellation_labels = [constellation_label(name) for name in constellations]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), sharey=True)
    isls = ["grid", "ring"]
    for ax, isl in zip(axes, isls):
        group = [row for row in filtered if row["isl_scenario"] == isl]
        values = defaultdict(dict)
        for row in group:
            values[row["constellation_name"]][row["family"]] = row[
                "timestep_mean_fstate_size_mean"
            ]

        x = list(range(len(constellations)))
        width = 0.18
        offsets = [-1.0, 0.0, 1.0]
        for family, offset in zip(families, offsets):
            family_vals = [values[c].get(family, 0.0) for c in constellations]
            ax.bar(
                [v + offset * width for v in x],
                family_vals,
                width,
                label=family,
                color=ALGORITHM_COLORS[family],
            )
        ax.set_title(f"Forwarding Table Entries by Constellation ({isl.upper()})")
        ax.set_xticks(x)
        ax.set_xticklabels(constellation_labels, rotation=18, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(frameon=False, ncol=2)

    axes[0].set_ylabel("Forwarding-state units (log scale)")
    fig.tight_layout()
    fig.savefig(output_dir / "fstate_constellation_bars.png")
    fig.savefig(output_dir / "fstate_all_algorithms_constellations.png")
    fig.savefig(output_dir / "fstate_all_algorithms_constellations.pdf")
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
