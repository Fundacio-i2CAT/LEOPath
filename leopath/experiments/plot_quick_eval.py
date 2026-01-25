import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quick evaluation outputs")
    parser.add_argument(
        "--input-dir",
        default="quick_eval_outputs",
        help="Directory containing evaluation outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="quick_eval_plots",
        help="Directory to write plots",
    )
    parser.add_argument(
        "--stretch-metric",
        choices=("distance", "hop"),
        default="distance",
        help="Stretch metric to plot",
    )
    parser.add_argument(
        "--log-fstate",
        action="store_true",
        help="Use log scale for forwarding state plot",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
        return rows


def time_minutes(rows: list[dict]) -> list[float]:
    return [row["time_since_epoch_ns"] / 1e9 / 60.0 for row in rows]


def collect_runs(input_dir: Path) -> dict:
    runs: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for algorithm_dir in input_dir.iterdir():
        if not algorithm_dir.is_dir():
            continue
        algo_name = algorithm_dir.name
        for isl_dir in algorithm_dir.iterdir():
            if not isl_dir.is_dir():
                continue
            isl_name = isl_dir.name
            timestep_path = isl_dir / "timestep_metrics.csv"
            delta_path = isl_dir / "delta_metrics.csv"
            if not timestep_path.exists() or not delta_path.exists():
                continue
            runs.setdefault(isl_name, {})[algo_name] = {
                "timestep": read_csv(timestep_path),
                "delta": read_csv(delta_path),
            }
    return runs


def plot_fstate_timeseries(output_dir: Path, runs: dict, isl: str, log_scale: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo, data in runs.items():
        rows = data["timestep"]
        if not rows:
            continue
        first_row = rows[0]
        if "fstate_size_mean" in first_row:
            mean_key = "fstate_size_mean"
            p95_key = "fstate_size_p95"
        else:
            mean_key = "fstate_sat_gs_mean"
            p95_key = "fstate_sat_gs_p95"
        times = time_minutes(rows)
        mean = [row[mean_key] for row in rows]
        p95 = [row[p95_key] for row in rows]
        ax.plot(times, mean, label=f"{algo} mean")
        ax.plot(times, p95, linestyle="--", label=f"{algo} p95")
    ax.set_title(f"Forwarding State Size ({isl})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("State units")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"fstate_timeseries_{isl}.png", dpi=150)
    plt.close(fig)


def plot_churn_timeseries(output_dir: Path, runs: dict, isl: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    churn_specs = [
        ("gs_handover_rate", "GS handover rate"),
        ("sat_gs_churn", "Sat→GS next-hop churn"),
        ("gs_gs_churn", "GS→GS next-hop churn"),
    ]
    for ax, (metric, title) in zip(axes, churn_specs):
        for algo, data in runs.items():
            rows = data["delta"]
            times = time_minutes(rows)
            series = [row[metric] for row in rows]
            ax.plot(times, series, label=algo)
        ax.set_title(title)
        ax.set_ylabel("Fraction")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(f"Routing Churn ({isl})", y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / f"churn_timeseries_{isl}.png", dpi=150)
    plt.close(fig)


def plot_churn_core_timeseries(output_dir: Path, runs: dict, isl: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
    churn_specs = [
        ("sat_gs_churn", "Sat→GS next-hop churn"),
        ("gs_gs_churn", "GS→GS next-hop churn"),
    ]
    for ax, (metric, title) in zip(axes, churn_specs):
        for algo, data in runs.items():
            rows = data["delta"]
            if not rows:
                continue
            times = time_minutes(rows)
            series = [row[metric] for row in rows]
            ax.plot(times, series, label=algo)
        ax.set_title(title)
        ax.set_ylabel("Fraction")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(f"Routing Churn (core, {isl})", y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / f"churn_core_timeseries_{isl}.png", dpi=150)
    plt.close(fig)


def plot_sat_gs_churn_timeseries(output_dir: Path, runs: dict, isl: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    metric = "sat_gs_churn"
    for algo, data in runs.items():
        rows = data["delta"]
        if not rows:
            continue
        times = time_minutes(rows)
        series = [row[metric] for row in rows]
        ax.plot(times, series, label=algo)
    ax.set_title(f"Sat→GS Next-hop Churn ({isl})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Fraction")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"churn_sat_gs_timeseries_{isl}.png", dpi=150)
    plt.close(fig)


def plot_stretch_timeseries(output_dir: Path, runs: dict, isl: str, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metric_suffix = "dist" if metric == "distance" else metric
    mean_col = f"stretch_{metric_suffix}_mean"
    p95_col = f"stretch_{metric_suffix}_p95"
    for algo, data in runs.items():
        rows = data["timestep"]
        times = time_minutes(rows)
        mean = [row[mean_col] for row in rows]
        p95 = [row[p95_col] for row in rows]
        ax.plot(times, mean, label=f"{algo} mean")
        ax.plot(times, p95, linestyle="--", label=f"{algo} p95")
    ax.set_title(f"Path Stretch ({isl}, {metric})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Stretch")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"stretch_{metric}_timeseries_{isl}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_by_isl = collect_runs(input_dir)
    if not runs_by_isl:
        raise SystemExit(f"No runs found in {input_dir}")

    for isl, runs in runs_by_isl.items():
        plot_fstate_timeseries(output_dir, runs, isl, args.log_fstate)
        plot_churn_timeseries(output_dir, runs, isl)
        plot_churn_core_timeseries(output_dir, runs, isl)
        plot_sat_gs_churn_timeseries(output_dir, runs, isl)
        plot_stretch_timeseries(output_dir, runs, isl, args.stretch_metric)


if __name__ == "__main__":
    main()
