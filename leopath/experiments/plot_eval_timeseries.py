import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import math
import statistics


REPRESENTATIVE_CONSTELLATION = "Starlink-550"


ALGORITHM_COLORS = {
    "Link-state": "#4c72b0",
    "Topological": "#55a868",
    "Explicit-path": "#8172b3",
    "Predictive LS (h=5m)": "#c44e52",
    "Predictive LS (h=10m)": "#c44e52",
    "Predictive LS (h=0m)": "#c44e52",
}


def apply_poster_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "lines.linewidth": 3.0,
            "lines.markersize": 5.0,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation time series")
    parser.add_argument("--input-dir", required=True, help="Evaluation output directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
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
            parsed = {key: float(value) for key, value in row.items()}
            if "gs_renumber_rate" not in parsed and "gs_handover_rate" in parsed:
                parsed["gs_renumber_rate"] = parsed["gs_handover_rate"]
            rows.append(parsed)
        return rows


def read_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def time_minutes(rows: list[dict]) -> list[float]:
    return [row["time_since_epoch_ns"] / 1e9 / 60.0 for row in rows]


def format_label(metadata: dict) -> str:
    algorithm = metadata.get("algorithm", "unknown")
    params = metadata.get("algorithm_params") or {}
    if algorithm == "predictive_link_state":
        horizon = params.get("prediction_horizon_minutes")
        if horizon is not None:
            return f"Predictive LS (h={int(horizon)}m)"
    if algorithm == "shortest_path_link_state":
        return "Link-state"
    if algorithm == "topological_routing":
        return "Topological"
    if algorithm == "explicit_path_routing":
        return "Explicit-path"
    return algorithm.replace("_", " ")


def should_include_run(metadata: dict) -> bool:
    algorithm = metadata.get("algorithm", "")
    params = metadata.get("algorithm_params") or {}
    if algorithm == "shortest_path_link_state":
        return True
    if algorithm == "topological_routing":
        return True
    if algorithm == "explicit_path_routing":
        return True
    if algorithm == "traditional_segment_routing":
        return False
    return False


def collect_runs(input_dir: Path) -> dict:
    runs: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for meta_path in input_dir.rglob("metadata.json"):
        run_dir = meta_path.parent
        timestep_path = run_dir / "timestep_metrics.csv"
        delta_path = run_dir / "delta_metrics.csv"
        if not timestep_path.exists() or not delta_path.exists():
            continue
        metadata = read_metadata(meta_path)
        if not should_include_run(metadata):
            continue
        isl = metadata.get("isl_scenario", "unknown")
        label = format_label(metadata)
        constellation_name = metadata.get("constellation", {}).get("name", "")
        existing = runs.setdefault(isl, {}).get(label)
        if existing is not None:
            existing_constellation = existing.get("constellation_name", "")
            if existing_constellation == REPRESENTATIVE_CONSTELLATION:
                continue
            if constellation_name != REPRESENTATIVE_CONSTELLATION:
                continue
        runs.setdefault(isl, {})[label] = {
            "timestep": read_csv(timestep_path),
            "delta": read_csv(delta_path),
            "constellation_name": constellation_name,
        }
    return runs


def algorithm_color(label: str):
    return ALGORITHM_COLORS.get(label)


def stretch_series(rows: list[dict], metric_suffix: str, stat: str) -> list[float]:
    value_col = f"stretch_{metric_suffix}_{stat}"
    count_col = f"stretch_{metric_suffix}_count"
    series = []
    for row in rows:
        value = row[value_col]
        if row.get(count_col, 0.0) <= 0.0:
            series.append(math.nan)
        else:
            series.append(value)
    return series


def rolling_median(series: list[float], window: int) -> list[float]:
    if window <= 1:
        return series[:]
    radius = window // 2
    smoothed = []
    for index in range(len(series)):
        start = max(0, index - radius)
        end = min(len(series), index + radius + 1)
        values = [value for value in series[start:end] if not math.isnan(value)]
        smoothed.append(statistics.median(values) if values else math.nan)
    return smoothed


def plot_smoothed_series(
    ax,
    times: list[float],
    series: list[float],
    label: str,
    color: str,
    window: int,
    show_raw: bool = True,
) -> None:
    if show_raw:
        ax.plot(times, series, color=color, linewidth=1.0, alpha=0.12)
    ax.plot(
        times,
        rolling_median(series, window),
        label=label,
        color=color,
        linewidth=2.6,
    )


def plot_fstate_timeseries(output_dir: Path, runs: dict, isl: str, log_scale: bool) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
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
        (mean_line,) = ax.plot(times, mean, label=algo, color=algorithm_color(algo))
        ax.plot(
            times,
            p95,
            linestyle="--",
            color=mean_line.get_color(),
            linewidth=1.8,
            alpha=0.45,
        )
    ax.set_title(f"Forwarding Table Entries ({isl})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("FIB entries (proxy)")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"fstate_timeseries_{isl}.png")
    plt.close(fig)


def plot_churn_timeseries(output_dir: Path, runs: dict, isl: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9.2, 12.2), sharex=True)
    churn_specs = [
        ("gs_renumber_rate", "GS renumbering rate"),
        ("gs_handover_rate", "GS handover rate"),
        ("sat_gs_churn", "Sat→GS next-hop churn"),
        ("gs_gs_churn", "GS→GS next-hop churn"),
    ]
    for ax, (metric, title) in zip(axes, churn_specs):
        for algo, data in runs.items():
            rows = data["delta"]
            times = time_minutes(rows)
            series = [row[metric] for row in rows]
            ax.plot(times, series, label=algo, linewidth=3.0, color=algorithm_color(algo))
        ax.set_title(title)
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.18, linewidth=0.8)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(f"Routing Churn ({isl})", y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / f"churn_timeseries_{isl}.png")
    plt.close(fig)


def plot_churn_core_timeseries(output_dir: Path, runs: dict, isl: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
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
            ax.plot(times, series, label=algo, linewidth=3.0, color=algorithm_color(algo))
        ax.set_title(title)
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.18, linewidth=0.8)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(f"Routing Churn (core, {isl})", y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / f"churn_core_timeseries_{isl}.png")
    plt.close(fig)


def plot_sat_gs_churn_timeseries(output_dir: Path, runs: dict, isl: str) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    metric = "sat_gs_churn"
    for algo, data in runs.items():
        rows = data["delta"]
        if not rows:
            continue
        times = time_minutes(rows)
        series = [row[metric] for row in rows]
        plot_smoothed_series(ax, times, series, algo, algorithm_color(algo), window=9)
    ax.set_title(f"Sat→GS Next-hop Churn ({isl})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Rate")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.08)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_dir / f"churn_sat_gs_timeseries_{isl}.png")
    plt.close(fig)


def plot_stretch_timeseries(output_dir: Path, runs: dict, isl: str, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    metric_suffix = "dist" if metric == "distance" else metric
    for algo, data in runs.items():
        rows = data["timestep"]
        times = time_minutes(rows)
        mean = stretch_series(rows, metric_suffix, "mean")
        plot_smoothed_series(ax, times, mean, algo, algorithm_color(algo), window=11)
    ax.set_title(f"Path Stretch ({isl}, {metric})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Stretch")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.08)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_dir / f"stretch_{metric}_timeseries_{isl}.png")
    plt.close(fig)


def plot_compute_timeseries(output_dir: Path, runs: dict, isl: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    metric = "compute_time_ms"
    for algo, data in runs.items():
        rows = data["timestep"]
        if not rows or metric not in rows[0]:
            continue
        times = time_minutes(rows)
        series = [row[metric] for row in rows]
        plot_smoothed_series(
            ax,
            times,
            series,
            algo,
            algorithm_color(algo),
            window=11,
            show_raw=False,
        )
    ax.set_title(f"Compute Time per Step ({isl})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Milliseconds (log scale)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_dir / f"compute_timeseries_{isl}.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_poster_style()
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
        plot_compute_timeseries(output_dir, runs, isl)


if __name__ == "__main__":
    main()
