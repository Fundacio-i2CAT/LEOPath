import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["algorithm"] = df["algorithm"].fillna("unknown")
    df["isl_scenario"] = df["isl_scenario"].fillna("unknown")
    return df


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stretch(df: pd.DataFrame, output_dir: str) -> None:
    metrics = [
        "timestep_mean_stretch_hop_mean",
        "timestep_mean_stretch_dist_mean",
    ]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for isl in sorted(df["isl_scenario"].unique()):
            subset = df[df["isl_scenario"] == isl]
            ax.scatter(
                subset["algorithm"],
                subset[metric],
                label=isl,
                alpha=0.8,
            )
        ax.set_title(metric.replace("timestep_mean_", "").replace("_", " "))
        ax.set_ylabel("mean stretch")
        ax.grid(True, alpha=0.2)
        ax.legend()
        _save(fig, os.path.join(output_dir, f"{metric}.png"))


def plot_fstate(df: pd.DataFrame, output_dir: str) -> None:
    metric = "timestep_mean_fstate_size_mean"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for isl in sorted(df["isl_scenario"].unique()):
        subset = df[df["isl_scenario"] == isl]
        ax.scatter(
            subset["algorithm"],
            subset[metric],
            label=isl,
            alpha=0.8,
        )
    ax.set_title("forwarding state size mean")
    ax.set_ylabel("entries per satellite")
    ax.grid(True, alpha=0.2)
    ax.legend()
    _save(fig, os.path.join(output_dir, "fstate_size_mean.png"))


def plot_churn(df: pd.DataFrame, output_dir: str) -> None:
    metrics = [
        "delta_mean_sat_gs_churn",
        "delta_mean_gs_gs_churn",
    ]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for isl in sorted(df["isl_scenario"].unique()):
            subset = df[df["isl_scenario"] == isl]
            ax.scatter(
                subset["algorithm"],
                subset[metric],
                label=isl,
                alpha=0.8,
            )
        ax.set_title(metric.replace("delta_mean_", "").replace("_", " "))
        ax.set_ylabel("mean churn")
        ax.grid(True, alpha=0.2)
        ax.legend()
        _save(fig, os.path.join(output_dir, f"{metric}.png"))


def plot_compute_time(df: pd.DataFrame, output_dir: str) -> None:
    metric = "timestep_mean_compute_time_ms"
    if metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for isl in sorted(df["isl_scenario"].unique()):
        subset = df[df["isl_scenario"] == isl]
        ax.scatter(
            subset["algorithm"],
            subset[metric],
            label=isl,
            alpha=0.8,
        )
    ax.set_title("compute time per step (ms)")
    ax.set_ylabel("milliseconds")
    ax.grid(True, alpha=0.2)
    ax.legend()
    _save(fig, os.path.join(output_dir, "compute_time_ms.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregate evaluation metrics")
    parser.add_argument("--input", required=True, help="Aggregate CSV path")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load(args.input)
    plot_stretch(df, args.output_dir)
    plot_fstate(df, args.output_dir)
    plot_churn(df, args.output_dir)
    plot_compute_time(df, args.output_dir)


if __name__ == "__main__":
    main()
