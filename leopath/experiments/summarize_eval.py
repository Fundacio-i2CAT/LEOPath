import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation outputs")
    parser.add_argument("--input-dir", required=True, help="Evaluation output directory")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def fget(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def summarize_run(run_dir: Path) -> dict:
    meta_path = run_dir / "metadata.json"
    timestep_path = run_dir / "timestep_metrics.csv"
    delta_path = run_dir / "delta_metrics.csv"

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    trows = read_csv(timestep_path) if timestep_path.exists() else []
    drows = read_csv(delta_path) if delta_path.exists() else []

    if trows and "fstate_size_mean" in trows[0]:
        fstate_mean = mean([fget(r, "fstate_size_mean") for r in trows])
    else:
        fstate_mean = mean([fget(r, "fstate_sat_gs_mean") for r in trows])

    summary = {
        "algorithm": meta.get("algorithm", run_dir.parts[-2]),
        "isl": meta.get("isl_scenario", run_dir.parts[-1]),
        "G": int(meta.get("ground_station_count", 0) or 0),
        "dt_min": float(meta.get("time_step_minutes", 0.0) or 0.0),
        "hours": float(meta.get("end_time_hours", 0.0) or 0.0),
        "snapshots": len(trows),
        "deltas": len(drows),
        "fstate_units_mean": fstate_mean,
        "handover_mean": mean([fget(r, "gs_handover_rate") for r in drows]),
        "sat_gs_churn_mean": mean([fget(r, "sat_gs_churn") for r in drows]),
        "sat_gs_break_mean": mean([fget(r, "sat_gs_break_rate") for r in drows]),
        "gs_gs_churn_mean": mean([fget(r, "gs_gs_churn") for r in drows]),
        "gs_gs_break_mean": mean([fget(r, "gs_gs_break_rate") for r in drows]),
        "stretch_dist_mean": mean([fget(r, "stretch_dist_mean") for r in trows]),
        "stretch_hop_mean": mean([fget(r, "stretch_hop_mean") for r in trows]),
        "stretch_dist_samples_mean": mean([fget(r, "stretch_dist_count") for r in trows]),
    }
    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    runs = []
    for algo_dir in sorted(input_dir.iterdir()):
        if not algo_dir.is_dir():
            continue
        for isl_dir in sorted(algo_dir.iterdir()):
            if not isl_dir.is_dir():
                continue
            if not (isl_dir / "metadata.json").exists():
                continue
            runs.append(summarize_run(isl_dir))

    if not runs:
        raise SystemExit(f"No runs found under {input_dir}")

    columns = [
        "algorithm",
        "isl",
        "G",
        "dt_min",
        "hours",
        "snapshots",
        "fstate_units_mean",
        "handover_mean",
        "sat_gs_churn_mean",
        "sat_gs_break_mean",
        "gs_gs_churn_mean",
        "gs_gs_break_mean",
        "stretch_dist_mean",
        "stretch_hop_mean",
        "stretch_dist_samples_mean",
    ]

    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for row in runs:
        out = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                out.append(f"{value:.3f}")
            else:
                out.append(str(value))
        print("| " + " | ".join(out) + " |")


if __name__ == "__main__":
    main()
