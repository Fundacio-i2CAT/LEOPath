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


def weighted_mean(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight


def fget(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def mean_with_positive_count(rows: list[dict], value_key: str, count_key: str) -> float:
    values = [fget(row, value_key) for row in rows if fget(row, count_key) > 0.0]
    return mean(values)


def weighted_mean_with_positive_count(rows: list[dict], value_key: str, count_key: str) -> float:
    values = [
        (fget(row, value_key), fget(row, count_key))
        for row in rows
        if fget(row, count_key) > 0.0
    ]
    return weighted_mean(values)


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

    ground_station_count = int(
        meta.get("ground_station_count", 0)
        or meta.get("ground_stations", {}).get("count", 0)
        or 0
    )
    handover_mean = mean([fget(r, "gs_handover_rate") for r in drows])
    gs_renumber_rate_mean = mean(
        [
            fget(r, "gs_renumber_rate")
            if r.get("gs_renumber_rate") not in (None, "")
            else fget(r, "gs_handover_rate")
            for r in drows
        ]
    )
    gs_renumber_count_mean = mean(
        [
            fget(r, "gs_renumber_count")
            if r.get("gs_renumber_count") not in (None, "")
            else fget(r, "gs_handover_rate") * ground_station_count
            for r in drows
        ]
    )

    summary = {
        "algorithm": meta.get("algorithm", run_dir.parts[-2]),
        "isl": meta.get("isl_scenario", run_dir.parts[-1]),
        "G": ground_station_count,
        "dt_min": float(meta.get("time_step_minutes", 0.0) or 0.0),
        "hours": float(meta.get("end_time_hours", 0.0) or 0.0),
        "snapshots": len(trows),
        "deltas": len(drows),
        "fstate_units_mean": fstate_mean,
        "strict_header_bytes_mean": mean([fget(r, "strict_header_bytes_mean") for r in trows]),
        "srv6_srh_bytes_mean": mean([fget(r, "srv6_srh_bytes_mean") for r in trows]),
        "explicit_delivered_rate_mean": mean([fget(r, "explicit_failover_delivered_rate") for r in trows]),
        "explicit_egress_not_visible_rate_mean": mean([fget(r, "explicit_failover_egress_not_visible_rate") for r in trows]),
        "handover_mean": handover_mean,
        "gs_renumber_count_mean": gs_renumber_count_mean,
        "gs_renumber_rate_mean": gs_renumber_rate_mean,
        "sat_gs_churn_mean": mean([fget(r, "sat_gs_churn") for r in drows]),
        "sat_gs_break_mean": mean([fget(r, "sat_gs_break_rate") for r in drows]),
        "sat_fstate_updates_total_mean": mean(
            [fget(r, "sat_fstate_updates_total_mean") for r in drows]
        ),
        "sat_fstate_updates_total_p95_mean": mean(
            [fget(r, "sat_fstate_updates_total_p95") for r in drows]
        ),
        "sat_fstate_updates_touched_satellite_rate_mean": mean(
            [fget(r, "sat_fstate_updates_touched_satellite_rate") for r in drows]
        ),
        "gs_gs_churn_mean": mean([fget(r, "gs_gs_churn") for r in drows]),
        "gs_gs_break_mean": mean([fget(r, "gs_gs_break_rate") for r in drows]),
        "stretch_dist_mean": weighted_mean_with_positive_count(
            trows, "stretch_dist_mean", "stretch_dist_count"
        ),
        "stretch_hop_mean": weighted_mean_with_positive_count(
            trows, "stretch_hop_mean", "stretch_hop_count"
        ),
        "stretch_dist_samples_mean": mean([fget(r, "stretch_dist_count") for r in trows]),
        "stretch_dist_valid_timestep_mean": mean_with_positive_count(
            trows, "stretch_dist_mean", "stretch_dist_count"
        ),
        "stretch_hop_valid_timestep_mean": mean_with_positive_count(
            trows, "stretch_hop_mean", "stretch_hop_count"
        ),
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
        "strict_header_bytes_mean",
        "srv6_srh_bytes_mean",
        "explicit_delivered_rate_mean",
        "explicit_egress_not_visible_rate_mean",
        "handover_mean",
        "gs_renumber_count_mean",
        "gs_renumber_rate_mean",
        "sat_gs_churn_mean",
        "sat_gs_break_mean",
        "sat_fstate_updates_total_mean",
        "sat_fstate_updates_total_p95_mean",
        "sat_fstate_updates_touched_satellite_rate_mean",
        "gs_gs_churn_mean",
        "gs_gs_break_mean",
        "stretch_dist_mean",
        "stretch_hop_mean",
        "stretch_dist_samples_mean",
        "stretch_dist_valid_timestep_mean",
        "stretch_hop_valid_timestep_mean",
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
