import argparse
import csv
import glob
import json
import os

STRETCH_COUNT_KEYS = {
    "stretch_dist_min": "stretch_dist_count",
    "stretch_dist_max": "stretch_dist_count",
    "stretch_dist_mean": "stretch_dist_count",
    "stretch_dist_median": "stretch_dist_count",
    "stretch_dist_p95": "stretch_dist_count",
    "stretch_hop_min": "stretch_hop_count",
    "stretch_hop_max": "stretch_hop_count",
    "stretch_hop_mean": "stretch_hop_count",
    "stretch_hop_median": "stretch_hop_count",
    "stretch_hop_p95": "stretch_hop_count",
}
WEIGHTED_MEAN_KEYS = {"stretch_dist_mean", "stretch_hop_mean"}


def _read_csv_mean(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    weighted_sums: dict[str, float] = {}
    weighted_counts: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if value is None or value == "":
                continue
            try:
                val = float(str(value))
            except ValueError:
                continue
            sums[key] = sums.get(key, 0.0) + val
            counts[key] = counts.get(key, 0) + 1

            weight_key = STRETCH_COUNT_KEYS.get(key)
            if weight_key is None:
                continue
            weight_raw = row.get(weight_key)
            if weight_raw is None or weight_raw == "":
                continue
            try:
                weight = float(weight_raw)
            except ValueError:
                continue
            if weight <= 0.0:
                continue
            if key in WEIGHTED_MEAN_KEYS:
                weighted_sums[key] = weighted_sums.get(key, 0.0) + (val * weight)
                weighted_counts[key] = weighted_counts.get(key, 0.0) + weight
                continue

            weighted_sums[key] = weighted_sums.get(key, 0.0) + val
            weighted_counts[key] = weighted_counts.get(key, 0.0) + 1.0

    means = {key: sums[key] / counts[key] for key in sums}
    for key, total_weight in weighted_counts.items():
        if total_weight > 0.0:
            means[key] = weighted_sums[key] / total_weight
    return means


def _read_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_runs(base_dir: str) -> list[dict]:
    metadata_paths = glob.glob(os.path.join(base_dir, "**", "metadata.json"), recursive=True)
    runs = []
    for meta_path in metadata_paths:
        run_dir = os.path.dirname(meta_path)
        metrics_path = os.path.join(run_dir, "timestep_metrics.csv")
        delta_path = os.path.join(run_dir, "delta_metrics.csv")
        runs.append(
            {
                "run_dir": run_dir,
                "metadata_path": meta_path,
                "timestep_metrics": metrics_path,
                "delta_metrics": delta_path,
            }
        )
    return runs


def aggregate(base_dir: str, output_csv: str) -> None:
    rows = []
    for run in _discover_runs(base_dir):
        metadata = _read_metadata(run["metadata_path"])
        timestep_mean = _read_csv_mean(run["timestep_metrics"])
        delta_mean = (
            _read_csv_mean(run["delta_metrics"]) if os.path.exists(run["delta_metrics"]) else {}
        )

        row = {
            "run_dir": run["run_dir"],
            "algorithm": metadata.get("algorithm"),
            "isl_scenario": metadata.get("isl_scenario"),
            "constellation_name": metadata.get("constellation", {}).get("name"),
            "num_orbits": metadata.get("constellation", {}).get("num_orbits"),
            "num_sats_per_orbit": metadata.get("constellation", {}).get("num_sats_per_orbit"),
            "time_step_minutes": metadata.get("time_step_minutes"),
            "end_time_hours": metadata.get("end_time_hours"),
        }

        algorithm_params = metadata.get("algorithm_params") or {}
        for key, value in algorithm_params.items():
            if key == "undirected_isls":
                continue
            row[f"param_{key}"] = value

        for key, value in timestep_mean.items():
            row[f"timestep_mean_{key}"] = value
        for key, value in delta_mean.items():
            row[f"delta_mean_{key}"] = value

        ground_station_count = float(metadata.get("ground_stations", {}).get("count", 0) or 0)
        if "delta_mean_gs_renumber_rate" not in row and "delta_mean_gs_handover_rate" in row:
            row["delta_mean_gs_renumber_rate"] = row["delta_mean_gs_handover_rate"]
        if "delta_mean_gs_renumber_count" not in row and "delta_mean_gs_handover_rate" in row:
            row["delta_mean_gs_renumber_count"] = (
                row["delta_mean_gs_handover_rate"] * ground_station_count
            )

        rows.append(row)

    if not rows:
        raise RuntimeError("No evaluation runs found")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation outputs")
    parser.add_argument("--input", required=True, help="Base eval output directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(args.input, args.output)


if __name__ == "__main__":
    main()
