"""Reduce a run's per-snapshot metrics to one value per metric.

Rates are pooled over snapshots instead of averaged per snapshot, so each
snapshot counts in proportion to the pairs it contributed, and stretch is
weighted by the number of pairs it was measured over.
"""

import csv
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows whose keys may differ, keeping first-seen column order."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


def column_sum(rows: list[dict[str, str]], key: str) -> float:
    return sum(_values(rows, key))


def column_mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = _values(rows, key)
    return sum(values) / len(values) if values else None


def column_max(rows: list[dict[str, str]], key: str) -> float | None:
    values = _values(rows, key)
    return max(values) if values else None


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def weighted_mean(rows: list[dict[str, str]], value_key: str, weight_key: str) -> float | None:
    total = 0.0
    weight = 0.0
    for row in rows:
        if row.get(value_key) in (None, "") or row.get(weight_key) in (None, ""):
            continue
        row_weight = float(row[weight_key])
        total += float(row[value_key]) * row_weight
        weight += row_weight
    return ratio(total, weight)


def pooled_delivery(rows: list[dict[str, str]]) -> dict[str, float | None]:
    """Delivery and stretch for one run, pooled over its snapshots."""
    deliverable = column_sum(rows, "delivery_deliverable")
    delivered = column_sum(rows, "delivery_delivered")
    return {
        "deliverable_per_snapshot": ratio(deliverable, len(rows)),
        "delivery_rate": ratio(delivered, deliverable),
        "non_optimal_egress_rate": ratio(
            column_sum(rows, "delivery_non_optimal_egress"), delivered
        ),
        "stretch_dist_shared": weighted_mean(
            rows, "stretch_dist_shared_mean", "stretch_dist_shared_count"
        ),
        "stretch_hop_shared": weighted_mean(
            rows, "stretch_hop_shared_mean", "stretch_hop_shared_count"
        ),
    }
