"""Reduce a run's per-snapshot metrics to one value per metric.

Rates are pooled over snapshots instead of averaged per snapshot, so each
snapshot counts in proportion to the pairs it contributed, and stretch is
weighted by the number of pairs it was measured over.
"""

import math
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
    # A run that has no value for a metric writes it as empty or as nan; neither
    # is a measurement, so pooling skips both rather than spreading a nan.
    values = (float(row[key]) for row in rows if row.get(key) not in (None, ""))
    return [value for value in values if math.isfinite(value)]


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
        value, row_weight = float(row[value_key]), float(row[weight_key])
        if not (math.isfinite(value) and math.isfinite(row_weight)):
            continue
        total += value * row_weight
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
        # 1.0 when every snapshot pinned one address pair at flow allocation;
        # 0.0 for any-attachment walks, including runs that predate the column.
        "fixed_address_forwarding": ratio(column_sum(rows, "fixed_address_forwarding"), len(rows)),
        "switched_egress_rate": ratio(
            column_sum(rows, "delivery_switched_egress"),
            column_sum(rows, "delivery_source_selected_egress"),
        ),
        "stretch_dist_shared": weighted_mean(
            rows, "stretch_dist_shared_mean", "stretch_dist_shared_count"
        ),
        "stretch_hop_shared": weighted_mean(
            rows, "stretch_hop_shared_mean", "stretch_hop_shared_count"
        ),
        # shared = egress x forwarding. The egress factor prices the choice of
        # egress, the forwarding factor what the algorithm did once that egress
        # was settled.
        "stretch_dist_egress": weighted_mean(
            rows, "stretch_dist_egress_mean", "stretch_dist_egress_count"
        ),
        "stretch_hop_egress": weighted_mean(
            rows, "stretch_hop_egress_mean", "stretch_hop_egress_count"
        ),
        "stretch_dist_forwarding": weighted_mean(rows, "stretch_dist_mean", "stretch_dist_count"),
        # One-way propagation delay over delivered paths, and its gap to the
        # any-egress shortest path, in milliseconds.
        "delay_ms": weighted_mean(rows, "delay_ms_mean", "delay_ms_count"),
        "delay_extra_ms": weighted_mean(rows, "delay_extra_ms_mean", "delay_extra_ms_count"),
        "delay_extra_p95_ms": weighted_mean(rows, "delay_extra_ms_p95", "delay_extra_ms_count"),
        "stretch_hop_forwarding": weighted_mean(rows, "stretch_hop_mean", "stretch_hop_count"),
    }
