import csv
import json
import math
import statistics
from pathlib import Path

import pytest

from leopath.experiments import state_accounting_tables, summarize_failure_sweep
from leopath.experiments.run_pooling import pooled_delivery, weighted_mean

CAUSES = ("loop", "dead_end", "link_down", "hop_limit", "egress_lost")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(
    run_dir: Path,
    rows: list[dict],
    deltas: list[dict] | None = None,
    metadata: dict | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata or {}), encoding="utf-8")
    _write_csv(run_dir / "timestep_metrics.csv", rows)
    if deltas:
        _write_csv(run_dir / "delta_metrics.csv", deltas)


def _snapshot(deliverable: int, delivered: int, stretch: float = 1.0, **causes: int) -> dict:
    row = {
        "delivery_deliverable": deliverable,
        "delivery_delivered": delivered,
        "delivery_forwarding_failure": deliverable - delivered,
        "delivery_non_optimal_egress": 0,
        "stretch_dist_shared_mean": stretch,
        "stretch_dist_shared_count": delivered,
    }
    row.update({f"delivery_failure_{cause}": causes.get(cause, 0) for cause in CAUSES})
    return row


def _as_text(row: dict) -> dict[str, str]:
    return {key: str(value) for key, value in row.items()}


def test_pooled_delivery_weights_snapshots_by_their_pairs() -> None:
    rows = [_as_text(_snapshot(10, 9, stretch=1.1)), _as_text(_snapshot(10, 7, stretch=1.0))]
    pooled = pooled_delivery(rows)

    assert pooled["delivery_rate"] == pytest.approx(16 / 20)
    assert pooled["stretch_dist_shared"] == pytest.approx((1.1 * 9 + 1.0 * 7) / 16)
    assert pooled["deliverable_per_snapshot"] == 10.0


def test_weighted_mean_skips_snapshots_without_samples() -> None:
    rows = [{"v": "1.5", "w": "2"}, {"v": "", "w": ""}, {"v": "0.0", "w": "0"}]
    assert weighted_mean(rows, "v", "w") == 1.5
    assert weighted_mean([], "v", "w") is None


def test_failure_sweep_pairs_each_seed_with_link_state(tmp_path: Path) -> None:
    cell = tmp_path / "telesat" / "isl_p0.10"
    for seed in (1, 2):
        _run(cell / f"seed{seed}" / "link_state", [_snapshot(10, 10), _snapshot(10, 10)])
    _run(
        cell / "seed1" / "topological_nominal",
        [_snapshot(10, 9, loop=1), _snapshot(10, 7, loop=2, dead_end=1)],
    )
    _run(cell / "seed2" / "topological_nominal", [_snapshot(10, 10), _snapshot(10, 8, loop=2)])

    runs = summarize_failure_sweep.discover_runs(tmp_path)
    summarize_failure_sweep.add_baseline_gaps(runs)
    summary = summarize_failure_sweep.combine_seeds(runs)
    by_variant = {row["variant"]: row for row in summary}
    topological = by_variant["topological_nominal"]

    assert topological["seeds"] == 2
    assert topological["delivery_rate_mean"] == pytest.approx(0.85)
    assert topological["delivery_rate_ci95"] == pytest.approx(
        12.706 * statistics.stdev([0.8, 0.9]) / math.sqrt(2)
    )
    assert topological["delivery_gap_vs_link_state_mean"] == pytest.approx(0.15)
    assert topological["failure_share_loop_mean"] == pytest.approx((0.75 + 1.0) / 2)
    assert by_variant["link_state"]["delivery_gap_vs_link_state_mean"] == 0.0

    markdown = summarize_failure_sweep.render_markdown(summary)
    assert "## telesat" in markdown
    assert "loop 88%" in markdown


def test_matrix_tables_pool_runs_and_keep_per_node_maxima(tmp_path: Path) -> None:
    rows = []
    for cache_max in (80, 96):
        row = _snapshot(552, 552)
        row.update(
            {
                "fstate_size_mean": 4.2,
                "fstate_installed_mean": 24,
                "fstate_markers_mean": 0,
                "fstate_neighbors_mean": 4.2,
                "compute_time_ms": 100,
                "aux_cache_pairs_per_sat_max": cache_max,
                "aux_geometry_row_edge_entries": 1584,
                "aux_geometry_plane_edge_entries": 1584,
                "aux_geometry_build_ms": 700,
            }
        )
        rows.append(row)
    run_dir = tmp_path / "starlink" / "topological_routing" / "grid"
    _run(run_dir, rows, deltas=[{"sat_fstate_updates_total_mean": 0.0}])

    runs = state_accounting_tables.discover_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0]["aux_cache_pairs_per_sat_max"] == 96.0
    assert runs[0]["aux_geometry_entries"] == 3168.0
    assert runs[0]["delivery_rate"] == 1.0
    markdown = state_accounting_tables.render_markdown(runs)
    assert "## grid" in markdown
    assert "3,168" in markdown


def test_failure_sweep_reports_exception_state_and_corrects_old_minima(tmp_path: Path) -> None:
    guarded = {
        "algorithm_params": {"forwarding_guard": "progress"},
        "constellation": {"num_orbits": 2, "num_sats_per_orbit": 5},
    }

    def snapshot(**extra: float) -> dict:
        # Four ground stations (12 ordered pairs) and one dead satellite.
        return {
            **_snapshot(12, 12),
            "delivery_total_pairs": 12,
            "failure_satellites_down": 1,
            **extra,
        }

    cell = tmp_path / "telesat" / "sat_p0.05" / "seed1"
    # Before the counter fix: the dead satellite's four decisions are included.
    _run(
        cell / "topological_nominal_progress",
        [snapshot(aux_forwarding_exceptions=10)],
        metadata=guarded,
    )
    _run(
        cell / "topological_nominal_progress_exceptions",
        [
            snapshot(
                aux_forwarding_exceptions=6,
                aux_forwarding_exceptions_isolated=4,
                aux_exception_entries=2,
                aux_exception_entries_one_pass=5,
                aux_exception_satellites=2,
                aux_exception_hops_to_failure_mean=0.5,
                aux_exception_hops_to_failure_max=1,
                aux_exception_unresolved=0,
            )
        ],
        metadata=guarded,
    )

    runs = {run["variant"]: run for run in summarize_failure_sweep.discover_runs(tmp_path)}
    old, new = runs["topological_nominal_progress"], runs["topological_nominal_progress_exceptions"]

    assert old["live_minima_per_snapshot"] == pytest.approx(6.0)
    assert new["live_minima_per_snapshot"] == pytest.approx(6.0)
    assert old["exception_entries_per_snapshot"] is None
    assert new["exception_entries_per_snapshot"] == 2.0
    assert new["exception_share_of_link_state"] == pytest.approx(2 / (10 * 4))
    assert new["exception_hops_to_failure_mean"] == pytest.approx(0.5)

    summarize_failure_sweep.add_baseline_gaps(list(runs.values()))
    summary = summarize_failure_sweep.combine_seeds(list(runs.values()))
    assert "2.0 (5)" in summarize_failure_sweep.render_markdown(summary)
