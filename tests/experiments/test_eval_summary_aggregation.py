from pathlib import Path

from leopath.experiments.aggregate_eval import _read_csv_mean
from leopath.experiments.summarize_eval import summarize_run


def test_summarize_run_weights_stretch_by_samples(tmp_path: Path) -> None:
    run_dir = tmp_path / "predictive_link_state" / "ring"
    run_dir.mkdir(parents=True)

    (run_dir / "metadata.json").write_text(
        '{"algorithm": "predictive_link_state", "isl_scenario": "ring"}',
        encoding="utf-8",
    )
    (run_dir / "timestep_metrics.csv").write_text(
        "\n".join(
            [
                "stretch_dist_mean,stretch_dist_count,stretch_hop_mean,stretch_hop_count,fstate_size_mean,strict_header_bytes_mean",
                "1.0,1,1.0,1,100,24",
                "0.0,0,0.0,0,100,0",
                "1.0,3,1.0,3,100,36",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "delta_metrics.csv").write_text(
        "gs_handover_rate,gs_renumber_count,gs_renumber_rate,sat_gs_churn,sat_gs_break_rate,gs_gs_churn,gs_gs_break_rate\n",
        encoding="utf-8",
    )

    summary = summarize_run(run_dir)

    assert summary["stretch_dist_mean"] == 1.0
    assert summary["stretch_hop_mean"] == 1.0
    assert summary["stretch_dist_valid_timestep_mean"] == 1.0
    assert summary["stretch_hop_valid_timestep_mean"] == 1.0
    assert summary["stretch_dist_samples_mean"] == 4 / 3
    assert summary["strict_header_bytes_mean"] == 20.0


def test_aggregate_eval_ignores_zero_sample_stretch_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "timestep_metrics.csv"
    csv_path.write_text(
        "\n".join(
            [
                "stretch_dist_min,stretch_dist_max,stretch_dist_mean,stretch_dist_count,stretch_hop_min,stretch_hop_max,stretch_hop_mean,stretch_hop_count,compute_time_ms",
                "1.0,1.0,1.0,1,1.0,1.0,1.0,1,10",
                "0.0,0.0,0.0,0,0.0,0.0,0.0,0,20",
                "1.2,1.4,1.3,3,1.1,1.5,1.25,3,30",
            ]
        ),
        encoding="utf-8",
    )

    means = _read_csv_mean(str(csv_path))

    assert means["stretch_dist_mean"] == 1.225
    assert means["stretch_hop_mean"] == 1.1875
    assert means["stretch_dist_min"] == 1.1
    assert means["stretch_dist_max"] == 1.2
    assert means["stretch_hop_min"] == 1.05
    assert means["stretch_hop_max"] == 1.25
    assert means["compute_time_ms"] == 20.0


def test_summarize_run_backfills_renumbering_from_handover(tmp_path: Path) -> None:
    run_dir = tmp_path / "topological_routing" / "grid"
    run_dir.mkdir(parents=True)

    (run_dir / "metadata.json").write_text(
        '{"algorithm": "topological_routing", "isl_scenario": "grid", "ground_stations": {"count": 4}}',
        encoding="utf-8",
    )
    (run_dir / "timestep_metrics.csv").write_text(
        "fstate_size_mean\n4\n4\n",
        encoding="utf-8",
    )
    (run_dir / "delta_metrics.csv").write_text(
        "gs_handover_rate,sat_gs_churn,sat_gs_break_rate,sat_fstate_updates_total_mean,sat_fstate_updates_total_p95,sat_fstate_updates_touched_satellite_rate,gs_gs_churn,gs_gs_break_rate\n"
        "0.25,0,0,0.5,1.0,0.25,0,0\n"
        "0.50,0,0,1.5,2.0,0.50,0,0\n",
        encoding="utf-8",
    )

    summary = summarize_run(run_dir)

    assert summary["G"] == 4
    assert summary["handover_mean"] == 0.375
    assert summary["gs_renumber_rate_mean"] == 0.375
    assert summary["gs_renumber_count_mean"] == 1.5
    assert summary["sat_fstate_updates_total_mean"] == 1.0
    assert summary["sat_fstate_updates_total_p95_mean"] == 1.5
    assert summary["sat_fstate_updates_touched_satellite_rate_mean"] == 0.375
