from leopath.experiments.eval_harness import (
    has_counter_rotating_seam,
    prepare_algorithm_params,
    select_isls,
)
from leopath.topology.constellation import ConstellationData


def test_explicit_path_preserves_explicit_refresh_interval() -> None:
    params = prepare_algorithm_params(
        simulation_config={"time_step_minutes": 10, "algorithm_params": {}},
        algorithm_name="explicit_path_routing",
        segment_count=3,
        segment_refresh_interval_steps=6,
        plane_weight=None,
        sat_weight=None,
        shell_weight=None,
        distance_mode=None,
        explicit_final_egress_mode=None,
        time_step_minutes=5,
    )

    assert params["segment_refresh_interval_steps"] == 6


def test_explicit_path_drops_unused_weighting_metadata() -> None:
    params = prepare_algorithm_params(
        simulation_config={
            "time_step_minutes": 10,
            "algorithm_params": {
                "plane_weight": 50.0,
                "sat_weight": 2.0,
                "shell_weight": 500.0,
            },
        },
        algorithm_name="explicit_path_routing",
        segment_count=2,
        segment_refresh_interval_steps=4,
        plane_weight=100.0,
        sat_weight=1.0,
        shell_weight=1000.0,
        distance_mode="torus_weighted_lookahead",
        explicit_final_egress_mode=None,
        time_step_minutes=5,
    )

    assert "plane_weight" not in params
    assert "sat_weight" not in params
    assert "shell_weight" not in params


def test_topological_routing_preserves_distance_mode() -> None:
    params = prepare_algorithm_params(
        simulation_config={"time_step_minutes": 10, "algorithm_params": {}},
        algorithm_name="topological_routing",
        segment_count=None,
        segment_refresh_interval_steps=None,
        plane_weight=None,
        sat_weight=None,
        shell_weight=None,
        distance_mode="torus_weighted",
        explicit_final_egress_mode=None,
        time_step_minutes=5,
    )

    assert params["distance_mode"] == "torus_weighted"


def test_explicit_path_preserves_final_egress_mode() -> None:
    params = prepare_algorithm_params(
        simulation_config={"time_step_minutes": 10, "algorithm_params": {}},
        algorithm_name="explicit_path_routing",
        segment_count=None,
        segment_refresh_interval_steps=3,
        plane_weight=None,
        sat_weight=None,
        shell_weight=None,
        distance_mode=None,
        explicit_final_egress_mode="dynamic",
        time_step_minutes=5,
    )

    assert params["final_egress_mode"] == "dynamic"


def test_failure_related_params_reach_only_their_algorithm() -> None:
    common = {
        "simulation_config": {"time_step_minutes": 1, "algorithm_params": {}},
        "segment_count": None,
        "segment_refresh_interval_steps": None,
        "plane_weight": None,
        "sat_weight": None,
        "shell_weight": None,
        "distance_mode": None,
        "explicit_final_egress_mode": None,
        "time_step_minutes": 1,
        "geometry_source": "nominal",
        "explicit_backup_adjacencies": True,
    }
    topological = prepare_algorithm_params(algorithm_name="topological_routing", **common)
    explicit = prepare_algorithm_params(algorithm_name="explicit_path_routing", **common)

    assert topological["geometry_source"] == "nominal"
    assert "include_backup_adjacencies" not in topological
    assert explicit["include_backup_adjacencies"] is True
    assert "geometry_source" not in explicit


def test_forwarding_guard_reaches_only_topological_routing() -> None:
    common = {
        "simulation_config": {"time_step_minutes": 1, "algorithm_params": {}},
        "segment_count": None,
        "segment_refresh_interval_steps": None,
        "plane_weight": None,
        "sat_weight": None,
        "shell_weight": None,
        "distance_mode": None,
        "explicit_final_egress_mode": None,
        "time_step_minutes": 1,
        "forwarding_guard": "progress",
    }
    assert (
        prepare_algorithm_params(algorithm_name="topological_routing", **common)["forwarding_guard"]
        == "progress"
    )
    assert "forwarding_guard" not in prepare_algorithm_params(
        algorithm_name="dra_routing", **common
    )


def test_local_repair_reaches_only_topological_routing() -> None:
    common = {
        "simulation_config": {"time_step_minutes": 1, "algorithm_params": {}},
        "segment_count": None,
        "segment_refresh_interval_steps": None,
        "plane_weight": None,
        "sat_weight": None,
        "shell_weight": None,
        "distance_mode": None,
        "explicit_final_egress_mode": None,
        "time_step_minutes": 1,
        "local_repair": "square",
    }
    assert (
        prepare_algorithm_params(algorithm_name="topological_routing", **common)["local_repair"]
        == "square"
    )
    assert "local_repair" not in prepare_algorithm_params(algorithm_name="dra_routing", **common)


def test_exception_policy_reaches_only_topological_routing() -> None:
    common = {
        "simulation_config": {"time_step_minutes": 1, "algorithm_params": {}},
        "segment_count": None,
        "segment_refresh_interval_steps": None,
        "plane_weight": None,
        "sat_weight": None,
        "shell_weight": None,
        "distance_mode": None,
        "explicit_final_egress_mode": None,
        "time_step_minutes": 1,
        "exception_policy": "grow",
    }
    assert (
        prepare_algorithm_params(algorithm_name="topological_routing", **common)["exception_policy"]
        == "grow"
    )
    assert "exception_policy" not in prepare_algorithm_params(
        algorithm_name="dra_routing", **common
    )


def _shell(orbits: int, sats: int) -> ConstellationData:
    return ConstellationData(
        orbits=orbits,
        sats_per_orbit=sats,
        epoch="00001.00000000",
        max_gsl_length_m=1.0,
        max_isl_length_m=1.0,
        satellites=[],
    )


def test_grid_on_walker_delta_keeps_the_seam_wrap() -> None:
    shell = _shell(6, 5)
    grid = set(select_isls(shell, "grid", raan_spread_degree=360.0))
    cylinder = set(select_isls(shell, "grid_seam", raan_spread_degree=360.0))
    # The delta wrap is one extra link per slot between the last and first plane.
    assert cylinder < grid
    assert len(grid - cylinder) == 5
    assert all({a // 5, b // 5} == {0, 5} for a, b in grid - cylinder)


def test_grid_on_walker_star_is_the_cylinder() -> None:
    shell = _shell(6, 5)
    grid = set(select_isls(shell, "grid", raan_spread_degree=180.0))
    cylinder = set(select_isls(shell, "grid_seam", raan_spread_degree=180.0))
    assert grid == cylinder


def test_counter_rotating_seam_only_on_star_shells() -> None:
    assert has_counter_rotating_seam(180.0)
    assert not has_counter_rotating_seam(360.0)
