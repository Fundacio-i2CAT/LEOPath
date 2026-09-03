from leopath.experiments.eval_harness import prepare_algorithm_params


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
