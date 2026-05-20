from leopath.experiments.eval_harness import prepare_algorithm_params


def test_traditional_sr_defaults_to_non_predictive_control() -> None:
    params = prepare_algorithm_params(
        simulation_config={
            "time_step_minutes": 5,
            "algorithm_params": {"prediction_horizon_minutes": 5, "segment_count": 2},
        },
        algorithm_name="traditional_segment_routing",
        prediction_horizon_minutes=None,
        segment_count=2,
        segment_refresh_interval_steps=2,
        segment_mode=None,
        plane_weight=None,
        sat_weight=None,
        shell_weight=None,
        time_step_minutes=None,
    )

    assert params["prediction_horizon_minutes"] == 0.0
    assert params["segment_refresh_interval_steps"] == 2
    assert params["time_step_minutes"] == 5


def test_predictive_override_is_preserved() -> None:
    params = prepare_algorithm_params(
        simulation_config={"time_step_minutes": 10, "algorithm_params": {}},
        algorithm_name="traditional_segment_routing",
        prediction_horizon_minutes=5,
        segment_count=2,
        segment_refresh_interval_steps=6,
        segment_mode=None,
        plane_weight=None,
        sat_weight=None,
        shell_weight=None,
        time_step_minutes=5,
    )

    assert params["prediction_horizon_minutes"] == 5
    assert params["segment_refresh_interval_steps"] == 6
    assert params["time_step_minutes"] == 5


def test_explicit_path_ignores_prediction_horizon_for_first_pass() -> None:
    params = prepare_algorithm_params(
        simulation_config={"time_step_minutes": 10, "algorithm_params": {}},
        algorithm_name="explicit_path_routing",
        prediction_horizon_minutes=7,
        segment_count=3,
        segment_refresh_interval_steps=None,
        segment_mode=None,
        plane_weight=None,
        sat_weight=None,
        shell_weight=None,
        time_step_minutes=5,
    )

    assert "prediction_horizon_minutes" not in params
    assert params["segment_count"] == 3
    assert params["segment_refresh_interval_steps"] == 1
    assert params["time_step_minutes"] == 5


def test_explicit_path_preserves_explicit_refresh_interval() -> None:
    params = prepare_algorithm_params(
        simulation_config={"time_step_minutes": 10, "algorithm_params": {}},
        algorithm_name="explicit_path_routing",
        prediction_horizon_minutes=None,
        segment_count=3,
        segment_refresh_interval_steps=6,
        segment_mode=None,
        plane_weight=None,
        sat_weight=None,
        shell_weight=None,
        time_step_minutes=5,
    )

    assert params["segment_refresh_interval_steps"] == 6


def test_explicit_path_drops_unused_weighting_metadata() -> None:
    params = prepare_algorithm_params(
        simulation_config={"time_step_minutes": 10, "algorithm_params": {}},
        algorithm_name="explicit_path_routing",
        prediction_horizon_minutes=None,
        segment_count=2,
        segment_refresh_interval_steps=4,
        segment_mode="plane_then_inplane",
        plane_weight=100.0,
        sat_weight=1.0,
        shell_weight=1000.0,
        time_step_minutes=5,
    )

    assert "segment_mode" not in params
    assert "plane_weight" not in params
    assert "sat_weight" not in params
    assert "shell_weight" not in params
