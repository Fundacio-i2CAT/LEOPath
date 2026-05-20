import argparse
import datetime
import time
import logging
import os

import yaml
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None
from astropy import units as astro_units

from leopath import logger
from leopath.main import (
    calculate_link_params,
    generate_plus_grid_isls,
    setup_ground_stations,
    setup_isls_in_the_same_orbit,
    setup_tles_and_satellites,
)
from leopath.network_state.generate_network_state import _build_topologies
from leopath.network_state.gsl_attachment.gsl_attachment_strategies import *  # noqa: F403, F401
from leopath.network_state.helpers import (
    _compute_ground_station_satellites_in_range,
    _compute_isls,
)
from leopath.network_state.routing_algorithms.routing_algorithm_factory import (
    get_routing_algorithm,
)
from leopath.topology.topology import ConstellationData

from .metrics import (
    build_interface_neighbor_map,
    compute_forwarding_state_stats,
    compute_gs_handover_rate,
    compute_gs_renumbering_stats,
    compute_gs_to_gs_churn,
    compute_path_stretch,
    compute_sat_to_gs_churn,
    get_gs_attachments,
    write_csv,
    write_json,
)

log = logger.get_logger(__name__)


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_ground_station_override(path: str | None) -> list[dict] | None:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground station config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if isinstance(payload, dict) and "ground_stations" in payload:
        return payload["ground_stations"]
    if isinstance(payload, list):
        return payload
    raise ValueError(
        "Ground station override must be a list or contain ground_stations"
    )


def select_isls(
    constellation: ConstellationData, scenario: str
) -> list[tuple[int, int]]:
    if scenario == "ring":
        return setup_isls_in_the_same_orbit(
            num_orbits=constellation.n_orbits,
            sats_per_orbit=constellation.n_sats_per_orbit,
        )
    if scenario == "grid":
        return generate_plus_grid_isls(
            n_orbits=constellation.n_orbits,
            n_sats_per_orbit=constellation.n_sats_per_orbit,
            idx_offset=0,
        )
    raise ValueError(f"Unknown ISL scenario: {scenario}")


def flatten_distribution(prefix: str, stats: dict) -> dict:
    return {
        f"{prefix}_min": stats["min"],
        f"{prefix}_max": stats["max"],
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_median": stats["median"],
        f"{prefix}_p95": stats["p95"],
        f"{prefix}_count": stats["count"],
    }


def prepare_algorithm_params(
    simulation_config: dict,
    algorithm_name: str,
    prediction_horizon_minutes: float | None,
    segment_count: int | None,
    segment_refresh_interval_steps: int | None,
    segment_mode: str | None,
    plane_weight: float | None,
    sat_weight: float | None,
    shell_weight: float | None,
    time_step_minutes: float | None,
) -> dict:
    algorithm_params = dict(simulation_config.get("algorithm_params") or {})

    if algorithm_name == "traditional_segment_routing":
        algorithm_params["prediction_horizon_minutes"] = (
            0.0 if prediction_horizon_minutes is None else prediction_horizon_minutes
        )
    elif algorithm_name == "explicit_path_routing":
        algorithm_params.pop("prediction_horizon_minutes", None)
    elif prediction_horizon_minutes is not None:
        algorithm_params["prediction_horizon_minutes"] = prediction_horizon_minutes

    if segment_count is not None:
        algorithm_params["segment_count"] = segment_count
    if segment_refresh_interval_steps is not None:
        algorithm_params["segment_refresh_interval_steps"] = segment_refresh_interval_steps
    elif algorithm_name == "explicit_path_routing":
        algorithm_params.setdefault("segment_refresh_interval_steps", 1)
    if segment_mode is not None and algorithm_name != "explicit_path_routing":
        algorithm_params["segment_mode"] = segment_mode
    if plane_weight is not None and algorithm_name != "explicit_path_routing":
        algorithm_params["plane_weight"] = plane_weight
    if sat_weight is not None and algorithm_name != "explicit_path_routing":
        algorithm_params["sat_weight"] = sat_weight
    if shell_weight is not None and algorithm_name != "explicit_path_routing":
        algorithm_params["shell_weight"] = shell_weight

    effective_time_step_minutes = time_step_minutes
    if effective_time_step_minutes is None:
        effective_time_step_minutes = simulation_config["time_step_minutes"]
    algorithm_params["time_step_minutes"] = effective_time_step_minutes
    return algorithm_params


def run_evaluation(
    config_path: str,
    output_dir: str,
    isl_scenario: str,
    algorithm_name: str | None,
    gs_override_path: str | None,
    end_time_hours: float | None,
    time_step_minutes: float | None,
    prediction_horizon_minutes: float | None,
    segment_count: int | None,
    segment_refresh_interval_steps: int | None,
    segment_mode: str | None,
    plane_weight: float | None,
    sat_weight: float | None,
    shell_weight: float | None,
) -> None:
    config = load_config(config_path)
    gs_override = load_ground_station_override(gs_override_path)
    if gs_override is not None:
        config["ground_stations"] = gs_override
    if end_time_hours is not None:
        config["simulation"]["end_time_hours"] = end_time_hours
    if time_step_minutes is not None:
        config["simulation"]["time_step_minutes"] = time_step_minutes

    effective_algorithm_name = (
        algorithm_name or config["simulation"]["dynamic_state_algorithm"]
    )
    config["simulation"]["dynamic_state_algorithm"] = effective_algorithm_name
    algorithm_params = prepare_algorithm_params(
        simulation_config=config["simulation"],
        algorithm_name=effective_algorithm_name,
        prediction_horizon_minutes=prediction_horizon_minutes,
        segment_count=segment_count,
        segment_refresh_interval_steps=segment_refresh_interval_steps,
        segment_mode=segment_mode,
        plane_weight=plane_weight,
        sat_weight=sat_weight,
        shell_weight=shell_weight,
        time_step_minutes=time_step_minutes,
    )
    if algorithm_params:
        config["simulation"]["algorithm_params"] = algorithm_params

    os.makedirs(output_dir, exist_ok=True)
    logger.setup_logger(
        is_debug=False, file_name=os.path.join(output_dir, "eval_harness.log")
    )
    logging.getLogger(logger.APP_LOGGER_NAME).setLevel(logging.ERROR)

    parsed_tles_data, sim_satellites = setup_tles_and_satellites(config)
    ground_stations = setup_ground_stations(config)

    max_gsl, max_isl = calculate_link_params(config)
    constellation_data = ConstellationData(
        orbits=parsed_tles_data["n_orbits"],
        sats_per_orbit=parsed_tles_data["n_sats_per_orbit"],
        epoch=parsed_tles_data["epoch"],
        max_gsl_length_m=max_gsl,
        max_isl_length_m=max_isl,
        satellites=sim_satellites,
    )

    undirected_isls = select_isls(constellation_data, isl_scenario)
    if config["simulation"]["dynamic_state_algorithm"] in {
        "predictive_link_state",
        "traditional_segment_routing",
    }:
        algorithm_params = {**algorithm_params, "undirected_isls": undirected_isls}
        config["simulation"]["algorithm_params"] = algorithm_params

    sim_config = config["simulation"]
    simulation_end_time_ns = int(sim_config["end_time_hours"] * 60 * 60 * 1e9)
    time_step_ns = int(sim_config["time_step_minutes"] * 60 * 1e9)
    offset_ns = int(sim_config.get("offset_ns", 0))

    satellite_ids = [sat.id for sat in sim_satellites]
    ground_station_ids = [gs.id for gs in ground_stations]

    time_steps = list(range(offset_ns, simulation_end_time_ns, time_step_ns))
    algorithm = get_routing_algorithm(sim_config["dynamic_state_algorithm"])
    max_hops = len(satellite_ids) + 2

    gsl_node_ids = list(range(len(sim_satellites))) + [gs.id for gs in ground_stations]
    gsl_interface_config = config["network"]["gsl_interfaces"]
    list_gsl_interfaces_info = [
        {
            "id": node_id,
            "number_of_interfaces": gsl_interface_config["number_of_interfaces"],
            "aggregate_max_bandwidth": gsl_interface_config["aggregate_max_bandwidth"],
        }
        for node_id in gsl_node_ids
    ]

    timestep_rows: list[dict] = []
    delta_rows: list[dict] = []
    control_plane_sample: dict | None = None
    prev_fstate: dict | None = None
    prev_attachments: list[tuple[int | None, float]] | None = None
    prev_route_plans: dict | None = None

    progress_iter = time_steps
    if tqdm is not None:
        constellation_name = config["constellation"]["name"]
        progress_iter = tqdm(
            time_steps,
            desc=(
                f"{constellation_name} "
                f"{sim_config['dynamic_state_algorithm']} {isl_scenario}"
            ),
            unit="step",
        )

    for step_index, time_since_epoch_ns in enumerate(progress_iter):
        time_absolute = parsed_tles_data["epoch"] + time_since_epoch_ns * astro_units.ns
        topology_with_isls, _ = _build_topologies(constellation_data, ground_stations)
        topology_with_isls.gsl_interfaces_info = list_gsl_interfaces_info
        _compute_isls(topology_with_isls, undirected_isls, time_absolute)
        gs_sat_visibility = _compute_ground_station_satellites_in_range(
            topology_with_isls, time_absolute
        )

        interface_neighbor_map = build_interface_neighbor_map(
            topology_with_isls.sat_neighbor_to_if
        )
        algorithm_params = sim_config.get("algorithm_params") or {}
        compute_start = time.perf_counter()
        fstate_output = algorithm.compute_state(
            time_since_epoch_ns=time_since_epoch_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            topology_with_isls=topology_with_isls,
            ground_station_satellites_in_range=gs_sat_visibility,
            list_gsl_interfaces_info=topology_with_isls.gsl_interfaces_info,
            algorithm_params=algorithm_params,
        )
        compute_duration_ms = (time.perf_counter() - compute_start) * 1000.0
        fstate = fstate_output.get("fstate", {})
        route_plans = fstate_output.get("route_plans", {})
        if control_plane_sample is None and fstate_output.get("control_plane"):
            control_plane_sample = fstate_output["control_plane"]

        attachments = get_gs_attachments(gs_sat_visibility)
        fstate_stats = compute_forwarding_state_stats(
            fstate,
            topology_with_isls.graph,
            sim_config["dynamic_state_algorithm"],
            satellite_ids,
            ground_station_ids,
            algorithm_params,
            route_plans,
        )
        stretch_stats = compute_path_stretch(
            fstate,
            topology_with_isls.graph,
            satellite_ids,
            ground_station_ids,
            attachments,
            interface_neighbor_map,
            max_hops,
            route_plans,
            gs_sat_visibility,
        )

        timestep_rows.append(
            {
                "time_index": step_index,
                "time_since_epoch_ns": time_since_epoch_ns,
                **flatten_distribution("fstate_size", fstate_stats),
                **flatten_distribution("stretch_hop", stretch_stats["hop"]),
                **flatten_distribution("stretch_dist", stretch_stats["distance"]),
                "compute_time_ms": compute_duration_ms,
            }
        )

        if prev_fstate is not None and prev_attachments is not None:
            gs_handover_rate = compute_gs_handover_rate(prev_attachments, attachments)
            gs_renumbering = compute_gs_renumbering_stats(prev_attachments, attachments)
            sat_gs_churn = compute_sat_to_gs_churn(
                prev_fstate,
                fstate,
                satellite_ids,
                ground_station_ids,
                interface_neighbor_map,
                prev_route_plans,
                route_plans,
            )
            gs_gs_churn = compute_gs_to_gs_churn(
                prev_fstate,
                fstate,
                ground_station_ids,
                prev_attachments,
                attachments,
                interface_neighbor_map,
                prev_route_plans,
                route_plans,
            )
            delta_rows.append(
                {
                    "time_index": step_index,
                    "time_since_epoch_ns": time_since_epoch_ns,
                    "gs_handover_rate": gs_handover_rate,
                    "gs_renumber_count": gs_renumbering["count"],
                    "gs_renumber_rate": gs_renumbering["rate"],
                    "sat_gs_churn": sat_gs_churn["churn"],
                    "sat_gs_break_rate": sat_gs_churn["break_rate"],
                    "gs_gs_churn": gs_gs_churn["churn"],
                    "gs_gs_break_rate": gs_gs_churn["break_rate"],
                }
            )

        prev_fstate = fstate
        prev_attachments = attachments
        prev_route_plans = route_plans
    metadata = {
        "algorithm": sim_config["dynamic_state_algorithm"],
        "algorithm_params": sim_config.get("algorithm_params") or {},
        "isl_scenario": isl_scenario,
        "constellation": {
            "name": config["constellation"]["name"],
            "num_orbits": config["constellation"]["num_orbits"],
            "num_sats_per_orbit": config["constellation"]["num_sats_per_orbit"],
            "altitude_m": config.get("satellite", {}).get("altitude_m"),
            "inclination_degree": config["constellation"].get("inclination_degree"),
        },
        "ground_stations": {
            "count": len(ground_stations),
            "override_path": gs_override_path,
        },
        "time_step_minutes": sim_config["time_step_minutes"],
        "end_time_hours": sim_config["end_time_hours"],
        "offset_ns": sim_config.get("offset_ns", 0),
        "generated_at": datetime.datetime.now().isoformat(),
        "forwarding_state_definition": {
            "shortest_path_link_state": "destination forwarding entries toward routable satellites (proxy: number of satellites)",
            "predictive_link_state": "destination forwarding entries toward routable satellites (proxy: number of satellites)",
            "explicit_path_routing": "stored pinned-path elements per source satellite across explicit route plans (real implementation state)",
            "traditional_segment_routing": "forwarding entries toward segment endpoints / routable satellites (proxy: number of satellites)",
            "topological_routing": "local neighbor-address forwarding entries (proxy: node degree)",
            "default": "reachable GS destinations per satellite",
        },
    }
    if control_plane_sample is not None:
        metadata["control_plane_sample"] = control_plane_sample

    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    write_csv(
        os.path.join(output_dir, "timestep_metrics.csv"),
        timestep_rows,
        fieldnames=list(timestep_rows[0].keys()) if timestep_rows else [],
    )
    write_csv(
        os.path.join(output_dir, "delta_metrics.csv"),
        delta_rows,
        fieldnames=list(delta_rows[0].keys()) if delta_rows else [],
    )

    log.info("Evaluation run complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LEOPath evaluation harness")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for CSV/JSON"
    )
    parser.add_argument(
        "--isl-scenario",
        choices=("ring", "grid"),
        default="grid",
        help="ISL scenario to evaluate",
    )
    parser.add_argument(
        "--algorithm", default=None, help="Routing algorithm name override"
    )
    parser.add_argument(
        "--gs-config", default=None, help="Ground station list override YAML"
    )
    parser.add_argument("--end-time-hours", type=float, default=None)
    parser.add_argument("--time-step-minutes", type=float, default=None)
    parser.add_argument("--prediction-horizon-minutes", type=float, default=None)
    parser.add_argument("--segment-count", type=int, default=None)
    parser.add_argument("--segment-refresh-interval-steps", type=int, default=None)
    parser.add_argument("--segment-mode", type=str, default=None)
    parser.add_argument("--plane-weight", type=float, default=None)
    parser.add_argument("--sat-weight", type=float, default=None)
    parser.add_argument("--shell-weight", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        config_path=args.config,
        output_dir=args.output_dir,
        isl_scenario=args.isl_scenario,
        algorithm_name=args.algorithm,
        gs_override_path=args.gs_config,
        end_time_hours=args.end_time_hours,
        time_step_minutes=args.time_step_minutes,
        prediction_horizon_minutes=args.prediction_horizon_minutes,
        segment_count=args.segment_count,
        segment_refresh_interval_steps=args.segment_refresh_interval_steps,
        segment_mode=args.segment_mode,
        plane_weight=args.plane_weight,
        sat_weight=args.sat_weight,
        shell_weight=args.shell_weight,
    )


if __name__ == "__main__":
    main()
