import argparse
import datetime
import logging
import os
import time

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
from leopath.network_state.gsl_attachment.multihoming import (
    ATTACHMENT_POLICIES,
    select_multihoming_attachments,
)
from leopath.network_state.helpers import (
    _compute_ground_station_satellites_in_range,
    _compute_isls,
)
from leopath.network_state.routing_algorithms.routing_algorithm_factory import (
    get_routing_algorithm,
)
from leopath.topology.topology import ConstellationData

from .failures import FAILURE_TYPES, FailureConfig, FailureProcess, satellite_latitudes_deg
from .metrics import (
    build_interface_neighbor_map,
    compute_explicit_failover_stats,
    compute_explicit_header_stats,
    compute_forwarding_state_stats,
    compute_gs_handover_rate,
    compute_gs_renumbering_stats,
    compute_gs_to_gs_churn,
    compute_installed_state_breakdown,
    compute_path_stretch,
    compute_sat_to_gs_churn,
    compute_satellite_forwarding_state_updates,
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
    raise ValueError("Ground station override must be a list or contain ground_stations")


def has_counter_rotating_seam(raan_spread_degree: float) -> bool:
    """True for a Walker star, whose nodes span less than a full circle.

    In a star the last plane and the first head in opposite directions, so a
    wrap ISL between them would join satellites up to half an orbit apart. A
    Walker delta spreads its nodes over 360 degrees and its wrap is an ordinary
    co-rotating neighbour link.
    """
    return raan_spread_degree < 360.0


def select_isls(
    constellation: ConstellationData, scenario: str, raan_spread_degree: float = 360.0
) -> list[tuple[int, int]]:
    if scenario == "ring":
        return setup_isls_in_the_same_orbit(
            num_orbits=constellation.n_orbits,
            sats_per_orbit=constellation.n_sats_per_orbit,
        )
    if scenario == "grid":
        # A star shell cannot close the torus, so its +Grid is the cylinder.
        return generate_plus_grid_isls(
            n_orbits=constellation.n_orbits,
            n_sats_per_orbit=constellation.n_sats_per_orbit,
            idx_offset=0,
            seam=has_counter_rotating_seam(raan_spread_degree),
        )
    if scenario == "grid_seam":
        return generate_plus_grid_isls(
            n_orbits=constellation.n_orbits,
            n_sats_per_orbit=constellation.n_sats_per_orbit,
            idx_offset=0,
            seam=True,
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


def _set_gs_addressing_params(
    algorithm_params: dict,
    algorithm_name: str,
    gs_addressing: str | None,
    gs_attachment_count: int | None,
    gs_attachment_policy: str | None,
) -> None:
    if algorithm_name not in ("topological_routing", "shortest_path_link_state"):
        return
    if gs_addressing is not None:
        algorithm_params["gs_addressing"] = gs_addressing
    if gs_attachment_count is not None:
        if gs_attachment_count < 1:
            raise ValueError("gs_attachment_count must be at least 1")
        algorithm_params["gs_attachment_count"] = gs_attachment_count
    if gs_attachment_policy is not None:
        if gs_attachment_policy not in ATTACHMENT_POLICIES:
            raise ValueError(
                f"Unknown gs_attachment_policy {gs_attachment_policy!r}, "
                f"expected one of {ATTACHMENT_POLICIES}"
            )
        algorithm_params["gs_attachment_policy"] = gs_attachment_policy


def prepare_algorithm_params(
    simulation_config: dict,
    algorithm_name: str,
    segment_count: int | None,
    segment_refresh_interval_steps: int | None,
    plane_weight: float | None,
    sat_weight: float | None,
    shell_weight: float | None,
    distance_mode: str | None,
    explicit_final_egress_mode: str | None,
    time_step_minutes: float | None,
    geometry_source: str | None = None,
    explicit_backup_adjacencies: bool = False,
    forwarding_guard: str | None = None,
    local_repair: str | None = None,
    exception_policy: str | None = None,
    gs_addressing: str | None = None,
    gs_attachment_count: int | None = None,
    gs_attachment_policy: str | None = None,
) -> dict:
    algorithm_params = dict(simulation_config.get("algorithm_params") or {})

    if algorithm_name == "explicit_path_routing":
        algorithm_params.pop("plane_weight", None)
        algorithm_params.pop("sat_weight", None)
        algorithm_params.pop("shell_weight", None)

    if segment_count is not None:
        algorithm_params["segment_count"] = segment_count
    if segment_refresh_interval_steps is not None:
        algorithm_params["segment_refresh_interval_steps"] = segment_refresh_interval_steps
    elif algorithm_name == "explicit_path_routing":
        algorithm_params.setdefault("segment_refresh_interval_steps", 1)
    if plane_weight is not None and algorithm_name != "explicit_path_routing":
        algorithm_params["plane_weight"] = plane_weight
    if sat_weight is not None and algorithm_name != "explicit_path_routing":
        algorithm_params["sat_weight"] = sat_weight
    if shell_weight is not None and algorithm_name != "explicit_path_routing":
        algorithm_params["shell_weight"] = shell_weight
    if distance_mode is not None and algorithm_name == "topological_routing":
        algorithm_params["distance_mode"] = distance_mode
    if explicit_final_egress_mode is not None and algorithm_name == "explicit_path_routing":
        algorithm_params["final_egress_mode"] = explicit_final_egress_mode
    if geometry_source is not None and algorithm_name == "topological_routing":
        algorithm_params["geometry_source"] = geometry_source
    if forwarding_guard is not None and algorithm_name == "topological_routing":
        algorithm_params["forwarding_guard"] = forwarding_guard
    if local_repair is not None and algorithm_name == "topological_routing":
        algorithm_params["local_repair"] = local_repair
    if exception_policy is not None and algorithm_name == "topological_routing":
        algorithm_params["exception_policy"] = exception_policy
    _set_gs_addressing_params(
        algorithm_params,
        algorithm_name,
        gs_addressing,
        gs_attachment_count,
        gs_attachment_policy,
    )
    if explicit_backup_adjacencies and algorithm_name == "explicit_path_routing":
        algorithm_params["include_backup_adjacencies"] = True

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
    segment_count: int | None,
    segment_refresh_interval_steps: int | None,
    plane_weight: float | None,
    sat_weight: float | None,
    shell_weight: float | None,
    distance_mode: str | None,
    explicit_final_egress_mode: str | None,
    geometry_source: str | None = None,
    explicit_backup_adjacencies: bool = False,
    failure_config: FailureConfig | None = None,
    forwarding_guard: str | None = None,
    local_repair: str | None = None,
    exception_policy: str | None = None,
    gs_addressing: str | None = None,
    gs_attachment_count: int | None = None,
    gs_attachment_policy: str | None = None,
) -> None:
    config = load_config(config_path)
    gs_override = load_ground_station_override(gs_override_path)
    if gs_override is not None:
        config["ground_stations"] = gs_override
    if end_time_hours is not None:
        config["simulation"]["end_time_hours"] = end_time_hours
    if time_step_minutes is not None:
        config["simulation"]["time_step_minutes"] = time_step_minutes

    effective_algorithm_name = algorithm_name or config["simulation"]["dynamic_state_algorithm"]
    config["simulation"]["dynamic_state_algorithm"] = effective_algorithm_name
    algorithm_params = prepare_algorithm_params(
        simulation_config=config["simulation"],
        algorithm_name=effective_algorithm_name,
        segment_count=segment_count,
        segment_refresh_interval_steps=segment_refresh_interval_steps,
        plane_weight=plane_weight,
        sat_weight=sat_weight,
        shell_weight=shell_weight,
        distance_mode=distance_mode,
        explicit_final_egress_mode=explicit_final_egress_mode,
        time_step_minutes=time_step_minutes,
        geometry_source=geometry_source,
        explicit_backup_adjacencies=explicit_backup_adjacencies,
        forwarding_guard=forwarding_guard,
        local_repair=local_repair,
        exception_policy=exception_policy,
        gs_addressing=gs_addressing,
        gs_attachment_count=gs_attachment_count,
        gs_attachment_policy=gs_attachment_policy,
    )
    if algorithm_params:
        config["simulation"]["algorithm_params"] = algorithm_params

    os.makedirs(output_dir, exist_ok=True)
    logger.setup_logger(is_debug=False, file_name=os.path.join(output_dir, "eval_harness.log"))
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

    raan_spread_degree = float(config["constellation"].get("raan_spread_degree", 360.0))
    undirected_isls = select_isls(constellation_data, isl_scenario, raan_spread_degree)
    sim_config = config["simulation"]
    simulation_end_time_ns = int(sim_config["end_time_hours"] * 60 * 60 * 1e9)
    time_step_ns = int(sim_config["time_step_minutes"] * 60 * 1e9)
    offset_ns = int(sim_config.get("offset_ns", 0))

    satellite_ids = [sat.id for sat in sim_satellites]
    ground_station_ids = [gs.id for gs in ground_stations]

    time_steps = list(range(offset_ns, simulation_end_time_ns, time_step_ns))
    # The failure pattern is drawn from the seed and the scenario, never from the
    # algorithm, so every algorithm routes over identical failures.
    failure_process = FailureProcess(
        failure_config or FailureConfig(),
        n_orbits=constellation_data.n_orbits,
        n_sats_per_orbit=constellation_data.n_sats_per_orbit,
        undirected_isls=undirected_isls,
        time_step_minutes=float(sim_config["time_step_minutes"]),
        stream_key=f"{config['constellation']['name']}|{isl_scenario}",
        latitude_provider=lambda time_absolute: satellite_latitudes_deg(
            sim_satellites, constellation_data.epoch, time_absolute
        ),
    )
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
    prev_interface_neighbor_map: dict[int, dict[int, int]] | None = None

    progress_iter = time_steps
    if tqdm is not None:
        constellation_name = config["constellation"]["name"]
        progress_iter = tqdm(
            time_steps,
            desc=(
                f"{constellation_name} " f"{sim_config['dynamic_state_algorithm']} {isl_scenario}"
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
        failure_stats = failure_process.inject(topology_with_isls, gs_sat_visibility, time_absolute)

        interface_neighbor_map = build_interface_neighbor_map(topology_with_isls.sat_neighbor_to_if)
        algorithm_params = sim_config.get("algorithm_params") or {}
        routing_gs_visibility = gs_sat_visibility
        attachment_assignment_stats: dict[str, float] = {}
        if algorithm_params.get("gs_addressing") == "attachment":
            routing_gs_visibility, attachment_assignment_stats = select_multihoming_attachments(
                gs_sat_visibility,
                int(algorithm_params.get("gs_attachment_count", 1)),
                str(algorithm_params.get("gs_attachment_policy", "independent")),
            )
        compute_start = time.perf_counter()
        fstate_output = algorithm.compute_state(
            time_since_epoch_ns=time_since_epoch_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            topology_with_isls=topology_with_isls,
            ground_station_satellites_in_range=routing_gs_visibility,
            list_gsl_interfaces_info=topology_with_isls.gsl_interfaces_info,
            algorithm_params=algorithm_params,
        )
        compute_duration_ms = (time.perf_counter() - compute_start) * 1000.0
        fstate = fstate_output.get("fstate", {})
        route_plans = fstate_output.get("route_plans", {})
        selected_egresses = fstate_output.get("selected_egresses", {})
        # Per-category auxiliary state: geometry and path-cost tables the
        # distance estimator maintains, reported separately from installed
        # forwarding entries rather than folded into them.
        auxiliary_state = fstate_output.get("auxiliary_state") or {}
        auxiliary_state.update(attachment_assignment_stats)
        if control_plane_sample is None and fstate_output.get("control_plane"):
            control_plane_sample = fstate_output["control_plane"]

        attachments = get_gs_attachments(routing_gs_visibility)
        fstate_stats = compute_forwarding_state_stats(
            fstate,
            topology_with_isls.graph,
            sim_config["dynamic_state_algorithm"],
            satellite_ids,
            ground_station_ids,
            attachments,
            algorithm_params,
            route_plans,
        )
        installed_state = compute_installed_state_breakdown(
            fstate,
            topology_with_isls.graph,
            satellite_ids,
            ground_station_ids,
        )
        explicit_header_stats = compute_explicit_header_stats(
            route_plans,
            attachments,
            ground_station_ids,
        )
        explicit_srv6_srh_stats = compute_explicit_header_stats(
            route_plans,
            attachments,
            ground_station_ids,
            bytes_key="srv6_srh_bytes",
        )
        explicit_failover_stats = compute_explicit_failover_stats(
            topology_with_isls.graph,
            route_plans,
            ground_station_ids,
            gs_sat_visibility,
            attachments,
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
            selected_egresses,
        )

        timestep_rows.append(
            {
                "time_index": step_index,
                "time_since_epoch_ns": time_since_epoch_ns,
                **flatten_distribution("fstate_size", fstate_stats),
                **flatten_distribution("fstate_installed", installed_state["installed"]),
                **flatten_distribution("fstate_markers", installed_state["unreachable_markers"]),
                **flatten_distribution("fstate_neighbors", installed_state["neighbor_entries"]),
                **flatten_distribution("strict_header_bytes", explicit_header_stats),
                **flatten_distribution("srv6_srh_bytes", explicit_srv6_srh_stats),
                **flatten_distribution("stretch_hop", stretch_stats["hop"]),
                **flatten_distribution("stretch_dist", stretch_stats["distance"]),
                **flatten_distribution("stretch_hop_shared", stretch_stats["hop_shared"]),
                **flatten_distribution("stretch_dist_shared", stretch_stats["distance_shared"]),
                **flatten_distribution("stretch_hop_egress", stretch_stats["hop_egress"]),
                **flatten_distribution("stretch_dist_egress", stretch_stats["distance_egress"]),
                **{f"delivery_{key}": value for key, value in stretch_stats["delivery"].items()},
                **{f"aux_{key}": value for key, value in auxiliary_state.items()},
                **{
                    f"explicit_failover_{key}": value
                    for key, value in explicit_failover_stats.items()
                },
                **{f"failure_{key}": value for key, value in failure_stats.items()},
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
            satellite_fstate_updates = compute_satellite_forwarding_state_updates(
                prev_fstate=prev_fstate,
                curr_fstate=fstate,
                algorithm_name=sim_config["dynamic_state_algorithm"],
                satellite_ids=satellite_ids,
                ground_station_ids=ground_station_ids,
                prev_attachments=prev_attachments,
                curr_attachments=attachments,
                prev_interface_neighbor_map=prev_interface_neighbor_map,
                curr_interface_neighbor_map=interface_neighbor_map,
                prev_route_plans=prev_route_plans,
                curr_route_plans=route_plans,
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
                    "sat_fstate_updates_add_mean": satellite_fstate_updates["add"]["mean"],
                    "sat_fstate_updates_add_p95": satellite_fstate_updates["add"]["p95"],
                    "sat_fstate_updates_delete_mean": satellite_fstate_updates["delete"]["mean"],
                    "sat_fstate_updates_delete_p95": satellite_fstate_updates["delete"]["p95"],
                    "sat_fstate_updates_modify_mean": satellite_fstate_updates["modify"]["mean"],
                    "sat_fstate_updates_modify_p95": satellite_fstate_updates["modify"]["p95"],
                    "sat_fstate_updates_total_mean": satellite_fstate_updates["total"]["mean"],
                    "sat_fstate_updates_total_p95": satellite_fstate_updates["total"]["p95"],
                    "sat_fstate_updates_touched_satellite_count": satellite_fstate_updates[
                        "touched_satellite_count"
                    ],
                    "sat_fstate_updates_touched_satellite_rate": satellite_fstate_updates[
                        "touched_satellite_rate"
                    ],
                }
            )

        prev_fstate = fstate
        prev_attachments = attachments
        prev_route_plans = route_plans
        prev_interface_neighbor_map = interface_neighbor_map
    metadata = {
        "algorithm": sim_config["dynamic_state_algorithm"],
        "algorithm_params": sim_config.get("algorithm_params") or {},
        "isl_scenario": isl_scenario,
        # Whether the +Grid wrap between the last and first plane was built. It is
        # absent for ring and grid_seam, and for grid on a Walker star shell.
        "isl_seam_wrap": isl_scenario == "grid"
        and not has_counter_rotating_seam(raan_spread_degree),
        "raan_spread_degree": raan_spread_degree,
        "failure_model": failure_process.describe(),
        # Set by the runner scripts to the image tag, so outputs from different
        # builds sharing one output tree can be told apart.
        "code_version": os.environ.get("LEOPATH_CODE_VERSION"),
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
            "explicit_path_routing": "local neighbor/interface entries on all satellites, plus destination-to-segment ingress bindings only on satellites that currently have attached ground stations; strict-adjacency header bytes are tracked separately in route plans",
            "topological_routing": "local neighbor-address forwarding entries (proxy: node degree)",
            "dra_routing": "local neighbor-address forwarding entries (proxy: node degree); DRA-style hop-only logical-coordinate baseline",
            "default": "reachable GS destinations per satellite",
        },
        "satellite_forwarding_state_update_definition": {
            "definition": "per-snapshot additions, deletions, or modifications of satellite-local forwarding-state units between consecutive snapshots",
            "shortest_path_link_state": "destination-to-next-hop forwarding entries toward currently attached destination satellites",
            "topological_routing": "mutable satellite-local forwarding state only; in the regular Ring/+Grid model this is limited to local GS delivery bindings and any explicit exception state, not algorithmic default forwarding",
            "dra_routing": "mutable satellite-local forwarding state only, as for topological_routing; the families differ solely in the distance metric used for default forwarding",
            "explicit_path_routing": "satellite-local GS delivery bindings plus ingress destination-to-path bindings on satellites that currently host GS attachments; packet-carried adjacency guidance is excluded",
        },
        "packet_guidance_definition": {
            "strict_header_bytes": "compact proxy overhead for active GS-to-GS traffic only: fixed 20-byte metadata plus 32-bit adjacency SIDs, zero for direct local delivery",
            "srv6_srh_bytes": "SRv6 SRH-equivalent overhead for the same active strict adjacency stack, excluding the IPv6 base header: 8 + 16 bytes per carried adjacency SID, zero for direct local delivery",
            "dynamic_egress_repair": "optional loose-final-egress repair for explicit paths: the cached strict core path terminates at its planned anchor, then final GS delivery is resolved toward the current visible egress",
        },
    }
    if control_plane_sample is not None:
        metadata["control_plane_sample"] = control_plane_sample
        effective_refresh_interval_steps = control_plane_sample.get(
            "effective_refresh_interval_steps"
        )
        if effective_refresh_interval_steps is not None:
            metadata["algorithm_params"][
                "segment_refresh_interval_steps"
            ] = effective_refresh_interval_steps

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
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV/JSON")
    parser.add_argument(
        "--isl-scenario",
        choices=("ring", "grid", "grid_seam"),
        default="grid",
        help="ISL scenario to evaluate",
    )
    parser.add_argument("--algorithm", default=None, help="Routing algorithm name override")
    parser.add_argument("--gs-config", default=None, help="Ground station list override YAML")
    parser.add_argument("--end-time-hours", type=float, default=None)
    parser.add_argument("--time-step-minutes", type=float, default=None)
    parser.add_argument("--segment-count", type=int, default=None)
    parser.add_argument("--segment-refresh-interval-steps", type=int, default=None)
    parser.add_argument("--plane-weight", type=float, default=None)
    parser.add_argument("--sat-weight", type=float, default=None)
    parser.add_argument("--shell-weight", type=float, default=None)
    parser.add_argument("--distance-mode", type=str, default=None)
    parser.add_argument(
        "--explicit-final-egress-mode",
        choices=("strict", "dynamic"),
        default=None,
    )
    parser.add_argument(
        "--explicit-backup-adjacencies",
        action="store_true",
        help="Give explicit paths single-hop local protection around a failed adjacency",
    )
    parser.add_argument(
        "--geometry-source",
        choices=("observed", "nominal"),
        default=None,
        help="Graph the topological pivot geometry is built from under failures",
    )
    parser.add_argument(
        "--forwarding-guard",
        choices=("none", "progress"),
        default=None,
        help="Topological routing: forward only to neighbours that lower the egress potential",
    )
    parser.add_argument(
        "--gs-addressing",
        choices=("visibility", "attachment"),
        default=None,
        help=(
            "Topological routing and link-state: 'attachment' makes a ground station's "
            "address name the satellite it is attached to, so satellites forward toward that "
            "address and need nothing about where the ground station sits, and link-state "
            "routes to that same single egress; 'visibility' keeps the address stable and "
            "minimises over every visible egress instead"
        ),
    )
    parser.add_argument(
        "--gs-attachment-count",
        type=int,
        default=None,
        help=(
            "With --gs-addressing attachment, advertise the K nearest live "
            "satellite addresses for each ground station (default: 1)"
        ),
    )
    parser.add_argument(
        "--gs-attachment-policy",
        choices=ATTACHMENT_POLICIES,
        default=None,
        help=(
            "How K satellite addresses are assigned: 'independent' is the top-K "
            "upper bound; 'exclusive' lets each satellite serve at most one station"
        ),
    )
    parser.add_argument(
        "--local-repair",
        choices=("none", "square"),
        default=None,
        help="Topological routing: reach a next hop cut off by a failed ISL over a 3-hop detour",
    )
    parser.add_argument(
        "--exception-policy",
        choices=("none", "grow"),
        default=None,
        help="Topological routing: install exception entries where the rules cannot deliver",
    )
    parser.add_argument("--failure-type", choices=FAILURE_TYPES, default="none")
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=0.0,
        help="Stationary probability that an ISL or satellite is down",
    )
    parser.add_argument("--failure-seed", type=int, default=0)
    parser.add_argument(
        "--failure-mean-duration-minutes",
        type=float,
        default=None,
        help="Mean outage length; defaults to 10 for ISLs and 60 for satellites",
    )
    parser.add_argument("--failure-void-size", type=int, default=2)
    parser.add_argument("--failure-polar-latitude-deg", type=float, default=75.0)
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
        segment_count=args.segment_count,
        segment_refresh_interval_steps=args.segment_refresh_interval_steps,
        plane_weight=args.plane_weight,
        sat_weight=args.sat_weight,
        shell_weight=args.shell_weight,
        distance_mode=args.distance_mode,
        explicit_final_egress_mode=args.explicit_final_egress_mode,
        geometry_source=args.geometry_source,
        explicit_backup_adjacencies=args.explicit_backup_adjacencies,
        forwarding_guard=args.forwarding_guard,
        local_repair=args.local_repair,
        exception_policy=args.exception_policy,
        gs_addressing=args.gs_addressing,
        gs_attachment_count=args.gs_attachment_count,
        gs_attachment_policy=args.gs_attachment_policy,
        failure_config=FailureConfig(
            failure_type=args.failure_type,
            rate=args.failure_rate,
            seed=args.failure_seed,
            mean_duration_minutes=args.failure_mean_duration_minutes,
            void_size=args.failure_void_size,
            polar_latitude_deg=args.failure_polar_latitude_deg,
        ),
    )


if __name__ == "__main__":
    main()
