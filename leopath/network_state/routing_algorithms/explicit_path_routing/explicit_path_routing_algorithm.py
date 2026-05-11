from __future__ import annotations

import networkx as nx
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

from leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing_algorithm import (
    _add_gs_to_gs_fstate,
    _calculate_bandwidth_state,
    _handle_direct_gs_path,
    _path_to_segments,
)


def algorithm_explicit_path_routing(
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,
    gsl_attachment_strategy: GSLAttachmentStrategy,
    current_time: Time,
    list_gsl_interfaces_info: list,
    algorithm_params: dict,
    current_ground_station_satellites_in_range: list | None = None,
    cached_route_plans: dict[tuple[int, int], dict] | None = None,
    control_plane_metadata: dict | None = None,
) -> dict:
    ground_station_satellites_in_range = current_ground_station_satellites_in_range
    if ground_station_satellites_in_range is None:
        gsl_attachments = gsl_attachment_strategy.select_attachments(
            topology_with_isls, ground_stations, current_time
        )
        ground_station_satellites_in_range = []
        for distance, sat_id in gsl_attachments:
            if sat_id != -1:
                ground_station_satellites_in_range.append([(distance, sat_id)])
            else:
                ground_station_satellites_in_range.append([])

    route_plans = cached_route_plans or _build_route_plans(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
    )
    waypoint_plans = _build_waypoint_plans(
        route_plans,
        int(algorithm_params.get("segment_count", 2)),
    )
    fstate = _materialize_first_hop_fstate(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        route_plans,
    )
    _add_gs_to_gs_fstate(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        fstate,
    )

    bandwidth_state = _calculate_bandwidth_state(
        constellation_data, ground_stations, list_gsl_interfaces_info
    )

    return {
        "fstate": fstate,
        "bandwidth": bandwidth_state,
        "route_plans": route_plans,
        "control_plane": {
            **(control_plane_metadata or {}),
            "sample_route_plans": [
                {
                    "source_satellite_id": src_sat_id,
                    "destination_ground_station_id": dst_gs_id,
                    "satellite_path": sat_path,
                    "waypoint_satellites": waypoint_plans[(src_sat_id, dst_gs_id)],
                }
                for (src_sat_id, dst_gs_id), sat_path in list(route_plans.items())[:5]
            ],
        },
    }


def _build_route_plans(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
) -> dict[tuple[int, int], dict]:
    sat_ids = sorted({sat.id for sat in topology_with_isls.get_satellites()})
    sat_graph = topology_with_isls.graph.subgraph(sat_ids)
    route_plans: dict[tuple[int, int], dict] = {}

    for src_sat_id in sat_ids:
        for gs_idx, dst_gs in enumerate(ground_stations):
            visible = ground_station_satellites_in_range[gs_idx]
            if not visible:
                route_plans[(src_sat_id, dst_gs.id)] = {}
                continue
            _, dst_sat_id = min(visible, key=lambda item: item[0])
            if src_sat_id == dst_sat_id:
                route_plans[(src_sat_id, dst_gs.id)] = {
                    "satellite_path": [src_sat_id],
                    "planned_dst_sat_id": dst_sat_id,
                }
                continue
            try:
                route_plans[(src_sat_id, dst_gs.id)] = {
                    "satellite_path": nx.shortest_path(
                        sat_graph,
                        source=src_sat_id,
                        target=dst_sat_id,
                        weight="weight",
                    ),
                    "planned_dst_sat_id": dst_sat_id,
                }
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                route_plans[(src_sat_id, dst_gs.id)] = {}
    return route_plans


def _build_waypoint_plans(
    route_plans: dict[tuple[int, int], dict],
    segment_count: int,
) -> dict[tuple[int, int], list[int]]:
    waypoint_plans: dict[tuple[int, int], list[int]] = {}
    for key, route_plan in route_plans.items():
        sat_path = route_plan.get("satellite_path", [])
        waypoint_plans[key] = _path_to_segments(sat_path, segment_count) if sat_path else []
    return waypoint_plans


def _materialize_first_hop_fstate(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
    route_plans: dict[tuple[int, int], dict],
) -> dict[tuple[int, int], tuple[int, int, int]]:
    fstate: dict[tuple[int, int], tuple[int, int, int]] = {}
    for src_sat_id, dst_gs_id in route_plans:
        route_plan = route_plans[(src_sat_id, dst_gs_id)]
        sat_path = route_plan.get("satellite_path", [])
        planned_dst_sat_id = route_plan.get("planned_dst_sat_id")
        if not sat_path:
            fstate[(src_sat_id, dst_gs_id)] = (-1, -1, -1)
            continue

        gs_idx = _ground_station_index(ground_stations, dst_gs_id)
        if gs_idx is None:
            fstate[(src_sat_id, dst_gs_id)] = (-1, -1, -1)
            continue

        visible = ground_station_satellites_in_range[gs_idx]
        if not visible:
            fstate[(src_sat_id, dst_gs_id)] = (-1, -1, -1)
            continue

        if src_sat_id == planned_dst_sat_id:
            if not _ground_station_visible_from_satellite(visible, planned_dst_sat_id):
                fstate[(src_sat_id, dst_gs_id)] = (-1, -1, -1)
                continue
            fstate[(src_sat_id, dst_gs_id)] = _handle_direct_gs_path(
                planned_dst_sat_id, dst_gs_id, topology_with_isls
            )
            continue

        if len(sat_path) < 2:
            fstate[(src_sat_id, dst_gs_id)] = (-1, -1, -1)
            continue
        next_sat = sat_path[1]
        my_if = topology_with_isls.sat_neighbor_to_if.get((src_sat_id, next_sat), -1)
        next_if = topology_with_isls.sat_neighbor_to_if.get((next_sat, src_sat_id), -1)
        if my_if < 0 or next_if < 0:
            fstate[(src_sat_id, dst_gs_id)] = (-1, -1, -1)
            continue
        fstate[(src_sat_id, dst_gs_id)] = (next_sat, my_if, next_if)
    return fstate


def _ground_station_index(
    ground_stations: list[GroundStation],
    ground_station_id: int,
) -> int | None:
    for index, ground_station in enumerate(ground_stations):
        if ground_station.id == ground_station_id:
            return index
    return None


def _ground_station_visible_from_satellite(
    visible_satellites: list[tuple[float, int]],
    satellite_id: int | None,
) -> bool:
    if satellite_id is None:
        return False
    return any(visible_sat_id == satellite_id for _, visible_sat_id in visible_satellites)
