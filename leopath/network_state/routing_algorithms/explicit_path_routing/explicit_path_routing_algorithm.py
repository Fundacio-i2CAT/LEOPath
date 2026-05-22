from __future__ import annotations

import networkx as nx
from heapq import heappop, heappush
from itertools import count
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
        include_backup_adjacencies=bool(
            algorithm_params.get("include_backup_adjacencies", False)
        ),
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
                    "satellite_path": route_plan.get("satellite_path", []),
                    "adjacency_sid_list": route_plan.get("adjacency_sid_list", []),
                    "backup_adjacency_sid_list": route_plan.get("backup_adjacency_sid_list", []),
                    "strict_header_bytes": route_plan.get("strict_header_bytes", 0),
                    "srv6_srh_bytes": route_plan.get("srv6_srh_bytes", 0),
                    "planned_dst_sat_id": route_plan.get("planned_dst_sat_id"),
                    "waypoint_satellites": waypoint_plans[(src_sat_id, dst_gs_id)],
                }
                for (src_sat_id, dst_gs_id), route_plan in list(route_plans.items())[:5]
            ],
        },
    }


def _build_route_plans(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
    include_backup_adjacencies: bool = False,
) -> dict[tuple[int, int], dict]:
    sat_ids = sorted({sat.id for sat in topology_with_isls.get_satellites()})
    sat_graph = topology_with_isls.graph.subgraph(sat_ids)
    route_plans: dict[tuple[int, int], dict] = {}
    destination_satellite_options: dict[int, list[tuple[float, int]]] = {}

    for gs_idx, dst_gs in enumerate(ground_stations):
        visible = ground_station_satellites_in_range[gs_idx]
        if not visible:
            destination_satellite_options[dst_gs.id] = []
            continue
        destination_satellite_options[dst_gs.id] = sorted(visible, key=lambda item: item[0])

    for dst_gs in ground_stations:
        visible = destination_satellite_options.get(dst_gs.id, [])
        if not visible:
            for src_sat_id in sat_ids:
                route_plans[(src_sat_id, dst_gs.id)] = {}
            continue

        best_paths = _single_destination_gs_shortest_paths(sat_graph, visible)

        for src_sat_id in sat_ids:
            path_info = best_paths.get(src_sat_id)
            if path_info is None:
                route_plans[(src_sat_id, dst_gs.id)] = {}
                continue
            total_distance, path_from_egress, dst_gsl_distance = path_info
            dst_sat_id = path_from_egress[0]
            sat_path_distance = float(total_distance) - float(dst_gsl_distance)
            sat_path = list(reversed(path_from_egress))
            route_plans[(src_sat_id, dst_gs.id)] = _build_strict_route_plan(
                sat_graph,
                src_sat_id,
                sat_path,
                dst_sat_id,
                sat_path_distance,
                dst_gsl_distance,
                include_backup_adjacencies=include_backup_adjacencies,
            )
    return route_plans


def _single_destination_gs_shortest_paths(
    sat_graph: nx.Graph,
    visible: list[tuple[float, int]],
) -> dict[int, tuple[float, list[int], float]]:
    best_seed_by_sat: dict[int, float] = {}
    for dst_gsl_distance, dst_sat_id in visible:
        if dst_sat_id not in sat_graph:
            continue
        distance = float(dst_gsl_distance)
        if distance < best_seed_by_sat.get(dst_sat_id, float("inf")):
            best_seed_by_sat[dst_sat_id] = distance

    tie_breaker = count()
    heap = []
    for dst_sat_id, dst_gsl_distance in best_seed_by_sat.items():
        heappush(
            heap,
            (dst_gsl_distance, next(tie_breaker), dst_sat_id, dst_sat_id, None, dst_gsl_distance),
        )

    settled: dict[int, tuple[float, int, int | None, float]] = {}
    while heap:
        total_distance, _, current_sat_id, root_sat_id, predecessor_sat_id, dst_gsl_distance = heappop(heap)
        if current_sat_id in settled:
            continue
        settled[current_sat_id] = (
            float(total_distance),
            root_sat_id,
            predecessor_sat_id,
            float(dst_gsl_distance),
        )

        for neighbor_sat_id in sat_graph.neighbors(current_sat_id):
            if neighbor_sat_id in settled:
                continue
            edge_weight = sat_graph.edges[current_sat_id, neighbor_sat_id].get("weight")
            if edge_weight is None:
                continue
            edge_weight = float(edge_weight)
            if edge_weight == float("inf"):
                continue
            heappush(
                heap,
                (
                    float(total_distance) + edge_weight,
                    next(tie_breaker),
                    neighbor_sat_id,
                    root_sat_id,
                    current_sat_id,
                    float(dst_gsl_distance),
                ),
            )

    best_paths: dict[int, tuple[float, list[int], float]] = {}
    for sat_id, (total_distance, _, _, dst_gsl_distance) in settled.items():
        path = []
        current_sat_id = sat_id
        while True:
            path.append(current_sat_id)
            predecessor_sat_id = settled[current_sat_id][2]
            if predecessor_sat_id is None:
                break
            current_sat_id = predecessor_sat_id
        best_paths[sat_id] = (
            float(total_distance),
            list(reversed(path)),
            float(dst_gsl_distance),
        )
    return best_paths


def _build_strict_route_plan(
    sat_graph: nx.Graph,
    src_sat_id: int,
    sat_path: list[int],
    planned_dst_sat_id: int,
    planned_sat_path_distance_m: float | None = None,
    planned_dst_gsl_distance_m: float | None = None,
    include_backup_adjacencies: bool = False,
) -> dict:
    adjacency_sid_list = sat_path[1:] if sat_path and sat_path[0] == src_sat_id else []
    backup_adjacency_sid_list = (
        _build_backup_adjacency_sid_list(sat_graph, sat_path)
        if include_backup_adjacencies
        else []
    )
    # Minimal strict-header proxy: fixed metadata plus 32-bit adjacency SIDs.
    strict_header_bytes = 20 + (4 * len(adjacency_sid_list)) if adjacency_sid_list else 0
    # SRv6 SRH equivalent for the same strict adjacency stack, excluding IPv6 base header.
    srv6_srh_bytes = 8 + (16 * len(adjacency_sid_list)) if adjacency_sid_list else 0
    return {
        "satellite_path": sat_path,
        "planned_dst_sat_id": planned_dst_sat_id,
        "planned_sat_path_distance_m": planned_sat_path_distance_m,
        "planned_dst_gsl_distance_m": planned_dst_gsl_distance_m,
        "forwarding_mode": "strict_adjacency_header",
        "local_protection_mode": "single_hop_rejoin" if include_backup_adjacencies else "none",
        "ingress_sat_id": src_sat_id,
        "adjacency_sid_list": adjacency_sid_list,
        "backup_adjacency_sid_list": backup_adjacency_sid_list,
        "strict_header_bytes": strict_header_bytes,
        "srv6_srh_bytes": srv6_srh_bytes,
    }


def _build_backup_adjacency_sid_list(
    sat_graph: nx.Graph,
    sat_path: list[int],
) -> list[int | None]:
    backup_sid_list: list[int | None] = []
    for current_sat_id, primary_next_sat_id in zip(sat_path, sat_path[1:]):
        backup_candidates = []
        for backup_next_sat_id in sat_graph.neighbors(current_sat_id):
            if backup_next_sat_id == primary_next_sat_id:
                continue
            if not sat_graph.has_edge(backup_next_sat_id, primary_next_sat_id):
                continue
            first_leg = sat_graph.edges[current_sat_id, backup_next_sat_id].get("weight", float("inf"))
            second_leg = sat_graph.edges[backup_next_sat_id, primary_next_sat_id].get(
                "weight", float("inf")
            )
            backup_candidates.append(
                (float(first_leg) + float(second_leg), int(backup_next_sat_id))
            )
        backup_sid_list.append(
            min(backup_candidates)[1] if backup_candidates else None
        )
    return backup_sid_list


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
        adjacency_sid_list = route_plan.get("adjacency_sid_list", [])
        planned_dst_sat_id = route_plan.get("planned_dst_sat_id")
        if planned_dst_sat_id is None:
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

        if not adjacency_sid_list:
            fstate[(src_sat_id, dst_gs_id)] = (-1, -1, -1)
            continue
        next_sat = adjacency_sid_list[0]
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
