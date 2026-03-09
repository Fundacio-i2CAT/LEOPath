from __future__ import annotations

import networkx as nx
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology


def algorithm_traditional_segment_routing(
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,
    gsl_attachment_strategy: GSLAttachmentStrategy,
    current_time: Time,
    list_gsl_interfaces_info: list,
    algorithm_params: dict,
) -> dict:
    segment_count = int(algorithm_params.get("segment_count", 2))
    if segment_count < 1:
        segment_count = 1

    gsl_attachments = gsl_attachment_strategy.select_attachments(
        topology_with_isls, ground_stations, current_time
    )
    ground_station_satellites_in_range = []
    for distance, sat_id in gsl_attachments:
        if sat_id != -1:
            ground_station_satellites_in_range.append([(distance, sat_id)])
        else:
            ground_station_satellites_in_range.append([])

    segment_plans = _build_segment_plans(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        segment_count,
    )

    fstate = _materialize_fstate_from_segments(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        segment_plans,
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
    }


def _build_segment_plans(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
    segment_count: int,
) -> dict[tuple[int, int], list[int]]:
    graph = topology_with_isls.graph
    sat_ids = {sat.id for sat in topology_with_isls.get_satellites()}
    sat_graph = graph.subgraph(sorted(sat_ids))

    plans: dict[tuple[int, int], list[int]] = {}
    for curr_sat_id in sorted(sat_ids):
        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_id = dst_gs.id
            visible = ground_station_satellites_in_range[gs_idx]
            if not visible:
                plans[(curr_sat_id, dst_gs_id)] = []
                continue
            _, dst_sat_id = min(visible, key=lambda item: item[0])
            if curr_sat_id == dst_sat_id:
                plans[(curr_sat_id, dst_gs_id)] = [dst_sat_id]
                continue
            try:
                shortest_sat_path = nx.shortest_path(
                    sat_graph,
                    source=curr_sat_id,
                    target=dst_sat_id,
                    weight="weight",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                plans[(curr_sat_id, dst_gs_id)] = []
                continue
            plans[(curr_sat_id, dst_gs_id)] = _path_to_segments(
                shortest_sat_path,
                segment_count,
            )
    return plans


def _path_to_segments(shortest_path: list[int], segment_count: int) -> list[int]:
    # shortest_path includes source and destination satellites.
    if len(shortest_path) <= 1:
        return shortest_path[:]

    max_segments = max(1, segment_count)
    hop_count = len(shortest_path) - 1
    if hop_count <= max_segments:
        return shortest_path[1:]

    # Evenly sample waypoint boundaries along the shortest path.
    segments: list[int] = []
    for idx in range(1, max_segments + 1):
        pos = int(round(idx * hop_count / max_segments))
        if pos <= 0:
            pos = 1
        if pos > hop_count:
            pos = hop_count
        sid = shortest_path[pos]
        if not segments or segments[-1] != sid:
            segments.append(sid)

    if segments[-1] != shortest_path[-1]:
        segments[-1] = shortest_path[-1]
    return segments


def _materialize_fstate_from_segments(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
    segment_plans: dict[tuple[int, int], list[int]],
) -> dict[tuple[int, int], tuple[int, int, int]]:
    graph = topology_with_isls.graph
    sat_ids = {sat.id for sat in topology_with_isls.get_satellites()}
    sat_graph = graph.subgraph(sorted(sat_ids))

    fstate: dict[tuple[int, int], tuple[int, int, int]] = {}
    for curr_sat_id in sorted(sat_ids):
        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_id = dst_gs.id
            visible = ground_station_satellites_in_range[gs_idx]
            if not visible:
                fstate[(curr_sat_id, dst_gs_id)] = (-1, -1, -1)
                continue
            _, dst_sat_id = min(visible, key=lambda item: item[0])
            if curr_sat_id == dst_sat_id:
                fstate[(curr_sat_id, dst_gs_id)] = _handle_direct_gs_path(
                    dst_sat_id, dst_gs_id, topology_with_isls
                )
                continue

            segments = segment_plans.get((curr_sat_id, dst_gs_id), [])
            if not segments:
                fstate[(curr_sat_id, dst_gs_id)] = (-1, -1, -1)
                continue

            next_sid = segments[0]
            if next_sid == curr_sat_id:
                if len(segments) == 1:
                    next_sid = dst_sat_id
                else:
                    next_sid = segments[1]

            next_hop = _next_hop_towards_sid(
                curr_sat_id,
                next_sid,
                sat_graph,
                topology_with_isls,
            )
            fstate[(curr_sat_id, dst_gs_id)] = next_hop

    return fstate


def _next_hop_towards_sid(
    curr_sat_id: int,
    sid_sat_id: int,
    sat_graph: nx.Graph,
    topology_with_isls: LEOTopology,
) -> tuple[int, int, int]:
    if curr_sat_id == sid_sat_id:
        return (-1, -1, -1)
    try:
        sat_path = nx.shortest_path(
            sat_graph,
            source=curr_sat_id,
            target=sid_sat_id,
            weight="weight",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return (-1, -1, -1)

    if len(sat_path) < 2:
        return (-1, -1, -1)
    next_sat = sat_path[1]
    my_if = topology_with_isls.sat_neighbor_to_if.get((curr_sat_id, next_sat), -1)
    next_if = topology_with_isls.sat_neighbor_to_if.get((next_sat, curr_sat_id), -1)
    if my_if < 0 or next_if < 0:
        return (-1, -1, -1)
    return (next_sat, my_if, next_if)


def _add_gs_to_gs_fstate(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
    fstate: dict,
) -> None:
    for src_idx, src_gs in enumerate(ground_stations):
        src_gs_node_id = src_gs.id
        if src_idx >= len(ground_station_satellites_in_range):
            continue
        visible = ground_station_satellites_in_range[src_idx]
        if not visible:
            continue
        _, src_sat_id = min(visible, key=lambda item: item[0])
        try:
            src_satellite = topology_with_isls.get_satellite(src_sat_id)
            my_gsl_if = 0
            next_hop_gsl_if = src_satellite.number_isls
            next_hop = (src_sat_id, my_gsl_if, next_hop_gsl_if)
        except KeyError:
            continue
        for dst_gs in ground_stations:
            if dst_gs.id == src_gs_node_id:
                continue
            fstate[(src_gs_node_id, dst_gs.id)] = next_hop


def _handle_direct_gs_path(
    dst_sat_id: int, dst_gs_node_id: int, topology_with_isls: LEOTopology
) -> tuple[int, int, int]:
    try:
        dst_satellite = topology_with_isls.get_satellite(dst_sat_id)
        my_gsl_if = dst_satellite.number_isls
        next_hop_gsl_if = 0
        return (dst_gs_node_id, my_gsl_if, next_hop_gsl_if)
    except KeyError:
        return (-1, -1, -1)


def _calculate_bandwidth_state(
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    list_gsl_interfaces_info: list,
) -> dict:
    num_satellites = constellation_data.number_of_satellites
    num_total_nodes = num_satellites + len(ground_stations)
    bandwidth_state = {}
    for i in range(num_total_nodes):
        if i < len(list_gsl_interfaces_info):
            node_info = list_gsl_interfaces_info[i]
            node_id = node_info.get("id", i)
            bandwidth = node_info.get("aggregate_max_bandwidth", 0.0)
        else:
            node_id = i
            bandwidth = 0.0
        bandwidth_state[node_id] = bandwidth
    return bandwidth_state
