from __future__ import annotations

import ipaddress
import math

import networkx as nx
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology


DEFAULT_SRV6_LOCATOR_PREFIX = "fd00:10:0:1::/64"


def algorithm_traditional_segment_routing(
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,
    gsl_attachment_strategy: GSLAttachmentStrategy,
    current_time: Time,
    list_gsl_interfaces_info: list,
    algorithm_params: dict,
    segment_plans: dict[tuple[int, int], list[str]] | None = None,
    planning_ground_station_satellites_in_range: list | None = None,
    current_ground_station_satellites_in_range: list | None = None,
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

    planning_visibility = (
        planning_ground_station_satellites_in_range or ground_station_satellites_in_range
    )
    current_visibility = (
        current_ground_station_satellites_in_range or ground_station_satellites_in_range
    )

    if segment_plans is None:
        segment_plans = _build_segment_plans(
            topology_with_isls,
            ground_stations,
            planning_visibility,
            segment_count,
            algorithm_params.get("srv6_locator_prefix", DEFAULT_SRV6_LOCATOR_PREFIX),
        )

    fstate = _materialize_fstate_from_segments(
        topology_with_isls,
        ground_stations,
        planning_visibility,
        segment_plans,
        current_visibility,
        algorithm_params.get("srv6_locator_prefix", DEFAULT_SRV6_LOCATOR_PREFIX),
    )

    _add_gs_to_gs_fstate(
        topology_with_isls,
        ground_stations,
        current_visibility,
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
    srv6_locator_prefix: str = DEFAULT_SRV6_LOCATOR_PREFIX,
) -> dict[tuple[int, int], list[str]]:
    sat_ids = {sat.id for sat in topology_with_isls.get_satellites()}
    sorted_sat_ids = sorted(sat_ids)

    plans: dict[tuple[int, int], list[str]] = {}
    for curr_sat_id in sorted_sat_ids:
        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_id = dst_gs.id
            visible = ground_station_satellites_in_range[gs_idx]
            if not visible:
                plans[(curr_sat_id, dst_gs_id)] = []
                continue
            _, dst_sat_id = min(visible, key=lambda item: item[0])
            plans[(curr_sat_id, dst_gs_id)] = [
                satellite_srv6_sid(dst_sat_id, srv6_locator_prefix)
            ]
    return plans


def satellite_srv6_sid(
    satellite_id: int,
    srv6_locator_prefix: str = DEFAULT_SRV6_LOCATOR_PREFIX,
) -> str:
    """Return a stable SRv6 endpoint SID for a satellite node."""
    if satellite_id < 0:
        raise ValueError(f"satellite_id must be non-negative, got {satellite_id}")
    network = ipaddress.IPv6Network(srv6_locator_prefix, strict=False)
    host_bits = 128 - network.prefixlen
    node_value = satellite_id + 1
    if node_value >= (1 << host_bits):
        raise ValueError(
            f"satellite_id {satellite_id} does not fit in locator {srv6_locator_prefix}"
        )
    return str(ipaddress.IPv6Address(int(network.network_address) + node_value))


def satellite_id_from_srv6_sid(
    sid: str,
    srv6_locator_prefix: str = DEFAULT_SRV6_LOCATOR_PREFIX,
) -> int | None:
    """Decode a simulator satellite id from a SID in the configured SRv6 locator."""
    try:
        network = ipaddress.IPv6Network(srv6_locator_prefix, strict=False)
        address = ipaddress.IPv6Address(sid)
    except ValueError:
        return None
    if address not in network:
        return None
    offset = int(address) - int(network.network_address)
    if offset <= 0:
        return None
    return offset - 1


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
    planning_ground_station_satellites_in_range: list,
    segment_plans: dict[tuple[int, int], list[str]],
    current_ground_station_satellites_in_range: list | None = None,
    srv6_locator_prefix: str = DEFAULT_SRV6_LOCATOR_PREFIX,
) -> dict[tuple[int, int], tuple[int, int, int]]:
    graph = topology_with_isls.graph
    sat_ids = {sat.id for sat in topology_with_isls.get_satellites()}
    sat_graph = graph.subgraph(sorted(sat_ids))
    next_hops = _build_next_hop_lookup(
        sat_graph,
        topology_with_isls,
        _collect_segment_endpoint_ids(
            ground_stations,
            planning_ground_station_satellites_in_range,
            segment_plans,
            srv6_locator_prefix,
        ),
    )

    fstate: dict[tuple[int, int], tuple[int, int, int]] = {}
    for curr_sat_id in sorted(sat_ids):
        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_id = dst_gs.id
            visible = planning_ground_station_satellites_in_range[gs_idx]
            if not visible:
                fstate[(curr_sat_id, dst_gs_id)] = (-1, -1, -1)
                continue
            _, dst_sat_id = min(visible, key=lambda item: item[0])
            if curr_sat_id == dst_sat_id:
                if _is_ground_station_currently_visible(
                    gs_idx,
                    dst_sat_id,
                    current_ground_station_satellites_in_range,
                ):
                    fstate[(curr_sat_id, dst_gs_id)] = _handle_direct_gs_path(
                        dst_sat_id, dst_gs_id, topology_with_isls
                    )
                else:
                    fstate[(curr_sat_id, dst_gs_id)] = (-1, -1, -1)
                continue

            segments = segment_plans.get((curr_sat_id, dst_gs_id), [])
            if not segments:
                fstate[(curr_sat_id, dst_gs_id)] = (-1, -1, -1)
                continue

            next_sid = satellite_id_from_srv6_sid(segments[0], srv6_locator_prefix)
            if next_sid is None:
                fstate[(curr_sat_id, dst_gs_id)] = (-1, -1, -1)
                continue
            if next_sid == curr_sat_id:
                if len(segments) == 1:
                    next_sid = dst_sat_id
                else:
                    next_sid = satellite_id_from_srv6_sid(
                        segments[1],
                        srv6_locator_prefix,
                    )
                    if next_sid is None:
                        fstate[(curr_sat_id, dst_gs_id)] = (-1, -1, -1)
                        continue

            next_hop = next_hops.get((curr_sat_id, next_sid), (-1, -1, -1))
            fstate[(curr_sat_id, dst_gs_id)] = next_hop

    return fstate


def _build_next_hop_lookup(
    sat_graph: nx.Graph,
    topology_with_isls: LEOTopology,
    target_sat_ids: set[int],
) -> dict[tuple[int, int], tuple[int, int, int]]:
    nodelist = sorted(sat_graph.nodes())
    node_to_index = {node_id: index for index, node_id in enumerate(nodelist)}
    if not nodelist or not target_sat_ids:
        return {}

    dist_matrix = nx.floyd_warshall_numpy(
        sat_graph,
        nodelist=nodelist,
        weight="weight",
    )
    next_hops: dict[tuple[int, int], tuple[int, int, int]] = {}
    for source in nodelist:
        source_idx = node_to_index[source]
        for destination in target_sat_ids:
            destination_idx = node_to_index.get(destination)
            if source == destination or destination_idx is None:
                continue
            if math.isinf(float(dist_matrix[source_idx, destination_idx])):
                continue
            next_sat = _select_next_hop_from_distance_matrix(
                source,
                destination_idx,
                sat_graph,
                node_to_index,
                dist_matrix,
            )
            if next_sat is None:
                continue
            my_if = topology_with_isls.sat_neighbor_to_if.get((source, next_sat), -1)
            next_if = topology_with_isls.sat_neighbor_to_if.get((next_sat, source), -1)
            if my_if >= 0 and next_if >= 0:
                next_hops[(source, destination)] = (next_sat, my_if, next_if)
    return next_hops


def _select_next_hop_from_distance_matrix(
    source: int,
    destination_idx: int,
    sat_graph: nx.Graph,
    node_to_index: dict[int, int],
    dist_matrix,
) -> int | None:
    best_neighbor = None
    best_distance = float("inf")
    for neighbor in sat_graph.neighbors(source):
        neighbor_idx = node_to_index.get(neighbor)
        if neighbor_idx is None:
            continue
        dist_neighbor_to_destination = float(dist_matrix[neighbor_idx, destination_idx])
        if math.isinf(dist_neighbor_to_destination):
            continue
        distance = sat_graph.edges[source, neighbor].get("weight", 1.0) + dist_neighbor_to_destination
        if distance < best_distance:
            best_distance = distance
            best_neighbor = neighbor
    return best_neighbor


def _collect_segment_endpoint_ids(
    ground_stations: list[GroundStation],
    planning_ground_station_satellites_in_range: list,
    segment_plans: dict[tuple[int, int], list[str]],
    srv6_locator_prefix: str,
) -> set[int]:
    endpoint_ids: set[int] = set()
    for gs_idx, _ in enumerate(ground_stations):
        if gs_idx >= len(planning_ground_station_satellites_in_range):
            continue
        visible = planning_ground_station_satellites_in_range[gs_idx]
        if visible:
            _, dst_sat_id = min(visible, key=lambda item: item[0])
            endpoint_ids.add(dst_sat_id)
    for segments in segment_plans.values():
        for sid in segments:
            sat_id = satellite_id_from_srv6_sid(sid, srv6_locator_prefix)
            if sat_id is not None:
                endpoint_ids.add(sat_id)
    return endpoint_ids


def _is_ground_station_currently_visible(
    gs_idx: int,
    dst_sat_id: int,
    current_ground_station_satellites_in_range: list | None,
) -> bool:
    if current_ground_station_satellites_in_range is None:
        return True
    if gs_idx >= len(current_ground_station_satellites_in_range):
        return False
    visible = current_ground_station_satellites_in_range[gs_idx]
    return any(sat_id == dst_sat_id for _, sat_id in visible)


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
