from __future__ import annotations

from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology


def algorithm_segment_routing(
    time_since_epoch_ns: int,
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,
    gsl_attachment_strategy: GSLAttachmentStrategy,
    current_time: Time,
    list_gsl_interfaces_info: list,
    algorithm_params: dict,
) -> dict:
    """
    Segment routing over ISLs using limited segments (default 2 segments).

    Routing logic:
    - GS attaches to nearest satellite.
    - For each destination GS, pick destination satellite.
    - Compute segment plan: route across planes to dest plane, then within plane.
    - Forwarding uses local next-hop chosen by plane alignment or intra-plane distance.
    """
    segment_mode = algorithm_params.get("segment_mode", "plane_then_inplane")
    plane_weight = algorithm_params.get("plane_weight", 100.0)
    sat_weight = algorithm_params.get("sat_weight", 1.0)
    shell_weight = algorithm_params.get("shell_weight", 1000.0)
    if algorithm_params.get("segment_count") is None:
        algorithm_params["segment_count"] = 2

    bandwidth_state = _calculate_bandwidth_state(
        constellation_data, ground_stations, list_gsl_interfaces_info
    )
    fstate = _calculate_forwarding_state(
        topology_with_isls,
        ground_stations,
        gsl_attachment_strategy,
        current_time,
        constellation_data,
        segment_mode,
        plane_weight,
        sat_weight,
        shell_weight,
    )
    return {
        "fstate": fstate,
        "bandwidth": bandwidth_state,
    }


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


def _calculate_forwarding_state(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    gsl_attachment_strategy: GSLAttachmentStrategy,
    current_time: Time,
    constellation_data: ConstellationData,
    segment_mode: str,
    plane_weight: float,
    sat_weight: float,
    shell_weight: float,
) -> dict:
    gsl_attachments = gsl_attachment_strategy.select_attachments(
        topology_with_isls, ground_stations, current_time
    )

    ground_station_satellites_in_range = []
    for distance, sat_id in gsl_attachments:
        if sat_id != -1:
            ground_station_satellites_in_range.append([(distance, sat_id)])
        else:
            ground_station_satellites_in_range.append([])

    all_sat_ids = {sat.id for sat in topology_with_isls.get_satellites()}
    sat_node_ids = [node_id for node_id in topology_with_isls.graph.nodes() if node_id in all_sat_ids]
    sat_node_ids = sorted(sat_node_ids)

    fstate: dict[tuple[int, int], tuple[int, int, int]] = {}

    # Map GS destinations first (sat -> GS)
    for curr_sat_id in sat_node_ids:
        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_node_id = dst_gs.id
            possible_dst_sats = ground_station_satellites_in_range[gs_idx]
            if not possible_dst_sats:
                fstate[(curr_sat_id, dst_gs_node_id)] = (-1, -1, -1)
                continue

            _, dst_sat_id = min(possible_dst_sats, key=lambda item: item[0])
            if curr_sat_id == dst_sat_id:
                next_hop = _handle_direct_gs_path(
                    curr_sat_id, dst_gs_node_id, topology_with_isls
                )
                fstate[(curr_sat_id, dst_gs_node_id)] = next_hop
                continue

            next_hop_decision = _segment_next_hop(
                curr_sat_id,
                dst_sat_id,
                topology_with_isls,
                constellation_data,
                segment_mode,
                plane_weight,
                sat_weight,
                shell_weight,
            )
            fstate[(curr_sat_id, dst_gs_node_id)] = next_hop_decision

    # Map GS -> GS via entry sat (for churn metrics)
    for src_idx, src_gs in enumerate(ground_stations):
        src_gs_node_id = src_gs.id
        for dst_idx, dst_gs in enumerate(ground_stations):
            if src_gs_node_id == dst_gs.id:
                continue
            if src_idx >= len(ground_station_satellites_in_range):
                fstate[(src_gs_node_id, dst_gs.id)] = (-1, -1, -1)
                continue
            visible = ground_station_satellites_in_range[src_idx]
            if not visible:
                fstate[(src_gs_node_id, dst_gs.id)] = (-1, -1, -1)
                continue
            _, src_sat_id = min(visible, key=lambda item: item[0])
            next_hop = _handle_gs_to_sat_entry(src_sat_id, topology_with_isls)
            fstate[(src_gs_node_id, dst_gs.id)] = next_hop

    return fstate


def _segment_next_hop(
    curr_sat_id: int,
    dst_sat_id: int,
    topology_with_isls: LEOTopology,
    constellation_data: ConstellationData,
    segment_mode: str,
    plane_weight: float,
    sat_weight: float,
    shell_weight: float,
) -> tuple[int, int, int]:
    neighbors = list(topology_with_isls.graph.neighbors(curr_sat_id))
    if not neighbors:
        return (-1, -1, -1)

    curr_addr = TopologicalNetworkAddress.set_address_from_constellation(
        curr_sat_id,
        constellation_data.n_orbits,
        constellation_data.n_sats_per_orbit,
    )
    dst_addr = TopologicalNetworkAddress.set_address_from_constellation(
        dst_sat_id,
        constellation_data.n_orbits,
        constellation_data.n_sats_per_orbit,
    )

    best_neighbor = None
    best_score = float("inf")
    for neighbor_id in neighbors:
        neighbor_addr = TopologicalNetworkAddress.set_address_from_constellation(
            neighbor_id,
            constellation_data.n_orbits,
            constellation_data.n_sats_per_orbit,
        )
        score = _segment_distance(
            neighbor_addr,
            dst_addr,
            segment_mode,
            plane_weight,
            sat_weight,
            shell_weight,
            constellation_data.n_orbits,
            constellation_data.n_sats_per_orbit,
        )
        if score < best_score:
            best_score = score
            best_neighbor = neighbor_id

    if best_neighbor is None:
        return (-1, -1, -1)

    curr_score = _segment_distance(
        curr_addr,
        dst_addr,
        segment_mode,
        plane_weight,
        sat_weight,
        shell_weight,
        constellation_data.n_orbits,
        constellation_data.n_sats_per_orbit,
    )
    if best_score >= curr_score:
        return (-1, -1, -1)

    my_if = topology_with_isls.sat_neighbor_to_if.get((curr_sat_id, best_neighbor), -1)
    next_if = topology_with_isls.sat_neighbor_to_if.get((best_neighbor, curr_sat_id), -1)
    return (best_neighbor, my_if, next_if)


def _segment_distance(
    neighbor_addr: TopologicalNetworkAddress,
    dst_addr: TopologicalNetworkAddress,
    segment_mode: str,
    plane_weight: float,
    sat_weight: float,
    shell_weight: float,
    max_planes: int,
    max_sats_per_plane: int,
) -> float:
    if segment_mode != "plane_then_inplane":
        return neighbor_addr.topological_distance_to(dst_addr)

    # Distance encourages plane alignment first, then intra-plane movement.
    if neighbor_addr.shell_id != dst_addr.shell_id:
        return shell_weight + abs(neighbor_addr.shell_id - dst_addr.shell_id) * (shell_weight / 10.0)
    if neighbor_addr.plane_id != dst_addr.plane_id:
        plane_diff = abs(neighbor_addr.plane_id - dst_addr.plane_id)
        plane_wrap = max(1, max_planes - plane_diff)
        plane_dist = min(plane_diff, plane_wrap)
        return plane_weight + plane_dist * (plane_weight / 10.0)
    sat_diff = abs(neighbor_addr.sat_index - dst_addr.sat_index)
    sat_wrap = max(1, max_sats_per_plane - sat_diff)
    sat_dist = min(sat_diff, sat_wrap)
    return sat_weight + sat_dist * sat_weight


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


def _handle_gs_to_sat_entry(src_sat_id: int, topology_with_isls: LEOTopology) -> tuple[int, int, int]:
    try:
        src_satellite = topology_with_isls.get_satellite(src_sat_id)
        my_gsl_if = 0
        next_hop_gsl_if = src_satellite.number_isls
        return (src_sat_id, my_gsl_if, next_hop_gsl_if)
    except KeyError:
        return (-1, -1, -1)
