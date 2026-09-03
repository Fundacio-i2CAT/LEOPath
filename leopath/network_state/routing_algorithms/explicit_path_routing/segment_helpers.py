"""Forwarding-state helpers shared by the explicit-path algorithm.

These four routines were factored out of the traditional segment-routing
module when that legacy baseline was removed. They are generic enough to
be used by any explicit-path family member: splitting a shortest path into
a segment list, installing ground-station-to-ground-station entries, the
final ground-station delivery hop, and the per-node bandwidth table.
"""

from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology


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
