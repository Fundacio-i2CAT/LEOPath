"""
Topological routing algorithm implementation that provides bandwidth and forwarding state calculation
using topological network addresses for LEO satellite networks.
"""

from astropy.time import Time

from leopath import logger
from leopath.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

from .fstate_calculation import calculate_fstate_topological_routing_no_gs_relay

log = logger.get_logger(__name__)


def algorithm_topological_routing(
    time_since_epoch_ns: int,
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,
    gsl_attachment_strategy: GSLAttachmentStrategy,
    current_time: Time,
    list_gsl_interfaces_info: list,
    prev_fstate: dict | None = None,
    graph_has_changed: bool = True,
) -> dict:
    """
    Calculates bandwidth and forwarding state using topological routing over ISLs only (no GS relaying).

    This algorithm implements topological addressing with 6GRUPA addresses and uses neighbor-based
    forwarding tables combined with shortest-path routing for satellite-to-ground-station communication.

    Args:
        time_since_epoch_ns: Current time step relative to epoch (integer ns).
        constellation_data: Holds satellite list, counts, max lengths, epoch string.
        ground_stations: List of GroundStation objects.
        topology_with_isls: LEOTopology object containing the graph with ISL links calculated.
        gsl_attachment_strategy: Strategy for selecting which satellites are visible to each ground station.
        current_time: Current simulation time for satellite positioning.
        list_gsl_interfaces_info: List of dicts, one per sat/GS, with bandwidth info.
        prev_fstate: Previous forwarding state (for optimization)
        graph_has_changed: Whether topology graph has changed since last computation

    Returns:
        Dictionary containing the new 'fstate' and 'bandwidth' state objects.
    """
    log.debug(f"Running topological routing algorithm for t={time_since_epoch_ns} ns")

    # Calculate bandwidth state (same as shortest path algorithm)
    bandwidth_state = _calculate_bandwidth_state(
        constellation_data, ground_stations, list_gsl_interfaces_info
    )

    # Calculate GSL attachment (satellites visible to each ground station)
    gsl_attachments = gsl_attachment_strategy.select_attachments(
        topology_with_isls, ground_stations, current_time
    )

    # Convert single attachments to the expected format for compatibility
    ground_station_satellites_in_range = []
    for gs_idx, (distance, sat_id) in enumerate(gsl_attachments):
        if sat_id != -1:  # Valid attachment
            ground_station_satellites_in_range.append([(distance, sat_id)])
        else:  # No attachment found
            ground_station_satellites_in_range.append([])
            log.warning(f"Ground station {gs_idx} has no satellite attachment")

    # Calculate forwarding state using topological routing
    fstate = calculate_fstate_topological_routing_no_gs_relay(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        constellation_data,
        time_since_epoch_ns,
        prev_fstate,
        graph_has_changed,
    )

    # Add GS -> GS entries (used by churn/stretches in evaluation)
    _add_gs_to_gs_fstate(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        fstate,
    )

    # Also add GS -> GS entries (used by churn/stretches in evaluation)
    _add_gs_to_gs_fstate(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        fstate,
    )

    return {
        "fstate": fstate,
        "bandwidth": bandwidth_state,
    }


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


def _calculate_bandwidth_state(
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    list_gsl_interfaces_info: list,
) -> dict:
    """
    Returns a dict mapping node_id to its aggregate_max_bandwidth.
    This is identical to the shortest path algorithm's bandwidth calculation.
    """
    num_satellites = constellation_data.number_of_satellites
    num_total_nodes = num_satellites + len(ground_stations)
    bandwidth_state = {}

    if len(list_gsl_interfaces_info) != num_total_nodes:
        log.warning(
            f"Length mismatch: list_gsl_interfaces_info ({len(list_gsl_interfaces_info)}) "
            f"vs total nodes ({num_total_nodes}). Bandwidth state might be incomplete."
        )

    for i in range(num_total_nodes):
        if i < len(list_gsl_interfaces_info):
            node_info = list_gsl_interfaces_info[i]
            node_id = node_info.get("id", i)
            bandwidth = node_info.get("aggregate_max_bandwidth", 0.0)
        else:
            node_id = i
            bandwidth = 0.0
            log.error(
                f"Index {i} out of bounds for list_gsl_interfaces_info, setting BW=0 for node {node_id}"
            )
        bandwidth_state[node_id] = bandwidth
        log.debug(f"  Bandwidth state: Node {node_id}, IF 0, BW = {bandwidth}")

    log.debug(f"  Calculated bandwidth state for {len(bandwidth_state)} nodes.")
    return bandwidth_state
