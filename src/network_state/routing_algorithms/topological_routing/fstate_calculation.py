import networkx as nx
import numpy as np

from src import logger
from src.topology.topology import GroundStation, LEOTopology, ConstellationData
from src.topology.satellite.topological_network_address import TopologicalNetworkAddress

log = logger.get_logger(__name__)


def algorithm_free_one_only_over_isls_topological(
    time_since_epoch_ns: int,
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,
    ground_station_satellites_in_range: list,
    list_gsl_interfaces_info: list,
    prev_output: dict | None = None,
    enable_verbose_logs: bool = False,
) -> dict:
    """
    Calculates bandwidth and forwarding state using topological routing (ISLs only, no GS relaying).

    This is the main entry point for the topological routing algorithm that integrates
    with the simulation framework.

    Args:
        time_since_epoch_ns: Current time step relative to epoch (integer ns)
        constellation_data: Holds satellite list, counts, max lengths, epoch string
        ground_stations: List of GroundStation objects
        topology_with_isls: LEOTopology object containing the graph with ISL links
        ground_station_satellites_in_range: List where index=gs_idx, value=list of (distance, sat_id) tuples
        list_gsl_interfaces_info: List of dicts, one per sat/GS, with bandwidth info
        prev_output: Dictionary containing 'fstate' and 'bandwidth' objects from the previous step
        enable_verbose_logs: Boolean to enable detailed logging

    Returns:
        Dictionary containing the new 'fstate' and 'bandwidth' state objects
    """
    log.debug(
        f"Running algorithm_free_one_only_over_isls_topological for t={time_since_epoch_ns} ns"
    )

    # Calculate bandwidth state (same as shortest path algorithm)
    bandwidth_state = _calculate_bandwidth_state(
        constellation_data, ground_stations, list_gsl_interfaces_info
    )

    # Check if graph has changed by comparing with previous state
    graph_has_changed = True
    prev_fstate = None
    if prev_output is not None:
        prev_fstate = prev_output.get("fstate")
        # For now, assume graph has changed unless we implement proper change detection
        # In a full implementation, you'd compare topology graphs here
        graph_has_changed = True

    # Calculate forwarding state using topological routing
    fstate = calculate_fstate_topological_routing_no_gs_relay(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        time_since_epoch_ns,
        prev_fstate,
        graph_has_changed,
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
    """
    Returns a dict mapping node_id to its aggregate_max_bandwidth.
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


def calculate_fstate_topological_routing_no_gs_relay(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
    time_since_epoch_ns: int = 0,
    prev_fstate: dict | None = None,
    graph_has_changed: bool = True,
) -> dict:
    """
    Calculates forwarding state using topological routing over ISLs only (no GS relays).

    Implements the topological routing algorithm with the following steps:
    1. At t=0: Set 6GRUPA addresses to all nodes and initialize forwarding tables
    2. If graph hasn't changed, reuse previous state
    3. Handle ground station link changes with renumbering
    4. Calculate satellite-to-GS routing decisions

    Args:
        topology_with_isls: Network topology with ISL links
        ground_stations: List of ground stations
        ground_station_satellites_in_range: List where index=gs_idx, value=list of (distance, sat_id) tuples
        time_since_epoch_ns: Time since epoch in nanoseconds
        prev_fstate: Previous forwarding state (for state comparison)
        graph_has_changed: Whether the topology graph has changed since last computation

    Returns:
        Dictionary containing forwarding state
    """
    log.debug("Calculating topological routing fstate object (no GS relay)")

    full_graph = topology_with_isls.graph

    try:
        all_satellite_ids = {sat.id for sat in topology_with_isls.get_satellites()}
    except Exception as e:
        log.exception(f"Error getting satellite IDs from topology: {e}")
        return {}

    satellite_node_ids = sorted(
        [node_id for node_id in full_graph.nodes() if node_id in all_satellite_ids]
    )

    if not satellite_node_ids:
        log.warning("No valid satellite nodes found in the graph for path calculation.")
        return {}

    satellite_only_subgraph = full_graph.subgraph(satellite_node_ids)

    if satellite_only_subgraph.number_of_nodes() == 0:
        log.warning("Satellite-only subgraph is empty. No ISL paths possible.")
        return {}

    # Step 1: Initialize at t=0 - set 6GRUPA addresses and neighbor forwarding tables
    if time_since_epoch_ns == 0:
        log.debug("t=0: Setting 6GRUPA addresses to all nodes and initializing forwarding tables")
        _set_sixgrupa_addresses_to_all_nodes(topology_with_isls)
        _fill_forwarding_tables_in_every_satellite(
            satellite_node_ids, satellite_only_subgraph, topology_with_isls
        )
        graph_has_changed = True  # Force recalculation on first run

    # Step 2: Check if we can reuse previous state
    if not graph_has_changed and prev_fstate is not None:
        log.debug("Graph unchanged, reusing previous fstate")
        return prev_fstate

    # Step 3: Handle ground station link changes (renumbering if needed)
    for gs_idx, gs in enumerate(ground_stations):
        if gs_idx < len(ground_station_satellites_in_range):
            gsl_satellites = ground_station_satellites_in_range[gs_idx]
            if gsl_satellites:  # GSL has changed
                log.debug(f"GSL changed for GS {gs.id}, performing renumbering")
                _perform_renumbering_for_gs(gs, gsl_satellites, topology_with_isls)

    # Step 4: Calculate satellite-to-GS forwarding state
    fstate: dict[tuple, tuple] = {}
    dist_satellite_to_ground_station: dict[tuple, float] = {}

    node_to_index = {node_id: index for index, node_id in enumerate(satellite_node_ids)}

    _calculate_sat_to_gs_fstate(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        satellite_node_ids,
        node_to_index,
        satellite_only_subgraph,
        topology_with_isls.sat_neighbor_to_if,
        dist_satellite_to_ground_station,
        fstate,
    )

    log.debug(f"Calculated fstate with {len(fstate)} entries")
    return fstate


def _set_sixgrupa_addresses_to_all_nodes(topology: LEOTopology):
    """
    Set 6GRUPA addresses to all satellite nodes in the topology.
    """
    log.debug("Setting 6GRUPA addresses to all satellite nodes")
    for satellite in topology.get_satellites():
        try:
            address = TopologicalNetworkAddress.from_6grupa(satellite.id)
            satellite.sixgrupa_addr = address
            log.debug(f"Assigned 6G-RUPA address {address} to satellite {satellite.id}")
        except Exception as e:
            log.error(f"Failed to assign 6G-RUPA address to satellite {satellite.id}: {e}")


def _perform_renumbering_for_gs(gs: GroundStation, gsl_satellites: list, topology: LEOTopology):
    """
    Perform renumbering when a ground station's satellite links change.

    This is a placeholder for the renumbering logic. In a full implementation,
    this would handle address reassignment when GSL topology changes.
    """
    log.debug(
        f"Renumbering for GS {gs.id} with satellites: {[sat_id for _, sat_id in gsl_satellites]}"
    )
    # TODO: Implement actual renumbering logic if needed
    # For now, this is a no-op as the basic algorithm doesn't require complex renumbering


def _fill_forwarding_tables_in_every_satellite(
    satellite_node_ids: list[int],
    satellite_only_subgraph: nx.Graph,
    topology_with_isls: LEOTopology,
):
    for satellite_id in satellite_node_ids:
        for neighbor_id in satellite_only_subgraph.neighbors(satellite_id):
            interface = topology_with_isls.sat_neighbor_to_if.get((satellite_id, neighbor_id))
            log.debug(f"Satellite {satellite_id} -> Neighbor {neighbor_id}: {interface}")
            satellite = topology_with_isls.get_satellite(satellite_id)
            neighbor_address = satellite.get_6grupa_addr_from(neighbor_id)
            if neighbor_address and interface is not None:
                satellite.forwarding_table[neighbor_address.to_integer()] = interface
                log.debug(
                    f"Forwarding entry added for satellite {satellite_id} to neighbor {neighbor_id}: address {neighbor_address}, interface {interface}"
                )


def _calculate_sat_to_gs_fstate(
    topology_with_isls,
    ground_stations,
    ground_station_satellites_in_range,
    nodelist,
    node_to_index,
    sat_subgraph,
    sat_neighbor_to_if,
    dist_satellite_to_ground_station,
    fstate,
):
    """
    Calculate satellite-to-ground-station forwarding state using topological routing.

    This implements the core routing logic:
    - For each satellite, determine next hop to reach each ground station
    - Use direct paths when available (single hop via GSL)
    - Use multi-hop ISL paths when direct connection not available
    """
    log.debug("Calculating satellite-to-GS forwarding state")

    # Calculate distance matrix for shortest paths if needed for multi-hop routing
    try:
        log.debug(f"Calculating shortest paths on satellite subgraph for {len(nodelist)} nodes...")
        dist_matrix = nx.floyd_warshall_numpy(sat_subgraph, nodelist=nodelist, weight="weight")
        log.debug("Shortest path calculation complete.")
    except (nx.NetworkXError, Exception) as e:
        log.error(f"Error during shortest path calculation: {e}")
        # Use simplified routing without distance matrix
        dist_matrix = None

    for curr_sat_id in nodelist:
        curr_sat_idx = node_to_index[curr_sat_id]
        try:
            curr_satellite = topology_with_isls.get_satellite(curr_sat_id)
        except KeyError:
            log.error(f"Could not find satellite object {curr_sat_id}")
            continue

        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_node_id = dst_gs.id
            if gs_idx >= len(ground_station_satellites_in_range):
                continue

            possible_dst_sats = ground_station_satellites_in_range[gs_idx]
            if possible_dst_sats:
                log.debug(
                    f"FSTATE: Sat {curr_sat_id} -> GS {dst_gs.id}. Visible sats: {[sat_id for _, sat_id in possible_dst_sats]}"
                )

            # Get possible paths to this ground station
            possibilities = _get_satellite_possibilities(
                possible_dst_sats, curr_sat_idx, node_to_index, dist_matrix
            )

            # Determine next hop decision
            next_hop_decision, distance_to_ground_station_m = _get_next_hop_decision(
                possibilities,
                curr_sat_id,
                curr_satellite,
                sat_subgraph,
                sat_neighbor_to_if,
                topology_with_isls,
                dst_gs_node_id,
            )

            if next_hop_decision is not None:
                dist_satellite_to_ground_station[(curr_sat_id, dst_gs_node_id)] = (
                    distance_to_ground_station_m
                )
                fstate[(curr_sat_id, dst_gs_node_id)] = next_hop_decision
                log.debug(
                    f"Fstate entry: Sat {curr_sat_id} -> GS {dst_gs_node_id} via {next_hop_decision}"
                )


def _get_satellite_possibilities(possible_dst_sats, curr_sat_idx, node_to_index, dist_matrix):
    """
    Get list of possible destination satellites for reaching a ground station.

    Returns list of (total_distance, visible_sat_id) tuples sorted by distance.
    """
    possibilities = []

    if not possible_dst_sats:
        return possibilities

    for visibility_info in possible_dst_sats:
        dist_gs_to_sat_m, visible_sat_id = visibility_info
        visible_sat_idx = node_to_index.get(visible_sat_id)

        if visible_sat_idx is not None:
            if dist_matrix is not None:
                # Use shortest path distance if available
                dist_curr_to_visible_sat = dist_matrix[curr_sat_idx, visible_sat_idx]
                if not np.isinf(dist_curr_to_visible_sat):
                    total_dist = dist_curr_to_visible_sat + dist_gs_to_sat_m
                    possibilities.append((total_dist, visible_sat_id))
            else:
                # Fallback: use GSL distance only (direct connection preferred)
                if curr_sat_idx == visible_sat_idx:
                    # Direct connection - use GSL distance
                    possibilities.append((dist_gs_to_sat_m, visible_sat_id))
                # For non-direct connections without distance matrix, skip

    possibilities.sort()  # Sort by total distance
    return possibilities


def _get_next_hop_decision(
    possibilities,
    curr_sat_id,
    curr_satellite,
    sat_subgraph,
    sat_neighbor_to_if,
    topology_with_isls,
    dst_gs_node_id,
):
    """
    Determine the next hop decision for reaching a ground station.

    Implements the routing logic:
    1. Try direct GSL connection if available
    2. Use multi-hop ISL path to reach a satellite with GSL connection

    Returns:
        tuple: (next_hop_interface_or_decision, distance) or (None, inf) if no path
    """
    if not possibilities:
        log.debug(f"No possibilities for satellite {curr_sat_id} to reach GS {dst_gs_node_id}")
        return None, float("inf")

    # Get the best (closest) destination satellite
    best_distance, best_dst_sat_id = possibilities[0]

    # Check if this is a direct connection (same satellite)
    if curr_sat_id == best_dst_sat_id:
        # Direct GSL connection - use special interface designation
        log.debug(f"Direct GSL path: Sat {curr_sat_id} -> GS {dst_gs_node_id}")
        return ("GSL", dst_gs_node_id), best_distance

    # Multi-hop path needed - find next hop neighbor
    try:
        if sat_subgraph.has_node(curr_sat_id) and sat_subgraph.has_node(best_dst_sat_id):
            # Use NetworkX to find shortest path
            path = nx.shortest_path(sat_subgraph, curr_sat_id, best_dst_sat_id, weight="weight")
            if len(path) > 1:
                next_hop_sat_id = path[1]  # Next satellite in path

                # Get the interface to the next hop
                interface = sat_neighbor_to_if.get((curr_sat_id, next_hop_sat_id))
                if interface is not None:
                    log.debug(
                        f"Multi-hop ISL path: Sat {curr_sat_id} -> {next_hop_sat_id} (if {interface}) -> ... -> Sat {best_dst_sat_id} -> GS {dst_gs_node_id}"
                    )
                    return interface, best_distance
                else:
                    log.warning(f"No interface found for ISL {curr_sat_id} -> {next_hop_sat_id}")
            else:
                log.warning(f"Path too short: {path}")
        else:
            log.warning(f"Nodes not found in subgraph: {curr_sat_id}, {best_dst_sat_id}")

    except (nx.NetworkXNoPath, nx.NetworkXError) as e:
        log.warning(
            f"No path found from satellite {curr_sat_id} to satellite {best_dst_sat_id}: {e}"
        )

    return None, float("inf")
