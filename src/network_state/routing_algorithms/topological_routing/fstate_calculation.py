import networkx as nx
from typing import Optional

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
        # Also assign GS addresses for initial GSL attachments
        for gs_idx, gs in enumerate(ground_stations):
            curr_sat_id = None
            if gs_idx < len(ground_station_satellites_in_range):
                satellites = ground_station_satellites_in_range[gs_idx]
                if satellites:
                    _, curr_sat_id = satellites[0]
            if curr_sat_id is not None:
                _perform_renumbering_for_gs(gs, None, curr_sat_id, topology_with_isls)
        graph_has_changed = True  # Force recalculation on first run

    # Step 2: Check if we can reuse previous state
    if not graph_has_changed and prev_fstate is not None:
        log.debug("Graph unchanged, reusing previous fstate")
        return prev_fstate

    # Step 3: Handle ground station link changes (renumbering if needed)
    gsl_changes = _detect_gsl_changes(ground_stations, ground_station_satellites_in_range)
    for gs_idx, (prev_sat_id, curr_sat_id) in gsl_changes.items():
        gs = ground_stations[gs_idx]
        log.debug(f"GSL changed for GS {gs.id}: {prev_sat_id} -> {curr_sat_id}")
        _perform_renumbering_for_gs(gs, prev_sat_id, curr_sat_id, topology_with_isls)

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
            address = TopologicalNetworkAddress.set_address_from_orbital_parameters(satellite.id)
            satellite.sixgrupa_addr = address
            log.debug(f"Assigned 6G-RUPA address {address} to satellite {satellite.id}")
        except Exception as e:
            log.error(f"Failed to assign 6G-RUPA address to satellite {satellite.id}: {e}")


def _detect_gsl_changes(
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
) -> dict[int, tuple[Optional[int], Optional[int]]]:
    """
    Detect GSL changes by comparing current attachments with previous ones.

    Args:
        ground_stations: List of ground stations
        ground_station_satellites_in_range: Current satellite attachments per GS

    Returns:
        dict: Maps GS index to (previous_sat_id, new_sat_id) for changed GSLs
    """
    gsl_changes = {}

    for gs_idx, gs in enumerate(ground_stations):
        if gs_idx >= len(ground_station_satellites_in_range):
            continue

        gsl_satellites = ground_station_satellites_in_range[gs_idx]
        current_sat_id = None

        # Get the best (closest) satellite currently attached to this GS
        if gsl_satellites:
            # Sort by distance and take the closest one
            sorted_sats = sorted(gsl_satellites, key=lambda x: x[0])
            if sorted_sats:
                current_sat_id = sorted_sats[0][1]  # (distance, sat_id)

        # Check if attachment has changed
        previous_sat_id = gs.previous_attached_satellite_id

        if previous_sat_id != current_sat_id:
            log.debug(f"GSL change detected for GS {gs.id}: {previous_sat_id} -> {current_sat_id}")
            gsl_changes[gs_idx] = (previous_sat_id, current_sat_id)

            # Update the stored previous satellite ID
            gs.previous_attached_satellite_id = current_sat_id

    return gsl_changes


def _assign_gs_address_from_satellite(
    gs: GroundStation, satellite_id: int, gs_subnet_index: int, topology: LEOTopology
) -> Optional[TopologicalNetworkAddress]:
    """
    Assign a 6grupa address to a ground station based on its attached satellite.

    Args:
        gs: The ground station
        satellite_id: ID of the satellite this GS is attached to
        gs_subnet_index: Subnet index for this GS under the satellite
        topology: Topology containing satellites

    Returns:
        TopologicalNetworkAddress for the ground station
    """
    try:
        # Get the satellite's 6grupa address
        satellite = topology.get_satellite(satellite_id)
        if not hasattr(satellite, "sixgrupa_addr") or not satellite.sixgrupa_addr:
            # If satellite doesn't have an address, create one
            sat_address = TopologicalNetworkAddress.set_address_from_orbital_parameters(
                satellite_id
            )
            satellite.sixgrupa_addr = sat_address
        else:
            sat_address = satellite.sixgrupa_addr

        # Create GS address with same shell, plane, sat_index but different subnet_index
        gs_address = TopologicalNetworkAddress(
            shell_id=sat_address.shell_id,
            plane_id=sat_address.plane_id,
            sat_index=sat_address.sat_index,
            subnet_index=gs_subnet_index,
        )

        log.debug(
            f"Assigned 6grupa address {gs_address} to GS {gs.id} "
            f"(attached to satellite {satellite_id})"
        )

        return gs_address

    except Exception as e:
        log.error(f"Failed to assign 6grupa address to GS {gs.id}: {e}")
        return None


def _perform_renumbering_for_gs(
    gs: GroundStation, prev_sat_id: Optional[int], curr_sat_id: Optional[int], topology: LEOTopology
):
    """
    Perform renumbering when a ground station's satellite links change.

    Updates the GS's 6grupa address to match the new satellite attachment.
    """
    log.debug(f"Renumbering for GS {gs.id} from satellite {prev_sat_id} to {curr_sat_id}")

    if curr_sat_id is not None:
        # Get the satellite's 6grupa address to match coordinates
        try:
            satellite = topology.get_satellite(curr_sat_id)
            if satellite.sixgrupa_addr is None:
                satellite.sixgrupa_addr = (
                    TopologicalNetworkAddress.set_address_from_orbital_parameters(curr_sat_id)
                )
            sat_addr = satellite.sixgrupa_addr
        except Exception:
            log.error(f"Failed to get satellite {curr_sat_id} address for GS {gs.id} renumbering")
            return

        # Count how many GSs are already attached to this satellite to assign unique subnet_index
        # Look for GSs that have 6grupa addresses matching this satellite's coordinates
        used_subnet_indices = set()
        for other_gs in topology.get_ground_stations():
            if (
                other_gs != gs
                and hasattr(other_gs, "sixgrupa_addr")
                and other_gs.sixgrupa_addr is not None
            ):
                other_addr = other_gs.sixgrupa_addr
                # Check if this GS is attached to the same satellite (same coordinates)
                if (
                    other_addr.shell_id == sat_addr.shell_id
                    and other_addr.plane_id == sat_addr.plane_id
                    and other_addr.sat_index == sat_addr.sat_index
                ):
                    used_subnet_indices.add(other_addr.subnet_index)

        # Find the next available subnet_index > 0 (0 is reserved for satellite)
        subnet_index = 1
        while subnet_index in used_subnet_indices:
            subnet_index += 1

        # Assign new address based on the current satellite
        gs_address = _assign_gs_address_from_satellite(gs, curr_sat_id, subnet_index, topology)
        if gs_address:
            gs.sixgrupa_addr = gs_address
            gs.previous_attached_satellite_id = curr_sat_id  # Update the previous attachment
            log.info(f"Renumbered GS {gs.id} to new address {gs_address}")
        else:
            log.warning(f"Renumbering GS {gs.id} failed, address assignment returned None")
    else:
        # No current satellite - clear the address
        gs.sixgrupa_addr = None
        gs.previous_attached_satellite_id = None  # Clear previous attachment
        log.debug(f"GS {gs.id} detached, cleared 6grupa address")


def _fill_forwarding_tables_in_every_satellite(
    satellite_node_ids: list[int],
    satellite_only_subgraph: nx.Graph,
    topology_with_isls: LEOTopology,
):
    """
    Fill forwarding tables in every satellite based on neighbor 6grupa addresses.
    """
    for satellite_id in satellite_node_ids:
        try:
            satellite = topology_with_isls.get_satellite(satellite_id)
            for neighbor_id in satellite_only_subgraph.neighbors(satellite_id):
                interface = topology_with_isls.sat_neighbor_to_if.get((satellite_id, neighbor_id))
                if interface is not None:
                    try:
                        neighbor_address = (
                            TopologicalNetworkAddress.set_address_from_orbital_parameters(
                                neighbor_id
                            )
                        )
                        satellite.forwarding_table[neighbor_address.to_integer()] = interface
                        log.debug(
                            f"Forwarding entry added for satellite {satellite_id} to neighbor {neighbor_id}: "
                            f"address {neighbor_address}, interface {interface}"
                        )
                    except Exception as e:
                        log.warning(
                            f"Failed to add forwarding entry for satellite {satellite_id} -> {neighbor_id}: {e}"
                        )
        except Exception as e:
            log.error(f"Failed to process satellite {satellite_id} for forwarding table: {e}")


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

    This implements the core topological routing logic:
    - For each satellite, determine next hop to reach each ground station
    - Use topological distance calculations based on 6grupa addresses
    """
    log.debug("Calculating satellite-to-GS forwarding state using topological routing")

    for curr_sat_id in nodelist:
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
            if not possible_dst_sats:
                continue

            log.debug(
                f"FSTATE: Sat {curr_sat_id} -> GS {dst_gs.id}. Visible sats: {[sat_id for _, sat_id in possible_dst_sats]}"
            )

            # Find the best destination satellite using topological distance
            best_dst_sat_id = None
            best_total_distance = float("inf")

            for dist_gs_to_sat_m, visible_sat_id in possible_dst_sats:
                try:
                    # Get 6grupa address of the destination satellite
                    dest_sat_address = (
                        TopologicalNetworkAddress.set_address_from_orbital_parameters(
                            visible_sat_id
                        )
                    )

                    # Calculate topological distance from current satellite to destination satellite
                    if hasattr(curr_satellite, "sixgrupa_addr") and curr_satellite.sixgrupa_addr:
                        topo_distance = curr_satellite.sixgrupa_addr.topological_distance_to(
                            dest_sat_address
                        )
                        total_distance = topo_distance + (
                            dist_gs_to_sat_m / 1000000.0
                        )  # Convert to similar scale

                        if total_distance < best_total_distance:
                            best_total_distance = total_distance
                            best_dst_sat_id = visible_sat_id
                    else:
                        log.warning(f"Satellite {curr_sat_id} has no 6grupa address assigned")

                except Exception as e:
                    log.warning(f"Failed to process destination satellite {visible_sat_id}: {e}")
                    continue

            if best_dst_sat_id is None:
                continue

            # Get destination satellite address for routing decision
            try:
                dest_sat_address = TopologicalNetworkAddress.set_address_from_orbital_parameters(
                    best_dst_sat_id
                )

                # Determine next hop using topological routing
                next_hop_decision, distance_to_ground_station_m = (
                    _get_next_hop_decision_topological(
                        curr_sat_id,
                        curr_satellite,
                        dest_sat_address,
                        sat_subgraph,
                        sat_neighbor_to_if,
                        topology_with_isls,
                        dst_gs_node_id,
                    )
                )

                if next_hop_decision is not None:
                    dist_satellite_to_ground_station[(curr_sat_id, dst_gs_node_id)] = (
                        distance_to_ground_station_m
                    )
                    fstate[(curr_sat_id, dst_gs_node_id)] = next_hop_decision
                    log.debug(
                        f"Fstate entry: Sat {curr_sat_id} -> GS {dst_gs_node_id} via {next_hop_decision}"
                    )

            except Exception as e:
                log.warning(
                    f"Failed to create routing decision for satellite {curr_sat_id} to GS {dst_gs_node_id}: {e}"
                )
                continue


def _get_next_hop_decision_topological(
    curr_sat_id: int,
    curr_satellite,
    destination_address: TopologicalNetworkAddress,
    sat_subgraph,
    sat_neighbor_to_if,
    topology_with_isls,
    dst_gs_node_id: int,
) -> tuple:
    """
    Determine the next hop decision using topological routing.

    This implements topological routing where each satellite computes the next hop
    by looking at neighbor's 6grupa addresses and performing a distance function.

    Args:
        curr_sat_id: Current satellite ID
        curr_satellite: Current satellite object
        destination_address: 6grupa address of the destination satellite
        sat_subgraph: Satellite-only subgraph
        sat_neighbor_to_if: Interface mapping
        topology_with_isls: Topology object
        dst_gs_node_id: Destination ground station ID

    Returns:
        tuple: (next_hop_interface_or_decision, distance) or (None, inf) if no path
    """
    if not hasattr(curr_satellite, "sixgrupa_addr") or not curr_satellite.sixgrupa_addr:
        log.warning(f"Satellite {curr_sat_id} has no 6grupa address assigned")
        return None, float("inf")

    my_address = curr_satellite.sixgrupa_addr
    my_distance_to_dest = my_address.topological_distance_to(destination_address)

    # Check if we are already at the destination satellite
    if my_distance_to_dest == 0.0:
        # Direct GSL connection - use GSL interface
        log.debug(f"Direct GSL path: Sat {curr_sat_id} -> GS {dst_gs_node_id}")
        return ("GSL", dst_gs_node_id), 0.0

    # Find the best neighbor using topological distance
    best_neighbor_id = None
    best_distance = my_distance_to_dest
    best_interface = None

    # Check all neighbors
    for neighbor_id in sat_subgraph.neighbors(curr_sat_id):
        try:
            neighbor_address = TopologicalNetworkAddress.set_address_from_orbital_parameters(
                neighbor_id
            )
            neighbor_distance_to_dest = neighbor_address.topological_distance_to(
                destination_address
            )

            # If this neighbor is closer to destination, consider it
            if neighbor_distance_to_dest < best_distance:
                interface = sat_neighbor_to_if.get((curr_sat_id, neighbor_id))
                if interface is not None:
                    best_distance = neighbor_distance_to_dest
                    best_neighbor_id = neighbor_id
                    best_interface = interface

        except Exception:
            # Skip this neighbor if we can't get its address
            continue

    if best_neighbor_id is not None:
        log.debug(
            f"Topological routing: Sat {curr_sat_id} -> {best_neighbor_id} (if {best_interface}) "
            f"towards destination with distance {best_distance}"
        )
        return best_interface, best_distance

    # No better neighbor found - this shouldn't happen in a connected graph
    log.warning(f"No better neighbor found for satellite {curr_sat_id} to reach destination")
    return None, float("inf")
