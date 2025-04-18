# fstate_calculation.py (Refactored Function)

import math
import networkx as nx
import numpy as np  # Import numpy for inf checking
from src.dynamic_state.topology import LEOTopology, GroundStation, Satellite
from src import logger  # Optional: if logging is desired

log = logger.get_logger(__name__)  # Optional


def calculate_fstate_shortest_path_object_no_gs_relay(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,  # Visibility: list[list[(dist, sat_id)]] per GS index
) -> dict:
    """
    Calculates forwarding state using shortest paths over ISLs only (no GS relays).

    Handles potentially non-sequential satellite IDs. Operates on topology
    objects and returns the fstate dictionary directly.

    :param topology_with_isls: LEOTopology object containing graph with ISLs,
                               satellite objects, and ISL interface mapping.
                               Graph nodes MUST be satellite IDs.
    :param ground_stations: List of GroundStation objects.
    :param ground_station_satellites_in_range: List where index=gs_idx,
                                                value=list of (distance, sat_id) tuples visible.
    :return: fstate dictionary {(src_node_id, dst_node_id): (next_hop_id, my_ifidx, next_hop_ifidx)}
             Returns empty dict {} if path calculation fails or no valid nodes exist.
    """
    log.debug("Calculating shortest path fstate object (no GS relay)")

    constellation_data = topology_with_isls.constellation_data
    sat_graph = topology_with_isls.graph  # Graph with ISLs
    sat_neighbor_to_if = topology_with_isls.sat_neighbor_to_if
    num_ground_stations = len(ground_stations)

    # --- Prepare for Floyd-Warshall with potentially non-sequential satellite IDs ---

    # 1. Create the explicit list of satellite nodes to use for path calculation.
    #    Only include nodes actually present in the graph.
    #    Assuming satellite IDs are the nodes in sat_graph intended for pathfinding.
    #    Filter out any potential non-satellite nodes if the graph construction allows them.
    #    Sorting ensures consistent matrix ordering if node order changes run-to-run.
    nodelist = sorted(
        [
            node
            for node in sat_graph.nodes()
            # Add check if needed: Ensure node represents a satellite, e.g., check type or ID range
            # if isinstance(node, int) and node < constellation_data.number_of_satellites # Example check
        ]
    )

    if not nodelist:
        log.warning("ISL topology graph contains no nodes for path calculation.")
        return {}

    # 2. Create ID-to-index mapping for accessing the distance matrix
    node_to_index = {node_id: index for index, node_id in enumerate(nodelist)}

    # 3. Calculate shortest path distances using the explicit nodelist
    try:
        log.debug(f"Calculating Floyd-Warshall on ISL graph for {len(nodelist)} nodes...")
        # The returned matrix `dist_matrix[i, j]` corresponds to distance
        # between `nodelist[i]` and `nodelist[j]`
        dist_matrix = nx.floyd_warshall_numpy(sat_graph, nodelist=nodelist, weight="weight")
        log.debug("Floyd-Warshall calculation complete.")
    except (nx.NetworkXError, Exception) as e:
        # Catch NetworkX specific errors and potentially others (e.g., NumPy issues)
        log.error(f"Error during Floyd-Warshall shortest path calculation: {e}")
        return {}  # Return empty state on error

    fstate = {}  # Initialize the forwarding state dictionary

    # --- Satellites to Ground Stations ---
    dist_satellite_to_ground_station = {}  # Helper dict: {(sat_id, dst_gs_id): distance_m}

    # Iterate through satellite IDs that are actually in the graph nodelist
    for curr_sat_id in nodelist:
        # Get the index corresponding to the current satellite ID for matrix access
        curr_sat_idx = node_to_index[curr_sat_id]

        # Get satellite object to access number_isls later
        try:
            current_satellite = topology_with_isls.get_satellite(curr_sat_id)
        except (KeyError, IndexError):  # Handle potential errors if get_satellite fails
            log.error(f"Could not find satellite object with ID {curr_sat_id} in topology.")
            continue  # Skip this satellite if object not found

        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_node_id = dst_gs.id  # Use the actual ground station ID

            # Find best exit satellite (dst_sat_id) for dst_gs among visible candidates
            # ground_station_satellites_in_range uses the index matching ground_stations list
            if gs_idx >= len(ground_station_satellites_in_range):
                log.warning(
                    f"Index {gs_idx} out of bounds for ground_station_satellites_in_range. Skipping GS {dst_gs_node_id}."
                )
                continue
            possible_dst_sats = ground_station_satellites_in_range[gs_idx]
            possibilities = []  # List of (total_dist_m, exit_sat_id)

            for visibility_info in possible_dst_sats:
                dist_gs_to_sat_m, visible_sat_id = visibility_info
                visible_sat_idx = node_to_index.get(visible_sat_id)

                # Check if the satellite visible from GS is part of the ISL graph pathfinding
                if visible_sat_idx is not None:
                    # Access distance matrix using indices
                    dist_curr_to_visible_sat = dist_matrix[curr_sat_idx, visible_sat_idx]
                    # Check reachability using numpy's isinf
                    if not np.isinf(dist_curr_to_visible_sat):
                        total_dist = dist_curr_to_visible_sat + dist_gs_to_sat_m
                        possibilities.append((total_dist, visible_sat_id))
                else:
                    # Satellite visible by GS is not in the ISL graph nodelist, cannot be exit node
                    pass
                    # log.debug(f"Sat {visible_sat_id} visible to GS {dst_gs_node_id} not in ISL nodelist.")

            possibilities.sort()  # Sort by total distance

            next_hop_decision = (-1, -1, -1)  # Default: drop packet
            distance_to_ground_station_m = float("inf")

            if possibilities:  # If at least one path exists
                distance_to_ground_station_m, dst_sat_id = possibilities[
                    0
                ]  # Best path distance and exit sat ID
                dst_sat_idx = node_to_index.get(dst_sat_id)  # Get index for best exit sat

                if (
                    dst_sat_idx is None
                ):  # Should not happen if possibilities is non-empty & consistent
                    log.error(
                        f"Logic error: Best exit satellite {dst_sat_id} not found in index map after sorting."
                    )
                    continue

                # If the current satellite is not the best exit satellite...
                if curr_sat_id != dst_sat_id:
                    # Find the best neighbor of curr_sat_id to route towards dst_sat_id
                    best_neighbor_dist_m = float("inf")
                    if curr_sat_id not in sat_graph:  # Defensive check
                        log.warning(f"Current satellite {curr_sat_id} disappeared from graph?")
                    else:
                        for neighbor_id in sat_graph.neighbors(curr_sat_id):
                            neighbor_idx = node_to_index.get(neighbor_id)
                            # Check if neighbor is in the ISL graph used for pathfinding
                            if neighbor_idx is not None:
                                try:
                                    link_weight = sat_graph.edges[curr_sat_id, neighbor_id][
                                        "weight"
                                    ]
                                    # Access distance matrix using indices
                                    dist_neighbor_to_dst_sat = dist_matrix[
                                        neighbor_idx, dst_sat_idx
                                    ]

                                    if not np.isinf(dist_neighbor_to_dst_sat):
                                        distance_m = link_weight + dist_neighbor_to_dst_sat
                                        if distance_m < best_neighbor_dist_m:
                                            # Get ISL interface IDs from topology map
                                            my_if = sat_neighbor_to_if.get(
                                                (curr_sat_id, neighbor_id), -1
                                            )
                                            next_hop_if = sat_neighbor_to_if.get(
                                                (neighbor_id, curr_sat_id), -1
                                            )
                                            if my_if == -1 or next_hop_if == -1:
                                                log.warning(
                                                    f"Missing ISL interface mapping for link ({curr_sat_id}, {neighbor_id})"
                                                )
                                            next_hop_decision = (neighbor_id, my_if, next_hop_if)
                                            best_neighbor_dist_m = distance_m
                                except KeyError:
                                    log.warning(
                                        f"Edge/weight missing for link ({curr_sat_id}, {neighbor_id}) in ISL graph"
                                    )
                                    continue  # Skip this neighbor
                            else:
                                # Neighbor node not in nodelist used for distance matrix
                                pass
                                # log.debug(f"Neighbor {neighbor_id} of Sat {curr_sat_id} not in ISL nodelist.")

                else:  # The current satellite IS the best exit satellite
                    # Next hop is the ground station itself.
                    try:
                        # Get the Satellite object for the current/destination satellite
                        dst_satellite = topology_with_isls.get_satellite(
                            dst_sat_id
                        )  # curr_sat_id == dst_sat_id
                        num_isls_dst_sat = dst_satellite.number_isls
                        # Assume GSL IF index follows ISL IF indices (0 to num_isls-1 are ISLs)
                        my_gsl_if = (
                            num_isls_dst_sat  # GSL IF = number of ISLs (assuming 0-based IF count)
                        )
                        # Assume ground station incoming GSL interface is always 0 ("one" algorithm)
                        next_hop_gsl_if = 0
                        next_hop_decision = (dst_gs_node_id, my_gsl_if, next_hop_gsl_if)
                    except (KeyError, IndexError):
                        log.error(
                            f"Could not find satellite object {dst_sat_id} for GSL hop calculation."
                        )
                        next_hop_decision = (-1, -1, -1)  # Fallback to drop

            # Store intermediate distance for GS->GS calculation
            dist_satellite_to_ground_station[(curr_sat_id, dst_gs_node_id)] = (
                distance_to_ground_station_m
            )

            # Store forwarding state entry: Sat -> GS
            # Key is (Current Sat ID, Destination GS ID)
            fstate[(curr_sat_id, dst_gs_node_id)] = next_hop_decision

    # --- Ground Stations to Ground Stations ---
    for src_idx, src_gs in enumerate(ground_stations):
        src_gs_node_id = src_gs.id
        for dst_idx, dst_gs in enumerate(ground_stations):
            if src_idx == dst_idx:
                continue  # Skip GS to itself
            dst_gs_node_id = dst_gs.id

            # Find best entry satellite (src_sat_id) visible to src_gs
            if src_idx >= len(ground_station_satellites_in_range):
                log.warning(
                    f"Index {src_idx} out of bounds for ground_station_satellites_in_range. Skipping GS {src_gs_node_id}."
                )
                continue
            possible_src_sats = ground_station_satellites_in_range[src_idx]
            possibilities = []  # Stores (total_dist_m, entry_sat_id)
            for visibility_info in possible_src_sats:
                dist_gs_to_sat_m, entry_sat_id = visibility_info

                # Check if entry satellite is in the ISL network before looking up distance
                if entry_sat_id in node_to_index:
                    # Look up pre-calculated distance from entry_sat to dst_gs
                    dist_entry_sat_to_dst_gs = dist_satellite_to_ground_station.get(
                        (entry_sat_id, dst_gs_node_id), float("inf")
                    )

                    if not math.isinf(
                        dist_entry_sat_to_dst_gs
                    ):  # Use math.isinf for standard float
                        total_dist = dist_gs_to_sat_m + dist_entry_sat_to_dst_gs
                        possibilities.append((total_dist, entry_sat_id))
                else:
                    # Entry satellite not in ISL network, cannot be used for path
                    pass

            possibilities.sort()

            next_hop_decision = (-1, -1, -1)  # Default: drop packet
            if possibilities:
                _, src_sat_id = possibilities[0]  # Best entry satellite ID

                # Next hop from GS is the chosen entry satellite
                try:
                    entry_satellite = topology_with_isls.get_satellite(src_sat_id)
                    num_isls_entry_sat = entry_satellite.number_isls
                    # Assume GS outgoing GSL IF is 0 ("one" algorithm)
                    my_gsl_if = 0
                    # Calculate incoming GSL IF on the satellite = num_isls
                    next_hop_gsl_if = num_isls_entry_sat
                    next_hop_decision = (src_sat_id, my_gsl_if, next_hop_gsl_if)
                except (KeyError, IndexError):
                    log.error(f"Could not find satellite object {src_sat_id} for GS->Sat hop.")
                    next_hop_decision = (-1, -1, -1)  # Fallback to drop

            # Store forwarding state entry: GS -> GS
            # Key is (Source GS ID, Destination GS ID)
            fstate[(src_gs_node_id, dst_gs_node_id)] = next_hop_decision

    log.debug(f"Calculated fstate object with {len(fstate)} entries.")
    # Return the calculated state dictionary
    return fstate
