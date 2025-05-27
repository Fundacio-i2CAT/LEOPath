# fstate_calculation.py (Refactored Function)

import math

import networkx as nx
import numpy as np

from src import logger
from src.topology.topology import GroundStation, LEOTopology

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

    # Get the full graph which might contain GS nodes too
    full_graph = topology_with_isls.graph
    sat_neighbor_to_if = topology_with_isls.sat_neighbor_to_if

    # --- Prepare for Floyd-Warshall on SATELLITE-ONLY subgraph ---

    # 1. Identify all known satellite IDs from the topology object
    try:
        all_satellite_ids = {sat.id for sat in topology_with_isls.get_satellites()}
    except Exception as e:
        log.exception(f"Error getting satellite IDs from topology: {e}")
        return {}

    # 2. Create the nodelist containing only satellite IDs that are present in the full graph
    nodelist = sorted([node_id for node_id in full_graph.nodes() if node_id in all_satellite_ids])

    if not nodelist:
        log.warning("No valid satellite nodes found in the graph for path calculation.")
        return {}

    # 3. Create the ID-to-index mapping for satellites
    node_to_index = {node_id: index for index, node_id in enumerate(nodelist)}

    # 4. **** Create a subgraph view containing only satellites and their ISLs ****
    # This view will only include nodes present in the nodelist and edges between them.
    sat_subgraph = full_graph.subgraph(nodelist)

    # Check if the satellite subgraph is empty (could happen if nodes exist but no ISLs connect them)
    if sat_subgraph.number_of_nodes() == 0:
        log.warning("Satellite subgraph is empty. No ISL paths possible.")
        # Fallback: Calculate fstate with only direct GSL hops possible?
        # For now, return empty fstate as ISL paths are expected.
        return {}

    # 5. Calculate shortest path distances using the SUBGRAPH and satellite nodelist
    try:
        log.debug(f"Calculating Floyd-Warshall on satellite subgraph for {len(nodelist)} nodes...")
        # Pass the subgraph and the corresponding nodelist
        dist_matrix = nx.floyd_warshall_numpy(sat_subgraph, nodelist=nodelist, weight="weight")
        log.debug("Floyd-Warshall calculation complete.")
    except (nx.NetworkXError, Exception) as e:
        # Check if error is due to disconnected components in sat_subgraph
        # Note: Floyd-Warshall handles disconnected nodes by returning np.inf
        # The error might be something else (e.g., non-numeric weights).
        log.error(f"Error during Floyd-Warshall shortest path calculation: {e}")
        return {}  # Return empty state on error

    fstate = {}
    dist_satellite_to_ground_station = {}  # Initialize the dictionary to store distances

    # --- Satellites to Ground Stations ---
    # Iterate only over satellite IDs in the calculated nodelist
    for curr_sat_id in nodelist:
        curr_sat_idx = node_to_index[curr_sat_id]
        try:
            current_satellite = topology_with_isls.get_satellite(curr_sat_id)
        except KeyError:
            log.error(
                f"Could not find satellite object {curr_sat_id} (should exist based on nodelist)."
            )
            continue

        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_node_id = dst_gs.id
            if gs_idx >= len(ground_station_satellites_in_range):
                continue  # Safety check
            possible_dst_sats = ground_station_satellites_in_range[gs_idx]
            possibilities = []

            for visibility_info in possible_dst_sats:
                dist_gs_to_sat_m, visible_sat_id = visibility_info
                visible_sat_idx = node_to_index.get(visible_sat_id)
                if visible_sat_idx is not None:  # Check if visible sat is in our ISL network
                    dist_curr_to_visible_sat = dist_matrix[curr_sat_idx, visible_sat_idx]
                    if not np.isinf(dist_curr_to_visible_sat):
                        total_dist = dist_curr_to_visible_sat + dist_gs_to_sat_m
                        possibilities.append((total_dist, visible_sat_id))

            possibilities.sort()

            next_hop_decision = (-1, -1, -1)
            distance_to_ground_station_m = float("inf")

            if possibilities:
                distance_to_ground_station_m, dst_sat_id = possibilities[0]
                dst_sat_idx = node_to_index.get(dst_sat_id)
                if dst_sat_idx is None:
                    continue  # Should not happen

                if curr_sat_id != dst_sat_id:
                    best_neighbor_dist_m = float("inf")
                    # **** Iterate over neighbors in the SUBGRAPH ****
                    for neighbor_id in sat_subgraph.neighbors(curr_sat_id):
                        neighbor_idx = node_to_index.get(neighbor_id)
                        if neighbor_idx is not None:
                            try:
                                # Get weight from subgraph (or original graph)
                                link_weight = sat_subgraph.edges[curr_sat_id, neighbor_id]["weight"]
                                dist_neighbor_to_dst_sat = dist_matrix[neighbor_idx, dst_sat_idx]
                                if not np.isinf(dist_neighbor_to_dst_sat):
                                    distance_m = link_weight + dist_neighbor_to_dst_sat
                                    if distance_m < best_neighbor_dist_m:
                                        my_if = sat_neighbor_to_if.get(
                                            (curr_sat_id, neighbor_id), -1
                                        )
                                        next_hop_if = sat_neighbor_to_if.get(
                                            (neighbor_id, curr_sat_id), -1
                                        )
                                        next_hop_decision = (neighbor_id, my_if, next_hop_if)
                                        best_neighbor_dist_m = distance_m
                            except KeyError:
                                log.warning(...)  # Keep warning
                        # else: # Neighbor not satellite -> ignore for ISL path
                else:  # Current satellite IS the best exit satellite
                    try:
                        dst_satellite = topology_with_isls.get_satellite(dst_sat_id)
                        num_isls_dst_sat = dst_satellite.number_isls
                        my_gsl_if = num_isls_dst_sat
                        next_hop_gsl_if = 0
                        next_hop_decision = (dst_gs_node_id, my_gsl_if, next_hop_gsl_if)
                    except KeyError:
                        log.error(...)
                        next_hop_decision = (-1, -1, -1)

            dist_satellite_to_ground_station[(curr_sat_id, dst_gs_node_id)] = (
                distance_to_ground_station_m
            )
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
