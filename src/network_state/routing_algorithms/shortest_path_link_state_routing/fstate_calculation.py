# fstate_calculation.py (Refactored Function)

import math
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

from src import logger
from src.topology.topology import GroundStation, LEOTopology

log = logger.get_logger(__name__)


def calculate_fstate_shortest_path_object_no_gs_relay(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
) -> dict:
    """
    Calculates forwarding state using shortest paths over ISLs only (no GS relays).
    """
    log.debug("Calculating shortest path fstate object (no GS relay)")

    full_graph = topology_with_isls.graph
    sat_neighbor_to_if = topology_with_isls.sat_neighbor_to_if

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
    # maps satellite node IDs to integer indices for efficient matrix operations
    # node_to_index dictionary allows efficient O(1) lookups to find the corresponding position
    # in the distance matrix for any given satellite ID
    node_to_index = {node_id: index for index, node_id in enumerate(satellite_node_ids)}
    # We compute subgraph because GSs are always either a src or dst, but never an intermediate node.
    satellite_only_subgraph = full_graph.subgraph(satellite_node_ids)

    if satellite_only_subgraph.number_of_nodes() == 0:
        log.warning("Satellite-only subgraph is empty. No ISL paths possible.")
        return {}

    try:
        log.debug(
            f"Calculating Floyd-Warshall on satellite subgraph for {len(satellite_node_ids)} nodes..."
        )
        dist_matrix = nx.floyd_warshall_numpy(
            satellite_only_subgraph, nodelist=satellite_node_ids, weight="weight"
        )
        log.debug("Floyd-Warshall calculation complete.")
    except (nx.NetworkXError, Exception) as e:
        log.error(f"Error during Floyd-Warshall shortest path calculation: {e}")
        return {}

    fstate: dict[tuple, tuple] = {}
    dist_satellite_to_ground_station: dict[tuple, float] = {}

    _calculate_sat_to_gs_fstate(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        satellite_node_ids,
        node_to_index,
        satellite_only_subgraph,
        dist_matrix,
        sat_neighbor_to_if,
        dist_satellite_to_ground_station,
        fstate,
    )

    _calculate_gs_to_gs_fstate(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        node_to_index,
        dist_satellite_to_ground_station,
        fstate,
    )

    log.debug(f"Calculated fstate object with {len(fstate)} entries.")
    return fstate


def _calculate_sat_to_gs_fstate(
    topology_with_isls: LEOTopology,
    ground_stations: List[GroundStation],
    ground_station_satellites_in_range: List[List[Tuple[float, int]]],
    nodelist: List[int],
    node_to_index: Dict[int, int],
    sat_subgraph: nx.Graph,
    dist_matrix: np.ndarray,
    sat_neighbor_to_if: Dict[Tuple[int, int], int],
    dist_satellite_to_ground_station: Dict[Tuple[int, int], float],
    fstate: Dict[Tuple[int, int], Tuple[int, int, int]],
) -> None:
    for curr_sat_id in nodelist:
        curr_sat_idx = node_to_index[curr_sat_id]
        try:
            topology_with_isls.get_satellite(curr_sat_id)
        except KeyError:
            log.error(
                f"Could not find satellite object {curr_sat_id} (should exist based on nodelist)."
            )
            continue

        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_node_id = dst_gs.id
            if gs_idx >= len(ground_station_satellites_in_range):
                continue
            possible_dst_sats = ground_station_satellites_in_range[gs_idx]
            if possible_dst_sats:
                log.debug(
                    f"  > FSTATE: Sat {curr_sat_id} -> GS {dst_gs.id}. Visible sats: {possible_dst_sats}"
                )
            possibilities = _get_satellite_possibilities(
                possible_dst_sats, curr_sat_idx, node_to_index, dist_matrix
            )

            next_hop_decision, distance_to_ground_station_m = _get_next_hop_decision(
                possibilities,
                curr_sat_id,
                node_to_index,
                sat_subgraph,
                dist_matrix,
                sat_neighbor_to_if,
                topology_with_isls,
                dst_gs_node_id,
            )

            dist_satellite_to_ground_station[(curr_sat_id, dst_gs_node_id)] = (
                distance_to_ground_station_m
            )
            fstate[(curr_sat_id, dst_gs_node_id)] = next_hop_decision


def _get_satellite_possibilities(
    possible_dst_sats: List[Tuple[float, int]],
    curr_sat_idx: int,
    node_to_index: Dict[int, int],
    dist_matrix: np.ndarray,
) -> List[Tuple[float, int]]:
    possibilities = []
    for visibility_info in possible_dst_sats:
        dist_gs_to_sat_m, visible_sat_id = visibility_info
        visible_sat_idx = node_to_index.get(visible_sat_id)
        if visible_sat_idx is not None:
            # Get the distance from current satellite to this visible satellite
            dist_curr_to_visible_sat = dist_matrix[curr_sat_idx, visible_sat_idx]
            # If a path exists (distance is not infinity):
            if not np.isinf(dist_curr_to_visible_sat):
                # distance(current_sat → visible_sat) + distance(visible_sat → ground_station)
                total_dist = dist_curr_to_visible_sat + dist_gs_to_sat_m
                possibilities.append((total_dist, visible_sat_id))
    possibilities.sort()  # Shortest path distance will be first
    return possibilities


def _get_next_hop_decision(
    possible_sat_gs_routes: List[Tuple[float, int]],
    curr_sat_id: int,
    node_to_index: Dict[int, int],
    sat_subgraph: nx.Graph,
    dist_matrix: np.ndarray,
    sat_neighbor_to_if: Dict[Tuple[int, int], int],
    topology_with_isls: LEOTopology,
    dst_gs_node_id: int,
) -> Tuple[Tuple[int, int, int], float]:
    next_hop_decision = (-1, -1, -1)
    distance_to_ground_station_m = float("inf")

    if possible_sat_gs_routes:
        distance_to_ground_station_m, dst_sat_id = possible_sat_gs_routes[0]  # Shortest is first
        dst_sat_idx = node_to_index.get(dst_sat_id)
        if dst_sat_idx is None:
            return next_hop_decision, distance_to_ground_station_m

        if curr_sat_id != dst_sat_id:
            next_hop_decision = _handle_multihop_path(
                curr_sat_id,
                dst_sat_idx,
                sat_subgraph,
                node_to_index,
                dist_matrix,
                sat_neighbor_to_if,
            )
        else:
            next_hop_decision = _handle_direct_gs_path(
                dst_sat_id, dst_gs_node_id, topology_with_isls
            )

    return next_hop_decision, distance_to_ground_station_m


def _handle_multihop_path(
    curr_sat_id: int,
    dst_sat_idx: int,
    sat_subgraph: nx.Graph,
    node_to_index: Dict[int, int],
    dist_matrix: np.ndarray,
    sat_neighbor_to_if: Dict[Tuple[int, int], int],
) -> Tuple[int, int, int]:
    """
    Handle routing when current satellite needs to route through other satellites.

    Returns:
        Tuple[int, int, int]: (next_hop_id, local_interface, remote_interface)
    """
    next_hop_decision = (-1, -1, -1)
    best_neighbor_dist_m = float("inf")

    for neighbor_id in sat_subgraph.neighbors(curr_sat_id):
        neighbor_idx = node_to_index.get(neighbor_id)
        if neighbor_idx is None:
            continue

        try:
            link_weight = sat_subgraph.edges[curr_sat_id, neighbor_id]["weight"]
            dist_neighbor_to_dst_sat = dist_matrix[neighbor_idx, dst_sat_idx]

            if not np.isinf(dist_neighbor_to_dst_sat):
                distance_m = link_weight + dist_neighbor_to_dst_sat
                if distance_m < best_neighbor_dist_m:
                    my_if = sat_neighbor_to_if.get((curr_sat_id, neighbor_id), -1)
                    next_hop_if = sat_neighbor_to_if.get((neighbor_id, curr_sat_id), -1)
                    next_hop_decision = (neighbor_id, my_if, next_hop_if)
                    best_neighbor_dist_m = distance_m
        except KeyError:
            log.warning(f"KeyError for edge ({curr_sat_id}, {neighbor_id}) in sat_subgraph.")

    return next_hop_decision


def _handle_direct_gs_path(
    dst_sat_id: int, dst_gs_node_id: int, topology_with_isls: LEOTopology
) -> Tuple[int, int, int]:
    """
    Handle routing when current satellite can directly communicate with the ground station.

    Returns:
        Tuple[int, int, int]: (ground_station_id, local_gsl_interface, gs_interface)
    """
    try:
        dst_satellite = topology_with_isls.get_satellite(dst_sat_id)
        num_isls_dst_sat = dst_satellite.number_isls
        my_gsl_if = num_isls_dst_sat
        next_hop_gsl_if = 0
        return (dst_gs_node_id, my_gsl_if, next_hop_gsl_if)
    except KeyError:
        log.error(f"Could not find satellite object {dst_sat_id} for GS exit hop.")
        return (-1, -1, -1)


def _calculate_gs_to_gs_fstate(
    topology_with_isls: LEOTopology,
    ground_stations: List[GroundStation],
    ground_station_satellites_in_range: List[List[Tuple[float, int]]],
    node_to_index: Dict[int, int],
    dist_satellite_to_ground_station: Dict[Tuple[int, int], float],
    fstate: Dict[Tuple[int, int], Tuple[int, int, int]],
) -> None:
    for src_idx, src_gs in enumerate(ground_stations):
        src_gs_node_id = src_gs.id
        for dst_idx, dst_gs in enumerate(ground_stations):
            if src_idx == dst_idx:
                continue
            dst_gs_node_id = dst_gs.id

            if src_idx >= len(ground_station_satellites_in_range):
                log.warning(
                    f"Index {src_idx} out of bounds for ground_station_satellites_in_range. Skipping GS {src_gs_node_id}."
                )
                continue
            possible_src_sats = ground_station_satellites_in_range[src_idx]
            possibilities = []
            for visibility_info in possible_src_sats:
                dist_gs_to_sat_m, entry_sat_id = visibility_info

                if entry_sat_id in node_to_index:
                    dist_entry_sat_to_dst_gs = dist_satellite_to_ground_station.get(
                        (entry_sat_id, dst_gs_node_id), float("inf")
                    )

                    if not math.isinf(dist_entry_sat_to_dst_gs):
                        total_dist = dist_gs_to_sat_m + dist_entry_sat_to_dst_gs
                        possibilities.append((total_dist, entry_sat_id))

            possibilities.sort()

            next_hop_decision = (-1, -1, -1)
            if possibilities:
                _, src_sat_id = possibilities[0]

                try:
                    entry_satellite = topology_with_isls.get_satellite(src_sat_id)
                    num_isls_entry_sat = entry_satellite.number_isls
                    my_gsl_if = 0
                    next_hop_gsl_if = num_isls_entry_sat
                    next_hop_decision = (src_sat_id, my_gsl_if, next_hop_gsl_if)
                except (KeyError, IndexError):
                    log.error(f"Could not find satellite object {src_sat_id} for GS->Sat hop.")
                    next_hop_decision = (-1, -1, -1)

            fstate[(src_gs_node_id, dst_gs_node_id)] = next_hop_decision
