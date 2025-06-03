# algorithm_free_one_only_over_isls.py (Refactored)

# The MIT License (MIT)
# ... (License remains the same) ...

from src import logger
from src.topology.topology import ConstellationData, GroundStation, LEOTopology

# --- Assume fstate_calculation is refactored ---
# We expect a function like calculate_fstate_shortest_path_object
# which takes topology, visibility, prev_state object and returns the fstate object
from .fstate_calculation import (  # Renamed/refactored function
    calculate_fstate_shortest_path_object_no_gs_relay,
)

# Remove graph utils import if _check_graph_is_valid is removed or uses topology methods directly
# from .utils import graph as graph_utils


log = logger.get_logger(__name__)

# Note: _check_graph_is_valid might be redundant if topology construction is robust
# def _check_graph_is_valid(topology: LEOTopology):
#     """
#     Check if the graph is valid by ensuring that all nodes are satellites and that there are no ground stations.
#     (May need adjustment based on how LEOTopology is constructed and used)
#     """
#     # This check might need reconsideration depending on the exact graph passed
#     if topology.graph.number_of_nodes() != topology.constellation_data.number_of_satellites:
#         # If topology_with_isls graph strictly contains *only* satellite nodes:
#         log.warning(f"Graph node count ({topology.graph.number_of_nodes()}) differs from satellite count ({topology.constellation_data.number_of_satellites})")
#         # If it can contain GS nodes without links, the check needs refinement
#     # graph_utils.validate_no_satellite_to_gs_links(...) # This check seems specific to GS relaying, maybe not needed here


def algorithm_free_one_only_over_isls(
    # output_dynamic_state_dir: str, # Removed: No longer writing state files here
    time_since_epoch_ns: int,
    constellation_data: ConstellationData,  # Passed as 'satellites' arg previously, use directly
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,  # Graph containing satellites and ISLs (maybe GS nodes too, depends on _build_topologies)
    ground_station_satellites_in_range: list,  # Visibility info: list[list[(dist, sat_id)]] per GS
    # Removed num_isls_per_sat - derivable from topology_with_isls.get_satellite(id).number_isls
    # Removed sat_neighbor_to_if - derivable from topology_with_isls.sat_neighbor_to_if
    list_gsl_interfaces_info: list,  # Info about bandwidth per node/interface
    prev_output: dict | None,  # Contains previous state: {'fstate': {...}, 'bandwidth': {...}}
) -> dict:  # Return the new state object
    """
    Refactored: FREE-ONE ONLY OVER INTER-SATELLITE LINKS ALGORITHM

    Calculates bandwidth and forwarding state (shortest paths via ISLs only)
    and returns them as state objects, without writing to files.

    Assumptions:
    - "one": Each satellite/GS has one GSL interface (index 0).
    - "free": Bandwidth is per-interface, no strict reciprocation enforced here.
    - "only_over_isls": Paths are GS -> Sat -> ... -> Sat -> GS.

    :param time_since_epoch_ns: Current time step relative to epoch (integer ns).
    :param constellation_data: Holds satellite list, counts, max lengths, epoch string.
    :param ground_stations: List of GroundStation objects.
    :param topology_with_isls: LEOTopology object containing the graph with ISL links calculated.
                               Also contains ISL interface mapping (sat_neighbor_to_if).
    :param ground_station_satellites_in_range: List where index=gs_idx, value=list of (distance, sat_id) tuples visible to that GS.
    :param list_gsl_interfaces_info: List of dicts, one per sat/GS, with bandwidth info.
    :param prev_output: Dictionary containing 'fstate' and 'bandwidth' objects from the previous step.
    :param enable_verbose_logs: Boolean to enable detailed logging.
    :return: Dictionary containing the new 'fstate' and 'bandwidth' state objects.
    """
    log.debug(f"Running algorithm_free_one_only_over_isls for t={time_since_epoch_ns} ns")

    # --- 1. Calculate Bandwidth State ---
    # Represents bandwidth capacity of the single GSL interface (IF=0) for each node.
    current_bandwidth_state = {}  # Key: node_id (int), Value: bandwidth (float)
    num_satellites = constellation_data.number_of_satellites
    num_total_nodes = num_satellites + len(ground_stations)

    if len(list_gsl_interfaces_info) != num_total_nodes:
        log.warning(
            f"Length mismatch: list_gsl_interfaces_info ({len(list_gsl_interfaces_info)}) "
            f"vs total nodes ({num_total_nodes}). Bandwidth state might be incomplete."
        )

    for i in range(num_total_nodes):
        if i < len(list_gsl_interfaces_info):
            # Assume the list order matches node IDs 0..N-1 (Sats 0..M-1, GSs M..N-1)
            # Or use an 'id' field if present in the list_gsl_interfaces_info dictionaries
            node_id = list_gsl_interfaces_info[i].get("id", i)
            bandwidth = list_gsl_interfaces_info[i].get("aggregate_max_bandwidth", 0.0)
            # Store bandwidth for the assumed single GSL interface (index 0) per node
            current_bandwidth_state[node_id] = bandwidth
            log.debug(f"  Bandwidth state: Node {node_id}, IF 0, BW = {bandwidth}")
        else:
            # Fallback if list is too short
            node_id = i  # Assume node ID is the index
            current_bandwidth_state[node_id] = 0.0
            log.error(
                f"Index {i} out of bounds for list_gsl_interfaces_info, setting BW=0 for node {node_id}"
            )
    log.debug(f"  Calculated bandwidth state for {len(current_bandwidth_state)} nodes.")

    # --- 2. Calculate Forwarding State ---
    # Extract previous fstate object if available
    # prev_fstate_obj = prev_output.get("fstate") if prev_output else None

    # The helper function needs access to the topology graph, ISL interface mapping,
    # satellite objects (via topology), ground station list, and GS->Sat visibility.

    # Call the refactored helper function (expected to be in fstate_calculation.py)
    # This function now returns the fstate object directly.
    try:
        current_fstate_obj = calculate_fstate_shortest_path_object_no_gs_relay(
            topology_with_isls,  # Contains graph, sat_neighbor_to_if, get_satellite()
            ground_stations,  # Needed to iterate through destinations
            ground_station_satellites_in_range,  # Needed for entry/exit points of paths
            # prev_fstate_obj,                  # Pass previous state if helper does delta logic
            # enable_verbose_logs               # Pass if helper needs it
        )
        log.debug("Calculated forwarding state object.")

    except NameError:
        log.exception(
            "Failed to call 'calculate_fstate_shortest_path_object'. "
            "Ensure fstate_calculation.py has been refactored."
        )
        # Handle error - perhaps return previous state or raise?
        # For now, return an empty fstate
        current_fstate_obj = {}
    except Exception as e:
        log.exception(f"Error during forwarding state calculation: {e}")
        current_fstate_obj = {}  # Return empty fstate on error

    # --- 3. Return Combined State ---
    new_state = {
        "fstate": current_fstate_obj,
        "bandwidth": current_bandwidth_state,
    }
    return new_state


# --- IMPORTANT ---
# The function `calculate_fstate_shortest_path_object` (previously part of
# `calculate_fstate_shortest_path_without_gs_relaying` in `fstate_calculation.py`)
# needs to be refactored separately. It should:
# 1. Accept `topology_with_isls`, `ground_stations`, `ground_station_satellites_in_range` etc.
# 2. Perform the shortest path calculations using NetworkX on `topology_with_isls.graph`.
# 3. Determine the next hop for each (source_sat, dest_gs_id) pair based on the path.
# 4. Use `topology_with_isls.sat_neighbor_to_if` to map ISL next hops to interface numbers.
# 5. Handle the mapping for the first GSL hop (GS -> Sat) and the last GSL hop (Sat -> GS),
#    potentially assuming interface 0 on the GS and deriving the satellite's GSL interface index.
# 6. Return the forwarding state as a dictionary structure, e.g.:
#    fstate = {
#        satellite_id: {
#            destination_gs_id: (next_hop_node_id, next_hop_interface_index),
#            ...
#        },
#        ...
#    }
# 7. Remove all file I/O operations related to writing the fstate file.
