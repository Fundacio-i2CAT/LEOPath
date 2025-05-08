# Filename: leo-routing-sim/src/post_analysis/graph_tools.py
# Content adapted from Hypatia's graph_tools.py, with modifications for leo-routing-sim.

# The MIT License (MIT)
#
# Copyright (c) 2020 ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Imports adapted for leo-routing-sim structure
# Assumes distance_tools.py is in src/distance_tools/ and this file is in src/post_analysis/
from ..distance_tools.distance_tools import (
    distance_m_between_satellites,
    distance_m_ground_station_to_satellite,
)

import networkx as nx  # For construct_graph_with_distances (not directly used by the main plotting script but kept)
from astropy import units as u  # For time calculations


# This function is USED by the plotting script.
def compute_path_length_without_graph(
    path,
    epoch,
    time_since_epoch_ns,
    satellites,
    ground_stations,
    list_isls,
    max_gsl_length_m,
    max_isl_length_m,
):
    # Convert nanosecond offset to days and add to epoch's Terrestrial Time (TT) Julian Date
    time_delta_days = (time_since_epoch_ns * 1e-9) / 86400.0  # 86400 seconds in a day
    time = epoch.ts.tt_jd(epoch.tt + time_delta_days)  # Create a new Skyfield Time object
    path_length_m = 0.0

    if path is None or len(path) < 2:
        return 0.0  # As per previous logic for short/None paths

    for i in range(1, len(path)):
        from_node_id = path[i - 1]
        to_node_id = path[i]

        if not (isinstance(from_node_id, int) and isinstance(to_node_id, int)):
            return None

        hop_length_m = None
        # Satellite to satellite
        if from_node_id < len(satellites) and to_node_id < len(satellites):
            is_isl_defined = False
            for isl_a, isl_b in list_isls:
                if (isl_a == from_node_id and isl_b == to_node_id) or (
                    isl_a == to_node_id and isl_b == from_node_id
                ):
                    is_isl_defined = True
                    break
            current_isl_distance_m = distance_m_between_satellites(
                satellites[from_node_id], satellites[to_node_id], str(epoch), str(time)
            )
            if current_isl_distance_m <= max_isl_length_m and is_isl_defined:
                hop_length_m = current_isl_distance_m
            else:
                return None
        # Ground station to satellite
        elif from_node_id >= len(satellites) and to_node_id < len(satellites):
            gs_idx = from_node_id - len(satellites)
            if not (0 <= gs_idx < len(ground_stations)):
                return None
            ground_station = ground_stations[gs_idx]
            current_gsl_distance_m = distance_m_ground_station_to_satellite(
                ground_station, satellites[to_node_id], str(epoch), str(time)
            )
            if current_gsl_distance_m <= max_gsl_length_m:
                hop_length_m = current_gsl_distance_m
            else:
                return None
        # Satellite to ground station
        elif from_node_id < len(satellites) and to_node_id >= len(satellites):
            gs_idx = to_node_id - len(satellites)
            if not (0 <= gs_idx < len(ground_stations)):
                return None
            ground_station = ground_stations[gs_idx]
            current_gsl_distance_m = distance_m_ground_station_to_satellite(
                ground_station, satellites[from_node_id], str(epoch), str(time)
            )
            if current_gsl_distance_m <= max_gsl_length_m:
                hop_length_m = current_gsl_distance_m
            else:
                return None
        else:
            return None

        if hop_length_m is None:
            return None
        path_length_m += hop_length_m

    return path_length_m


# This function IS USED by the plotting script. REVISED for robustness.
def get_path(src, dst, forward_state):
    """
    Reconstructs a path from src to dst using the forward_state dictionary.
    forward_state is expected to be {(current_node, ultimate_destination): next_hop_node_id}.
    Returns the path as a list of node IDs, or None if no path is found or an issue occurs.
    """
    if src == dst:
        return [src]

    # Check if an initial hop is defined for the (src, dst) pair.
    # The forward_state passed to this function (fstate_for_path_module)
    # is already filtered to exclude keys where next_hop was -1.
    if (src, dst) not in forward_state:
        # print(f"Debug get_path: No initial forwarding entry for ({src}, {dst}).")
        return None

    path = [src]
    current_hop = src
    # Define a reasonable maximum number of hops to prevent infinite loops
    # This could be based on the total number of nodes in the network if available
    max_hops = 200  # Adjust as needed, e.g., len(satellites) + len(ground_stations)

    while current_hop != dst:
        # Check if the current hop has a rule for the ultimate destination
        if (current_hop, dst) not in forward_state:
            # print(f"Debug get_path: Path broken. No forwarding entry for ({current_hop}, {dst}). Path so far: {path}")
            return None

        next_hop = forward_state[(current_hop, dst)]

        # Check for loops
        if next_hop in path:
            # print(f"Debug get_path: Loop detected. Next hop {next_hop} already in path {path}.")
            return None

        path.append(next_hop)
        current_hop = next_hop

        # Check for excessive path length
        if len(path) > max_hops:
            # print(f"Debug get_path: Exceeded max_hops ({max_hops}). Path: {path}")
            return None

    return path


# --- Functions below are from original Hypatia graph_tools.py, ---
# --- kept for completeness but NOT directly used by the plotting script provided. ---


def construct_graph_with_distances(
    epoch,
    time_since_epoch_ns,
    satellites,
    ground_stations,
    list_isls,
    max_gsl_length_m,
    max_isl_length_m,
):
    # Convert nanosecond offset to days and add to epoch's Terrestrial Time (TT) Julian Date
    time_delta_days = (time_since_epoch_ns * 1e-9) / 86400.0
    time = epoch.ts.tt_jd(epoch.tt + time_delta_days)  # Create a new Skyfield Time object
    sat_net_graph_with_gs = nx.Graph()

    for a, b in list_isls:
        sat_distance_m = distance_m_between_satellites(
            satellites[a], satellites[b], str(epoch), str(time)
        )
        if sat_distance_m <= max_isl_length_m:
            sat_net_graph_with_gs.add_edge(a, b, weight=sat_distance_m)

    for ground_station in ground_stations:
        # Ensure ground_station["gid"] is the local ID (0 to num_gs-1)
        gs_global_id = len(satellites) + ground_station["gid"]
        for sid in range(len(satellites)):
            distance_m = distance_m_ground_station_to_satellite(
                ground_station, satellites[sid], str(epoch), str(time)
            )
            if distance_m <= max_gsl_length_m:
                sat_net_graph_with_gs.add_edge(gs_global_id, sid, weight=distance_m)
    return sat_net_graph_with_gs


def compute_path_length_with_graph(path, graph):
    if path is None:
        return 0.0  # Or None
    return sum_path_weights(augment_path_with_weights(path, graph))


def get_path_with_weights(src, dst, forward_state, sat_net_graph_with_gs):
    if (src, dst) not in forward_state:  # Check if key exists first
        return None
    if forward_state[(src, dst)] == -1:  # Original Hypatia check for no path
        return None

    curr = src
    path = []
    max_hops = 200
    count = 0
    while curr != dst:
        if count > max_hops:
            return None  # Loop guard
        if (curr, dst) not in forward_state:
            return None  # Path broken

        next_hop = forward_state[(curr, dst)]
        if next_hop == -1:
            return None  # Explicit no path

        if not sat_net_graph_with_gs.has_edge(curr, next_hop):
            # print(f"Debug get_path_with_weights: Edge ({curr},{next_hop}) not in graph required for weights.")
            return None  # Path invalid if edge weights are needed but edge doesn't exist

        w = sat_net_graph_with_gs.get_edge_data(curr, next_hop)["weight"]
        path.append((curr, next_hop, w))
        curr = next_hop
        count += 1
    return path


def augment_path_with_weights(path, sat_net_graph_with_gs):
    res = []
    if path is None or len(path) < 2:
        return res
    for i in range(1, len(path)):
        u_node, v_node = path[i - 1], path[i]
        if not sat_net_graph_with_gs.has_edge(u_node, v_node):
            # print(f"Debug augment_path_with_weights: Edge ({u_node},{v_node}) not in graph.")
            return []  # Cannot augment if edge doesn't exist
        res.append((u_node, v_node, sat_net_graph_with_gs.get_edge_data(u_node, v_node)["weight"]))
    return res


def sum_path_weights(weighted_path):
    res = 0.0
    if weighted_path is None:
        return res
    for i in weighted_path:
        res += i[2]
    return res
