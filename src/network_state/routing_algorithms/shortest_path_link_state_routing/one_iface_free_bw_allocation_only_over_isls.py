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


from astropy.time import Time

from src import logger
from src.topology.topology import ConstellationData, GroundStation, LEOTopology
from src.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy

from .fstate_calculation import calculate_fstate_shortest_path_object_no_gs_relay

log = logger.get_logger(__name__)


def algorithm_free_one_only_over_isls(
    time_since_epoch_ns: int,
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,
    gsl_attachment_strategy: GSLAttachmentStrategy,
    current_time: Time,
    list_gsl_interfaces_info: list,  # Info about bandwidth per node/interface
) -> dict:
    """
    Calculates bandwidth and forwarding state (shortest paths via ISLs only, no GS relaying)
    and returns them as state objects, without writing to files.

    Assumptions:
    - "one": Each satellite/GS has one GSL interface (index 0).
    - "free": Bandwidth is managed per interface, and there is no strict requirement that
      bandwidth usage is reciprocated between nodes. Each node's interface tracks its own
      available bandwidth independently, allowing for flexible allocation without enforcing
      that if Node A can send to Node B, Node B must also be able to send to Node A.
    - "only_over_isls": Paths are GS -> Sat -> ... -> Sat -> GS.

    :param time_since_epoch_ns: Current time step relative to epoch (integer ns).
    :param constellation_data: Holds satellite list, counts, max lengths, epoch string.
    :param ground_stations: List of GroundStation objects.
    :param topology_with_isls: LEOTopology object containing the graph with ISL links calculated.
                               Also contains ISL interface mapping (sat_neighbor_to_if).
    :param gsl_attachment_strategy: Strategy for selecting which satellites are visible to each ground station.
    :param current_time: Current simulation time for satellite positioning.
    :param list_gsl_interfaces_info: List of dicts, one per sat/GS, with bandwidth info.
    :param prev_output: Dictionary containing 'fstate' and 'bandwidth' objects from the previous step.
    :param enable_verbose_logs: Boolean to enable detailed logging.
    :return: Dictionary containing the new 'fstate' and 'bandwidth' state objects.
    """
    log.debug(f"Running algorithm_free_one_only_over_isls for t={time_since_epoch_ns} ns")

    bandwidth_state = _calculate_bandwidth_state(
        constellation_data, ground_stations, list_gsl_interfaces_info
    )
    fstate = _calculate_forwarding_state(
        topology_with_isls, ground_stations, gsl_attachment_strategy, current_time
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


def _calculate_forwarding_state(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    gsl_attachment_strategy: GSLAttachmentStrategy,
    current_time: Time,
) -> dict:
    """
    Returns the forwarding state object using shortest path calculation.
    """
    try:
        fstate = calculate_fstate_shortest_path_object_no_gs_relay(
            topology_with_isls,
            ground_stations,
            gsl_attachment_strategy,
            current_time,
        )
        log.debug("Calculated forwarding state object.")
        return fstate
    except NameError:
        log.exception(
            "Failed to call 'calculate_fstate_shortest_path_object'. "
            "Ensure fstate_calculation.py has been refactored."
        )
        return {}
    except Exception as e:
        log.exception(f"Error during forwarding state calculation: {e}")
        return {}
