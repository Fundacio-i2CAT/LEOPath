from leopath.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

from .algorithm_topological_routing import (
    _calculate_bandwidth_state,
    algorithm_topological_routing,
)


class TopologicalRoutingAlgorithm(RoutingAlgorithm):
    """
    Routing algorithm using topological routing (ISLs only, no GS relaying).

    This algorithm implements topological routing with the following key features:
    - Assigns 6GRUPA addresses to all satellites at t=0
    - Populates forwarding tables based on topological neighbor relationships
    - Handles satellite-to-ground-station routing without GS relays
    - Supports both direct GSL connections and multi-hop ISL paths
    """

    def __init__(self) -> None:
        self._cached_bandwidth_signature: tuple | None = None
        self._cached_bandwidth_state: dict | None = None

    def compute_state(
        self,
        time_since_epoch_ns: int,
        constellation_data: ConstellationData,
        ground_stations: list[GroundStation],
        topology_with_isls: LEOTopology,
        ground_station_satellites_in_range: list,
        list_gsl_interfaces_info: list,
        algorithm_params: dict | None = None,
    ) -> dict:
        """
        Calculates bandwidth and forwarding state for the current network state using topological routing.

        Args:
            time_since_epoch_ns: Current time step relative to epoch (integer ns)
            constellation_data: Holds satellite list, counts, max lengths, epoch string
            ground_stations: List of GroundStation objects
            topology_with_isls: LEOTopology object containing the graph with ISL links
            ground_station_satellites_in_range: List where index=gs_idx, value=list of (distance, sat_id) tuples
            list_gsl_interfaces_info: List of dicts, one per sat/GS, with bandwidth info

        Returns:
            Dictionary containing the new 'fstate' and 'bandwidth' state objects
        """
        bandwidth_signature = tuple(
            (
                node_info.get("id", index),
                node_info.get("aggregate_max_bandwidth", 0.0),
            )
            for index, node_info in enumerate(list_gsl_interfaces_info)
        )
        if bandwidth_signature != self._cached_bandwidth_signature:
            self._cached_bandwidth_state = _calculate_bandwidth_state(
                constellation_data,
                ground_stations,
                list_gsl_interfaces_info,
            )
            self._cached_bandwidth_signature = bandwidth_signature

        nearest_satellite_attachments = []
        for visible_satellites in ground_station_satellites_in_range:
            if visible_satellites:
                nearest_satellite_attachments.append(
                    [min(visible_satellites, key=lambda item: item[0])]
                )
            else:
                nearest_satellite_attachments.append([])

        return algorithm_topological_routing(
            time_since_epoch_ns,
            constellation_data,
            ground_stations,
            topology_with_isls,
            nearest_satellite_attachments,
            self._cached_bandwidth_state or {},
            prev_fstate=None,
            graph_has_changed=True,
        )
