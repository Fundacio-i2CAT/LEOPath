from abc import ABC, abstractmethod

from src.topology.topology import ConstellationData, GroundStation, LEOTopology


class RoutingAlgorithm(ABC):
    """
    Abstract base class for routing algorithms.
    """

    @abstractmethod
    def compute_state(
        self,
        time_since_epoch_ns: int,
        constellation_data: ConstellationData,
        ground_stations: list[GroundStation],
        topology_with_isls: LEOTopology,
        ground_station_satellites_in_range: list,
        list_gsl_interfaces_info: list,
    ) -> dict:
        """
        Calculates bandwidth and forwarding state for the current network state.

        Assumptions:
        - Each satellite/GS has one GSL interface (index 0).
        - Bandwidth is managed per interface, independently for each node.
        - Only ISL paths are considered (GS -> Sat -> ... -> Sat -> GS), no GS relaying.

        :param time_since_epoch_ns: Current time step relative to epoch (integer ns).
        :param constellation_data: Holds satellite list, counts, max lengths, epoch string.
        :param ground_stations: List of GroundStation objects.
        :param topology_with_isls: LEOTopology object containing the graph with ISL links calculated.
                                   Also contains ISL interface mapping (sat_neighbor_to_if).
        :param ground_station_satellites_in_range: List where index=gs_idx, value=list of (distance, sat_id) tuples visible to that GS.
        :param list_gsl_interfaces_info: List of dicts, one per sat/GS, with bandwidth info.
        :return: Dictionary containing the new 'fstate' and 'bandwidth' state objects.
        """
        pass
