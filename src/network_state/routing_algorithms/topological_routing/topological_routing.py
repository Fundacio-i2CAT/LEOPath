from src.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from src.topology.topology import ConstellationData, GroundStation, LEOTopology


class TopologicalRoutingAlgorithm(RoutingAlgorithm):
    """
    Routing algorithm using topological information (ISLs only, no GS relaying).
    """

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
        """
        return algorithm_free_one_only_over_isls(
            time_since_epoch_ns,
            constellation_data,
            ground_stations,
            topology_with_isls,
            ground_station_satellites_in_range,
            list_gsl_interfaces_info,
        )
