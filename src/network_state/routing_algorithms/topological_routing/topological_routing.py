from astropy.time import Time
from astropy import units as astro_units

from src.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from src.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from src.topology.topology import ConstellationData, GroundStation, LEOTopology

from .algorithm_topological_routing import algorithm_topological_routing


class TopologicalRoutingAlgorithm(RoutingAlgorithm):
    """
    Routing algorithm using topological routing (ISLs only, no GS relaying).

    This algorithm implements topological routing with the following key features:
    - Assigns 6GRUPA addresses to all satellites at t=0
    - Populates forwarding tables based on topological neighbor relationships
    - Handles satellite-to-ground-station routing without GS relays
    - Supports both direct GSL connections and multi-hop ISL paths
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
        # Get the GSL attachment strategy (default to nearest satellite, same as shortest path)
        gsl_strategy = GSLAttachmentFactory.get_strategy("nearest_satellite")

        # Create a current_time object to match the pattern used in generate_network_state.py
        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        current_time = epoch + time_since_epoch_ns * astro_units.ns

        return algorithm_topological_routing(
            time_since_epoch_ns,
            constellation_data,
            ground_stations,
            topology_with_isls,
            gsl_strategy,
            current_time,
            list_gsl_interfaces_info,
        )
