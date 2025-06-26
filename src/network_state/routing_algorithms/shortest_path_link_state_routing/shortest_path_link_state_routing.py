from astropy.time import Time
from astropy import units as astro_units

from src.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from src.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
# Import to trigger strategy registration
# This import is necessary for the factory to have the strategy registered
from src.network_state.gsl_attachment.gsl_attachment_strategies import nearest_satellite  # noqa: F401
from src.topology.topology import ConstellationData, GroundStation, LEOTopology

from .one_iface_free_bw_allocation_only_over_isls import algorithm_free_one_only_over_isls


class ShortestPathLinkStateRoutingAlgorithm(RoutingAlgorithm):
    """
    Routing algorithm using shortest path link-state routing (ISLs only, no GS relaying).
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
        # Get the GSL attachment strategy (default to nearest satellite)
        gsl_strategy = GSLAttachmentFactory.get_strategy("nearest_satellite")
        
        # Create a current_time object to match the pattern used in generate_network_state.py
        # Use the same epoch and time calculation as the working system
        # This should match: time_absolute = epoch + time_since_epoch_ns * astro_units.ns
        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        current_time = epoch + time_since_epoch_ns * astro_units.ns
        
        return algorithm_free_one_only_over_isls(
            time_since_epoch_ns,
            constellation_data,
            ground_stations,
            topology_with_isls,
            gsl_strategy,
            current_time,
            list_gsl_interfaces_info,
        )
