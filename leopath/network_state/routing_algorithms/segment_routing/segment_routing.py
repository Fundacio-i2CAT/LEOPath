from astropy import units as astro_units
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from leopath.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

from .segment_routing_algorithm import algorithm_segment_routing


class SegmentRoutingAlgorithm(RoutingAlgorithm):
    """
    Segment routing over ISLs using limited segments.

    Default behavior uses two segments: first hop to a target orbital plane,
    then intra-plane routing to the destination's plane/satellite.
    """

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
        gsl_strategy = GSLAttachmentFactory.get_strategy("nearest_satellite")

        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        current_time = epoch + time_since_epoch_ns * astro_units.ns

        return algorithm_segment_routing(
            time_since_epoch_ns,
            constellation_data,
            ground_stations,
            topology_with_isls,
            gsl_strategy,
            current_time,
            list_gsl_interfaces_info,
            algorithm_params or {},
        )
