from astropy import units as astro_units
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from leopath.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

from .traditional_segment_routing_algorithm import algorithm_traditional_segment_routing


class TraditionalSegmentRoutingAlgorithm(RoutingAlgorithm):
    """
    Traditional segment routing baseline with explicit segment lists.

    The control plane computes a per-(source sat, destination GS) segment list
    [sid1, sid2, ..., dst_sat], where each sid is a satellite node ID. During
    forwarding-table materialization, each node follows shortest path to the
    current active segment endpoint.
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

        return algorithm_traditional_segment_routing(
            constellation_data,
            ground_stations,
            topology_with_isls,
            gsl_strategy,
            current_time,
            list_gsl_interfaces_info,
            algorithm_params or {},
        )
