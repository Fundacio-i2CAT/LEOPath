from astropy import units as astro_units
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from leopath.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm

# Import to trigger strategy registration
# This import is necessary for the factory to have the strategy registered
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

from .one_iface_free_bw_allocation_only_over_isls import algorithm_free_one_only_over_isls

GS_ADDRESSING = ("visibility", "attachment")


def egresses_for_addressing(ground_station_satellites_in_range: list, gs_addressing: str) -> list:
    """Satellites each ground station can be reached through, under the addressing policy.

    Under ``visibility`` every satellite above the station's horizon is an egress.
    Under ``attachment`` the station holds one ground link, to its nearest live
    visible satellite, the rule topological routing uses to pick the attachment,
    so link-state routes to the same single egress. That makes it the like-for-like
    peer of topological routing under attachment addressing; ``visibility`` stays
    the any-egress optimum both are scored against.
    """
    if gs_addressing not in GS_ADDRESSING:
        raise ValueError(
            f"Unknown gs_addressing {gs_addressing!r}, expected one of {GS_ADDRESSING}"
        )
    if gs_addressing == "visibility":
        return ground_station_satellites_in_range
    return [
        [min(visible, key=lambda egress: egress[0])] if visible else []
        for visible in ground_station_satellites_in_range
    ]


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
        algorithm_params: dict | None = None,
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

        # Route toward the ground station, not toward one chosen satellite:
        # any satellite currently above the destination's horizon is a valid
        # egress, and the fstate calculation picks whichever minimises path
        # length plus GSL length. Attachment addressing narrows that to the one
        # satellite the station is attached to.
        return algorithm_free_one_only_over_isls(
            time_since_epoch_ns,
            constellation_data,
            ground_stations,
            topology_with_isls,
            gsl_strategy,
            current_time,
            list_gsl_interfaces_info,
            egresses_for_addressing(
                ground_station_satellites_in_range,
                str((algorithm_params or {}).get("gs_addressing", "visibility")),
            ),
        )
