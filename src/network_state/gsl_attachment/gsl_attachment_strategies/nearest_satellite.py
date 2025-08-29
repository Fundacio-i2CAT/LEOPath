from typing import List, Tuple

from astropy.time import Time

from src import logger
from src.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy
from src.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from src.topology.topology import LEOTopology, GroundStation
from src.topology import distance_tools

log = logger.get_logger(__name__)


class NearestSatelliteStrategy(GSLAttachmentStrategy):
    """Attaches each ground station to its nearest visible satellites."""

    def name(self) -> str:
        return "nearest_satellite"

    def select_attachments(
        self, topology: LEOTopology, ground_stations: List[GroundStation], current_time: Time
    ) -> List[Tuple[float, int]]:
        """
        Find the nearest visible satellite for each ground station.

        Args:
            topology: Network topology with satellite positions
            ground_stations: List of ground stations
            current_time: The current simulation time for calculating satellite positions

        Returns:
            List of tuples (distance, satellite_id) where each tuple represents
            the nearest satellite to the corresponding ground station by index.
            Returns (-1, -1) for ground stations that cannot connect to any satellite.
        """
        result = []

        # Prepare time strings for distance calculation
        time_str_for_ephem = str(current_time.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3])
        epoch_str_for_ephem = topology.constellation_data.epoch

        for gs in ground_stations:
            nearest_satellite = (-1.0, -1)  # Default: no satellite found (distance, sat_id)
            min_distance = float("inf")

            for sat in topology.get_satellites():
                try:
                    distance = distance_tools.distance_m_ground_station_to_satellite(
                        gs,  # Pass GroundStation object
                        sat,  # Pass Satellite object
                        epoch_str_for_ephem,  # Pass epoch string
                        time_str_for_ephem,  # Pass formatted time string
                    )

                    max_gs_range = topology.constellation_data.max_gsl_length_m

                    # Check if satellite is visible and closer than current best
                    if distance <= max_gs_range and distance < min_distance:
                        min_distance = distance
                        nearest_satellite = (distance, sat.id)

                except Exception as e:
                    log.warning(
                        f"Error calculating distance for GS {gs.id} to satellite {sat.id}: {e}"
                    )

            result.append(nearest_satellite)
            if nearest_satellite[1] != -1:
                log.debug(
                    f"GS {gs.id}: Attached to satellite {nearest_satellite[1]} at distance {nearest_satellite[0]:.2f}m"
                )
            else:
                log.warning(f"GS {gs.id}: No visible satellites found")

        return result


# Register the strategy with the factory
GSLAttachmentFactory.register_strategy(NearestSatelliteStrategy)
