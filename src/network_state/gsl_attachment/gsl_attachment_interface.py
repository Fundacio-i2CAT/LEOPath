from abc import ABC, abstractmethod
from typing import List, Tuple

from astropy.time import Time

from src.topology.topology import LEOTopology, GroundStation


class GSLAttachmentStrategy(ABC):
    """Interface for ground station to satellite attachment strategies."""

    @abstractmethod
    def name(self) -> str:
        """Return the name of the strategy."""
        pass

    @abstractmethod
    def select_attachments(
        self,
        topology: LEOTopology,
        ground_stations: List[GroundStation],
        current_time: Time
    ) -> List[Tuple[float, int]]:
        """
        Select a single attachment point for each ground station.

        Args:
            topology: Network topology containing satellites and links
            ground_stations: List of ground stations to connect
            current_time: The current simulation time for calculating satellite positions

        Returns:
            List of tuples (distance, satellite_id) where each tuple represents
            the selected attachment for the corresponding ground station by index.
            Returns (-1, -1) for ground stations that cannot connect to any satellite.
        """
        pass
