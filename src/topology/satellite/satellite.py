from typing import Optional

import ephem

from src.topology.satellite.topological_network_address import TopologicalNetworkAddress


class SatelliteEphemeris:

    def __init__(self, ephem_obj_manual: ephem.Body, ephem_obj_direct: ephem.Body):
        """
        Class to hold the ephemeris data of a satellite.
        :param ephem_obj: Object representing the ephemeris data.
        :param ephem_obj_direct: Object representing the direct ephemeris data.
        """
        self.ephem_obj_manual = ephem_obj_manual
        self.ephem_obj_direct = ephem_obj_direct


class Satellite:
    """
    Class to a represent a satellite within a constellation.

    :param ephem_obj_manual: Object representing the manual ephemeris data.
    :param ephem_obj_direct: Object representing the direct ephemeris data.
    """

    def __init__(
        self,
        id: int,
        ephem_obj_manual: ephem.Body,
        ephem_obj_direct: ephem.Body,
        orbital_plane_id: Optional[int] = None,
        satellite_id: Optional[int] = None,
        sixgrupa_addr: Optional[TopologicalNetworkAddress] = None,
    ):
        """
        Class to represent a satellite within a constellation.
        :param id: Satellite ID
        :param ephem_obj_manual: Object representing the manual ephemeris data.
        :param ephem_obj_direct: Object representing the direct ephemeris data.
        :param 6grupa_addr: Optional address to be used in 6G-RUPA-based networks
        """
        self.position = SatelliteEphemeris(ephem_obj_manual, ephem_obj_direct)
        self.number_isls = 0
        self.number_gsls = 0
        self.id = id
        self.sixgrupa_addr = sixgrupa_addr
        self.orbital_plane_id = orbital_plane_id
        self.satellite_id = satellite_id
        self.forwarding_table: dict[int, int] = (
            {}
        )  # Maps 6G-RUPA (serialized) address  to interface number

    def get_6grupa_addr_from(
        self, neighbor_satellite_id: int
    ) -> Optional[TopologicalNetworkAddress]:
        """
        Get the 6G-RUPA address of a neighbor satellite.
        :param neighbor_satellite_id: Neighbor satellite ID
        :return: 6G-RUPA address of the neighbor satellite or None if not available
        """
        try:
            # Generate the topological address for the neighbor satellite
            return TopologicalNetworkAddress.set_address_from_orbital_parameters(neighbor_satellite_id)
        except Exception:
            # Log the error but don't crash the system
            return None
