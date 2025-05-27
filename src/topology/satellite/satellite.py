import ephem
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
