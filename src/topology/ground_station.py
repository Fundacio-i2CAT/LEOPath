from typing import Optional

from src.topology.satellite.topological_network_address import TopologicalNetworkAddress


class GroundStation:
    def __init__(
        self,
        gid: int,
        name: str,
        latitude_degrees_str: str,
        longitude_degrees_str: str,
        elevation_m_float: float,
        cartesian_x: float,
        cartesian_y: float,
        cartesian_z: float,
    ):
        """
        Class that represents a ground station.
        :param gid: Ground station ID
        :param name: Name of the ground station
        :param latitude_degrees_str: Latitude in degrees as a string
        :param longitude_degrees_str: Longitude in degrees as a string
        :param elevation_m_float: Elevation in meters
        :param cartesian_x: Cartesian X coordinate
        :param cartesian_y: Cartesian Y coordinate
        :param cartesian_z: Cartesian Z coordinate
        """
        self.id = gid
        self.name = name
        self.latitude_degrees_str = latitude_degrees_str
        self.longitude_degrees_str = longitude_degrees_str
        self.elevation_m_float = elevation_m_float
        self.cartesian_x = cartesian_x
        self.cartesian_y = cartesian_y
        self.cartesian_z = cartesian_z
        
        # Topological routing attributes
        self.sixgrupa_addr: Optional[TopologicalNetworkAddress] = None
        self.previous_attached_satellite_id: Optional[int] = None
