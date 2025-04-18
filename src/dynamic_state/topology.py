import ephem
import networkx as nx


class ConstellationData:
    def __init__(
        self,
        orbits: int,
        sats_per_orbit: int,
        epoch: str,
        max_gsl_length_m: float,
        max_isl_length_m: float,
        satellites: list[ephem.Body],
    ):
        """
        Class to hold the orbital configuration data.
        :param orbits: Number of orbits
        :param sats_per_orbit: Number of satellites per orbit
        :param epoch: In the TLE, the epoch is given with a Julian date of yyddd.fraction
            - ddd is actually one-based, meaning e.g. 18001 is 1st of January, or 2018-01-01 00:00.
            - As such, to convert it to Astropy Time, we add (ddd - 1) days to it.
            - See also: https://www.celestrak.com/columns/v04n03/#FAQ04
        :param max_gsl_length_m: Maximum ground station link length in meters
        :param max_isl_length_m: Maximum inter-satellite link length in meters
        """
        self.n_orbits = orbits
        self.n_sats_per_orbit = sats_per_orbit
        self.epoch = epoch
        self.max_gsl_length_m = max_gsl_length_m
        self.max_isl_length_m = max_isl_length_m
        self.number_of_satellites = orbits * sats_per_orbit
        self.satellites = satellites


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

    def __init__(self, id: int, ephem_obj_manual: ephem.Body, ephem_obj_direct: ephem.Body):
        # NOTE Position is not 100% accurate because ephemeris data contains also velocity, but in the code gets more readable
        self.position = SatelliteEphemeris(ephem_obj_manual, ephem_obj_direct)
        self.number_isls = 0
        self.number_gsls = 0
        self.id = id


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


class ISL:
    def __init__(self, sat1: Satellite, sat2: Satellite):
        """
        Class to represent an inter-satellite link (ISL) between two satellites.
        :param sat1: First satellite address
        :param sat2: Second satellite address
        """
        self.sat1 = sat1
        self.sat2 = sat2


class LEOTopology:
    def __init__(
        self,
        constellation_data: ConstellationData,
        ground_stations: list[GroundStation],
    ):
        """
        Class to represent the topology of a Low Earth Orbit (LEO) satellite network.
        :param graph: A NetworkX graph representing the satellite network topology.
        """
        self.graph = nx.Graph()
        self.constellation_data = constellation_data
        self.ground_stations = ground_stations
        self.number_of_ground_stations = len(ground_stations)
        # TODO This info is in the graph, probably we do not need it here. If for some reason we do, I think it should be placed inside the satellite object
        self.sat_neighbor_to_if: dict = {}  # TODO Specify the type of this dictionary
        self.number_of_isls = 0
        # TODO This info is probably in the graph. If we still need it, I think it should be placed inside the satellite object
        self.gsl_interfaces_info: list  # TODO Specify the type of this list

    def get_satellites(self) -> list[Satellite]:
        """
        Get the satellites in the constellation.
        :return: List of satellites
        """
        return self.constellation_data.satellites

    def get_satellite(self, id: int) -> Satellite:
        """
        Get a satellite by its ID.
        :param id: Satellite ID
        :return: Satellite object
        :raises KeyError: if satellite with the given ID is not found.
        """
        for satellite in self.constellation_data.satellites:
            if satellite.id == id:
                return satellite
        raise KeyError(f"Satellite with ID {id} not found in constellation data.")

    def get_ground_stations(self) -> list[GroundStation]:
        """
        Get the ground stations in the constellation.
        :return: List of ground stations
        """
        return self.ground_stations

    def get_ground_station(self, gid: int) -> GroundStation:
        """
        Get a ground station by its ID.
        :param gid: Ground station ID
        :return: Ground station object
        :raises KeyError: if ground station with the given ID is not found.
        """
        for gs in self.ground_stations:
            if gs.id == gid:
                return gs
        raise KeyError(f"Ground station with ID {gid} not found.")
