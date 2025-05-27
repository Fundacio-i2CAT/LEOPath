import networkx as nx
from src.topology.constellation import ConstellationData
from src.topology.satellite.satellite import Satellite
from src.topology.ground_station import GroundStation


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
