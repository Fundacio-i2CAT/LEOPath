from src.topology.satellite.satellite import Satellite


class ISL:
    def __init__(self, sat1: Satellite, sat2: Satellite):
        """
        Class to represent an inter-satellite link (ISL) between two satellites.
        :param sat1: First satellite address
        :param sat2: Second satellite address
        """
        self.sat1 = sat1
        self.sat2 = sat2
