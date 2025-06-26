# Import all GSL attachment strategies to ensure they register themselves
from .nearest_satellite import NearestSatelliteStrategy

__all__ = ["NearestSatelliteStrategy"]
