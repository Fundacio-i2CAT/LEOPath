from src.topology.satellite.satellite import Satellite


class ConstellationData:
    def __init__(
        self,
        orbits: int,
        sats_per_orbit: int,
        epoch: str,
        max_gsl_length_m: float,
        max_isl_length_m: float,
        satellites: list[Satellite],
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
