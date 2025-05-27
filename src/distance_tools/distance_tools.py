# The MIT License (MIT)
#
# Copyright (c) 2020 ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import ephem
from geopy.distance import great_circle  # Keep existing imports

from src import logger

# Import your Satellite class definition (adjust path if necessary)
from src.topology.topology import GroundStation, Satellite

log = logger.get_logger(__name__)


def distance_m_between_satellites(
    sat1: Satellite, sat2: Satellite, epoch_str: str, date_str: str
) -> float:
    """
    Computes the straight distance between two satellites in meters.

    Accepts custom Satellite wrapper objects.

    :param sat1:       The first Satellite object.
    :param sat2:       The other Satellite object.
    :param epoch_str:  Epoch time string (e.g., "2000-01-01 00:00:00").
    :param date_str:   The time instant string (e.g., "2000-01-01 00:00:00").

    :return: The distance between the satellites in meters (float).
    :raises AttributeError: If satellite objects lack expected position/ephem attributes.
    :raises ValueError: If ephem objects are invalid.
    :raises RuntimeError: For other ephem calculation errors.
    """
    try:
        # 1. Extract underlying ephem.Body objects
        #    Choose manual or direct based on which one is populated/needed
        if not hasattr(sat1, "position") or not hasattr(sat1.position, "ephem_obj_manual"):
            raise AttributeError(
                f"Satellite object for sat_id {getattr(sat1, 'id', 'UNKNOWN')} is missing position or ephem_obj_manual"
            )
        ephem_body1 = sat1.position.ephem_obj_manual  # Or ephem_obj_direct

        if not hasattr(sat2, "position") or not hasattr(sat2.position, "ephem_obj_manual"):
            raise AttributeError(
                f"Satellite object for sat_id {getattr(sat2, 'id', 'UNKNOWN')} is missing position or ephem_obj_manual"
            )
        ephem_body2 = sat2.position.ephem_obj_manual  # Or ephem_obj_direct

        if not isinstance(ephem_body1, ephem.Body) or not isinstance(ephem_body2, ephem.Body):
            raise ValueError("Extracted ephem objects are not valid ephem.Body types.")

        # 2. Create an observer (position doesn't strictly matter for separation angle,
        #    but range calculation depends on it - using 0,0 is fine here)
        observer = ephem.Observer()
        observer.epoch = epoch_str  # TLE epoch
        observer.date = date_str  # Current time
        observer.lat = "0"  # degrees string
        observer.lon = "0"  # degrees string
        observer.elevation = 0.0

        # 3. Calculate the relative location by computing the ephem.Body objects
        ephem_body1.compute(observer)
        ephem_body2.compute(observer)

        # 4. Calculate the separation angle using the computed ephem.Body objects
        # Ensure ephem.separation result is converted correctly (it's often an ephem.Angle)
        angle_radians = float(ephem.separation(ephem_body1, ephem_body2))

        # 5. Get ranges from the *computed* ephem.Body objects
        range1 = ephem_body1.range
        range2 = ephem_body2.range

        # Check for potential issues (e.g., satellite below horizon for the arbitrary observer)
        if range1 is None or range2 is None or range1 <= 0 or range2 <= 0:
            # This might happen if a satellite is below the horizon for the observer at 0,0
            # The geometric approach might be less stable in such cases.
            # Consider alternative (e.g., Cartesian distance) if this occurs frequently.
            # For now, return infinity or raise an error.
            # print(f"Warning: Invalid range detected for ISL distance ({range1}, {range2})") # Use logger
            return float("inf")  # Indicate invalid distance

        # 6. Calculate distance using Law of Cosines
        # c^2 = a^2 + b^2 - 2 * a * b * cos(C)
        cos_term = 2 * range1 * range2 * math.cos(angle_radians)
        distance_sq = (range1**2) + (range2**2) - cos_term

        # Handle potential floating point issues near zero
        if distance_sq < 0:
            # This can happen due to floating point errors if angle is near 0 or pi
            # and ranges are almost equal. Treat distance as near zero.
            # print(f"Warning: Negative value in sqrt for ISL distance ({distance_sq}). Clamping to 0.") # Use logger
            return 0.0

        distance_m = math.sqrt(distance_sq)
        return distance_m

    except (AttributeError, ValueError) as e:
        log.error(f"[distance_tools] Input Error calculating ISL distance: {e}")  # Use logger
        raise e  # Re-raise configuration/input errors
    except Exception as e:
        log.error(f"[distance_tools] Runtime Error calculating ISL distance: {e}")  # Use logger
        raise


def distance_m_ground_station_to_satellite(
    ground_station: GroundStation,  # Expect GroundStation object
    satellite: Satellite,  # Expect Satellite object
    epoch_str: str,  # Expect epoch string ('YYYY/MM/DD' format recommended)
    date_str: str,  # Expect date string ('YYYY/MM/DD HH:MM:SS.sss' recommended)
) -> float:
    """
    Computes the straight distance in meters between a ground station and a satellite.

    Accepts GroundStation and Satellite wrapper objects.

    :param ground_station: The GroundStation object.
    :param satellite:      The Satellite object (must contain ephem.Body).
    :param epoch_str:      Epoch time string for the observer (e.g., 'YYYY/MM/DD').
    :param date_str:       The time instant string for the observer (e.g., 'YYYY/MM/DD HH:MM:SS.sss').

    :return: The distance between the ground station and the satellite in meters (float).
             Returns float('inf') if satellite is below horizon or on calculation error.
    :raises AttributeError: If objects lack expected attributes.
    :raises ValueError: If ephem objects/data are invalid or time strings cannot be parsed.
    """
    gs_id_str = f"GS {getattr(ground_station, 'id', 'UNKNOWN')}"
    sat_id_str = f"Sat {getattr(satellite, 'id', 'UNKNOWN')}"
    try:
        # 1. Validate GroundStation object and extract data
        if not isinstance(ground_station, GroundStation):
            raise TypeError(f"Expected GroundStation object, got {type(ground_station)}")
        # Check for necessary attributes using hasattr
        if not all(
            hasattr(ground_station, attr)
            for attr in ["latitude_degrees_str", "longitude_degrees_str", "elevation_m_float"]
        ):
            raise AttributeError(
                "GroundStation object missing required attributes (lat, lon, elv)."
            )
        gs_lat_str = str(ground_station.latitude_degrees_str)
        gs_lon_str = str(ground_station.longitude_degrees_str)
        gs_elev_float = float(ground_station.elevation_m_float)

        # 2. Extract ephem.Body object from Satellite object
        if not isinstance(satellite, Satellite):
            raise TypeError(f"Expected Satellite object, got {type(satellite)}")
        if not hasattr(satellite, "position") or not hasattr(
            satellite.position, "ephem_obj_manual"
        ):  # Or _direct
            raise AttributeError(
                f"Satellite object for {sat_id_str} missing position/ephem_obj_manual"
            )
        ephem_body = satellite.position.ephem_obj_manual  # Or _direct

        if not isinstance(ephem_body, ephem.Body):
            raise ValueError(
                f"Extracted ephem object for {sat_id_str} is not a valid ephem.Body type."
            )

        # 3. Create ephem.Observer from GroundStation object data
        observer = ephem.Observer()
        observer.lat = gs_lat_str
        observer.lon = gs_lon_str
        observer.elevation = gs_elev_float

        # 4. Set observer time context using the provided strings
        try:
            observer.epoch = epoch_str
            observer.date = date_str
        except ValueError as e:
            raise ValueError(
                f"Invalid date/epoch format for ephem Observer. "
                f"Received date='{date_str}', epoch='{epoch_str}'. Original error: {e}"
            ) from e

        # 5. Compute satellite position relative to observer
        ephem_body.compute(observer)

        # 6. Return distance (ephem's range is in meters)
        if ephem_body.alt < 0:  # Check if below horizon
            return float("inf")
        if ephem_body.range is None:  # Should not happen if above horizon, but check
            log.error(
                f"[distance_tools] Warning: ephem range is None for {sat_id_str} from {gs_id_str} even though alt>=0."
            )  # Use logger
            return float("inf")

        return float(ephem_body.range)

    except (AttributeError, ValueError, TypeError) as e:
        log.error(
            f"[distance_tools] Input/Type Error calculating GSL distance for {gs_id_str} <-> {sat_id_str}: {e}"
        )  # Use logger
        return float("inf")  # Return inf on error
    except Exception as e:
        log.error(
            f"[distance_tools] Runtime Error calculating GSL distance for {gs_id_str} <-> {sat_id_str}: {e}"
        )  # Use logger
        return float("inf")  # Return inf on error


def geodesic_distance_m_between_ground_stations(
    ground_station_1: GroundStation,  # Expect object
    ground_station_2: GroundStation,  # Expect object
) -> float:
    """
    Calculate the geodesic distance between two ground stations.
    Accepts GroundStation objects.

    :param ground_station_1: First GroundStation object.
    :param ground_station_2: Another GroundStation object.
    :return: Geodesic distance in meters.
    :raises AttributeError: If objects lack required attributes.
    """
    if not isinstance(ground_station_1, GroundStation) or not isinstance(
        ground_station_2, GroundStation
    ):
        raise TypeError("Inputs must be GroundStation objects.")
    if (
        not hasattr(ground_station_1, "latitude_degrees_str")
        or not hasattr(ground_station_1, "longitude_degrees_str")
        or not hasattr(ground_station_2, "latitude_degrees_str")
        or not hasattr(ground_station_2, "longitude_degrees_str")
    ):
        raise AttributeError("GroundStation objects missing lat/lon attributes.")

    earth_radius_km = 6378.135

    try:
        lat1 = float(ground_station_1.latitude_degrees_str)  # Use attribute access
        lon1 = float(ground_station_1.longitude_degrees_str)  # Use attribute access
        lat2 = float(ground_station_2.latitude_degrees_str)  # Use attribute access
        lon2 = float(ground_station_2.longitude_degrees_str)  # Use attribute access

        distance = great_circle((lat1, lon1), (lat2, lon2), radius=earth_radius_km).m
        return distance
    except (ValueError, TypeError) as e:
        print(
            f"[distance_tools] Error converting lat/lon to float for GS distance: {e}"
        )  # Use logger
        raise ValueError("Invalid lat/lon format in GroundStation object") from e


def straight_distance_m_between_ground_stations(
    ground_station_1: GroundStation,  # Expect object
    ground_station_2: GroundStation,  # Expect object
) -> float:
    """
    Calculate the straight distance between two ground stations (goes through the Earth).
    Accepts GroundStation objects.

    :param ground_station_1: First GroundStation object.
    :param ground_station_2: Another GroundStation object.
    :return: Straight distance in meters (goes through the Earth).
    """
    # WGS72 value
    earth_radius_m = 6378135.0

    # Calculate geodesic distance first (now uses objects correctly)
    geo_dist_m = geodesic_distance_m_between_ground_stations(ground_station_1, ground_station_2)

    # Check for potential issues with geo_dist calculation (e.g., identical points)
    if geo_dist_m == 0:
        return 0.0
    if earth_radius_m <= 0:  # Avoid division by zero
        return float("inf")  # Or raise error

    # Calculate angle - handle potential division by zero if radius is bad
    try:
        fraction_of_earth_circumference = geo_dist_m / (earth_radius_m * 2.0 * math.pi)
        angle_radians = fraction_of_earth_circumference * 2 * math.pi
    except ZeroDivisionError:
        print(
            "[distance_tools] Error: Earth radius is zero in straight distance calc."
        )  # Use logger
        return float("inf")  # Or raise

    # Calculate polygon side (straight distance)
    polygon_side_m = 2 * math.sin(angle_radians / 2.0) * earth_radius_m

    return polygon_side_m


def create_basic_ground_station_for_satellite_shadow(satellite, epoch_str, date_str):
    """
    Calculate the (latitude, longitude) of the satellite shadow on the Earth and creates a ground station there.

    :param satellite:   Satellite
    :param epoch_str:   Epoch (string)
    :param date_str:    Time moment (string)

    :return: Basic ground station
    """

    satellite.compute(date_str, epoch=epoch_str)

    return {
        "gid": -1,
        "name": "Shadow of " + satellite.name,
        "latitude_degrees_str": str(math.degrees(satellite.sublat)),
        "longitude_degrees_str": str(math.degrees(satellite.sublong)),
        "elevation_m_float": 0,
    }


def geodetic2cartesian(lat_degrees, lon_degrees, ele_m):
    """
    Compute geodetic coordinates (latitude, longitude, elevation) to Cartesian coordinates.

    :param lat_degrees: Latitude in degrees (float)
    :param lon_degrees: Longitude in degrees (float)
    :param ele_m:  Elevation in meters

    :return: Cartesian coordinate as 3-tuple of (x, y, z)
    """

    #
    # Adapted from: https://github.com/andykee/pygeodesy/blob/master/pygeodesy/transform.py
    #

    # WGS72 value,
    # Source: https://geographiclib.sourceforge.io/html/NET/NETGeographicLib_8h_source.html
    a = 6378135.0

    # Ellipsoid flattening factor; WGS72 value
    # Taken from https://geographiclib.sourceforge.io/html/NET/NETGeographicLib_8h_source.html
    f = 1.0 / 298.26

    # First numerical eccentricity of ellipsoid
    e = math.sqrt(2.0 * f - f * f)
    lat = lat_degrees * (math.pi / 180.0)
    lon = lon_degrees * (math.pi / 180.0)

    # Radius of curvature in the prime vertical of the surface of the geodetic ellipsoid
    v = a / math.sqrt(1.0 - e * e * math.sin(lat) * math.sin(lat))

    x = (v + ele_m) * math.cos(lat) * math.cos(lon)
    y = (v + ele_m) * math.cos(lat) * math.sin(lon)
    z = (v * (1.0 - e * e) + ele_m) * math.sin(lat)

    return x, y, z
