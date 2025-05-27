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
from src.distance_tools import geodetic2cartesian
from src.topology.topology import GroundStation


def read_ground_stations_basic(filename_ground_stations_basic: str) -> list[GroundStation]:
    """
    Reads ground stations from the input file (basic format) and
    returns a list of GroundStation objects. Calculates Cartesian coordinates.

    Expected format per line:
    gid,name,latitude_degrees,longitude_degrees,elevation_m

    :param filename_ground_stations_basic: Filename of ground stations basic.
    :return: List of GroundStation objects.
    :raises FileNotFoundError: If the file cannot be opened.
    :raises ValueError: If file format is incorrect or data cannot be parsed.
    """
    ground_stations_list = []
    gid_counter = 0
    try:
        with open(filename_ground_stations_basic, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):  # Skip empty/comment lines
                    continue

                parts = line.split(",")
                if len(parts) != 5:
                    raise ValueError(
                        f"Basic ground station file expects 5 columns, "
                        f"got {len(parts)} on line {line_num}: {line}"
                    )

                try:
                    # Parse basic data
                    gid_from_file = int(parts[0])
                    name_str = parts[1]
                    lat_str = parts[2]
                    lon_str = parts[3]
                    ele_float = float(parts[4])

                    # Validate sequential GID
                    if gid_from_file != gid_counter:
                        raise ValueError(
                            f"Ground station id must increment sequentially starting from 0. "
                            f"Expected {gid_counter}, got {gid_from_file} on line {line_num}."
                        )

                    # Calculate Cartesian coordinates
                    lat_float = float(lat_str)
                    lon_float = float(lon_str)
                    cart_x, cart_y, cart_z = geodetic2cartesian(lat_float, lon_float, ele_float)

                    # Create GroundStation object
                    gs_object = GroundStation(
                        gid=gid_counter,  # Use counter as the reliable ID
                        name=name_str,
                        latitude_degrees_str=lat_str,
                        longitude_degrees_str=lon_str,
                        elevation_m_float=ele_float,
                        cartesian_x=cart_x,
                        cartesian_y=cart_y,
                        cartesian_z=cart_z,
                    )
                    ground_stations_list.append(gs_object)
                    gid_counter += 1

                except ValueError as e:  # Catch errors during conversion (int, float)
                    raise ValueError(f"Error parsing data on line {line_num}: {line} - {e}") from e
                except Exception as e:  # Catch errors from geodetic2cartesian
                    raise RuntimeError(
                        f"Error calculating cartesian coords for line {line_num}: {line} - {e}"
                    ) from e

    except FileNotFoundError:
        print(
            f"[ERROR] Ground station file not found: {filename_ground_stations_basic}"
        )  # Use logger
        raise  # Re-raise the specific error
    except Exception as e:
        print(
            f"[ERROR] Failed to read ground stations from {filename_ground_stations_basic}: {e}"
        )  # Use logger
        raise  # Re-raise other errors

    return ground_stations_list


def read_ground_stations_extended(filename_ground_stations_extended):
    # TODO: Consider refactoring this to return list[GroundStation]
    """
    Reads ground stations from the input file.

    :param filename_ground_stations_extended: Filename of ground stations basic (typically /path/to/ground_stations.txt)

    :return: List of ground stations
    """
    ground_stations_extended = []
    gid = 0
    with open(filename_ground_stations_extended, "r") as f:
        for line in f:
            split = line.split(",")
            if len(split) != 8:
                raise ValueError("Extended ground station file has 8 columns: " + line)
            if int(split[0]) != gid:
                raise ValueError("Ground station id must increment each line")
            ground_station_basic = {
                "gid": gid,
                "name": split[1],
                "latitude_degrees_str": split[2],
                "longitude_degrees_str": split[3],
                "elevation_m_float": float(split[4]),
                "cartesian_x": float(split[5]),
                "cartesian_y": float(split[6]),
                "cartesian_z": float(split[7]),
            }
            ground_stations_extended.append(ground_station_basic)
            gid += 1
    return ground_stations_extended
