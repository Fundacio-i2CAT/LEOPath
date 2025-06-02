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

import os
import unittest

import src
from src.ground_stations import read_ground_stations_basic, read_ground_stations_extended
from src.topology.topology import GroundStation


class TestGroundStations(unittest.TestCase):

    def test_ground_stations_normal(self):
        """Tests reading a valid basic ground station file."""
        gs_file = "ground_stations.temp.txt"
        # Provide valid float strings for lat/lon/elv
        gs_content = "0,abc,33.0,11.0,77.0"
        with open(gs_file, "w+") as f_out:
            f_out.write(gs_content)
        try:
            ground_stations = read_ground_stations_basic(gs_file)  # Assuming fixed version
            self.assertEqual(1, len(ground_stations))
            self.assertTrue(
                isinstance(ground_stations[0], GroundStation),
                "Did not return a GroundStation object",
            )
            self.assertEqual(0, ground_stations[0].id)
            self.assertEqual("abc", ground_stations[0].name)
            self.assertEqual("33.0", ground_stations[0].latitude_degrees_str)
            self.assertEqual("11.0", ground_stations[0].longitude_degrees_str)
            self.assertEqual(77.0, ground_stations[0].elevation_m_float)
            self.assertTrue(
                hasattr(ground_stations[0], "cartesian_x"),
                "GroundStation object missing cartesian_x",
            )
            self.assertTrue(
                hasattr(ground_stations[0], "cartesian_y"),
                "GroundStation object missing cartesian_y",
            )
            self.assertTrue(
                hasattr(ground_stations[0], "cartesian_z"),
                "GroundStation object missing cartesian_z",
            )
            # Optional: Add a check for non-None or approximate value if geodetic2cartesian is reliable
            self.assertIsNotNone(ground_stations[0].cartesian_x)
            # Example (rough check based on lat/lon - replace with actual expected values if needed):
            # self.assertGreater(ground_stations[0].cartesian_x, 0)
            # self.assertGreater(ground_stations[0].cartesian_y, 0)
            # self.assertGreater(ground_stations[0].cartesian_z, 0)

        finally:
            if os.path.exists(gs_file):
                os.remove(gs_file)

    def test_ground_stations_valid(self):

        # Empty
        with open("ground_stations.temp.txt", "w+") as f_out:
            f_out.write("")
        self.assertEqual(0, len(read_ground_stations_basic("ground_stations.temp.txt")))
        os.remove("ground_stations.temp.txt")

        # Two lines
        with open("ground_stations.temp.txt", "w+") as f_out:
            f_out.write("0,abc,33,11,5\n")
            f_out.write("1,abc,33,11,5")
        self.assertEqual(2, len(read_ground_stations_basic("ground_stations.temp.txt")))
        os.remove("ground_stations.temp.txt")

    def test_ground_stations_invalid(self):

        # Missing column
        with open("ground_stations.temp.txt", "w+") as f_out:
            f_out.write("0,abc,33,11")
        try:
            read_ground_stations_basic("ground_stations.temp.txt")
            self.fail()
        except ValueError:
            self.assertTrue(True)
        os.remove("ground_stations.temp.txt")

        # Invalid non-ascending gid
        with open("ground_stations.temp.txt", "w+") as f_out:
            f_out.write("0,abc,33,11,5\n")
            f_out.write("0,abc,33,11,5")
        try:
            read_ground_stations_basic("ground_stations.temp.txt")
            self.fail()
        except ValueError:
            self.assertTrue(True)
        os.remove("ground_stations.temp.txt")

        # Missing column
        with open("ground_stations_extended.temp.txt", "w+") as f_out:
            f_out.write("0,abc,33,11,2,3,3")
        try:
            read_ground_stations_extended("ground_stations_extended.temp.txt")
            self.fail()
        except ValueError:
            self.assertTrue(True)
        os.remove("ground_stations_extended.temp.txt")

        # Invalid non-ascending gid
        with open("ground_stations_extended.temp.txt", "w+") as f_out:
            f_out.write("0,abc,33,11,5,2,3,3\n")
            f_out.write("0,abc,33,11,5,2,3,3")
        try:
            read_ground_stations_extended("ground_stations_extended.temp.txt")
            self.fail()
        except ValueError:
            self.assertTrue(True)
        os.remove("ground_stations_extended.temp.txt")
