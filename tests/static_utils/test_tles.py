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

from src.tles.generate_tles_from_scratch import (
    generate_tles_from_scratch_manual,
    generate_tles_from_scratch_with_sgp,
)
from src.tles.read_tles import read_tles, satellite_ephem_to_str
import unittest
import os
from astropy.time import Time
import math
import ephem


class TestTles(unittest.TestCase):

    def tles_generation(
        self,
        constellation_name,
        num_orbits,
        num_sats_per_orbit,
        phase_diff,
        inclination_degree,
        eccentricity,
        arg_of_perigee_degree,
        mean_motion_rev_per_day,
    ):

        # Manual
        generate_tles_from_scratch_manual(
            "tles_manual.txt.tmp",
            constellation_name,
            num_orbits,
            num_sats_per_orbit,
            phase_diff,
            inclination_degree,
            eccentricity,
            arg_of_perigee_degree,
            mean_motion_rev_per_day,
        )

        # Using SGP
        generate_tles_from_scratch_with_sgp(
            "tles_sgp.txt.tmp",
            constellation_name,
            num_orbits,
            num_sats_per_orbit,
            phase_diff,
            inclination_degree,
            eccentricity,
            arg_of_perigee_degree,
            mean_motion_rev_per_day,
        )

        # Now compare they are the same
        manual_lines = []
        with open("tles_manual.txt.tmp", "r") as f_in:
            for line in f_in:
                manual_lines.append(line.strip())
        sgp_lines = []
        with open("tles_sgp.txt.tmp", "r") as f_in:
            for line in f_in:
                sgp_lines.append(line.strip())
        self.assertEqual(manual_lines, sgp_lines)

        # Now read both of them in
        tles_manual = read_tles("tles_manual.txt.tmp")
        tles_sgp = read_tles("tles_sgp.txt.tmp")

        # The files were already identical, but just to be sure, a quick check
        self.assertEqual(tles_manual["n_orbits"], tles_sgp["n_orbits"])
        self.assertEqual(tles_manual["n_sats_per_orbit"], tles_sgp["n_sats_per_orbit"])
        self.assertEqual(tles_manual["epoch"], tles_sgp["epoch"])
        self.assertEqual(len(tles_manual["satellites"]), len(tles_sgp["satellites"]))
        for i in range(num_orbits * num_sats_per_orbit):
            # Direct equal does not work as in the string they added a memory address
            self.assertEqual(
                satellite_ephem_to_str(tles_sgp["satellites"][i]),
                satellite_ephem_to_str(tles_manual["satellites"][i]),
            )

        # Now we just continue with tles_manual to test it set the values correctly
        self.assertEqual(tles_manual["n_orbits"], num_orbits)
        self.assertEqual(tles_manual["n_sats_per_orbit"], num_sats_per_orbit)
        self.assertEqual(tles_manual["epoch"], Time("2000-01-01 00:00:00", scale="tdb"))
        for i in range(num_orbits * num_sats_per_orbit):
            orbit = int(math.floor(i / num_sats_per_orbit))
            orbit_wise_shift = 0
            if orbit % 2 == 1:
                if phase_diff:
                    orbit_wise_shift = 360.0 / (num_sats_per_orbit * 2.0)
            mean_anomaly_degree = orbit_wise_shift + (
                (i % num_sats_per_orbit) * 360 / num_sats_per_orbit
            )

            self.assertEqual(tles_manual["satellites"][i]._ap, arg_of_perigee_degree)
            self.assertEqual(tles_manual["satellites"][i]._decay, 0)
            self.assertEqual(tles_manual["satellites"][i]._drag, 0)
            self.assertAlmostEqual(tles_manual["satellites"][i]._e, eccentricity, 5)
            self.assertEqual(
                tles_manual["satellites"][i]._epoch, ephem.Date("2000-01-01 00:00:00")
            )
            self.assertAlmostEqual(
                tles_manual["satellites"][i]._inc, math.radians(inclination_degree), 5
            )
            self.assertAlmostEqual(
                tles_manual["satellites"][i]._M, math.radians(mean_anomaly_degree), 5
            )
            self.assertEqual(tles_manual["satellites"][i]._n, mean_motion_rev_per_day)
            self.assertEqual(
                tles_manual["satellites"][i]._orbit, 0
            )  # Number of orbit revolutions done at epoch
            self.assertAlmostEqual(
                tles_manual["satellites"][i]._raan,
                math.radians(orbit * 360.0 / num_orbits),
                5,
            )

        # Finally remove the temporary files
        os.remove("tles_manual.txt.tmp")
        os.remove("tles_sgp.txt.tmp")

    def test_tles_generation_kuiper_starlink_telesat(self):

        # Kuiper-630
        self.tles_generation("Kuiper-630-K1", 34, 34, True, 51.9, 0.0000001, 0.0, 14.80)

        # Starlink 550
        self.tles_generation("Starlink-550", 72, 22, True, 53, 0.0000001, 0.0, 13.66)

        # Telesat 1015
        self.tles_generation("Telesat-1015", 27, 13, True, 98.98, 0.0000001, 0.0, 13.66)
