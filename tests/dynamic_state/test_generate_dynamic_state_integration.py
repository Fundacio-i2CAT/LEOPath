# tests/dynamic_state/test_generate_dynamic_state_integration.py
# FINAL VERSION with TLE fixes & Non-Sequential Test Update

import math
import unittest
import pprint  # For printing the actual fstate nicely

import ephem
from astropy.time import Time
from astropy import units as astro_units

# Modules and classes to test/use
from src.dynamic_state.generate_dynamic_state import generate_dynamic_state_at
from src.dynamic_state.topology import ConstellationData, GroundStation, Satellite


class TestDynamicStateIntegration(unittest.TestCase):

    def test_equator_scenario_t0(self):
        """
        Integration test using sequential IDs 0..N-1.
        Checks fstate and bandwidth at t=0.
        """
        output_dir = None
        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        time_since_epoch_ns = 0
        dynamic_state_algorithm = "algorithm_free_one_only_over_isls"
        prev_output = None
        max_gsl_length_m = 1089686.4181956202
        max_isl_length_m = 5016591.2330984278

        tle_data = {
            0: (
                "Starlink-550 0",
                "1 01308U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
                "2 01308  53.0000 295.0000 0000001   0.0000 155.4545 15.19000000    04",
            ),
            1: (
                "Starlink-550 1",
                "1 01309U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    06",
                "2 01309  53.0000 295.0000 0000001   0.0000 171.8182 15.19000000    04",
            ),
            2: (
                "Starlink-550 2",
                "1 01310U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    08",
                "2 01310  53.0000 295.0000 0000001   0.0000 188.1818 15.19000000    03",
            ),
            3: (
                "Starlink-550 3",
                "1 01311U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    09",
                "2 01311  53.0000 295.0000 0000001   0.0000 204.5455 15.19000000    04",
            ),
        }
        satellites = []
        for sat_id, tle_lines in tle_data.items():
            try:
                ephem_obj = ephem.readtle(tle_lines[0], tle_lines[1], tle_lines[2])
                satellites.append(
                    Satellite(id=sat_id, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
                )
            except ValueError as e:
                self.fail(f"Failed to read TLE for sat_id {sat_id} in equator_scenario: {e}")

        gs_data = [
            {
                "gid": 4,
                "name": "Luanda",
                "lat": "-8.836820",
                "lon": "13.234320",
                "elv": 0.0,
                "x": 6135530.18,
                "y": 1442953.50,
                "z": -973332.34,
            },
            {
                "gid": 5,
                "name": "Lagos",
                "lat": "6.453060",
                "lon": "3.395830",
                "elv": 0.0,
                "x": 6326864.17,
                "y": 375422.89,
                "z": 712064.78,
            },
            {
                "gid": 6,
                "name": "Kinshasa",
                "lat": "-4.327580",
                "lon": "15.313570",
                "elv": 0.0,
                "x": 6134256.67,
                "y": 1679704.40,
                "z": -478073.16,
            },
            {
                "gid": 7,
                "name": "Ar-Riyadh-(Riyadh)",
                "lat": "24.690466",
                "lon": "46.709566",
                "elv": 0.0,
                "x": 3975957.34,
                "y": 4220595.03,
                "z": 2647959.98,
            },
        ]
        # Use correct keyword args for GroundStation constructor
        ground_stations = [
            GroundStation(
                gid=d["gid"],
                name=d["name"],
                latitude_degrees_str=d["lat"],
                longitude_degrees_str=d["lon"],
                elevation_m_float=d["elv"],
                cartesian_x=d["x"],
                cartesian_y=d["y"],
                cartesian_z=d["z"],
            )
            for d in gs_data
        ]

        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellites),
            epoch="00001.00000000",  # Match TLE
            max_gsl_length_m=max_gsl_length_m,
            max_isl_length_m=max_isl_length_m,
            satellites=satellites,
        )
        undirected_isls = [(0, 1), (1, 2), (2, 3)]
        list_gsl_interfaces_info = [
            {"id": i, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0} for i in range(8)
        ]

        # --- Execute ---
        result_state = generate_dynamic_state_at(
            output_dynamic_state_dir=output_dir,
            epoch=epoch,
            time_since_epoch_ns=time_since_epoch_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
            prev_output=prev_output,
        )

        # --- Assertions ---
        self.assertIsNotNone(result_state, "generate_dynamic_state_at returned None")
        self.assertIn("fstate", result_state)
        self.assertIn("bandwidth", result_state)
        expected_bandwidth = {i: 1.0 for i in range(8)}
        self.assertDictEqual(result_state["bandwidth"], expected_bandwidth)

        expected_fstate = {  # Expected state based on previous runs/manual calculation
            (0, 4): (1, 0, 0),
            (0, 5): (1, 0, 0),
            (0, 6): (1, 0, 0),
            (0, 7): (-1, -1, -1),
            (1, 4): (2, 1, 0),
            (1, 5): (5, 2, 0),
            (1, 6): (2, 1, 0),
            (1, 7): (-1, -1, -1),
            (2, 4): (4, 2, 0),
            (2, 5): (1, 0, 1),
            (2, 6): (6, 2, 0),
            (2, 7): (-1, -1, -1),
            (3, 4): (2, 0, 1),
            (3, 5): (2, 0, 1),
            (3, 6): (2, 0, 1),
            (3, 7): (-1, -1, -1),
            (4, 5): (2, 0, 2),
            (4, 6): (2, 0, 2),
            (4, 7): (-1, -1, -1),
            (5, 4): (1, 0, 2),
            (5, 6): (1, 0, 2),
            (5, 7): (-1, -1, -1),
            (6, 4): (2, 0, 2),
            (6, 5): (2, 0, 2),
            (6, 7): (-1, -1, -1),
            (7, 4): (-1, -1, -1),
            (7, 5): (-1, -1, -1),
            (7, 6): (-1, -1, -1),
        }
        self.maxDiff = None
        self.assertDictEqual(result_state["fstate"], expected_fstate)

    def test_non_sequential_ids(self):
        """
        Integration test with non-sequential IDs for satellites and ground stations.
        Ensures the system handles non-sequential IDs correctly.
        """
        # --- Inputs ---
        output_dir = None  # Not writing files
        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        time_since_epoch_ns = 0  # Test at t=0
        dynamic_state_algorithm = "algorithm_free_one_only_over_isls"
        prev_output = None

        # Max lengths
        max_gsl_length_m = 1089686.4181956202
        max_isl_length_m = 5016591.2330984278

        # TLE Data (non-sequential satellite IDs)
        tle_data = {
            10: (
                "Starlink-550 10",
                "1 01308U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
                "2 01308  53.0000 295.0000 0000001   0.0000 155.4545 15.19000000    04",
            ),
            20: (
                "Starlink-550 20",
                "1 01309U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    06",
                "2 01309  53.0000 295.0000 0000001   0.0000 171.8182 15.19000000    04",
            ),
            30: (
                "Starlink-550 30",
                "1 01310U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    08",
                "2 01310  53.0000 295.0000 0000001   0.0000 188.1818 15.19000000    03",
            ),
            40: (
                "Starlink-550 40",
                "1 01311U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    09",
                "2 01311  53.0000 295.0000 0000001   0.0000 204.5455 15.19000000    04",
            ),
        }
        satellites = []
        for sat_id, tle_lines in tle_data.items():
            ephem_obj = ephem.readtle(tle_lines[0], tle_lines[1], tle_lines[2])
            satellites.append(
                Satellite(id=sat_id, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
            )

        # Ground Station Data (non-sequential IDs)
        gs_data = [
            {
                "gid": 100,
                "name": "Luanda",
                "lat": "-8.836820",
                "lon": "13.234320",
                "elv": 0.0,
                "x": 6135530.18,
                "y": 1442953.50,
                "z": -973332.34,
            },
            {
                "gid": 200,
                "name": "Lagos",
                "lat": "6.453060",
                "lon": "3.395830",
                "elv": 0.0,
                "x": 6326864.17,
                "y": 375422.89,
                "z": 712064.78,
            },
            {
                "gid": 300,
                "name": "Kinshasa",
                "lat": "-4.327580",
                "lon": "15.313570",
                "elv": 0.0,
                "x": 6134256.67,
                "y": 1679704.40,
                "z": -478073.16,
            },
            {
                "gid": 400,
                "name": "Ar-Riyadh-(Riyadh)",
                "lat": "24.690466",
                "lon": "46.709566",
                "elv": 0.0,
                "x": 3975957.34,
                "y": 4220595.03,
                "z": 2647959.98,
            },
        ]
        ground_stations = [
            GroundStation(
                gid=d["gid"],
                name=d["name"],
                latitude_degrees_str=d["lat"],
                longitude_degrees_str=d["lon"],
                elevation_m_float=d["elv"],
                cartesian_x=d["x"],
                cartesian_y=d["y"],
                cartesian_z=d["z"],
            )
            for d in gs_data
        ]

        # ConstellationData
        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellites),
            epoch="00001.00000000",
            max_gsl_length_m=max_gsl_length_m,
            max_isl_length_m=max_isl_length_m,
            satellites=satellites,
        )

        # Undirected ISLs (using non-sequential sat IDs)
        undirected_isls = [(10, 20), (20, 30), (30, 40)]

        # GSL Interface Info (Nodes 10-40 are Sats, 100-400 are GSs)
        list_gsl_interfaces_info = [
            {"id": 10, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 20, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 30, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 40, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 100, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 200, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 300, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 400, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0},
        ]

        # --- Execute ---
        result_state = generate_dynamic_state_at(
            output_dynamic_state_dir=output_dir,
            epoch=epoch,
            time_since_epoch_ns=time_since_epoch_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
            prev_output=prev_output,
        )

        # --- Assertions ---
        self.assertIsNotNone(result_state, "generate_dynamic_state_at returned None")
        self.assertIn("fstate", result_state)
        self.assertIn("bandwidth", result_state)

        # Assert Bandwidth state
        expected_bandwidth = {
            10: 1.0,
            20: 1.0,
            30: 1.0,
            40: 1.0,
            100: 1.0,
            200: 1.0,
            300: 1.0,
            400: 1.0,
        }
        self.assertDictEqual(result_state["bandwidth"], expected_bandwidth)

        expected_fstate = {  # Expected state based on previous runs/manual calculation
            (10, 100): (20, 0, 0),
            (10, 200): (20, 0, 0),
            (10, 300): (20, 0, 0),
            (10, 400): (-1, -1, -1),
            (20, 100): (30, 1, 0),
            (20, 200): (30, 2, 0),
            (20, 300): (30, 1, 0),
            (20, 400): (-1, -1, -1),
            (30, 100): (40, 2, 0),
            (30, 200): (40, 0, 1),
            (30, 300): (40, 2, 0),
            (30, 400): (-1, -1, -1),
            (40, 100): (30, 0, 1),
            (40, 200): (30, 0, 1),
            (40, 300): (30, 0, 1),
            (40, 400): (-1, -1, -1),
            (100, 200): (30, 0, 2),
            (100, 300): (30, 0, 2),
            (100, 400): (-1, -1, -1),
            (200, 100): (20, 0, 2),
            (200, 300): (20, 0, 2),
            (200, 400): (-1, -1, -1),
            (300, 100): (30, 0, 2),
            (300, 200): (30, 0, 2),
            (300, 400): (-1, -1, -1),
            (400, 100): (-1, -1, -1),
            (400, 200): (-1, -1, -1),
            (400, 300): (-1, -1, -1),
        }
        self.assertDictEqual(result_state["fstate"], expected_fstate)
