# tests/dynamic_state/test_generate_dynamic_state_integration.py
# FINAL VERSION with TLE fixes & Non-Sequential Test Update

import math
import pprint  # For printing the actual fstate nicely
import unittest

import ephem
from astropy import units as astro_units
from astropy.time import Time

# Modules and classes to test/use
from src.dynamic_state.generate_dynamic_state import (
    generate_dynamic_state,
    generate_dynamic_state_at,
)
from src.topology.topology import ConstellationData, GroundStation, Satellite


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
        result_state_dict, result_topology = generate_dynamic_state_at(
            output_dynamic_state_dir=output_dir,
            epoch=epoch,
            time_since_epoch_ns=time_since_epoch_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
            prev_output=prev_output,
            prev_topology=None,
        )

        # --- Assertions ---
        self.assertIsNotNone(result_state_dict, "generate_dynamic_state_at returned None")
        self.assertIsNotNone(result_state_dict, "result_state_dict is None")
        self.assertIn("fstate", result_state_dict)
        self.assertIn("bandwidth", result_state_dict)
        expected_bandwidth = {i: 1.0 for i in range(8)}
        self.assertDictEqual(result_state_dict["bandwidth"], expected_bandwidth)

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
        self.assertDictEqual(result_state_dict["fstate"], expected_fstate)

    # In tests/dynamic_state/test_generate_dynamic_state_integration.py

    def test_non_sequential_ids(self):
        """
        Integration test using non-sequential IDs with valid TLE data.
        Checks fstate and bandwidth calculation for non-sequential IDs.
        Uses TLE data known to work from test_equator_scenario_t0.
        """
        # --- Inputs ---
        output_dir = None
        epoch = Time("2000-01-01 00:00:00", scale="tdb")  # Match TLE epoch
        time_since_epoch_ns = 0
        dynamic_state_algorithm = "algorithm_free_one_only_over_isls"
        prev_output = None
        # Use max lengths from other test for consistency with TLEs
        max_gsl_length_m = 1089686.4181956202
        max_isl_length_m = 5016591.2330984278

        # Define non-sequential IDs to use
        sat_ids = [10, 20, 30, 40]
        gs_ids = [100, 200, 300, 400]
        all_node_ids = sat_ids + gs_ids

        # Use known VALID TLE data from the first test
        tle_data_valid = {
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

        # Create Satellite objects assigning the NON-SEQUENTIAL IDs
        satellites = []
        for i, new_id in enumerate(sat_ids):
            tle_lines = tle_data_valid[i]  # Get corresponding valid TLE
            try:
                ephem_obj = ephem.readtle(tle_lines[0], tle_lines[1], tle_lines[2])
                satellites.append(
                    Satellite(id=new_id, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
                )
            except ValueError as e:
                self.fail(f"Failed to read known-good TLE for original index {i}: {e}")

        # Use same ground station location data, assign NON-SEQUENTIAL IDs
        gs_data = [
            {
                "lat": "-8.836820",
                "lon": "13.234320",
                "elv": 0.0,
                "x": 6135530.18,
                "y": 1442953.50,
                "z": -973332.34,
                "name": "Luanda",
            },
            {
                "lat": "6.453060",
                "lon": "3.395830",
                "elv": 0.0,
                "x": 6326864.17,
                "y": 375422.89,
                "z": 712064.78,
                "name": "Lagos",
            },
            {
                "lat": "-4.327580",
                "lon": "15.313570",
                "elv": 0.0,
                "x": 6134256.67,
                "y": 1679704.40,
                "z": -478073.16,
                "name": "Kinshasa",
            },
            {
                "lat": "24.690466",
                "lon": "46.709566",
                "elv": 0.0,
                "x": 3975957.34,
                "y": 4220595.03,
                "z": 2647959.98,
                "name": "Ar-Riyadh-(Riyadh)",
            },
        ]
        ground_stations = []
        for i, new_id in enumerate(gs_ids):
            d = gs_data[i]
            ground_stations.append(
                GroundStation(
                    gid=new_id,
                    name=d["name"],
                    latitude_degrees_str=d["lat"],
                    longitude_degrees_str=d["lon"],
                    elevation_m_float=d["elv"],
                    cartesian_x=d["x"],
                    cartesian_y=d["y"],
                    cartesian_z=d["z"],
                )
            )

        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellites),
            epoch="00001.00000000",
            max_gsl_length_m=max_gsl_length_m,
            max_isl_length_m=max_isl_length_m,
            satellites=satellites,
        )

        # Define ISLs using NON-SEQUENTIAL IDs corresponding to original 0-1, 1-2, 2-3
        undirected_isls = [(10, 20), (20, 30), (30, 40)]

        # Define GSL Interface Info using NON-SEQUENTIAL IDs
        list_gsl_interfaces_info = [
            {"id": node_id, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0}
            for node_id in all_node_ids
        ]

        # --- Execute ---
        result_state, _ = generate_dynamic_state_at(
            output_dynamic_state_dir=output_dir,
            epoch=epoch,
            time_since_epoch_ns=time_since_epoch_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
            prev_output=prev_output,
            prev_topology=None,
        )

        # --- Assertions ---
        self.assertIsNotNone(result_state, "generate_dynamic_state_at returned None")
        self.assertIn("fstate", result_state)
        self.assertIn("bandwidth", result_state)

        # Assert Bandwidth state (uses non-sequential IDs)
        expected_bandwidth = {node_id: 1.0 for node_id in all_node_ids}
        self.assertDictEqual(result_state["bandwidth"], expected_bandwidth)

        # Assert Forwarding state (using the correctly translated dictionary)
        expected_fstate = {
            # Sat 10 -> GS
            (10, 100): (20, 0, 0),
            (10, 200): (20, 0, 0),
            (10, 300): (20, 0, 0),
            (10, 400): (-1, -1, -1),
            # Sat 20 -> GS
            (20, 100): (30, 1, 0),
            (20, 200): (200, 2, 0),
            (20, 300): (30, 1, 0),
            (20, 400): (-1, -1, -1),
            # Sat 30 -> GS
            (30, 100): (100, 2, 0),
            (30, 200): (20, 0, 1),
            (30, 300): (300, 2, 0),
            (30, 400): (-1, -1, -1),
            # Sat 40 -> GS
            (40, 100): (30, 0, 1),
            (40, 200): (30, 0, 1),
            (40, 300): (30, 0, 1),
            (40, 400): (-1, -1, -1),
            # GS 100 -> GS
            (100, 200): (30, 0, 2),
            (100, 300): (30, 0, 2),
            (100, 400): (-1, -1, -1),
            # GS 200 -> GS
            (200, 100): (20, 0, 2),
            (200, 300): (20, 0, 2),
            (200, 400): (-1, -1, -1),
            # GS 300 -> GS
            (300, 100): (30, 0, 2),
            (300, 200): (30, 0, 2),
            (300, 400): (-1, -1, -1),
            # GS 400 -> GS
            (400, 100): (-1, -1, -1),
            (400, 200): (-1, -1, -1),
            (400, 300): (-1, -1, -1),
        }

        self.maxDiff = None
        self.assertDictEqual(
            result_state["fstate"], expected_fstate, "F-state mismatch for non-sequential IDs."
        )

    def test_full_loop_short_run(self):
        """
        Integration test for the main generate_dynamic_state loop.
        Runs for 3 steps (t=0, 1s, 2s) and checks the exact final state.
        Uses non-sequential IDs with equator TLE data.
        """
        # --- Inputs ---
        output_dir = None
        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        dynamic_state_algorithm = "algorithm_free_one_only_over_isls"

        # Simulation Time Parameters (3 steps)
        time_step_s = 1
        duration_s = 3
        offset_s = 0
        time_step_ns = int(time_step_s * 1e9)
        simulation_end_time_ns = int(duration_s * 1e9)  # Includes t=0, t=1s, t=2s
        offset_ns = int(offset_s * 1e9)

        # Max lengths
        max_gsl_length_m = 1089686.4181956202
        max_isl_length_m = 5016591.2330984278

        # Use non-sequential IDs
        SAT_A_ID = 10
        SAT_B_ID = 20
        SAT_C_ID = 30
        SAT_D_ID = 40
        GS_X_ID = 100
        GS_Y_ID = 200
        GS_Z_ID = 300
        GS_W_ID = 400
        sat_ids = [SAT_A_ID, SAT_B_ID, SAT_C_ID, SAT_D_ID]
        gs_ids = [GS_X_ID, GS_Y_ID, GS_Z_ID, GS_W_ID]
        all_node_ids = sat_ids + gs_ids

        # Use valid TLEs from equator test
        tle_data_orig = {  # Keyed by original sequential ID (0-3)
            0: (
                "Starlink-A",
                "1 01308U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
                "2 01308  53.0000 295.0000 0000001   0.0000 155.4545 15.19000000    04",
            ),
            1: (
                "Starlink-B",
                "1 01309U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    06",
                "2 01309  53.0000 295.0000 0000001   0.0000 171.8182 15.19000000    04",
            ),
            2: (
                "Starlink-C",
                "1 01310U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    08",
                "2 01310  53.0000 295.0000 0000001   0.0000 188.1818 15.19000000    03",
            ),
            3: (
                "Starlink-D",
                "1 01311U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    09",
                "2 01311  53.0000 295.0000 0000001   0.0000 204.5455 15.19000000    04",
            ),
        }
        satellites = []
        for i, new_id in enumerate(sat_ids):
            tle_lines = tle_data_orig[i]
            ephem_obj = ephem.readtle(*tle_lines)
            satellites.append(
                Satellite(id=new_id, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
            )

        # Use same ground station location data, assign non-sequential IDs
        gs_data_orig = [
            {
                "lat": "-8.836820",
                "lon": "13.234320",
                "elv": 0.0,
                "x": 6135530.18,
                "y": 1442953.50,
                "z": -973332.34,
                "name": "Luanda",
            },
            {
                "lat": "6.453060",
                "lon": "3.395830",
                "elv": 0.0,
                "x": 6326864.17,
                "y": 375422.89,
                "z": 712064.78,
                "name": "Lagos",
            },
            {
                "lat": "-4.327580",
                "lon": "15.313570",
                "elv": 0.0,
                "x": 6134256.67,
                "y": 1679704.40,
                "z": -478073.16,
                "name": "Kinshasa",
            },
            {
                "lat": "24.690466",
                "lon": "46.709566",
                "elv": 0.0,
                "x": 3975957.34,
                "y": 4220595.03,
                "z": 2647959.98,
                "name": "Ar-Riyadh",
            },
        ]
        ground_stations = []
        for i, new_id in enumerate(gs_ids):
            d = gs_data_orig[i]
            ground_stations.append(
                GroundStation(
                    gid=new_id,
                    name=d["name"],
                    latitude_degrees_str=d["lat"],
                    longitude_degrees_str=d["lon"],
                    elevation_m_float=d["elv"],
                    cartesian_x=d["x"],
                    cartesian_y=d["y"],
                    cartesian_z=d["z"],
                )
            )

        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellites),
            epoch="00001.00000000",
            max_gsl_length_m=max_gsl_length_m,
            max_isl_length_m=max_isl_length_m,
            satellites=satellites,
        )
        undirected_isls = [(10, 20), (20, 30), (30, 40)]
        list_gsl_interfaces_info = [
            {"id": node_id, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0}
            for node_id in all_node_ids
        ]

        # --- Execute the Full Loop ---
        final_state = generate_dynamic_state(
            output_dynamic_state_dir=output_dir,
            epoch=epoch,
            simulation_end_time_ns=simulation_end_time_ns,
            time_step_ns=time_step_ns,
            offset_ns=offset_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
        )

        # --- Assertions ---
        self.assertIsNotNone(final_state, "generate_dynamic_state loop returned None")
        self.assertIsInstance(final_state, list, "Final state object should be a list")
        self.assertIn("fstate", final_state[0])
        self.assertIn("bandwidth", final_state[0])

        # Check bandwidth state
        expected_bandwidth = {node_id: 1.0 for node_id in all_node_ids}
        self.assertDictEqual(
            final_state[0]["bandwidth"], expected_bandwidth, "Final bandwidth state mismatch"
        )

        # Check fstate is not empty
        self.assertTrue(final_state[0]["fstate"], "Final fstate dictionary is empty")

        expected_final_fstate = {
            (10, 100): (20, 0, 0),
            (10, 200): (20, 0, 0),
            (10, 300): (20, 0, 0),
            (10, 400): (-1, -1, -1),
            (20, 100): (30, 1, 0),
            (20, 200): (200, 2, 0),
            (20, 300): (30, 1, 0),
            (20, 400): (-1, -1, -1),
            (30, 100): (100, 2, 0),
            (30, 200): (20, 0, 1),
            (30, 300): (300, 2, 0),
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

        self.maxDiff = None  # Show full diff on failure
        self.assertDictEqual(
            final_state[0]["fstate"], expected_final_fstate, "Final fstate mismatch after loop"
        )
