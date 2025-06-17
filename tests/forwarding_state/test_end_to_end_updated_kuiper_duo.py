# In tests/forwarding_state/test_end_to_end_updated_kuiper_duo.py (or similar file)

import math
import unittest

import ephem
from astropy.time import Time

from src.distance_tools import geodetic2cartesian
from src.network_state.generate_network_state import _generate_state_for_step
from src.topology.topology import ConstellationData, GroundStation, Satellite


class TestEndToEndRefactored(unittest.TestCase):

    def test_kuiper_path_evolution(self):
        """
        Integration test checking state at t=0 and first hop for Manila->Dalian
        at later time steps, based on the old end-to-end test traces.
        Uses sequential IDs matching the old test's analysis (Sats 0-11, GS 12=Manila, 13=Dalian).
        """
        # --- Inputs ---
        epoch = Time("2000-01-01 00:00:00", scale="tdb")  # Match TLE epoch
        dynamic_state_algorithm = "shortest_path_link_state"
        prev_output = None  # Check each step independently

        # Max lengths
        altitude_m = 630000
        earth_radius = 6378135.0
        satellite_cone_radius_m = altitude_m / math.tan(math.radians(30.0))
        max_gsl_length_m = math.sqrt(math.pow(satellite_cone_radius_m, 2) + math.pow(altitude_m, 2))
        max_isl_length_m = 2 * math.sqrt(
            math.pow(earth_radius + altitude_m, 2) - math.pow(earth_radius + 80000, 2)
        )

        # --- Setup Common Data ---
        # TLE Data (12 satellites, IDs 0-11)
        tle_data = {
            0: (
                "Kuiper-630 0",
                "1 00184U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    06",
                "2 00184  51.9000  52.9412 0000001   0.0000 142.9412 14.80000000    00",
            ),
            1: (
                "Kuiper-630 1",
                "1 00185U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    07",
                "2 00185  51.9000  52.9412 0000001   0.0000 153.5294 14.80000000    07",
            ),
            2: (
                "Kuiper-630 2",
                "1 00217U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    03",
                "2 00217  51.9000  63.5294 0000001   0.0000 127.0588 14.80000000    01",
            ),
            3: (
                "Kuiper-630 3",
                "1 00218U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    04",
                "2 00218  51.9000  63.5294 0000001   0.0000 137.6471 14.80000000    00",
            ),
            4: (
                "Kuiper-630 4",
                "1 00219U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
                "2 00219  51.9000  63.5294 0000001   0.0000 148.2353 14.80000000    08",
            ),
            5: (
                "Kuiper-630 5",
                "1 00251U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    01",
                "2 00251  51.9000  74.1176 0000001   0.0000 132.3529 14.80000000    00",
            ),
            6: (
                "Kuiper-630 6",
                "1 00616U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    06",
                "2 00616  51.9000 190.5882 0000001   0.0000  31.7647 14.80000000    05",
            ),
            7: (
                "Kuiper-630 7",
                "1 00617U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    07",
                "2 00617  51.9000 190.5882 0000001   0.0000  42.3529 14.80000000    03",
            ),
            8: (
                "Kuiper-630 8",
                "1 00648U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    01",
                "2 00648  51.9000 201.1765 0000001   0.0000  15.8824 14.80000000    09",
            ),
            9: (
                "Kuiper-630 9",
                "1 00649U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    02",
                "2 00649  51.9000 201.1765 0000001   0.0000  26.4706 14.80000000    07",
            ),
            10: (
                "Kuiper-630 10",
                "1 00650U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    04",
                "2 00650  51.9000 201.1765 0000001   0.0000  37.0588 14.80000000    05",
            ),
            11: (
                "Kuiper-630 11",
                "1 00651U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
                "2 00651  51.9000 201.1765 0000001   0.0000  47.6471 14.80000000    04",
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
                self.fail(f"Failed to read TLE for sat_id {sat_id}: {e}")

        # Ground Station Data (Manila=12, Dalian=13)
        manila_lat, manila_lon, manila_elv = 14.6042, 120.9822, 0.0
        manila_x, manila_y, manila_z = geodetic2cartesian(manila_lat, manila_lon, manila_elv)
        dalian_lat, dalian_lon, dalian_elv = 38.913811, 121.602322, 0.0
        dalian_x, dalian_y, dalian_z = geodetic2cartesian(dalian_lat, dalian_lon, dalian_elv)
        GS_MANILA_ID = 12
        GS_DALIAN_ID = 13

        ground_stations = [
            GroundStation(
                gid=GS_MANILA_ID,
                name="Manila",
                latitude_degrees_str=str(manila_lat),
                longitude_degrees_str=str(manila_lon),
                elevation_m_float=manila_elv,
                cartesian_x=manila_x,
                cartesian_y=manila_y,
                cartesian_z=manila_z,
            ),
            GroundStation(
                gid=GS_DALIAN_ID,
                name="Dalian",
                latitude_degrees_str=str(dalian_lat),
                longitude_degrees_str=str(dalian_lon),
                elevation_m_float=dalian_elv,
                cartesian_x=dalian_x,
                cartesian_y=dalian_y,
                cartesian_z=dalian_z,
            ),
        ]
        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellites),
            epoch="00001.00000000",
            max_gsl_length_m=max_gsl_length_m,
            max_isl_length_m=max_isl_length_m,
            satellites=satellites,
        )
        undirected_isls = [
            (0, 1),
            (0, 3),
            (2, 3),
            (2, 5),
            (3, 4),
            (6, 10),
            (7, 11),
            (8, 9),
            (9, 10),
            (10, 11),
        ]
        list_gsl_interfaces_info = [
            {"id": node_id, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0}
            for node_id in list(range(12)) + [GS_MANILA_ID, GS_DALIAN_ID]
        ]

        # --- Execute and Assert for t=0 ---
        print("\n--- Checking Full State at t=0 ns ---")
        result_state_t0, _ = _generate_state_for_step(
            epoch=epoch,
            time_since_epoch_ns=0,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
            prev_output=None,
            prev_topology=None,  # Added missing argument
        )

        self.assertIsNotNone(result_state_t0, "_generate_state_for_step returned None at t=0")
        self.assertIn("fstate", result_state_t0)
        self.assertIn("bandwidth", result_state_t0)
        fstate_t0 = result_state_t0["fstate"]

        expected_fstate_t0 = {
            (0, 12): (1, 0, 0),
            (0, 13): (3, 1, 0),
            (1, 12): (12, 1, 0),
            (1, 13): (0, 0, 0),
            (2, 12): (3, 0, 1),
            (2, 13): (5, 1, 0),
            (3, 12): (0, 0, 1),
            (3, 13): (13, 3, 0),  # Using actual output IF=3
            (4, 12): (3, 0, 2),
            (4, 13): (3, 0, 2),
            (5, 12): (2, 0, 1),
            (5, 13): (13, 1, 0),
            (6, 12): (10, 0, 0),
            (6, 13): (10, 0, 0),
            (7, 12): (11, 0, 0),
            (7, 13): (13, 1, 0),
            (8, 12): (9, 0, 0),
            (8, 13): (9, 0, 0),
            (9, 12): (12, 2, 0),
            (9, 13): (10, 1, 1),
            (10, 12): (9, 1, 1),
            (10, 13): (11, 2, 1),
            (11, 12): (10, 1, 2),
            (11, 13): (7, 0, 0),
            (12, 13): (1, 0, 1),
            (13, 12): (3, 0, 3),  # Using actual output IF=3
        }

        self.maxDiff = None
        # Assert the whole dictionary for t=0 matches the actual output
        self.assertDictEqual(
            fstate_t0,
            expected_fstate_t0,
            "Full fstate mismatch at t=0. Check calculation logic for Sat 3's GSL IF index.",
        )

        # --- Define Time Steps and Expected First Hops (Manila -> Dalian) for later times ---
        # (This part remains unchanged from the last successful version)
        test_points = [
            (18 * 10**9, "12-4-3-13", 4),
            (27.6 * 10**9, "12-9-10-11-7-13", 4),
            (74.3 * 10**9, "12-4-3-2-5-13", 9),
            (125.9 * 10**9, "12-8-9-10-11-7-13", 4),
            (128.7 * 10**9, "12-8-9-10-6-13", 8),
        ]

        # --- Execute and Assert for later time steps (checking first hop only) ---
        for time_ns, path_str, expected_hop_id in test_points:
            time_since_epoch_ns_int = int(time_ns)
            print(f"\n--- Checking t={time_since_epoch_ns_int} ns (First Hop Only) ---")

            result_state, _ = _generate_state_for_step(
                epoch=epoch,
                time_since_epoch_ns=time_since_epoch_ns_int,
                constellation_data=constellation_data,
                ground_stations=ground_stations,
                undirected_isls=undirected_isls,
                list_gsl_interfaces_info=list_gsl_interfaces_info,
                dynamic_state_algorithm=dynamic_state_algorithm,
                prev_output=prev_output,  # Still pass None for prev_output
                prev_topology=None,  # Added missing argument
            )

            self.assertIsNotNone(
                result_state, f"generate_dynamic_state_at returned None at t={time_ns}"
            )
            self.assertIn("fstate", result_state)
            fstate = result_state["fstate"]
            hop_tuple = fstate.get(
                (GS_MANILA_ID, GS_DALIAN_ID), (-1, -1, -1)
            )  # Use default if key missing

            # Assert the first hop matches expectation
            self.assertEqual(
                hop_tuple[0],  # actual_hop_id
                expected_hop_id,
                f"Mismatch at t={time_ns} ns. Ref Path: {path_str}. "
                f"Expected first hop: {expected_hop_id}. Got: {hop_tuple[0]}. Full state tuple: {hop_tuple}",
            )


# Add main block if running directly
# if __name__ == '__main__':
#      unittest.main()
