# In a new file, e.g., tests/dynamic_state/test_end_to_end_kuiper_triangle.py
# Or add to test_generate_dynamic_state_integration.py

import math
import unittest
import pprint

import ephem
from astropy.time import Time
from astropy import units as u

# Modules and classes to test/use
from src.dynamic_state.generate_dynamic_state import generate_dynamic_state_at
from src.dynamic_state.topology import ConstellationData, GroundStation, Satellite
from src.distance_tools import geodetic2cartesian  # Assuming this is available


class TestEndToEndKuiperTriangle(unittest.TestCase):

    def test_kuiper_triangle_t0(self):
        """
        Integration test for Kuiper subset scenario at t=0.
        Checks basic state generation.
        """
        # --- Inputs ---
        output_dir = None
        epoch = Time("2000-01-01 00:00:00", scale="tdb")  # Match TLE epoch
        time_since_epoch_ns = 0
        dynamic_state_algorithm = "algorithm_free_one_only_over_isls"
        prev_output = None

        # Max lengths (same calculation as old test)
        altitude_m = 630000
        earth_radius = 6378135.0
        satellite_cone_radius_m = altitude_m / math.tan(math.radians(30.0))
        max_gsl_length_m = math.sqrt(math.pow(satellite_cone_radius_m, 2) + math.pow(altitude_m, 2))
        max_isl_length_m = 2 * math.sqrt(
            math.pow(earth_radius + altitude_m, 2) - math.pow(earth_radius + 80000, 2)
        )

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

        # Ground Station Data (Manila=12, Dalian=13, StPete=14)
        GS_MANILA_ID = 12
        GS_DALIAN_ID = 13
        GS_STPETE_ID = 14

        gs_defs = {
            GS_MANILA_ID: {"name": "Manila", "lat": 14.6042, "lon": 120.9822, "elv": 0.0},
            GS_DALIAN_ID: {"name": "Dalian", "lat": 38.913811, "lon": 121.602322, "elv": 0.0},
            GS_STPETE_ID: {"name": "StPete", "lat": 59.929858, "lon": 30.326228, "elv": 0.0},
        }
        ground_stations = []
        for gid, data in gs_defs.items():
            x, y, z = geodetic2cartesian(data["lat"], data["lon"], data["elv"])
            ground_stations.append(
                GroundStation(
                    gid=gid,
                    name=data["name"],
                    latitude_degrees_str=str(data["lat"]),
                    longitude_degrees_str=str(data["lon"]),
                    elevation_m_float=data["elv"],
                    cartesian_x=x,
                    cartesian_y=y,
                    cartesian_z=z,
                )
            )

        # ConstellationData
        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellites),
            epoch="00001.00000000",
            max_gsl_length_m=max_gsl_length_m,
            max_isl_length_m=max_isl_length_m,
            satellites=satellites,
        )

        # ISLs based on old test mapping
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

        # GSL Interface Info (12 Sats + 3 GS)
        list_gsl_interfaces_info = [
            {"id": node_id, "number_of_interfaces": 1, "aggregate_max_bandwidth": 1.0}
            for node_id in list(range(12)) + [GS_MANILA_ID, GS_DALIAN_ID, GS_STPETE_ID]
        ]

        # --- Execute for t=0 ---
        result_state_t0, _ = generate_dynamic_state_at(
            output_dynamic_state_dir=output_dir,
            epoch=epoch,
            time_since_epoch_ns=0,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
            prev_output=None,
            prev_topology=None,
        )

        # --- Assertions for t=0 ---
        self.assertIsNotNone(result_state_t0, "generate_dynamic_state_at returned None at t=0")
        self.assertIn("fstate", result_state_t0)
        self.assertIn("bandwidth", result_state_t0)
        self.assertIsInstance(result_state_t0["fstate"], dict)
        self.assertIsInstance(result_state_t0["bandwidth"], dict)

        # Check bandwidth calculation looks okay
        self.assertEqual(len(result_state_t0["bandwidth"]), 12 + 3)
        self.assertEqual(result_state_t0["bandwidth"].get(0), 1.0)  # Check first sat
        self.assertEqual(result_state_t0["bandwidth"].get(GS_MANILA_ID), 1.0)  # Check first GS

        # Check specific first hop from old trace for t=0
        fstate_t0 = result_state_t0["fstate"]
        hop_tuple_12_13 = fstate_t0.get((GS_MANILA_ID, GS_DALIAN_ID))
        self.assertIsNotNone(
            hop_tuple_12_13, f"fstate missing for ({GS_MANILA_ID}, {GS_DALIAN_ID}) at t=0"
        )
        self.assertEqual(
            hop_tuple_12_13[0], 1, f"First hop mismatch for ({GS_MANILA_ID}, {GS_DALIAN_ID}) at t=0"
        )  # Expected hop Sat 1 (Original 184)

        # Optional: Print fstate for manual inspection or capture
        print("\n" + "=" * 20)
        print("Actual F-State for Kuiper Triangle Test (t=0):")
        pprint.pprint(fstate_t0)
        print("=" * 20 + "\n")

        # Optional: Run for another time step and check path change if desired
        # time_ns_18s = 18 * 10**9
        # result_state_t18 = generate_dynamic_state_at(...)
        # hop_tuple_t18 = result_state_t18["fstate"].get((GS_MANILA_ID, GS_DALIAN_ID))
        # self.assertEqual(hop_tuple_t18[0], 4, ...) # Expected hop Sat 4 (Original 218)
