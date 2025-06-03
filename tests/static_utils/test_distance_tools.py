import math
import unittest

import ephem
from astropy import units as u
from astropy.time import Time

from src.distance_tools import (
    create_basic_ground_station_for_satellite_shadow,
    distance_m_between_satellites,
    distance_m_ground_station_to_satellite,
    geodesic_distance_m_between_ground_stations,
    geodetic2cartesian,
    straight_distance_m_between_ground_stations,
)
from src.topology.topology import GroundStation, Satellite


class TestDistanceTools(unittest.TestCase):

    def test_distance_between_satellites(self):
        ephem_sat_0 = ephem.readtle(
            "Kuiper-630 0",
            "1 00001U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    04",
            "2 00001  51.9000   0.0000 0000001   0.0000   0.0000 14.80000000    02",
        )
        ephem_sat_1 = ephem.readtle(
            "Kuiper-630 1",
            "1 00002U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
            "2 00002  51.9000   0.0000 0000001   0.0000  10.5882 14.80000000    07",
        )
        ephem_sat_17 = ephem.readtle(
            "Kuiper-630 17",
            "1 00018U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    02",
            "2 00018  51.9000   0.0000 0000001   0.0000 180.0000 14.80000000    09",
        )
        ephem_sat_18 = ephem.readtle(
            "Kuiper-630 18",
            "1 00019U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    03",
            "2 00019  51.9000   0.0000 0000001   0.0000 190.5882 14.80000000    04",
        )
        ephem_sat_19 = ephem.readtle(
            "Kuiper-630 19",
            "1 00020U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
            "2 00020  51.9000   0.0000 0000001   0.0000 201.1765 14.80000000    05",
        )
        sat_obj_0 = Satellite(id=0, ephem_obj_manual=ephem_sat_0, ephem_obj_direct=ephem_sat_0)
        sat_obj_1 = Satellite(id=1, ephem_obj_manual=ephem_sat_1, ephem_obj_direct=ephem_sat_1)
        sat_obj_17 = Satellite(id=17, ephem_obj_manual=ephem_sat_17, ephem_obj_direct=ephem_sat_17)
        sat_obj_18 = Satellite(id=18, ephem_obj_manual=ephem_sat_18, ephem_obj_direct=ephem_sat_18)
        sat_obj_19 = Satellite(id=19, ephem_obj_manual=ephem_sat_19, ephem_obj_direct=ephem_sat_19)

        for extra_time_ns in [
            0,
            1,
            1000,
            1000000,
            1000000000,
            60000000000,
            10 * 60000000000,
            20 * 60000000000,
            30 * 60000000000,
            40 * 60000000000,
            50 * 60000000000,
            60 * 60000000000,
            70 * 60000000000,
            80 * 60000000000,
            90 * 60000000000,
            100 * 60000000000,
        ]:
            epoch = Time("2000-01-01 00:00:00", scale="tdb")
            time_obj = epoch + extra_time_ns * u.ns
            epoch_str_for_ephem = str(epoch.strftime("%Y/%m/%d"))
            # Add fractional seconds, remove trailing zeros if needed (ephem might be picky)
            time_str_for_ephem = time_obj.strftime("%Y/%m/%d %H:%M:%S.%f")
            if time_str_for_ephem.endswith(".000000"):
                time_str_for_ephem = time_str_for_ephem[:-7]  # Remove if exactly zero
            elif "." in time_str_for_ephem:
                time_str_for_ephem = time_str_for_ephem.rstrip("0")  # Remove trailing zeros
            self.assertAlmostEqual(
                distance_m_between_satellites(
                    sat_obj_0, sat_obj_0, epoch_str_for_ephem, time_str_for_ephem
                ),
                0,
                delta=1e-3,  # Use delta for float comparison
            )
            # ... (rest of self checks) ...

            dist_0_1 = distance_m_between_satellites(
                sat_obj_0, sat_obj_1, epoch_str_for_ephem, time_str_for_ephem
            )
            dist_1_0 = distance_m_between_satellites(
                sat_obj_1, sat_obj_0, epoch_str_for_ephem, time_str_for_ephem
            )
            self.assertAlmostEqual(dist_0_1, dist_1_0, delta=1e-3)
            # ... (rest of symmetry checks) ...

            dist_0_18 = distance_m_between_satellites(
                sat_obj_0, sat_obj_18, epoch_str_for_ephem, time_str_for_ephem
            )
            self.assertGreater(dist_0_18, dist_0_1)

            dist_18_19 = distance_m_between_satellites(
                sat_obj_18, sat_obj_19, epoch_str_for_ephem, time_str_for_ephem
            )
            dist_17_18 = distance_m_between_satellites(
                sat_obj_17, sat_obj_18, epoch_str_for_ephem, time_str_for_ephem
            )
            dist_17_19 = distance_m_between_satellites(
                sat_obj_17, sat_obj_19, epoch_str_for_ephem, time_str_for_ephem
            )
            epsilon = 1e-3  # Tolerance for float comparison
            self.assertGreaterEqual(  # Use GreaterEqual for robustness with floats
                dist_17_18 + dist_18_19,
                dist_17_19 - epsilon,  # Check A+B >= C - epsilon
                f"Triangle inequality failed: {dist_17_18} + {dist_18_19} <= {dist_17_19}",
            )

            # Polygon side calculation check
            num_sats_per_plane = 34
            polygon_side_m = 2 * (
                7008135.0 * math.sin(math.radians(360.0 / num_sats_per_plane) / 2.0)
            )
            lower_bound = 0.85 * polygon_side_m
            upper_bound = 1.15 * polygon_side_m

            dist_17_18 = distance_m_between_satellites(
                sat_obj_17, sat_obj_18, epoch_str_for_ephem, time_str_for_ephem
            )
            self.assertTrue(
                lower_bound <= dist_17_18 <= upper_bound,
                f"Dist 17-18 ({dist_17_18:.2f}) vs expected polygon side ({polygon_side_m:.2f}) out of bounds",
            )
            # ... (rest of polygon checks) ...

    def test_distance_between_ground_stations(self):
        gs_content = (
            "0,Amsterdam,52.379189,4.899431,0\n"
            "1,Paris,48.864716,2.349014,0\n"
            "2,Rio de Janeiro,-22.970722,-43.182365,0\n"
            "3,Manila,14.599512,120.984222,0\n"
            "4,Perth,-31.953512,115.857048,0\n"
            "5,Antarctica Base,-72.927148,33.450844,0\n"
            "6,New York,40.730610,-73.935242,0\n"
            "7,Greenland Base,79.741382,-53.143087,0"
        )

        ground_stations = []
        for line in gs_content.strip().splitlines():
            parts = line.strip().split(",")
            gid = int(parts[0])
            name = parts[1]
            lat_str = parts[2]
            lon_str = parts[3]
            elev = float(parts[4])
            cart_x, cart_y, cart_z = geodetic2cartesian(float(lat_str), float(lon_str), elev)
            gs = GroundStation(
                gid=gid,
                name=name,
                latitude_degrees_str=lat_str,
                longitude_degrees_str=lon_str,
                elevation_m_float=elev,
                cartesian_x=cart_x,
                cartesian_y=cart_y,
                cartesian_z=cart_z,
            )
            ground_stations.append(gs)

        self.assertTrue(ground_stations, "No ground stations created")
        self.assertTrue(
            all(isinstance(gs, GroundStation) for gs in ground_stations),
            "Not all objects are GroundStation instances",
        )

        # Distance to itself is always 0
        for i in range(len(ground_stations)):
            self.assertEqual(
                geodesic_distance_m_between_ground_stations(ground_stations[i], ground_stations[i]),
                0,
            )
            self.assertEqual(
                straight_distance_m_between_ground_stations(ground_stations[i], ground_stations[i]),
                0,
            )

        # Direction does not matter
        for i in range(len(ground_stations)):
            for j in range(i + 1, len(ground_stations)):
                dist_geo_ij = geodesic_distance_m_between_ground_stations(
                    ground_stations[i], ground_stations[j]
                )
                dist_geo_ji = geodesic_distance_m_between_ground_stations(
                    ground_stations[j], ground_stations[i]
                )
                self.assertAlmostEqual(dist_geo_ij, dist_geo_ji, delta=1e-3)

                dist_str_ij = straight_distance_m_between_ground_stations(
                    ground_stations[i], ground_stations[j]
                )
                dist_str_ji = straight_distance_m_between_ground_stations(
                    ground_stations[j], ground_stations[i]
                )
                self.assertAlmostEqual(dist_str_ij, dist_str_ji, delta=1e-3)

                # Geodesic >= Straight
                self.assertGreaterEqual(dist_geo_ij, dist_str_ij)

        # Check specific distances
        self.assertAlmostEqual(
            geodesic_distance_m_between_ground_stations(ground_stations[0], ground_stations[1]),
            430000,
            delta=1000.0,
        )
        self.assertAlmostEqual(
            geodesic_distance_m_between_ground_stations(ground_stations[0], ground_stations[6]),
            5861000,
            delta=5000.0,
        )
        self.assertAlmostEqual(
            geodesic_distance_m_between_ground_stations(ground_stations[6], ground_stations[5]),
            14861000,
            delta=20000.0,
        )

    def test_distance_ground_station_to_satellite(self):
        # ASSUMPTION: distance_m_ground_station_to_satellite now accepts (GS_Obj, Sat_Obj, epoch_str, date_str)
        # ASSUMPTION: create_basic_... still returns dict, GS dist funcs accept GS_Obj

        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        time_obj = epoch + 100 * 1_000_000_000 * u.ns  # 100 seconds

        # Use compatible time formats for ephem functions
        epoch_str_for_ephem = str(epoch.strftime("%Y/%m/%d"))
        time_str_for_ephem = time_obj.strftime("%Y/%m/%d %H:%M:%S.%f")
        if time_str_for_ephem.endswith(".000000"):
            time_str_for_ephem = time_str_for_ephem[:-7]
        elif "." in time_str_for_ephem:
            time_str_for_ephem = time_str_for_ephem.rstrip("0")

        # Create ephem satellite objects
        ephem_sat_18 = ephem.readtle(
            "Telesat-1015 18",
            "1 00019U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    03",
            "2 00019  98.9800  13.3333 0000001   0.0000 152.3077 13.66000000    04",
        )
        ephem_sat_19 = ephem.readtle(
            "Telesat-1015 19",
            "1 00020U 00000ABC 00001.00000000  .00000000  00000-0  00000+0 0    05",
            "2 00020  98.9800  13.3333 0000001   0.0000 180.0000 13.66000000    00",
        )

        # Wrap in Satellite objects
        sat_obj_18 = Satellite(id=18, ephem_obj_manual=ephem_sat_18, ephem_obj_direct=ephem_sat_18)
        sat_obj_19 = Satellite(id=19, ephem_obj_manual=ephem_sat_19, ephem_obj_direct=ephem_sat_19)

        # Create shadow ground station dicts using the original ephem objects
        shadow_dict_18 = create_basic_ground_station_for_satellite_shadow(
            ephem_sat_18, epoch_str_for_ephem, time_str_for_ephem
        )
        shadow_dict_19 = create_basic_ground_station_for_satellite_shadow(
            ephem_sat_19, epoch_str_for_ephem, time_str_for_ephem
        )

        # Convert shadow dicts to GroundStation objects
        # Requires geodetic2cartesian to be available and correct
        try:
            cart_18 = geodetic2cartesian(
                float(shadow_dict_18["latitude_degrees_str"]),
                float(shadow_dict_18["longitude_degrees_str"]),
                shadow_dict_18["elevation_m_float"],
            )
            cart_19 = geodetic2cartesian(
                float(shadow_dict_19["latitude_degrees_str"]),
                float(shadow_dict_19["longitude_degrees_str"]),
                shadow_dict_19["elevation_m_float"],
            )
        except Exception as e:
            self.fail(f"geodetic2cartesian failed during test setup: {e}")

        shadow_gs_18 = GroundStation(
            gid=shadow_dict_18["gid"],
            name=shadow_dict_18["name"],
            latitude_degrees_str=shadow_dict_18["latitude_degrees_str"],
            longitude_degrees_str=shadow_dict_18["longitude_degrees_str"],
            elevation_m_float=shadow_dict_18["elevation_m_float"],
            cartesian_x=cart_18[0],
            cartesian_y=cart_18[1],
            cartesian_z=cart_18[2],
        )
        shadow_gs_19 = GroundStation(
            gid=shadow_dict_19["gid"],
            name=shadow_dict_19["name"],
            latitude_degrees_str=shadow_dict_19["latitude_degrees_str"],
            longitude_degrees_str=shadow_dict_19["longitude_degrees_str"],
            elevation_m_float=shadow_dict_19["elevation_m_float"],
            cartesian_x=cart_19[0],
            cartesian_y=cart_19[1],
            cartesian_z=cart_19[2],
        )

        # --- Use GroundStation and Satellite objects in calls ---
        dist_shadow_18_to_sat_18 = distance_m_ground_station_to_satellite(
            shadow_gs_18, sat_obj_18, epoch_str_for_ephem, time_str_for_ephem
        )
        self.assertAlmostEqual(dist_shadow_18_to_sat_18, 1015000, delta=5000)

        dist_shadow_19_to_sat_19 = distance_m_ground_station_to_satellite(
            shadow_gs_19, sat_obj_19, epoch_str_for_ephem, time_str_for_ephem
        )
        self.assertAlmostEqual(dist_shadow_19_to_sat_19, 1015000, delta=5000)

        # Assuming GS distance functions take GS_Obj and use dot notation internally
        shadow_distance_m = geodesic_distance_m_between_ground_stations(shadow_gs_18, shadow_gs_19)
        self.assertAlmostEqual(shadow_distance_m, 3080640, delta=5000)

        dist_shadow_18_to_sat_19 = distance_m_ground_station_to_satellite(
            shadow_gs_18, sat_obj_19, epoch_str_for_ephem, time_str_for_ephem
        )
        # Check Pythagoras relationship (approximate for sphere)
        pythag_dist_sq = shadow_distance_m**2 + dist_shadow_19_to_sat_19**2
        self.assertAlmostEqual(
            math.sqrt(pythag_dist_sq),
            dist_shadow_18_to_sat_19,
            delta=0.1 * math.sqrt(pythag_dist_sq),  # Allow 10% deviation
        )

        # Check straight line distance relationship
        straight_shadow_distance_m = straight_distance_m_between_ground_stations(
            shadow_gs_18, shadow_gs_19
        )
        pythag_straight_dist_sq = straight_shadow_distance_m**2 + dist_shadow_19_to_sat_19**2
        # Dist to other sat should be greater than straight line + altitude path approx
        self.assertGreaterEqual(dist_shadow_18_to_sat_19, math.sqrt(pythag_straight_dist_sq))

        # Check cartesian calculation agrees with straight line GS distance
        # Uses dot notation on GS object now
        a = geodetic2cartesian(
            float(shadow_gs_18.latitude_degrees_str),
            float(shadow_gs_18.longitude_degrees_str),
            shadow_gs_18.elevation_m_float,
        )
        b = geodetic2cartesian(
            float(shadow_gs_19.latitude_degrees_str),
            float(shadow_gs_19.longitude_degrees_str),
            shadow_gs_19.elevation_m_float,
        )
        calc_straight_dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        self.assertAlmostEqual(
            calc_straight_dist, straight_shadow_distance_m, delta=20000
        )  # 20km tolerance
