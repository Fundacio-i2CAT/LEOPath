# tests/dynamic_state/test_fstate_calculation_refactored.py

import unittest
from unittest.mock import MagicMock

import ephem  # For mocking spec
import networkx as nx

# Function to test
from src.dynamic_state.fstate_calculation import calculate_fstate_shortest_path_object_no_gs_relay

# Classes needed for setup
from src.dynamic_state.topology import (
    ConstellationData,
    GroundStation,
    LEOTopology,
    Satellite,
)


# --- Test Class ---
class TestFstateCalculationRefactored(unittest.TestCase):

    def _setup_scenario(
        self, satellite_list, ground_station_list, isl_edges_with_weights, gsl_visibility_list
    ):
        """Helper to build topology and visibility structures for fstate tests."""
        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellite_list),
            epoch="25001.0",
            max_gsl_length_m=5000000,
            max_isl_length_m=5000000,
            satellites=satellite_list,
        )
        topology = LEOTopology(constellation_data, ground_station_list)
        num_isls_per_sat_map = {sat.id: 0 for sat in satellite_list}
        topology.sat_neighbor_to_if = {}
        for sat in satellite_list:
            topology.graph.add_node(sat.id)
            sat.number_isls = 0
        for u_id, v_id, weight in isl_edges_with_weights:
            if topology.graph.has_node(u_id) and topology.graph.has_node(v_id):
                topology.graph.add_edge(u_id, v_id, weight=weight)
                u_if = num_isls_per_sat_map[u_id]
                v_if = num_isls_per_sat_map[v_id]
                topology.sat_neighbor_to_if[(u_id, v_id)] = u_if
                topology.sat_neighbor_to_if[(v_id, u_id)] = v_if
                num_isls_per_sat_map[u_id] += 1
                num_isls_per_sat_map[v_id] += 1
            else:
                print(f"Warning in test setup: Skipping edge ({u_id},{v_id}) - node(s) not found.")
        for sat in topology.constellation_data.satellites:
            sat.number_isls = num_isls_per_sat_map.get(sat.id, 0)
        if len(gsl_visibility_list) != len(ground_station_list):
            raise ValueError(f"Length mismatch: gsl_visibility_list vs ground_station_list")
        ground_station_satellites_in_range = gsl_visibility_list
        return topology, ground_station_satellites_in_range

    # --- Test Cases ---

    def test_one_sat_two_gs_refactored(self):
        """Scenario: 1 Sat (ID 10), 2 GS (IDs 100, 101), GSLs only"""
        # Diagram:
        #      10 (Sat)
        #     /  \
        # 100(GS) 101(GS)
        SAT_ID = 10
        GS_A_ID = 100
        GS_B_ID = 101
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [Satellite(id=SAT_ID, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body)]
        ground_stations = [
            GroundStation(
                gid=GS_A_ID,
                name="GA",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_B_ID,
                name="GB",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        isl_edges = []
        gsl_visibility = [[(1000, SAT_ID)], [(1000, SAT_ID)]]
        topology, visibility = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        fstate = calculate_fstate_shortest_path_object_no_gs_relay(
            topology, ground_stations, visibility
        )
        expected_fstate = {
            (SAT_ID, GS_A_ID): (GS_A_ID, 0, 0),
            (SAT_ID, GS_B_ID): (GS_B_ID, 0, 0),
            (GS_A_ID, GS_B_ID): (SAT_ID, 0, 0),
            (GS_B_ID, GS_A_ID): (SAT_ID, 0, 0),
        }
        self.assertDictEqual(fstate, expected_fstate)

    def test_two_sat_two_gs_refactored(self):
        """Scenario: Sat 10 -- Sat 11, GS 100 -> Sat 10, GS 101 -> Sat 11"""
        # Diagram: 100(GS) -- 10(Sat) -- 11(Sat) -- 101(GS)
        SAT_A = 10
        SAT_B = 11
        GS_X = 100
        GS_Y = 101
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        ground_stations = [
            GroundStation(
                gid=GS_X,
                name="GX",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_Y,
                name="GY",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        isl_edges = [(SAT_A, SAT_B, 1000)]
        gsl_visibility = [[(500, SAT_A)], [(600, SAT_B)]]
        topology, visibility = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        fstate = calculate_fstate_shortest_path_object_no_gs_relay(
            topology, ground_stations, visibility
        )
        expected_fstate = {
            (SAT_A, GS_X): (GS_X, 1, 0),
            (SAT_A, GS_Y): (SAT_B, 0, 0),
            (SAT_B, GS_X): (SAT_A, 0, 0),
            (SAT_B, GS_Y): (GS_Y, 1, 0),
            (GS_X, GS_Y): (SAT_A, 0, 1),
            (GS_Y, GS_X): (SAT_B, 0, 1),
        }
        self.assertDictEqual(fstate, expected_fstate)

    def test_two_sat_three_gs_refactored(self):
        """Scenario: Sat 10 -- Sat 11, GS 100 -> S10, GS 101 -> S10 & S11, GS 102 -> S11"""
        # Diagram:
        #            100(GS)
        #             /
        #   10(Sat) ----- 11(Sat)
        #      \       /       \
        #       101(GS)      102(GS)
        SAT_A = 10
        SAT_B = 11
        GS_X = 100
        GS_Y = 101
        GS_Z = 102
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        ground_stations = [
            GroundStation(
                gid=GS_X,
                name="GX",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_Y,
                name="GY",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_Z,
                name="GZ",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        isl_edges = [(SAT_A, SAT_B, 1000)]
        gsl_visibility = [[(500, SAT_A)], [(200, SAT_A), (100, SAT_B)], [(400, SAT_B)]]
        topology, visibility = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        fstate = calculate_fstate_shortest_path_object_no_gs_relay(
            topology, ground_stations, visibility
        )
        expected_fstate = {
            (SAT_A, GS_X): (GS_X, 1, 0),
            (SAT_A, GS_Y): (GS_Y, 1, 0),
            (SAT_A, GS_Z): (SAT_B, 0, 0),
            (SAT_B, GS_X): (SAT_A, 0, 0),
            (SAT_B, GS_Y): (GS_Y, 1, 0),
            (SAT_B, GS_Z): (GS_Z, 1, 0),
            (GS_X, GS_Y): (SAT_A, 0, 1),
            (GS_X, GS_Z): (SAT_A, 0, 1),
            (GS_Y, GS_X): (SAT_A, 0, 1),
            (GS_Y, GS_Z): (SAT_B, 0, 1),
            (GS_Z, GS_X): (SAT_B, 0, 1),
            (GS_Z, GS_Y): (SAT_B, 0, 1),
        }
        self.assertDictEqual(fstate, expected_fstate)

    def test_five_sat_five_gs_refactored(self):
        """Scenario: 5 Sats (10-14), 5 GS (105-109), complex ISLs"""
        # Diagram:
        #  107(GS)-- 10(Sat)      11(Sat) -- 106(GS)
        #           / |          | \
        #     108(GS) 13(Sat)   14(Sat)   109(GS)
        #      |         \     /          |
        #      +--------- 12(Sat) --------+
        #                 |
        #               105(GS)
        # --- Setup ---
        # Satellite IDs (map from old 0-4)
        SAT_0 = 10
        SAT_1 = 11
        SAT_2 = 12
        SAT_3 = 13
        SAT_4 = 14
        # Ground Station IDs (map from old 5-9)
        GS_5 = 105
        GS_6 = 106
        GS_7 = 107
        GS_8 = 108
        GS_9 = 109

        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_0, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_1, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_2, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_3, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_4, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        ground_stations = [
            GroundStation(
                gid=GS_5,
                name="G5",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_6,
                name="G6",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_7,
                name="G7",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_8,
                name="G8",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_9,
                name="G9",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]  # Order: GS_5(idx0), GS_6(idx1), GS_7(idx2), GS_8(idx3), GS_9(idx4)

        # Old ISLs: (0,3,600), (1,4,300), (2,3,400), (2,4,400)
        isl_edges = [
            (SAT_0, SAT_3, 600),
            (SAT_1, SAT_4, 300),
            (SAT_2, SAT_3, 400),
            (SAT_2, SAT_4, 400),
        ]

        # Old GSLs: (0,7,500), (0,8,600), (1,6,500), (1,9,300), (2,5,500), (2,8,200), (2,9,500)
        gsl_visibility = [
            # GS_5 (idx 0) sees Sat 2 (12) dist 500
            [(500, SAT_2)],
            # GS_6 (idx 1) sees Sat 1 (11) dist 500
            [(500, SAT_1)],
            # GS_7 (idx 2) sees Sat 0 (10) dist 500
            [(500, SAT_0)],
            # GS_8 (idx 3) sees Sat 0 (10) dist 600 AND Sat 2 (12) dist 200
            [(600, SAT_0), (200, SAT_2)],
            # GS_9 (idx 4) sees Sat 1 (11) dist 300 AND Sat 2 (12) dist 500
            [(300, SAT_1), (500, SAT_2)],
        ]

        topology, visibility = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )

        # --- Call Function ---
        fstate = calculate_fstate_shortest_path_object_no_gs_relay(
            topology, ground_stations, visibility
        )

        # --- Assertions ---
        # ISL Counts: SAT_0:1, SAT_1:1, SAT_2:2, SAT_3:2, SAT_4:2
        # IF Map (built by helper):
        # (10,13):0, (13,10):0
        # (11,14):0, (14,11):0
        # (12,13):0, (13,12):1
        # (12,14):1, (14,12):1
        # GSL IFs: SAT_0=1, SAT_1=1, SAT_2=2, SAT_3=2, SAT_4=2; All GS=0
        expected_fstate = {
            # Old Name -> New Name: Recalculated IFs
            # (0, 5) -> (10, 105): Path 10->13->12->105. Hop 13. IFs: (10,13)=0, (13,10)=0. -> (13, 0, 0)
            (SAT_0, GS_5): (SAT_3, 0, 0),
            # (0, 6) -> (10, 106): Path 10->13->12->14->11->106. Hop 13. IFs: (10,13)=0, (13,10)=0. -> (13, 0, 0)
            (SAT_0, GS_6): (SAT_3, 0, 0),
            # (0, 7) -> (10, 107): Direct. Hop 107. IFs: Sat IF=1, GS IF=0. -> (107, 1, 0)
            (SAT_0, GS_7): (GS_7, 1, 0),
            # (0, 8) -> (10, 108): Direct. Hop 108. IFs: Sat IF=1, GS IF=0. -> (108, 1, 0)
            (SAT_0, GS_8): (GS_8, 1, 0),
            # (0, 9) -> (10, 109): Path 10->13->12->14->11->109. Hop 13. IFs: (10,13)=0, (13,10)=0. -> (13, 0, 0)
            (SAT_0, GS_9): (SAT_3, 0, 0),
            # (1, 5) -> (11, 105): Path 11->14->12->105. Hop 14. IFs: (11,14)=0, (14,11)=0. -> (14, 0, 0)
            (SAT_1, GS_5): (SAT_4, 0, 0),
            # (1, 6) -> (11, 106): Direct. Hop 106. IFs: Sat IF=1, GS IF=0. -> (106, 1, 0)
            (SAT_1, GS_6): (GS_6, 1, 0),
            # (1, 7) -> (11, 107): Path 11->14->12->13->10->107. Hop 14. IFs: (11,14)=0, (14,11)=0. -> (14, 0, 0)
            (SAT_1, GS_7): (SAT_4, 0, 0),
            # (1, 8) -> (11, 108): Path 11->14->12->108. Hop 14. IFs: (11,14)=0, (14,11)=0. -> (14, 0, 0)
            (SAT_1, GS_8): (SAT_4, 0, 0),
            # (1, 9) -> (11, 109): Direct. Hop 109. IFs: Sat IF=1, GS IF=0. -> (109, 1, 0)
            (SAT_1, GS_9): (GS_9, 1, 0),
            # (2, 5) -> (12, 105): Direct. Hop 105. IFs: Sat IF=2, GS IF=0. -> (105, 2, 0)
            (SAT_2, GS_5): (GS_5, 2, 0),
            # (2, 6) -> (12, 106): Path 12->14->11->106. Hop 14. IFs: (12,14)=1, (14,12)=1. -> (14, 1, 1)
            (SAT_2, GS_6): (SAT_4, 1, 1),
            # (2, 7) -> (12, 107): Path 12->13->10->107. Hop 13. IFs: (12,13)=0, (13,12)=1. -> (13, 0, 1)
            (SAT_2, GS_7): (SAT_3, 0, 1),
            # (2, 8) -> (12, 108): Direct. Hop 108. IFs: Sat IF=2, GS IF=0. -> (108, 2, 0)
            (SAT_2, GS_8): (GS_8, 2, 0),
            # (2, 9) -> (12, 109): Direct. Hop 109. IFs: Sat IF=2, GS IF=0. -> (109, 2, 0)
            (SAT_2, GS_9): (GS_9, 2, 0),
            # (3, 5) -> (13, 105): Path 13->12->105. Hop 12. IFs: (13,12)=1, (12,13)=0. -> (12, 1, 0)
            (SAT_3, GS_5): (SAT_2, 1, 0),
            # (3, 6) -> (13, 106): Path 13->12->14->11->106. Hop 12. IFs: (13,12)=1, (12,13)=0. -> (12, 1, 0)
            (SAT_3, GS_6): (SAT_2, 1, 0),
            # (3, 7) -> (13, 107): Path 13->10->107. Hop 10. IFs: (13,10)=0, (10,13)=0. -> (10, 0, 0)
            (SAT_3, GS_7): (SAT_0, 0, 0),
            # (3, 8) -> (13, 108): Path 13->12->108. Hop 12. IFs: (13,12)=1, (12,13)=0. -> (12, 1, 0)
            (SAT_3, GS_8): (SAT_2, 1, 0),
            # (3, 9) -> (13, 109): Path 13->12->14->11->109. Hop 12. IFs: (13,12)=1, (12,13)=0. -> (12, 1, 0)
            (SAT_3, GS_9): (SAT_2, 1, 0),
            # (4, 5) -> (14, 105): Path 14->12->105. Hop 12. IFs: (14,12)=1, (12,14)=1. -> (12, 1, 1)
            (SAT_4, GS_5): (SAT_2, 1, 1),
            # (4, 6) -> (14, 106): Path 14->11->106. Hop 11. IFs: (14,11)=0, (11,14)=0. -> (11, 0, 0)
            (SAT_4, GS_6): (SAT_1, 0, 0),
            # (4, 7) -> (14, 107): Path 14->12->13->10->107. Hop 12. IFs: (14,12)=1, (12,14)=1. -> (12, 1, 1)
            (SAT_4, GS_7): (SAT_2, 1, 1),
            # (4, 8) -> (14, 108): Path 14->12->108. Hop 12. IFs: (14,12)=1, (12,14)=1. -> (12, 1, 1)
            (SAT_4, GS_8): (SAT_2, 1, 1),
            # (4, 9) -> (14, 109): Path 14->11->109. Hop 11. IFs: (14,11)=0, (11,14)=0. -> (11, 0, 0)
            (SAT_4, GS_9): (SAT_1, 0, 0),
            # GS -> GS Calculations (Example: GS_5 -> GS_6)
            # (5, 6) -> (105, 106): Path 105->12->14->11->106. Entry=12. Hop 12. IFs: GS IF=0, Sat GSL IF=2. -> (12, 0, 2)
            (GS_5, GS_6): (SAT_2, 0, 2),
            (GS_5, GS_7): (SAT_2, 0, 2),  # Path 105->12->13->10->107. Entry=12. Hop 12.
            (GS_5, GS_8): (SAT_2, 0, 2),  # Path 105->12->108. Entry=12. Hop 12.
            (GS_5, GS_9): (
                SAT_2,
                0,
                2,
            ),  # Path 105->12->14->11->109 or 105->12->109. Entry=12. Hop 12.
            (GS_6, GS_5): (
                SAT_1,
                0,
                1,
            ),  # Path 106->11->14->12->105. Entry=11. Hop 11. IFs: GS IF=0, Sat GSL IF=1.
            (GS_6, GS_7): (SAT_1, 0, 1),  # Path 106->11->14->12->13->10->107. Entry=11. Hop 11.
            (GS_6, GS_8): (SAT_1, 0, 1),  # Path 106->11->14->12->108. Entry=11. Hop 11.
            (GS_6, GS_9): (SAT_1, 0, 1),  # Path 106->11->109. Entry=11. Hop 11.
            (GS_7, GS_5): (
                SAT_0,
                0,
                1,
            ),  # Path 107->10->13->12->105. Entry=10. Hop 10. IFs: GS IF=0, Sat GSL IF=1.
            (GS_7, GS_6): (SAT_0, 0, 1),  # Path 107->10->13->12->14->11->106. Entry=10. Hop 10.
            (GS_7, GS_8): (SAT_0, 0, 1),  # Path 107->10->108. Entry=10. Hop 10.
            (GS_7, GS_9): (SAT_0, 0, 1),  # Path 107->10->13->12->14->11->109. Entry=10. Hop 10.
            (GS_8, GS_5): (
                SAT_2,
                0,
                2,
            ),  # Path 108->12->105. Entry=12 (cheaper). Hop 12. IFs: GS IF=0, Sat GSL IF=2.
            (GS_8, GS_6): (SAT_2, 0, 2),  # Path 108->12->14->11->106. Entry=12. Hop 12.
            (GS_8, GS_7): (
                SAT_0,
                0,
                1,
            ),  # Path 108->10->107. Entry=10 (cheaper). Hop 10. IFs: GS IF=0, Sat GSL IF=1.
            (GS_8, GS_9): (SAT_2, 0, 2),  # Path 108->12->109. Entry=12. Hop 12.
            (GS_9, GS_5): (
                SAT_1,
                0,
                1,
            ),  # Path 109->11->14->12->105. Entry=11 (cheaper). Hop 11. IFs: GS IF=0, Sat GSL IF=1.
            # (GS_9, GS_5): Old path 109->12->105. Entry=12. Hop 12. IFs: (109, 12) not direct. -> Let's trace. 109 sees 11(300), 12(500). GS5 sees 12(500).
            # Path via 11: 109->11->14->12->105. GSL(300)+ISL(300)+ISL(400)+GSL(500) = 1500.
            # Path via 12: 109->12->105. GSL(500)+GSL(500) = 1000. Choose entry 12. Hop 12. IFs: (GS=0, SatGSL=2) -> (12, 0, 2) <<-- Corrected
            (GS_9, GS_5): (SAT_2, 0, 2),
            (GS_9, GS_6): (SAT_1, 0, 1),  # Path 109->11->106. Entry=11. Hop 11.
            (GS_9, GS_7): (SAT_1, 0, 1),  # Path 109->11->14->12->13->10->107. Entry=11. Hop 11.
            # (GS_9, GS_7): Path via 12: 109->12->13->10->107. GSL(500)+ISL(400)+ISL(600)+GSL(500)=2000. Path via 11: GSL(300)+ISL(300)+ISL(400)+ISL(600)+GSL(500)=2100. Choose 12. Hop 12. -> (12, 0, 2) <<-- Corrected
            (GS_9, GS_7): (SAT_2, 0, 2),
            (GS_9, GS_8): (SAT_2, 0, 2),  # Path 109->12->108. Entry=12. Hop 12.
        }
        # Carefully compare calculated vs old, using new IF logic
        self.maxDiff = None  # Show full diff on failure
        self.assertDictEqual(fstate, expected_fstate)

    def test_two_sat_two_gs_no_isl_refactored(self):
        """Scenario: Sat 10, Sat 11 (no ISL). GS 100->S10, GS 101->S10&S11, GS 102->S11"""
        # Diagram:
        # 100(GS)    101(GS)    102(GS)
        #   \       /    \       /
        #    10(Sat)      11(Sat)
        #   /       \    /       \
        #         (No ISL)
        # --- Setup ---
        SAT_A = 10
        SAT_B = 11
        GS_X = 100
        GS_Y = 101
        GS_Z = 102
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        ground_stations = [
            GroundStation(
                gid=GS_X,
                name="GX",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_Y,
                name="GY",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_Z,
                name="GZ",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        isl_edges = []  # No ISLs
        gsl_visibility = [
            [(100, SAT_A)],  # GS X (idx 0) sees Sat A
            [(100, SAT_A), (100, SAT_B)],  # GS Y (idx 1) sees Sat A & B
            [(100, SAT_B)],  # GS Z (idx 2) sees Sat B
        ]

        topology, visibility = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )

        # --- Call Function ---
        fstate = calculate_fstate_shortest_path_object_no_gs_relay(
            topology, ground_stations, visibility
        )

        # --- Assertions ---
        # Sats A, B have 0 ISLs. Sat GSL IF = 0. GS GSL IF = 0.
        # No paths possible between sats, or between GSs via different sats.
        expected_fstate = {
            # Sat -> GS (Only direct GSLs possible)
            (SAT_A, GS_X): (GS_X, 0, 0),
            (SAT_A, GS_Y): (GS_Y, 0, 0),
            (SAT_A, GS_Z): (-1, -1, -1),  # Cannot reach
            (SAT_B, GS_X): (-1, -1, -1),  # Cannot reach
            (SAT_B, GS_Y): (GS_Y, 0, 0),
            (SAT_B, GS_Z): (GS_Z, 0, 0),
            # GS -> GS (Only possible if entry/exit via SAME satellite)
            (GS_X, GS_Y): (SAT_A, 0, 0),  # Path X->A->Y
            (GS_X, GS_Z): (-1, -1, -1),  # Needs ISL
            (GS_Y, GS_X): (SAT_A, 0, 0),  # Path Y->A->X (assume Y->A is chosen over Y->B->(fail))
            (GS_Y, GS_Z): (SAT_B, 0, 0),  # Path Y->B->Z (assume Y->B is chosen over Y->A->(fail))
            (GS_Z, GS_X): (-1, -1, -1),  # Needs ISL
            (GS_Z, GS_Y): (SAT_B, 0, 0),  # Path Z->B->Y
        }
        self.assertDictEqual(fstate, expected_fstate)

    def test_gsl_interface_index_calculation(self):
        """
        Unit test focusing on GSL interface index calculation for satellites.
        Scenario: Sat 10 -- Sat 20 -- Sat 30. GS 100 sees Sat 20. GS 101 sees Sat 10.
        Checks hops involving Sat 20 (which has 2 ISLs) to verify its GSL IF index.
        Expects Satellite GSL Interface Index = number of ISLs for that satellite.
        """
        # --- Setup ---
        # Define IDs
        SAT_A = 10
        SAT_B = 20  # The satellite whose GSL IF we are testing (should be 2)
        SAT_C = 30
        GS_X = 100
        GS_Y = 101

        # Create Satellite objects (ephem data is mocked, not used for path logic)
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_C, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        # Create GroundStation objects
        ground_stations = [
            GroundStation(
                gid=GS_X,
                name="GX",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_Y,
                name="GY",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]

        # Define ISLs: A(10) -- B(20) -- C(30)
        # Sat B (20) will have number_isls = 2
        isl_edges = [(SAT_A, SAT_B, 100), (SAT_C, SAT_B, 100)]  # Simple weights

        # Define GSL Visibility: GS X (100) sees Sat B (20); GS Y (101) sees Sat A (10)
        # Index 0 corresponds to GS_X, Index 1 corresponds to GS_Y
        gsl_visibility = [
            [(500, SAT_B)],  # GS X (index 0) sees Sat B
            [(500, SAT_A)],  # GS Y (index 1) sees Sat A
        ]

        # Use the helper to create topology object and formatted visibility list
        topology, visibility = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )

        # --- Verification of Setup ---
        # Double-check the ISL count for the satellite under test (Sat B / ID 20)
        try:
            sat_b_obj = topology.get_satellite(SAT_B)
            self.assertEqual(
                sat_b_obj.number_isls,
                2,
                "Test setup error: Satellite 20 should have 2 ISLs based on input.",
            )
            # Check ISL interface mapping created by helper (optional)
            self.assertEqual(
                topology.sat_neighbor_to_if.get((SAT_B, SAT_A)), 0, "IF Map incorrect"
            )  # B's 1st ISL
            self.assertEqual(
                topology.sat_neighbor_to_if.get((SAT_B, SAT_C)), 1, "IF Map incorrect"
            )  # B's 2nd ISL
        except KeyError as e:
            self.fail(f"Test setup error: Failed to get satellite or interface mapping: {e}")

        # --- Call Function Under Test ---
        fstate = calculate_fstate_shortest_path_object_no_gs_relay(
            topology, ground_stations, visibility
        )

        # --- Assertions ---
        # 1. Check Sat B (20) -> GS X (100) : Direct hop via GSL
        #    Expected Sat B GSL IF = number_isls = 2.
        #    Expected GS X IF = 0.
        #    Expected tuple: (GS_X, 2, 0) = (100, 2, 0)
        hop_tuple_sat_gs = fstate.get((SAT_B, GS_X))
        self.assertIsNotNone(hop_tuple_sat_gs, f"fstate missing for ({SAT_B=}, {GS_X=})")
        self.assertEqual(
            hop_tuple_sat_gs,
            (GS_X, 2, 0),
            "Incorrect hop/IFs for direct Sat->GS (Expecting Sat GSL IF=num_isls=2)",
        )

        # 2. Check GS X (100) -> GS Y (101): Path X -> B(20) -> A(10) -> Y
        #    First hop from GS X should be Sat B (20).
        #    Expected GS X IF = 0.
        #    Expected Sat B Incoming GSL IF = number_isls = 2.
        #    Expected tuple: (SAT_B, 0, 2) = (20, 0, 2)
        hop_tuple_gs_gs = fstate.get((GS_X, GS_Y))
        self.assertIsNotNone(hop_tuple_gs_gs, f"fstate missing for ({GS_X=}, {GS_Y=})")
        self.assertEqual(
            hop_tuple_gs_gs,
            (SAT_B, 0, 2),
            "Incorrect hop/IFs for GS->GS via Sat (Expecting Sat GSL IF=num_isls=2)",
        )
