import networkx as nx
import pytest

from src.network_state.utils.graph import validate_no_satellite_to_gs_links
from src.topology.topology import GroundStation, Satellite


def test_validate_no_satellite_to_gs_links_valid_graph():
    # Create a valid graph with only satellite-to-satellite and GS-to-GS links
    graph = nx.Graph()
    satellites = [
        Satellite(id=0, ephem_obj_manual=None, ephem_obj_direct=None),
        Satellite(id=1, ephem_obj_manual=None, ephem_obj_direct=None),
    ]
    ground_stations = [
        GroundStation(
            gid=2,
            name="GS1",
            latitude_degrees_str="0",
            longitude_degrees_str="0",
            elevation_m_float=0,
            cartesian_x=0,
            cartesian_y=0,
            cartesian_z=0,
        ),
        GroundStation(
            gid=3,
            name="GS2",
            latitude_degrees_str="0",
            longitude_degrees_str="0",
            elevation_m_float=0,
            cartesian_x=0,
            cartesian_y=0,
            cartesian_z=0,
        ),
    ]

    # Add valid edges
    graph.add_edge(0, 1)  # Satellite-to-satellite
    graph.add_edge(2, 3)  # GS-to-GS

    # Validate the graph
    validate_no_satellite_to_gs_links(graph, satellites, ground_stations)


def test_validate_no_satellite_to_gs_links_invalid_graph():
    # Create an invalid graph with a satellite-to-GS link
    graph = nx.Graph()
    satellites = [Satellite(id=0, ephem_obj_manual=None, ephem_obj_direct=None)]
    ground_stations = [
        GroundStation(
            gid=1,
            name="GS1",
            latitude_degrees_str="0",
            longitude_degrees_str="0",
            elevation_m_float=0,
            cartesian_x=0,
            cartesian_y=0,
            cartesian_z=0,
        )
    ]

    # Add invalid edge
    graph.add_edge(0, 1)  # Satellite-to-GS

    # Validate the graph and expect a ValueError
    with pytest.raises(ValueError, match="Invalid edge between satellite 0 and ground station 1"):
        validate_no_satellite_to_gs_links(graph, satellites, ground_stations)


def test_validate_no_satellite_to_gs_links_empty_graph():
    # Create an empty graph
    graph = nx.Graph()
    satellites = []
    ground_stations = []

    # Validate the empty graph
    validate_no_satellite_to_gs_links(graph, satellites, ground_stations)
