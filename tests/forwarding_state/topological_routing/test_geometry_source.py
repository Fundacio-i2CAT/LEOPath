import math
from types import SimpleNamespace

import networkx as nx
import pytest

from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    _build_torus_weight_model,
    _geometry_subgraph,
)
from leopath.topology.constellation import ConstellationData
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress

PLANES = 3
SATS = 4


def _grid() -> nx.Graph:
    graph = nx.Graph()
    for plane in range(PLANES):
        for index in range(SATS):
            sat = plane * SATS + index
            graph.add_edge(sat, plane * SATS + (index + 1) % SATS, weight=600.0)
            graph.add_edge(sat, ((plane + 1) % PLANES) * SATS + index, weight=1500.0)
    return graph


def _addresses() -> dict:
    return {
        plane * SATS + index: TopologicalNetworkAddress(0, plane, index, 0)
        for plane in range(PLANES)
        for index in range(SATS)
    }


def test_nominal_geometry_ignores_a_failed_link_the_observed_geometry_sees() -> None:
    nominal = _grid()
    observed = nominal.copy()
    observed.remove_edge(0, 1)  # plane 0, between indices 0 and 1
    topology = SimpleNamespace(nominal_graph=nominal)
    satellite_ids = sorted(nominal.nodes())
    constellation = ConstellationData(PLANES, SATS, "20001.0", 1e6, 5e6, [])

    def first_row_edge_cost(source: str) -> float:
        graph = _geometry_subgraph(
            topology, satellite_ids, observed.subgraph(satellite_ids), source
        )
        model = _build_torus_weight_model(graph, _addresses(), constellation)
        return model["row_edge_costs"][0][0]

    assert math.isinf(first_row_edge_cost("observed"))
    assert first_row_edge_cost("nominal") == 600.0


def test_nominal_falls_back_to_the_snapshot_without_injected_failures() -> None:
    satellite_ids = list(range(PLANES * SATS))
    observed = _grid().subgraph(satellite_ids)
    topology = SimpleNamespace(nominal_graph=None)
    assert _geometry_subgraph(topology, satellite_ids, observed, "nominal") is observed


def test_unknown_geometry_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        _geometry_subgraph(SimpleNamespace(nominal_graph=None), [], nx.Graph(), "oracle")
