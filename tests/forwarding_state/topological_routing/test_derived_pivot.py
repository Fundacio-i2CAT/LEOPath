"""The table-free pivot estimator must agree with the tabled one everywhere.

If it does, the P*S^2 + S*P^2 path-cost tables are provably a cache: a
satellite holding only the shell's constants computes the same distance.
"""

from types import SimpleNamespace

import networkx as nx
import pytest

from leopath.network_state.routing_algorithms.topological_routing.derived_pivot import (
    DerivedPivotEstimator,
)
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    _build_torus_weight_model,
    _torus_weighted_pivot_distance,
)
from leopath.topology.walker_geometry import WalkerShell


def _address(plane: int, slot: int) -> SimpleNamespace:
    satellite = SimpleNamespace(shell_id=0, plane_id=plane, sat_index=slot)
    return SimpleNamespace(get_satellite_address=lambda: satellite)


def _tabled_model(shell: WalkerShell, time_s: float, wrap: bool):
    planes, slots = shell.planes, shell.sats_per_plane
    graph = nx.Graph()
    addresses = {}
    for plane in range(planes):
        for slot in range(slots):
            addresses[plane * slots + slot] = _address(plane, slot)
    for plane in range(planes):
        for slot in range(slots):
            node = plane * slots + slot
            rail_to = plane * slots + (slot + 1) % slots
            graph.add_edge(
                node,
                rail_to,
                weight=shell.distance_m((plane, slot), (plane, (slot + 1) % slots), time_s),
            )
            if plane + 1 < planes or wrap:
                rung_to = ((plane + 1) % planes) * slots + slot
                graph.add_edge(
                    node,
                    rung_to,
                    weight=shell.distance_m((plane, slot), ((plane + 1) % planes, slot), time_s),
                )
    constellation = SimpleNamespace(n_orbits=planes, n_sats_per_orbit=slots)
    return _build_torus_weight_model(graph, addresses, constellation), addresses


SHELLS = [
    ("delta, even planes", WalkerShell(6, 8, 53.0, 15.05), True),
    ("delta, odd planes", WalkerShell(7, 8, 53.0, 15.05), True),
    ("star", WalkerShell(6, 9, 87.9, 13.16, raan_spread_deg=180.0), False),
    ("no phasing", WalkerShell(5, 6, 70.0, 14.8, phase_diff=False), True),
]


@pytest.mark.parametrize("label,shell,wrap", SHELLS, ids=[s[0] for s in SHELLS])
@pytest.mark.parametrize("time_s", [0.0, 2500.0, 11000.0])
def test_table_free_pivot_matches_the_tabled_pivot(label, shell, wrap, time_s) -> None:
    model, addresses = _tabled_model(shell, time_s, wrap)
    estimator = DerivedPivotEstimator(shell, time_s)
    assert estimator.cross_plane_wrap == wrap
    slots = shell.sats_per_plane
    for source in addresses:
        for destination in addresses:
            tabled = _torus_weighted_pivot_distance(
                addresses[source], addresses[destination], model
            )
            derived = estimator.distance(
                (source // slots, source % slots), (destination // slots, destination % slots)
            )
            assert derived == pytest.approx(tabled, rel=1e-9, abs=1e-6), (source, destination)
