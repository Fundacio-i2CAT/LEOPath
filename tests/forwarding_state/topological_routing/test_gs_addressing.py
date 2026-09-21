"""Attachment addressing: a ground station's address names its attachment.

Under ``gs_addressing: attachment`` the destination a packet carries is one
satellite, so a forwarding satellite chooses nothing and needs nothing about
where the ground station sits on the surface. Under ``visibility`` the address
is stable and every satellite minimises over all visible egresses instead.
"""

import ephem
import networkx as nx
import pytest

from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    GS_ADDRESSING,
    _exception_egresses,
    _select_gs_attachments,
    calculate_fstate_topological_routing_no_gs_relay,
)
from leopath.topology.satellite.satellite import Satellite
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology


def _address(plane: int, index: int) -> TopologicalNetworkAddress:
    return TopologicalNetworkAddress(0, plane, index, 0)


def _candidates(*entries: tuple[float, int]) -> list:
    return [(distance, sat_id, _address(sat_id // 8, sat_id % 8)) for distance, sat_id in entries]


def test_attachment_is_the_shortest_ground_link() -> None:
    candidates = [_candidates((900_000.0, 12), (400_000.0, 7), (650_000.0, 20))]

    attachments = _select_gs_attachments(candidates)

    assert len(attachments) == 1
    assert len(attachments[0]) == 1
    distance, sat_id, address = attachments[0][0]
    assert sat_id == 7
    assert distance == 400_000.0
    assert address.get_satellite_address().sat_index == 7


def test_a_ground_station_with_no_visible_satellite_keeps_no_attachment() -> None:
    assert _select_gs_attachments([[]]) == [[]]


def test_every_ground_station_keeps_its_index() -> None:
    candidates = [
        _candidates((500_000.0, 3)),
        [],
        _candidates((800_000.0, 9), (100_000.0, 31)),
    ]

    attachments = _select_gs_attachments(candidates)

    assert len(attachments) == 3
    assert attachments[0][0][1] == 3
    assert attachments[1] == []
    assert attachments[2][0][1] == 31


def test_a_failed_attachment_is_replaced_rather_than_stranding_the_station() -> None:
    # apply_failures strips failed satellites from the visibility list before
    # routing runs, so the next snapshot simply attaches to the best survivor.
    before = _candidates((400_000.0, 7), (650_000.0, 20))
    assert _select_gs_attachments([before])[0][0][1] == 7

    after_failure = _candidates((650_000.0, 20))
    assert _select_gs_attachments([after_failure])[0][0][1] == 20


def test_attachment_leaves_one_egress_to_evaluate_instead_of_many() -> None:
    candidates = [_candidates(*((1_000.0 * n, n) for n in range(1, 17)))]

    assert len(candidates[0]) == 16
    assert len(_select_gs_attachments(candidates)[0]) == 1


def test_only_the_two_addressing_policies_exist() -> None:
    assert GS_ADDRESSING == ("visibility", "attachment")


def _topology(satellite_ids: list[int], gs_ids: list[int], isl_edges: list[tuple]):
    """A small topology the real entry point can run over."""
    satellites = []
    for offset, sat_id in enumerate(satellite_ids):
        body = ephem.EarthSatellite(
            "1 25544U 98067A   21001.00000000  .00001000  00000-0  23027-4 0  9990",
            f"2 25544  51.640{offset:02d} 339.704{offset:02d} 0003572  "
            f"86.486{offset:02d} 273.608{offset:02d} 15.48919103270233",
        )
        satellite = Satellite(id=sat_id, ephem_obj_manual=body, ephem_obj_direct=body)
        satellite.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(
            sat_id
        )
        satellites.append(satellite)

    ground_stations = [
        GroundStation(
            gid=gs_id,
            name=f"GS{gs_id}",
            latitude_degrees_str="0",
            longitude_degrees_str="0",
            elevation_m_float=0,
            cartesian_x=0,
            cartesian_y=0,
            cartesian_z=0,
        )
        for gs_id in gs_ids
    ]

    constellation = ConstellationData(
        orbits=1,
        sats_per_orbit=len(satellites),
        epoch="2024-01-01T00:00:00.000000000",
        max_gsl_length_m=5_000_000,
        max_isl_length_m=5_000_000,
        satellites=satellites,
    )
    topology = LEOTopology(constellation, ground_stations)
    topology.sat_neighbor_to_if = {}
    interfaces = {sat_id: 0 for sat_id in satellite_ids}
    for satellite in satellites:
        topology.graph.add_node(satellite.id)
        satellite.number_isls = 0
    for node_a, node_b, weight in isl_edges:
        topology.graph.add_edge(node_a, node_b, weight=weight)
        topology.sat_neighbor_to_if[(node_a, node_b)] = interfaces[node_a]
        topology.sat_neighbor_to_if[(node_b, node_a)] = interfaces[node_b]
        interfaces[node_a] += 1
        interfaces[node_b] += 1
    for satellite in satellites:
        satellite.number_isls = interfaces[satellite.id]
    return topology, ground_stations


def _run(
    gs_addressing: str | None,
    visibility: list,
    state_report: dict | None = None,
    built: tuple | None = None,
    time_since_epoch_ns: int = 0,
) -> dict:
    topology, ground_stations = built or _topology([10, 11], [100], [(10, 11, 1000.0)])
    params = {} if gs_addressing is None else {"gs_addressing": gs_addressing}
    return calculate_fstate_topological_routing_no_gs_relay(
        topology,
        ground_stations,
        visibility,
        time_since_epoch_ns=time_since_epoch_ns,
        prev_fstate=None,
        graph_has_changed=True,
        algorithm_params=params,
        state_report=state_report,
    )


def test_an_unknown_addressing_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown gs_addressing"):
        _run("nearest-thing", [[(500.0, 10)]])


def test_both_policies_agree_when_only_one_satellite_is_visible() -> None:
    visibility = [[(500.0, 10)]]

    assert _run("visibility", visibility) == _run("attachment", visibility)


def test_visibility_is_the_default() -> None:
    visibility = [[(500.0, 10), (900.0, 11)]]

    assert _run(None, visibility) == _run("visibility", visibility)


def test_the_first_attachment_is_not_counted_as_a_renumbering() -> None:
    report: dict = {}
    _run("attachment", [[(500.0, 10)]], state_report=report)

    # The ground station had no address to change, so nothing was renumbered.
    assert report["aux_gs_renumberings"] == 0.0


def test_moving_the_attachment_counts_one_renumbering() -> None:
    built = _topology([10, 11], [100], [(10, 11, 1000.0)])

    first: dict = {}
    _run("attachment", [[(500.0, 10)]], state_report=first, built=built)
    assert first["aux_gs_renumberings"] == 0.0

    # Satellite 10 sets, satellite 11 rises: the address has to follow.
    second: dict = {}
    _run(
        "attachment",
        [[(600.0, 11)]],
        state_report=second,
        built=built,
        time_since_epoch_ns=60_000_000_000,
    )
    assert second["aux_gs_renumberings"] == 1.0


def test_a_steady_attachment_costs_no_renumbering() -> None:
    built = _topology([10, 11], [100], [(10, 11, 1000.0)])
    _run("attachment", [[(500.0, 10)]], state_report={}, built=built)

    report: dict = {}
    _run(
        "attachment",
        [[(520.0, 10)]],
        state_report=report,
        built=built,
        time_since_epoch_ns=60_000_000_000,
    )

    assert report["aux_gs_renumberings"] == 0.0


def test_exceptions_deliver_only_through_the_attachment() -> None:
    visibility = [[(500.0, 10), (900.0, 11)]]
    candidates = [[(500.0, 10, _address(0, 0)), (900.0, 11, _address(0, 1))]]
    attached = _select_gs_attachments(candidates)

    assert _exception_egresses(visibility, attached, "attachment") == [[(500.0, 10)]]


def test_exceptions_keep_every_visible_egress_under_visibility() -> None:
    visibility = [[(500.0, 10), (900.0, 11)]]
    candidates = [[(500.0, 10, _address(0, 0)), (900.0, 11, _address(0, 1))]]

    assert _exception_egresses(visibility, candidates, "visibility") is visibility


def test_no_satellite_but_the_attachment_gets_a_ground_link_entry() -> None:
    # Satellite 10 is the attachment but has lost both of its links; 11 and 12
    # still talk to each other, and 11 also sees the station. An exception that
    # treated every visible satellite as an egress would hand 11 a ground link
    # the station does not have.
    built = _topology([10, 11, 12], [100], [(11, 12, 1000.0)])
    topology, _ = built
    topology.nominal_graph = nx.Graph([(10, 11), (11, 12), (12, 10)])
    fstate = calculate_fstate_topological_routing_no_gs_relay(
        topology,
        built[1],
        [[(500.0, 10), (900.0, 11)]],
        time_since_epoch_ns=0,
        prev_fstate=None,
        graph_has_changed=True,
        algorithm_params={
            "gs_addressing": "attachment",
            "distance_mode": "torus_unit",
            "forwarding_guard": "progress",
            "exception_policy": "grow",
        },
    )

    ground_links = {
        sat for (sat, gs), entry in fstate.items() if gs == 100 and entry == ("GSL", 100)
    }
    assert ground_links <= {10}
