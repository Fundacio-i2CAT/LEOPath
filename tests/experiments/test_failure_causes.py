import networkx as nx

from leopath.experiments.metrics import compute_path_stretch

GS_SRC = 100
GS_DST = 101


def _delivery(
    fstate: dict,
    graph: nx.Graph,
    satellite_ids: list[int],
    src_sat: int,
    dst_sat: int,
    route_plans: dict | None = None,
) -> dict:
    return compute_path_stretch(
        fstate=fstate,
        topology_graph=graph,
        satellite_ids=satellite_ids,
        ground_station_ids=[GS_SRC, GS_DST],
        attachments=[(src_sat, 10.0), (dst_sat, 20.0)],
        interface_neighbor_map={},
        max_hops=len(satellite_ids) + 2,
        route_plans=route_plans,
        ground_station_satellites_in_range=[[(10.0, src_sat)], [(20.0, dst_sat)]],
    )["delivery"]


def _assert_causes_account_for_every_failure(delivery: dict) -> None:
    causes = sum(value for key, value in delivery.items() if key.startswith("failure_"))
    assert causes == delivery["forwarding_failure"]


def test_forwarding_loop_and_missing_route_are_told_apart() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0)])
    fstate = {
        (0, GS_DST): (1, 0, 0),
        (1, GS_DST): (0, 0, 0),  # sent straight back: a loop
    }
    delivery = _delivery(fstate, graph, [0, 1, 2], src_sat=0, dst_sat=2)

    assert delivery["deliverable"] == 2.0
    assert delivery["failure_loop"] == 1.0
    assert delivery["failure_dead_end"] == 1.0  # nothing installed toward GS_SRC
    _assert_causes_account_for_every_failure(delivery)


def test_entry_over_a_failed_link_is_link_down() -> None:
    # Link 1-2 has failed but satellite 1 still forwards over it; 0-3-2 survives.
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (0, 3, 1.0), (3, 2, 1.0)])
    fstate = {
        (0, GS_DST): (1, 0, 0),
        (1, GS_DST): (2, 0, 0),
        (2, GS_SRC): (3, 0, 0),
        (3, GS_SRC): (0, 0, 0),
        (0, GS_SRC): (GS_SRC, 0, 0),
    }
    delivery = _delivery(fstate, graph, [0, 1, 2, 3], src_sat=0, dst_sat=2)

    assert delivery["deliverable"] == 2.0
    assert delivery["delivered"] == 1.0
    assert delivery["failure_link_down"] == 1.0
    _assert_causes_account_for_every_failure(delivery)


def test_stale_explicit_plan_over_a_failed_link_is_link_down() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (0, 3, 1.0), (3, 2, 1.0)])
    route_plans = {
        (0, GS_DST): {
            "satellite_path": [0, 1, 2],
            "adjacency_sid_list": [1, 2],
            "planned_dst_sat_id": 2,
        }
    }
    delivery = _delivery({}, graph, [0, 1, 2, 3], src_sat=0, dst_sat=2, route_plans=route_plans)

    assert delivery["failure_link_down"] == 1.0
    _assert_causes_account_for_every_failure(delivery)
