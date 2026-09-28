from types import SimpleNamespace

from leopath.network_state.routing_algorithms.flow_allocation import (
    allocate_address_pair,
    update_current_addresses,
    new_flow_allocation_counters,
)


def _cheapest(sources: list, destinations: list) -> tuple | None:
    pairs = [
        (src_dist + dst_dist, (src_dist, src_sat, dst_dist, dst_sat))
        for src_dist, src_sat in sources
        for dst_dist, dst_sat in destinations
    ]
    return min(pairs)[1] if pairs else None


def _allocate(station, sources, destinations, counters):
    return allocate_address_pair("family", station, 7, sources, destinations, _cheapest, counters)


def test_first_allocation_picks_the_cheapest_pair() -> None:
    station = SimpleNamespace(allocated_address_pairs={})
    counters = new_flow_allocation_counters()

    selected = _allocate(station, [(5.0, 1), (1.0, 2)], [(3.0, 10), (1.0, 11)], counters)

    assert (selected[1], selected[3]) == (2, 11)
    assert counters == {**new_flow_allocation_counters(), "flow_allocations": 1}


def test_flow_keeps_its_pair_while_both_synonyms_are_advertised() -> None:
    station = SimpleNamespace(allocated_address_pairs={("family", 7): (1, 10)})
    counters = new_flow_allocation_counters()

    # (2, 11) is now cheaper, but the flow's addresses are still valid.
    selected = _allocate(station, [(5.0, 1), (1.0, 2)], [(3.0, 10), (1.0, 11)], counters)

    assert (selected[1], selected[3]) == (1, 10)
    assert counters == new_flow_allocation_counters()


def test_withdrawn_destination_synonym_moves_the_flow_with_one_update() -> None:
    station = SimpleNamespace(allocated_address_pairs={("family", 7): (1, 10)})
    counters = new_flow_allocation_counters()

    selected = _allocate(station, [(1.0, 1)], [(1.0, 11)], counters)

    assert (selected[1], selected[3]) == (1, 11)
    assert station.allocated_address_pairs[("family", 7)] == (1, 11)
    assert counters == {
        **new_flow_allocation_counters(),
        "flow_updates": 1,
        "flow_update_messages": 1,
    }


def test_both_ends_renumbered_costs_two_update_messages() -> None:
    station = SimpleNamespace(allocated_address_pairs={("family", 7): (1, 10)})
    counters = new_flow_allocation_counters()

    _allocate(station, [(1.0, 2)], [(1.0, 11)], counters)

    assert counters["flow_updates"] == 1
    assert counters["flow_update_messages"] == 2


def test_unreachable_flow_keeps_its_pin_until_a_pair_returns() -> None:
    station = SimpleNamespace(allocated_address_pairs={("family", 7): (1, 10)})
    counters = new_flow_allocation_counters()

    assert _allocate(station, [], [(1.0, 10)], counters) is None
    assert station.allocated_address_pairs[("family", 7)] == (1, 10)
    assert counters == new_flow_allocation_counters()


def _stations(count: int) -> list:
    return [SimpleNamespace(current_address_satellite_id=None) for _ in range(count)]


def test_nearest_policy_renumbers_whenever_the_nearest_satellite_changes() -> None:
    stations = _stations(3)
    counters = new_flow_allocation_counters()
    update_current_addresses(stations, [[(1.0, 5), (2.0, 6)], [], []], "nearest", counters)
    update_current_addresses(stations, [[(3.0, 5), (2.0, 6)], [], []], "nearest", counters)

    assert stations[0].current_address_satellite_id == 6
    # One renumbering, one flow update to each of the two peers.
    assert counters["gs_current_address_changes"] == 1
    assert counters["flow_update_messages"] == 2


def test_sticky_policy_keeps_an_address_whose_attachment_survives() -> None:
    stations = _stations(3)
    counters = new_flow_allocation_counters()
    update_current_addresses(stations, [[(1.0, 5), (2.0, 6)], [], []], "sticky_nearest", counters)
    update_current_addresses(stations, [[(3.0, 5), (2.0, 6)], [], []], "sticky_nearest", counters)

    assert stations[0].current_address_satellite_id == 5
    assert counters == new_flow_allocation_counters()


def test_detached_station_keeps_its_last_address_and_first_address_is_free() -> None:
    stations = _stations(2)
    counters = new_flow_allocation_counters()
    update_current_addresses(stations, [[(1.0, 5)], []], "sticky_nearest", counters)
    update_current_addresses(stations, [[], []], "sticky_nearest", counters)

    assert stations[0].current_address_satellite_id == 5
    assert stations[1].current_address_satellite_id is None
    assert counters == new_flow_allocation_counters()
