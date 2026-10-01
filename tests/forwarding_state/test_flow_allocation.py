from types import SimpleNamespace

from leopath.network_state.routing_algorithms.flow_allocation import (
    allocate_address_pair,
    resolve_flow_address_pair,
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
    # The source has the lower id, so it holds the flow's pin.
    station.id = 1
    peer = SimpleNamespace(id=7, allocated_address_pairs={})
    return allocate_address_pair(
        "family", station, peer, sources, destinations, _cheapest, counters
    )


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


def test_both_directions_of_a_flow_share_one_pin_and_pay_once() -> None:
    low = SimpleNamespace(id=1, allocated_address_pairs={})
    high = SimpleNamespace(id=7, allocated_address_pairs={})
    low_sats, high_sats = [(1.0, 1), (2.0, 2)], [(1.0, 10), (2.0, 11)]
    counters = new_flow_allocation_counters()

    allocate_address_pair("family", low, high, low_sats, high_sats, _cheapest, counters)
    reverse = allocate_address_pair("family", high, low, high_sats, low_sats, _cheapest, counters)

    assert high.allocated_address_pairs == {}
    assert low.allocated_address_pairs[("family", 7)] == (1, 10)
    assert (reverse[1], reverse[3]) == (10, 1)
    assert counters["flow_allocations"] == 1

    # Sat 10 goes: the first direction to notice moves the flow, the other keeps it.
    high_sats = [(2.0, 11)]
    allocate_address_pair("family", high, low, high_sats, low_sats, _cheapest, counters)
    allocate_address_pair("family", low, high, low_sats, high_sats, _cheapest, counters)

    assert low.allocated_address_pairs[("family", 7)] == (1, 11)
    assert counters["flow_updates"] == 1
    assert counters["flow_update_messages"] == 1


def _requester_station(gs_id: int, current: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        id=gs_id,
        current_address_satellite_id=current,
        previous_current_address_satellite_id=current,
        requested_synonyms={},
    )


def test_requester_aware_resolves_the_best_synonym_and_keeps_it() -> None:
    a, b = _requester_station(1, 1), _requester_station(7, 10)
    counters = new_flow_allocation_counters()
    a_sats, b_sats = [(1.0, 1), (5.0, 2)], [(3.0, 10), (1.0, 11)]

    first = resolve_flow_address_pair(
        "requester_aware", "family", a, b, a_sats, b_sats, _cheapest, counters
    )
    assert (first[1], first[3]) == (1, 11)  # B's synonym 11 is best for A, though 10 is B's current
    assert counters["flow_allocations"] == 1

    # A cheaper uplink appears: the source end moves freely, the destination synonym stays.
    later = resolve_flow_address_pair(
        "requester_aware",
        "family",
        a,
        b,
        [(9.0, 1), (0.5, 2)],
        [(0.1, 10), (1.0, 11)],
        _cheapest,
        counters,
    )
    assert (later[1], later[3]) == (2, 11)
    assert counters["flow_updates"] == 0


def test_requester_aware_moves_a_withdrawn_synonym_to_the_current_address() -> None:
    a, b = _requester_station(1, 1), _requester_station(7, 10)
    a.requested_synonyms[("family", 7)] = 11
    counters = new_flow_allocation_counters()

    moved = resolve_flow_address_pair(
        "requester_aware", "family", a, b, [(1.0, 1)], [(3.0, 10)], _cheapest, counters
    )
    assert moved[3] == 10
    assert counters["flow_updates"] == 1 and counters["flow_update_messages"] == 1


def test_requester_aware_does_not_double_count_a_renumbering() -> None:
    a, b = _requester_station(1, 1), _requester_station(7, 12)
    b.previous_current_address_satellite_id = 11  # B renumbered 11 -> 12 this snapshot
    a.requested_synonyms[("family", 7)] = 11
    counters = new_flow_allocation_counters()

    moved = resolve_flow_address_pair(
        "requester_aware", "family", a, b, [(1.0, 1)], [(3.0, 12)], _cheapest, counters
    )
    assert moved[3] == 12
    assert counters["flow_update_messages"] == 0
