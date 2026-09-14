import networkx as nx
import pytest

from leopath.experiments.failures import (
    FailureConfig,
    FailureProcess,
    SnapshotFailures,
    apply_failures,
)

PLANES = 6
SATS_PER_PLANE = 8


def _grid_isls(planes: int = PLANES, sats: int = SATS_PER_PLANE) -> list[tuple[int, int]]:
    isls = set()
    for plane in range(planes):
        for index in range(sats):
            sat = plane * sats + index
            same_plane = plane * sats + (index + 1) % sats
            next_plane = ((plane + 1) % planes) * sats + index
            isls.add((min(sat, same_plane), max(sat, same_plane)))
            isls.add((min(sat, next_plane), max(sat, next_plane)))
    return sorted(isls)


def _process(config: FailureConfig, latitude_provider=None) -> FailureProcess:
    return FailureProcess(
        config,
        n_orbits=PLANES,
        n_sats_per_orbit=SATS_PER_PLANE,
        undirected_isls=_grid_isls(),
        time_step_minutes=1.0,
        stream_key="test|grid",
        latitude_provider=latitude_provider,
    )


def _pattern(config: FailureConfig, steps: int = 20) -> list[set]:
    process = _process(config)
    return [process.snapshot(None).failed_isls for _ in range(steps)]


class _Topology:
    """Three satellites in a line, the middle and last one seen by GS 100."""

    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_edge(0, 1, weight=1.0)
        self.graph.add_edge(1, 2, weight=1.0)
        self.graph.add_edge(1, 100, weight=5.0)
        self.graph.add_edge(2, 100, weight=6.0)
        self.sat_neighbor_to_if = {(0, 1): 0, (1, 0): 0, (1, 2): 1, (2, 1): 0}
        self.nominal_graph = None


def test_no_failures_by_default() -> None:
    process = _process(FailureConfig())
    assert all(process.snapshot(None).is_empty() for _ in range(5))


def test_isl_outage_holds_stationary_rate_and_mean_duration() -> None:
    process = _process(
        FailureConfig(failure_type="isl", rate=0.1, mean_duration_minutes=10.0, seed=3)
    )
    steps = 4000
    down_samples = 0
    outage_lengths = []
    running: dict = {}
    previous: set = set()
    for _ in range(steps):
        down = process.snapshot(None).failed_isls
        down_samples += len(down)
        for isl in down:
            running[isl] = running.get(isl, 0) + 1
        for isl in previous - down:
            outage_lengths.append(running.pop(isl))
        previous = down

    assert down_samples / (steps * len(_grid_isls())) == pytest.approx(0.1, abs=0.01)
    assert sum(outage_lengths) / len(outage_lengths) == pytest.approx(10.0, rel=0.1)


def test_same_seed_reproduces_the_pattern_and_other_seeds_differ() -> None:
    config = FailureConfig(failure_type="isl", rate=0.05, seed=7)
    assert _pattern(config) == _pattern(config)
    assert _pattern(config) != _pattern(FailureConfig(failure_type="isl", rate=0.05, seed=8))


def test_satellite_outage_draws_satellites_not_links() -> None:
    failures = _process(FailureConfig(failure_type="satellite", rate=0.2, seed=1)).snapshot(None)
    assert failures.failed_satellites
    assert all(0 <= sat < PLANES * SATS_PER_PLANE for sat in failures.failed_satellites)
    assert not failures.failed_isls


def test_void_is_a_static_contiguous_block() -> None:
    process = _process(FailureConfig(failure_type="void", void_size=3, seed=5))
    block = process.snapshot(None).failed_satellites
    planes = {sat // SATS_PER_PLANE for sat in block}
    indices = {sat % SATS_PER_PLANE for sat in block}

    assert len(block) == 9
    assert any(planes == {(s + d) % PLANES for d in range(3)} for s in range(PLANES))
    assert any(
        indices == {(s + d) % SATS_PER_PLANE for d in range(3)} for s in range(SATS_PER_PLANE)
    )
    assert process.snapshot(None).failed_satellites == block


def test_plane_cut_partitions_the_torus() -> None:
    failures = _process(FailureConfig(failure_type="cut", seed=2)).snapshot(None)
    graph = nx.Graph(_grid_isls())
    graph.remove_edges_from(failures.failed_isls)

    assert nx.number_connected_components(graph) == 2
    assert all(a // SATS_PER_PLANE != b // SATS_PER_PLANE for a, b in failures.failed_isls)


def test_polar_deactivation_removes_only_high_latitude_inter_plane_links() -> None:
    def latitudes(_time: object) -> dict[int, float]:
        # Index 0 of every plane sits above the threshold, everything else on the equator.
        return {
            sat: (80.0 if sat % SATS_PER_PLANE == 0 else 0.0)
            for sat in range(PLANES * SATS_PER_PLANE)
        }

    config = FailureConfig(failure_type="polar", polar_latitude_deg=75.0)
    failures = _process(config, latitudes).snapshot(None)

    assert len(failures.failed_isls) == PLANES
    for sat_a, sat_b in failures.failed_isls:
        assert sat_a // SATS_PER_PLANE != sat_b // SATS_PER_PLANE
        assert sat_a % SATS_PER_PLANE == 0 and sat_b % SATS_PER_PLANE == 0


def test_polar_deactivation_needs_latitudes() -> None:
    with pytest.raises(ValueError):
        _process(FailureConfig(failure_type="polar"))


def test_rejects_rates_the_model_cannot_hold() -> None:
    with pytest.raises(ValueError):
        FailureConfig(failure_type="isl", rate=1.0)
    with pytest.raises(ValueError):
        _process(FailureConfig(failure_type="isl", rate=0.6, mean_duration_minutes=1.0))


def test_apply_failures_removes_links_interfaces_and_visibility() -> None:
    topology = _Topology()
    visibility = [[(5.0, 1), (6.0, 2)]]
    stats = apply_failures(topology, visibility, SnapshotFailures(failed_satellites={1}))

    assert list(topology.graph.edges()) == [(2, 100)]
    assert topology.sat_neighbor_to_if == {}
    assert visibility == [[(6.0, 2)]]
    assert stats == {"isls_removed": 2.0, "satellites_down": 1.0}


def test_inject_keeps_the_failure_free_graph() -> None:
    topology = _Topology()
    process = FailureProcess(
        FailureConfig(failure_type="void", void_size=1, seed=0),
        n_orbits=1,
        n_sats_per_orbit=3,
        undirected_isls=[(0, 1), (1, 2)],
        time_step_minutes=1.0,
        stream_key="test",
    )
    stats = process.inject(topology, [[(5.0, 1), (6.0, 2)]], None)

    assert topology.nominal_graph is not topology.graph
    assert topology.nominal_graph.number_of_edges() == 4
    assert topology.graph.number_of_edges() < 4
    assert stats["satellites_down"] == 1.0


def test_inject_without_failures_shares_the_graph() -> None:
    topology = _Topology()
    process = FailureProcess(
        FailureConfig(),
        n_orbits=1,
        n_sats_per_orbit=3,
        undirected_isls=[(0, 1), (1, 2)],
        time_step_minutes=1.0,
        stream_key="test",
    )
    process.inject(topology, [], None)
    assert topology.nominal_graph is topology.graph
