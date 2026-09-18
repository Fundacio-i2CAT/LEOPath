"""Seeded failure injection for evaluation runs.

Failures are applied to each snapshot after ISLs and ground visibility are
computed and before any routing algorithm runs, so every algorithm routes over
the same degraded graph. The failure pattern depends only on the seed, the
constellation, the ISL scenario and the failure parameters, never on the
algorithm, which makes comparisons between algorithms paired.

Every algorithm observes the post-failure graph in the snapshot a failure
occurs. For link-state that is close to reality at one-minute snapshots, since
flooding converges within seconds. The topological scheme's pivot geometry is
the one structure that should not see failures: under
``geometry_source="nominal"`` it is built from the failure-free graph kept in
``LEOTopology.nominal_graph``, as a satellite deriving geometry from
ephemerides would, while next hops still consider only live neighbours.
"""

import math
import random
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import ephem

from leopath.topology.distance_tools import _to_clean_ephem_string
from leopath.topology.satellite.satellite import Satellite
from leopath.topology.topology import LEOTopology

FAILURE_TYPES = ("none", "isl", "satellite", "void", "cut", "polar")

# Mean outage length used when a run does not set one.
DEFAULT_MEAN_DURATION_MINUTES = {"isl": 10.0, "satellite": 60.0}

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class FailureConfig:
    """Parameters of the failure model for one evaluation run.

    ``rate`` is the stationary probability that an element is down and applies
    to ``isl`` and ``satellite`` failures. ``void`` takes down a contiguous
    ``void_size`` by ``void_size`` block of satellites. ``cut`` removes every
    inter-plane link across two opposite plane boundaries, splitting a torus
    into two components. ``polar`` switches off inter-plane links with an
    endpoint above ``polar_latitude_deg``. Void and cut placement is drawn from
    ``seed``.
    """

    failure_type: str = "none"
    rate: float = 0.0
    seed: int = 0
    mean_duration_minutes: float | None = None
    void_size: int = 2
    polar_latitude_deg: float = 75.0

    def __post_init__(self) -> None:
        if self.failure_type not in FAILURE_TYPES:
            raise ValueError(f"Unknown failure type: {self.failure_type}")
        if not 0.0 <= self.rate < 1.0:
            raise ValueError(f"Failure rate must be in [0, 1), got {self.rate}")
        if self.mean_duration_minutes is not None and self.mean_duration_minutes <= 0.0:
            raise ValueError("Mean outage duration must be positive")
        if self.void_size < 1:
            raise ValueError("Void size must be at least 1")
        if not 0.0 < self.polar_latitude_deg < 90.0:
            raise ValueError("Polar latitude threshold must be in (0, 90) degrees")

    def resolved_mean_duration_minutes(self) -> float | None:
        if self.mean_duration_minutes is not None:
            return self.mean_duration_minutes
        return DEFAULT_MEAN_DURATION_MINUTES.get(self.failure_type)


@dataclass
class SnapshotFailures:
    """Elements down in one snapshot."""

    failed_isls: set[tuple[int, int]] = field(default_factory=set)
    failed_satellites: set[int] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not self.failed_isls and not self.failed_satellites


class _MarkovOutage(Generic[T]):
    """Independent two-state on/off process per element.

    Each element is down with stationary probability ``rate`` and, once down,
    stays down for ``mean_duration_steps`` snapshots on average. Initial states
    are drawn from the stationary distribution, so a run starts in steady state
    rather than failure-free. One random draw is consumed per element per
    snapshot whatever its state, so the pattern is reproducible from the seed.
    """

    def __init__(
        self,
        elements: Iterable[T],
        rate: float,
        mean_duration_steps: float,
        rng: random.Random,
    ) -> None:
        self._elements = list(elements)
        self._rng = rng
        self._repair_probability = 1.0 / max(1.0, mean_duration_steps)
        self._failure_probability = self._repair_probability * rate / (1.0 - rate)
        if self._failure_probability > 1.0:
            raise ValueError(
                f"Failure rate {rate} cannot be sustained with a mean outage of "
                f"{mean_duration_steps} snapshots"
            )
        self._down: set[T] = {element for element in self._elements if rng.random() < rate}
        self._initialised = False

    def step(self) -> set[T]:
        if not self._initialised:
            self._initialised = True
            return set(self._down)
        for element in self._elements:
            draw = self._rng.random()
            if element in self._down:
                if draw < self._repair_probability:
                    self._down.discard(element)
            elif draw < self._failure_probability:
                self._down.add(element)
        return set(self._down)


class FailureProcess:
    """Failure pattern for one run, advanced one snapshot at a time."""

    def __init__(
        self,
        config: FailureConfig,
        n_orbits: int,
        n_sats_per_orbit: int,
        undirected_isls: Iterable[tuple[int, int]],
        time_step_minutes: float,
        stream_key: str,
        latitude_provider: Callable[[object], dict[int, float]] | None = None,
    ) -> None:
        self.config = config
        self._planes = n_orbits
        self._sats_per_plane = n_sats_per_orbit
        self._isls = sorted({(min(a, b), max(a, b)) for a, b in undirected_isls})
        self._latitude_provider = latitude_provider
        # String seeds are hashed deterministically by random.Random, so the
        # pattern depends neither on PYTHONHASHSEED nor on the algorithm run.
        rng = random.Random(
            f"{config.seed}|{stream_key}|{config.failure_type}|{config.rate}|{config.void_size}"
        )
        duration_minutes = config.resolved_mean_duration_minutes() or time_step_minutes
        duration_steps = duration_minutes / time_step_minutes
        self._isl_outage: _MarkovOutage[tuple[int, int]] | None = None
        self._satellite_outage: _MarkovOutage[int] | None = None
        self._static = SnapshotFailures()
        self._previous: SnapshotFailures | None = None

        if config.failure_type == "isl":
            self._isl_outage = _MarkovOutage(self._isls, config.rate, duration_steps, rng)
        elif config.failure_type == "satellite":
            satellites = range(n_orbits * n_sats_per_orbit)
            self._satellite_outage = _MarkovOutage(satellites, config.rate, duration_steps, rng)
        elif config.failure_type == "void":
            self._static.failed_satellites = self._void_block(rng)
        elif config.failure_type == "cut":
            self._static.failed_isls = self._plane_cut(rng)
        elif config.failure_type == "polar" and latitude_provider is None:
            raise ValueError("Polar deactivation needs satellite latitudes")

    def snapshot(self, time_absolute: object) -> SnapshotFailures:
        """Elements down at this snapshot. Call once per snapshot, in order."""
        failures = SnapshotFailures(
            failed_isls=set(self._static.failed_isls),
            failed_satellites=set(self._static.failed_satellites),
        )
        if self._isl_outage is not None:
            failures.failed_isls = self._isl_outage.step()
        if self._satellite_outage is not None:
            failures.failed_satellites = self._satellite_outage.step()
        if self.config.failure_type == "polar" and self._latitude_provider is not None:
            failures.failed_isls = self._polar_links(self._latitude_provider(time_absolute))
        return failures

    def inject(
        self,
        topology: LEOTopology,
        ground_station_satellites_in_range: list,
        time_absolute: object,
    ) -> dict[str, float]:
        """Apply this snapshot's failures to the topology and ground visibility.

        The failure-free graph is kept on ``topology.nominal_graph`` first, so a
        structure that should not observe failures can still be built from it.
        """
        failures = self.snapshot(time_absolute)
        topology.nominal_graph = topology.graph if failures.is_empty() else topology.graph.copy()
        stats = apply_failures(topology, ground_station_satellites_in_range, failures)
        # Elements that failed or recovered since the previous snapshot: what a
        # failure-only flooding scheme would have to advertise. The first snapshot
        # counts every failure present, since none has been advertised yet.
        previous = self._previous or SnapshotFailures()
        stats["events"] = float(
            len(failures.failed_isls ^ previous.failed_isls)
            + len(failures.failed_satellites ^ previous.failed_satellites)
        )
        self._previous = failures
        return stats

    def describe(self) -> dict:
        return {
            "failure_type": self.config.failure_type,
            "rate": self.config.rate,
            "seed": self.config.seed,
            "mean_duration_minutes": self.config.resolved_mean_duration_minutes(),
            "void_size": self.config.void_size,
            "polar_latitude_deg": self.config.polar_latitude_deg,
            "knowledge": (
                "all algorithms route over the post-failure graph of each snapshot; "
                "topological routing with geometry_source=nominal builds its pivot "
                "geometry from the failure-free graph"
            ),
        }

    def _plane_of(self, satellite_id: int) -> int:
        return satellite_id // self._sats_per_plane

    def _void_block(self, rng: random.Random) -> set[int]:
        size = self.config.void_size
        if size > self._planes or size > self._sats_per_plane:
            raise ValueError(f"A void of size {size} does not fit the constellation")
        first_plane = rng.randrange(self._planes)
        first_index = rng.randrange(self._sats_per_plane)
        return {
            ((first_plane + plane_step) % self._planes) * self._sats_per_plane
            + (first_index + index_step) % self._sats_per_plane
            for plane_step in range(size)
            for index_step in range(size)
        }

    def _plane_cut(self, rng: random.Random) -> set[tuple[int, int]]:
        first = rng.randrange(self._planes)
        boundaries = {first, (first + self._planes // 2) % self._planes}
        cut: set[tuple[int, int]] = set()
        for sat_a, sat_b in self._isls:
            planes = {self._plane_of(sat_a), self._plane_of(sat_b)}
            if len(planes) == 1:
                continue
            if any(planes == {b, (b + 1) % self._planes} for b in boundaries):
                cut.add((sat_a, sat_b))
        return cut

    def _polar_links(self, latitudes_deg: dict[int, float]) -> set[tuple[int, int]]:
        threshold = self.config.polar_latitude_deg
        return {
            (sat_a, sat_b)
            for sat_a, sat_b in self._isls
            if self._plane_of(sat_a) != self._plane_of(sat_b)
            and max(abs(latitudes_deg.get(sat_a, 0.0)), abs(latitudes_deg.get(sat_b, 0.0)))
            > threshold
        }


def apply_failures(
    topology: LEOTopology,
    ground_station_satellites_in_range: list,
    failures: SnapshotFailures,
) -> dict[str, float]:
    """Remove failed links and satellites from a snapshot, in place.

    A failed satellite loses every link, ISL and GSL alike, and disappears from
    ground visibility. Removed ISLs also leave the interface map, so forwarding
    state pointing at them resolves to no neighbour. Interfaces of surviving
    links keep their indices, as a real node's would.
    """
    graph = topology.graph
    failed_edges: set[tuple[int, int]] = set(failures.failed_isls)
    for satellite_id in failures.failed_satellites:
        if graph.has_node(satellite_id):
            failed_edges.update(
                (satellite_id, neighbor) for neighbor in graph.neighbors(satellite_id)
            )

    isls_removed = 0
    for node_a, node_b in failed_edges:
        if not graph.has_edge(node_a, node_b):
            continue
        graph.remove_edge(node_a, node_b)
        if topology.sat_neighbor_to_if.pop((node_a, node_b), None) is not None:
            isls_removed += 1
        topology.sat_neighbor_to_if.pop((node_b, node_a), None)

    for gs_index, visible in enumerate(ground_station_satellites_in_range):
        ground_station_satellites_in_range[gs_index] = [
            (distance, satellite_id)
            for distance, satellite_id in visible
            if satellite_id not in failures.failed_satellites
        ]
    return {
        "isls_removed": float(isls_removed),
        "satellites_down": float(len(failures.failed_satellites)),
    }


def satellite_latitudes_deg(
    satellites: list[Satellite],
    epoch: object,
    time_absolute: object,
) -> dict[int, float]:
    """Sub-satellite latitude of every satellite at one snapshot, in degrees."""
    observer = ephem.Observer()
    observer.epoch = _to_clean_ephem_string(str(epoch))
    observer.date = _to_clean_ephem_string(str(time_absolute))
    latitudes: dict[int, float] = {}
    for satellite in satellites:
        body = satellite.position.ephem_obj_manual
        body.compute(observer)
        latitudes[satellite.id] = math.degrees(float(body.sublat))
    return latitudes
