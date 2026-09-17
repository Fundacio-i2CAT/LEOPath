"""Exception entries for topological forwarding under failures.

Topological forwarding derives next hops from addresses, with the progress guard
and local repair as its rules. Where those rules cannot deliver to a ground
station, a satellite needs explicit state: an exception entry naming the next hop
on the shortest live path toward the station's best visible egress, as in the
rule-and-exception forwarding policy of Leon Gaixas et al. (2017). Satellites are
assumed to learn failures by flooding only failed links and satellites over a
topology they already know, so the entries are computed here from the live graph.

Two placements are computed for every destination. The one-pass set gives an entry
to every live satellite whose rule-based walk fails. Exception hops strictly
shorten the live distance until the packet reaches a satellite whose rules
deliver, and that suffix uses rules only, so the set is loop-free by construction;
it also covers every satellite whose traffic drains toward a local minimum, which
makes it an upper bound. The grow set, which is the one installed, starts empty
and adds only the satellite where a walk breaks: the local minimum, or the first
satellite of a loop without an entry. It repeats until every walk from a reachable
live satellite delivers, and terminates because the set only grows and is bounded
by the reachable satellites.
"""

import heapq
from collections.abc import Iterable

import networkx as nx

EXCEPTION_POLICIES = ("none", "grow")

_DELIVER = "deliver"


def apply_exception_policy(
    fstate: dict,
    live_subgraph: nx.Graph,
    sat_neighbor_to_if: dict,
    nominal_graph: nx.Graph | None,
    ground_stations: list,
    ground_station_satellites_in_range: list,
    policy: str,
    detour_marker: str,
    state_report: dict | None = None,
) -> None:
    """Install exception entries where the rules cannot deliver, and report their cost."""
    if policy not in EXCEPTION_POLICIES:
        raise ValueError(f"Unknown exception policy: {policy}")
    if policy == "none":
        return

    context = _WalkContext(fstate, sat_neighbor_to_if, detour_marker)
    live_satellites = sorted(s for s in live_subgraph.nodes() if live_subgraph.degree(s) > 0)
    hops_to_failure = _hops_to_degraded_satellites(live_subgraph, nominal_graph)
    totals = {"entries": 0, "one_pass": 0, "unresolved": 0}
    satellites_with_entries: set[int] = set()
    groups: set[tuple] = set()
    distances: list[int] = []

    for gs_idx, ground_station in enumerate(ground_stations):
        visible_list = (
            ground_station_satellites_in_range[gs_idx]
            if gs_idx < len(ground_station_satellites_in_range)
            else []
        )
        if not visible_list:
            continue
        paths = _shortest_live_paths(live_subgraph, visible_list)
        reachable = [sat for sat in live_satellites if sat in paths.distance]
        visible = {sat for _distance, sat in visible_list}

        rules_memo: set[int] = set()
        totals["one_pass"] += sum(
            1
            for sat in reachable
            if not context.walk(sat, ground_station.id, visible, {}, rules_memo)[0]
        )
        exceptions, unresolved = _grow_exceptions(
            context, reachable, ground_station.id, visible, paths, sat_neighbor_to_if
        )
        totals["unresolved"] += unresolved

        for sat, entry in exceptions.items():
            fstate[(sat, ground_station.id)] = entry
            groups.add((sat, entry, paths.egress[sat]))
            if sat in hops_to_failure:
                distances.append(hops_to_failure[sat])
        totals["entries"] += len(exceptions)
        satellites_with_entries.update(exceptions)

    if state_report is not None:
        state_report.update(
            {
                "exception_entries": float(totals["entries"]),
                "exception_entries_one_pass": float(totals["one_pass"]),
                "exception_satellites": float(len(satellites_with_entries)),
                "exception_groups": float(len(groups)),
                "exception_unresolved": float(totals["unresolved"]),
                "exception_hops_to_failure_mean": (
                    float(sum(distances)) / len(distances) if distances else 0.0
                ),
                "exception_hops_to_failure_max": float(max(distances)) if distances else 0.0,
            }
        )


class _ShortestLivePaths:
    """Distance, next hop and egress toward one ground station, per satellite."""

    def __init__(self) -> None:
        self.distance: dict[int, float] = {}
        self.next_hop: dict[int, object] = {}
        self.egress: dict[int, int] = {}


def _shortest_live_paths(live_subgraph: nx.Graph, visible_list: Iterable) -> _ShortestLivePaths:
    """Dijkstra from every visible egress, each starting at its own GSL length."""
    paths = _ShortestLivePaths()
    for gsl_distance, sat in visible_list:
        if live_subgraph.has_node(sat) and gsl_distance < paths.distance.get(sat, float("inf")):
            paths.distance[sat] = float(gsl_distance)
            paths.next_hop[sat] = _DELIVER
            paths.egress[sat] = sat
    heap = [(distance, sat) for sat, distance in paths.distance.items()]
    heapq.heapify(heap)
    settled: set[int] = set()
    while heap:
        distance, sat = heapq.heappop(heap)
        if sat in settled:
            continue
        settled.add(sat)
        for neighbour in live_subgraph.neighbors(sat):
            candidate = distance + float(live_subgraph.edges[sat, neighbour].get("weight", 1.0))
            if candidate < paths.distance.get(neighbour, float("inf")):
                paths.distance[neighbour] = candidate
                paths.next_hop[neighbour] = sat
                paths.egress[neighbour] = paths.egress[sat]
                heapq.heappush(heap, (candidate, neighbour))
    return paths


class _WalkContext:
    """Follows forwarding state for one destination, with exceptions overlaid."""

    def __init__(self, fstate: dict, sat_neighbor_to_if: dict, detour_marker: str) -> None:
        self._fstate = fstate
        self._detour_marker = detour_marker
        self._neighbour_by_interface: dict[int, dict[int, int]] = {}
        for (sat, neighbour), interface in sat_neighbor_to_if.items():
            self._neighbour_by_interface.setdefault(sat, {})[interface] = neighbour

    def walk(
        self,
        start: int,
        gs_id: int,
        visible: set[int],
        exceptions: dict,
        delivering: set[int],
    ) -> tuple[bool, int | None]:
        """Whether a walk from ``start`` delivers, and where it breaks if it does not.

        ``delivering`` memoises satellites already known to deliver under the same
        exceptions; the walk adds its own path to it on success.
        """
        path: list[int] = []
        position: dict[int, int] = {}
        current = start
        while current not in delivering:
            if current in position:
                cycle = path[position[current] :]
                return False, next((sat for sat in cycle if sat not in exceptions), None)
            position[current] = len(path)
            path.append(current)
            entry = exceptions.get(current, self._fstate.get((current, gs_id)))
            hop = self._next(current, entry, gs_id)
            if hop == _DELIVER and current in visible:
                break
            if hop is None or hop == _DELIVER:
                return False, current
            current = hop
        delivering.update(path)
        return True, None

    def _next(self, sat: int, entry: object, gs_id: int):
        if isinstance(entry, int):
            return self._neighbour_by_interface.get(sat, {}).get(entry)
        if not isinstance(entry, tuple):
            return None
        if len(entry) == 2 and entry[0] == "GSL" and entry[1] == gs_id:
            return _DELIVER
        if len(entry) == 4 and entry[0] == self._detour_marker:
            return entry[3]
        return None


def _grow_exceptions(
    context: _WalkContext,
    reachable: list[int],
    gs_id: int,
    visible: set[int],
    paths: _ShortestLivePaths,
    sat_neighbor_to_if: dict,
) -> tuple[dict, int]:
    """Add entries only where walks break, until every reachable walk delivers."""
    exceptions: dict[int, object] = {}
    while True:
        delivering: set[int] = set()
        added = False
        unresolved = 0
        for sat in reachable:
            delivered, breaking = context.walk(sat, gs_id, visible, exceptions, delivering)
            if delivered:
                continue
            entry = _exception_entry(breaking, gs_id, paths, sat_neighbor_to_if)
            if breaking is None or entry is None or breaking in exceptions:
                unresolved += 1
                continue
            exceptions[breaking] = entry
            delivering.clear()
            added = True
        if not added:
            return exceptions, unresolved


def _exception_entry(
    sat: int | None, gs_id: int, paths: _ShortestLivePaths, sat_neighbor_to_if: dict
) -> object:
    """Forwarding entry toward the next hop on the shortest live path, if any."""
    if sat is None or sat not in paths.next_hop:
        return None
    next_hop = paths.next_hop[sat]
    if next_hop == _DELIVER:
        return ("GSL", gs_id)
    return sat_neighbor_to_if.get((sat, next_hop))


def _hops_to_degraded_satellites(
    live_subgraph: nx.Graph, nominal_graph: nx.Graph | None
) -> dict[int, int]:
    """Live hop distance from each satellite to the nearest one that lost an ISL."""
    if nominal_graph is None:
        return {}
    degraded = {
        sat
        for sat in live_subgraph.nodes()
        if live_subgraph.degree(sat) > 0
        and nominal_graph.has_node(sat)
        and sum(1 for n in nominal_graph.neighbors(sat) if live_subgraph.has_node(n))
        > live_subgraph.degree(sat)
    }
    if not degraded:
        return {}
    return dict(nx.multi_source_dijkstra_path_length(live_subgraph, degraded, weight=lambda *_: 1))
