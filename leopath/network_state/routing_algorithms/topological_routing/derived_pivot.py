"""Pivot distance computed from a shell's constants, with no tables at all.

The tabled estimator in ``fstate_calculation`` precomputes every row-to-row and
plane-to-plane path cost, P*S^2 + S*P^2 entries per shell. For a shell laid out
the way LEOPath lays it out, those tables are pure cache:

- every rail (in-plane link) of a circular shell has the same length;
- a rung (cross-plane link) joins (p, r) to (p + 1, r). Rotating the shell
  about the pole doesn't change its length, and the only thing that varies
  with p is the half-slot shift of odd planes, so its length depends on the
  row r, the parity of p and the time, never on p itself. The one exception
  is the wrap rung from the last plane back to plane 0 when the plane count is
  odd, where both ends are unshifted.

So a pivot query needs, per pivot row, one rail length times a ring distance
plus two rung lengths times how many even and odd rungs the crossing uses.
That is O(S) arithmetic per query, from seven constants and the clock.
``DerivedPivotEstimator.distance`` must equal the tabled estimator over the
same derived geometry; the test suite checks that it does.
"""

from __future__ import annotations

import math

from leopath.topology.walker_geometry import WalkerShell


class DerivedPivotEstimator:
    """Table-free pivot distance on a +Grid Walker shell at one instant."""

    def __init__(self, shell: WalkerShell, time_s: float, cross_plane_wrap: bool | None = None):
        self.shell = shell
        self.time_s = time_s
        # A Walker star (nodes over 180 degrees) has no rung from the last plane
        # back to the first: those planes counter-rotate.
        self.cross_plane_wrap = (
            shell.raan_spread_deg >= 360.0 if cross_plane_wrap is None else cross_plane_wrap
        )
        self._rail = shell.rail_length_m()
        # Per row: rung length leaving an even plane, leaving an odd plane, and
        # the wrap rung. Computed lazily; a satellite would evaluate the closed
        # form for the rows its query touches.
        self._rungs: dict[int, tuple[float, float, float]] = {}

    def _rung_lengths(self, row: int) -> tuple[float, float, float]:
        cached = self._rungs.get(row)
        if cached is None:
            planes = self.shell.planes
            even = self.shell.rung_length_m(0, row, self.time_s)
            odd = self.shell.rung_length_m(1, row, self.time_s) if planes > 2 else even
            wrap = self.shell.rung_length_m(planes - 1, row, self.time_s)
            cached = (even, odd, wrap)
            self._rungs[row] = cached
        return cached

    def _row_cost(self, source_slot: int, destination_slot: int) -> float:
        slots = self.shell.sats_per_plane
        steps = (destination_slot - source_slot) % slots
        return self._rail * min(steps, slots - steps)

    def _plane_cost(self, row: int, source_plane: int, destination_plane: int) -> float:
        if source_plane == destination_plane:
            return 0.0
        planes = self.shell.planes
        forward = self._crossing_cost(
            row, source_plane, (destination_plane - source_plane) % planes
        )
        backward = self._crossing_cost(
            row, destination_plane, (source_plane - destination_plane) % planes
        )
        return min(forward, backward)

    def _crossing_cost(self, row: int, start_plane: int, steps: int) -> float:
        """Cost of the rungs from start_plane up to start_plane + steps along a row."""
        planes = self.shell.planes
        end_plane = start_plane + steps  # exclusive, may run past the last plane
        # The rung leaving plane P - 1 is the wrap back to plane 0.
        if end_plane >= planes and not self.cross_plane_wrap:
            return math.inf
        lengths = self._rung_lengths(row)
        total = self._parity_sum(start_plane, min(end_plane, planes) - start_plane, lengths)
        if end_plane > planes:
            total += self._parity_sum(0, end_plane - planes, lengths)
        return total

    def _parity_sum(
        self, first_plane: int, count: int, lengths: tuple[float, float, float]
    ) -> float:
        """Rungs leaving planes first_plane .. first_plane + count - 1, none past P - 1."""
        if count <= 0:
            return 0.0
        even, odd, wrap = lengths
        # With an odd plane count the wrap rung joins two unshifted planes, so it
        # differs from every other rung leaving an even plane.
        wrap_included = self.shell.planes % 2 == 1 and first_plane + count == self.shell.planes
        regular = count - (1 if wrap_included else 0)
        evens = (regular + (1 if first_plane % 2 == 0 else 0)) // 2
        odds = regular - evens
        return evens * even + odds * odd + (wrap if wrap_included else 0.0)

    def distance(self, source: tuple[int, int], destination: tuple[int, int]) -> float:
        """Pivot distance between two satellites given as (plane, slot)."""
        if source == destination:
            return 0.0
        best = math.inf
        for pivot_row in range(self.shell.sats_per_plane):
            candidate = (
                self._row_cost(source[1], pivot_row)
                + self._plane_cost(pivot_row, source[0], destination[0])
                + self._row_cost(pivot_row, destination[1])
            )
            best = min(best, candidate)
        return best
