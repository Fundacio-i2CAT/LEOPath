"""Closed-form geometry of a Walker shell, from its constants and the clock.

A satellite running topological forwarding needs a distance estimate across
its whole shell, but it can only range its own links. Every other ISL length
has to come from somewhere. Measuring and flooding them would reintroduce the
topology state the design removes; this module derives them instead, from the
handful of constants a shell is built from (planes, satellites per plane,
inclination, mean motion, how far the ascending nodes are spread, and the
half-slot phasing of odd planes) plus the time since epoch.

The model is a circular orbit carried forward with SGP4's secular rates: the
mean motion recovered from the TLE's Kozai value, and the drift of the mean
anomaly, the argument of perigee and the node. SGP4's initialisation is itself
a closed-form function of the same constants, and every satellite of a shell
shares the same rates, so they are computed once per shell (in 6G-RUPA terms,
once at enrollment) and hold no per-satellite state. What the model ignores is
SGP4's periodic terms and any real deployment irregularity, which is exactly
what the comparison against SGP4 and against CelesTrak measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

from sgp4.api import WGS72, Satrec


@dataclass(frozen=True)
class _SecularRates:
    semi_major_axis_m: float
    mean_anomaly_rate: float  # rad/s
    perigee_rate: float  # rad/s
    node_rate: float  # rad/s


@dataclass(frozen=True)
class WalkerShell:
    """The constants that fix a shell's geometry at any time."""

    planes: int
    sats_per_plane: int
    inclination_deg: float
    mean_motion_rev_per_day: float
    raan_spread_deg: float = 360.0
    phase_diff: bool = True

    @cached_property
    def _rates(self) -> _SecularRates:
        """SGP4's secular rates for this shell, from one reference satellite.

        Every satellite of the shell has the same inclination and mean motion,
        so they all share these numbers.
        """
        reference = Satrec()
        reference.sgp4init(
            WGS72,
            "i",
            1,
            18262.5,  # 2000-01-01 12:00 in days since 1949-12-31; the rates don't depend on it
            0.0,
            0.0,
            0.0,
            1e-7,
            0.0,
            math.radians(self.inclination_deg),
            0.0,
            self.mean_motion_rev_per_day * 60.0 / 13750.9870831397,
            0.0,
        )
        per_minute = 1.0 / 60.0
        return _SecularRates(
            semi_major_axis_m=reference.a * reference.radiusearthkm * 1000.0,
            mean_anomaly_rate=reference.mdot * per_minute,
            perigee_rate=reference.argpdot * per_minute,
            node_rate=reference.nodedot * per_minute,
        )

    @property
    def semi_major_axis_m(self) -> float:
        return self._rates.semi_major_axis_m

    def node_rad(self, plane: int, time_s: float) -> float:
        return (
            math.radians(plane * self.raan_spread_deg / self.planes)
            + self._rates.node_rate * time_s
        )

    def argument_of_latitude_rad(self, plane: int, slot: int, time_s: float) -> float:
        rates = self._rates
        shift = math.pi / self.sats_per_plane if (self.phase_diff and plane % 2 == 1) else 0.0
        return (
            shift
            + 2.0 * math.pi * slot / self.sats_per_plane
            + (rates.mean_anomaly_rate + rates.perigee_rate) * time_s
        )

    def unit_position(self, plane: int, slot: int, time_s: float) -> tuple[float, float, float]:
        node = self.node_rad(plane, time_s)
        latitude = self.argument_of_latitude_rad(plane, slot, time_s)
        inclination = math.radians(self.inclination_deg)
        cos_n, sin_n = math.cos(node), math.sin(node)
        cos_u, sin_u = math.cos(latitude), math.sin(latitude)
        cos_i, sin_i = math.cos(inclination), math.sin(inclination)
        return (
            cos_n * cos_u - sin_n * sin_u * cos_i,
            sin_n * cos_u + cos_n * sin_u * cos_i,
            sin_u * sin_i,
        )

    def distance_m(self, a: tuple[int, int], b: tuple[int, int], time_s: float) -> float:
        """Chord between two satellites given as (plane, slot)."""
        ax, ay, az = self.unit_position(a[0], a[1], time_s)
        bx, by, bz = self.unit_position(b[0], b[1], time_s)
        separation = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
        return self.semi_major_axis_m * separation

    def rail_length_m(self) -> float:
        """Every in-plane link of a circular shell has this one length."""
        return 2.0 * self.semi_major_axis_m * math.sin(math.pi / self.sats_per_plane)

    def rung_length_m(self, plane: int, row: int, time_s: float) -> float:
        """Length of the cross-plane link from (plane, row) to (plane + 1, row)."""
        return self.distance_m((plane, row), ((plane + 1) % self.planes, row), time_s)


def walker_shell_from_config(constellation_config: dict) -> WalkerShell:
    """Build the shell constants from a LEOPath ``constellation`` config block."""
    return WalkerShell(
        planes=int(constellation_config["num_orbits"]),
        sats_per_plane=int(constellation_config["num_sats_per_orbit"]),
        inclination_deg=float(constellation_config["inclination_degree"]),
        mean_motion_rev_per_day=float(constellation_config["mean_motion_rev_per_day"]),
        raan_spread_deg=float(constellation_config.get("raan_spread_degree", 360.0)),
        phase_diff=bool(constellation_config.get("phase_diff", True)),
    )
