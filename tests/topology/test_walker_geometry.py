import math

import pytest
from sgp4.api import WGS72, Satrec, jday

from leopath.topology.walker_geometry import WalkerShell, walker_shell_from_config

STARLINK_LIKE = WalkerShell(
    planes=72, sats_per_plane=22, inclination_deg=53.0, mean_motion_rev_per_day=15.05
)


def test_every_rail_has_the_same_length() -> None:
    rail = STARLINK_LIKE.rail_length_m()
    for plane, slot, time_s in [(0, 0, 0.0), (5, 3, 1234.0), (71, 21, 20000.0)]:
        neighbour = (plane, (slot + 1) % STARLINK_LIKE.sats_per_plane)
        assert STARLINK_LIKE.distance_m((plane, slot), neighbour, time_s) == pytest.approx(rail)


def test_rung_length_depends_on_row_and_plane_parity_not_on_plane() -> None:
    for row in (0, 7, 15):
        for time_s in (0.0, 3000.0):
            even = STARLINK_LIKE.rung_length_m(0, row, time_s)
            odd = STARLINK_LIKE.rung_length_m(1, row, time_s)
            for plane in range(0, 72, 2):
                assert STARLINK_LIKE.rung_length_m(plane, row, time_s) == pytest.approx(even)
            for plane in range(1, 72, 2):
                assert STARLINK_LIKE.rung_length_m(plane, row, time_s) == pytest.approx(odd)


def test_distance_is_symmetric() -> None:
    a, b = (3, 4), (10, 17)
    assert STARLINK_LIKE.distance_m(a, b, 500.0) == pytest.approx(
        STARLINK_LIKE.distance_m(b, a, 500.0)
    )


def test_position_follows_sgp4_to_within_short_period_terms() -> None:
    shell = STARLINK_LIKE
    plane, slot = 3, 5
    satellite = Satrec()
    mean_anomaly = math.radians(360.0 / shell.sats_per_plane) * slot + (
        math.pi / shell.sats_per_plane if plane % 2 == 1 else 0.0
    )
    satellite.sgp4init(
        WGS72,
        "i",
        1,
        18263.0,  # 2000-01-01 00:00, the epoch LEOPath writes into its TLEs
        0.0,
        0.0,
        0.0,
        1e-7,
        0.0,
        math.radians(shell.inclination_deg),
        mean_anomaly,
        shell.mean_motion_rev_per_day * 60.0 / 13750.9870831397,
        math.radians(plane * 360.0 / shell.planes),
    )
    jd, fr = jday(2000, 1, 1, 0, 0, 0)
    for time_s in (0.0, 3600.0, 6 * 3600.0):
        _error, position_km, _velocity = satellite.sgp4(jd, fr + time_s / 86400.0)
        derived = [c * shell.semi_major_axis_m for c in shell.unit_position(plane, slot, time_s)]
        error_m = math.dist(derived, [c * 1000.0 for c in position_km])
        # SGP4's periodic J2 terms, which a secular model leaves out, stay near 10 km.
        assert error_m < 20_000.0


def test_shell_constants_come_from_the_constellation_config() -> None:
    shell = walker_shell_from_config(
        {
            "num_orbits": 12,
            "num_sats_per_orbit": 49,
            "inclination_degree": 87.9,
            "mean_motion_rev_per_day": 13.16,
            "raan_spread_degree": 180.0,
            "phase_diff": True,
        }
    )
    assert (shell.planes, shell.sats_per_plane, shell.raan_spread_deg) == (12, 49, 180.0)
    assert 7_500_000 < shell.semi_major_axis_m < 7_600_000
