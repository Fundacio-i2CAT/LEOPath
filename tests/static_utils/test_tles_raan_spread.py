from pathlib import Path

import pytest

from leopath.tles.generate_tles_from_scratch import generate_tles_from_scratch_with_sgp


def _plane_raans(path: Path, orbits: int, sats: int) -> list[float]:
    lines = path.read_text().splitlines()
    # Header line, then name / line 1 / line 2 per satellite. RAAN is columns 18-25 of line 2.
    line2 = [lines[1 + 3 * i + 2] for i in range(orbits * sats)]
    return [float(line2[p * sats][17:25]) for p in range(orbits)]


def _generate(path: Path, orbits: int, sats: int, **kwargs) -> None:
    generate_tles_from_scratch_with_sgp(
        str(path), "test", orbits, sats, True, 87.9, 0.0000001, 0.0, 13.16, **kwargs
    )


def test_default_spreads_nodes_over_a_full_circle(tmp_path: Path) -> None:
    out = tmp_path / "delta.txt"
    _generate(out, 12, 4)
    assert _plane_raans(out, 12, 4) == pytest.approx([30.0 * p for p in range(12)], abs=1e-3)


def test_walker_star_spreads_nodes_over_half_a_circle(tmp_path: Path) -> None:
    out = tmp_path / "star.txt"
    _generate(out, 12, 4, raan_spread_degree=180.0)
    assert _plane_raans(out, 12, 4) == pytest.approx([15.0 * p for p in range(12)], abs=1e-3)
