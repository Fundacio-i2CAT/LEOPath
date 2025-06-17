from src import logger

log = logger.get_logger(__name__)


def setup_isls_in_the_same_orbit(num_orbits: int, sats_per_orbit: int, idx_offset: int = 0):
    """
    Returns a list of undirected ISLs for satellites in the same orbit.
    Each orbit is a ring of satellites connected in a closed loop.

    :param num_orbits: Number of orbits
    :param sats_per_orbit: Number of satellites per orbit
    :param idx_offset: Index offset to apply to satellite IDs
    """
    undirected_isls = []
    for orbit in range(num_orbits):
        for sat in range(sats_per_orbit):
            # Connect to next satellite in the same orbit (also wraps around)
            sat_id = idx_offset + orbit * sats_per_orbit + sat
            next_sat_id = idx_offset + orbit * sats_per_orbit + ((sat + 1) % sats_per_orbit)
            undirected_isls.append((sat_id, next_sat_id))

    log.info(
        f"Created {len(undirected_isls)} intra-orbit ISLs (rings) for {num_orbits} orbits; undirected_isls={undirected_isls}"
    )
    return undirected_isls


def generate_plus_grid_isls(n_orbits, n_sats_per_orbit, isl_shift=0, idx_offset=0):
    """
    Generate plus grid ISL file.

    :param n_orbits: Number of orbits
    :param n_sats_per_orbit: Number of satellites per orbit
    :param isl_shift: ISL shift between orbits
    :param idx_offset: Index offset (e.g., if you have multiple shells)
    """
    if n_orbits < 3 or n_sats_per_orbit < 3:
        log.warning("Need at least 3 orbits and 3 satellites per orbit for plus grid ISLs")
        return []

    # Pass idx_offset to setup_isls_in_the_same_orbit
    list_isls = setup_isls_in_the_same_orbit(n_orbits, n_sats_per_orbit, idx_offset)

    # Add ISLs between orbits (adjacent orbits)
    for i in range(n_orbits):
        next_orbit = (i + 1) % n_orbits
        for j in range(n_sats_per_orbit):
            # Connect satellite j in orbit i to satellite j+isl_shift in adjacent orbit
            sat_id = idx_offset + i * n_sats_per_orbit + j
            adj_sat_id = (
                idx_offset + next_orbit * n_sats_per_orbit + ((j + isl_shift) % n_sats_per_orbit)
            )
            list_isls.append((sat_id, adj_sat_id))

    log.info(f"Created {len(list_isls)} ISLs (plus grid); undirected_isls='{list_isls}'")
    return list_isls
