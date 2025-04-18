def validate_no_satellite_to_gs_links(graph, satellites, ground_stations):
    """
    Validates that there are no edges between satellites and ground stations in the graph.

    :param graph: The NetworkX graph representing the topology.
    :param satellites: List of Satellite objects.
    :param ground_stations: List of GroundStation objects.
    :raises ValueError: If a satellite is connected to a ground station.
    """
    satellite_ids = {sat.id for sat in satellites}
    ground_station_ids = {gs.id for gs in ground_stations}

    for u, v in graph.edges:
        if (u in satellite_ids and v in ground_station_ids) or (
            u in ground_station_ids and v in satellite_ids
        ):
            raise ValueError(f"Invalid edge between satellite {u} and ground station {v}")
