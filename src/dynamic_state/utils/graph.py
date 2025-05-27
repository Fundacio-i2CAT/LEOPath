import math

from src import logger
from src.topology.topology import LEOTopology

log = logger.get_logger(__name__)


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


def _topologies_are_equal(
    prev_topo: LEOTopology | None,  # Puede ser None en el primer paso
    curr_topo: LEOTopology,
    weight_tolerance: float = 1e-6,  # Tolerancia para comparar pesos (distancias en metros)
) -> bool:
    """
    Compara dos topologías (LEOTopology) para determinar si son
    estructuralmente iguales (nodos, enlaces y pesos de enlaces dentro de una tolerancia).

    Args:
        prev_topo: La instancia LEOTopology del paso de tiempo anterior (o None).
        curr_topo: La instancia LEOTopology del paso de tiempo actual.
        weight_tolerance: Tolerancia absoluta (en metros) para comparar pesos/distancias.

    Returns:
        True si las topologías se consideran iguales, False en caso contrario.
    """
    # 1. Manejar el caso del primer paso de tiempo (no hay topología previa)
    if prev_topo is None:
        log.debug("_topologies_are_equal: No previous topology, returning False.")
        return False  # No puede ser igual si no había una anterior

    # 2. Obtener los grafos de NetworkX
    graph_prev = prev_topo.graph
    graph_curr = curr_topo.graph

    # 3. Comparar conjuntos de nodos
    nodes_prev = set(graph_prev.nodes())
    nodes_curr = set(graph_curr.nodes())
    if nodes_prev != nodes_curr:
        log.debug(
            f"_topologies_are_equal: Node sets differ. Prev: {len(nodes_prev)}, Curr: {len(nodes_curr)}. Returning False."
        )
        # Para más detalle (opcional):
        # log.debug(f"Nodes added: {nodes_curr - nodes_prev}")
        # log.debug(f"Nodes removed: {nodes_prev - nodes_curr}")
        return False

    # 4. Comparar conjuntos de enlaces (sin considerar datos/pesos aún)
    # Compara tuplas (u, v) independientemente del orden interno si el grafo es no dirigido
    # NetworkX maneja esto internamente al obtener .edges()
    edges_prev = set(graph_prev.edges())
    edges_curr = set(graph_curr.edges())
    if edges_prev != edges_curr:
        log.debug(
            f"_topologies_are_equal: Edge sets differ. Prev: {len(edges_prev)}, Curr: {len(edges_curr)}. Returning False."
        )
        # Para más detalle (opcional):
        # log.debug(f"Edges added: {edges_curr - edges_prev}")
        # log.debug(f"Edges removed: {edges_prev - edges_curr}")
        return False

    # 5. Comparar pesos de los enlaces (si nodos y enlaces coinciden)
    # Iteramos sobre los enlaces actuales (sabemos que son los mismos que los anteriores)
    for u, v in edges_curr:
        try:
            # Obtener pesos, usando .get() para manejar posible ausencia (aunque no debería ocurrir aquí)
            weight_curr = graph_curr.get_edge_data(u, v, default={}).get("weight")
            weight_prev = graph_prev.get_edge_data(u, v, default={}).get("weight")

            # Si algún peso falta (inesperado si los edges son iguales, pero por seguridad)
            if weight_curr is None or weight_prev is None:
                if weight_curr is weight_prev:  # Ambos son None, considerarlo igual
                    continue
                else:  # Uno tiene peso y el otro no, considerarlo diferente
                    log.debug(
                        f"_topologies_are_equal: Weight missing for edge ({u},{v}). Prev: {weight_prev}, Curr: {weight_curr}. Returning False."
                    )
                    return False

            # Comparar pesos usando math.isclose con tolerancia absoluta
            if not math.isclose(weight_curr, weight_prev, abs_tol=weight_tolerance):
                log.debug(
                    f"_topologies_are_equal: Weight differs for edge ({u},{v}). Prev: {weight_prev:.4f}, Curr: {weight_curr:.4f}. Returning False."
                )
                return False
        except KeyError:
            # Esto no debería ocurrir si los conjuntos de enlaces son iguales, pero por si acaso
            log.error(
                f"_topologies_are_equal: Error accessing edge data for ({u},{v}) despite equal edge sets. Returning False."
            )
            return False

    # 6. Si todas las comprobaciones pasan, las topologías son iguales
    log.debug("_topologies_are_equal: Topologies are equal. Returning True.")
    return True
