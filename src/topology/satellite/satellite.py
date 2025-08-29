from typing import Optional, TYPE_CHECKING

import ephem

from src.topology.satellite.topological_network_address import TopologicalNetworkAddress

if TYPE_CHECKING:
    from src.topology.topology import LEOTopology


class SatelliteEphemeris:

    def __init__(self, ephem_obj_manual: ephem.Body, ephem_obj_direct: ephem.Body):
        """
        Class to hold the ephemeris data of a satellite.
        :param ephem_obj: Object representing the ephemeris data.
        :param ephem_obj_direct: Object representing the direct ephemeris data.
        """
        self.ephem_obj_manual = ephem_obj_manual
        self.ephem_obj_direct = ephem_obj_direct


class Satellite:
    """
    Class to a represent a satellite within a constellation.

    :param ephem_obj_manual: Object representing the manual ephemeris data.
    :param ephem_obj_direct: Object representing the direct ephemeris data.
    """

    def __init__(
        self,
        id: int,
        ephem_obj_manual: ephem.Body,
        ephem_obj_direct: ephem.Body,
        orbital_plane_id: Optional[int] = None,
        satellite_id: Optional[int] = None,
        sixgrupa_addr: Optional[TopologicalNetworkAddress] = None,
    ):
        """
        Class to represent a satellite within a constellation.
        :param id: Satellite ID
        :param ephem_obj_manual: Object representing the manual ephemeris data.
        :param ephem_obj_direct: Object representing the direct ephemeris data.
        :param 6grupa_addr: Optional address to be used in 6G-RUPA-based networks
        """
        self.position = SatelliteEphemeris(ephem_obj_manual, ephem_obj_direct)
        self.number_isls = 0
        self.number_gsls = 0
        self.id = id
        self.sixgrupa_addr = sixgrupa_addr
        self.orbital_plane_id = orbital_plane_id
        self.satellite_id = satellite_id
        self.forwarding_table: dict[int, int] = (
            {}
        )  # Maps 6G-RUPA (serialized) address  to interface number

    def get_6grupa_addr_from(
        self, neighbor_satellite_id: int
    ) -> Optional[TopologicalNetworkAddress]:
        """
        Get the 6G-RUPA address of a neighbor satellite.
        :param neighbor_satellite_id: Neighbor satellite ID
        :return: 6G-RUPA address of the neighbor satellite or None if not available
        """
        try:
            # Generate the topological address for the neighbor satellite
            return TopologicalNetworkAddress.set_address_from_orbital_parameters(neighbor_satellite_id)
        except Exception:
            # Log the error but don't crash the system
            return None

    def get_best_neighbor_for_destination(
        self, destination_address: TopologicalNetworkAddress, topology: "LEOTopology"
    ) -> Optional[int]:
        """
        Get the best neighbor satellite to route towards a destination using topological distance.

        Args:
            destination_address: The 6grupa address of the destination
            topology: The LEO topology containing neighbor information

        Returns:
            int: The satellite ID of the best neighbor, or None if no neighbors available
        """
        if not hasattr(self, 'sixgrupa_addr') or not self.sixgrupa_addr:
            return None

        my_address = self.sixgrupa_addr
        my_distance_to_dest = my_address.topological_distance_to(destination_address)

        best_neighbor_id = None
        best_distance = my_distance_to_dest

        # Get satellite subgraph to find neighbors
        try:
            all_satellite_ids = {sat.id for sat in topology.get_satellites()}
            satellite_node_ids = [node_id for node_id in topology.graph.nodes() if node_id in all_satellite_ids]
            satellite_only_subgraph = topology.graph.subgraph(satellite_node_ids)

            # Check all neighbors
            for neighbor_id in satellite_only_subgraph.neighbors(self.id):
                try:
                    neighbor_address = TopologicalNetworkAddress.set_address_from_orbital_parameters(neighbor_id)
                    neighbor_distance_to_dest = neighbor_address.topological_distance_to(destination_address)
                    
                    # If this neighbor is closer to destination, consider it
                    if neighbor_distance_to_dest < best_distance:
                        best_distance = neighbor_distance_to_dest
                        best_neighbor_id = neighbor_id
                        
                except Exception:
                    # Skip this neighbor if we can't get its address
                    continue
                    
        except Exception:
            # If we can't access topology, return None
            return None

        return best_neighbor_id
