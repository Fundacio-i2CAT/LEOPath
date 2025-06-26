from src import logger
from src.topology.constellation import ConstellationData
from src.topology.satellite.topological_network_address import TopologicalNetworkAddress
from src.topology.topology import LEOTopology

log = logger.get_logger(__name__)


# def assign_6grupa_addresses_to_satellitest_in(topology: LEOTopology):
#     for satellite in topology.get_satellites():
#         # Assign 6G-RUPA addresses to each satellite
#         address = TopologicalNetworkAddress.from_6grupa(satellite.id)
#         satellite.set_network_address(address)
#         log.info(f"Assigned 6G-RUPA address {address} to satellite {satellite.id}")
