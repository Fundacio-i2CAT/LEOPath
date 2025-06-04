from dataclasses import dataclass

from src import logger

log = logger.get_logger(__name__)

# ==============================================================================
# Topological Address Component Limits and Bit Allocation
# ==============================================================================
# These constants define the assumed maximum range for each component of the
# topological address (sh, o, s, x). They are crucial for determining the
# number of bits required for each component in the bit-packing serialization
# (`to_integer` method).
#
# Reasoning for estimates is based on public info for LEO constellations
# (Starlink, Kuiper, OneWeb, etc.) as of early 2025, plus headroom.
# ------------------------------------------------------------------------------

# Maximum number of distinct orbital shells (sh).
# Starlink/Kuiper plan multiple shells. Allowing 0-15 seems generous.
MAX_SHELLS = 16  # Max index = 15. Requires 4 bits.

# Maximum number of orbital planes within any single shell (o).
# Starlink Gen1 has 72 planes/shell. Allowing 0-127 covers this with room.
MAX_PLANES = 128  # Max index = 127. Requires 7 bits.

# Maximum number of satellites within any single plane (s).
# Current constellations range from ~20 to ~40. Allowing 0-63 provides headroom.
MAX_SATS_PER_PLANE = 64  # Max index = 63. Requires 6 bits.

# Maximum number of Ground Stations simultaneously associated with (homed to)
# a single satellite's sub-network (x > 0). Index x=0 is reserved for the satellite itself.
# This depends on satellite antenna capability and network design.
# Assuming up to 31 GSs can be addressed under one satellite seems reasonable,
# providing flexibility without demanding excessive bits.
MAX_GS_PER_SAT_SUBNET = 31

# Total number of unique endpoints addressable under one satellite's sh,o,s prefix.
# Includes the satellite (index 0) + the max number of associated GSs.
MAX_ENDPOINTS_PER_SAT = 1 + MAX_GS_PER_SAT_SUBNET  # e.g., 1 + 31 = 32

# ------------------------------------------------------------------------------
# Calculate bits required for each component based on the MAX values above.
# Using (MAX-1).bit_length() handles cases where MAX is not a power of 2.
# If MAX is 1 (e.g., only 1 shell), we still allocate 1 bit.
# ------------------------------------------------------------------------------
SHELL_BITS = (MAX_SHELLS - 1).bit_length() if MAX_SHELLS > 1 else 1  # e.g., 4 bits
PLANE_BITS = (MAX_PLANES - 1).bit_length() if MAX_PLANES > 1 else 1  # e.g., 7 bits
SAT_IDX_BITS = (
    (MAX_SATS_PER_PLANE - 1).bit_length() if MAX_SATS_PER_PLANE > 1 else 1
)  # e.g., 6 bits
# Bits for subnet_index (0 to MAX_GS_PER_SAT_SUBNET inclusive)
SUBNET_IDX_BITS = (
    (MAX_ENDPOINTS_PER_SAT - 1).bit_length() if MAX_ENDPOINTS_PER_SAT > 1 else 1
)  # e.g., 5 bits for 0..31

# ------------------------------------------------------------------------------
# Verify total bits fit within a standard 64-bit integer.
# ------------------------------------------------------------------------------
TOTAL_BITS = SHELL_BITS + PLANE_BITS + SAT_IDX_BITS + SUBNET_IDX_BITS
log.debug(
    f"Address bit allocation: Shell={SHELL_BITS}, Plane={PLANE_BITS}, SatIdx={SAT_IDX_BITS}, SubnetIdx={SUBNET_IDX_BITS}. Total={TOTAL_BITS}"
)
if TOTAL_BITS > 64:
    # If this occurs, consider reducing MAX values, using fewer components,
    # or switching serialization (e.g., storing as a tuple/string and hashing).
    raise ValueError(f"Total bits required ({TOTAL_BITS}) exceeds 64 bits for address components")

# ------------------------------------------------------------------------------
# Define bit masks and shifts for packing/unpacking based on the calculated bits.
# Packing order (MSB to LSB): SHELL | PLANE | SAT_IDX | SUBNET_IDX
# ------------------------------------------------------------------------------
SUBNET_IDX_MASK = (1 << SUBNET_IDX_BITS) - 1
SAT_IDX_MASK = (1 << SAT_IDX_BITS) - 1
PLANE_MASK = (1 << PLANE_BITS) - 1
SHELL_MASK = (1 << SHELL_BITS) - 1

# Shifts are determined by the number of bits to the right of the component
SAT_IDX_SHIFT = SUBNET_IDX_BITS
PLANE_SHIFT = SUBNET_IDX_BITS + SAT_IDX_BITS
SHELL_SHIFT = SUBNET_IDX_BITS + SAT_IDX_BITS + PLANE_BITS
# ==============================================================================


@dataclass(frozen=True)
class TopologicalNetworkAddress:
    """
    Represents a topological network address based on (shell, plane, sat_idx, subnet_idx).
    Uses constants defined above for validation and serialization.

    Attributes:
        shell_id: Shell index (sh).
        plane_id: Orbital plane index (o).
        sat_index: Satellite index within plane (s).
        subnet_index: Endpoint index within the satellite's sub-network (x).
                      0 indicates the satellite itself.
                      > 0 indicates a ground station homed to this satellite.
    """

    shell_id: int
    plane_id: int
    sat_index: int
    subnet_index: int

    def __post_init__(self):
        if not (0 <= self.shell_id < MAX_SHELLS):
            raise ValueError(f"shell_id {self.shell_id} out of range [0, {MAX_SHELLS - 1}]")
        if not (0 <= self.plane_id < MAX_PLANES):
            raise ValueError(f"plane_id {self.plane_id} out of range [0, {MAX_PLANES - 1}]")
        if self.subnet_index == 0 and not (0 <= self.sat_index < MAX_SATS_PER_PLANE):
            raise ValueError(
                f"sat_index {self.sat_index} out of range [0, {MAX_SATS_PER_PLANE - 1}] for satellite address (subnet_index 0)"
            )
        if not (0 <= self.subnet_index < MAX_ENDPOINTS_PER_SAT):
            raise ValueError(
                f"subnet_index {self.subnet_index} out of range [0, {MAX_ENDPOINTS_PER_SAT - 1}]"
            )

    @property
    def is_satellite(self) -> bool:
        return self.subnet_index == 0

    @property
    def is_ground_station(self) -> bool:
        return self.subnet_index > 0

    def get_satellite_address(self) -> "TopologicalNetworkAddress":
        if self.is_satellite:
            return self
        else:
            return TopologicalNetworkAddress(self.shell_id, self.plane_id, self.sat_index, 0)

    def to_integer(self) -> int:
        packed_address = 0
        packed_address |= (self.shell_id & SHELL_MASK) << SHELL_SHIFT
        packed_address |= (self.plane_id & PLANE_MASK) << PLANE_SHIFT
        packed_address |= (self.sat_index & SAT_IDX_MASK) << SAT_IDX_SHIFT
        packed_address |= self.subnet_index & SUBNET_IDX_MASK
        return packed_address

    @staticmethod
    def from_integer(packed_address: int) -> "TopologicalNetworkAddress":
        if not isinstance(packed_address, int) or packed_address < 0:
            raise ValueError("Packed address must be a non-negative integer")

        shell_id = (packed_address >> SHELL_SHIFT) & SHELL_MASK
        plane_id = (packed_address >> PLANE_SHIFT) & PLANE_MASK
        sat_index = (packed_address >> SAT_IDX_SHIFT) & SAT_IDX_MASK
        subnet_index = packed_address & SUBNET_IDX_MASK
        try:
            return TopologicalNetworkAddress(
                shell_id=shell_id, plane_id=plane_id, sat_index=sat_index, subnet_index=subnet_index
            )
        except ValueError as e:
            log.error(
                f"Failed to create address from integer {packed_address}: {e}. Decoded components: sh={shell_id}, o={plane_id}, s={sat_index}, x={subnet_index}"
            )
            raise

    def __str__(self) -> str:
        kind = "Sat" if self.is_satellite else f"GS[{self.subnet_index}]"
        return f"TopoAddr(sh:{self.shell_id}, o:{self.plane_id}, s:{self.sat_index}, x:{kind})"
