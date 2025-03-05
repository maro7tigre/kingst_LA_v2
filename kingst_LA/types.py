"""
Pythonic wrappers for Kingst Logic Analyzer basic types.

This module provides Python-friendly interfaces to the low-level C++ types
used in the Kingst Logic Analyzer SDK. It includes enums, classes, and utility
functions that make it easier to work with the SDK from Python code.
"""

from enum import Enum, IntEnum
from typing import Optional, Union, Tuple, List, Dict, Any, TypeVar, ClassVar, overload
import sys

# Import the underlying C++ bindings
from ._core import (
    # Enums
    BitState as _BitState,
    DisplayBase as _DisplayBase,
    
    # Classes
    Channel as _Channel,
    
    # Analyzer Enums
    AnalyzerEnums as _AnalyzerEnums,
    
    # Constants
    UNDEFINED_CHANNEL as _UNDEFINED_CHANNEL,
    
    # Functions
    toggle_bit as _toggle_bit,
    invert_bit as _invert_bit,
    
    # Integer boundaries
    S8_MIN, S8_MAX, U8_MAX,
    S16_MIN, S16_MAX, U16_MAX,
    S32_MIN, S32_MAX, U32_MAX,
    S64_MIN, S64_MAX, U64_MAX,
)

# Re-export the numeric limits
__all__ = [
    # Enums
    'BitState', 'DisplayBase',
    
    # Classes
    'Channel',
    
    # Analyzer Enums namespace
    'AnalyzerEnums',
    
    # Functions
    'toggle_bit', 'invert_bit',
    
    # Constants
    'UNDEFINED_CHANNEL',
    
    # Integer boundaries
    'S8_MIN', 'S8_MAX', 'U8_MAX',
    'S16_MIN', 'S16_MAX', 'U16_MAX',
    'S32_MIN', 'S32_MAX', 'U32_MAX',
    'S64_MIN', 'S64_MAX', 'U64_MAX',
]

# =============================================================================
# Enums
# =============================================================================

class BitState(IntEnum):
    """
    Represents the logical state of a digital signal.
    
    Attributes:
        LOW: Logic low (0)
        HIGH: Logic high (1)
    """
    LOW = _BitState.LOW
    HIGH = _BitState.HIGH
    
    def __str__(self) -> str:
        return "LOW" if self == BitState.LOW else "HIGH"
    
    def __repr__(self) -> str:
        return f"BitState.{self.name}"


class DisplayBase(IntEnum):
    """
    Controls how numbers are displayed in the UI.
    
    Attributes:
        Binary: Display as binary (e.g., "10110")
        Decimal: Display as decimal (e.g., "22")
        Hexadecimal: Display as hex (e.g., "0x16")
        ASCII: Display as ASCII characters
        AsciiHex: Display as ASCII with hex for non-printable characters
    """
    Binary = _DisplayBase.Binary
    Decimal = _DisplayBase.Decimal
    Hexadecimal = _DisplayBase.Hexadecimal
    ASCII = _DisplayBase.ASCII
    AsciiHex = _DisplayBase.AsciiHex
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"DisplayBase.{self.name}"


# =============================================================================
# Analyzer Enums Namespace
# =============================================================================

class AnalyzerEnums:
    """
    Enumerations for analyzer configuration.
    
    This class contains enumerations used to configure various aspects 
    of protocol analyzers.
    """
    
    class ShiftOrder(IntEnum):
        """
        Specifies the bit order for serial protocols.
        
        Attributes:
            MsbFirst: Most significant bit first (e.g., standard SPI)
            LsbFirst: Least significant bit first
        """
        MsbFirst = _AnalyzerEnums.ShiftOrder.MsbFirst
        LsbFirst = _AnalyzerEnums.ShiftOrder.LsbFirst
        
        def __str__(self) -> str:
            return "MSB First" if self == self.MsbFirst else "LSB First"
    
    class EdgeDirection(IntEnum):
        """
        Defines edge directions for triggering or sampling.
        
        Attributes:
            PosEdge: Rising edge (low to high transition)
            NegEdge: Falling edge (high to low transition)
        """
        PosEdge = _AnalyzerEnums.EdgeDirection.PosEdge
        NegEdge = _AnalyzerEnums.EdgeDirection.NegEdge
        
        def __str__(self) -> str:
            return "Rising Edge" if self == self.PosEdge else "Falling Edge"
    
    class Edge(IntEnum):
        """
        Defines specific edges for analysis.
        
        Attributes:
            LeadingEdge: First edge of a sequence
            TrailingEdge: Last edge of a sequence
        """
        LeadingEdge = _AnalyzerEnums.Edge.LeadingEdge
        TrailingEdge = _AnalyzerEnums.Edge.TrailingEdge
        
        def __str__(self) -> str:
            return "Leading Edge" if self == self.LeadingEdge else "Trailing Edge"
    
    class Parity(IntEnum):
        """
        Specifies parity settings for serial protocols.
        
        Attributes:
            None_: No parity bit
            Even: Even parity (bit ensures even number of 1s)
            Odd: Odd parity (bit ensures odd number of 1s)
        """
        None_ = getattr(_AnalyzerEnums.Parity, 'None')
        Even = _AnalyzerEnums.Parity.Even
        Odd = _AnalyzerEnums.Parity.Odd
        
        def __str__(self) -> str:
            if self == self.None_:
                return "No Parity"
            elif self == self.Even:
                return "Even Parity"
            else:
                return "Odd Parity"
    
    class Acknowledge(IntEnum):
        """
        Represents acknowledgment states in protocols.
        
        Attributes:
            Ack: Acknowledged (typically low for I2C)
            Nak: Not acknowledged (typically high for I2C)
        """
        Ack = _AnalyzerEnums.Acknowledge.Ack
        Nak = _AnalyzerEnums.Acknowledge.Nak
        
        def __str__(self) -> str:
            return "ACK" if self == self.Ack else "NAK"
    
    class Sign(IntEnum):
        """
        Specifies if values should be interpreted as signed or unsigned.
        
        Attributes:
            UnsignedInteger: Treat as unsigned value (all bits are magnitude)
            SignedInteger: Treat as signed value (MSB is sign bit)
        """
        UnsignedInteger = _AnalyzerEnums.Sign.UnsignedInteger
        SignedInteger = _AnalyzerEnums.Sign.SignedInteger
        
        def __str__(self) -> str:
            return "Unsigned" if self == self.UnsignedInteger else "Signed"


# =============================================================================
# Classes
# =============================================================================

class Channel:
    """
    Represents a physical channel on the logic analyzer.
    
    Each channel is identified by a device ID and a channel index.
    The device ID identifies the specific logic analyzer hardware,
    and the channel index identifies the specific channel on that device.
    """
    
    def __init__(self, device_id: int = 0, channel_index: int = 0):
        """
        Initialize a new Channel.
        
        Args:
            device_id: Identifier for the logic analyzer device
            channel_index: Channel number on the device (zero-based)
        """
        if isinstance(device_id, _Channel):
            # Copy constructor case
            self._channel = _Channel(device_id)
        else:
            # Regular constructor case
            self._channel = _Channel(device_id, channel_index)
    
    @property
    def device_id(self) -> int:
        """
        Identifier for the logic analyzer device.
        
        Each physical logic analyzer has a unique device ID.
        """
        return self._channel.device_id
    
    @device_id.setter
    def device_id(self, value: int) -> None:
        self._channel.device_id = value
    
    @property
    def channel_index(self) -> int:
        """
        Channel number on the device (zero-based).
        
        Identifies which specific channel on the device this
        Channel object represents.
        """
        return self._channel.channel_index
    
    @channel_index.setter
    def channel_index(self, value: int) -> None:
        self._channel.channel_index = value
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Channel):
            return NotImplemented
        return self._channel == other._channel
    
    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, Channel):
            return NotImplemented
        return self._channel != other._channel
    
    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Channel):
            return NotImplemented
        return self._channel < other._channel
    
    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Channel):
            return NotImplemented
        return self._channel > other._channel
    
    def __repr__(self) -> str:
        return f"Channel(device_id={self.device_id}, channel_index={self.channel_index})"
    
    def __str__(self) -> str:
        return f"Channel {self.channel_index} on Device {self.device_id}"
    
    def __hash__(self) -> int:
        return hash((self.device_id, self.channel_index))
    
    # Access to the underlying C++ object
    @property
    def _internal(self) -> _Channel:
        """Get the internal C++ Channel object."""
        return self._channel


# =============================================================================
# Constants
# =============================================================================

# Undefined channel constant
UNDEFINED_CHANNEL = Channel(_UNDEFINED_CHANNEL.device_id, _UNDEFINED_CHANNEL.channel_index)


# =============================================================================
# Utility Functions
# =============================================================================

def toggle_bit(bit: Union[BitState, int]) -> BitState:
    """
    Toggle a bit state (equivalent to C++ Toggle macro).
    
    Args:
        bit: The bit state to toggle
    
    Returns:
        The toggled bit state (LOW->HIGH or HIGH->LOW)
    """
    if isinstance(bit, int):
        bit = BitState(bit)
    return BitState(_toggle_bit(bit))


def invert_bit(bit: Union[BitState, int]) -> BitState:
    """
    Invert a bit state (equivalent to C++ Invert macro).
    
    Args:
        bit: The bit state to invert
    
    Returns:
        The inverted bit state (LOW->HIGH or HIGH->LOW)
    """
    if isinstance(bit, int):
        bit = BitState(bit)
    return BitState(_invert_bit(bit))