"""
Pythonic wrappers for Kingst Logic Analyzer basic types.

This module provides Python-friendly interfaces to the low-level C++ types
used in the Kingst Logic Analyzer SDK. It includes enums, classes, and utility
functions that make it easier to work with the SDK from Python code.

The module bridges the gap between the C++ implementation and Python users,
offering type safety, comprehensive documentation, and idiomatic Python patterns
while maintaining full compatibility with the underlying SDK.
"""

from enum import Enum, IntEnum
from typing import Optional, Union, Tuple, List, Dict, Any, TypeVar, ClassVar, overload, cast
import warnings
import sys
from dataclasses import dataclass

# Import the underlying C++ bindings - would reference the compiled pyd module
try:
    from kingst_analyzer._core import (
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
except ImportError as e:
    # Provide stubs for documentation and development without the actual bindings
    # This enables IDE completion and documentation generation
    warnings.warn(f"Could not import C++ bindings (_core module): {e}. Using stub classes.")
    
    class _BitState(Enum):
        LOW = 0
        HIGH = 1
    
    class _DisplayBase(Enum):
        Binary = 0
        Decimal = 1
        Hexadecimal = 2
        ASCII = 3
        AsciiHex = 4
    
    class _Channel:
        def __init__(self, *args):
            self.device_id = 0 if len(args) == 0 else args[0]
            self.channel_index = 0 if len(args) < 2 else args[1]
    
    # Stub class for AnalyzerEnums
    class _AnalyzerEnums:
        class ShiftOrder(Enum):
            MsbFirst = 0
            LsbFirst = 1
        
        class EdgeDirection(Enum):
            PosEdge = 0
            NegEdge = 1
        
        class Edge(Enum):
            LeadingEdge = 0
            TrailingEdge = 1
        
        class Parity(IntEnum):
            None_ = int(getattr(_AnalyzerEnums.Parity, 'None'))  # Use getattr to access 'None' as a string
            Even = int(_AnalyzerEnums.Parity.Even)
            Odd = int(_AnalyzerEnums.Parity.Odd)
        
        class Acknowledge(Enum):
            Ack = 0
            Nak = 1
        
        class Sign(Enum):
            UnsignedInteger = 0
            SignedInteger = 1
    
    # Stub constants
    _UNDEFINED_CHANNEL = _Channel(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF)
    
    # Stub functions
    def _toggle_bit(bit): 
        return _BitState.HIGH if bit == _BitState.LOW else _BitState.LOW
    
    def _invert_bit(bit):
        return _BitState.HIGH if bit == _BitState.LOW else _BitState.LOW
    
    # Integer boundaries
    S8_MIN, S8_MAX, U8_MAX = -128, 127, 255
    S16_MIN, S16_MAX, U16_MAX = -32768, 32767, 65535
    S32_MIN, S32_MAX, U32_MAX = -2147483648, 2147483647, 4294967295
    S64_MIN, S64_MAX, U64_MAX = -9223372036854775808, 9223372036854775807, 18446744073709551615

# Define exported names
__all__ = [
    # Enums
    'BitState', 'DisplayBase',
    
    # Classes
    'Channel',
    
    # Analyzer Enums
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
    
    This enum provides a type-safe way to represent binary states in digital signals.
    It offers convenient string representations and compatibility with both numeric
    and symbolic comparisons.
    
    Attributes:
        LOW: Logic low (0)
        HIGH: Logic high (1)
    
    Example:
        >>> state = BitState.HIGH
        >>> if state == BitState.HIGH:
        ...     print("Signal is HIGH")
        >>> # Numeric comparison also works
        >>> state == 1  # True
    """
    LOW = int(_BitState.LOW)
    HIGH = int(_BitState.HIGH)
    
    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return "LOW" if self == BitState.LOW else "HIGH"
    
    def __repr__(self) -> str:
        """Return a code-like representation for debugging."""
        return f"BitState.{self.name}"


class DisplayBase(IntEnum):
    """
    Controls how numeric values are displayed in the UI and exported data.
    
    This enum defines the various number formats available for displaying
    protocol analyzer results, frame data, and other numeric values.
    
    Attributes:
        Binary: Display as binary (e.g., "10110")
        Decimal: Display as decimal (e.g., "22")
        Hexadecimal: Display as hex (e.g., "0x16")
        ASCII: Display as ASCII characters
        AsciiHex: Display as ASCII with hex for non-printable characters
    
    Example:
        >>> # Configure an analyzer to show results in hexadecimal
        >>> analyzer_settings.set_number_format(DisplayBase.Hexadecimal)
    """
    Binary = int(_DisplayBase.Binary)
    Decimal = int(_DisplayBase.Decimal)
    Hexadecimal = int(_DisplayBase.Hexadecimal)
    ASCII = int(_DisplayBase.ASCII)
    AsciiHex = int(_DisplayBase.AsciiHex)
    
    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return self.name
    
    def __repr__(self) -> str:
        """Return a code-like representation for debugging."""
        return f"DisplayBase.{self.name}"
    
    def format_value(self, value: int, num_bits: int = 8) -> str:
        """
        Format a numeric value according to this display base.
        
        Args:
            value: Integer value to format
            num_bits: Number of bits to consider (for Binary/Hex padding)
            
        Returns:
            Formatted string representation
            
        Example:
            >>> DisplayBase.Binary.format_value(10, num_bits=8)
            '00001010'
            >>> DisplayBase.Hexadecimal.format_value(255)
            '0xFF'
        """
        if self == DisplayBase.Binary:
            return format(value, f'0{num_bits}b')
        elif self == DisplayBase.Decimal:
            return str(value)
        elif self == DisplayBase.Hexadecimal:
            hex_digits = (num_bits + 3) // 4  # Round up to nearest multiple of 4
            return f"0x{value:0{hex_digits}X}"
        elif self == DisplayBase.ASCII:
            if 32 <= value <= 126:  # Printable ASCII
                return chr(value)
            return f"."  # Non-printable
        elif self == DisplayBase.AsciiHex:
            if 32 <= value <= 126:  # Printable ASCII
                return chr(value)
            return f"<{value:02X}>"  # Non-printable as hex
        
        # Should never reach here
        return str(value)


# =============================================================================
# Analyzer Enums Namespace
# =============================================================================

class AnalyzerEnums:
    """
    Enumerations for analyzer configuration and result interpretation.
    
    This class encapsulates various enumeration types used to configure protocol
    analyzers and to interpret analyzer results. Each nested enum class represents
    a different aspect of analyzer configuration.
    
    The nested enums follow the C++ AnalyzerEnums namespace structure, but provide
    a more Pythonic interface with better documentation, type safety, and enhanced
    functionality.
    """
    
    class ShiftOrder(IntEnum):
        """
        Specifies the bit order for serial protocols.
        
        Many serial protocols allow configuration of whether data is transmitted
        with the most significant bit first or the least significant bit first.
        This setting is critical for correctly interpreting the data.
        
        Attributes:
            MsbFirst: Most significant bit first (e.g., standard SPI, I2C)
            LsbFirst: Least significant bit first (e.g., some variants of SPI)
        
        Example:
            >>> # Configure a SPI analyzer to use LSB first
            >>> spi_settings.set_shift_order(AnalyzerEnums.ShiftOrder.LsbFirst)
        """
        MsbFirst = int(_AnalyzerEnums.ShiftOrder.MsbFirst)
        LsbFirst = int(_AnalyzerEnums.ShiftOrder.LsbFirst)
        
        def __str__(self) -> str:
            """Return a human-readable string representation."""
            return "MSB First" if self == self.MsbFirst else "LSB First"
        
        def __repr__(self) -> str:
            """Return a code-like representation for debugging."""
            return f"AnalyzerEnums.ShiftOrder.{self.name}"
    
    class EdgeDirection(IntEnum):
        """
        Defines edge directions for triggering or sampling.
        
        Edge direction is crucial for many protocols where data is sampled
        on either the rising or falling edge of a clock signal.
        
        Attributes:
            PosEdge: Rising edge (low to high transition)
            NegEdge: Falling edge (high to low transition)
        
        Example:
            >>> # Configure an analyzer to sample on the rising edge
            >>> settings.set_clock_edge(AnalyzerEnums.EdgeDirection.PosEdge)
        """
        PosEdge = int(_AnalyzerEnums.EdgeDirection.PosEdge)
        NegEdge = int(_AnalyzerEnums.EdgeDirection.NegEdge)
        
        def __str__(self) -> str:
            """Return a human-readable string representation."""
            return "Rising Edge" if self == self.PosEdge else "Falling Edge"
        
        def __repr__(self) -> str:
            """Return a code-like representation for debugging."""
            return f"AnalyzerEnums.EdgeDirection.{self.name}"
    
    class Edge(IntEnum):
        """
        Defines specific edges for analysis and interpretation.
        
        In some protocols, the leading and trailing edges of signals have
        distinct meanings and must be handled differently.
        
        Attributes:
            LeadingEdge: First edge of a sequence (often used for start conditions)
            TrailingEdge: Last edge of a sequence (often used for stop conditions)
        
        Example:
            >>> # Check if an edge is a leading edge in a sequence
            >>> if edge_type == AnalyzerEnums.Edge.LeadingEdge:
            ...     print("Start of sequence detected")
        """
        LeadingEdge = int(_AnalyzerEnums.Edge.LeadingEdge)
        TrailingEdge = int(_AnalyzerEnums.Edge.TrailingEdge)
        
        def __str__(self) -> str:
            """Return a human-readable string representation."""
            return "Leading Edge" if self == self.LeadingEdge else "Trailing Edge"
        
        def __repr__(self) -> str:
            """Return a code-like representation for debugging."""
            return f"AnalyzerEnums.Edge.{self.name}"
    
    class Parity(IntEnum):
        """
        Specifies parity settings for serial communication protocols.
        
        Parity is an error-detection mechanism used in many serial protocols
        like UART. It adds an extra bit to ensure the total number of 1 bits
        follows a specific pattern, enabling basic error detection.
        
        Attributes:
            None_: No parity bit (no error detection)
            Even: Even parity (total count of 1s including parity bit is even)
            Odd: Odd parity (total count of 1s including parity bit is odd)
        
        Example:
            >>> # Configure a UART analyzer to use even parity
            >>> uart_settings.set_parity(AnalyzerEnums.Parity.Even)
        """
        # Handling 'None' which is a Python keyword
        None_ = int(getattr(_AnalyzerEnums.Parity, 'None'))
        Even = int(_AnalyzerEnums.Parity.Even)
        Odd = int(_AnalyzerEnums.Parity.Odd)
        
        def __str__(self) -> str:
            """Return a human-readable string representation."""
            if self == self.None_:
                return "No Parity"
            elif self == self.Even:
                return "Even Parity"
            else:
                return "Odd Parity"
        
        def __repr__(self) -> str:
            """Return a code-like representation for debugging."""
            return f"AnalyzerEnums.Parity.{self.name}"
        
        def calculate(self, value: int, bits: int) -> BitState:
            """
            Calculate the expected parity bit for a given value.
            
            Args:
                value: The data value to calculate parity for
                bits: Number of data bits
                
            Returns:
                BitState representing the expected parity bit
                
            Raises:
                ValueError: If parity type is None_
                
            Example:
                >>> # Calculate parity for a byte
                >>> parity = AnalyzerEnums.Parity.Even.calculate(0x53, 8)
                >>> print(parity)  # BitState.HIGH or BitState.LOW
            """
            if self == self.None_:
                raise ValueError("Cannot calculate parity for 'None' parity type")
            
            # Count the number of 1 bits in the value
            ones_count = bin(value & ((1 << bits) - 1)).count('1')
            
            if self == self.Even:
                # For even parity, parity bit is 1 if count of 1s is odd
                return BitState.HIGH if ones_count % 2 else BitState.LOW
            else:  # Odd parity
                # For odd parity, parity bit is 1 if count of 1s is even
                return BitState.HIGH if ones_count % 2 == 0 else BitState.LOW
    
    class Acknowledge(IntEnum):
        """
        Represents acknowledgment states in communication protocols.
        
        Many protocols like I2C use an acknowledgment mechanism where the
        receiver confirms receipt of data. This enum represents the possible
        acknowledgment states.
        
        Attributes:
            Ack: Acknowledged (typically LOW for I2C - successful receipt)
            Nak: Not acknowledged (typically HIGH for I2C - error or end condition)
        
        Example:
            >>> # Check if a device acknowledged the transmission
            >>> if ack_state == AnalyzerEnums.Acknowledge.Ack:
            ...     print("Device acknowledged the data")
            >>> else:
            ...     print("Device did not acknowledge - possible error")
        """
        Ack = int(_AnalyzerEnums.Acknowledge.Ack)
        Nak = int(_AnalyzerEnums.Acknowledge.Nak)
        
        def __str__(self) -> str:
            """Return a human-readable string representation."""
            return "ACK" if self == self.Ack else "NAK"
        
        def __repr__(self) -> str:
            """Return a code-like representation for debugging."""
            return f"AnalyzerEnums.Acknowledge.{self.name}"
    
    class Sign(IntEnum):
        """
        Specifies if values should be interpreted as signed or unsigned.
        
        This affects how multi-byte values are displayed and interpreted
        in the analyzer interface and in exported data.
        
        Attributes:
            UnsignedInteger: Treat as unsigned value (all bits are magnitude)
            SignedInteger: Treat as signed value (MSB is sign bit)
        
        Example:
            >>> # Configure an analyzer to interpret data as signed integers
            >>> settings.set_number_format(DisplayBase.Decimal, 
            ...                           AnalyzerEnums.Sign.SignedInteger)
        """
        UnsignedInteger = int(_AnalyzerEnums.Sign.UnsignedInteger)
        SignedInteger = int(_AnalyzerEnums.Sign.SignedInteger)
        
        def __str__(self) -> str:
            """Return a human-readable string representation."""
            return "Unsigned" if self == self.UnsignedInteger else "Signed"
        
        def __repr__(self) -> str:
            """Return a code-like representation for debugging."""
            return f"AnalyzerEnums.Sign.{self.name}"
        
        def interpret_value(self, value: int, num_bits: int) -> int:
            """
            Interpret a numeric value according to this sign type.
            
            Args:
                value: Raw integer value to interpret
                num_bits: Number of bits in the value
                
            Returns:
                Interpreted value as signed or unsigned
                
            Example:
                >>> # Interpret 0xFF as an 8-bit value
                >>> SignedInteger.interpret_value(0xFF, 8)
                -1
                >>> UnsignedInteger.interpret_value(0xFF, 8)
                255
            """
            if self == self.UnsignedInteger:
                # Mask to ensure correct bit width
                return value & ((1 << num_bits) - 1)
            else:  # SignedInteger
                # Check if the sign bit is set
                if value & (1 << (num_bits - 1)):
                    # Convert to negative value using two's complement
                    return value - (1 << num_bits)
                else:
                    return value


# =============================================================================
# Classes
# =============================================================================

class Channel:
    """
    Represents a physical channel on the logic analyzer.
    
    Each channel is identified by a device ID and a channel index.
    The device ID identifies the specific logic analyzer hardware,
    and the channel index identifies the specific channel on that device.
    
    Channels are used to specify which inputs to use for protocol analyzers,
    to access captured data, and to configure device settings.
    
    Attributes:
        device_id (int): Identifier for the logic analyzer device
        channel_index (int): Channel number on the device (zero-based)
    
    Example:
        >>> # Create a channel representing the first channel on the default device
        >>> clk_channel = Channel(device_id=0, channel_index=0)
        >>> # Create a channel for the second device's fourth channel
        >>> data_channel = Channel(device_id=1, channel_index=3)
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
        In multi-device setups, this identifies which device
        the channel belongs to.
        
        Returns:
            Int: The device identifier
        """
        return self._channel.device_id
    
    @device_id.setter
    def device_id(self, value: int) -> None:
        """
        Set the device identifier for this channel.
        
        Args:
            value: New device ID
            
        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError("Device ID cannot be negative")
        self._channel.device_id = value
    
    @property
    def channel_index(self) -> int:
        """
        Channel number on the device (zero-based).
        
        Identifies which specific channel on the device this
        Channel object represents.
        
        Returns:
            Int: The channel index
        """
        return self._channel.channel_index
    
    @channel_index.setter
    def channel_index(self, value: int) -> None:
        """
        Set the channel index for this channel.
        
        Args:
            value: New channel index
            
        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError("Channel index cannot be negative")
        self._channel.channel_index = value
    
    def is_valid(self) -> bool:
        """
        Check if this channel is valid (not UNDEFINED_CHANNEL).
        
        Returns:
            bool: True if the channel is valid, False otherwise
            
        Example:
            >>> chan = Channel(0, 1)
            >>> chan.is_valid()  # True
            >>> undef = UNDEFINED_CHANNEL
            >>> undef.is_valid()  # False
        """
        return self != UNDEFINED_CHANNEL
    
    def __eq__(self, other: Any) -> bool:
        """Check if two channels are equal (same device and index)."""
        if not isinstance(other, Channel):
            return NotImplemented
        return (self.device_id == other.device_id and 
                self.channel_index == other.channel_index)
    
    def __ne__(self, other: Any) -> bool:
        """Check if two channels are not equal."""
        if not isinstance(other, Channel):
            return NotImplemented
        return not (self == other)
    
    def __lt__(self, other: Any) -> bool:
        """
        Compare channels for ordering (device_id first, then channel_index).
        
        This enables sorting channels in a consistent order.
        """
        if not isinstance(other, Channel):
            return NotImplemented
        if self.device_id != other.device_id:
            return self.device_id < other.device_id
        return self.channel_index < other.channel_index
    
    def __gt__(self, other: Any) -> bool:
        """
        Compare channels for ordering (device_id first, then channel_index).
        
        This enables sorting channels in a consistent order.
        """
        if not isinstance(other, Channel):
            return NotImplemented
        if self.device_id != other.device_id:
            return self.device_id > other.device_id
        return self.channel_index > other.channel_index
    
    def __repr__(self) -> str:
        """Return a code-like representation for debugging."""
        return f"Channel(device_id={self.device_id}, channel_index={self.channel_index})"
    
    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"Channel {self.channel_index} on Device {self.device_id}"
    
    def __hash__(self) -> int:
        """
        Calculate a hash value for the channel.
        
        This allows channels to be used as dictionary keys or in sets.
        """
        return hash((self.device_id, self.channel_index))
    
    # Access to the underlying C++ object
    @property
    def _internal(self) -> _Channel:
        """
        Get the internal C++ Channel object.
        
        This is primarily for internal use by other components
        that need to interact with the C++ SDK.
        
        Returns:
            The underlying C++ Channel object
        """
        return self._channel


# =============================================================================
# Constants
# =============================================================================

# Undefined channel constant - used to represent no channel selected
UNDEFINED_CHANNEL = Channel(_UNDEFINED_CHANNEL.device_id, _UNDEFINED_CHANNEL.channel_index)


# =============================================================================
# Utility Functions
# =============================================================================

def toggle_bit(bit: Union[BitState, int]) -> BitState:
    """
    Toggle a bit state (equivalent to C++ Toggle macro).
    
    This function inverts the state of a digital bit: LOW becomes HIGH, 
    and HIGH becomes LOW.
    
    Args:
        bit: The bit state to toggle (can be BitState enum or int 0/1)
    
    Returns:
        The toggled bit state (LOW->HIGH or HIGH->LOW)
        
    Raises:
        ValueError: If bit is not a valid BitState or 0/1 integer
        
    Example:
        >>> toggle_bit(BitState.LOW)
        BitState.HIGH
        >>> toggle_bit(BitState.HIGH)
        BitState.LOW
        >>> toggle_bit(0)  # Also works with integers
        BitState.HIGH
    """
    # Handle integer input (0/1)
    if isinstance(bit, int):
        if bit == 0:
            bit = BitState.LOW
        elif bit == 1:
            bit = BitState.HIGH
        else:
            raise ValueError(f"Integer bit value must be 0 or 1, got {bit}")
    
    # Handle BitState input
    elif not isinstance(bit, BitState):
        raise ValueError(f"Expected BitState or int, got {type(bit).__name__}")
    
    # Call the underlying function
    return BitState(_toggle_bit(bit))


def invert_bit(bit: Union[BitState, int]) -> BitState:
    """
    Invert a bit state (equivalent to C++ Invert macro).
    
    This function inverts the state of a digital bit: LOW becomes HIGH, 
    and HIGH becomes LOW. It's functionally identical to toggle_bit.
    
    Args:
        bit: The bit state to invert (can be BitState enum or int 0/1)
    
    Returns:
        The inverted bit state (LOW->HIGH or HIGH->LOW)
        
    Raises:
        ValueError: If bit is not a valid BitState or 0/1 integer
        
    Example:
        >>> invert_bit(BitState.LOW)
        BitState.HIGH
        >>> invert_bit(1)  # Also works with integers
        BitState.LOW
    """
    # Handle integer input (0/1)
    if isinstance(bit, int):
        if bit == 0:
            bit = BitState.LOW
        elif bit == 1:
            bit = BitState.HIGH
        else:
            raise ValueError(f"Integer bit value must be 0 or 1, got {bit}")
    
    # Handle BitState input
    elif not isinstance(bit, BitState):
        raise ValueError(f"Expected BitState or int, got {type(bit).__name__}")
    
    # Call the underlying function
    return BitState(_invert_bit(bit))


# =============================================================================
# Extended helper functions (not in original C++ API)
# =============================================================================

def bits_to_int(bits: List[BitState], msb_first: bool = True) -> int:
    """
    Convert a list of bits to an integer value.
    
    Args:
        bits: List of BitState values
        msb_first: If True, interpret bits as most significant bit first
        
    Returns:
        Integer value
        
    Example:
        >>> # Convert bits [HIGH, LOW, LOW, HIGH] to integer (binary 1001 = 9)
        >>> bits_to_int([BitState.HIGH, BitState.LOW, BitState.LOW, BitState.HIGH])
        9
    """
    if not bits:
        return 0
    
    result = 0
    if msb_first:
        for bit in bits:
            result = (result << 1) | (1 if bit == BitState.HIGH else 0)
    else:
        for bit in reversed(bits):
            result = (result << 1) | (1 if bit == BitState.HIGH else 0)
    
    return result


def int_to_bits(value: int, num_bits: int, msb_first: bool = True) -> List[BitState]:
    """
    Convert an integer to a list of bit states.
    
    Args:
        value: Integer value to convert
        num_bits: Number of bits to include in the result
        msb_first: If True, return bits with most significant bit first
        
    Returns:
        List of BitState values
        
    Example:
        >>> # Convert 9 (binary 1001) to list of bits with 4 bits
        >>> int_to_bits(9, 4)
        [BitState.HIGH, BitState.LOW, BitState.LOW, BitState.HIGH]
    """
    # Create a list of bits
    bits = []
    for i in range(num_bits):
        bit_value = (value >> i) & 1
        bits.append(BitState.HIGH if bit_value else BitState.LOW)
    
    # Reverse for MSB first (default)
    if msb_first:
        bits.reverse()
    
    return bits


def check_parity(value: int, num_bits: int, parity_type: AnalyzerEnums.Parity, 
                parity_bit: BitState) -> bool:
    """
    Check if a parity bit is correct for a given value.
    
    Args:
        value: Data value to check parity for
        num_bits: Number of bits in the data
        parity_type: Type of parity to check
        parity_bit: The actual parity bit to verify
        
    Returns:
        True if parity is correct, False otherwise
        
    Raises:
        ValueError: If parity_type is None_
        
    Example:
        >>> # Check if parity is correct for value 0x53 with even parity
        >>> data_value = 0x53  # 01010011 in binary (4 bits are 1)
        >>> parity_bit = BitState.LOW  # Makes total 1-bits even (4+0=4)
        >>> check_parity(data_value, 8, AnalyzerEnums.Parity.Even, parity_bit)
        True
    """
    if parity_type == AnalyzerEnums.Parity.None_:
        raise ValueError("Cannot check parity for 'None' parity type")
    
    expected_parity = parity_type.calculate(value, num_bits)
    return expected_parity == parity_bit


def count_ones(value: int, num_bits: int = None) -> int:
    """
    Count the number of 1 bits in an integer value.
    
    Args:
        value: Integer value to count bits in
        num_bits: Number of bits to consider (default: all bits)
        
    Returns:
        Count of 1 bits
        
    Example:
        >>> count_ones(0x53, 8)  # Binary 01010011
        4
    """
    if num_bits is not None:
        # Mask to the specified number of bits
        value &= (1 << num_bits) - 1
    
    # Use bin() to convert to binary string and count '1' characters
    # Skipping the '0b' prefix with [2:]
    return bin(value)[2:].count('1')


def format_value(value: int, display_base: DisplayBase, 
                num_bits: int = 8, sign: AnalyzerEnums.Sign = AnalyzerEnums.Sign.UnsignedInteger) -> str:
    """
    Format a value according to display base and sign.
    
    This function combines the display base formatting with sign interpretation
    to produce consistent, readable output for numeric values.
    
    Args:
        value: Value to format
        display_base: Display format to use
        num_bits: Number of bits in the value
        sign: Whether to interpret as signed or unsigned
        
    Returns:
        Formatted string representation
        
    Example:
        >>> # Format -5 as 8-bit signed hexadecimal
        >>> format_value(251, DisplayBase.Hexadecimal, 8, AnalyzerEnums.Sign.SignedInteger)
        '0xFB (-5)'
        >>> # Format same value as unsigned
        >>> format_value(251, DisplayBase.Hexadecimal, 8, AnalyzerEnums.Sign.UnsignedInteger)
        '0xFB'
    """
    # Apply sign interpretation
    interpreted_value = sign.interpret_value(value, num_bits)
    
    # Format according to display base
    formatted = display_base.format_value(value, num_bits)
    
    # For signed values that are negative, add the decimal interpretation
    if (sign == AnalyzerEnums.Sign.SignedInteger and 
        interpreted_value < 0 and 
        display_base != DisplayBase.Decimal):
        formatted += f" ({interpreted_value})"
    
    return formatted


# =============================================================================
# Extended numeric validation and conversion utilities
# =============================================================================

def validate_int_range(value: int, min_value: int, max_value: int, name: str = "value") -> int:
    """
    Validate that an integer value is within a specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        name: Name of the value for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValueError: If value is outside the allowed range
        
    Example:
        >>> # Validate a U8 value
        >>> validate_int_range(128, 0, 255, "byte_value")
        128
        >>> # This would raise an error:
        >>> # validate_int_range(300, 0, 255, "byte_value")
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}, got {value}")
    
    return value


def validate_u8(value: int, name: str = "U8 value") -> int:
    """Validate an 8-bit unsigned integer."""
    return validate_int_range(value, 0, U8_MAX, name)


def validate_s8(value: int, name: str = "S8 value") -> int:
    """Validate an 8-bit signed integer."""
    return validate_int_range(value, S8_MIN, S8_MAX, name)


def validate_u16(value: int, name: str = "U16 value") -> int:
    """Validate a 16-bit unsigned integer."""
    return validate_int_range(value, 0, U16_MAX, name)


def validate_s16(value: int, name: str = "S16 value") -> int:
    """Validate a 16-bit signed integer."""
    return validate_int_range(value, S16_MIN, S16_MAX, name)


def validate_u32(value: int, name: str = "U32 value") -> int:
    """Validate a 32-bit unsigned integer."""
    return validate_int_range(value, 0, U32_MAX, name)


def validate_s32(value: int, name: str = "S32 value") -> int:
    """Validate a 32-bit signed integer."""
    return validate_int_range(value, S32_MIN, S32_MAX, name)


def validate_u64(value: int, name: str = "U64 value") -> int:
    """Validate a 64-bit unsigned integer."""
    return validate_int_range(value, 0, U64_MAX, name)


def validate_s64(value: int, name: str = "S64 value") -> int:
    """Validate a 64-bit signed integer."""
    return validate_int_range(value, S64_MIN, S64_MAX, name)


# =============================================================================
# Type conversion utilities
# =============================================================================

def bytes_to_value(data: bytes, msb_first: bool = True) -> int:
    """
    Convert a sequence of bytes to an integer value.
    
    Args:
        data: Bytes to convert
        msb_first: Whether the most significant byte is first (big-endian)
        
    Returns:
        Integer value
        
    Example:
        >>> # Convert bytes [0x12, 0x34] to integer
        >>> bytes_to_value(b'\x12\x34')  # Default is MSB first (big-endian)
        4660  # 0x1234
        >>> bytes_to_value(b'\x12\x34', msb_first=False)  # LSB first (little-endian)
        13330  # 0x3412
    """
    if not data:
        return 0
    
    if msb_first:
        # Big-endian (most significant byte first)
        return int.from_bytes(data, byteorder='big')
    else:
        # Little-endian (least significant byte first)
        return int.from_bytes(data, byteorder='little')


def value_to_bytes(value: int, length: int, msb_first: bool = True) -> bytes:
    """
    Convert an integer value to a sequence of bytes.
    
    Args:
        value: Integer value to convert
        length: Number of bytes to produce
        msb_first: Whether to put the most significant byte first (big-endian)
        
    Returns:
        Bytes object
        
    Example:
        >>> # Convert 0x1234 to bytes with MSB first (big-endian)
        >>> value_to_bytes(0x1234, 2)
        b'\x12\x34'
        >>> # Convert 0x1234 to bytes with LSB first (little-endian)
        >>> value_to_bytes(0x1234, 2, msb_first=False)
        b'\x34\x12'
    """
    if msb_first:
        # Big-endian (most significant byte first)
        return value.to_bytes(length, byteorder='big')
    else:
        # Little-endian (least significant byte first)
        return value.to_bytes(length, byteorder='little')


# Update exports to include the new functions
__all__ += [
    # Extended utility functions
    'bits_to_int', 'int_to_bits', 'check_parity', 'count_ones',
    'format_value',
    
    # Validation utilities
    'validate_int_range', 
    'validate_u8', 'validate_s8',
    'validate_u16', 'validate_s16',
    'validate_u32', 'validate_s32',
    'validate_u64', 'validate_s64',
    
    # Type conversion utilities
    'bytes_to_value', 'value_to_bytes',
]