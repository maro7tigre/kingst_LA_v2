"""
Utility functions for working with the Kingst Logic Analyzer.

This module provides helper functions for common operations when working with
logic analyzer data, including formatting, bit manipulation, timing conversions,
data transformation utilities, and file operations. It serves as a Pythonic wrapper
around the C++ AnalyzerHelpers class and related utility classes.

The module is organized into several functional categories:
- Bit manipulation: Functions for working with individual bits and binary data
- Formatting: Converting values to readable strings
- Timing: Converting between samples and time measurements
- File operations: Saving and loading data
- Bit conversion: Transforming between different bit representations
- Data transformation: Helper functions for manipulating data structures

Example:
    >>> from kingst_LA import helpers
    >>> # Format a value as hexadecimal
    >>> hex_str = helpers.format_number(42, helpers.DisplayBase.Hexadecimal, 8)
    >>> print(hex_str)
    '0x2A'
    
    >>> # Calculate time from samples
    >>> time_s = helpers.samples_to_seconds(1000, 20_000_000)
    >>> print(f"Time: {time_s*1000:.3f} ms")
    'Time: 0.050 ms'
"""

from typing import List, Tuple, Dict, Optional, Union, BinaryIO, Any, Callable, Iterator, TypeVar, Iterable
import math
import io
import struct
import warnings
from enum import Enum
from dataclasses import dataclass
import functools

# Try to import NumPy for advanced operations, but make it optional
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Try to import Matplotlib for visualization, but make it optional
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Import the underlying C++ bindings
from kingst_analyzer._kingst_analyzer import (
    AnalyzerHelpers as _AnalyzerHelpers,
    BitExtractor, BitState, DataBuilder, SimpleArchive, ClockGenerator,
    DisplayBase, ShiftOrder
)

# Type aliases for improved type hints
SampleNumber = int
TimeSeconds = float
T = TypeVar('T')  # Generic type for data transformation functions

# Re-export the helper classes and enums
__all__ = [
    # Enums and classes
    'BitState', 'DisplayBase', 'ShiftOrder',
    'BitExtractor', 'DataBuilder', 'SimpleArchive', 'ClockGenerator',
    
    # Bit manipulation
    'is_even', 'is_odd', 'get_ones_count', 'diff32', 
    'bit_count', 'convert_to_signed', 'convert_from_signed',
    'has_bit_set', 'set_bit', 'clear_bit', 'toggle_bit', 'mask_bits',
    
    # Formatting
    'format_number', 'format_time', 'format_value',
    'format_bytes_as_hex', 'format_binary', 'format_engineering',
    
    # Timing
    'samples_to_seconds', 'seconds_to_samples', 
    'time_delta', 'phase_to_samples', 'calculate_frequency',
    'calculate_period', 'calculate_duty_cycle',
    
    # File operations
    'save_to_file', 'export_data', 'export_csv', 'export_binary',
    'export_waveform_data', 'read_from_file',
    
    # Bit conversion
    'bits_to_bytes', 'bytes_to_bits', 'extract_bits', 'build_value',
    'bit_slice', 'reverse_bits', 'count_transitions', 'find_edges',
    
    # Convenience data transformation
    'group_by_n', 'interleave', 'deinterleave', 'smooth_signal',
    'moving_average', 'downsample', 'upsample', 'threshold', 'hysteresis',
    
    # Context manager
    'TimingBlockScope'
]


# ============================================================================
# Exception classes
# ============================================================================

class HelperError(Exception):
    """Base exception for all helper-related errors."""
    pass


class BitManipulationError(HelperError):
    """Exception raised for errors in bit manipulation functions."""
    pass


class FormattingError(HelperError):
    """Exception raised for errors in formatting functions."""
    pass


class TimingError(HelperError):
    """Exception raised for errors in timing conversion functions."""
    pass


class FileOperationError(HelperError):
    """Exception raised for errors in file operations."""
    pass


class DataTransformationError(HelperError):
    """Exception raised for errors in data transformation functions."""
    pass


# ============================================================================
# Context Manager for Timing Blocks
# ============================================================================

@dataclass
class TimingBlockScope:
    """
    Context manager for measuring timing between blocks of code.
    
    This context manager helps track sample numbers or time durations
    between different sections of code, making it easier to generate
    accurate timing for simulations or analysis.
    
    Attributes:
        start_sample (int): Starting sample number for this block
        sample_count (int): Number of samples in this block
        sample_rate_hz (int): Sample rate in Hz, for time calculations
        
    Example:
        >>> # Create a timing block starting at sample 1000 with 100MHz sample rate
        >>> with TimingBlockScope(1000, sample_rate_hz=100_000_000) as timing:
        >>>     # Do some operations...
        >>>     timing.advance(50)  # Advance by 50 samples
        >>>     # Do more operations...
        >>>     timing.advance_by_time(0.000001)  # Advance by 1μs
        >>>     
        >>> # Get the final sample number after the block
        >>> final_sample = timing.end_sample
        >>> print(f"Block took {timing.duration_seconds * 1e6:.1f} μs")
    """
    
    start_sample: int
    sample_count: int = 0
    sample_rate_hz: Optional[int] = None
    
    def __enter__(self) -> 'TimingBlockScope':
        """Enter the timing block context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the timing block context."""
        pass
    
    @property
    def current_sample(self) -> int:
        """Get the current sample number."""
        return self.start_sample + self.sample_count
    
    @property
    def end_sample(self) -> int:
        """Get the final sample number after this block."""
        return self.current_sample
    
    @property
    def duration_samples(self) -> int:
        """Get the total number of samples in this block."""
        return self.sample_count
    
    @property
    def duration_seconds(self) -> float:
        """Get the total time duration of this block in seconds."""
        if self.sample_rate_hz is None or self.sample_rate_hz <= 0:
            raise TimingError("Sample rate must be set to calculate duration in seconds")
        return self.sample_count / self.sample_rate_hz
    
    @property
    def start_time(self) -> float:
        """Get the start time of this block in seconds."""
        if self.sample_rate_hz is None or self.sample_rate_hz <= 0:
            raise TimingError("Sample rate must be set to calculate time")
        return self.start_sample / self.sample_rate_hz
    
    @property
    def end_time(self) -> float:
        """Get the end time of this block in seconds."""
        if self.sample_rate_hz is None or self.sample_rate_hz <= 0:
            raise TimingError("Sample rate must be set to calculate time")
        return self.current_sample / self.sample_rate_hz
    
    def advance(self, samples: int) -> int:
        """
        Advance by a specific number of samples.
        
        Args:
            samples (int): Number of samples to advance
            
        Returns:
            int: New current sample number
            
        Raises:
            ValueError: If samples is negative
        """
        if samples < 0:
            raise ValueError("Cannot advance by a negative number of samples")
        self.sample_count += samples
        return self.current_sample
    
    def advance_by_time(self, seconds: float) -> int:
        """
        Advance by a specific time duration.
        
        Args:
            seconds (float): Time to advance in seconds
            
        Returns:
            int: New current sample number
            
        Raises:
            TimingError: If sample rate is not set
            ValueError: If seconds is negative
        """
        if self.sample_rate_hz is None or self.sample_rate_hz <= 0:
            raise TimingError("Sample rate must be set to advance by time")
        if seconds < 0:
            raise ValueError("Cannot advance by a negative time")
        
        samples = round(seconds * self.sample_rate_hz)
        return self.advance(samples)
    
    def reset(self, new_start_sample: Optional[int] = None) -> None:
        """
        Reset the timing block with a new starting point.
        
        Args:
            new_start_sample (int, optional): New starting sample.
                                             If None, reset to original start_sample.
        """
        if new_start_sample is not None:
            self.start_sample = new_start_sample
        self.sample_count = 0


# ============================================================================
# Bit Manipulation Functions
# ============================================================================

def is_even(value: int) -> bool:
    """
    Check if a number is even.
    
    Args:
        value: Integer to check
        
    Returns:
        True if the number is even, False otherwise
        
    Examples:
        >>> is_even(42)
        True
        >>> is_even(7)
        False
    """
    return bool(_AnalyzerHelpers.is_even(value))


def is_odd(value: int) -> bool:
    """
    Check if a number is odd.
    
    Args:
        value: Integer to check
        
    Returns:
        True if the number is odd, False otherwise
        
    Examples:
        >>> is_odd(42)
        False
        >>> is_odd(7)
        True
    """
    return bool(_AnalyzerHelpers.is_odd(value))


def get_ones_count(value: int) -> int:
    """
    Count the number of bits set to 1 in a value.
    
    Args:
        value: Integer to check
        
    Returns:
        Number of bits set to 1
        
    Examples:
        >>> get_ones_count(0xA5)  # 10100101
        4
    """
    return int(_AnalyzerHelpers.get_ones_count(value))


# Alias for get_ones_count with a more Pythonic name
bit_count = get_ones_count


def diff32(a: int, b: int) -> int:
    """
    Calculate the absolute difference between two 32-bit values.
    
    Args:
        a: First value
        b: Second value
        
    Returns:
        Absolute difference |a - b|
        
    Examples:
        >>> diff32(100, 50)
        50
        >>> diff32(50, 100)
        50
    """
    return int(_AnalyzerHelpers.diff32(a, b))


def convert_to_signed(value: int, num_bits: int) -> int:
    """
    Convert an unsigned number to a signed number with the specified bit width.
    
    Args:
        value: Unsigned integer to convert
        num_bits: Number of bits in the value
        
    Returns:
        Signed representation of the number
        
    Raises:
        ValueError: If num_bits is less than 1 or greater than 64
        
    Examples:
        >>> convert_to_signed(0xFF, 8)  # 11111111 in 8 bits = -1
        -1
        >>> convert_to_signed(0x80, 8)  # 10000000 in 8 bits = -128
        -128
    """
    if num_bits < 1 or num_bits > 64:
        raise ValueError("Number of bits must be between 1 and 64")
    
    return int(_AnalyzerHelpers.convert_to_signed_number(value, num_bits))


def convert_from_signed(value: int, num_bits: int) -> int:
    """
    Convert a signed number to an unsigned number with the specified bit width.
    
    Args:
        value: Signed integer to convert
        num_bits: Number of bits in the value
        
    Returns:
        Unsigned representation of the number
        
    Raises:
        ValueError: If num_bits is less than 1 or greater than 64
        
    Examples:
        >>> convert_from_signed(-1, 8)  # -1 in 8 bits = 11111111 = 0xFF
        255
        >>> convert_from_signed(-128, 8)  # -128 in 8 bits = 10000000 = 0x80
        128
    """
    if num_bits < 1 or num_bits > 64:
        raise ValueError("Number of bits must be between 1 and 64")
    
    if value < 0:
        # Compute two's complement
        mask = (1 << num_bits) - 1
        return ((~abs(value) + 1) & mask)
    else:
        return value & ((1 << num_bits) - 1)


def has_bit_set(value: int, bit_position: int) -> bool:
    """
    Check if a specific bit is set in a value.
    
    Args:
        value: Integer to check
        bit_position: Position of the bit to check (0-based, from LSB)
        
    Returns:
        True if the bit is set (1), False if clear (0)
        
    Raises:
        ValueError: If bit_position is negative
        
    Examples:
        >>> has_bit_set(0xA5, 0)  # 10100101, bit 0 is 1
        True
        >>> has_bit_set(0xA5, 1)  # 10100101, bit 1 is 0
        False
    """
    if bit_position < 0:
        raise ValueError("Bit position cannot be negative")
    
    return bool((value >> bit_position) & 1)


def set_bit(value: int, bit_position: int) -> int:
    """
    Set a specific bit in a value.
    
    Args:
        value: Integer to modify
        bit_position: Position of the bit to set (0-based, from LSB)
        
    Returns:
        New value with the bit set
        
    Raises:
        ValueError: If bit_position is negative
        
    Examples:
        >>> set_bit(0xA0, 0)  # 10100000 -> 10100001
        161  # 0xA1
        >>> set_bit(0xA5, 1)  # 10100101 -> 10100111
        167  # 0xA7
    """
    if bit_position < 0:
        raise ValueError("Bit position cannot be negative")
    
    return value | (1 << bit_position)


def clear_bit(value: int, bit_position: int) -> int:
    """
    Clear a specific bit in a value.
    
    Args:
        value: Integer to modify
        bit_position: Position of the bit to clear (0-based, from LSB)
        
    Returns:
        New value with the bit cleared
        
    Raises:
        ValueError: If bit_position is negative
        
    Examples:
        >>> clear_bit(0xA5, 0)  # 10100101 -> 10100100
        164  # 0xA4
        >>> clear_bit(0xA5, 2)  # 10100101 -> 10100001
        161  # 0xA1
    """
    if bit_position < 0:
        raise ValueError("Bit position cannot be negative")
    
    return value & ~(1 << bit_position)


def toggle_bit(value: int, bit_position: int) -> int:
    """
    Toggle a specific bit in a value.
    
    Args:
        value: Integer to modify
        bit_position: Position of the bit to toggle (0-based, from LSB)
        
    Returns:
        New value with the bit toggled
        
    Raises:
        ValueError: If bit_position is negative
        
    Examples:
        >>> toggle_bit(0xA5, 0)  # 10100101 -> 10100100
        164  # 0xA4
        >>> toggle_bit(0xA5, 1)  # 10100101 -> 10100111
        167  # 0xA7
    """
    if bit_position < 0:
        raise ValueError("Bit position cannot be negative")
    
    return value ^ (1 << bit_position)


def mask_bits(value: int, mask: int) -> int:
    """
    Apply a bit mask to a value.
    
    Args:
        value: Integer to mask
        mask: Bit mask to apply
        
    Returns:
        Masked value (value & mask)
        
    Examples:
        >>> mask_bits(0xA5, 0x0F)  # 10100101 & 00001111
        5  # 0x05
        >>> mask_bits(0xA5, 0xF0)  # 10100101 & 11110000
        160  # 0xA0
    """
    return value & mask


# ============================================================================
# Formatting Functions
# ============================================================================

def format_number(value: int, display_base: DisplayBase, num_bits: int) -> str:
    """
    Format a number as a string using the specified display base.
    
    Args:
        value: Number to format
        display_base: Base to display in (Binary, Decimal, Hexadecimal, etc.)
        num_bits: Number of data bits in the value
        
    Returns:
        Formatted string representation
        
    Raises:
        ValueError: If num_bits is less than 1 or greater than 64
        
    Examples:
        >>> format_number(42, DisplayBase.Decimal, 8)
        '42'
        >>> format_number(42, DisplayBase.Hexadecimal, 8)
        '0x2A'
        >>> format_number(42, DisplayBase.Binary, 8)
        '00101010'
    """
    if num_bits < 1 or num_bits > 64:
        raise ValueError("Number of bits must be between 1 and 64")
    
    try:
        return _AnalyzerHelpers.format_number(value, display_base, num_bits)
    except Exception as e:
        raise FormattingError(f"Error formatting number: {e}")


def format_time(sample: int, trigger_sample: int, sample_rate_hz: int) -> str:
    """
    Format a sample number as a time string.
    
    Args:
        sample: Sample number to format
        trigger_sample: Trigger sample number (reference point)
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        Formatted time string
        
    Raises:
        ValueError: If sample_rate_hz is less than or equal to 0
        
    Examples:
        >>> format_time(1000, 0, 1_000_000)  # 1000 samples at 1MHz
        '+1.00 ms'
        >>> format_time(0, 1000, 1_000_000)  # 0 samples with trigger at 1000
        '-1.00 ms'
    """
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be greater than 0")
    
    try:
        return _AnalyzerHelpers.format_time(sample, trigger_sample, sample_rate_hz)
    except Exception as e:
        raise FormattingError(f"Error formatting time: {e}")


def format_value(value: Union[int, float], 
                format_type: str = "auto", 
                num_bits: int = 8,
                prefix: bool = True) -> str:
    """
    Versatile formatter for values with automatic type detection.
    
    Args:
        value: Value to format
        format_type: How to format the value:
            - "auto": Automatic detection based on value type
            - "dec" or "decimal": Decimal format
            - "hex" or "hexadecimal": Hexadecimal format
            - "bin" or "binary": Binary format
            - "ascii": ASCII character (for 0-127 values)
            - "float": Floating point with automatic precision
            - "eng": Engineering notation with SI prefix
        num_bits: Number of bits for integer formatting
        prefix: Whether to include prefix (0x, 0b) for hex/binary
        
    Returns:
        Formatted string representation
        
    Raises:
        ValueError: If format_type is invalid or num_bits is out of range
        
    Examples:
        >>> format_value(42)
        '42'
        >>> format_value(42, "hex")
        '0x2A'
        >>> format_value(42, "bin")
        '0b00101010'
        >>> format_value(42, "ascii")
        "'*' (42)"
        >>> format_value(1.23456e-6, "eng")
        '1.23 µ'
    """
    if num_bits < 1 or num_bits > 64:
        raise ValueError("Number of bits must be between 1 and 64")
    
    valid_formats = {"auto", "dec", "decimal", "hex", "hexadecimal", 
                    "bin", "binary", "ascii", "float", "eng"}
    if format_type not in valid_formats:
        raise ValueError(f"Invalid format type: {format_type}. "
                         f"Valid options are: {', '.join(valid_formats)}")
    
    # Determine format type automatically if set to auto
    if format_type == "auto":
        if isinstance(value, float):
            format_type = "float"
        elif isinstance(value, int):
            if 0 <= value <= 127:
                format_type = "ascii"
            else:
                format_type = "hex"
    
    # Format based on type
    if format_type in ("dec", "decimal"):
        return str(value)
    
    elif format_type in ("hex", "hexadecimal"):
        hex_str = format(value, 'x').upper()
        num_chars = math.ceil(num_bits / 4)
        hex_str = hex_str.zfill(num_chars)
        return f"0x{hex_str}" if prefix else hex_str
    
    elif format_type in ("bin", "binary"):
        bin_str = format(value, 'b').zfill(num_bits)
        return f"0b{bin_str}" if prefix else bin_str
    
    elif format_type == "ascii":
        if 0 <= value <= 127:
            char = chr(value)
            if char.isprintable():
                return f"'{char}' ({value})"
            else:
                ctrl_chars = {
                    0: "NUL", 1: "SOH", 2: "STX", 3: "ETX", 4: "EOT", 5: "ENQ",
                    6: "ACK", 7: "BEL", 8: "BS", 9: "HT", 10: "LF", 11: "VT",
                    12: "FF", 13: "CR", 14: "SO", 15: "SI", 16: "DLE", 17: "DC1",
                    18: "DC2", 19: "DC3", 20: "DC4", 21: "NAK", 22: "SYN", 23: "ETB",
                    24: "CAN", 25: "EM", 26: "SUB", 27: "ESC", 28: "FS", 29: "GS",
                    30: "RS", 31: "US", 127: "DEL"
                }
                return f"'{ctrl_chars.get(value, 'CTRL')}' ({value})"
        return str(value)
    
    elif format_type == "float":
        if abs(value) < 1e-3 or abs(value) >= 1e5:
            return f"{value:.6e}"
        else:
            return f"{value:.6g}"
    
    elif format_type == "eng":
        return format_engineering(value)
    
    else:
        return str(value)


def format_bytes_as_hex(data: Union[bytes, List[int]], 
                      bytes_per_line: int = 16,
                      include_ascii: bool = True) -> str:
    """
    Format bytes as a hexadecimal dump.
    
    Args:
        data: Bytes or list of integers to format
        bytes_per_line: Number of bytes to display per line
        include_ascii: Whether to include ASCII representation
        
    Returns:
        Formatted hexadecimal dump
        
    Examples:
        >>> data = bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x20, 0x57, 0x6F, 0x72, 0x6C, 0x64])
        >>> print(format_bytes_as_hex(data, bytes_per_line=8))
        00000000: 48 65 6C 6C 6F 20 57 6F  Hello Wo
        00000008: 72 6C 64                 rld
    """
    # Convert list of integers to bytes if needed
    if isinstance(data, list):
        data = bytes(data)
    
    result = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        # Address
        line = f"{i:08X}: "
        
        # Hex values with spacing
        hex_values = " ".join(f"{byte:02X}" for byte in chunk)
        padding = " " * (bytes_per_line * 3 - len(hex_values))
        line += hex_values + padding
        
        # ASCII representation
        if include_ascii:
            ascii_repr = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            line += " " + ascii_repr
        
        result.append(line)
    
    return "\n".join(result)


def format_binary(value: int, 
                 num_bits: int = 8, 
                 group_size: int = 4, 
                 prefix: bool = True) -> str:
    """
    Format a value as a binary string with optional grouping.
    
    Args:
        value: Integer to format
        num_bits: Number of bits to display
        group_size: Number of bits per group (0 for no grouping)
        prefix: Whether to include '0b' prefix
        
    Returns:
        Formatted binary string
        
    Raises:
        ValueError: If num_bits or group_size is invalid
        
    Examples:
        >>> format_binary(42, 8)
        '0b0010_1010'
        >>> format_binary(42, 8, group_size=0)
        '0b00101010'
        >>> format_binary(42, 8, prefix=False)
        '0010_1010'
    """
    if num_bits < 1:
        raise ValueError("Number of bits must be at least 1")
    if group_size < 0:
        raise ValueError("Group size cannot be negative")
    
    # Format as binary string with leading zeros
    bin_str = format(value & ((1 << num_bits) - 1), f"0{num_bits}b")
    
    # Apply grouping if requested
    if group_size > 0:
        groups = []
        for i in range(0, len(bin_str), group_size):
            groups.append(bin_str[max(0, len(bin_str) - i - group_size):
                               len(bin_str) - i])
        bin_str = "_".join(reversed(groups))
    
    # Add prefix if requested
    if prefix:
        bin_str = "0b" + bin_str
    
    return bin_str


def format_engineering(value: float, 
                      precision: int = 2, 
                      unit: str = "") -> str:
    """
    Format a value in engineering notation with SI prefix.
    
    Args:
        value: Value to format
        precision: Number of decimal places to show
        unit: Optional unit string to append
        
    Returns:
        Formatted string with SI prefix
        
    Examples:
        >>> format_engineering(1234.5678)
        '1.23 k'
        >>> format_engineering(0.0012345, precision=3)
        '1.235 m'
        >>> format_engineering(1.234e6, unit="Hz")
        '1.23 MHz'
    """
    if value == 0:
        return f"0{' ' + unit if unit else ''}"
    
    prefixes = {
        -24: "y", -21: "z", -18: "a", -15: "f", -12: "p", -9: "n",
        -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G",
        12: "T", 15: "P", 18: "E", 21: "Z", 24: "Y"
    }
    
    exponent = int(math.floor(math.log10(abs(value)) / 3) * 3)
    exponent = max(-24, min(24, exponent))  # Limit to standard SI prefixes
    
    mantissa = value / (10 ** exponent)
    si_prefix = prefixes[exponent]
    
    if unit:
        return f"{mantissa:.{precision}f} {si_prefix}{unit}"
    else:
        return f"{mantissa:.{precision}f} {si_prefix}".rstrip()


# ============================================================================
# Timing Conversion Functions
# ============================================================================

def samples_to_seconds(samples: Union[int, float], sample_rate_hz: int) -> float:
    """
    Convert samples to seconds at the given sample rate.
    
    Args:
        samples: Number of samples
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        Time in seconds
        
    Raises:
        ValueError: If sample_rate_hz is less than or equal to 0
        
    Examples:
        >>> samples_to_seconds(1000, 1_000_000)  # 1000 samples at 1MHz
        0.001  # 1ms
    """
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be greater than 0")
    
    return float(samples) / float(sample_rate_hz)


def seconds_to_samples(seconds: float, sample_rate_hz: int) -> int:
    """
    Convert time in seconds to samples at the given sample rate.
    
    Args:
        seconds: Time in seconds
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        Number of samples (rounded to nearest integer)
        
    Raises:
        ValueError: If sample_rate_hz is less than or equal to 0
        
    Examples:
        >>> seconds_to_samples(0.001, 1_000_000)  # 1ms at 1MHz
        1000
    """
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be greater than 0")
    
    return round(float(seconds) * float(sample_rate_hz))


def time_delta(sample1: int, sample2: int, sample_rate_hz: int) -> float:
    """
    Calculate time difference between two sample points.
    
    Args:
        sample1: First sample number
        sample2: Second sample number
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        Time difference in seconds (positive if sample2 > sample1)
        
    Raises:
        ValueError: If sample_rate_hz is less than or equal to 0
        
    Examples:
        >>> time_delta(1000, 1500, 1_000_000)  # 500 samples at 1MHz
        0.0005  # 0.5ms
        >>> time_delta(1500, 1000, 1_000_000)  # -500 samples at 1MHz
        -0.0005  # -0.5ms
    """
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be greater than 0")
    
    return samples_to_seconds(sample2 - sample1, sample_rate_hz)


def phase_to_samples(phase_deg: float, period_samples: int) -> int:
    """
    Convert a phase in degrees to an equivalent number of samples.
    
    Args:
        phase_deg: Phase in degrees (0-360)
        period_samples: Number of samples in one complete period
        
    Returns:
        Number of samples corresponding to the phase
        
    Raises:
        ValueError: If period_samples is less than or equal to 0
        
    Examples:
        >>> phase_to_samples(90, 100)  # 90° phase in a 100-sample period
        25
        >>> phase_to_samples(180, 100)  # 180° phase in a 100-sample period
        50
    """
    if period_samples <= 0:
        raise ValueError("Period samples must be greater than 0")
    
    return round((phase_deg % 360) * period_samples / 360.0)


def calculate_frequency(period_samples: int, sample_rate_hz: int) -> float:
    """
    Calculate frequency from period in samples.
    
    Args:
        period_samples: Number of samples in one complete period
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        Frequency in Hz
        
    Raises:
        ValueError: If period_samples or sample_rate_hz is less than or equal to 0
        
    Examples:
        >>> calculate_frequency(100, 1_000_000)  # 100 samples at 1MHz
        10000.0  # 10kHz
    """
    if period_samples <= 0:
        raise ValueError("Period samples must be greater than 0")
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be greater than 0")
    
    return float(sample_rate_hz) / float(period_samples)


def calculate_period(frequency_hz: float, sample_rate_hz: int) -> int:
    """
    Calculate period in samples from frequency.
    
    Args:
        frequency_hz: Frequency in Hz
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        Period in samples (rounded to nearest integer)
        
    Raises:
        ValueError: If frequency_hz or sample_rate_hz is less than or equal to 0
        
    Examples:
        >>> calculate_period(10000, 1_000_000)  # 10kHz at 1MHz
        100
    """
    if frequency_hz <= 0:
        raise ValueError("Frequency must be greater than 0")
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be greater than 0")
    
    return round(float(sample_rate_hz) / float(frequency_hz))


def calculate_duty_cycle(high_samples: int, total_period_samples: int) -> float:
    """
    Calculate duty cycle from high and total period samples.
    
    Args:
        high_samples: Number of samples in the high state
        total_period_samples: Total number of samples in the period
        
    Returns:
        Duty cycle as a percentage (0-100)
        
    Raises:
        ValueError: If high_samples is negative or total_period_samples is less than or equal to 0
        
    Examples:
        >>> calculate_duty_cycle(25, 100)  # 25 high samples in a 100-sample period
        25.0  # 25% duty cycle
    """
    if high_samples < 0:
        raise ValueError("High samples cannot be negative")
    if total_period_samples <= 0:
        raise ValueError("Total period samples must be greater than 0")
    
    return (float(high_samples) / float(total_period_samples)) * 100.0


# ============================================================================
# File Export Functions
# ============================================================================

def save_to_file(file_name: str, data: Union[bytes, List[int]], is_binary: bool = False) -> None:
    """
    Save data to a file.
    
    Args:
        file_name: Name of the file to save
        data: Data to save (bytes or list of integers)
        is_binary: True for binary file, False for text file
        
    Raises:
        FileOperationError: If the file cannot be saved
        
    Examples:
        >>> save_to_file("data.txt", b"Hello, world!")
        >>> save_to_file("data.bin", [0x00, 0x01, 0x02, 0x03], is_binary=True)
    """
    # Convert list of integers to bytes if needed
    if isinstance(data, list):
        data = bytes(data)
    
    try:
        _AnalyzerHelpers.save_file(file_name, data, is_binary)
    except Exception as e:
        raise FileOperationError(f"Error saving file: {e}")


def export_data(file_name: str, 
                data: Any,
                format_func: Optional[Callable] = None,
                is_binary: bool = False) -> None:
    """
    Export data to a file with optional formatting.
    
    This function uses the streaming file API for efficient memory usage.
    
    Args:
        file_name: Name of the file to save
        data: Data to export (depends on format_func)
        format_func: Function to format each data item
                    If None, data is written as-is
        is_binary: True for binary file, False for text file
        
    Raises:
        FileOperationError: If the file cannot be exported
        
    Examples:
        >>> # Export a list of numbers as hex values
        >>> export_data(
        ...     "data.txt", 
        ...     [10, 20, 30, 40], 
        ...     lambda x: f"0x{x:02X}\\n"
        ... )
        
        >>> # Export binary data
        >>> export_data(
        ...     "data.bin",
        ...     [0x00, 0x01, 0x02, 0x03],
        ...     lambda x: bytes([x]),
        ...     is_binary=True
        ... )
    """
    try:
        file = _AnalyzerHelpers.start_file(file_name, is_binary)
    except Exception as e:
        raise FileOperationError(f"Error starting file export: {e}")
    
    try:
        if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
            # Iterable data - process each item
            for item in data:
                if format_func:
                    formatted = format_func(item)
                    if isinstance(formatted, str):
                        formatted = formatted.encode("utf-8")
                    _AnalyzerHelpers.append_to_file(formatted, file)
                else:
                    # Default formatting
                    if isinstance(item, str):
                        _AnalyzerHelpers.append_to_file(item.encode("utf-8"), file)
                    elif isinstance(item, bytes):
                        _AnalyzerHelpers.append_to_file(item, file)
                    else:
                        _AnalyzerHelpers.append_to_file(str(item).encode("utf-8"), file)
        else:
            # Non-iterable data - process as a single item
            if format_func:
                formatted = format_func(data)
                if isinstance(formatted, str):
                    formatted = formatted.encode("utf-8")
                _AnalyzerHelpers.append_to_file(formatted, file)
            else:
                # Default formatting
                if isinstance(data, str):
                    _AnalyzerHelpers.append_to_file(data.encode("utf-8"), file)
                elif isinstance(data, bytes):
                    _AnalyzerHelpers.append_to_file(data, file)
                else:
                    _AnalyzerHelpers.append_to_file(str(data).encode("utf-8"), file)
    except Exception as e:
        raise FileOperationError(f"Error during file export: {e}")
    finally:
        _AnalyzerHelpers.end_file(file)


def export_csv(file_name: str, 
              data: List[List[Any]], 
              headers: Optional[List[str]] = None,
              delimiter: str = ",") -> None:
    """
    Export data to a CSV file.
    
    Args:
        file_name: Name of the CSV file to save
        data: List of rows, where each row is a list of values
        headers: Optional list of column headers
        delimiter: Delimiter character (default: comma)
        
    Raises:
        FileOperationError: If the CSV file cannot be exported
        
    Examples:
        >>> # Export a list of rows with headers
        >>> export_csv(
        ...     "data.csv",
        ...     [[1, "John", 25], [2, "Alice", 30]],
        ...     ["ID", "Name", "Age"]
        ... )
    """
    try:
        file = _AnalyzerHelpers.start_file(file_name, False)
    except Exception as e:
        raise FileOperationError(f"Error starting CSV export: {e}")
    
    try:
        # Write headers if provided
        if headers:
            header_line = delimiter.join(str(h) for h in headers) + "\n"
            _AnalyzerHelpers.append_to_file(header_line.encode("utf-8"), file)
        
        # Write data rows
        for row in data:
            line = delimiter.join(str(cell) for cell in row) + "\n"
            _AnalyzerHelpers.append_to_file(line.encode("utf-8"), file)
    except Exception as e:
        raise FileOperationError(f"Error during CSV export: {e}")
    finally:
        _AnalyzerHelpers.end_file(file)


def export_binary(file_name: str, 
                 data: List[int], 
                 element_size: int = 1) -> None:
    """
    Export data to a binary file.
    
    Args:
        file_name: Name of the binary file to save
        data: List of integers to save
        element_size: Size of each element in bytes (1, 2, 4, or 8)
        
    Raises:
        ValueError: If element_size is not 1, 2, 4, or 8
        FileOperationError: If the binary file cannot be exported
        
    Examples:
        >>> # Export as bytes (8-bit values)
        >>> export_binary("data.bin", [0x00, 0x01, 0x02, 0x03])
        
        >>> # Export as 16-bit values
        >>> export_binary("data.bin", [0x1234, 0x5678], element_size=2)
        
        >>> # Export as 32-bit values
        >>> export_binary("data.bin", [0x12345678, 0x9ABCDEF0], element_size=4)
    """
    if element_size not in (1, 2, 4, 8):
        raise ValueError("Element size must be 1, 2, 4, or 8 bytes")
    
    try:
        with open(file_name, "wb") as f:
            format_char = {1: "B", 2: "H", 4: "I", 8: "Q"}[element_size]
            for value in data:
                f.write(struct.pack(f"<{format_char}", value))
    except Exception as e:
        raise FileOperationError(f"Error exporting binary file: {e}")


def export_waveform_data(file_name: str, 
                        samples: List[int], 
                        states: List[BitState],
                        sample_rate_hz: Optional[int] = None,
                        format_type: str = "csv",
                        include_time: bool = True) -> None:
    """
    Export digital waveform data to a file.
    
    Args:
        file_name: Name of the file to save
        samples: List of sample numbers
        states: List of bit states (BitState.HIGH or BitState.LOW)
        sample_rate_hz: Sample rate in Hz (needed for time calculation)
        format_type: Export format ("csv", "vcd", or "json")
        include_time: Whether to include time column (requires sample_rate_hz)
        
    Raises:
        ValueError: If samples and states have different lengths
        FileOperationError: If the file cannot be exported
        
    Examples:
        >>> # Export waveform to CSV
        >>> export_waveform_data(
        ...     "waveform.csv",
        ...     [0, 100, 200, 300],
        ...     [BitState.LOW, BitState.HIGH, BitState.LOW, BitState.HIGH],
        ...     sample_rate_hz=1_000_000,
        ...     format_type="csv"
        ... )
    """
    if len(samples) != len(states):
        raise ValueError("Number of samples must match number of states")
    
    if include_time and sample_rate_hz is None:
        raise ValueError("Sample rate must be specified to include time column")
    
    try:
        if format_type.lower() == "csv":
            # Export as CSV
            with open(file_name, "w") as f:
                # Write header
                header = "Sample"
                if include_time:
                    header += ",Time (s)"
                header += ",State"
                f.write(header + "\n")
                
                # Write data rows
                for i, (sample, state) in enumerate(zip(samples, states)):
                    row = str(sample)
                    if include_time:
                        time_s = float(sample) / sample_rate_hz
                        row += f",{time_s:.9f}"
                    row += f",{1 if state == BitState.HIGH else 0}"
                    f.write(row + "\n")
        
        elif format_type.lower() == "vcd":
            # Export as Value Change Dump (VCD) format
            with open(file_name, "w") as f:
                # Write VCD header
                f.write("$date\n")
                from datetime import datetime
                f.write(f"  {datetime.now().isoformat()}\n")
                f.write("$end\n")
                f.write("$version\n")
                f.write("  Kingst Logic Analyzer Export\n")
                f.write("$end\n")
                f.write("$timescale\n")
                if sample_rate_hz is not None:
                    time_unit = "1s"
                    if sample_rate_hz >= 1e9:
                        time_unit = "1ps"
                    elif sample_rate_hz >= 1e6:
                        time_unit = "1ns"
                    elif sample_rate_hz >= 1e3:
                        time_unit = "1us"
                    else:
                        time_unit = "1ms"
                else:
                    time_unit = "1ns"
                f.write(f"  {time_unit}\n")
                f.write("$end\n")
                
                # Define the signal
                f.write("$scope module logic $end\n")
                f.write("$var wire 1 # signal $end\n")
                f.write("$upscope $end\n")
                f.write("$enddefinitions $end\n")
                
                # Write initial value
                f.write("$dumpvars\n")
                f.write(f"{'1' if states[0] == BitState.HIGH else '0'}#\n")
                f.write("$end\n")
                
                # Write value changes
                prev_state = states[0]
                for sample, state in zip(samples[1:], states[1:]):
                    if state != prev_state:
                        # Convert sample to time based on sample rate
                        if sample_rate_hz is not None:
                            time_value = sample
                            if time_unit == "1ps":
                                time_value *= 1e12 / sample_rate_hz
                            elif time_unit == "1ns":
                                time_value *= 1e9 / sample_rate_hz
                            elif time_unit == "1us":
                                time_value *= 1e6 / sample_rate_hz
                            elif time_unit == "1ms":
                                time_value *= 1e3 / sample_rate_hz
                            else:
                                time_value /= sample_rate_hz
                        else:
                            time_value = sample
                        
                        f.write(f"#{int(time_value)}\n")
                        f.write(f"{'1' if state == BitState.HIGH else '0'}#\n")
                        prev_state = state
        
        elif format_type.lower() == "json":
            # Export as JSON
            import json
            data = {
                "samples": samples,
                "states": [1 if state == BitState.HIGH else 0 for state in states]
            }
            if sample_rate_hz is not None:
                data["sample_rate_hz"] = sample_rate_hz
                if include_time:
                    data["times"] = [float(sample) / sample_rate_hz for sample in samples]
            
            with open(file_name, "w") as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    except Exception as e:
        raise FileOperationError(f"Error exporting waveform data: {e}")


def read_from_file(file_name: str, is_binary: bool = False) -> Union[str, bytes]:
    """
    Read data from a file.
    
    Args:
        file_name: Name of the file to read
        is_binary: True for binary file, False for text file
        
    Returns:
        File contents as string (text) or bytes (binary)
        
    Raises:
        FileOperationError: If the file cannot be read
        
    Examples:
        >>> # Read text file
        >>> content = read_from_file("data.txt")
        >>> print(content)
        'Hello, world!'
        
        >>> # Read binary file
        >>> data = read_from_file("data.bin", is_binary=True)
        >>> print(list(data[:4]))
        [0, 1, 2, 3]
    """
    try:
        mode = "rb" if is_binary else "r"
        with open(file_name, mode) as f:
            return f.read()
    except Exception as e:
        raise FileOperationError(f"Error reading file: {e}")


# ============================================================================
# Bit Conversion Functions
# ============================================================================

def bits_to_bytes(bits: List[BitState]) -> bytes:
    """
    Convert a list of BitState values to bytes.
    
    Args:
        bits: List of BitState values (BitState.HIGH or BitState.LOW)
        
    Returns:
        Bytes representation (MSB first)
        
    Raises:
        ValueError: If bits is empty
        
    Examples:
        >>> # Convert 8 bits to a byte
        >>> bits_to_bytes([
        ...     BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
        ...     BitState.LOW, BitState.LOW, BitState.HIGH, BitState.HIGH
        ... ])
        b'\\xa3'  # 10100011
    """
    if not bits:
        raise ValueError("Bits list cannot be empty")
    
    result = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        # Pad with zeros if needed
        while len(chunk) < 8:
            chunk.append(BitState.LOW)
        
        byte_val = 0
        for j, bit in enumerate(chunk):
            if bit == BitState.HIGH:
                byte_val |= (1 << (7 - j))
        
        result.append(byte_val)
    
    return bytes(result)


def bytes_to_bits(data: bytes) -> List[BitState]:
    """
    Convert bytes to a list of BitState values.
    
    Args:
        data: Bytes to convert
        
    Returns:
        List of BitState values (BitState.HIGH or BitState.LOW)
        
    Examples:
        >>> # Convert a byte to 8 bits
        >>> bytes_to_bits(b'\\xa3')  # 10100011
        [BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
         BitState.LOW, BitState.LOW, BitState.HIGH, BitState.HIGH]
    """
    result = []
    for byte in data:
        for i in range(7, -1, -1):  # MSB first
            bit = (byte >> i) & 1
            result.append(BitState.HIGH if bit else BitState.LOW)
    
    return result


def extract_bits(value: int, shift_order: ShiftOrder, num_bits: int) -> List[BitState]:
    """
    Extract individual bits from a value.
    
    Args:
        value: Integer to extract bits from
        shift_order: Order to extract bits (MSBFirst or LSBFirst)
        num_bits: Number of bits to extract
        
    Returns:
        List of BitState values
        
    Raises:
        ValueError: If num_bits is less than 1 or greater than 64
        
    Examples:
        >>> # Extract bits from 0xA5 (10100101) MSB first
        >>> extract_bits(0xA5, ShiftOrder.MSBFirst, 8)
        [BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
         BitState.LOW, BitState.HIGH, BitState.LOW, BitState.HIGH]
    """
    if num_bits < 1 or num_bits > 64:
        raise ValueError("Number of bits must be between 1 and 64")
    
    extractor = BitExtractor(value, shift_order, num_bits)
    return [extractor.get_next_bit() for _ in range(num_bits)]


def build_value(bits: List[BitState], shift_order: ShiftOrder) -> int:
    """
    Build a value from a list of bits.
    
    Args:
        bits: List of BitState values
        shift_order: Order to add bits (MSBFirst or LSBFirst)
        
    Returns:
        Integer value constructed from bits
        
    Raises:
        ValueError: If bits is empty
        
    Examples:
        >>> # Build a value from bits (10100101 = 0xA5)
        >>> build_value([
        ...     BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
        ...     BitState.LOW, BitState.HIGH, BitState.LOW, BitState.HIGH
        ... ], ShiftOrder.MSBFirst)
        165  # 0xA5
    """
    if not bits:
        raise ValueError("Bits list cannot be empty")
    
    builder = DataBuilder()
    data = 0
    builder.reset(data, shift_order, len(bits))
    
    for bit in bits:
        builder.add_bit(bit)
    
    return data


def bit_slice(value: int, start_bit: int, num_bits: int, msb_order: bool = True) -> int:
    """
    Extract a slice of bits from an integer value.
    
    Args:
        value: Integer to extract bits from
        start_bit: Starting bit position (0-based)
        num_bits: Number of bits to extract
        msb_order: If True, bit 0 is the MSB; if False, bit 0 is the LSB
        
    Returns:
        Integer containing the extracted bits
        
    Raises:
        ValueError: If start_bit or num_bits is negative
        
    Examples:
        >>> # Extract 4 bits (5-8) from 0xA5 (10100101) in MSB order
        >>> bit_slice(0xA5, 4, 4, msb_order=True)
        5  # 0101
    """
    if start_bit < 0:
        raise ValueError("Start bit cannot be negative")
    if num_bits < 1:
        raise ValueError("Number of bits must be at least 1")
    
    if msb_order:
        shift = 8 * ((value.bit_length() + 7) // 8) - start_bit - num_bits
        if shift < 0:
            raise ValueError("Start bit and num_bits exceed the value's bit length")
    else:
        shift = start_bit
    
    mask = (1 << num_bits) - 1
    return (value >> shift) & mask


def reverse_bits(value: int, num_bits: int) -> int:
    """
    Reverse the bit order of an integer.
    
    Args:
        value: Integer to reverse
        num_bits: Number of bits to consider
        
    Returns:
        Integer with reversed bit order
        
    Raises:
        ValueError: If num_bits is less than 1 or greater than 64
        
    Examples:
        >>> # Reverse 0xA5 (10100101) -> 10100101
        >>> reverse_bits(0xA5, 8)
        165  # 0xA5 (no change because it's symmetric)
        
        >>> # Reverse 0x12 (00010010) -> 01001000
        >>> reverse_bits(0x12, 8)
        72  # 0x48
    """
    if num_bits < 1 or num_bits > 64:
        raise ValueError("Number of bits must be between 1 and 64")
    
    result = 0
    for i in range(num_bits):
        if (value >> i) & 1:
            result |= 1 << (num_bits - 1 - i)
    
    return result


def count_transitions(bits: List[BitState]) -> int:
    """
    Count the number of transitions (edges) in a sequence of bits.
    
    Args:
        bits: List of BitState values
        
    Returns:
        Number of transitions between HIGH and LOW states
        
    Examples:
        >>> # Count transitions in 10101010
        >>> count_transitions([
        ...     BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
        ...     BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW
        ... ])
        7
    """
    if not bits:
        return 0
    
    transitions = 0
    prev_bit = bits[0]
    
    for bit in bits[1:]:
        if bit != prev_bit:
            transitions += 1
            prev_bit = bit
    
    return transitions


def find_edges(bits: List[BitState]) -> List[int]:
    """
    Find all edge positions in a sequence of bits.
    
    Args:
        bits: List of BitState values
        
    Returns:
        List of indices where transitions occur
        
    Examples:
        >>> # Find edges in 10101010
        >>> find_edges([
        ...     BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
        ...     BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW
        ... ])
        [1, 2, 3, 4, 5, 6, 7]
    """
    if not bits:
        return []
    
    edges = []
    prev_bit = bits[0]
    
    for i, bit in enumerate(bits[1:], 1):
        if bit != prev_bit:
            edges.append(i)
            prev_bit = bit
    
    return edges


# ============================================================================
# Data Transformation Functions
# ============================================================================

def group_by_n(data: List[T], n: int) -> List[List[T]]:
    """
    Group elements in a list by groups of n.
    
    Args:
        data: List of elements to group
        n: Group size
        
    Returns:
        List of groups, where each group has n elements (except possibly the last one)
        
    Raises:
        ValueError: If n is less than 1
        
    Examples:
        >>> group_by_n([1, 2, 3, 4, 5, 6, 7], 3)
        [[1, 2, 3], [4, 5, 6], [7]]
    """
    if n < 1:
        raise ValueError("Group size must be at least 1")
    
    return [data[i:i+n] for i in range(0, len(data), n)]


def interleave(*lists: List[Any]) -> List[Any]:
    """
    Interleave elements from multiple lists.
    
    Args:
        *lists: Lists to interleave
        
    Returns:
        List with interleaved elements
        
    Examples:
        >>> interleave([1, 3, 5], [2, 4, 6])
        [1, 2, 3, 4, 5, 6]
        
        >>> interleave([1, 4], [2, 5], [3, 6])
        [1, 2, 3, 4, 5, 6]
    """
    if not lists:
        return []
    
    result = []
    min_len = min(len(lst) for lst in lists)
    
    # Interleave up to the shortest list length
    for i in range(min_len):
        for lst in lists:
            result.append(lst[i])
    
    # Append remaining elements from longer lists
    for i, lst in enumerate(lists):
        if len(lst) > min_len:
            result.extend(lst[min_len:])
    
    return result


def deinterleave(data: List[Any], n_streams: int) -> List[List[Any]]:
    """
    Split an interleaved list into n separate streams.
    
    Args:
        data: Interleaved data
        n_streams: Number of streams to extract
        
    Returns:
        List of n lists, each containing one stream
        
    Raises:
        ValueError: If n_streams is less than 1
        
    Examples:
        >>> deinterleave([1, 2, 3, 4, 5, 6], 2)
        [[1, 3, 5], [2, 4, 6]]
        
        >>> deinterleave([1, 2, 3, 4, 5, 6], 3)
        [[1, 4], [2, 5], [3, 6]]
    """
    if n_streams < 1:
        raise ValueError("Number of streams must be at least 1")
    
    result = [[] for _ in range(n_streams)]
    
    for i, item in enumerate(data):
        result[i % n_streams].append(item)
    
    return result


def smooth_signal(data: List[float], window_size: int = 3, method: str = "moving_avg") -> List[float]:
    """
    Smooth a signal using various methods.
    
    Args:
        data: List of values to smooth
        window_size: Size of the smoothing window
        method: Smoothing method:
            - "moving_avg": Simple moving average
            - "weighted_avg": Weighted moving average (triangular window)
            - "median": Median filter
        
    Returns:
        List of smoothed values
        
    Raises:
        ValueError: If window_size is less than 1 or method is invalid
        
    Examples:
        >>> smooth_signal([1, 5, 3, 8, 2, 9, 7], window_size=3)
        [1.0, 3.0, 5.33, 4.33, 6.33, 6.0, 7.0]
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    
    if not data:
        return []
    
    # If window_size is 1, no smoothing needed
    if window_size == 1:
        return list(data)
    
    valid_methods = {"moving_avg", "weighted_avg", "median"}
    if method not in valid_methods:
        raise ValueError(f"Invalid smoothing method: {method}. "
                         f"Valid methods are: {', '.join(valid_methods)}")
    
    result = []
    half_window = window_size // 2
    
    # Handle numpy availability
    if method == "median" and not _HAS_NUMPY:
        warnings.warn("NumPy not available, falling back to simple moving average")
        method = "moving_avg"
    
    if method == "moving_avg":
        # Simple moving average
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window = data[start:end]
            result.append(sum(window) / len(window))
    
    elif method == "weighted_avg":
        # Weighted moving average (triangular window)
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window = data[start:end]
            
            # Generate triangular weights
            weights = []
            center = i - start
            for j in range(len(window)):
                weight = 1.0 - abs(j - center) / (half_window + 1)
                weights.append(max(0.0, weight))
            
            # Normalize weights
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
            else:
                weights = [1.0 / len(window)] * len(window)
            
            # Calculate weighted average
            result.append(sum(w * v for w, v in zip(weights, window)))
    
    elif method == "median":
        # Median filter (requires NumPy)
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window = data[start:end]
            result.append(float(np.median(window)))
    
    return result


# Alias for smooth_signal with moving average method
def moving_average(data: List[float], window_size: int = 3) -> List[float]:
    """
    Apply a simple moving average filter to a signal.
    
    Args:
        data: List of values to smooth
        window_size: Size of the moving average window
        
    Returns:
        List of smoothed values
        
    Examples:
        >>> moving_average([1, 5, 3, 8, 2, 9, 7], window_size=3)
        [1.0, 3.0, 5.33, 4.33, 6.33, 6.0, 7.0]
    """
    return smooth_signal(data, window_size, method="moving_avg")


def downsample(data: List[Any], factor: int, method: str = "decimate") -> List[Any]:
    """
    Downsample a signal by keeping only every nth sample.
    
    Args:
        data: List of values to downsample
        factor: Downsampling factor (keep every nth sample)
        method: Downsampling method:
            - "decimate": Keep every nth sample
            - "average": Average groups of n samples
            - "max": Keep maximum value in each group
            - "min": Keep minimum value in each group
        
    Returns:
        Downsampled list
        
    Raises:
        ValueError: If factor is less than 1 or method is invalid
        
    Examples:
        >>> downsample([1, 2, 3, 4, 5, 6, 7, 8], factor=2)
        [1, 3, 5, 7]
        
        >>> downsample([1, 2, 3, 4, 5, 6, 7, 8], factor=2, method="average")
        [1.5, 3.5, 5.5, 7.5]
    """
    if factor < 1:
        raise ValueError("Downsampling factor must be at least 1")
    
    if factor == 1:
        return list(data)
    
    valid_methods = {"decimate", "average", "max", "min"}
    if method not in valid_methods:
        raise ValueError(f"Invalid downsampling method: {method}. "
                         f"Valid methods are: {', '.join(valid_methods)}")
    
    if method == "decimate":
        # Keep every nth sample
        return data[::factor]
    
    # Group data into chunks of size 'factor'
    chunks = group_by_n(data, factor)
    
    if method == "average":
        # Average each group
        return [sum(chunk) / len(chunk) for chunk in chunks]
    
    elif method == "max":
        # Keep maximum value in each group
        return [max(chunk) for chunk in chunks]
    
    elif method == "min":
        # Keep minimum value in each group
        return [min(chunk) for chunk in chunks]
    
    # Should never get here due to validation above
    return []


def upsample(data: List[Any], factor: int, method: str = "repeat") -> List[Any]:
    """
    Upsample a signal by inserting additional samples.
    
    Args:
        data: List of values to upsample
        factor: Upsampling factor (number of samples to insert between original samples)
        method: Upsampling method:
            - "repeat": Repeat each sample
            - "zero": Insert zeros between samples
            - "linear": Linear interpolation between samples
        
    Returns:
        Upsampled list
        
    Raises:
        ValueError: If factor is less than 1 or method is invalid
        
    Examples:
        >>> upsample([1, 3, 5], factor=2, method="repeat")
        [1, 1, 1, 3, 3, 3, 5, 5, 5]
        
        >>> upsample([1, 3, 5], factor=2, method="zero")
        [1, 0, 0, 3, 0, 0, 5, 0, 0]
        
        >>> upsample([1, 3, 5], factor=2, method="linear")
        [1, 1.67, 2.33, 3, 3.67, 4.33, 5, 5, 5]
    """
    if factor < 1:
        raise ValueError("Upsampling factor must be at least 1")
    
    if factor == 1:
        return list(data)
    
    valid_methods = {"repeat", "zero", "linear"}
    if method not in valid_methods:
        raise ValueError(f"Invalid upsampling method: {method}. "
                         f"Valid methods are: {', '.join(valid_methods)}")
    
    result = []
    
    if method == "repeat":
        # Repeat each sample factor+1 times
        for value in data:
            for _ in range(factor):
                result.append(value)
    
    elif method == "zero":
        # Insert zeros between samples
        for value in data:
            result.append(value)
            for _ in range(factor - 1):
                result.append(0)
    
    elif method == "linear":
        # Linear interpolation between samples
        for i in range(len(data) - 1):
            current = data[i]
            next_val = data[i + 1]
            result.append(current)
            
            # Insert interpolated values
            for j in range(1, factor):
                t = j / factor
                interp = current * (1 - t) + next_val * t
                result.append(interp)
        
        # Add the last value
        if data:
            result.append(data[-1])
            # Extrapolate beyond the last value if needed
            for _ in range(factor - 1):
                result.append(data[-1])
    
    return result


def threshold(data: List[float], threshold_value: float, 
             high_value: float = 1.0, low_value: float = 0.0) -> List[float]:
    """
    Apply a threshold to a signal, converting it to a binary signal.
    
    Args:
        data: List of values to threshold
        threshold_value: Threshold level
        high_value: Value to use for samples above threshold
        low_value: Value to use for samples below threshold
        
    Returns:
        Thresholded list
        
    Examples:
        >>> threshold([0.1, 0.5, 0.8, 0.3, 0.9], threshold_value=0.5)
        [0.0, 0.0, 1.0, 0.0, 1.0]
    """
    return [high_value if x >= threshold_value else low_value for x in data]


def hysteresis(data: List[float], low_threshold: float, high_threshold: float,
              high_value: float = 1.0, low_value: float = 0.0) -> List[float]:
    """
    Apply hysteresis thresholding to a signal to reduce noise.
    
    This implements a Schmitt trigger with separate rising and falling thresholds.
    
    Args:
        data: List of values to process
        low_threshold: Threshold for falling edges
        high_threshold: Threshold for rising edges
        high_value: Value to use for high state
        low_value: Value to use for low state
        
    Returns:
        Processed list with hysteresis
        
    Raises:
        ValueError: If low_threshold is not less than high_threshold
        
    Examples:
        >>> hysteresis([0.1, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2], 0.3, 0.6)
        [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    """
    if low_threshold >= high_threshold:
        raise ValueError("Low threshold must be less than high threshold")
    
    if not data:
        return []
    
    result = []
    # Start in low state
    state = low_value if data[0] <= high_threshold else high_value
    result.append(state)
    
    for value in data[1:]:
        if state == low_value and value >= high_threshold:
            # Rising edge detected
            state = high_value
        elif state == high_value and value <= low_threshold:
            # Falling edge detected
            state = low_value
        
        result.append(state)
    
    return result


# ============================================================================
# Additional helper functions for convenience
# ============================================================================

def calculate_rms(data: List[float]) -> float:
    """
    Calculate the Root Mean Square (RMS) value of a signal.
    
    Args:
        data: List of values
        
    Returns:
        RMS value
        
    Raises:
        ValueError: If data is empty
        
    Examples:
        >>> calculate_rms([1.0, 2.0, 3.0, 4.0, 5.0])
        3.3166247903554
    """
    if not data:
        raise ValueError("Cannot calculate RMS of empty data")
    
    squared_sum = sum(x * x for x in data)
    return math.sqrt(squared_sum / len(data))


def calculate_snr(signal: List[float], noise: List[float]) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        signal: List of signal values
        noise: List of noise values
        
    Returns:
        SNR in dB
        
    Raises:
        ValueError: If signal or noise is empty
        
    Examples:
        >>> calculate_snr([1.0, 2.0, 3.0], [0.1, 0.2, 0.1])
        20.0
    """
    if not signal or not noise:
        raise ValueError("Signal and noise cannot be empty")
    
    signal_power = sum(x * x for x in signal) / len(signal)
    noise_power = sum(x * x for x in noise) / len(noise)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * math.log10(signal_power / noise_power)


def find_peaks(data: List[float], 
              threshold: Optional[float] = None,
              min_distance: int = 1) -> List[int]:
    """
    Find indices of peaks in a signal.
    
    Args:
        data: List of values to search for peaks
        threshold: Minimum peak height, or None to detect all peaks
        min_distance: Minimum number of samples between peaks
        
    Returns:
        List of peak indices
        
    Examples:
        >>> find_peaks([1, 3, 2, 6, 4, 8, 5, 7, 9, 2, 1])
        [1, 3, 5, 8]
    """
    if not data:
        return []
    
    # Find all local maxima
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            if threshold is None or data[i] >= threshold:
                peaks.append(i)
    
    # Filter by minimum distance
    if min_distance > 1 and peaks:
        filtered_peaks = [peaks[0]]
        last_peak = peaks[0]
        
        for peak in peaks[1:]:
            if peak - last_peak >= min_distance:
                filtered_peaks.append(peak)
                last_peak = peak
        
        return filtered_peaks
    
    return peaks


def find_zero_crossings(data: List[float]) -> List[int]:
    """
    Find indices of zero crossings in a signal.
    
    Args:
        data: List of values to search for zero crossings
        
    Returns:
        List of indices where the signal crosses zero
        
    Examples:
        >>> find_zero_crossings([-1, -0.5, 0.2, 0.8, -0.3, -0.7, 0.1])
        [2, 4, 6]
    """
    if not data or len(data) < 2:
        return []
    
    crossings = []
    for i in range(1, len(data)):
        if (data[i-1] <= 0 and data[i] > 0) or (data[i-1] >= 0 and data[i] < 0):
            crossings.append(i)
    
    return crossings


def measure_delay(signal1: List[float], signal2: List[float], 
                 method: str = "cross_correlation") -> int:
    """
    Measure the delay between two signals.
    
    Args:
        signal1: First signal
        signal2: Second signal
        method: Method to use:
            - "cross_correlation": Use cross-correlation (requires NumPy)
            - "zero_crossings": Compare zero crossing positions
            - "peaks": Compare peak positions
        
    Returns:
        Delay in samples (positive means signal2 is delayed relative to signal1)
        
    Raises:
        ValueError: If method is invalid or signals are empty
        
    Examples:
        >>> # signal2 is delayed by 2 samples
        >>> measure_delay([1, 2, 3, 4, 5], [0, 0, 1, 2, 3, 4, 5])
        2
    """
    if not signal1 or not signal2:
        raise ValueError("Signals cannot be empty")
    
    valid_methods = {"cross_correlation", "zero_crossings", "peaks"}
    if method not in valid_methods:
        raise ValueError(f"Invalid delay measurement method: {method}. "
                         f"Valid methods are: {', '.join(valid_methods)}")
    
    if method == "cross_correlation":
        if not _HAS_NUMPY:
            raise ImportError("NumPy is required for cross-correlation method")
        
        # Compute cross-correlation
        correlation = np.correlate(signal1, signal2, mode="full")
        delay = np.argmax(correlation) - len(signal1) + 1
        
        return int(delay)
    
    elif method == "zero_crossings":
        # Compare zero crossing positions
        zc1 = find_zero_crossings(signal1)
        zc2 = find_zero_crossings(signal2)
        
        if not zc1 or not zc2:
            raise ValueError("No zero crossings found in one or both signals")
        
        # Try to match patterns of zero crossings
        delays = []
        for i in range(min(3, len(zc1))):
            for j in range(min(3, len(zc2))):
                delay = zc2[j] - zc1[i]
                if delay >= 0:
                    delays.append(delay)
        
        if not delays:
            return 0
        
        # Return most common delay
        return max(set(delays), key=delays.count)
    
    elif method == "peaks":
        # Compare peak positions
        peaks1 = find_peaks(signal1)
        peaks2 = find_peaks(signal2)
        
        if not peaks1 or not peaks2:
            raise ValueError("No peaks found in one or both signals")
        
        # Try to match patterns of peaks
        delays = []
        for i in range(min(3, len(peaks1))):
            for j in range(min(3, len(peaks2))):
                delay = peaks2[j] - peaks1[i]
                if delay >= 0:
                    delays.append(delay)
        
        if not delays:
            return 0
        
        # Return most common delay
        return max(set(delays), key=delays.count)
    
    return 0


# ============================================================================
# Visualization helpers
# ============================================================================

def plot_signal(data: List[float], 
               sample_rate_hz: Optional[int] = None,
               title: str = "Signal Plot",
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               figsize: Tuple[int, int] = (10, 6)) -> Any:
    """
    Plot a signal using matplotlib.
    
    Args:
        data: Signal values to plot
        sample_rate_hz: Sample rate in Hz (for time axis), or None to use sample numbers
        title: Plot title
        xlabel: X-axis label (auto-generated if None)
        ylabel: Y-axis label
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Matplotlib figure and axes objects
        
    Raises:
        ImportError: If matplotlib is not available
        
    Examples:
        >>> # Plot a sine wave
        >>> import math
        >>> signal = [math.sin(2 * math.pi * i / 100) for i in range(200)]
        >>> fig, ax = plot_signal(signal, sample_rate_hz=1000, title="Sine Wave")
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if sample_rate_hz is not None:
        # Use time axis
        time_axis = [i / sample_rate_hz for i in range(len(data))]
        ax.plot(time_axis, data)
        
        # Set x-axis label if not specified
        if xlabel is None:
            xlabel = "Time (s)"
    else:
        # Use sample numbers
        ax.plot(data)
        
        # Set x-axis label if not specified
        if xlabel is None:
            xlabel = "Sample"
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    return fig, ax


def plot_digital_signal(data: Union[List[BitState], List[int]], 
                       sample_rate_hz: Optional[int] = None,
                       title: str = "Digital Signal",
                       figsize: Tuple[int, int] = (10, 3)) -> Any:
    """
    Plot a digital signal as a waveform.
    
    Args:
        data: List of BitState values or integers (0/1)
        sample_rate_hz: Sample rate in Hz (for time axis), or None to use sample numbers
        title: Plot title
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Matplotlib figure and axes objects
        
    Raises:
        ImportError: If matplotlib is not available
        
    Examples:
        >>> # Plot a clock signal
        >>> clock = [BitState.HIGH, BitState.HIGH, BitState.LOW, BitState.LOW] * 5
        >>> fig, ax = plot_digital_signal(clock, title="Clock Signal")
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for plotting")
    
    # Convert data to integers if they are BitState
    if data and isinstance(data[0], BitState):
        int_data = [1 if bit == BitState.HIGH else 0 for bit in data]
    else:
        int_data = data
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if sample_rate_hz is not None:
        # Use time axis
        time_axis = [i / sample_rate_hz for i in range(len(int_data))]
        ax.step(time_axis, int_data, where="post")
        ax.set_xlabel("Time (s)")
    else:
        # Use sample numbers
        ax.step(range(len(int_data)), int_data, where="post")
        ax.set_xlabel("Sample")
    
    # Customize appearance
    ax.set_title(title)
    ax.set_ylabel("State")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LOW", "HIGH"])
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    return fig, ax


def plot_fft(data: List[float], 
            sample_rate_hz: int,
            title: str = "Frequency Spectrum",
            figsize: Tuple[int, int] = (10, 6)) -> Any:
    """
    Plot the frequency spectrum of a signal using FFT.
    
    Args:
        data: Signal values to analyze
        sample_rate_hz: Sample rate in Hz
        title: Plot title
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Matplotlib figure and axes objects
        
    Raises:
        ImportError: If matplotlib or NumPy is not available
        ValueError: If sample_rate_hz is less than or equal to 0
        
    Examples:
        >>> # Plot the spectrum of a 1kHz sine wave
        >>> import math
        >>> sample_rate = 44100
        >>> frequency = 1000
        >>> duration = 0.1  # seconds
        >>> samples = int(sample_rate * duration)
        >>> signal = [math.sin(2 * math.pi * frequency * i / sample_rate) for i in range(samples)]
        >>> fig, ax = plot_fft(signal, sample_rate_hz=sample_rate)
    """
    if not _HAS_MATPLOTLIB or not _HAS_NUMPY:
        raise ImportError("Matplotlib and NumPy are required for FFT plotting")
    
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be greater than 0")
    
    # Compute FFT
    fft_result = np.fft.rfft(data)
    fft_magnitude = np.abs(fft_result)
    
    # Create frequency axis
    freq_axis = np.fft.rfftfreq(len(data), 1 / sample_rate_hz)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq_axis, fft_magnitude)
    
    # Customize appearance
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    return fig, ax