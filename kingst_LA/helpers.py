"""
Utility functions for working with the Kingst Logic Analyzer.

This module provides helper functions for common operations when working with
logic analyzer data, including formatting, bit manipulation, timing conversions,
and data transformation utilities.
"""

from typing import List, Tuple, Dict, Optional, Union, BinaryIO, Any
import math
import io
import struct
from enum import Enum

# Import the underlying C++ bindings
from kingst_analyzer._kingst_analyzer import (
    AnalyzerHelpers as _AnalyzerHelpers,
    BitExtractor, BitState, DataBuilder, SimpleArchive, ClockGenerator,
    DisplayBase, ShiftOrder
)


# Re-export the helper classes
__all__ = [
    # Bit manipulation
    'is_even', 'is_odd', 'get_ones_count', 'diff32', 
    'bit_count', 'convert_to_signed', 'convert_from_signed',
    
    # Formatting
    'format_number', 'format_time', 'format_value',
    
    # Timing
    'samples_to_seconds', 'seconds_to_samples', 
    'time_delta', 'phase_to_samples',
    
    # File operations
    'save_to_file', 'export_data', 'export_csv',
    
    # Bit conversion
    'bits_to_bytes', 'bytes_to_bits', 'extract_bits',
    'build_value', 'bit_slice', 'reverse_bits',
    
    # Convenience data transformation
    'group_by_n', 'interleave', 'deinterleave',
    
    # Re-exports
    'BitExtractor', 'BitState', 'DataBuilder', 
    'SimpleArchive', 'ClockGenerator',
    'DisplayBase', 'ShiftOrder'
]


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
        
    Examples:
        >>> convert_to_signed(0xFF, 8)  # 11111111 in 8 bits = -1
        -1
        >>> convert_to_signed(0x80, 8)  # 10000000 in 8 bits = -128
        -128
    """
    return int(_AnalyzerHelpers.convert_to_signed_number(value, num_bits))


def convert_from_signed(value: int, num_bits: int) -> int:
    """
    Convert a signed number to an unsigned number with the specified bit width.
    
    Args:
        value: Signed integer to convert
        num_bits: Number of bits in the value
        
    Returns:
        Unsigned representation of the number
        
    Examples:
        >>> convert_from_signed(-1, 8)  # -1 in 8 bits = 11111111 = 0xFF
        255
        >>> convert_from_signed(-128, 8)  # -128 in 8 bits = 10000000 = 0x80
        128
    """
    if value < 0:
        # Compute two's complement
        mask = (1 << num_bits) - 1
        return ((~abs(value) + 1) & mask)
    else:
        return value


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
        
    Examples:
        >>> format_number(42, DisplayBase.Decimal, 8)
        '42'
        >>> format_number(42, DisplayBase.Hexadecimal, 8)
        '0x2A'
        >>> format_number(42, DisplayBase.Binary, 8)
        '00101010'
    """
    return _AnalyzerHelpers.format_number(value, display_base, num_bits)


def format_time(sample: int, trigger_sample: int, sample_rate_hz: int) -> str:
    """
    Format a sample number as a time string.
    
    Args:
        sample: Sample number to format
        trigger_sample: Trigger sample number (reference point)
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        Formatted time string
        
    Examples:
        >>> format_time(1000, 0, 1_000_000)  # 1000 samples at 1MHz
        '+1.00 ms'
        >>> format_time(0, 1000, 1_000_000)  # 0 samples with trigger at 1000
        '-1.00 ms'
    """
    return _AnalyzerHelpers.format_time(sample, trigger_sample, sample_rate_hz)


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
        # Engineering notation with SI prefix
        if value == 0:
            return "0"
        
        prefixes = {
            -24: "y", -21: "z", -18: "a", -15: "f", -12: "p", -9: "n",
            -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G",
            12: "T", 15: "P", 18: "E", 21: "Z", 24: "Y"
        }
        
        exponent = int(math.floor(math.log10(abs(value)) / 3) * 3)
        exponent = max(-24, min(24, exponent))  # Limit to standard SI prefixes
        
        mantissa = value / (10 ** exponent)
        return f"{mantissa:.2f} {prefixes[exponent]}"
    
    else:
        return str(value)


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
        
    Examples:
        >>> samples_to_seconds(1000, 1_000_000)  # 1000 samples at 1MHz
        0.001  # 1ms
    """
    return float(samples) / float(sample_rate_hz)


def seconds_to_samples(seconds: float, sample_rate_hz: int) -> int:
    """
    Convert time in seconds to samples at the given sample rate.
    
    Args:
        seconds: Time in seconds
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        Number of samples (rounded to nearest integer)
        
    Examples:
        >>> seconds_to_samples(0.001, 1_000_000)  # 1ms at 1MHz
        1000
    """
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
        
    Examples:
        >>> time_delta(1000, 1500, 1_000_000)  # 500 samples at 1MHz
        0.0005  # 0.5ms
        >>> time_delta(1500, 1000, 1_000_000)  # -500 samples at 1MHz
        -0.0005  # -0.5ms
    """
    return samples_to_seconds(sample2 - sample1, sample_rate_hz)


def phase_to_samples(phase_deg: float, period_samples: int) -> int:
    """
    Convert a phase in degrees to an equivalent number of samples.
    
    Args:
        phase_deg: Phase in degrees (0-360)
        period_samples: Number of samples in one complete period
        
    Returns:
        Number of samples corresponding to the phase
        
    Examples:
        >>> phase_to_samples(90, 100)  # 90° phase in a 100-sample period
        25
        >>> phase_to_samples(180, 100)  # 180° phase in a 100-sample period
        50
    """
    return round((phase_deg % 360) * period_samples / 360.0)


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
        
    Examples:
        >>> save_to_file("data.txt", b"Hello, world!")
        >>> save_to_file("data.bin", [0x00, 0x01, 0x02, 0x03], is_binary=True)
    """
    # Convert list of integers to bytes if needed
    if isinstance(data, list):
        data = bytes(data)
    
    _AnalyzerHelpers.save_file(file_name, data, is_binary)


def export_data(file_name: str, 
                data: Any,
                format_func: Optional[callable] = None,
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
    file = _AnalyzerHelpers.start_file(file_name, is_binary)
    
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
        
    Examples:
        >>> # Export a list of rows with headers
        >>> export_csv(
        ...     "data.csv",
        ...     [[1, "John", 25], [2, "Alice", 30]],
        ...     ["ID", "Name", "Age"]
        ... )
    """
    file = _AnalyzerHelpers.start_file(file_name, False)
    
    try:
        # Write headers if provided
        if headers:
            header_line = delimiter.join(str(h) for h in headers) + "\n"
            _AnalyzerHelpers.append_to_file(header_line.encode("utf-8"), file)
        
        # Write data rows
        for row in data:
            line = delimiter.join(str(cell) for cell in row) + "\n"
            _AnalyzerHelpers.append_to_file(line.encode("utf-8"), file)
    finally:
        _AnalyzerHelpers.end_file(file)


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
        
    Examples:
        >>> # Convert 8 bits to a byte
        >>> bits_to_bytes([
        ...     BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
        ...     BitState.LOW, BitState.LOW, BitState.HIGH, BitState.HIGH
        ... ])
        b'\\xa3'  # 10100011
    """
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
        
    Examples:
        >>> # Extract bits from 0xA5 (10100101) MSB first
        >>> extract_bits(0xA5, ShiftOrder.MSBFirst, 8)
        [BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
         BitState.LOW, BitState.HIGH, BitState.LOW, BitState.HIGH]
    """
    extractor = BitExtractor(value, shift_order, num_bits)
    return extractor.get_all_bits(num_bits)


def build_value(bits: List[BitState], shift_order: ShiftOrder) -> int:
    """
    Build a value from a list of bits.
    
    Args:
        bits: List of BitState values
        shift_order: Order to add bits (MSBFirst or LSBFirst)
        
    Returns:
        Integer value constructed from bits
        
    Examples:
        >>> # Build a value from bits (10100101 = 0xA5)
        >>> build_value([
        ...     BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW,
        ...     BitState.LOW, BitState.HIGH, BitState.LOW, BitState.HIGH
        ... ], ShiftOrder.MSBFirst)
        165  # 0xA5
    """
    builder = DataBuilder()
    return builder.build_from_bits(bits, shift_order)


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
        
    Examples:
        >>> # Extract 4 bits (5-8) from 0xA5 (10100101) in MSB order
        >>> bit_slice(0xA5, 4, 4, msb_order=True)
        5  # 0101
    """
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
        
    Examples:
        >>> # Reverse 0xA5 (10100101) -> 10100101
        >>> reverse_bits(0xA5, 8)
        165  # 0xA5 (no change because it's symmetric)
        
        >>> # Reverse 0x12 (00010010) -> 01001000
        >>> reverse_bits(0x12, 8)
        72  # 0x48
    """
    result = 0
    for i in range(num_bits):
        if (value >> i) & 1:
            result |= 1 << (num_bits - 1 - i)
    
    return result


# ============================================================================
# Data Transformation Functions
# ============================================================================

def group_by_n(data: List[Any], n: int) -> List[List[Any]]:
    """
    Group elements in a list by groups of n.
    
    Args:
        data: List of elements to group
        n: Group size
        
    Returns:
        List of groups, where each group has n elements (except possibly the last one)
        
    Examples:
        >>> group_by_n([1, 2, 3, 4, 5, 6, 7], 3)
        [[1, 2, 3], [4, 5, 6], [7]]
    """
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
        
    Examples:
        >>> deinterleave([1, 2, 3, 4, 5, 6], 2)
        [[1, 3, 5], [2, 4, 6]]
        
        >>> deinterleave([1, 2, 3, 4, 5, 6], 3)
        [[1, 4], [2, 5], [3, 6]]
    """
    result = [[] for _ in range(n_streams)]
    
    for i, item in enumerate(data):
        result[i % n_streams].append(item)
    
    return result