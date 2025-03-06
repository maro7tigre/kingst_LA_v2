"""
Unit tests for the kingst_analyzer.types module.

These tests verify the basic types, enums, and utility functions in the types module 
without requiring hardware connection.
"""

import pytest
import sys
from pathlib import Path
from typing import List, Union, Any

# Add the package root to the Python path if running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module to test
from kingst_analyzer.types import (
    # Enums
    BitState, DisplayBase, AnalyzerEnums,
    
    # Classes
    Channel,
    
    # Constants
    UNDEFINED_CHANNEL,
    
    # Utility functions
    toggle_bit, invert_bit, bits_to_int, int_to_bits, check_parity, count_ones,
    format_value, validate_int_range, validate_u8, validate_s8, validate_u16, 
    validate_s16, validate_u32, validate_s32, validate_u64, validate_s64,
    bytes_to_value, value_to_bytes,
    
    # Integer boundaries
    S8_MIN, S8_MAX, U8_MAX,
    S16_MIN, S16_MAX, U16_MAX,
    S32_MIN, S32_MAX, U32_MAX,
    S64_MIN, S64_MAX, U64_MAX
)


class TestBitState:
    """Tests for the BitState enum implementation."""
    
    def test_bitstate_values(self):
        """Test that BitState enum has the correct values."""
        assert BitState.LOW == 0, "BitState.LOW should equal 0"
        assert BitState.HIGH == 1, "BitState.HIGH should equal 1"
    
    def test_bitstate_str(self):
        """Test string representation of BitState values."""
        assert str(BitState.LOW) == "LOW", "String representation of BitState.LOW should be 'LOW'"
        assert str(BitState.HIGH) == "HIGH", "String representation of BitState.HIGH should be 'HIGH'"
    
    def test_bitstate_repr(self):
        """Test repr of BitState values."""
        assert repr(BitState.LOW) == "BitState.LOW", "Repr of BitState.LOW incorrect"
        assert repr(BitState.HIGH) == "BitState.HIGH", "Repr of BitState.HIGH incorrect"
    
    def test_bitstate_integer_comparison(self):
        """Test that BitState can be compared with integers."""
        assert BitState.LOW == 0, "BitState.LOW should compare equal to 0"
        assert BitState.HIGH == 1, "BitState.HIGH should compare equal to 1"
        assert BitState.LOW != 1, "BitState.LOW should not equal 1"
        assert BitState.HIGH != 0, "BitState.HIGH should not equal 0"
    
    def test_bitstate_enum_comparison(self):
        """Test that BitState values can be compared with each other."""
        assert BitState.LOW != BitState.HIGH, "BitState.LOW should not equal BitState.HIGH"
        assert BitState.LOW == BitState.LOW, "BitState.LOW should equal itself"
        assert BitState.HIGH == BitState.HIGH, "BitState.HIGH should equal itself"


class TestDisplayBase:
    """Tests for the DisplayBase enum implementation."""
    
    def test_displaybase_values(self):
        """Test that DisplayBase enum has the correct values."""
        assert DisplayBase.Binary == 0, "DisplayBase.Binary should equal 0"
        assert DisplayBase.Decimal == 1, "DisplayBase.Decimal should equal 1"
        assert DisplayBase.Hexadecimal == 2, "DisplayBase.Hexadecimal should equal 2"
        assert DisplayBase.ASCII == 3, "DisplayBase.ASCII should equal 3"
        assert DisplayBase.AsciiHex == 4, "DisplayBase.AsciiHex should equal 4"
    
    def test_displaybase_str(self):
        """Test string representation of DisplayBase values."""
        assert str(DisplayBase.Binary) == "Binary", "String representation incorrect"
        assert str(DisplayBase.Decimal) == "Decimal", "String representation incorrect"
        assert str(DisplayBase.Hexadecimal) == "Hexadecimal", "String representation incorrect"
        assert str(DisplayBase.ASCII) == "ASCII", "String representation incorrect"
        assert str(DisplayBase.AsciiHex) == "AsciiHex", "String representation incorrect"
    
    def test_displaybase_format_value_binary(self):
        """Test formatting values in binary display base."""
        assert DisplayBase.Binary.format_value(10, 8) == "00001010", "Binary formatting incorrect"
        assert DisplayBase.Binary.format_value(15, 4) == "1111", "Binary formatting with exact bits incorrect"
        assert DisplayBase.Binary.format_value(255, 8) == "11111111", "Binary formatting max value incorrect"
    
    def test_displaybase_format_value_decimal(self):
        """Test formatting values in decimal display base."""
        assert DisplayBase.Decimal.format_value(10) == "10", "Decimal formatting incorrect"
        assert DisplayBase.Decimal.format_value(0) == "0", "Decimal formatting zero incorrect"
        assert DisplayBase.Decimal.format_value(255) == "255", "Decimal formatting max value incorrect"
    
    def test_displaybase_format_value_hexadecimal(self):
        """Test formatting values in hexadecimal display base."""
        assert DisplayBase.Hexadecimal.format_value(10, 8) == "0x0A", "Hex formatting incorrect"
        assert DisplayBase.Hexadecimal.format_value(15, 4) == "0xF", "Hex formatting with exact bits incorrect"
        assert DisplayBase.Hexadecimal.format_value(255, 8) == "0xFF", "Hex formatting max value incorrect"
        assert DisplayBase.Hexadecimal.format_value(0x1234, 16) == "0x1234", "Hex formatting 16-bit incorrect"
    
    def test_displaybase_format_value_ascii(self):
        """Test formatting values in ASCII display base."""
        assert DisplayBase.ASCII.format_value(65) == "A", "ASCII formatting for 'A' incorrect"
        assert DisplayBase.ASCII.format_value(97) == "a", "ASCII formatting for 'a' incorrect"
        assert DisplayBase.ASCII.format_value(0) == ".", "ASCII formatting for non-printable incorrect"
        assert DisplayBase.ASCII.format_value(127) == ".", "ASCII formatting for DEL incorrect"
    
    def test_displaybase_format_value_asciihex(self):
        """Test formatting values in AsciiHex display base."""
        assert DisplayBase.AsciiHex.format_value(65) == "A", "AsciiHex formatting for 'A' incorrect"
        assert DisplayBase.AsciiHex.format_value(0) == "<00>", "AsciiHex formatting for NUL incorrect"
        assert DisplayBase.AsciiHex.format_value(31) == "<1F>", "AsciiHex formatting for Unit Separator incorrect"


class TestAnalyzerEnums:
    """Tests for the AnalyzerEnums namespace and its contained enums."""
    
    def test_shift_order_enum(self):
        """Test ShiftOrder enum values and string representations."""
        assert AnalyzerEnums.ShiftOrder.MsbFirst == 0, "MsbFirst value incorrect"
        assert AnalyzerEnums.ShiftOrder.LsbFirst == 1, "LsbFirst value incorrect"
        assert str(AnalyzerEnums.ShiftOrder.MsbFirst) == "MSB First", "String representation incorrect"
        assert str(AnalyzerEnums.ShiftOrder.LsbFirst) == "LSB First", "String representation incorrect"
    
    def test_edge_direction_enum(self):
        """Test EdgeDirection enum values and string representations."""
        assert AnalyzerEnums.EdgeDirection.PosEdge == 0, "PosEdge value incorrect"
        assert AnalyzerEnums.EdgeDirection.NegEdge == 1, "NegEdge value incorrect"
        assert str(AnalyzerEnums.EdgeDirection.PosEdge) == "Rising Edge", "String representation incorrect"
        assert str(AnalyzerEnums.EdgeDirection.NegEdge) == "Falling Edge", "String representation incorrect"
    
    def test_edge_enum(self):
        """Test Edge enum values and string representations."""
        assert AnalyzerEnums.Edge.LeadingEdge == 0, "LeadingEdge value incorrect"
        assert AnalyzerEnums.Edge.TrailingEdge == 1, "TrailingEdge value incorrect"
        assert str(AnalyzerEnums.Edge.LeadingEdge) == "Leading Edge", "String representation incorrect"
        assert str(AnalyzerEnums.Edge.TrailingEdge) == "Trailing Edge", "String representation incorrect"
    
    def test_parity_enum(self):
        """Test Parity enum values and string representations."""
        assert AnalyzerEnums.Parity.None_ == 0, "None_ value incorrect"
        assert AnalyzerEnums.Parity.Even == 1, "Even value incorrect"
        assert AnalyzerEnums.Parity.Odd == 2, "Odd value incorrect"
        assert str(AnalyzerEnums.Parity.None_) == "No Parity", "String representation incorrect"
        assert str(AnalyzerEnums.Parity.Even) == "Even Parity", "String representation incorrect"
        assert str(AnalyzerEnums.Parity.Odd) == "Odd Parity", "String representation incorrect"
    
    def test_parity_calculate_even(self):
        """Test calculation of even parity bits."""
        # Test values with even number of 1 bits (should return LOW for even parity)
        assert AnalyzerEnums.Parity.Even.calculate(0b00110011, 8) == BitState.LOW
        assert AnalyzerEnums.Parity.Even.calculate(0b10101010, 8) == BitState.LOW
        
        # Test values with odd number of 1 bits (should return HIGH for even parity)
        assert AnalyzerEnums.Parity.Even.calculate(0b10000000, 8) == BitState.HIGH
        assert AnalyzerEnums.Parity.Even.calculate(0b00000001, 8) == BitState.HIGH
        assert AnalyzerEnums.Parity.Even.calculate(0b10101011, 8) == BitState.HIGH
    
    def test_parity_calculate_odd(self):
        """Test calculation of odd parity bits."""
        # Test values with even number of 1 bits (should return HIGH for odd parity)
        assert AnalyzerEnums.Parity.Odd.calculate(0b00110011, 8) == BitState.HIGH
        assert AnalyzerEnums.Parity.Odd.calculate(0b10101010, 8) == BitState.HIGH
        
        # Test values with odd number of 1 bits (should return LOW for odd parity)
        assert AnalyzerEnums.Parity.Odd.calculate(0b10000000, 8) == BitState.LOW
        assert AnalyzerEnums.Parity.Odd.calculate(0b00000001, 8) == BitState.LOW
        assert AnalyzerEnums.Parity.Odd.calculate(0b10101011, 8) == BitState.LOW
    
    def test_parity_calculate_none_raises_error(self):
        """Test that calculating parity with None_ parity type raises an error."""
        with pytest.raises(ValueError, match="Cannot calculate parity for 'None' parity type"):
            AnalyzerEnums.Parity.None_.calculate(0x55, 8)
    
    def test_acknowledge_enum(self):
        """Test Acknowledge enum values and string representations."""
        assert AnalyzerEnums.Acknowledge.Ack == 0, "Ack value incorrect"
        assert AnalyzerEnums.Acknowledge.Nak == 1, "Nak value incorrect"
        assert str(AnalyzerEnums.Acknowledge.Ack) == "ACK", "String representation incorrect"
        assert str(AnalyzerEnums.Acknowledge.Nak) == "NAK", "String representation incorrect"
    
    def test_sign_enum(self):
        """Test Sign enum values and string representations."""
        assert AnalyzerEnums.Sign.UnsignedInteger == 0, "UnsignedInteger value incorrect"
        assert AnalyzerEnums.Sign.SignedInteger == 1, "SignedInteger value incorrect"
        assert str(AnalyzerEnums.Sign.UnsignedInteger) == "Unsigned", "String representation incorrect"
        assert str(AnalyzerEnums.Sign.SignedInteger) == "Signed", "String representation incorrect"
    
    def test_sign_interpret_value_unsigned(self):
        """Test interpretation of unsigned values."""
        sign = AnalyzerEnums.Sign.UnsignedInteger
        assert sign.interpret_value(0, 8) == 0, "Unsigned interpretation of 0 incorrect"
        assert sign.interpret_value(255, 8) == 255, "Unsigned interpretation of 255 incorrect"
        assert sign.interpret_value(0x7FFF, 16) == 0x7FFF, "Unsigned interpretation of 0x7FFF incorrect"
        assert sign.interpret_value(0xFFFF, 16) == 0xFFFF, "Unsigned interpretation of 0xFFFF incorrect"
    
    def test_sign_interpret_value_signed(self):
        """Test interpretation of signed values."""
        sign = AnalyzerEnums.Sign.SignedInteger
        assert sign.interpret_value(0, 8) == 0, "Signed interpretation of 0 incorrect"
        assert sign.interpret_value(127, 8) == 127, "Signed interpretation of 127 incorrect"
        assert sign.interpret_value(128, 8) == -128, "Signed interpretation of 128 incorrect"
        assert sign.interpret_value(255, 8) == -1, "Signed interpretation of 255 incorrect"
        assert sign.interpret_value(0x7FFF, 16) == 0x7FFF, "Signed interpretation of 0x7FFF incorrect"
        assert sign.interpret_value(0x8000, 16) == -0x8000, "Signed interpretation of 0x8000 incorrect"
        assert sign.interpret_value(0xFFFF, 16) == -1, "Signed interpretation of 0xFFFF incorrect"


class TestChannel:
    """Tests for the Channel class implementation."""
    
    def test_channel_init_default(self):
        """Test Channel initialization with default parameters."""
        channel = Channel()
        assert channel.device_id == 0, "Default device_id should be 0"
        assert channel.channel_index == 0, "Default channel_index should be 0"
    
    def test_channel_init_with_params(self):
        """Test Channel initialization with specific parameters."""
        channel = Channel(device_id=1, channel_index=5)
        assert channel.device_id == 1, "device_id not set correctly"
        assert channel.channel_index == 5, "channel_index not set correctly"
    
    def test_channel_properties(self):
        """Test Channel property getters and setters."""
        channel = Channel()
        
        # Test property setters
        channel.device_id = 2
        channel.channel_index = 7
        
        # Test property getters
        assert channel.device_id == 2, "device_id getter/setter not working"
        assert channel.channel_index == 7, "channel_index getter/setter not working"
    
    def test_channel_property_validation(self):
        """Test validation in Channel property setters."""
        channel = Channel()
        
        # Test negative values
        with pytest.raises(ValueError, match="Device ID cannot be negative"):
            channel.device_id = -1
            
        with pytest.raises(ValueError, match="Channel index cannot be negative"):
            channel.channel_index = -1
    
    def test_channel_equality(self):
        """Test Channel equality comparison."""
        ch1 = Channel(1, 2)
        ch2 = Channel(1, 2)
        ch3 = Channel(1, 3)
        ch4 = Channel(2, 2)
        
        assert ch1 == ch2, "Equal channels should compare equal"
        assert ch1 != ch3, "Channels with different indexes should not be equal"
        assert ch1 != ch4, "Channels with different device IDs should not be equal"
    
    def test_channel_ordering(self):
        """Test Channel ordering comparison."""
        ch1 = Channel(1, 2)
        ch2 = Channel(1, 3)
        ch3 = Channel(2, 1)
        
        # Test less than
        assert ch1 < ch2, "Channels should be ordered by index if device ID is the same"
        assert ch1 < ch3, "Channels should be ordered by device ID first"
        
        # Test greater than
        assert ch2 > ch1, "Channels should be ordered by index if device ID is the same"
        assert ch3 > ch1, "Channels should be ordered by device ID first"
    
    def test_channel_hash(self):
        """Test that Channel objects can be used as dictionary keys."""
        channels = {
            Channel(1, 2): "Channel 1-2",
            Channel(1, 3): "Channel 1-3"
        }
        
        assert channels[Channel(1, 2)] == "Channel 1-2", "Channel hash not working correctly"
        assert channels[Channel(1, 3)] == "Channel 1-3", "Channel hash not working correctly"
    
    def test_channel_is_valid(self):
        """Test the is_valid method for Channel."""
        valid_channel = Channel(1, 2)
        undefined_channel = UNDEFINED_CHANNEL
        
        assert valid_channel.is_valid(), "Regular channel should be valid"
        assert not undefined_channel.is_valid(), "UNDEFINED_CHANNEL should not be valid"
    
    def test_channel_repr(self):
        """Test the __repr__ method for Channel."""
        channel = Channel(1, 2)
        repr_str = repr(channel)
        
        assert "Channel" in repr_str, "Repr should contain class name"
        assert "device_id=1" in repr_str, "Repr should contain device_id"
        assert "channel_index=2" in repr_str, "Repr should contain channel_index"
    
    def test_channel_str(self):
        """Test the __str__ method for Channel."""
        channel = Channel(1, 2)
        str_value = str(channel)
        
        assert "Channel 2" in str_value, "String representation should contain channel number"
        assert "Device 1" in str_value, "String representation should contain device ID"


class TestBitUtilityFunctions:
    """Tests for bit manipulation utility functions."""
    
    def test_toggle_bit(self):
        """Test the toggle_bit function."""
        assert toggle_bit(BitState.LOW) == BitState.HIGH, "toggle_bit(LOW) should return HIGH"
        assert toggle_bit(BitState.HIGH) == BitState.LOW, "toggle_bit(HIGH) should return LOW"
    
    def test_toggle_bit_with_integers(self):
        """Test the toggle_bit function with integer inputs."""
        assert toggle_bit(0) == BitState.HIGH, "toggle_bit(0) should return HIGH"
        assert toggle_bit(1) == BitState.LOW, "toggle_bit(1) should return LOW"
    
    def test_toggle_bit_invalid_input(self):
        """Test toggle_bit with invalid inputs."""
        with pytest.raises(ValueError, match="Integer bit value must be 0 or 1"):
            toggle_bit(2)
        
        with pytest.raises(ValueError, match="Expected BitState or int"):
            toggle_bit("not a bit")
    
    def test_invert_bit(self):
        """Test the invert_bit function."""
        assert invert_bit(BitState.LOW) == BitState.HIGH, "invert_bit(LOW) should return HIGH"
        assert invert_bit(BitState.HIGH) == BitState.LOW, "invert_bit(HIGH) should return LOW"
    
    def test_invert_bit_with_integers(self):
        """Test the invert_bit function with integer inputs."""
        assert invert_bit(0) == BitState.HIGH, "invert_bit(0) should return HIGH"
        assert invert_bit(1) == BitState.LOW, "invert_bit(1) should return LOW"
    
    def test_invert_bit_invalid_input(self):
        """Test invert_bit with invalid inputs."""
        with pytest.raises(ValueError, match="Integer bit value must be 0 or 1"):
            invert_bit(2)
        
        with pytest.raises(ValueError, match="Expected BitState or int"):
            invert_bit("not a bit")


class TestBitConversionFunctions:
    """Tests for functions that convert between bits and integers."""
    
    def test_bits_to_int_msb_first(self):
        """Test the bits_to_int function with MSB first ordering."""
        # Test basic conversion
        bits = [BitState.HIGH, BitState.LOW, BitState.LOW, BitState.HIGH]  # 1001 binary = 9 decimal
        assert bits_to_int(bits) == 9, "bits_to_int conversion incorrect for 1001"
        
        # Test all zeros
        assert bits_to_int([BitState.LOW, BitState.LOW, BitState.LOW]) == 0, "bits_to_int incorrect for all zeros"
        
        # Test all ones
        assert bits_to_int([BitState.HIGH, BitState.HIGH, BitState.HIGH]) == 7, "bits_to_int incorrect for all ones"
        
        # Test empty list
        assert bits_to_int([]) == 0, "bits_to_int should return 0 for empty list"
    
    def test_bits_to_int_lsb_first(self):
        """Test the bits_to_int function with LSB first ordering."""
        # In LSB first, [HIGH, LOW, LOW, HIGH] would be interpreted as binary 1001 read from right to left,
        # which is 9 decimal
        bits = [BitState.HIGH, BitState.LOW, BitState.LOW, BitState.HIGH]
        assert bits_to_int(bits, msb_first=False) == 9, "bits_to_int LSB first conversion incorrect"
        
        # A different example: [LOW, HIGH, HIGH, LOW] in LSB first is binary 0110 read right to left,
        # which is 6 decimal
        bits = [BitState.LOW, BitState.HIGH, BitState.HIGH, BitState.LOW]
        assert bits_to_int(bits, msb_first=False) == 6, "bits_to_int LSB first conversion incorrect"
    
    def test_int_to_bits_msb_first(self):
        """Test the int_to_bits function with MSB first ordering."""
        # Test basic conversion
        expected = [BitState.HIGH, BitState.LOW, BitState.LOW, BitState.HIGH]  # 1001 binary = 9 decimal
        assert int_to_bits(9, 4) == expected, "int_to_bits conversion incorrect for 9"
        
        # Test zero
        assert all(bit == BitState.LOW for bit in int_to_bits(0, 3)), "int_to_bits incorrect for zero"
        
        # Test padding with zeros
        expected = [BitState.LOW, BitState.LOW, BitState.LOW, BitState.HIGH]  # 0001 binary = 1 decimal
        assert int_to_bits(1, 4) == expected, "int_to_bits padding incorrect"
    
    def test_int_to_bits_lsb_first(self):
        """Test the int_to_bits function with LSB first ordering."""
        # For 9 decimal (1001 binary), LSB first would be [1, 0, 0, 1] from left to right
        expected = [BitState.HIGH, BitState.LOW, BitState.LOW, BitState.HIGH]
        assert int_to_bits(9, 4, msb_first=False) == expected, "int_to_bits LSB first conversion incorrect"
        
        # For 6 decimal (0110 binary), LSB first would be [0, 1, 1, 0] from left to right
        expected = [BitState.LOW, BitState.HIGH, BitState.HIGH, BitState.LOW]
        assert int_to_bits(6, 4, msb_first=False) == expected, "int_to_bits LSB first conversion incorrect"


class TestParityFunctions:
    """Tests for parity-related functions."""
    
    def test_check_parity_even(self):
        """Test the check_parity function with even parity."""
        # Check correct even parity (total number of 1 bits is even)
        assert check_parity(0x53, 8, AnalyzerEnums.Parity.Even, BitState.HIGH), "Even parity check incorrect"
        assert check_parity(0x55, 8, AnalyzerEnums.Parity.Even, BitState.LOW), "Even parity check incorrect"
        
        # Check incorrect even parity
        assert not check_parity(0x53, 8, AnalyzerEnums.Parity.Even, BitState.LOW), "Even parity check incorrect"
        assert not check_parity(0x55, 8, AnalyzerEnums.Parity.Even, BitState.HIGH), "Even parity check incorrect"
    
    def test_check_parity_odd(self):
        """Test the check_parity function with odd parity."""
        # Check correct odd parity (total number of 1 bits is odd)
        assert check_parity(0x53, 8, AnalyzerEnums.Parity.Odd, BitState.LOW), "Odd parity check incorrect"
        assert check_parity(0x55, 8, AnalyzerEnums.Parity.Odd, BitState.HIGH), "Odd parity check incorrect"
        
        # Check incorrect odd parity
        assert not check_parity(0x53, 8, AnalyzerEnums.Parity.Odd, BitState.HIGH), "Odd parity check incorrect"
        assert not check_parity(0x55, 8, AnalyzerEnums.Parity.Odd, BitState.LOW), "Odd parity check incorrect"
    
    def test_check_parity_none_raises_error(self):
        """Test that check_parity with None_ parity type raises an error."""
        with pytest.raises(ValueError, match="Cannot check parity for 'None' parity type"):
            check_parity(0x55, 8, AnalyzerEnums.Parity.None_, BitState.LOW)
    
    def test_count_ones(self):
        """Test the count_ones function."""
        assert count_ones(0) == 0, "count_ones incorrect for 0"
        assert count_ones(1) == 1, "count_ones incorrect for 1"
        assert count_ones(3) == 2, "count_ones incorrect for 3 (binary 11)"
        assert count_ones(0x55) == 4, "count_ones incorrect for 0x55 (binary 01010101)"
        assert count_ones(0xFF) == 8, "count_ones incorrect for 0xFF (all bits set)"
    
    def test_count_ones_with_bit_limit(self):
        """Test the count_ones function with a bit limit."""
        # Only count the lowest 4 bits of 0xF5 (binary 11110101)
        # The lowest 4 bits are 0101, which has 2 bits set
        assert count_ones(0xF5, 4) == 2, "count_ones with bit limit incorrect"
        
        # Only count the lowest 6 bits of 0xF5 (binary 11110101)
        # The lowest 6 bits are 110101, which has 4 bits set
        assert count_ones(0xF5, 6) == 4, "count_ones with bit limit incorrect"


class TestValueFormattingFunctions:
    """Tests for functions that format or interpret values."""
    
    def test_format_value_unsigned(self):
        """Test the format_value function with unsigned values."""
        # Binary format
        assert format_value(10, DisplayBase.Binary) == "00001010", "Binary formatting incorrect"
        
        # Decimal format
        assert format_value(10, DisplayBase.Decimal) == "10", "Decimal formatting incorrect"
        
        # Hexadecimal format
        assert format_value(10, DisplayBase.Hexadecimal) == "0x0A", "Hex formatting incorrect"
        
        # ASCII format
        assert format_value(65, DisplayBase.ASCII) == "A", "ASCII formatting incorrect"
        assert format_value(0, DisplayBase.ASCII) == ".", "ASCII formatting for non-printable incorrect"
        
        # AsciiHex format
        assert format_value(65, DisplayBase.AsciiHex) == "A", "AsciiHex formatting incorrect"
        assert format_value(0, DisplayBase.AsciiHex) == "<00>", "AsciiHex formatting for non-printable incorrect"
    
    def test_format_value_signed(self):
        """Test the format_value function with signed values."""
        signed = AnalyzerEnums.Sign.SignedInteger
        
        # Hexadecimal format with negative value
        # 0xFE is -2 in 8-bit signed interpretation
        assert format_value(0xFE, DisplayBase.Hexadecimal, 8, signed) == "0xFE (-2)", "Signed hex formatting incorrect"
        
        # Decimal format directly shows the signed value
        assert format_value(0xFE, DisplayBase.Decimal, 8, signed) == "-2", "Signed decimal formatting incorrect"
        
        # Binary format with negative value shows two's complement representation with signed indication
        assert format_value(0xFE, DisplayBase.Binary, 8, signed) == "11111110 (-2)", "Signed binary formatting incorrect"
    
    def test_format_value_custom_bit_width(self):
        """Test the format_value function with custom bit widths."""
        # Binary format with 4 bits
        assert format_value(3, DisplayBase.Binary, 4) == "0011", "Binary formatting with custom width incorrect"
        
        # Hexadecimal format with 16 bits (4 hex digits)
        assert format_value(0x123, DisplayBase.Hexadecimal, 16) == "0x0123", "Hex formatting with custom width incorrect"

class TestValidationFunctions:
    """Tests for value validation functions."""
    
    def test_validate_int_range_valid(self):
        """Test validate_int_range with valid values."""
        assert validate_int_range(5, 0, 10) == 5, "validate_int_range should return value if valid"
        assert validate_int_range(0, 0, 10) == 0, "validate_int_range should accept minimum value"
        assert validate_int_range(10, 0, 10) == 10, "validate_int_range should accept maximum value"
    
    def test_validate_int_range_invalid(self):
        """Test validate_int_range with invalid values."""
        # Test value below minimum
        with pytest.raises(ValueError, match="must be between 0 and 10"):
            validate_int_range(-1, 0, 10)
        
        # Test value above maximum
        with pytest.raises(ValueError, match="must be between 0 and 10"):
            validate_int_range(11, 0, 10)
        
        # Test non-integer value
        with pytest.raises(TypeError, match="must be an integer"):
            validate_int_range(5.5, 0, 10)
    
    def test_validate_int_range_custom_name(self):
        """Test validate_int_range with a custom value name."""
        # Test with custom name
        with pytest.raises(ValueError, match="test_value must be between 0 and 10"):
            validate_int_range(11, 0, 10, name="test_value")
    
    def test_validate_u8(self):
        """Test validate_u8 function."""
        # Valid cases
        assert validate_u8(0) == 0, "validate_u8 should accept minimum value"
        assert validate_u8(255) == 255, "validate_u8 should accept maximum value"
        assert validate_u8(128) == 128, "validate_u8 should accept middle value"
        
        # Invalid cases
        with pytest.raises(ValueError, match="must be between 0 and 255"):
            validate_u8(-1)
        
        with pytest.raises(ValueError, match="must be between 0 and 255"):
            validate_u8(256)
    
    def test_validate_s8(self):
        """Test validate_s8 function."""
        # Valid cases
        assert validate_s8(-128) == -128, "validate_s8 should accept minimum value"
        assert validate_s8(127) == 127, "validate_s8 should accept maximum value"
        assert validate_s8(0) == 0, "validate_s8 should accept zero"
        
        # Invalid cases
        with pytest.raises(ValueError, match="must be between -128 and 127"):
            validate_s8(-129)
        
        with pytest.raises(ValueError, match="must be between -128 and 127"):
            validate_s8(128)
    
    def test_validate_u16(self):
        """Test validate_u16 function."""
        # Valid cases
        assert validate_u16(0) == 0, "validate_u16 should accept minimum value"
        assert validate_u16(65535) == 65535, "validate_u16 should accept maximum value"
        assert validate_u16(32768) == 32768, "validate_u16 should accept middle value"
        
        # Invalid cases
        with pytest.raises(ValueError, match="must be between 0 and 65535"):
            validate_u16(-1)
        
        with pytest.raises(ValueError, match="must be between 0 and 65535"):
            validate_u16(65536)
    
    def test_validate_s16(self):
        """Test validate_s16 function."""
        # Valid cases
        assert validate_s16(-32768) == -32768, "validate_s16 should accept minimum value"
        assert validate_s16(32767) == 32767, "validate_s16 should accept maximum value"
        assert validate_s16(0) == 0, "validate_s16 should accept zero"
        
        # Invalid cases
        with pytest.raises(ValueError, match="must be between -32768 and 32767"):
            validate_s16(-32769)
        
        with pytest.raises(ValueError, match="must be between -32768 and 32767"):
            validate_s16(32768)
    
    def test_validate_u32(self):
        """Test validate_u32 function."""
        # Valid cases
        assert validate_u32(0) == 0, "validate_u32 should accept minimum value"
        assert validate_u32(4294967295) == 4294967295, "validate_u32 should accept maximum value"
        assert validate_u32(2147483648) == 2147483648, "validate_u32 should accept middle value"
        
        # Invalid cases
        with pytest.raises(ValueError, match="must be between 0 and"):
            validate_u32(-1)
        
        with pytest.raises(ValueError, match="must be between 0 and"):
            validate_u32(4294967296)
    
    def test_validate_s32(self):
        """Test validate_s32 function."""
        # Valid cases
        assert validate_s32(-2147483648) == -2147483648, "validate_s32 should accept minimum value"
        assert validate_s32(2147483647) == 2147483647, "validate_s32 should accept maximum value"
        assert validate_s32(0) == 0, "validate_s32 should accept zero"
        
        # Invalid cases
        with pytest.raises(ValueError, match="must be between"):
            validate_s32(-2147483649)
        
        with pytest.raises(ValueError, match="must be between"):
            validate_s32(2147483648)
    
    def test_validate_u64(self):
        """Test validate_u64 function."""
        # Valid cases
        assert validate_u64(0) == 0, "validate_u64 should accept minimum value"
        assert validate_u64(18446744073709551615) == 18446744073709551615, "validate_u64 should accept maximum value"
        assert validate_u64(9223372036854775808) == 9223372036854775808, "validate_u64 should accept middle value"
        
        # Invalid cases
        with pytest.raises(ValueError, match="must be between 0 and"):
            validate_u64(-1)
        
        with pytest.raises(ValueError, match="must be between 0 and"):
            validate_u64(18446744073709551616)
    
    def test_validate_s64(self):
        """Test validate_s64 function."""
        # Valid cases
        assert validate_s64(-9223372036854775808) == -9223372036854775808, "validate_s64 should accept minimum value"
        assert validate_s64(9223372036854775807) == 9223372036854775807, "validate_s64 should accept maximum value"
        assert validate_s64(0) == 0, "validate_s64 should accept zero"
        
        # Invalid cases
        with pytest.raises(ValueError, match="must be between"):
            validate_s64(-9223372036854775809)
        
        with pytest.raises(ValueError, match="must be between"):
            validate_s64(9223372036854775808)


class TestByteConversionFunctions:
    """Tests for byte conversion functions."""
    
    def test_bytes_to_value_msb_first(self):
        """Test bytes_to_value with MSB first (big-endian) ordering."""
        # Test single byte
        assert bytes_to_value(b'\x12') == 0x12, "Single byte conversion incorrect"
        
        # Test multiple bytes
        assert bytes_to_value(b'\x12\x34') == 0x1234, "Two byte conversion incorrect"
        assert bytes_to_value(b'\x12\x34\x56\x78') == 0x12345678, "Four byte conversion incorrect"
        
        # Test empty bytes
        assert bytes_to_value(b'') == 0, "Empty bytes should convert to 0"
    
    def test_bytes_to_value_lsb_first(self):
        """Test bytes_to_value with LSB first (little-endian) ordering."""
        # Test single byte (same as MSB first for a single byte)
        assert bytes_to_value(b'\x12', msb_first=False) == 0x12, "Single byte conversion incorrect"
        
        # Test multiple bytes
        assert bytes_to_value(b'\x12\x34', msb_first=False) == 0x3412, "Two byte conversion incorrect"
        assert bytes_to_value(b'\x12\x34\x56\x78', msb_first=False) == 0x78563412, "Four byte conversion incorrect"
        
        # Test empty bytes
        assert bytes_to_value(b'', msb_first=False) == 0, "Empty bytes should convert to 0"
    
    def test_value_to_bytes_msb_first(self):
        """Test value_to_bytes with MSB first (big-endian) ordering."""
        # Test single byte
        assert value_to_bytes(0x12, 1) == b'\x12', "Single byte conversion incorrect"
        
        # Test multiple bytes
        assert value_to_bytes(0x1234, 2) == b'\x12\x34', "Two byte conversion incorrect"
        assert value_to_bytes(0x12345678, 4) == b'\x12\x34\x56\x78', "Four byte conversion incorrect"
        
        # Test zero
        assert value_to_bytes(0, 1) == b'\x00', "Zero conversion incorrect"
        assert value_to_bytes(0, 4) == b'\x00\x00\x00\x00', "Zero with padding incorrect"
        
        # Test padding with zeros
        assert value_to_bytes(0x12, 2) == b'\x00\x12', "Padding with zeros incorrect"
        assert value_to_bytes(0x1234, 4) == b'\x00\x00\x12\x34', "Padding with zeros incorrect"
    
    def test_value_to_bytes_lsb_first(self):
        """Test value_to_bytes with LSB first (little-endian) ordering."""
        # Test single byte (same as MSB first for a single byte)
        assert value_to_bytes(0x12, 1, msb_first=False) == b'\x12', "Single byte conversion incorrect"
        
        # Test multiple bytes
        assert value_to_bytes(0x1234, 2, msb_first=False) == b'\x34\x12', "Two byte conversion incorrect"
        assert value_to_bytes(0x12345678, 4, msb_first=False) == b'\x78\x56\x34\x12', "Four byte conversion incorrect"
        
        # Test zero
        assert value_to_bytes(0, 1, msb_first=False) == b'\x00', "Zero conversion incorrect"
        assert value_to_bytes(0, 4, msb_first=False) == b'\x00\x00\x00\x00', "Zero with padding incorrect"
        
        # Test padding with zeros
        assert value_to_bytes(0x12, 2, msb_first=False) == b'\x12\x00', "Padding with zeros incorrect"
        assert value_to_bytes(0x1234, 4, msb_first=False) == b'\x34\x12\x00\x00', "Padding with zeros incorrect"
    
    def test_value_to_bytes_value_too_large(self):
        """Test value_to_bytes with a value too large for the specified length."""
        # This should fail because 0x1234 requires at least 2 bytes
        with pytest.raises(OverflowError):
            value_to_bytes(0x1234, 1)
        
        # This should fail because 0x12345678 requires at least 4 bytes
        with pytest.raises(OverflowError):
            value_to_bytes(0x12345678, 3)


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""
    
    def test_integer_boundary_constants(self):
        """Test that integer boundary constants have correct values."""
        # 8-bit integers
        assert S8_MIN == -128, "S8_MIN incorrect"
        assert S8_MAX == 127, "S8_MAX incorrect"
        assert U8_MAX == 255, "U8_MAX incorrect"
        
        # 16-bit integers
        assert S16_MIN == -32768, "S16_MIN incorrect"
        assert S16_MAX == 32767, "S16_MAX incorrect"
        assert U16_MAX == 65535, "U16_MAX incorrect"
        
        # 32-bit integers
        assert S32_MIN == -2147483648, "S32_MIN incorrect"
        assert S32_MAX == 2147483647, "S32_MAX incorrect"
        assert U32_MAX == 4294967295, "U32_MAX incorrect"
        
        # 64-bit integers
        assert S64_MIN == -9223372036854775808, "S64_MIN incorrect"
        assert S64_MAX == 9223372036854775807, "S64_MAX incorrect"
        assert U64_MAX == 18446744073709551615, "U64_MAX incorrect"
    
    def test_undefined_channel_constant(self):
        """Test that UNDEFINED_CHANNEL has the correct values."""
        from kingst_analyzer.types import UNDEFINED_CHANNEL
        
        assert UNDEFINED_CHANNEL.device_id == 0xFFFFFFFFFFFFFFFF, "UNDEFINED_CHANNEL device_id incorrect"
        assert UNDEFINED_CHANNEL.channel_index == 0xFFFFFFFF, "UNDEFINED_CHANNEL channel_index incorrect"
        assert not UNDEFINED_CHANNEL.is_valid(), "UNDEFINED_CHANNEL should not be valid"
    
    def test_bit_conversion_edge_cases(self):
        """Test edge cases for bit conversion functions."""
        from kingst_analyzer.types import bits_to_int, int_to_bits, BitState
        
        # Test with maximum 8-bit value
        bits = [BitState.HIGH] * 8  # All 1's
        assert bits_to_int(bits) == 255, "bits_to_int incorrect for all 1's"
        
        # Test conversion roundtrip
        for value in [0, 1, 127, 128, 255]:
            bits = int_to_bits(value, 8)
            assert bits_to_int(bits) == value, f"Conversion roundtrip failed for {value}"
        
        # Test with a large number of bits
        bits = int_to_bits(0xFFFFFFFF, 32)
        assert len(bits) == 32, "int_to_bits should produce 32 bits"
        assert all(bit == BitState.HIGH for bit in bits), "All bits should be HIGH"
        assert bits_to_int(bits) == 0xFFFFFFFF, "bits_to_int should handle 32 bits"
    
    def test_format_value_edge_cases(self):
        """Test edge cases for format_value function."""
        from kingst_analyzer.types import format_value, DisplayBase, AnalyzerEnums
        
        # Test with minimum and maximum 8-bit values
        assert format_value(0, DisplayBase.Binary, 8) == "00000000", "Binary formatting for 0 incorrect"
        assert format_value(255, DisplayBase.Binary, 8) == "11111111", "Binary formatting for 255 incorrect"
        
        # Test with minimum and maximum 8-bit signed values
        signed = AnalyzerEnums.Sign.SignedInteger
        assert format_value(128, DisplayBase.Decimal, 8, signed) == "-128", "Signed decimal for -128 incorrect"
        assert format_value(127, DisplayBase.Decimal, 8, signed) == "127", "Signed decimal for 127 incorrect"
        
        # Test with non-printable ASCII characters
        assert format_value(0, DisplayBase.ASCII) == ".", "ASCII for NUL incorrect"
        assert format_value(127, DisplayBase.ASCII) == ".", "ASCII for DEL incorrect"
        
        # Test with printable and non-printable ASCII in AsciiHex mode
        assert format_value(65, DisplayBase.AsciiHex) == "A", "AsciiHex for 'A' incorrect"
        assert format_value(0, DisplayBase.AsciiHex) == "<00>", "AsciiHex for NUL incorrect"


class TestComprehensiveChecks:
    """Comprehensive tests that combine multiple functions."""
    
    def test_validation_and_formatting(self):
        """Test combining validation with formatting functions."""
        from kingst_analyzer.types import (
            validate_u8, format_value, DisplayBase
        )
        
        # Validate and format a byte value
        value = validate_u8(42)
        assert format_value(value, DisplayBase.Binary, 8) == "00101010", "Binary formatting after validation incorrect"
        assert format_value(value, DisplayBase.Hexadecimal, 8) == "0x2A", "Hex formatting after validation incorrect"
    
    def test_bit_manipulation_chain(self):
        """Test a chain of bit manipulation operations."""
        from kingst_analyzer.types import (
            BitState, toggle_bit, int_to_bits, bits_to_int
        )
        
        # Convert integer to bits, toggle one bit, convert back
        bits = int_to_bits(0x55, 8)  # 01010101
        bits[3] = toggle_bit(bits[3])  # Toggle the 4th bit to get 01011101
        value = bits_to_int(bits)
        assert value == 0x5D, "Bit manipulation chain produced incorrect result"
    
    def test_byte_conversion_roundtrip(self):
        """Test roundtrip conversion between values and bytes."""
        from kingst_analyzer.types import bytes_to_value, value_to_bytes
        
        # Test MSB first roundtrip
        for value in [0, 1, 0xFF, 0x1234, 0x12345678]:
            # Determine minimum byte length needed
            length = max(1, (value.bit_length() + 7) // 8)
            bytes_data = value_to_bytes(value, length)
            roundtrip_value = bytes_to_value(bytes_data)
            assert roundtrip_value == value, f"MSB roundtrip failed for {value}"
        
        # Test LSB first roundtrip
        for value in [0, 1, 0xFF, 0x1234, 0x12345678]:
            # Determine minimum byte length needed
            length = max(1, (value.bit_length() + 7) // 8)
            bytes_data = value_to_bytes(value, length, msb_first=False)
            roundtrip_value = bytes_to_value(bytes_data, msb_first=False)
            assert roundtrip_value == value, f"LSB roundtrip failed for {value}"
    
    def test_parity_calculation_and_checking(self):
        """Test combination of parity calculation and checking."""
        from kingst_analyzer.types import (
            AnalyzerEnums, BitState, check_parity
        )
        
        # Generate a sequence of test values
        test_values = [0x00, 0x55, 0xAA, 0xFF, 0x12, 0x34, 0x78, 0x9A]
        
        # For each value, calculate parity bits and then verify them
        for value in test_values:
            # Test even parity
            even_parity_bit = AnalyzerEnums.Parity.Even.calculate(value, 8)
            assert check_parity(value, 8, AnalyzerEnums.Parity.Even, even_parity_bit), \
                f"Even parity check failed for {value:02X}"
            
            # Test odd parity
            odd_parity_bit = AnalyzerEnums.Parity.Odd.calculate(value, 8)
            assert check_parity(value, 8, AnalyzerEnums.Parity.Odd, odd_parity_bit), \
                f"Odd parity check failed for {value:02X}"
            
            # Verify even and odd parity bits are opposites
            assert even_parity_bit != odd_parity_bit, \
                f"Even and odd parity bits should be opposites for {value:02X}"