"""
Simulation module for Kingst Logic Analyzer.

This module provides Pythonic interfaces for creating and configuring simulations
with the Kingst Logic Analyzer. It includes builder patterns for complex simulation
setups and helper functions for common test signals and protocol patterns.

The module bridges between high-level Python code and the low-level C++ bindings,
providing comprehensive access to the simulator functionality with a clean,
intuitive API and strong type checking.

Example:
    Create a simple UART simulation:
    
    ```python
    from kingst_LA import simulation as sim
    from kingst_LA.types import Channel, BitState
    
    # Create a simulation manager
    manager = sim.SimulationManager(sample_rate=10_000_000)  # 10 MHz
    
    # Configure a UART simulation
    uart = manager.add_uart(
        tx_channel=Channel(0, "TX"),
        data=b"Hello, World!",
        baud_rate=115200
    )
    
    # Access the simulation channels
    tx_channel = uart.tx_channel
    
    # Run the simulation
    manager.run()
    ```
"""

import math
import struct
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Union, Callable, Any, Sequence, Iterator, Set

# Import the low-level bindings
try:
    from .. import _kingst_analyzer as _ka
    from ..types import Channel, BitState
except ImportError:
    # For standalone testing and documentation
    class _ka:
        class SimulationChannelDescriptor:
            pass
        class SimulationChannelDescriptorGroup:
            pass
        class Channel:
            pass
        class BitState:
            HIGH = 1
            LOW = 0
    
    class Channel:
        def __init__(self, index, name=None):
            self.index = index
            self.name = name
    
    class BitState:
        HIGH = 1
        LOW = 0


# Type aliases for better type hinting and documentation
U8 = int
U16 = int
U32 = int
U64 = int


class SimulationChannelDescriptor:
    """
    Pythonic wrapper for a simulation channel descriptor.
    
    This class provides a more pythonic interface to the underlying
    C++ SimulationChannelDescriptor, adding helper methods and
    better error handling.
    
    Attributes:
        name (str): User-defined name of the channel
        channel_index (int): Index of the channel in the analyzer
        sample_rate_hz (int): Sample rate in Hz
    """
    
    def __init__(self, descriptor: _ka.SimulationChannelDescriptor, 
                 name: str = "", channel_index: int = -1, 
                 sample_rate_hz: int = 0):
        """
        Initialize a simulation channel wrapper.
        
        Args:
            descriptor: The underlying SimulationChannelDescriptor
            name: User-defined name for this channel
            channel_index: Index of this channel (0-based)
            sample_rate_hz: Sample rate in Hz
            
        Raises:
            ValueError: If descriptor is None
        """
        if descriptor is None:
            raise ValueError("SimulationChannelDescriptor cannot be None")
            
        self._descriptor = descriptor
        self.name = name
        self.channel_index = channel_index
        self.sample_rate_hz = sample_rate_hz or descriptor.get_sample_rate()
        
        # Cache the initial state for reference
        self._initial_state = descriptor.get_initial_bit_state()
    
    # -------------------------------------------------------------------------
    # Core methods (wrapping underlying C++ methods)
    # -------------------------------------------------------------------------
    
    def transition(self) -> None:
        """
        Toggle the current bit state (from HIGH to LOW or LOW to HIGH).
        
        Creates a transition at the current sample position.
        """
        self._descriptor.transition()
    
    def transition_if_needed(self, bit_state: BitState) -> None:
        """
        Transition to a specific bit state if not already at that state.
        
        Args:
            bit_state: Desired bit state (BitState.HIGH or BitState.LOW)
            
        Raises:
            TypeError: If bit_state is not a valid BitState
        """
        if not isinstance(bit_state, (BitState, int)):
            raise TypeError(f"Expected BitState, got {type(bit_state)}")
            
        self._descriptor.transition_if_needed(bit_state)
    
    def advance(self, num_samples: int) -> None:
        """
        Move forward by a specified number of samples.
        
        This advances the current sample position without changing the bit state.
        
        Args:
            num_samples: Number of samples to move forward
            
        Raises:
            ValueError: If num_samples is negative
        """
        if num_samples < 0:
            raise ValueError("Number of samples to advance must be positive")
            
        self._descriptor.advance(num_samples)
    
    @property
    def current_bit_state(self) -> BitState:
        """Get the current bit state."""
        return self._descriptor.get_current_bit_state()
    
    @property
    def current_sample_number(self) -> int:
        """Get the current sample position."""
        return self._descriptor.get_current_sample_number()
    
    @property
    def current_time_s(self) -> float:
        """
        Get the current time in seconds from the start of simulation.
        
        Returns:
            Current time in seconds
            
        Raises:
            ValueError: If sample rate is not set
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to calculate time")
            
        return self.current_sample_number / self.sample_rate_hz
    
    # -------------------------------------------------------------------------
    # Enhanced signal generation methods
    # -------------------------------------------------------------------------
    
    def add_pulse(self, width: int, count: int = 1, gap: int = 0) -> None:
        """
        Add a series of pulses.
        
        Creates a series of pulses (transitions from the current state
        and back) with specified width and gaps between pulses.
        
        Args:
            width: Width of each pulse in samples
            count: Number of pulses to create
            gap: Gap between pulses in samples
            
        Raises:
            ValueError: If width, count, or gap is negative
        """
        if width <= 0:
            raise ValueError("Pulse width must be positive")
        if count <= 0:
            raise ValueError("Pulse count must be positive")
        if gap < 0:
            raise ValueError("Gap cannot be negative")
            
        self._descriptor.add_pulse(width, count, gap)
    
    def add_pulse_train(self, pulse_width_s: float, gap_s: float, 
                       count: int) -> None:
        """
        Add a train of pulses using time-based specification.
        
        Args:
            pulse_width_s: Width of each pulse in seconds
            gap_s: Gap between pulses in seconds
            count: Number of pulses to generate
            
        Raises:
            ValueError: If sample rate is not set or parameters are invalid
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for time-based operations")
            
        if pulse_width_s <= 0:
            raise ValueError("Pulse width must be positive")
        if gap_s < 0:
            raise ValueError("Gap cannot be negative")
        if count <= 0:
            raise ValueError("Pulse count must be positive")
            
        pulse_width_samples = int(pulse_width_s * self.sample_rate_hz)
        gap_samples = int(gap_s * self.sample_rate_hz)
        
        self.add_pulse(pulse_width_samples, count, gap_samples)
    
    def add_bit_pattern(self, pattern: List[BitState], samples_per_bit: int) -> None:
        """
        Add a specific bit pattern.
        
        Creates a sequence of bits with specified duration per bit.
        
        Args:
            pattern: List of BitStates to generate
            samples_per_bit: Number of samples for each bit
            
        Raises:
            ValueError: If pattern is empty or samples_per_bit is invalid
            TypeError: If pattern contains invalid types
            
        Example:
            >>> # Create a pattern with alternating HIGH and LOW bits
            >>> pattern = [BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW]
            >>> sim_channel.add_bit_pattern(pattern, 100)
        """
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        if samples_per_bit <= 0:
            raise ValueError("Samples per bit must be positive")
            
        # Validate pattern elements
        for i, bit in enumerate(pattern):
            if not isinstance(bit, (BitState, int)):
                raise TypeError(f"Pattern element at index {i} is not a valid BitState")
        
        self._descriptor.add_bit_pattern(pattern, samples_per_bit)
    
    def add_byte_pattern(self, data: Union[bytes, bytearray, List[int]], 
                        samples_per_bit: int, msb_first: bool = True) -> None:
        """
        Add a pattern from bytes or integer data.
        
        Converts bytes or integer values to bit patterns and generates
        the corresponding signal.
        
        Args:
            data: Bytes or list of integers to convert to bit patterns
            samples_per_bit: Number of samples for each bit
            msb_first: Whether bits should be sent MSB first
            
        Raises:
            ValueError: If data is empty or samples_per_bit is invalid
            TypeError: If data contains invalid values
            
        Example:
            >>> # Send the byte 0xA5 (10100101) MSB first
            >>> sim_channel.add_byte_pattern(bytes([0xA5]), 100)
        """
        if not data:
            raise ValueError("Data cannot be empty")
        if samples_per_bit <= 0:
            raise ValueError("Samples per bit must be positive")
        
        # Convert to bytes if it's a list of integers
        if isinstance(data, list):
            try:
                data = bytes(data)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Invalid data in list: {e}")
        
        # Process each byte in the data
        for byte in data:
            bit_pattern = []
            
            # Create bit pattern based on MSB/LSB order
            if msb_first:
                for bit_pos in range(7, -1, -1):
                    bit_value = (byte >> bit_pos) & 0x01
                    bit_pattern.append(BitState.HIGH if bit_value else BitState.LOW)
            else:
                for bit_pos in range(8):
                    bit_value = (byte >> bit_pos) & 0x01
                    bit_pattern.append(BitState.HIGH if bit_value else BitState.LOW)
            
            # Add the bit pattern
            self.add_bit_pattern(bit_pattern, samples_per_bit)
    
    def add_square_wave(self, frequency_hz: int, duty_cycle_percent: float = 50.0, 
                       cycles: int = 10) -> None:
        """
        Add a square wave signal.
        
        Args:
            frequency_hz: Frequency in Hz
            duty_cycle_percent: Duty cycle (0-100%)
            cycles: Number of cycles to generate
            
        Raises:
            ValueError: If sample rate is not set or parameters are invalid
            
        Example:
            >>> # Generate a 1kHz square wave with 25% duty cycle for 10 cycles
            >>> sim_channel.add_square_wave(1000, 25.0, 10)
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to generate frequency-based signals")
            
        if frequency_hz <= 0:
            raise ValueError("Frequency must be positive")
        if duty_cycle_percent <= 0 or duty_cycle_percent >= 100:
            raise ValueError("Duty cycle must be between 0 and 100 exclusive")
        if cycles <= 0:
            raise ValueError("Cycle count must be positive")
            
        # Calculate period and high/low durations
        period_samples = self.sample_rate_hz // frequency_hz
        high_samples = int(period_samples * duty_cycle_percent / 100.0)
        low_samples = period_samples - high_samples
        
        for _ in range(cycles):
            self.transition_if_needed(BitState.HIGH)
            self.advance(high_samples)
            self.transition_if_needed(BitState.LOW)
            self.advance(low_samples)
    
    def add_clock(self, frequency_hz: int, cycles: int = 10, 
                 duty_cycle_percent: float = 50.0) -> None:
        """
        Add a clock signal (alias for square_wave with clearer intent).
        
        Args:
            frequency_hz: Clock frequency in Hz
            cycles: Number of clock cycles
            duty_cycle_percent: Duty cycle (0-100%)
            
        Raises:
            ValueError: If sample rate is not set or parameters are invalid
            
        Example:
            >>> # Generate a 10MHz clock signal for 100 cycles
            >>> sim_channel.add_clock(10_000_000, 100)
        """
        self.add_square_wave(frequency_hz, duty_cycle_percent, cycles)
    
    # -------------------------------------------------------------------------
    # Protocol-specific generation methods
    # -------------------------------------------------------------------------
    
    def add_uart_byte(self, byte: int, bit_width: int, with_start_bit: bool = True,
                     with_stop_bit: bool = True, with_parity_bit: bool = False,
                     even_parity: bool = True) -> None:
        """
        Add a UART byte transmission.
        
        Creates a complete UART byte transmission including optional
        start bit, stop bit, and parity bit.
        
        Args:
            byte: Data byte to transmit (0-255)
            bit_width: Width of each bit in samples
            with_start_bit: Include start bit
            with_stop_bit: Include stop bit
            with_parity_bit: Include parity bit
            even_parity: Use even parity (if with_parity_bit is True)
            
        Raises:
            ValueError: If byte is not in range 0-255 or bit_width is invalid
            
        Example:
            >>> # Send the ASCII character 'A' with standard 8N1 format
            >>> sim_channel.add_uart_byte(ord('A'), 100, True, True, False)
        """
        if not 0 <= byte <= 255:
            raise ValueError("Byte value must be in range 0-255")
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
            
        self._descriptor.add_uart_byte(
            byte, bit_width, with_start_bit, with_stop_bit, with_parity_bit, even_parity)
    
    def add_uart_string(self, data: Union[str, bytes, bytearray], baud_rate: int,
                       with_start_bit: bool = True, with_stop_bit: bool = True,
                       with_parity_bit: bool = False, even_parity: bool = True) -> None:
        """
        Add a UART string transmission.
        
        Creates a complete UART transmission of a string or byte sequence.
        
        Args:
            data: String or bytes to transmit
            baud_rate: UART baud rate in bits per second
            with_start_bit: Include start bit
            with_stop_bit: Include stop bit
            with_parity_bit: Include parity bit
            even_parity: Use even parity (if with_parity_bit is True)
            
        Raises:
            ValueError: If sample rate is not set or parameters are invalid
            
        Example:
            >>> # Send "Hello World" at 9600 baud
            >>> sim_channel.add_uart_string("Hello World", 9600)
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for baud rate operations")
            
        if baud_rate <= 0:
            raise ValueError("Baud rate must be positive")
            
        # Calculate bit width based on baud rate
        bit_width = self.sample_rate_hz // baud_rate
        
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Add idle time before transmission (10 bit widths)
        current_state = self.current_bit_state
        self.transition_if_needed(BitState.HIGH)  # UART idle state is HIGH
        self.advance(bit_width * 10)
        
        # Send each byte
        for byte in data:
            self.add_uart_byte(
                byte, bit_width, with_start_bit, with_stop_bit, with_parity_bit, even_parity)
        
        # Add idle time after transmission (10 bit widths)
        self.transition_if_needed(BitState.HIGH)
        self.advance(bit_width * 10)
    
    def add_i2c_start(self, bit_width: int) -> None:
        """
        Add an I2C START condition.
        
        Creates an I2C START condition (SDA falling edge while SCL is HIGH).
        
        Args:
            bit_width: Width of the START condition in samples
            
        Raises:
            ValueError: If bit_width is invalid
            
        Note:
            This should be called on the SDA channel. The SCL channel
            should be set to HIGH during this transition.
        """
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
            
        self._descriptor.add_i2c_start(bit_width, False)
    
    def add_i2c_stop(self, bit_width: int) -> None:
        """
        Add an I2C STOP condition.
        
        Creates an I2C STOP condition (SDA rising edge while SCL is HIGH).
        
        Args:
            bit_width: Width of the STOP condition in samples
            
        Raises:
            ValueError: If bit_width is invalid
            
        Note:
            This should be called on the SDA channel. The SCL channel
            should be set to HIGH during this transition.
        """
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
            
        self._descriptor.add_i2c_stop(bit_width, False)
    
    def add_i2c_bit(self, bit_value: bool, bit_width: int, is_scl: bool = False) -> None:
        """
        Add a single I2C bit.
        
        Creates the proper signal pattern for a single I2C bit on either SCL or SDA.
        
        Args:
            bit_value: Value of the bit (for SDA only, ignored for SCL)
            bit_width: Width of each bit in samples
            is_scl: True if this is the SCL (clock) channel, False for SDA (data)
            
        Raises:
            ValueError: If bit_width is invalid
            
        Note:
            This is a low-level primitive that doesn't handle I2C protocol specifics.
            For full I2C transactions, use the SimulationManager's higher-level methods.
        """
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
            
        self._descriptor.add_i2c_bit(bit_value, bit_width, is_scl)
    
    def add_i2c_byte(self, byte: int, with_ack: bool = True, 
                    bit_width: int = 100, is_scl: bool = False) -> None:
        """
        Add an I2C byte transmission.
        
        Creates an I2C byte transmission including the ACK bit.
        
        Args:
            byte: Data byte to transmit (0-255)
            with_ack: Include ACK bit (pulled low)
            bit_width: Width of each bit in samples
            is_scl: True if this is the SCL channel, False for SDA
            
        Raises:
            ValueError: If byte is not in range 0-255 or bit_width is invalid
            
        Note:
            I2C bytes are transmitted MSB first.
            This method handles only the data phase, not START/STOP conditions.
        """
        if not 0 <= byte <= 255:
            raise ValueError("Byte value must be in range 0-255")
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
            
        self._descriptor.add_i2c_byte(byte, with_ack, bit_width, is_scl)
    
    def add_spi_bit(self, bit_value: bool, bit_width: int, cpol: bool = False,
                   cpha: bool = False, is_clock: bool = False) -> None:
        """
        Add a single SPI bit.
        
        Creates the proper signal pattern for a single SPI bit on either clock or data lines.
        
        Args:
            bit_value: Value of the bit (for data lines only, ignored for clock)
            bit_width: Width of each bit in samples
            cpol: Clock polarity (0 = idle low, 1 = idle high)
            cpha: Clock phase (0 = sample on first edge, 1 = sample on second edge)
            is_clock: True if this is the clock channel, False for data lines
            
        Raises:
            ValueError: If bit_width is invalid
            
        Note:
            This is a low-level primitive that doesn't handle full SPI transactions.
            For full SPI transactions, use the SimulationManager's higher-level methods.
        """
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
            
        self._descriptor.add_spi_bit(bit_value, bit_width, cpol, cpha, is_clock)
    
    # -------------------------------------------------------------------------
    # Advanced pattern generation methods
    # -------------------------------------------------------------------------
    
    def add_ramp(self, start_value: int, end_value: int, steps: int, 
                bit_width: int, bits: int = 8) -> None:
        """
        Add a digital ramp pattern (stair/steps).
        
        Creates a stair/ramp pattern with values increasing from
        start_value to end_value in steps steps.
        
        Args:
            start_value: Starting value (0-255 for 8-bit)
            end_value: Ending value (0-255 for 8-bit)
            steps: Number of steps in the ramp
            bit_width: Width of each bit in samples
            bits: Number of bits to use (default: 8)
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> # Create an 8-bit ramp from 0 to 255 in 8 steps
            >>> sim_channel.add_ramp(0, 255, 8, 100)
        """
        if steps <= 0:
            raise ValueError("Steps must be positive")
        if not 1 <= bits <= 32:
            raise ValueError("Bits must be between 1 and 32")
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
            
        max_val = (1 << bits) - 1
        if not 0 <= start_value <= max_val:
            raise ValueError(f"Start value must be in range 0-{max_val}")
        if not 0 <= end_value <= max_val:
            raise ValueError(f"End value must be in range 0-{max_val}")
        
        # Calculate step size
        step_size = (end_value - start_value) / (steps - 1) if steps > 1 else 0
        
        for step in range(steps):
            # Calculate value for this step
            value = round(start_value + step * step_size)
            value = max(0, min(max_val, value))  # Ensure within bounds
            
            # Create a bit pattern representing the value
            pattern = []
            for bit in range(bits - 1, -1, -1):  # MSB first
                pattern.append(BitState.HIGH if (value & (1 << bit)) else BitState.LOW)
            
            # Add the bit pattern
            self.add_bit_pattern(pattern, bit_width)
    
    def add_walking_ones(self, bit_width: int, bits: int = 8) -> None:
        """
        Add a walking 1's pattern.
        
        Creates a pattern where a single 1 bit walks through
        an otherwise all 0's pattern.
        
        Args:
            bit_width: Width of each bit in samples
            bits: Number of bits in the pattern
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> # Create an 8-bit walking 1's pattern (00000001, 00000010, 00000100, ...)
            >>> sim_channel.add_walking_ones(100, 8)
        """
        if not 1 <= bits <= 32:
            raise ValueError("Bits must be between 1 and 32")
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
        
        for pos in range(bits):
            # Create a bit pattern with a single 1 at position pos
            pattern = [BitState.LOW] * bits
            pattern[bits - 1 - pos] = BitState.HIGH  # MSB first
            
            # Add the bit pattern
            self.add_bit_pattern(pattern, bit_width)
    
    def add_walking_zeros(self, bit_width: int, bits: int = 8) -> None:
        """
        Add a walking 0's pattern.
        
        Creates a pattern where a single 0 bit walks through
        an otherwise all 1's pattern.
        
        Args:
            bit_width: Width of each bit in samples
            bits: Number of bits in the pattern
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> # Create an 8-bit walking 0's pattern (11111110, 11111101, 11111011, ...)
            >>> sim_channel.add_walking_zeros(100, 8)
        """
        if not 1 <= bits <= 32:
            raise ValueError("Bits must be between 1 and 32")
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
        
        for pos in range(bits):
            # Create a bit pattern with a single 0 at position pos
            pattern = [BitState.HIGH] * bits
            pattern[bits - 1 - pos] = BitState.LOW  # MSB first
            
            # Add the bit pattern
            self.add_bit_pattern(pattern, bit_width)
    
    def add_counter(self, start_value: int, end_value: int, 
                   bit_width: int, bits: int = 8) -> None:
        """
        Add a binary counter pattern.
        
        Creates a sequence of binary values counting from
        start_value to end_value.
        
        Args:
            start_value: Starting value
            end_value: Ending value
            bit_width: Width of each bit in samples
            bits: Number of bits in each value
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> # Create a 4-bit counter from 0 to 15
            >>> sim_channel.add_counter(0, 15, 100, 4)
        """
        if not 1 <= bits <= 32:
            raise ValueError("Bits must be between 1 and 32")
        if bit_width <= 0:
            raise ValueError("Bit width must be positive")
            
        max_val = (1 << bits) - 1
        if not 0 <= start_value <= max_val:
            raise ValueError(f"Start value must be in range 0-{max_val}")
        if not 0 <= end_value <= max_val:
            raise ValueError(f"End value must be in range 0-{max_val}")
        
        # Determine step direction
        step = 1 if start_value <= end_value else -1
        
        # Generate each value in the sequence
        for value in range(start_value, end_value + step, step):
            # Create a bit pattern representing the value
            pattern = []
            for bit in range(bits - 1, -1, -1):  # MSB first
                pattern.append(BitState.HIGH if (value & (1 << bit)) else BitState.LOW)
            
            # Add the bit pattern
            self.add_bit_pattern(pattern, bit_width)
    
    def add_manchester_data(self, data: Union[bytes, bytearray, List[int]], 
                          bit_rate: int, standard_manchester: bool = True,
                          msb_first: bool = True) -> None:
        """
        Add Manchester-encoded data.
        
        Creates a Manchester-encoded signal from the provided data.
        
        Args:
            data: Bytes to encode using Manchester encoding
            bit_rate: Bit rate in bits per second
            standard_manchester: Whether to use IEEE 802.3 (True) or Thomas/G.E. (False) encoding
            msb_first: Whether to transmit most significant bit first
            
        Raises:
            ValueError: If sample rate is not set or parameters are invalid
            
        Note:
            IEEE 802.3 Manchester:
            - 0 = high-to-low transition in middle of bit
            - 1 = low-to-high transition in middle of bit
            
            Thomas/G.E. Manchester:
            - 0 = low-to-high transition in middle of bit
            - 1 = high-to-low transition in middle of bit
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for bit rate operations")
            
        if bit_rate <= 0:
            raise ValueError("Bit rate must be positive")
            
        # Calculate half bit width (transition occurs in middle of each bit)
        half_bit_samples = self.sample_rate_hz // (bit_rate * 2)
        
        # Convert to bytes if needed
        if isinstance(data, list):
            try:
                data = bytes(data)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Invalid data in list: {e}")
        
        # Process each byte
        for byte in data:
            # Process each bit
            for bit_idx in range(8):
                # Determine actual bit position based on MSB/LSB order
                actual_bit_idx = 7 - bit_idx if msb_first else bit_idx
                bit_value = (byte >> actual_bit_idx) & 0x01
                
                # Encode bit using Manchester encoding
                if standard_manchester:
                    # IEEE 802.3 Manchester
                    if bit_value:
                        # Bit 1: low->high transition
                        self.transition_if_needed(BitState.LOW)
                        self.advance(half_bit_samples)
                        self.transition_if_needed(BitState.HIGH)
                        self.advance(half_bit_samples)
                    else:
                        # Bit 0: high->low transition
                        self.transition_if_needed(BitState.HIGH)
                        self.advance(half_bit_samples)
                        self.transition_if_needed(BitState.LOW)
                        self.advance(half_bit_samples)
                else:
                    # Thomas/G.E. Manchester (inverted)
                    if bit_value:
                        # Bit 1: high->low transition
                        self.transition_if_needed(BitState.HIGH)
                        self.advance(half_bit_samples)
                        self.transition_if_needed(BitState.LOW)
                        self.advance(half_bit_samples)
                    else:
                        # Bit 0: low->high transition
                        self.transition_if_needed(BitState.LOW)
                        self.advance(half_bit_samples)
                        self.transition_if_needed(BitState.HIGH)
                        self.advance(half_bit_samples)

    # -------------------------------------------------------------------------
    # Time-based convenience methods
    # -------------------------------------------------------------------------
    
    def advance_by_time(self, time_s: float) -> None:
        """
        Advance by a specified amount of time in seconds.
        
        Args:
            time_s: Time to advance in seconds
            
        Raises:
            ValueError: If sample rate is not set or time is negative
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for time-based operations")
            
        if time_s < 0:
            raise ValueError("Time cannot be negative")
            
        samples = int(time_s * self.sample_rate_hz)
        self.advance(samples)
    
    def add_idle_time(self, time_s: float) -> None:
        """
        Add idle time without changing the current state.
        
        This is a convenience method for advancing while maintaining the current state.
        
        Args:
            time_s: Idle time to add in seconds
            
        Raises:
            ValueError: If sample rate is not set or time is negative
        """
        self.advance_by_time(time_s)
    
    # -------------------------------------------------------------------------
    # Information methods
    # -------------------------------------------------------------------------
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this simulation channel.
        
        Returns:
            Dictionary with channel information
        """
        return {
            "name": self.name,
            "channel_index": self.channel_index,
            "sample_rate": self.sample_rate_hz,
            "current_sample": self.current_sample_number,
            "current_state": self.current_bit_state,
            "initial_state": self._initial_state,
        }
    
    def __str__(self) -> str:
        """Create a string representation of this channel."""
        name = self.name or f"Channel {self.channel_index}"
        return (f"SimulationChannel({name}, sample_rate={self.sample_rate_hz} Hz, "
                f"current_sample={self.current_sample_number}, "
                f"current_state={'HIGH' if self.current_bit_state == BitState.HIGH else 'LOW'})")
    
    def __repr__(self) -> str:
        """Create a detailed string representation of this channel."""
        return self.__str__()


class SignalGenerator:
    """
    Utility class for generating common digital signal patterns.
    
    This class provides static methods for generating standard digital
    signals on simulation channels.
    """
    
    @staticmethod
    def square_wave(channel: SimulationChannelDescriptor, 
                   frequency_hz: U32, 
                   duty_cycle_percent: float = 50.0, 
                   count: U32 = 10) -> None:
        """
        Generate a square wave signal.
        
        Args:
            channel: The channel to generate the signal on
            frequency_hz: Frequency in Hz
            duty_cycle_percent: Duty cycle (0-100%)
            count: Number of cycles to generate
            
        Raises:
            ValueError: If parameters are invalid or sample rate is not set
            
        Example:
            >>> # Generate a 1kHz square wave with 30% duty cycle for 5 cycles
            >>> SignalGenerator.square_wave(channel, 1000, 30.0, 5)
        """
        channel.add_square_wave(frequency_hz, duty_cycle_percent, count)
    
    @staticmethod
    def pulse_train(channel: SimulationChannelDescriptor,
                   pulse_width_s: float,
                   gap_s: float,
                   count: U32) -> None:
        """
        Generate a train of pulses.
        
        Args:
            channel: The channel to generate the signal on
            pulse_width_s: Width of each pulse in seconds
            gap_s: Gap between pulses in seconds
            count: Number of pulses to generate
            
        Raises:
            ValueError: If parameters are invalid or sample rate is not set
            
        Example:
            >>> # Generate 10 pulses of 1ms width with 2ms gaps
            >>> SignalGenerator.pulse_train(channel, 0.001, 0.002, 10)
        """
        channel.add_pulse_train(pulse_width_s, gap_s, count)
    
    @staticmethod
    def clock(channel: SimulationChannelDescriptor,
             frequency_hz: U32,
             count: U32,
             duty_cycle_percent: float = 50.0) -> None:
        """
        Generate a clock signal.
        
        Args:
            channel: The channel to generate the signal on
            frequency_hz: Clock frequency in Hz
            count: Number of clock cycles
            duty_cycle_percent: Duty cycle (0-100%)
            
        Raises:
            ValueError: If parameters are invalid or sample rate is not set
            
        Example:
            >>> # Generate a 10MHz clock signal for 100 cycles
            >>> SignalGenerator.clock(channel, 10_000_000, 100)
        """
        channel.add_clock(frequency_hz, count, duty_cycle_percent)
    
    @staticmethod
    def bit_pattern(channel: SimulationChannelDescriptor,
                   pattern: List[BitState],
                   samples_per_bit: U32) -> None:
        """
        Generate a specific bit pattern.
        
        Args:
            channel: The channel to generate the signal on
            pattern: List of bit states to generate
            samples_per_bit: Number of samples per bit
            
        Raises:
            ValueError: If pattern is empty or samples_per_bit is invalid
            
        Example:
            >>> # Generate a pattern with alternating HIGH and LOW bits
            >>> pattern = [BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW]
            >>> SignalGenerator.bit_pattern(channel, pattern, 100)
        """
        channel.add_bit_pattern(pattern, samples_per_bit)
    
    @staticmethod
    def byte_pattern(channel: SimulationChannelDescriptor,
                    data: bytes,
                    samples_per_bit: U32,
                    msb_first: bool = True) -> None:
        """
        Generate a pattern from a bytes object.
        
        Args:
            channel: The channel to generate the signal on
            data: Bytes to convert to a bit pattern
            samples_per_bit: Number of samples per bit
            msb_first: Whether bits should be sent MSB first
            
        Raises:
            ValueError: If data is empty or samples_per_bit is invalid
            
        Example:
            >>> # Generate a pattern from bytes with MSB first
            >>> SignalGenerator.byte_pattern(channel, b'\xA5\x3C', 100, True)
        """
        channel.add_byte_pattern(data, samples_per_bit, msb_first)
    
    @staticmethod
    def ramp(channel: SimulationChannelDescriptor,
            start_value: int,
            end_value: int,
            steps: int,
            bit_width: int,
            bits: int = 8) -> None:
        """
        Generate a digital ramp (stair) pattern.
        
        Args:
            channel: The channel to generate the signal on
            start_value: Starting value
            end_value: Ending value
            steps: Number of steps in the ramp
            bit_width: Width of each bit in samples
            bits: Number of bits to use (default: 8)
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> # Generate an 8-bit ramp from 0 to 255 in 8 steps
            >>> SignalGenerator.ramp(channel, 0, 255, 8, 100)
        """
        channel.add_ramp(start_value, end_value, steps, bit_width, bits)
    
    @staticmethod
    def counter(channel: SimulationChannelDescriptor,
               start_value: int,
               end_value: int,
               bit_width: int,
               bits: int = 8) -> None:
        """
        Generate a binary counter pattern.
        
        Args:
            channel: The channel to generate the signal on
            start_value: Starting value
            end_value: Ending value
            bit_width: Width of each bit in samples
            bits: Number of bits in each value
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> # Generate a 4-bit counter from 0 to 15
            >>> SignalGenerator.counter(channel, 0, 15, 100, 4)
        """
        channel.add_counter(start_value, end_value, bit_width, bits)
    
    @staticmethod
    def walking_ones(channel: SimulationChannelDescriptor,
                    bit_width: int,
                    bits: int = 8) -> None:
        """
        Generate a walking 1's pattern.
        
        Args:
            channel: The channel to generate the signal on
            bit_width: Width of each bit in samples
            bits: Number of bits in the pattern
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> # Generate an 8-bit walking 1's pattern
            >>> SignalGenerator.walking_ones(channel, 100, 8)
        """
        channel.add_walking_ones(bit_width, bits)
    
    @staticmethod
    def walking_zeros(channel: SimulationChannelDescriptor,
                     bit_width: int,
                     bits: int = 8) -> None:
        """
        Generate a walking 0's pattern.
        
        Args:
            channel: The channel to generate the signal on
            bit_width: Width of each bit in samples
            bits: Number of bits in the pattern
            
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> # Generate an 8-bit walking 0's pattern
            >>> SignalGenerator.walking_zeros(channel, 100, 8)
        """
        channel.add_walking_zeros(bit_width, bits)


# Protocol Configuration Classes

@dataclass
class UARTConfig:
    """
    Configuration for UART simulation.
    
    This class holds the parameters needed for UART signal generation.
    
    Attributes:
        baud_rate: UART baud rate in bits per second
        data_bits: Number of data bits (5-9)
        stop_bits: Number of stop bits (1, 1.5, or 2)
        parity: Parity mode (None, 'even', or 'odd')
        
    Example:
        >>> # Standard 9600 baud, 8 data bits, no parity, 1 stop bit (8N1)
        >>> config = UARTConfig(baud_rate=9600)
        >>> 
        >>> # 115200 baud, 8 data bits, even parity, 1 stop bit (8E1)
        >>> config = UARTConfig(baud_rate=115200, parity='even')
    """
    
    baud_rate: U32
    data_bits: int = 8
    stop_bits: float = 1.0
    parity: Optional[str] = None  # None, 'even', 'odd'
    
    def __post_init__(self):
        """Validate the configuration."""
        if self.baud_rate <= 0:
            raise ValueError(f"Invalid baud rate: {self.baud_rate}. Must be positive.")
            
        if self.data_bits not in (5, 6, 7, 8, 9):
            raise ValueError(f"Invalid data bits: {self.data_bits}. Must be 5-9.")
        
        if self.stop_bits not in (1.0, 1.5, 2.0):
            raise ValueError(f"Invalid stop bits: {self.stop_bits}. Must be 1, 1.5, or 2.")
        
        if self.parity not in (None, 'even', 'odd'):
            raise ValueError(f"Invalid parity: {self.parity}. Must be None, 'even', or 'odd'.")
    
    @property
    def frame_bits(self) -> float:
        """
        Calculate the total number of bits in a frame.
        
        Returns:
            Total bits including start, data, parity, and stop bits
        """
        return (
            1.0 +  # Start bit
            self.data_bits +
            (1.0 if self.parity is not None else 0.0) +
            self.stop_bits
        )
    
    @property
    def frame_time(self) -> float:
        """
        Calculate the time required for one frame at the configured baud rate.
        
        Returns:
            Frame time in seconds
        """
        return self.frame_bits / self.baud_rate
    
    def __str__(self) -> str:
        """Create a string representation of the configuration."""
        parity_str = self.parity[0].upper() if self.parity else 'N'
        stop_str = str(int(self.stop_bits)) if self.stop_bits == int(self.stop_bits) else str(self.stop_bits)
        return f"UART({self.baud_rate} baud, {self.data_bits}{parity_str}{stop_str})"


@dataclass
class I2CConfig:
    """
    Configuration for I2C simulation.
    
    This class holds the parameters needed for I2C signal generation.
    
    Attributes:
        clock_rate: I2C clock rate in Hz
        address_bits: Address bits (7 or 10)
        
    Example:
        >>> # Standard mode (100 kHz)
        >>> config = I2CConfig(clock_rate=100000)
        >>> 
        >>> # Fast mode (400 kHz)
        >>> config = I2CConfig(clock_rate=400000)
    """
    
    clock_rate: U32 = 100000  # Standard mode (100 kHz)
    address_bits: int = 7     # 7 or 10 bit addressing
    
    def __post_init__(self):
        """Validate the configuration."""
        if self.clock_rate <= 0:
            raise ValueError(f"Invalid clock rate: {self.clock_rate}. Must be positive.")
            
        if self.address_bits not in (7, 10):
            raise ValueError(f"Invalid address bits: {self.address_bits}. Must be 7 or 10.")
    
    @property
    def standard_mode(self) -> bool:
        """Check if this is standard mode (100 kHz)."""
        return 90000 <= self.clock_rate <= 110000
    
    @property
    def fast_mode(self) -> bool:
        """Check if this is fast mode (400 kHz)."""
        return 350000 <= self.clock_rate <= 450000
    
    @property
    def fast_mode_plus(self) -> bool:
        """Check if this is fast mode plus (1 MHz)."""
        return 900000 <= self.clock_rate <= 1100000
    
    @property
    def high_speed_mode(self) -> bool:
        """Check if this is high-speed mode (3.4 MHz)."""
        return self.clock_rate > 1100000
    
    def __str__(self) -> str:
        """Create a string representation of the configuration."""
        mode = "Standard"
        if self.fast_mode:
            mode = "Fast"
        elif self.fast_mode_plus:
            mode = "Fast+"
        elif self.high_speed_mode:
            mode = "High-Speed"
            
        return f"I2C({self.clock_rate} Hz, {mode}, {self.address_bits}-bit addressing)"


@dataclass
class SPIConfig:
    """
    Configuration for SPI simulation.
    
    This class holds the parameters needed for SPI signal generation.
    
    Attributes:
        clock_rate: SPI clock rate in Hz
        mode: SPI mode (0-3)
        msb_first: Whether MSB is transmitted first
        bits_per_word: Number of bits per word
        cs_active_low: Whether chip select is active low
        
    Example:
        >>> # Standard SPI mode 0 at 1 MHz
        >>> config = SPIConfig(clock_rate=1000000, mode=0)
        >>> 
        >>> # SPI mode 3 at 5 MHz, LSB first
        >>> config = SPIConfig(clock_rate=5000000, mode=3, msb_first=False)
    """
    
    clock_rate: U32
    mode: int = 0  # SPI modes 0-3
    msb_first: bool = True
    bits_per_word: int = 8
    cs_active_low: bool = True
    
    def __post_init__(self):
        """Validate the configuration and derive CPOL/CPHA."""
        if self.clock_rate <= 0:
            raise ValueError(f"Invalid clock rate: {self.clock_rate}. Must be positive.")
            
        if self.mode not in (0, 1, 2, 3):
            raise ValueError(f"Invalid SPI mode: {self.mode}. Must be 0-3.")
            
        if self.bits_per_word < 1:
            raise ValueError(f"Invalid bits per word: {self.bits_per_word}. Must be positive.")
        
        # SPI mode to CPOL/CPHA mapping:
        # Mode 0: CPOL=0, CPHA=0
        # Mode 1: CPOL=0, CPHA=1
        # Mode 2: CPOL=1, CPHA=0
        # Mode 3: CPOL=1, CPHA=1
        self.cpol = (self.mode & 2) > 0
        self.cpha = (self.mode & 1) > 0
    
    @property
    def idle_state(self) -> BitState:
        """Get the idle (inactive) state of the clock line."""
        return BitState.HIGH if self.cpol else BitState.LOW
    
    @property
    def active_state(self) -> BitState:
        """Get the active state of the clock line."""
        return BitState.LOW if self.cpol else BitState.HIGH
    
    @property
    def cs_idle_state(self) -> BitState:
        """Get the idle (inactive) state of the chip select line."""
        return BitState.HIGH if self.cs_active_low else BitState.LOW
    
    @property
    def cs_active_state(self) -> BitState:
        """Get the active state of the chip select line."""
        return BitState.LOW if self.cs_active_low else BitState.HIGH
    
    def __str__(self) -> str:
        """Create a string representation of the configuration."""
        direction = "MSB" if self.msb_first else "LSB"
        cs_type = "active-low" if self.cs_active_low else "active-high"
        return f"SPI({self.clock_rate} Hz, mode {self.mode}, {direction} first, {self.bits_per_word} bits, {cs_type})"


@dataclass
class CANConfig:
    """
    Configuration for CAN simulation.
    
    This class holds the parameters needed for CAN signal generation.
    
    Attributes:
        bit_rate: CAN bit rate in bits per second
        sample_point: Sample point as percentage of bit time (0-100)
        extended_id: Whether to use extended identifiers (29-bit)
        
    Example:
        >>> # Standard CAN at 500 kbps
        >>> config = CANConfig(bit_rate=500000)
        >>> 
        >>> # CAN with extended IDs at 1 Mbps
        >>> config = CANConfig(bit_rate=1000000, extended_id=True)
    """
    
    bit_rate: U32
    sample_point: float = 75.0  # Percentage (0-100)
    extended_id: bool = False
    
    def __post_init__(self):
        """Validate the configuration."""
        if self.bit_rate <= 0:
            raise ValueError(f"Invalid bit rate: {self.bit_rate}. Must be positive.")
            
        if not 0 < self.sample_point < 100:
            raise ValueError(f"Invalid sample point: {self.sample_point}. Must be between 0 and 100.")
    
    @property
    def bit_time(self) -> float:
        """Calculate the time of one bit in seconds."""
        return 1.0 / self.bit_rate
    
    @property
    def max_identifier(self) -> int:
        """Get the maximum valid identifier value."""
        return 0x1FFFFFFF if self.extended_id else 0x7FF
    
    def validate_identifier(self, identifier: int) -> None:
        """
        Validate a CAN identifier.
        
        Args:
            identifier: CAN identifier value
            
        Raises:
            ValueError: If identifier is invalid
        """
        max_id = self.max_identifier
        if not 0 <= identifier <= max_id:
            raise ValueError(f"Invalid CAN identifier: {identifier}. Must be 0-{max_id}.")
    
    def __str__(self) -> str:
        """Create a string representation of the configuration."""
        id_type = "Extended ID" if self.extended_id else "Standard ID"
        return f"CAN({self.bit_rate} bps, {id_type}, sample point {self.sample_point}%)"


@dataclass
class ManchesterConfig:
    """
    Configuration for Manchester encoding simulation.
    
    This class holds the parameters needed for Manchester signal generation.
    
    Attributes:
        bit_rate: Bit rate in bits per second
        standard_encoding: Whether to use IEEE 802.3 (True) or Thomas/G.E. (False) encoding
        msb_first: Whether MSB is transmitted first
        
    Example:
        >>> # Standard IEEE 802.3 Manchester at 1 Mbps
        >>> config = ManchesterConfig(bit_rate=1000000)
        >>> 
        >>> # Thomas/G.E. Manchester at 100 kbps, LSB first
        >>> config = ManchesterConfig(bit_rate=100000, standard_encoding=False, msb_first=False)
    """
    
    bit_rate: U32
    standard_encoding: bool = True
    msb_first: bool = True
    
    def __post_init__(self):
        """Validate the configuration."""
        if self.bit_rate <= 0:
            raise ValueError(f"Invalid bit rate: {self.bit_rate}. Must be positive.")
    
    @property
    def encoding_type(self) -> str:
        """Get the name of the encoding type."""
        return "IEEE 802.3" if self.standard_encoding else "Thomas/G.E."
    
    @property
    def bit_time(self) -> float:
        """Calculate the time of one bit in seconds."""
        return 1.0 / self.bit_rate
    
    def __str__(self) -> str:
        """Create a string representation of the configuration."""
        bit_order = "MSB first" if self.msb_first else "LSB first"
        return f"Manchester({self.bit_rate} bps, {self.encoding_type}, {bit_order})"


# Simulation Builder Classes

class UARTSimulation:
    """
    Builder for UART simulations.
    
    This class provides a fluent interface for building UART simulations.
    
    Attributes:
        tx_channel: Transmit channel descriptor
        rx_channel: Receive channel descriptor (optional)
        config: UART configuration
        sample_rate_hz: Sample rate in Hz
        
    Example:
        >>> # Create a UART simulation
        >>> uart = UARTSimulation(tx_channel, config=UARTConfig(baud_rate=9600))
        >>> uart.add_tx_data("Hello, World!")
        >>> uart.generate()
    """
    
    def __init__(self, 
                tx_channel: SimulationChannelDescriptor,
                rx_channel: Optional[SimulationChannelDescriptor] = None,
                config: Optional[UARTConfig] = None,
                sample_rate_hz: U32 = 0):
        """
        Initialize a UART simulation.
        
        Args:
            tx_channel: The TX channel (from master/DUT)
            rx_channel: Optional RX channel (to master/DUT)
            config: UART configuration
            sample_rate_hz: Sample rate in Hz (if 0, use channel's rate)
            
        Raises:
            ValueError: If tx_channel is None
        """
        if tx_channel is None:
            raise ValueError("TX channel cannot be None")
            
        self.tx_channel = tx_channel
        self.rx_channel = rx_channel
        self.config = config or UARTConfig(baud_rate=9600)
        self.sample_rate_hz = sample_rate_hz or tx_channel.sample_rate_hz
        self.tx_data = bytearray()
        self.rx_data = bytearray()
    
    def add_tx_data(self, data: Union[bytes, bytearray, str]) -> 'UARTSimulation':
        """
        Add data to transmit on the TX channel.
        
        Args:
            data: Data to transmit (will be converted to bytes)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> uart.add_tx_data("Hello")
            >>> uart.add_tx_data(b"\x01\x02\x03")
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        self.tx_data.extend(data)
        return self
    
    def add_rx_data(self, data: Union[bytes, bytearray, str]) -> 'UARTSimulation':
        """
        Add data to receive on the RX channel.
        
        Args:
            data: Data to receive (will be converted to bytes)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If RX channel is not configured
            
        Example:
            >>> uart.add_rx_data("ACK")
        """
        if not self.rx_channel:
            raise ValueError("RX channel not configured")
            
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        self.rx_data.extend(data)
        return self
    
    def generate(self) -> None:
        """
        Generate the UART simulation.
        
        Creates the UART signals according to the configured parameters
        and data.
        
        Raises:
            ValueError: If sample rate is not set
            
        Example:
            >>> uart.add_tx_data("Hello")
            >>> uart.generate()
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for UART simulation")
            
        # Calculate timing parameters
        bit_width = self.sample_rate_hz // self.config.baud_rate
        
        # Generate TX data
        if self.tx_data:
            self._generate_uart_data(self.tx_channel, self.tx_data, bit_width)
        
        # Generate RX data
        if self.rx_channel and self.rx_data:
            self._generate_uart_data(self.rx_channel, self.rx_data, bit_width)
    
    def _generate_uart_data(self, 
                          channel: SimulationChannelDescriptor, 
                          data: bytes, 
                          bit_width: int) -> None:
        """
        Generate UART data on a channel.
        
        Args:
            channel: Channel to generate data on
            data: Data bytes to send
            bit_width: Width of each bit in samples
        """
        # Set channel to idle state (HIGH)
        channel.transition_if_needed(BitState.HIGH)
        
        # Initial idle time (10-bit width)
        channel.advance(bit_width * 10)
        
        for byte in data:
            # Start bit (always LOW)
            channel.transition_if_needed(BitState.LOW)
            channel.advance(bit_width)
            
            # Data bits (LSB first)
            for i in range(self.config.data_bits):
                bit_value = (byte >> i) & 0x01
                bit_state = BitState.HIGH if bit_value else BitState.LOW
                channel.transition_if_needed(bit_state)
                channel.advance(bit_width)
            
            # Parity bit (if enabled)
            if self.config.parity:
                # Count 1-bits for parity
                bit_count = bin(byte).count('1')
                if self.config.parity == 'even':
                    # Even parity: ensure even number of 1s
                    parity_bit = BitState.HIGH if bit_count % 2 else BitState.LOW
                else:  # 'odd'
                    # Odd parity: ensure odd number of 1s
                    parity_bit = BitState.LOW if bit_count % 2 else BitState.HIGH
                    
                channel.transition_if_needed(parity_bit)
                channel.advance(bit_width)
            
            # Stop bit(s)
            channel.transition_if_needed(BitState.HIGH)
            
            # Calculate stop bit timing
            if self.config.stop_bits == 1.0:
                channel.advance(bit_width)
            elif self.config.stop_bits == 1.5:
                channel.advance(bit_width + bit_width // 2)
            else:  # 2.0
                channel.advance(bit_width * 2)
            
            # Inter-byte gap (1 bit width)
            channel.advance(bit_width)
        
        # Final idle time
        channel.advance(bit_width * 10)


class I2CSimulation:
    """
    Builder for I2C simulations.
    
    This class provides a fluent interface for building I2C simulations.
    
    Attributes:
        scl_channel: SCL (clock) channel descriptor
        sda_channel: SDA (data) channel descriptor
        config: I2C configuration
        sample_rate_hz: Sample rate in Hz
        
    Example:
        >>> # Create an I2C simulation
        >>> i2c = I2CSimulation(scl_channel, sda_channel, config=I2CConfig(clock_rate=100000))
        >>> i2c.add_write(0x50, b'\x00\x10\x20\x30')  # Write to device 0x50
        >>> i2c.generate()
    """
    
    def __init__(self,
                scl_channel: SimulationChannelDescriptor,
                sda_channel: SimulationChannelDescriptor,
                config: Optional[I2CConfig] = None,
                sample_rate_hz: U32 = 0):
        """
        Initialize an I2C simulation.
        
        Args:
            scl_channel: The SCL (clock) channel
            sda_channel: The SDA (data) channel
            config: I2C configuration
            sample_rate_hz: Sample rate in Hz (if 0, use channel's rate)
            
        Raises:
            ValueError: If channels are None or sample rate is incompatible
        """
        if scl_channel is None or sda_channel is None:
            raise ValueError("SCL and SDA channels cannot be None")
            
        self.scl_channel = scl_channel
        self.sda_channel = sda_channel
        self.config = config or I2CConfig()
        
        # Use higher of the two sample rates if they differ
        scl_rate = scl_channel.sample_rate_hz
        sda_rate = sda_channel.sample_rate_hz
        
        if scl_rate != sda_rate and scl_rate > 0 and sda_rate > 0:
            warnings.warn(f"SCL and SDA channels have different sample rates "
                         f"({scl_rate} Hz and {sda_rate} Hz). "
                         f"Using the higher rate.")
                         
        self.sample_rate_hz = sample_rate_hz or max(scl_rate, sda_rate)
        
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for I2C simulation")
            
        self.transactions = []
    
    def add_transaction(self, 
                       address: int, 
                       data: Union[bytes, bytearray, List[int]], 
                       is_read: bool = False) -> 'I2CSimulation':
        """
        Add an I2C transaction.
        
        Args:
            address: 7-bit or 10-bit device address
            data: Data bytes to transmit or receive
            is_read: True for read operation, False for write
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If address is invalid
            
        Example:
            >>> # Add a write transaction to device 0x50
            >>> i2c.add_transaction(0x50, b'\x00\x10\x20\x30', is_read=False)
        """
        # Validate address
        if self.config.address_bits == 7 and address > 0x7F:
            raise ValueError(f"Invalid 7-bit address: 0x{address:X}. Must be 0-0x7F.")
        elif self.config.address_bits == 10 and address > 0x3FF:
            raise ValueError(f"Invalid 10-bit address: 0x{address:X}. Must be 0-0x3FF.")
        
        # Convert list to bytes if needed
        if isinstance(data, list):
            data = bytes(data)
        
        # Store the transaction
        self.transactions.append({
            'address': address,
            'data': data if isinstance(data, (bytes, bytearray)) else bytes(data),
            'is_read': is_read
        })
        
        return self
    
    def add_write(self, address: int, data: Union[bytes, bytearray, List[int]]) -> 'I2CSimulation':
        """
        Add an I2C write transaction.
        
        Args:
            address: 7-bit or 10-bit device address
            data: Data bytes to write
            
        Returns:
            Self for method chaining
            
        Example:
            >>> # Write to EEPROM at address 0x50
            >>> i2c.add_write(0x50, b'\x00\x10\x20\x30')
        """
        return self.add_transaction(address, data, is_read=False)
    
    def add_read(self, address: int, data: Union[bytes, bytearray, List[int]]) -> 'I2CSimulation':
        """
        Add an I2C read transaction.
        
        Args:
            address: 7-bit or 10-bit device address
            data: Data bytes to read (from slave)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> # Read from EEPROM at address 0x50
            >>> i2c.add_read(0x50, b'\x10\x20\x30\x40')
        """
        return self.add_transaction(address, data, is_read=True)
    
    def add_write_read(self, address: int, 
                      write_data: Union[bytes, bytearray, List[int]],
                      read_data: Union[bytes, bytearray, List[int]]) -> 'I2CSimulation':
        """
        Add an I2C write followed by read transaction (with repeated start).
        
        Args:
            address: 7-bit or 10-bit device address
            write_data: Data bytes to write
            read_data: Data bytes to read
            
        Returns:
            Self for method chaining
            
        Example:
            >>> # Write register address then read data from sensor
            >>> i2c.add_write_read(0x68, b'\x3B', b'\x41\x42\x43\x44')
        """
        # Add write transaction
        self.add_write(address, write_data)
        
        # Mark for repeated start instead of stop
        if len(self.transactions) > 0:
            self.transactions[-1]['repeated_start'] = True
        
        # Add read transaction
        return self.add_read(address, read_data)
    
    def generate(self) -> None:
        """
        Generate the I2C simulation.
        
        Creates the I2C signals according to the configured parameters
        and transactions.
        
        Raises:
            ValueError: If sample rate is not set
            
        Example:
            >>> i2c.add_write(0x50, b'\x00\x10\x20\x30')
            >>> i2c.generate()
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for I2C simulation")
            
        bit_width = self.sample_rate_hz // self.config.clock_rate
        
        # Initial idle state (both lines HIGH)
        self.scl_channel.transition_if_needed(BitState.HIGH)
        self.sda_channel.transition_if_needed(BitState.HIGH)
        
        # Initial idle time
        self.scl_channel.advance(bit_width * 4)
        self.sda_channel.advance(bit_width * 4)
        
        # Generate each transaction
        for i, transaction in enumerate(self.transactions):
            address = transaction['address']
            data = transaction['data']
            is_read = transaction['is_read']
            repeated_start = transaction.get('repeated_start', False)
            
            # START condition
            self._generate_start_condition(bit_width)
            
            # Address byte (7-bit address + R/W bit)
            address_byte = (address << 1) | (1 if is_read else 0)
            self._generate_byte_transfer(address_byte, bit_width)
            
            # Get ACK from slave
            self._generate_ack(bit_width)
            
            # Data bytes
            for byte_idx, byte in enumerate(data):
                self._generate_byte_transfer(byte, bit_width)
                
                # ACK/NACK bit
                # For write: slave sends ACK (SDA LOW)
                # For last read byte: master sends NACK (SDA HIGH)
                # For non-last read byte: master sends ACK (SDA LOW)
                is_last_byte = byte_idx == len(data) - 1
                send_nack = is_read and is_last_byte
                
                self._generate_ack(bit_width, send_nack)
            
            # STOP condition or repeated START
            if not repeated_start or i == len(self.transactions) - 1:
                self._generate_stop_condition(bit_width)
            else:
                # Extra clock cycle before repeated START
                self.scl_channel.transition_if_needed(BitState.LOW)
                self.scl_channel.advance(bit_width // 2)
                self.sda_channel.transition_if_needed(BitState.HIGH)
                self.sda_channel.advance(bit_width // 2)
                
                self.scl_channel.transition_if_needed(BitState.HIGH)
                self.scl_channel.advance(bit_width)
                self.sda_channel.advance(bit_width)
            
            # Idle time between transactions
            if i < len(self.transactions) - 1 and not repeated_start:
                self.sda_channel.advance(bit_width * 2)
                self.scl_channel.advance(bit_width * 2)
    
    def _generate_start_condition(self, bit_width: int) -> None:
        """
        Generate an I2C START condition.
        
        Args:
            bit_width: Width of each bit in samples
        """
        # Ensure both lines are HIGH
        self.scl_channel.transition_if_needed(BitState.HIGH)
        self.sda_channel.transition_if_needed(BitState.HIGH)
        self.scl_channel.advance(bit_width // 2)
        self.sda_channel.advance(bit_width // 2)
        
        # SDA transitions from HIGH to LOW while SCL is HIGH (START)
        self.sda_channel.transition_if_needed(BitState.LOW)
        self.sda_channel.advance(bit_width // 2)
        self.scl_channel.advance(bit_width // 2)
        
        # SCL goes LOW to prepare for first data bit
        self.scl_channel.transition_if_needed(BitState.LOW)
        self.scl_channel.advance(bit_width // 2)
        self.sda_channel.advance(bit_width // 2)
    
    def _generate_stop_condition(self, bit_width: int) -> None:
        """
        Generate an I2C STOP condition.
        
        Args:
            bit_width: Width of each bit in samples
        """
        # Ensure SCL is LOW and SDA is LOW
        self.scl_channel.transition_if_needed(BitState.LOW)
        self.sda_channel.transition_if_needed(BitState.LOW)
        self.scl_channel.advance(bit_width // 2)
        self.sda_channel.advance(bit_width // 2)
        
        # SCL goes HIGH
        self.scl_channel.transition_if_needed(BitState.HIGH)
        self.scl_channel.advance(bit_width // 2)
        self.sda_channel.advance(bit_width // 2)
        
        # SDA transitions from LOW to HIGH while SCL is HIGH (STOP)
        self.sda_channel.transition_if_needed(BitState.HIGH)
        self.sda_channel.advance(bit_width // 2)
        self.scl_channel.advance(bit_width // 2)
        
        # Idle time after STOP
        self.sda_channel.advance(bit_width)
        self.scl_channel.advance(bit_width)
    
    def _generate_byte_transfer(self, byte: int, bit_width: int) -> None:
        """
        Generate an I2C byte transfer.
        
        Args:
            byte: Byte value to transfer
            bit_width: Width of each bit in samples
        """
        # Send 8 data bits (MSB first)
        for i in range(7, -1, -1):
            bit_value = (byte >> i) & 0x01
            
            # SCL is LOW while data changes
            self.scl_channel.transition_if_needed(BitState.LOW)
            self.scl_channel.advance(bit_width // 4)
            
            # Set SDA value
            self.sda_channel.transition_if_needed(BitState.HIGH if bit_value else BitState.LOW)
            self.sda_channel.advance(bit_width // 4)
            
            # SCL goes HIGH (data is valid)
            self.scl_channel.transition_if_needed(BitState.HIGH)
            self.scl_channel.advance(bit_width // 2)
            self.sda_channel.advance(bit_width // 2)
            
            # SCL goes LOW again
            self.scl_channel.transition_if_needed(BitState.LOW)
            self.scl_channel.advance(bit_width // 4)
            self.sda_channel.advance(bit_width // 4)
    
    def _generate_ack(self, bit_width: int, nack: bool = False) -> None:
        """
        Generate an I2C ACK or NACK bit.
        
        Args:
            bit_width: Width of each bit in samples
            nack: True to generate NACK, False for ACK
        """
        # SCL is LOW while data changes
        self.scl_channel.transition_if_needed(BitState.LOW)
        self.scl_channel.advance(bit_width // 4)
        
        # Set SDA value (LOW for ACK, HIGH for NACK)
        self.sda_channel.transition_if_needed(BitState.HIGH if nack else BitState.LOW)
        self.sda_channel.advance(bit_width // 4)
        
        # SCL goes HIGH (ACK/NACK is valid)
        self.scl_channel.transition_if_needed(BitState.HIGH)
        self.scl_channel.advance(bit_width // 2)
        self.sda_channel.advance(bit_width // 2)
        
        # SCL goes LOW again
        self.scl_channel.transition_if_needed(BitState.LOW)
        self.scl_channel.advance(bit_width // 4)
        self.sda_channel.advance(bit_width // 4)


class SPISimulation:
    """
    Builder for SPI simulations.
    
    This class provides a fluent interface for building SPI simulations.
    
    Attributes:
        sck_channel: SCK (clock) channel descriptor
        mosi_channel: MOSI (Master Out Slave In) channel descriptor
        miso_channel: MISO (Master In Slave Out) channel descriptor
        cs_channel: CS (Chip Select) channel descriptor
        config: SPI configuration
        sample_rate_hz: Sample rate in Hz
        
    Example:
        >>> # Create an SPI simulation
        >>> spi = SPISimulation(sck, mosi, miso, cs, config=SPIConfig(clock_rate=1000000))
        >>> spi.add_transaction(b'\x01\x02\x03', b'\xA1\xA2\xA3')
        >>> spi.generate()
    """
    
    def __init__(self,
                sck_channel: SimulationChannelDescriptor,
                mosi_channel: Optional[SimulationChannelDescriptor] = None,
                miso_channel: Optional[SimulationChannelDescriptor] = None,
                cs_channel: Optional[SimulationChannelDescriptor] = None,
                config: Optional[SPIConfig] = None,
                sample_rate_hz: U32 = 0):
        """
        Initialize an SPI simulation.
        
        Args:
            sck_channel: The SCK (clock) channel
            mosi_channel: Optional MOSI (Master Out Slave In) channel
            miso_channel: Optional MISO (Master In Slave Out) channel
            cs_channel: Optional CS (Chip Select) channel
            config: SPI configuration
            sample_rate_hz: Sample rate in Hz (if 0, use channel's rate)
            
        Raises:
            ValueError: If SCK channel is None
        """
        if sck_channel is None:
            raise ValueError("SCK channel cannot be None")
            
        self.sck_channel = sck_channel
        self.mosi_channel = mosi_channel
        self.miso_channel = miso_channel
        self.cs_channel = cs_channel
        self.config = config or SPIConfig(clock_rate=1000000)
        self.sample_rate_hz = sample_rate_hz or sck_channel.sample_rate_hz
        
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for SPI simulation")
            
        self.transactions = []
    
    def add_transaction(self, 
                       mosi_data: Optional[Union[bytes, bytearray, List[int]]] = None,
                       miso_data: Optional[Union[bytes, bytearray, List[int]]] = None) -> 'SPISimulation':
        """
        Add an SPI transaction.
        
        Args:
            mosi_data: Data to send from master to slave (optional)
            miso_data: Data to send from slave to master (optional)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If both mosi_data and miso_data are None
            
        Example:
            >>> # Add an SPI transaction with both MOSI and MISO data
            >>> spi.add_transaction(b'\x01\x02\x03', b'\xA1\xA2\xA3')
        """
        if mosi_data is None and miso_data is None:
            raise ValueError("At least one of mosi_data or miso_data must be provided")
        
        # Convert list to bytes if needed
        if isinstance(mosi_data, list):
            mosi_data = bytes(mosi_data) if mosi_data is not None else None
        
        if isinstance(miso_data, list):
            miso_data = bytes(miso_data) if miso_data is not None else None
        
        self.transactions.append({
            'mosi_data': mosi_data if mosi_data is not None else bytes(),
            'miso_data': miso_data if miso_data is not None else bytes()
        })
        
        return self
    
    def generate(self) -> None:
        """
        Generate the SPI simulation.
        
        Creates the SPI signals according to the configured parameters
        and transactions.
        
        Raises:
            ValueError: If sample rate is not set
            
        Example:
            >>> spi.add_transaction(b'\x01\x02\x03', b'\xA1\xA2\xA3')
            >>> spi.generate()
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set for SPI simulation")
            
        bit_width = self.sample_rate_hz // self.config.clock_rate
        
        # Initial idle state
        idle_state = BitState.HIGH if self.config.cpol else BitState.LOW
        self.sck_channel.transition_if_needed(idle_state)
        
        if self.cs_channel:
            self.cs_channel.transition_if_needed(
                BitState.HIGH if self.config.cs_active_low else BitState.LOW
            )
        
        # Process each transaction
        for i, transaction in enumerate(self.transactions):
            mosi_data = transaction['mosi_data']
            miso_data = transaction['miso_data']
            
            # Activate chip select
            if self.cs_channel:
                self.cs_channel.transition_if_needed(
                    BitState.LOW if self.config.cs_active_low else BitState.HIGH
                )
                
                # Short delay after CS activation
                if self.cs_channel:
                    self.cs_channel.advance(bit_width // 2)
                if self.mosi_channel:
                    self.mosi_channel.advance(bit_width // 2)
                if self.miso_channel:
                    self.miso_channel.advance(bit_width // 2)
                self.sck_channel.advance(bit_width // 2)
            
            # Get max length of data
            max_bytes = max(len(mosi_data), len(miso_data))
            
            # Process each byte
            for byte_idx in range(max_bytes):
                mosi_byte = mosi_data[byte_idx] if byte_idx < len(mosi_data) else 0
                miso_byte = miso_data[byte_idx] if byte_idx < len(miso_data) else 0
                
                # Process each bit in the byte
                for bit_idx in range(self.config.bits_per_word):
                    # Determine bit position based on MSB/LSB first
                    actual_bit_idx = 7 - bit_idx if self.config.msb_first else bit_idx
                    
                    # Extract bit values
                    mosi_bit = (mosi_byte >> actual_bit_idx) & 0x01 if self.mosi_channel else 0
                    miso_bit = (miso_byte >> actual_bit_idx) & 0x01 if self.miso_channel else 0
                    
                    self._generate_spi_bit(mosi_bit, miso_bit, bit_width)
                
                # Inter-byte gap (optional, depending on protocol)
                if byte_idx < max_bytes - 1:
                    if self.cs_channel:
                        self.cs_channel.advance(bit_width // 4)
                    if self.mosi_channel:
                        self.mosi_channel.advance(bit_width // 4)
                    if self.miso_channel:
                        self.miso_channel.advance(bit_width // 4)
                    self.sck_channel.advance(bit_width // 4)
            
            # Deactivate chip select
            if self.cs_channel:
                self.cs_channel.transition_if_needed(
                    BitState.HIGH if self.config.cs_active_low else BitState.LOW
                )
                
                # Idle time after transaction
                idle_time = bit_width * 2
                if self.cs_channel:
                    self.cs_channel.advance(idle_time)
                if self.mosi_channel:
                    self.mosi_channel.advance(idle_time)
                if self.miso_channel:
                    self.miso_channel.advance(idle_time)
                self.sck_channel.advance(idle_time)
    
    def _generate_spi_bit(self, mosi_bit: int, miso_bit: int, bit_width: int) -> None:
        """
        Generate a single SPI bit.
        
        Args:
            mosi_bit: MOSI bit value (0 or 1)
            miso_bit: MISO bit value (0 or 1)
            bit_width: Width of each bit in samples
        """
        if self.config.cpha == 0:
            # CPHA=0: Data is valid on leading edge
            
            # Set data lines before clock toggle
            if self.mosi_channel:
                self.mosi_channel.transition_if_needed(
                    BitState.HIGH if mosi_bit else BitState.LOW
                )
            
            if self.miso_channel:
                self.miso_channel.transition_if_needed(
                    BitState.HIGH if miso_bit else BitState.LOW
                )
            
            # Setup time before clock edge
            setup_time = bit_width // 4
            if self.cs_channel:
                self.cs_channel.advance(setup_time)
            if self.mosi_channel:
                self.mosi_channel.advance(setup_time)
            if self.miso_channel:
                self.miso_channel.advance(setup_time)
            self.sck_channel.advance(setup_time)
            
            # First clock edge (leading)
            self.sck_channel.transition()
            
            # Hold time after clock edge
            hold_time = bit_width // 2
            if self.cs_channel:
                self.cs_channel.advance(hold_time)
            if self.mosi_channel:
                self.mosi_channel.advance(hold_time)
            if self.miso_channel:
                self.miso_channel.advance(hold_time)
            self.sck_channel.advance(hold_time)
            
            # Second clock edge (trailing)
            self.sck_channel.transition()
            
            # Final time
            final_time = bit_width // 4
            if self.cs_channel:
                self.cs_channel.advance(final_time)
            if self.mosi_channel:
                self.mosi_channel.advance(final_time)
            if self.miso_channel:
                self.miso_channel.advance(final_time)
            self.sck_channel.advance(final_time)
        else:
            # CPHA=1: Data is valid on trailing edge
            
            # First clock edge
            self.sck_channel.transition()
            
            # Setup time after first clock edge
            setup_time = bit_width // 4
            if self.cs_channel:
                self.cs_channel.advance(setup_time)
            if self.mosi_channel:
                self.mosi_channel.advance(setup_time)
            if self.miso_channel:
                self.miso_channel.advance(setup_time)
            self.sck_channel.advance(setup_time)
            
            # Set data after first clock edge
            if self.mosi_channel:
                self.mosi_channel.transition_if_needed(
                    BitState.HIGH if mosi_bit else BitState.LOW
                )
            
            if self.miso_channel:
                self.miso_channel.transition_if_needed(
                    BitState.HIGH if miso_bit else BitState.LOW
                )
            
            # Hold time before second edge
            hold_time = bit_width // 2
            if self.cs_channel:
                self.cs_channel.advance(hold_time)
            if self.mosi_channel:
                self.mosi_channel.advance(hold_time)
            if self.miso_channel:
                self.miso_channel.advance(hold_time)
            self.sck_channel.advance(hold_time)
            
            # Second clock edge
            self.sck_channel.transition()
            
            # Final time
            final_time = bit_width // 4
            if self.cs_channel:
                self.cs_channel.advance(final_time)
            if self.mosi_channel:
                self.mosi_channel.advance(final_time)
            if self.miso_channel:
                self.miso_channel.advance(final_time)
            self.sck_channel.advance(final_time)


class SimulationManager:
    """
    High-level manager for simulations.
    
    This class provides a convenient interface for creating and managing
    simulation channels and protocols.
    
    Attributes:
        sample_rate_hz: Default sample rate in Hz for all channels
        
    Example:
        >>> # Create a simulation manager with 10 MHz sample rate
        >>> manager = SimulationManager(sample_rate=10_000_000)
        >>> 
        >>> # Add a UART simulation
        >>> manager.add_uart(tx_channel=Channel(0, "TX"), data=b"Hello, World!", baud_rate=115200)
        >>> 
        >>> # Run the simulation
        >>> manager.run()
    """
    
    def __init__(self, sample_rate: U32 = 10_000_000):
        """
        Initialize a simulation manager.
        
        Args:
            sample_rate: Default sample rate in Hz for all channels
            
        Raises:
            ValueError: If sample rate is not positive
        """
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
            
        self.group = _ka.SimulationChannelDescriptorGroup()
        self.sample_rate_hz = sample_rate
        self.channels = {}
        self.simulations = []
        self._channel_descriptors = {}
    
    def get_channel(self, channel_index: int, channel_name: Optional[str] = None) -> SimulationChannelDescriptor:
        """
        Get or create a simulation channel.
        
        Args:
            channel_index: Channel index
            channel_name: Optional channel name
            
        Returns:
            The simulation channel descriptor
            
        Example:
            >>> # Get channel 0 named "CLK"
            >>> clk_channel = manager.get_channel(0, "CLK")
        """
        if channel_index in self._channel_descriptors:
            return self._channel_descriptors[channel_index]
        
        # Create a new channel
        ka_channel = Channel(channel_index, channel_name or f"Channel_{channel_index}")
        descriptor = self.group.add(ka_channel, self.sample_rate_hz, BitState.LOW)
        
        # Create the wrapper
        sim_channel = SimulationChannelDescriptor(
            descriptor, 
            name=channel_name or f"Channel_{channel_index}", 
            channel_index=channel_index, 
            sample_rate_hz=self.sample_rate_hz
        )
        
        self._channel_descriptors[channel_index] = sim_channel
        return sim_channel
    
    def add_uart(self, 
                tx_channel: Union[Channel, int],
                data: Union[bytes, bytearray, str],
                baud_rate: U32 = 9600,
                rx_channel: Optional[Union[Channel, int]] = None,
                rx_data: Optional[Union[bytes, bytearray, str]] = None,
                parity: Optional[str] = None,
                stop_bits: float = 1.0,
                data_bits: int = 8) -> UARTSimulation:
        """
        Add a UART simulation.
        
        Args:
            tx_channel: TX channel (index or Channel object)
            data: TX data to send
            baud_rate: Baud rate in bits per second
            rx_channel: Optional RX channel (index or Channel object)
            rx_data: Optional RX data to receive
            parity: Parity mode (None, 'even', or 'odd')
            stop_bits: Number of stop bits (1, 1.5, or 2)
            data_bits: Number of data bits (5-9)
            
        Returns:
            The UART simulation object
            
        Example:
            >>> # Create a UART simulation at 115200 baud
            >>> uart = manager.add_uart(0, "Hello, World!", 115200)
        """
        # Convert channel index to Channel object if needed
        if isinstance(tx_channel, int):
            tx_channel = Channel(tx_channel, f"UART_TX_{tx_channel}")
        
        # Get simulation channels
        tx_sim_channel = self.get_channel(tx_channel.index, tx_channel.name)
        
        rx_sim_channel = None
        if rx_channel is not None:
            if isinstance(rx_channel, int):
                rx_channel = Channel(rx_channel, f"UART_RX_{rx_channel}")
            rx_sim_channel = self.get_channel(rx_channel.index, rx_channel.name)
        
        # Create UART config
        config = UARTConfig(
            baud_rate=baud_rate,
            data_bits=data_bits,
            stop_bits=stop_bits,
            parity=parity
        )
        
        # Create UART simulation
        uart_sim = UARTSimulation(tx_sim_channel, rx_sim_channel, config, self.sample_rate_hz)
        uart_sim.add_tx_data(data)
        
        if rx_sim_channel and rx_data:
            uart_sim.add_rx_data(rx_data)
        
        # Register simulation
        self.simulations.append(uart_sim)
        
        return uart_sim
    
    def add_i2c(self,
               scl_channel: Union[Channel, int],
               sda_channel: Union[Channel, int],
               clock_rate: U32 = 100000,
               address_bits: int = 7) -> I2CSimulation:
        """
        Add an I2C simulation.
        
        Args:
            scl_channel: SCL channel (index or Channel object)
            sda_channel: SDA channel (index or Channel object)
            clock_rate: Clock rate in Hz
            address_bits: Address bits (7 or 10)
            
        Returns:
            The I2C simulation object
            
        Example:
            >>> # Create an I2C simulation at 400kHz
            >>> i2c = manager.add_i2c(0, 1, 400000)
        """
        # Convert channel index to Channel object if needed
        if isinstance(scl_channel, int):
            scl_channel = Channel(scl_channel, f"I2C_SCL_{scl_channel}")
        
        if isinstance(sda_channel, int):
            sda_channel = Channel(sda_channel, f"I2C_SDA_{sda_channel}")
        
        # Get simulation channels
        scl_sim_channel = self.get_channel(scl_channel.index, scl_channel.name)
        sda_sim_channel = self.get_channel(sda_channel.index, sda_channel.name)
        
        # Create I2C config
        config = I2CConfig(clock_rate=clock_rate, address_bits=address_bits)
        
        # Create I2C simulation
        i2c_sim = I2CSimulation(scl_sim_channel, sda_sim_channel, config, self.sample_rate_hz)
        
        # Register simulation
        self.simulations.append(i2c_sim)
        
        return i2c_sim
    
    def add_spi(self,
               sck_channel: Union[Channel, int],
               mosi_channel: Optional[Union[Channel, int]] = None,
               miso_channel: Optional[Union[Channel, int]] = None,
               cs_channel: Optional[Union[Channel, int]] = None,
               clock_rate: U32 = 1000000,
               mode: int = 0,
               msb_first: bool = True,
               bits_per_word: int = 8,
               cs_active_low: bool = True) -> SPISimulation:
        """
        Add an SPI simulation.
        
        Args:
            sck_channel: SCK channel (index or Channel object)
            mosi_channel: Optional MOSI channel (index or Channel object)
            miso_channel: Optional MISO channel (index or Channel object)
            cs_channel: Optional CS channel (index or Channel object)
            clock_rate: Clock rate in Hz
            mode: SPI mode (0-3)
            msb_first: Whether MSB is transmitted first
            bits_per_word: Number of bits per word
            cs_active_low: Whether chip select is active low
            
        Returns:
            The SPI simulation object
            
        Example:
            >>> # Create an SPI simulation in mode 0 at 1MHz
            >>> spi = manager.add_spi(0, 1, 2, 3, 1000000, 0)
        """
        # Convert channel indices to Channel objects if needed
        if isinstance(sck_channel, int):
            sck_channel = Channel(sck_channel, f"SPI_SCK_{sck_channel}")
        
        # Get simulation channels
        sck_sim_channel = self.get_channel(sck_channel.index, sck_channel.name)
        
        mosi_sim_channel = None
        if mosi_channel is not None:
            if isinstance(mosi_channel, int):
                mosi_channel = Channel(mosi_channel, f"SPI_MOSI_{mosi_channel}")
            mosi_sim_channel = self.get_channel(mosi_channel.index, mosi_channel.name)
        
        miso_sim_channel = None
        if miso_channel is not None:
            if isinstance(miso_channel, int):
                miso_channel = Channel(miso_channel, f"SPI_MISO_{miso_channel}")
            miso_sim_channel = self.get_channel(miso_channel.index, miso_channel.name)
        
        cs_sim_channel = None
        if cs_channel is not None:
            if isinstance(cs_channel, int):
                cs_channel = Channel(cs_channel, f"SPI_CS_{cs_channel}")
            cs_sim_channel = self.get_channel(cs_channel.index, cs_channel.name)
        
        # Create SPI config
        config = SPIConfig(
            clock_rate=clock_rate,
            mode=mode,
            msb_first=msb_first,
            bits_per_word=bits_per_word,
            cs_active_low=cs_active_low
        )
        
        # Create SPI simulation
        spi_sim = SPISimulation(
            sck_sim_channel, 
            mosi_sim_channel, 
            miso_sim_channel, 
            cs_sim_channel, 
            config, 
            self.sample_rate_hz
        )
        
        # Register simulation
        self.simulations.append(spi_sim)
        
        return spi_sim
    
    def add_pwm(self,
               channel: Union[Channel, int],
               frequency_hz: U32,
               duty_cycle_percent: float = 50.0,
               count: U32 = 10,
               start_high: bool = True) -> SimulationChannelDescriptor:
        """
        Add a PWM simulation.
        
        Args:
            channel: PWM channel (index or Channel object)
            frequency_hz: Frequency in Hz
            duty_cycle_percent: Duty cycle (0-100%)
            count: Number of cycles to generate
            start_high: Whether to start with high pulse
            
        Returns:
            The simulation channel descriptor
            
        Example:
            >>> # Create a 1kHz PWM signal with 25% duty cycle
            >>> pwm_channel = manager.add_pwm(0, 1000, 25.0, 100)
        """
        # Convert channel index to Channel object if needed
        if isinstance(channel, int):
            channel = Channel(channel, f"PWM_{channel}")
        
        # Get simulation channel
        sim_channel = self.get_channel(channel.index, channel.name)
        
        # Create PWM signal
        sim_channel.add_square_wave(frequency_hz, duty_cycle_percent, count)
        
        return sim_channel
    
    def add_clock(self,
                 channel: Union[Channel, int],
                 frequency_hz: U32,
                 duty_cycle_percent: float = 50.0,
                 count: U32 = 100) -> SimulationChannelDescriptor:
        """
        Add a clock signal simulation.
        
        Args:
            channel: Clock channel (index or Channel object)
            frequency_hz: Frequency in Hz
            duty_cycle_percent: Duty cycle (0-100%)
            count: Number of cycles to generate
            
        Returns:
            The simulation channel descriptor
            
        Example:
            >>> # Create a 10MHz clock signal for 1000 cycles
            >>> clk_channel = manager.add_clock(0, 10_000_000, 50.0, 1000)
        """
        return self.add_pwm(channel, frequency_hz, duty_cycle_percent, count, True)
    
    def add_pattern(self, 
                   channel: Union[Channel, int],
                   pattern_type: str,
                   **kwargs) -> SimulationChannelDescriptor:
        """
        Add a standard test pattern simulation.
        
        Args:
            channel: Channel (index or Channel object)
            pattern_type: Type of pattern ('square', 'counter', 'walking_ones', etc.)
            **kwargs: Additional pattern-specific parameters
            
        Returns:
            The simulation channel descriptor
            
        Raises:
            ValueError: If pattern_type is unknown
            
        Example:
            >>> # Create a walking ones pattern
            >>> channel = manager.add_pattern(0, 'walking_ones', bit_width=100, bits=8)
        """
        # Convert channel index to Channel object if needed
        if isinstance(channel, int):
            channel = Channel(channel, f"Pattern_{channel}")
        
        # Get simulation channel
        sim_channel = self.get_channel(channel.index, channel.name)
        
        # Generate the requested pattern
        if pattern_type == 'square' or pattern_type == 'square_wave':
            frequency_hz = kwargs.get('frequency_hz', 1000)
            duty_cycle_percent = kwargs.get('duty_cycle_percent', 50.0)
            count = kwargs.get('count', 10)
            sim_channel.add_square_wave(frequency_hz, duty_cycle_percent, count)
            
        elif pattern_type == 'pulse' or pattern_type == 'pulse_train':
            pulse_width_s = kwargs.get('pulse_width_s', 0.001)
            gap_s = kwargs.get('gap_s', 0.001)
            count = kwargs.get('count', 10)
            sim_channel.add_pulse_train(pulse_width_s, gap_s, count)
            
        elif pattern_type == 'clock':
            frequency_hz = kwargs.get('frequency_hz', 1000)
            duty_cycle_percent = kwargs.get('duty_cycle_percent', 50.0)
            count = kwargs.get('count', 100)
            sim_channel.add_clock(frequency_hz, count, duty_cycle_percent)
            
        elif pattern_type == 'bit_pattern':
            pattern = kwargs.get('pattern', [BitState.HIGH, BitState.LOW])
            samples_per_bit = kwargs.get('samples_per_bit', 100)
            sim_channel.add_bit_pattern(pattern, samples_per_bit)
            
        elif pattern_type == 'byte_pattern':
            data = kwargs.get('data', b'\x55\xAA')  # Alternating 01010101 10101010
            samples_per_bit = kwargs.get('samples_per_bit', 100)
            msb_first = kwargs.get('msb_first', True)
            sim_channel.add_byte_pattern(data, samples_per_bit, msb_first)
            
        elif pattern_type == 'ramp' or pattern_type == 'stair':
            start_value = kwargs.get('start_value', 0)
            end_value = kwargs.get('end_value', 255)
            steps = kwargs.get('steps', 8)
            bit_width = kwargs.get('bit_width', 100)
            bits = kwargs.get('bits', 8)
            sim_channel.add_ramp(start_value, end_value, steps, bit_width, bits)
            
        elif pattern_type == 'counter':
            start_value = kwargs.get('start_value', 0)
            end_value = kwargs.get('end_value', 255)
            bit_width = kwargs.get('bit_width', 100)
            bits = kwargs.get('bits', 8)
            sim_channel.add_counter(start_value, end_value, bit_width, bits)
            
        elif pattern_type == 'walking_ones':
            bit_width = kwargs.get('bit_width', 100)
            bits = kwargs.get('bits', 8)
            sim_channel.add_walking_ones(bit_width, bits)
            
        elif pattern_type == 'walking_zeros':
            bit_width = kwargs.get('bit_width', 100)
            bits = kwargs.get('bits', 8)
            sim_channel.add_walking_zeros(bit_width, bits)
            
        elif pattern_type == 'manchester':
            data = kwargs.get('data', b'\x55\xAA')
            bit_rate = kwargs.get('bit_rate', 10000)
            standard_manchester = kwargs.get('standard_manchester', True)
            msb_first = kwargs.get('msb_first', True)
            sim_channel.add_manchester_data(data, bit_rate, standard_manchester, msb_first)
            
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        return sim_channel
    
    def run(self) -> None:
        """
        Run all simulations.
        
        Generates all the simulation data for the configured channels.
        
        Example:
            >>> # Configure simulations
            >>> manager.add_uart(0, "Hello", 9600)
            >>> manager.add_i2c(1, 2, 100000)
            >>> # Run all simulations
            >>> manager.run()
        """
        for simulation in self.simulations:
            simulation.generate()
    
    def get_channel_count(self) -> int:
        """
        Get the number of channels in the simulation.
        
        Returns:
            Number of channels
        """
        return len(self._channel_descriptors)
    
    def get_all_channels(self) -> List[SimulationChannelDescriptor]:
        """
        Get all simulation channels.
        
        Returns:
            List of all simulation channel descriptors
        """
        return list(self._channel_descriptors.values())
    
    def clear(self) -> None:
        """
        Clear all simulations and channels.
        
        This resets the simulation manager to its initial state.
        """
        self.simulations = []
        self._channel_descriptors = {}
        self.group = _ka.SimulationChannelDescriptorGroup()
    
    def __str__(self) -> str:
        """Create a string representation of the simulation manager."""
        return (f"SimulationManager(sample_rate={self.sample_rate_hz} Hz, "
                f"channels={self.get_channel_count()}, "
                f"simulations={len(self.simulations)})")


# Predefined Protocol Standards

class UARTStandards:
    """
    Common UART configurations.
    
    This class provides predefined UART configurations for
    standard baud rates and formats.
    
    Example:
        >>> # Use a standard 115200 baud, 8N1 configuration
        >>> uart = UARTSimulation(tx_channel, config=UARTStandards.STANDARD_115200_8N1)
    """
    
    STANDARD_9600_8N1 = UARTConfig(baud_rate=9600, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_115200_8N1 = UARTConfig(baud_rate=115200, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_9600_8E1 = UARTConfig(baud_rate=9600, data_bits=8, stop_bits=1.0, parity='even')
    STANDARD_9600_8O1 = UARTConfig(baud_rate=9600, data_bits=8, stop_bits=1.0, parity='odd')
    STANDARD_9600_8N2 = UARTConfig(baud_rate=9600, data_bits=8, stop_bits=2.0, parity=None)
    STANDARD_19200_8N1 = UARTConfig(baud_rate=19200, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_38400_8N1 = UARTConfig(baud_rate=38400, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_57600_8N1 = UARTConfig(baud_rate=57600, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_230400_8N1 = UARTConfig(baud_rate=230400, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_460800_8N1 = UARTConfig(baud_rate=460800, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_921600_8N1 = UARTConfig(baud_rate=921600, data_bits=8, stop_bits=1.0, parity=None)


class I2CStandards:
    """
    Common I2C configurations.
    
    This class provides predefined I2C configurations for
    standard speeds and modes.
    
    Example:
        >>> # Use a standard Fast Mode configuration
        >>> i2c = I2CSimulation(scl_channel, sda_channel, config=I2CStandards.FAST_MODE)
    """
    
    STANDARD_MODE = I2CConfig(clock_rate=100000)  # 100 kHz
    FAST_MODE = I2CConfig(clock_rate=400000)      # 400 kHz
    FAST_MODE_PLUS = I2CConfig(clock_rate=1000000)  # 1 MHz
    HIGH_SPEED_MODE = I2CConfig(clock_rate=3400000)  # 3.4 MHz
    ULTRA_FAST_MODE = I2CConfig(clock_rate=5000000)  # 5 MHz


class SPIStandards:
    """
    Common SPI configurations.
    
    This class provides predefined SPI configurations for
    standard modes and speeds.
    
    Example:
        >>> # Use a standard Mode 0, 1MHz configuration
        >>> spi = SPISimulation(sck, mosi, miso, cs, config=SPIStandards.MODE_0_1MHZ)
    """
    
    # Mode 0: CPOL=0, CPHA=0
    MODE_0_1MHZ = SPIConfig(clock_rate=1000000, mode=0)
    MODE_0_5MHZ = SPIConfig(clock_rate=5000000, mode=0)
    MODE_0_10MHZ = SPIConfig(clock_rate=10000000, mode=0)
    
    # Mode 1: CPOL=0, CPHA=1
    MODE_1_1MHZ = SPIConfig(clock_rate=1000000, mode=1)
    MODE_1_5MHZ = SPIConfig(clock_rate=5000000, mode=1)
    MODE_1_10MHZ = SPIConfig(clock_rate=10000000, mode=1)
    
    # Mode 2: CPOL=1, CPHA=0
    MODE_2_1MHZ = SPIConfig(clock_rate=1000000, mode=2)
    MODE_2_5MHZ = SPIConfig(clock_rate=5000000, mode=2)
    MODE_2_10MHZ = SPIConfig(clock_rate=10000000, mode=2)
    
    # Mode 3: CPOL=1, CPHA=1
    MODE_3_1MHZ = SPIConfig(clock_rate=1000000, mode=3)
    MODE_3_5MHZ = SPIConfig(clock_rate=5000000, mode=3)
    MODE_3_10MHZ = SPIConfig(clock_rate=10000000, mode=3)


# Utility function for creating test patterns

def create_test_pattern(manager: SimulationManager, 
                       pattern_type: str, 
                       channel: Union[Channel, int],
                       **kwargs) -> SimulationChannelDescriptor:
    """
    Create a standardized test pattern.
    
    This is a convenience function that delegates to
    SimulationManager.add_pattern().
    
    Args:
        manager: Simulation manager
        pattern_type: Type of pattern ('square', 'counter', 'walking_ones', etc.)
        channel: Channel to output pattern on
        **kwargs: Additional pattern-specific parameters
        
    Returns:
        Simulation channel descriptor
        
    Example:
        >>> # Create a counter pattern
        >>> channel = create_test_pattern(manager, 'counter', 0, 
        ...                              start_value=0, end_value=15, 
        ...                              bit_width=100, bits=4)
    """
    return manager.add_pattern(channel, pattern_type, **kwargs)