"""
Simulation module for Kingst Logic Analyzer.

This module provides Pythonic interfaces for creating and configuring simulations
with the Kingst Logic Analyzer. It includes builder patterns for complex simulation
setups and helper functions for common test signals and protocol patterns.

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
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Callable, Any, Sequence

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


# Type aliases
U8 = int
U16 = int
U32 = int
U64 = int


class SignalGenerator:
    """Utility class for generating common digital signal patterns."""
    
    @staticmethod
    def square_wave(channel: _ka.SimulationChannelDescriptor, 
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
        """
        sample_rate = channel.get_sample_rate()
        period_samples = sample_rate // frequency_hz
        high_samples = int(period_samples * duty_cycle_percent / 100.0)
        low_samples = period_samples - high_samples
        
        for _ in range(count):
            channel.transition_if_needed(BitState.HIGH)
            channel.advance(high_samples)
            channel.transition_if_needed(BitState.LOW)
            channel.advance(low_samples)
    
    @staticmethod
    def pulse_train(channel: _ka.SimulationChannelDescriptor,
                   pulse_width_samples: U32,
                   gap_samples: U32,
                   count: U32) -> None:
        """
        Generate a train of pulses.
        
        Args:
            channel: The channel to generate the signal on
            pulse_width_samples: Width of each pulse in samples
            gap_samples: Gap between pulses in samples
            count: Number of pulses to generate
        """
        for _ in range(count):
            # Toggle to opposite state
            channel.transition()
            # Pulse width
            channel.advance(pulse_width_samples)
            # Toggle back
            channel.transition()
            # Gap between pulses
            if _ < count - 1:  # No gap after the last pulse
                channel.advance(gap_samples)
    
    @staticmethod
    def clock(channel: _ka.SimulationChannelDescriptor,
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
        """
        SignalGenerator.square_wave(channel, frequency_hz, duty_cycle_percent, count)
    
    @staticmethod
    def bit_pattern(channel: _ka.SimulationChannelDescriptor,
                   pattern: List[BitState],
                   samples_per_bit: U32) -> None:
        """
        Generate a specific bit pattern.
        
        Args:
            channel: The channel to generate the signal on
            pattern: List of bit states to generate
            samples_per_bit: Number of samples per bit
        """
        channel.add_bit_pattern(pattern, samples_per_bit)
    
    @staticmethod
    def byte_pattern(channel: _ka.SimulationChannelDescriptor,
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
        """
        for byte in data:
            for bit_index in range(8):
                # Get the correct bit based on MSB/LSB order
                bit_pos = 7 - bit_index if msb_first else bit_index
                bit_value = (byte >> bit_pos) & 0x01
                bit_state = BitState.HIGH if bit_value else BitState.LOW
                
                # Set the bit
                channel.transition_if_needed(bit_state)
                channel.advance(samples_per_bit)


@dataclass
class UARTConfig:
    """Configuration for UART simulation."""
    
    baud_rate: U32
    data_bits: int = 8
    stop_bits: float = 1.0
    parity: Optional[str] = None  # None, 'even', 'odd'
    
    def __post_init__(self):
        """Validate the configuration."""
        if self.data_bits not in (5, 6, 7, 8, 9):
            raise ValueError(f"Invalid data bits: {self.data_bits}. Must be 5-9.")
        
        if self.stop_bits not in (1.0, 1.5, 2.0):
            raise ValueError(f"Invalid stop bits: {self.stop_bits}. Must be 1, 1.5, or 2.")
        
        if self.parity not in (None, 'even', 'odd'):
            raise ValueError(f"Invalid parity: {self.parity}. Must be None, 'even', or 'odd'.")


@dataclass
class I2CConfig:
    """Configuration for I2C simulation."""
    
    clock_rate: U32 = 100000  # Standard mode (100 kHz)
    address_bits: int = 7     # 7 or 10 bit addressing


@dataclass
class SPIConfig:
    """Configuration for SPI simulation."""
    
    clock_rate: U32
    mode: int = 0  # SPI modes 0-3
    msb_first: bool = True
    bits_per_word: int = 8
    cs_active_low: bool = True
    
    def __post_init__(self):
        """Validate the configuration and derive CPOL/CPHA."""
        if self.mode not in (0, 1, 2, 3):
            raise ValueError(f"Invalid SPI mode: {self.mode}. Must be 0-3.")
        
        # SPI mode to CPOL/CPHA mapping:
        # Mode 0: CPOL=0, CPHA=0
        # Mode 1: CPOL=0, CPHA=1
        # Mode 2: CPOL=1, CPHA=0
        # Mode 3: CPOL=1, CPHA=1
        self.cpol = (self.mode & 2) > 0
        self.cpha = (self.mode & 1) > 0


class UARTSimulation:
    """Builder for UART simulations."""
    
    def __init__(self, 
                tx_channel: _ka.SimulationChannelDescriptor,
                rx_channel: Optional[_ka.SimulationChannelDescriptor] = None,
                config: Optional[UARTConfig] = None,
                sample_rate: U32 = 0):
        """
        Initialize a UART simulation.
        
        Args:
            tx_channel: The TX channel (from master/DUT)
            rx_channel: Optional RX channel (to master/DUT)
            config: UART configuration
            sample_rate: Sample rate in Hz (if 0, use channel's rate)
        """
        self.tx_channel = tx_channel
        self.rx_channel = rx_channel
        self.config = config or UARTConfig(baud_rate=9600)
        self.sample_rate = sample_rate or tx_channel.get_sample_rate()
        self.tx_data = bytearray()
        self.rx_data = bytearray()
    
    def add_tx_data(self, data: Union[bytes, bytearray, str]) -> 'UARTSimulation':
        """
        Add data to transmit on the TX channel.
        
        Args:
            data: Data to transmit (will be converted to bytes)
            
        Returns:
            Self for method chaining
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
        """
        if not self.rx_channel:
            raise ValueError("RX channel not configured")
            
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        self.rx_data.extend(data)
        return self
    
    def generate(self) -> None:
        """Generate the UART simulation."""
        # Calculate timing parameters
        bit_width = self.sample_rate // self.config.baud_rate
        
        # Generate TX data
        if self.tx_data:
            for byte in self.tx_data:
                # Start bit (always LOW)
                self.tx_channel.transition_if_needed(BitState.LOW)
                self.tx_channel.advance(bit_width)
                
                # Data bits (LSB first)
                for i in range(self.config.data_bits):
                    bit_value = (byte >> i) & 0x01
                    bit_state = BitState.HIGH if bit_value else BitState.LOW
                    self.tx_channel.transition_if_needed(bit_state)
                    self.tx_channel.advance(bit_width)
                
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
                        
                    self.tx_channel.transition_if_needed(parity_bit)
                    self.tx_channel.advance(bit_width)
                
                # Stop bit(s)
                self.tx_channel.transition_if_needed(BitState.HIGH)
                
                # Calculate stop bit timing
                if self.config.stop_bits == 1.0:
                    self.tx_channel.advance(bit_width)
                elif self.config.stop_bits == 1.5:
                    self.tx_channel.advance(bit_width + bit_width // 2)
                else:  # 2.0
                    self.tx_channel.advance(bit_width * 2)
                
                # Inter-byte gap (1 bit width)
                self.tx_channel.advance(bit_width)
        
        # Generate RX data
        if self.rx_channel and self.rx_data:
            for byte in self.rx_data:
                # Start bit (always LOW)
                self.rx_channel.transition_if_needed(BitState.LOW)
                self.rx_channel.advance(bit_width)
                
                # Data bits (LSB first)
                for i in range(self.config.data_bits):
                    bit_value = (byte >> i) & 0x01
                    bit_state = BitState.HIGH if bit_value else BitState.LOW
                    self.rx_channel.transition_if_needed(bit_state)
                    self.rx_channel.advance(bit_width)
                
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
                        
                    self.rx_channel.transition_if_needed(parity_bit)
                    self.rx_channel.advance(bit_width)
                
                # Stop bit(s)
                self.rx_channel.transition_if_needed(BitState.HIGH)
                
                # Calculate stop bit timing
                if self.config.stop_bits == 1.0:
                    self.rx_channel.advance(bit_width)
                elif self.config.stop_bits == 1.5:
                    self.rx_channel.advance(bit_width + bit_width // 2)
                else:  # 2.0
                    self.rx_channel.advance(bit_width * 2)
                
                # Inter-byte gap (1 bit width)
                self.rx_channel.advance(bit_width)


class I2CSimulation:
    """Builder for I2C simulations."""
    
    def __init__(self,
                scl_channel: _ka.SimulationChannelDescriptor,
                sda_channel: _ka.SimulationChannelDescriptor,
                config: Optional[I2CConfig] = None,
                sample_rate: U32 = 0):
        """
        Initialize an I2C simulation.
        
        Args:
            scl_channel: The SCL (clock) channel
            sda_channel: The SDA (data) channel
            config: I2C configuration
            sample_rate: Sample rate in Hz (if 0, use channel's rate)
        """
        self.scl_channel = scl_channel
        self.sda_channel = sda_channel
        self.config = config or I2CConfig()
        self.sample_rate = sample_rate or scl_channel.get_sample_rate()
        self.transactions = []
    
    def add_transaction(self, 
                       address: int, 
                       data: Union[bytes, bytearray], 
                       is_read: bool = False) -> 'I2CSimulation':
        """
        Add an I2C transaction.
        
        Args:
            address: 7-bit or 10-bit device address
            data: Data bytes to transmit or receive
            is_read: True for read operation, False for write
            
        Returns:
            Self for method chaining
        """
        # Validate address
        if self.config.address_bits == 7 and address > 0x7F:
            raise ValueError(f"Invalid 7-bit address: 0x{address:X}. Must be 0-0x7F.")
        elif self.config.address_bits == 10 and address > 0x3FF:
            raise ValueError(f"Invalid 10-bit address: 0x{address:X}. Must be 0-0x3FF.")
        
        # Store the transaction
        self.transactions.append({
            'address': address,
            'data': data if isinstance(data, (bytes, bytearray)) else bytes(data),
            'is_read': is_read
        })
        
        return self
    
    def add_write(self, address: int, data: Union[bytes, bytearray]) -> 'I2CSimulation':
        """
        Add an I2C write transaction.
        
        Args:
            address: 7-bit or 10-bit device address
            data: Data bytes to write
            
        Returns:
            Self for method chaining
        """
        return self.add_transaction(address, data, is_read=False)
    
    def add_read(self, address: int, data: Union[bytes, bytearray]) -> 'I2CSimulation':
        """
        Add an I2C read transaction.
        
        Args:
            address: 7-bit or 10-bit device address
            data: Data bytes to read (from slave)
            
        Returns:
            Self for method chaining
        """
        return self.add_transaction(address, data, is_read=True)
    
    def generate(self) -> None:
        """Generate the I2C simulation."""
        bit_width = self.sample_rate // self.config.clock_rate
        
        # Initial idle state (both lines HIGH)
        self.scl_channel.transition_if_needed(BitState.HIGH)
        self.sda_channel.transition_if_needed(BitState.HIGH)
        
        # Generate each transaction
        for transaction in self.transactions:
            address = transaction['address']
            data = transaction['data']
            is_read = transaction['is_read']
            
            # START condition
            self.sda_channel.add_i2c_start(bit_width, is_scl=False)
            self.scl_channel.add_i2c_start(bit_width, is_scl=True)
            
            # Address byte (7-bit address + R/W bit)
            address_byte = (address << 1) | (1 if is_read else 0)
            
            # Send address byte (MSB first)
            for i in range(7, -1, -1):
                bit_value = (address_byte >> i) & 0x01
                
                # SCL goes low
                self.scl_channel.transition_if_needed(BitState.LOW)
                self.scl_channel.advance(bit_width // 4)
                
                # Set SDA while SCL is low
                self.sda_channel.transition_if_needed(BitState.HIGH if bit_value else BitState.LOW)
                self.sda_channel.advance(bit_width // 4)
                
                # SCL goes high
                self.scl_channel.transition_if_needed(BitState.HIGH)
                self.scl_channel.advance(bit_width // 2)
                
                # SCL goes low again
                self.scl_channel.transition_if_needed(BitState.LOW)
                self.scl_channel.advance(bit_width // 4)
                
                # Keep SDA at same level
                self.sda_channel.advance(bit_width // 4)
            
            # ACK from slave (SDA pulled LOW)
            self.scl_channel.transition_if_needed(BitState.LOW)
            self.scl_channel.advance(bit_width // 4)
            
            self.sda_channel.transition_if_needed(BitState.LOW)  # ACK
            self.sda_channel.advance(bit_width // 4)
            
            self.scl_channel.transition_if_needed(BitState.HIGH)
            self.scl_channel.advance(bit_width // 2)
            
            self.scl_channel.transition_if_needed(BitState.LOW)
            self.scl_channel.advance(bit_width // 4)
            
            self.sda_channel.advance(bit_width // 4)
            
            # Data bytes
            for byte_idx, byte in enumerate(data):
                for i in range(7, -1, -1):
                    bit_value = (byte >> i) & 0x01
                    
                    # SCL goes low
                    self.scl_channel.transition_if_needed(BitState.LOW)
                    self.scl_channel.advance(bit_width // 4)
                    
                    # Set SDA while SCL is low
                    self.sda_channel.transition_if_needed(BitState.HIGH if bit_value else BitState.LOW)
                    self.sda_channel.advance(bit_width // 4)
                    
                    # SCL goes high
                    self.scl_channel.transition_if_needed(BitState.HIGH)
                    self.scl_channel.advance(bit_width // 2)
                    
                    # SCL goes low again
                    self.scl_channel.transition_if_needed(BitState.LOW)
                    self.scl_channel.advance(bit_width // 4)
                    
                    # Keep SDA at same level
                    self.sda_channel.advance(bit_width // 4)
                
                # ACK/NACK bit
                # For write: slave sends ACK (SDA LOW)
                # For last read byte: master sends NACK (SDA HIGH)
                # For non-last read byte: master sends ACK (SDA LOW)
                is_last_byte = byte_idx == len(data) - 1
                send_nack = is_read and is_last_byte
                
                self.scl_channel.transition_if_needed(BitState.LOW)
                self.scl_channel.advance(bit_width // 4)
                
                self.sda_channel.transition_if_needed(BitState.HIGH if send_nack else BitState.LOW)
                self.sda_channel.advance(bit_width // 4)
                
                self.scl_channel.transition_if_needed(BitState.HIGH)
                self.scl_channel.advance(bit_width // 2)
                
                self.scl_channel.transition_if_needed(BitState.LOW)
                self.scl_channel.advance(bit_width // 4)
                
                self.sda_channel.advance(bit_width // 4)
            
            # STOP condition
            self.scl_channel.transition_if_needed(BitState.LOW)
            self.scl_channel.advance(bit_width // 4)
            
            self.sda_channel.transition_if_needed(BitState.LOW)
            self.sda_channel.advance(bit_width // 4)
            
            self.scl_channel.transition_if_needed(BitState.HIGH)
            self.scl_channel.advance(bit_width // 2)
            
            # SDA goes from LOW to HIGH while SCL is HIGH (STOP)
            self.sda_channel.transition_if_needed(BitState.HIGH)
            self.sda_channel.advance(bit_width // 2)
            
            # Idle time between transactions
            self.sda_channel.advance(bit_width * 2)
            self.scl_channel.advance(bit_width * 2)


class SPISimulation:
    """Builder for SPI simulations."""
    
    def __init__(self,
                sck_channel: _ka.SimulationChannelDescriptor,
                mosi_channel: Optional[_ka.SimulationChannelDescriptor] = None,
                miso_channel: Optional[_ka.SimulationChannelDescriptor] = None,
                cs_channel: Optional[_ka.SimulationChannelDescriptor] = None,
                config: Optional[SPIConfig] = None,
                sample_rate: U32 = 0):
        """
        Initialize an SPI simulation.
        
        Args:
            sck_channel: The SCK (clock) channel
            mosi_channel: Optional MOSI (Master Out Slave In) channel
            miso_channel: Optional MISO (Master In Slave Out) channel
            cs_channel: Optional CS (Chip Select) channel
            config: SPI configuration
            sample_rate: Sample rate in Hz (if 0, use channel's rate)
        """
        self.sck_channel = sck_channel
        self.mosi_channel = mosi_channel
        self.miso_channel = miso_channel
        self.cs_channel = cs_channel
        self.config = config or SPIConfig(clock_rate=1000000)
        self.sample_rate = sample_rate or sck_channel.get_sample_rate()
        self.transactions = []
    
    def add_transaction(self, 
                       mosi_data: Optional[Union[bytes, bytearray]] = None,
                       miso_data: Optional[Union[bytes, bytearray]] = None) -> 'SPISimulation':
        """
        Add an SPI transaction.
        
        Args:
            mosi_data: Data to send from master to slave (optional)
            miso_data: Data to send from slave to master (optional)
            
        Returns:
            Self for method chaining
        """
        if mosi_data is None and miso_data is None:
            raise ValueError("At least one of mosi_data or miso_data must be provided")
        
        self.transactions.append({
            'mosi_data': mosi_data if mosi_data is not None else bytes(),
            'miso_data': miso_data if miso_data is not None else bytes()
        })
        
        return self
    
    def generate(self) -> None:
        """Generate the SPI simulation."""
        bit_width = self.sample_rate // self.config.clock_rate
        
        # Initial idle state
        idle_state = BitState.HIGH if self.config.cpol else BitState.LOW
        self.sck_channel.transition_if_needed(idle_state)
        
        if self.cs_channel:
            self.cs_channel.transition_if_needed(
                BitState.HIGH if self.config.cs_active_low else BitState.LOW
            )
        
        # Process each transaction
        for transaction in self.transactions:
            mosi_data = transaction['mosi_data']
            miso_data = transaction['miso_data']
            
            # Activate chip select
            if self.cs_channel:
                self.cs_channel.transition_if_needed(
                    BitState.LOW if self.config.cs_active_low else BitState.HIGH
                )
                self.cs_channel.advance(bit_width // 2)
            
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
                        
                        # Advance all active channels
                        self.sck_channel.advance(bit_width // 4)
                        if self.mosi_channel:
                            self.mosi_channel.advance(bit_width // 4)
                        if self.miso_channel:
                            self.miso_channel.advance(bit_width // 4)
                        if self.cs_channel:
                            self.cs_channel.advance(bit_width // 4)
                        
                        # First clock edge (leading)
                        self.sck_channel.transition()
                        self.sck_channel.advance(bit_width // 2)
                        if self.mosi_channel:
                            self.mosi_channel.advance(bit_width // 2)
                        if self.miso_channel:
                            self.miso_channel.advance(bit_width // 2)
                        if self.cs_channel:
                            self.cs_channel.advance(bit_width // 2)
                        
                        # Second clock edge (trailing)
                        self.sck_channel.transition()
                        self.sck_channel.advance(bit_width // 4)
                        if self.mosi_channel:
                            self.mosi_channel.advance(bit_width // 4)
                        if self.miso_channel:
                            self.miso_channel.advance(bit_width // 4)
                        if self.cs_channel:
                            self.cs_channel.advance(bit_width // 4)
                    else:
                        # CPHA=1: Data is valid on trailing edge
                        
                        # First clock edge
                        self.sck_channel.transition()
                        self.sck_channel.advance(bit_width // 4)
                        if self.mosi_channel:
                            self.mosi_channel.advance(bit_width // 4)
                        if self.miso_channel:
                            self.miso_channel.advance(bit_width // 4)
                        if self.cs_channel:
                            self.cs_channel.advance(bit_width // 4)
                        
                        # Set data after first clock edge
                        if self.mosi_channel:
                            self.mosi_channel.transition_if_needed(
                                BitState.HIGH if mosi_bit else BitState.LOW
                            )
                        
                        if self.miso_channel:
                            self.miso_channel.transition_if_needed(
                                BitState.HIGH if miso_bit else BitState.LOW
                            )
                        
                        # Advance all active channels
                        self.sck_channel.advance(bit_width // 2)
                        if self.mosi_channel:
                            self.mosi_channel.advance(bit_width // 2)
                        if self.miso_channel:
                            self.miso_channel.advance(bit_width // 2)
                        if self.cs_channel:
                            self.cs_channel.advance(bit_width // 2)
                        
                        # Second clock edge
                        self.sck_channel.transition()
                        self.sck_channel.advance(bit_width // 4)
                        if self.mosi_channel:
                            self.mosi_channel.advance(bit_width // 4)
                        if self.miso_channel:
                            self.miso_channel.advance(bit_width // 4)
                        if self.cs_channel:
                            self.cs_channel.advance(bit_width // 4)
            
            # Deactivate chip select
            if self.cs_channel:
                self.cs_channel.transition_if_needed(
                    BitState.HIGH if self.config.cs_active_low else BitState.LOW
                )
                
                # Idle time after transaction
                self.sck_channel.advance(bit_width * 2)
                if self.mosi_channel:
                    self.mosi_channel.advance(bit_width * 2)
                if self.miso_channel:
                    self.miso_channel.advance(bit_width * 2)
                self.cs_channel.advance(bit_width * 2)


class SimulationManager:
    """High-level manager for simulations."""
    
    def __init__(self, sample_rate: U32 = 10_000_000):
        """
        Initialize a simulation manager.
        
        Args:
            sample_rate: Default sample rate in Hz for all channels
        """
        self.group = _ka.SimulationChannelDescriptorGroup()
        self.sample_rate = sample_rate
        self.channels = {}
        self.simulations = []
    
    def get_channel(self, channel_index: int, channel_name: Optional[str] = None) -> _ka.SimulationChannelDescriptor:
        """
        Get or create a simulation channel.
        
        Args:
            channel_index: Channel index
            channel_name: Optional channel name
            
        Returns:
            The simulation channel descriptor
        """
        if channel_index in self.channels:
            return self.channels[channel_index]
        
        # Create a new channel
        ka_channel = Channel(channel_index, channel_name)
        sim_channel = self.group.add(ka_channel, self.sample_rate, BitState.LOW)
        self.channels[channel_index] = sim_channel
        
        return sim_channel
    
    def add_uart(self, 
                tx_channel: Union[Channel, int],
                data: Union[bytes, bytearray, str],
                baud_rate: U32 = 9600,
                rx_channel: Optional[Union[Channel, int]] = None,
                rx_data: Optional[Union[bytes, bytearray, str]] = None) -> UARTSimulation:
        """
        Add a UART simulation.
        
        Args:
            tx_channel: TX channel (index or Channel object)
            data: TX data to send
            baud_rate: Baud rate in bits per second
            rx_channel: Optional RX channel (index or Channel object)
            rx_data: Optional RX data to receive
            
        Returns:
            The UART simulation object
        """
        # Convert channel index to Channel object if needed
        if isinstance(tx_channel, int):
            tx_channel = Channel(tx_channel, f"UART_TX_{tx_channel}")
        
        # Get simulation channels
        tx_sim_channel = self.group.add(tx_channel, self.sample_rate, BitState.HIGH)
        
        rx_sim_channel = None
        if rx_channel is not None:
            if isinstance(rx_channel, int):
                rx_channel = Channel(rx_channel, f"UART_RX_{rx_channel}")
            rx_sim_channel = self.group.add(rx_channel, self.sample_rate, BitState.HIGH)
        
        # Create UART config
        config = UARTConfig(baud_rate=baud_rate)
        
        # Create UART simulation
        uart_sim = UARTSimulation(tx_sim_channel, rx_sim_channel, config, self.sample_rate)
        uart_sim.add_tx_data(data)
        
        if rx_sim_channel and rx_data:
            uart_sim.add_rx_data(rx_data)
        
        # Register simulation
        self.simulations.append(uart_sim)
        
        return uart_sim
    
    def add_i2c(self,
               scl_channel: Union[Channel, int],
               sda_channel: Union[Channel, int],
               clock_rate: U32 = 100000) -> I2CSimulation:
        """
        Add an I2C simulation.
        
        Args:
            scl_channel: SCL channel (index or Channel object)
            sda_channel: SDA channel (index or Channel object)
            clock_rate: Clock rate in Hz
            
        Returns:
            The I2C simulation object
        """
        # Convert channel index to Channel object if needed
        if isinstance(scl_channel, int):
            scl_channel = Channel(scl_channel, f"I2C_SCL_{scl_channel}")
        
        if isinstance(sda_channel, int):
            sda_channel = Channel(sda_channel, f"I2C_SDA_{sda_channel}")
        
        # Get simulation channels
        scl_sim_channel = self.group.add(scl_channel, self.sample_rate, BitState.HIGH)
        sda_sim_channel = self.group.add(sda_channel, self.sample_rate, BitState.HIGH)
        
        # Create I2C config
        config = I2CConfig(clock_rate=clock_rate)
        
        # Create I2C simulation
        i2c_sim = I2CSimulation(scl_sim_channel, sda_sim_channel, config, self.sample_rate)
        
        # Register simulation
        self.simulations.append(i2c_sim)
        
        return i2c_sim
    
    def add_spi(self,
               sck_channel: Union[Channel, int],
               mosi_channel: Optional[Union[Channel, int]] = None,
               miso_channel: Optional[Union[Channel, int]] = None,
               cs_channel: Optional[Union[Channel, int]] = None,
               clock_rate: U32 = 1000000,
               mode: int = 0) -> SPISimulation:
        """
        Add an SPI simulation.
        
        Args:
            sck_channel: SCK channel (index or Channel object)
            mosi_channel: Optional MOSI channel (index or Channel object)
            miso_channel: Optional MISO channel (index or Channel object)
            cs_channel: Optional CS channel (index or Channel object)
            clock_rate: Clock rate in Hz
            mode: SPI mode (0-3)
            
        Returns:
            The SPI simulation object
        """
        # Convert channel indices to Channel objects if needed
        if isinstance(sck_channel, int):
            sck_channel = Channel(sck_channel, f"SPI_SCK_{sck_channel}")
        
        # Get simulation channels
        sck_sim_channel = self.group.add(sck_channel, self.sample_rate, 
                                        BitState.HIGH if mode & 2 else BitState.LOW)
        
        mosi_sim_channel = None
        if mosi_channel is not None:
            if isinstance(mosi_channel, int):
                mosi_channel = Channel(mosi_channel, f"SPI_MOSI_{mosi_channel}")
            mosi_sim_channel = self.group.add(mosi_channel, self.sample_rate, BitState.LOW)
        
        miso_sim_channel = None
        if miso_channel is not None:
            if isinstance(miso_channel, int):
                miso_channel = Channel(miso_channel, f"SPI_MISO_{miso_channel}")
            miso_sim_channel = self.group.add(miso_channel, self.sample_rate, BitState.LOW)
        
        cs_sim_channel = None
        if cs_channel is not None:
            if isinstance(cs_channel, int):
                cs_channel = Channel(cs_channel, f"SPI_CS_{cs_channel}")
            cs_sim_channel = self.group.add(cs_channel, self.sample_rate, BitState.HIGH)
        
        # Create SPI config
        config = SPIConfig(clock_rate=clock_rate, mode=mode)
        
        # Create SPI simulation
        spi_sim = SPISimulation(
            sck_sim_channel, 
            mosi_sim_channel, 
            miso_sim_channel, 
            cs_sim_channel, 
            config, 
            self.sample_rate
        )
        
        # Register simulation
        self.simulations.append(spi_sim)
        
        return spi_sim
    
    def add_pwm(self,
               channel: Union[Channel, int],
               frequency_hz: U32,
               duty_cycle_percent: float = 50.0,
               count: U32 = 10,
               start_high: bool = True) -> _ka.SimulationChannelDescriptor:
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
        """
        # Convert channel index to Channel object if needed
        if isinstance(channel, int):
            channel = Channel(channel, f"PWM_{channel}")
        
        # Get simulation channel
        sim_channel = self.group.add(channel, self.sample_rate, 
                                    BitState.HIGH if start_high else BitState.LOW)
        
        # Create PWM signal
        period_samples = self.sample_rate // frequency_hz
        high_samples = int(period_samples * duty_cycle_percent / 100.0)
        low_samples = period_samples - high_samples
        
        for _ in range(count):
            if start_high:
                sim_channel.transition_if_needed(BitState.HIGH)
                sim_channel.advance(high_samples)
                sim_channel.transition_if_needed(BitState.LOW)
                sim_channel.advance(low_samples)
            else:
                sim_channel.transition_if_needed(BitState.LOW)
                sim_channel.advance(low_samples)
                sim_channel.transition_if_needed(BitState.HIGH)
                sim_channel.advance(high_samples)
        
        return sim_channel
    
    def add_clock(self,
                 channel: Union[Channel, int],
                 frequency_hz: U32,
                 duty_cycle_percent: float = 50.0,
                 count: U32 = 100) -> _ka.SimulationChannelDescriptor:
        """
        Add a clock signal simulation.
        
        Args:
            channel: Clock channel (index or Channel object)
            frequency_hz: Frequency in Hz
            duty_cycle_percent: Duty cycle (0-100%)
            count: Number of cycles to generate
            
        Returns:
            The simulation channel descriptor
        """
        return self.add_pwm(channel, frequency_hz, duty_cycle_percent, count, True)
    
    def run(self) -> None:
        """
        Run all simulations.
        
        This generates all the simulation data for the configured channels.
        """
        for simulation in self.simulations:
            simulation.generate()


# Predefined Protocol Standards

class UARTStandards:
    """Common UART configurations."""
    
    STANDARD_9600_8N1 = UARTConfig(baud_rate=9600, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_115200_8N1 = UARTConfig(baud_rate=115200, data_bits=8, stop_bits=1.0, parity=None)
    STANDARD_9600_8E1 = UARTConfig(baud_rate=9600, data_bits=8, stop_bits=1.0, parity='even')
    STANDARD_9600_8O1 = UARTConfig(baud_rate=9600, data_bits=8, stop_bits=1.0, parity='odd')
    STANDARD_9600_8N2 = UARTConfig(baud_rate=9600, data_bits=8, stop_bits=2.0, parity=None)


class I2CStandards:
    """Common I2C configurations."""
    
    STANDARD_MODE = I2CConfig(clock_rate=100000)  # 100 kHz
    FAST_MODE = I2CConfig(clock_rate=400000)      # 400 kHz
    FAST_MODE_PLUS = I2CConfig(clock_rate=1000000)  # 1 MHz
    HIGH_SPEED_MODE = I2CConfig(clock_rate=3400000)  # 3.4 MHz


class SPIStandards:
    """Common SPI configurations."""
    
    # Mode 0: CPOL=0, CPHA=0
    MODE_0_1MHZ = SPIConfig(clock_rate=1000000, mode=0)
    # Mode 1: CPOL=0, CPHA=1
    MODE_1_1MHZ = SPIConfig(clock_rate=1000000, mode=1)
    # Mode 2: CPOL=1, CPHA=0
    MODE_2_1MHZ = SPIConfig(clock_rate=1000000, mode=2)
    # Mode 3: CPOL=1, CPHA=1
    MODE_3_1MHZ = SPIConfig(clock_rate=1000000, mode=3)


# Function to create standardized test patterns

def create_test_pattern(manager: SimulationManager, 
                       pattern_type: str, 
                       channel: Union[Channel, int],
                       **kwargs) -> Any:
    """
    Create a standardized test pattern.
    
    Args:
        manager: Simulation manager
        pattern_type: Type of pattern ('stair', 'counter', 'walking1', 'walking0', etc.)
        channel: Channel to output pattern on
        **kwargs: Additional pattern-specific parameters
        
    Returns:
        Simulation object or channel descriptor depending on pattern type
    """
    # Convert channel index to Channel object if needed
    if isinstance(channel, int):
        channel = Channel(channel, f"Pattern_{channel}")
    
    # Get simulation channel
    sim_channel = manager.group.add(channel, manager.sample_rate, BitState.LOW)
    
    if pattern_type == 'stair':
        # Create a stair/ramp pattern
        bit_width = kwargs.get('bit_width', 100)
        steps = kwargs.get('steps', 8)
        
        for step in range(steps):
            # Create a bit pattern representing the step
            pattern = []
            for i in range(8):
                if i < step:
                    pattern.append(BitState.HIGH)
                else:
                    pattern.append(BitState.LOW)
            
            # Output the pattern
            SignalGenerator.bit_pattern(sim_channel, pattern, bit_width)
    
    elif pattern_type == 'counter':
        # Create a binary counter pattern
        bit_width = kwargs.get('bit_width', 100)
        count = kwargs.get('count', 256)
        bits = kwargs.get('bits', 8)
        
        for value in range(count):
            # Create a bit pattern representing the value
            pattern = []
            for i in range(bits):
                if (value >> i) & 0x01:
                    pattern.append(BitState.HIGH)
                else:
                    pattern.append(BitState.LOW)
            
            # Output the pattern
            SignalGenerator.bit_pattern(sim_channel, pattern, bit_width)
    
    elif pattern_type == 'walking1':
        # Create a walking 1 pattern
        bit_width = kwargs.get('bit_width', 100)
        bits = kwargs.get('bits', 8)
        
        for pos in range(bits):
            # Create a bit pattern with a single 1 at position pos
            pattern = []
            for i in range(bits):
                if i == pos:
                    pattern.append(BitState.HIGH)
                else:
                    pattern.append(BitState.LOW)
            
            # Output the pattern
            SignalGenerator.bit_pattern(sim_channel, pattern, bit_width)
    
    elif pattern_type == 'walking0':
        # Create a walking 0 pattern
        bit_width = kwargs.get('bit_width', 100)
        bits = kwargs.get('bits', 8)
        
        for pos in range(bits):
            # Create a bit pattern with a single 0 at position pos
            pattern = []
            for i in range(bits):
                if i == pos:
                    pattern.append(BitState.LOW)
                else:
                    pattern.append(BitState.HIGH)
            
            # Output the pattern
            SignalGenerator.bit_pattern(sim_channel, pattern, bit_width)
    
    elif pattern_type == 'pwm':
        # Create a PWM signal with varying duty cycle
        frequency_hz = kwargs.get('frequency_hz', 1000)
        start_percent = kwargs.get('start_percent', 10)
        end_percent = kwargs.get('end_percent', 90)
        steps = kwargs.get('steps', 9)
        
        for step in range(steps):
            # Calculate duty cycle for this step
            duty_cycle = start_percent + (end_percent - start_percent) * step / (steps - 1)
            # Generate a few cycles at this duty cycle
            manager.add_pwm(channel, frequency_hz, duty_cycle, count=5)
    
    return sim_channel