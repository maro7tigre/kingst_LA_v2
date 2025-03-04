// bind_simulation.cpp
//
// Python bindings for the SimulationChannelDescriptor classes

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "SimulationChannelDescriptor.h"
#include "LogicPublicTypes.h"

namespace py = pybind11;

void init_simulation(py::module_ &m) {
    //------------------------------------------------------------------
    // SimulationChannelDescriptor class
    //------------------------------------------------------------------
    py::class_<SimulationChannelDescriptor> sim_channel(m, "SimulationChannelDescriptor", R"pbdoc(
        Represents a channel in simulation mode.
        
        SimulationChannelDescriptor is used to create simulated data for testing
        analyzers without requiring physical hardware. It allows you to generate
        bit patterns by setting transitions at specific sample points.
        
        Note:
            Instances of this class are typically created and managed by a
            SimulationChannelDescriptorGroup. Do not delete returned pointers.
    )pbdoc");

    //------------------------------------------------------------------
    // Core methods (primary functional methods)
    //------------------------------------------------------------------
    
    sim_channel.def("transition", &SimulationChannelDescriptor::Transition, R"pbdoc(
        Toggle the current bit state (from HIGH to LOW or LOW to HIGH).
        
        Creates a transition at the current sample position.
        
        Returns:
            None
    )pbdoc");

    sim_channel.def("transition_if_needed", &SimulationChannelDescriptor::TransitionIfNeeded, R"pbdoc(
        Transition to a specific bit state if not already at that state.
        
        Args:
            bit_state (BitState): Desired bit state (HIGH or LOW)
            
        Returns:
            None
    )pbdoc", py::arg("bit_state"));

    sim_channel.def("advance", &SimulationChannelDescriptor::Advance, R"pbdoc(
        Move forward by a specified number of samples.
        
        This advances the current sample position without changing the bit state.
        
        Args:
            num_samples_to_advance (U32): Number of samples to move forward
            
        Returns:
            None
    )pbdoc", py::arg("num_samples_to_advance"));

    //------------------------------------------------------------------
    // State methods (query current state)
    //------------------------------------------------------------------
    
    sim_channel.def("get_current_bit_state", &SimulationChannelDescriptor::GetCurrentBitState, R"pbdoc(
        Get the current bit state.
        
        Returns:
            BitState: Current bit state (HIGH or LOW)
    )pbdoc");

    sim_channel.def("get_current_sample_number", &SimulationChannelDescriptor::GetCurrentSampleNumber, R"pbdoc(
        Get the current sample position.
        
        Returns:
            U64: Current sample number
    )pbdoc");

    //------------------------------------------------------------------
    // Configuration methods
    //------------------------------------------------------------------
    
    sim_channel.def("set_channel", &SimulationChannelDescriptor::SetChannel, R"pbdoc(
        Set the channel that this descriptor is associated with.
        
        Args:
            channel (Channel): Channel to associate with this descriptor
            
        Returns:
            None
    )pbdoc", py::arg("channel"));

    sim_channel.def("set_sample_rate", &SimulationChannelDescriptor::SetSampleRate, R"pbdoc(
        Set the sample rate for simulation.
        
        Args:
            sample_rate_hz (U32): Sample rate in Hz
            
        Returns:
            None
    )pbdoc", py::arg("sample_rate_hz"));

    sim_channel.def("set_initial_bit_state", &SimulationChannelDescriptor::SetInitialBitState, R"pbdoc(
        Set the initial bit state for the channel.
        
        Args:
            initial_bit_state (BitState): Initial bit state (HIGH or LOW)
            
        Returns:
            None
    )pbdoc", py::arg("initial_bit_state"));

    //------------------------------------------------------------------
    // Access methods
    //------------------------------------------------------------------
    
    sim_channel.def("get_channel", &SimulationChannelDescriptor::GetChannel, R"pbdoc(
        Get the channel associated with this descriptor.
        
        Returns:
            Channel: Associated channel
    )pbdoc");

    sim_channel.def("get_sample_rate", &SimulationChannelDescriptor::GetSampleRate, R"pbdoc(
        Get the sample rate for simulation.
        
        Returns:
            U32: Sample rate in Hz
    )pbdoc");

    sim_channel.def("get_initial_bit_state", &SimulationChannelDescriptor::GetInitialBitState, R"pbdoc(
        Get the initial bit state for the channel.
        
        Returns:
            BitState: Initial bit state (HIGH or LOW)
    )pbdoc");

    // _get_data is intentionally not exposed as it's marked "don't use" in the header
    
    //------------------------------------------------------------------
    // Python-specific helper methods (digital signal generation)
    //------------------------------------------------------------------
    
    sim_channel.def("add_pulse", [](SimulationChannelDescriptor &self, U32 width, U32 count, U32 gap) {
        BitState current_state = self.GetCurrentBitState();
        
        for (U32 i = 0; i < count; i++) {
            // Toggle to create leading edge of pulse
            self.Transition();
            
            // Advance for pulse width
            self.Advance(width);
            
            // Toggle to create trailing edge of pulse
            self.Transition();
            
            // Advance for gap between pulses (if not the last pulse)
            if (i < count - 1) {
                self.Advance(gap);
            }
        }
    }, R"pbdoc(
        Add a series of pulses.
        
        Creates a series of pulses (transitions from the current state
        and back) with specified width and gaps between pulses.
        
        Args:
            width (U32): Width of each pulse in samples
            count (U32, optional): Number of pulses to create. Defaults to 1.
            gap (U32, optional): Gap between pulses in samples. Defaults to 0.
            
        Returns:
            None
    )pbdoc", py::arg("width"), py::arg("count") = 1, py::arg("gap") = 0);

    sim_channel.def("add_bit_pattern", [](SimulationChannelDescriptor &self, 
                                         const std::vector<BitState> &pattern, 
                                         U32 samples_per_bit) {
        for (BitState bit : pattern) {
            // Transition if needed to match the desired bit state
            self.TransitionIfNeeded(bit);
            
            // Advance for the duration of this bit
            self.Advance(samples_per_bit);
        }
    }, R"pbdoc(
        Add a specific bit pattern.
        
        Creates a sequence of bits with specified duration per bit.
        
        Args:
            pattern (list[BitState]): List of BitStates to generate
            samples_per_bit (U32): Number of samples for each bit
            
        Returns:
            None
            
        Example:
            >>> # Create a pattern with alternating HIGH and LOW bits
            >>> pattern = [BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW]
            >>> sim_channel.add_bit_pattern(pattern, 100)
    )pbdoc", py::arg("pattern"), py::arg("samples_per_bit"));

    sim_channel.def("add_uart_byte", [](SimulationChannelDescriptor &self, U8 byte, U32 bit_width, 
                                       bool with_start_bit, bool with_stop_bit, bool with_parity_bit,
                                       bool even_parity) {
        // UART idle state is HIGH, so ensure we're at that state
        self.TransitionIfNeeded(BIT_HIGH);
        
        // Start bit (if requested)
        if (with_start_bit) {
            // Start bit is always LOW
            self.Transition(); // Transition to LOW
            self.Advance(bit_width);
        }
        
        // Data bits (LSB first for UART)
        U8 parity = 0; // For tracking parity bit value
        for (int i = 0; i < 8; i++) {
            bool bit_value = (byte >> i) & 0x01;
            parity ^= bit_value; // XOR for parity calculation
            
            BitState bit_state = bit_value ? BIT_HIGH : BIT_LOW;
            self.TransitionIfNeeded(bit_state);
            self.Advance(bit_width);
        }
        
        // Parity bit (if requested)
        if (with_parity_bit) {
            // For even parity, parity bit = 1 if odd number of 1's in data
            // For odd parity, parity bit = 1 if even number of 1's in data
            bool parity_value = even_parity ? (parity == 1) : (parity == 0);
            BitState parity_state = parity_value ? BIT_HIGH : BIT_LOW;
            
            self.TransitionIfNeeded(parity_state);
            self.Advance(bit_width);
        }
        
        // Stop bit (if requested)
        if (with_stop_bit) {
            // Stop bit is always HIGH
            self.TransitionIfNeeded(BIT_HIGH);
            self.Advance(bit_width);
        }
    }, R"pbdoc(
        Add a UART byte transmission.
        
        Creates a complete UART byte transmission including optional
        start bit, stop bit, and parity bit.
        
        Args:
            byte (U8): Data byte to transmit
            bit_width (U32): Width of each bit in samples
            with_start_bit (bool, optional): Include start bit. Defaults to True.
            with_stop_bit (bool, optional): Include stop bit. Defaults to True.
            with_parity_bit (bool, optional): Include parity bit. Defaults to False.
            even_parity (bool, optional): Use even parity (if with_parity_bit is True).
                                         Defaults to True.
        
        Returns:
            None
            
        Note:
            For standard UART, start bit is LOW, stop bit is HIGH, and data is LSB first.
    )pbdoc", 
    py::arg("byte"), 
    py::arg("bit_width"), 
    py::arg("with_start_bit") = true, 
    py::arg("with_stop_bit") = true, 
    py::arg("with_parity_bit") = false, 
    py::arg("even_parity") = true);

    sim_channel.def("add_i2c_bit", [](SimulationChannelDescriptor &self, bool bit_value, U32 bit_width,
                                     bool is_scl) {
        if (is_scl) {
            // SCL
            // Clock starts low
            self.TransitionIfNeeded(BIT_LOW);
            self.Advance(bit_width / 4);
            
            // Clock goes high
            self.Transition();
            self.Advance(bit_width / 2);
            
            // Clock goes low again
            self.Transition();
            self.Advance(bit_width / 4);
        } else {
            // SDA - set data before SCL rises
            self.TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
            self.Advance(bit_width);
        }
    }, R"pbdoc(
        Add a single I2C bit.
        
        Creates the proper signal pattern for a single I2C bit on either SCL or SDA.
        
        Args:
            bit_value (bool): Value of the bit (for SDA only, ignored for SCL)
            bit_width (U32): Width of each bit in samples
            is_scl (bool): True if this is the SCL (clock) channel, False for SDA (data)
            
        Returns:
            None
            
        Note:
            This is a low-level primitive that doesn't handle I2C protocol specifics.
            For full I2C transactions, use the simulator group's higher-level methods.
    )pbdoc",
    py::arg("bit_value"),
    py::arg("bit_width"),
    py::arg("is_scl"));

    sim_channel.def("add_i2c_byte", [](SimulationChannelDescriptor &self, U8 byte, bool with_ack, 
                                      U32 bit_width, bool is_scl) {
        if (is_scl) {
            // For SCL, generate 9 clock cycles (8 data bits + 1 ACK bit)
            for (int i = 0; i < 9; i++) {
                // Clock starts low
                self.TransitionIfNeeded(BIT_LOW);
                self.Advance(bit_width / 4);
                
                // Clock goes high
                self.Transition();
                self.Advance(bit_width / 2);
                
                // Clock goes low again
                self.Transition();
                self.Advance(bit_width / 4);
            }
        } else {
            // For SDA, generate 8 data bits (MSB first) + ACK bit
            for (int i = 7; i >= 0; i--) {
                bool bit_value = (byte >> i) & 0x01;
                BitState bit_state = bit_value ? BIT_HIGH : BIT_LOW;
                
                // Set data while clock is low
                self.TransitionIfNeeded(bit_state);
                self.Advance(bit_width / 2);
                
                // Hold data while clock is high
                self.Advance(bit_width / 2);
            }
            
            // ACK bit (pulled low by receiver to acknowledge)
            if (with_ack) {
                // ACK is active low
                self.TransitionIfNeeded(BIT_LOW);
            } else {
                // NACK is high (pulled up)
                self.TransitionIfNeeded(BIT_HIGH);
            }
            
            // Hold ACK for a full bit width
            self.Advance(bit_width);
        }
    }, R"pbdoc(
        Add an I2C byte transmission.
        
        Creates an I2C byte transmission including the ACK bit.
        
        Args:
            byte (U8): Data byte to transmit
            with_ack (bool, optional): Include ACK bit (pulled low). Defaults to True.
            bit_width (U32, optional): Width of each bit in samples. Defaults to 100.
            is_scl (bool, optional): True if this is the SCL (clock) channel, 
                                    False for SDA (data). Defaults to False.
        
        Returns:
            None
            
        Note:
            I2C bytes are transmitted MSB first.
            This method handles only the data phase, not START/STOP conditions.
    )pbdoc", 
    py::arg("byte"), 
    py::arg("with_ack") = true, 
    py::arg("bit_width") = 100, 
    py::arg("is_scl") = false);

    sim_channel.def("add_i2c_start", [](SimulationChannelDescriptor &self, U32 bit_width, bool is_scl) {
        if (is_scl) {
            // SCL - remains HIGH during SDA transition
            self.TransitionIfNeeded(BIT_HIGH);
            self.Advance(bit_width);
        } else {
            // SDA - transitions from HIGH to LOW while SCL is HIGH
            self.TransitionIfNeeded(BIT_HIGH);
            self.Advance(bit_width / 2);
            self.Transition(); // HIGH to LOW
            self.Advance(bit_width / 2);
        }
    }, R"pbdoc(
        Add an I2C START condition.
        
        Creates an I2C START condition (SDA falling edge while SCL is HIGH).
        
        Args:
            bit_width (U32): Width of the START condition in samples
            is_scl (bool): True if this is the SCL channel, False for SDA
            
        Returns:
            None
    )pbdoc",
    py::arg("bit_width"),
    py::arg("is_scl") = false);

    sim_channel.def("add_i2c_stop", [](SimulationChannelDescriptor &self, U32 bit_width, bool is_scl) {
        if (is_scl) {
            // SCL - remains HIGH during SDA transition
            self.TransitionIfNeeded(BIT_HIGH);
            self.Advance(bit_width);
        } else {
            // SDA - transitions from LOW to HIGH while SCL is HIGH
            self.TransitionIfNeeded(BIT_LOW);
            self.Advance(bit_width / 2);
            self.Transition(); // LOW to HIGH
            self.Advance(bit_width / 2);
        }
    }, R"pbdoc(
        Add an I2C STOP condition.
        
        Creates an I2C STOP condition (SDA rising edge while SCL is HIGH).
        
        Args:
            bit_width (U32): Width of the STOP condition in samples
            is_scl (bool): True if this is the SCL channel, False for SDA
            
        Returns:
            None
    )pbdoc",
    py::arg("bit_width"),
    py::arg("is_scl") = false);

    sim_channel.def("add_spi_bit", [](SimulationChannelDescriptor &self, bool bit_value, U32 bit_width,
                                     bool cpol, bool cpha, bool is_clock) {
        if (is_clock) {
            // CPOL determines idle state
            BitState idle_state = cpol ? BIT_HIGH : BIT_LOW;
            
            // First ensure we're at idle state
            self.TransitionIfNeeded(idle_state);
            
            // First clock edge
            self.Advance(bit_width / 4);
            self.Transition();
            
            // Second clock edge
            self.Advance(bit_width / 2);
            self.Transition();
            
            // Return to idle position
            self.Advance(bit_width / 4);
        } else {
            // For data line
            self.TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
            self.Advance(bit_width);
        }
    }, R"pbdoc(
        Add a single SPI bit.
        
        Creates the proper signal pattern for a single SPI bit on either clock or data lines.
        
        Args:
            bit_value (bool): Value of the bit (for data lines only, ignored for clock)
            bit_width (U32): Width of each bit in samples
            cpol (bool): Clock polarity (0 = idle low, 1 = idle high)
            cpha (bool): Clock phase (0 = sample on first edge, 1 = sample on second edge)
            is_clock (bool): True if this is the clock channel, False for data lines
            
        Returns:
            None
    )pbdoc",
    py::arg("bit_value"),
    py::arg("bit_width"),
    py::arg("cpol") = false,
    py::arg("cpha") = false,
    py::arg("is_clock") = false);

    //------------------------------------------------------------------
    // SimulationChannelDescriptorGroup class
    //------------------------------------------------------------------

    py::class_<SimulationChannelDescriptorGroup> sim_group(m, "SimulationChannelDescriptorGroup", R"pbdoc(
        A group of simulation channels.
        
        This class manages a collection of SimulationChannelDescriptor objects
        for convenient multichannel simulation.
        
        Example:
            >>> group = SimulationChannelDescriptorGroup()
            >>> channel0 = Channel(0, 0)
            >>> sim_ch0 = group.add(channel0, 1000000, BitState.LOW)
    )pbdoc");

    sim_group.def(py::init<>(), R"pbdoc(
        Initialize a new simulation channel group.
        
        Returns:
            SimulationChannelDescriptorGroup: A new group instance
    )pbdoc");

    sim_group.def("add", &SimulationChannelDescriptorGroup::Add, R"pbdoc(
        Add a channel to the group.
        
        Args:
            channel (Channel): Channel to add
            sample_rate (U32): Sample rate in Hz
            initial_bit_state (BitState): Initial bit state (HIGH or LOW)
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created descriptor
            
        Note:
            Do not delete the returned pointer. The pointer is managed by the group.
    )pbdoc", 
    py::arg("channel"), 
    py::arg("sample_rate"), 
    py::arg("initial_bit_state"), 
    py::return_value_policy::reference);

    sim_group.def("advance_all", &SimulationChannelDescriptorGroup::AdvanceAll, R"pbdoc(
        Advance all channels by the same number of samples.
        
        This method moves all channels forward together while maintaining
        their individual bit states.
        
        Args:
            num_samples_to_advance (U32): Number of samples to advance
            
        Returns:
            None
    )pbdoc", py::arg("num_samples_to_advance"));

    sim_group.def("get_array", &SimulationChannelDescriptorGroup::GetArray, R"pbdoc(
        Get the array of channel descriptors.
        
        Returns:
            SimulationChannelDescriptor*: Pointer to the array of descriptors
            
        Note:
            Usually not needed in Python; use list access instead.
    )pbdoc", py::return_value_policy::reference);

    sim_group.def("get_count", &SimulationChannelDescriptorGroup::GetCount, R"pbdoc(
        Get the number of channels in the group.
        
        Returns:
            U32: Number of channels
    )pbdoc");

    //------------------------------------------------------------------
    // Python-specific sequence interface for SimulationChannelDescriptorGroup
    //------------------------------------------------------------------
    
    sim_group.def("__len__", &SimulationChannelDescriptorGroup::GetCount, R"pbdoc(
        Get the number of channels in the group.
        
        Returns:
            int: Number of channels
    )pbdoc");
    
    sim_group.def("__getitem__", [](SimulationChannelDescriptorGroup &self, U32 index) {
        if (index >= self.GetCount()) {
            throw py::index_error("Index out of range");
        }
        return &(self.GetArray()[index]);
    }, R"pbdoc(
        Get a channel descriptor by index.
        
        Args:
            index (int): Zero-based index of the descriptor
            
        Returns:
            SimulationChannelDescriptor: The descriptor at the given index
            
        Raises:
            IndexError: If index is out of range
    )pbdoc", py::return_value_policy::reference);

    //------------------------------------------------------------------
    // Protocol simulation helpers for SimulationChannelDescriptorGroup
    //------------------------------------------------------------------
    
    sim_group.def("add_uart_simulation", [](SimulationChannelDescriptorGroup &self, 
                                           Channel &uart_tx_channel,
                                           const std::vector<U8> &data,
                                           U32 baud_rate,
                                           U32 sample_rate,
                                           bool with_start_bit,
                                           bool with_stop_bit,
                                           bool with_parity_bit,
                                           bool even_parity) {
        // Calculate bit width based on baud rate and sample rate
        U32 bit_width = sample_rate / baud_rate;
        
        // Add UART TX channel to group
        SimulationChannelDescriptor *uart_tx = self.Add(uart_tx_channel, sample_rate, BIT_HIGH);
        
        // Start with some idle time (10 bit widths)
        uart_tx->Advance(bit_width * 10);
        
        // Add each byte
        for (U8 byte : data) {
            // Start bit (if requested)
            if (with_start_bit) {
                // Start bit is always LOW
                uart_tx->Transition(); // Transition to LOW
                uart_tx->Advance(bit_width);
            }
            
            // Data bits (LSB first for UART)
            U8 parity = 0; // For tracking parity bit value
            for (int i = 0; i < 8; i++) {
                bool bit_value = ((byte >> i) & 0x01) != 0;
                parity ^= (bit_value ? 1 : 0); // Safe way to XOR boolean with integer

                BitState bit_state = bit_value ? BIT_HIGH : BIT_LOW;
                // ... rest of the code
            }
            
            // Parity bit (if requested)
            if (with_parity_bit) {
                // For even parity, parity bit = 1 if odd number of 1's in data
                // For odd parity, parity bit = 1 if even number of 1's in data
                bool parity_value = even_parity ? (parity == 1) : (parity == 0);
                BitState parity_state = parity_value ? BIT_HIGH : BIT_LOW;
                
                uart_tx->TransitionIfNeeded(parity_state);
                uart_tx->Advance(bit_width);
            }
            
            // Stop bit (if requested)
            if (with_stop_bit) {
                // Stop bit is always HIGH
                uart_tx->TransitionIfNeeded(BIT_HIGH);
                uart_tx->Advance(bit_width);
            }
            
            // Inter-byte gap (1 bit width)
            uart_tx->Advance(bit_width);
        }
        
        // Final idle time
        uart_tx->Advance(bit_width * 10);
        
        return uart_tx;
    }, R"pbdoc(
        Add a UART transmission simulation.
        
        Creates a complete UART transmission for a sequence of bytes.
        
        Args:
            uart_tx_channel (Channel): Channel for UART TX
            data (list[U8]): List of bytes to transmit
            baud_rate (U32): UART baud rate in bits per second
            sample_rate (U32): Sample rate in Hz
            with_start_bit (bool, optional): Include start bit. Defaults to True.
            with_stop_bit (bool, optional): Include stop bit. Defaults to True.
            with_parity_bit (bool, optional): Include parity bit. Defaults to False.
            even_parity (bool, optional): Use even parity (if with_parity_bit is True).
                                         Defaults to True.
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created TX channel descriptor
            
        Note:
            The returned pointer is owned by the group and should not be deleted.
    )pbdoc", 
    py::arg("uart_tx_channel"), 
    py::arg("data"), 
    py::arg("baud_rate"), 
    py::arg("sample_rate"), 
    py::arg("with_start_bit") = true, 
    py::arg("with_stop_bit") = true, 
    py::arg("with_parity_bit") = false, 
    py::arg("even_parity") = true,
    py::return_value_policy::reference);

    sim_group.def("add_uart_simulation_with_rx", [](SimulationChannelDescriptorGroup &self, 
                                           Channel &uart_tx_channel,
                                           Channel &uart_rx_channel,
                                           const std::vector<U8> &tx_data,
                                           const std::vector<U8> &rx_data,
                                           U32 baud_rate,
                                           U32 sample_rate,
                                           bool with_start_bit,
                                           bool with_stop_bit,
                                           bool with_parity_bit,
                                           bool even_parity) {
        // Calculate bit width based on baud rate and sample rate
        U32 bit_width = sample_rate / baud_rate;
        
        // Add UART channels to group
        SimulationChannelDescriptor *uart_tx = self.Add(uart_tx_channel, sample_rate, BIT_HIGH);
        SimulationChannelDescriptor *uart_rx = self.Add(uart_rx_channel, sample_rate, BIT_HIGH);
        
        // Start with some idle time (10 bit widths)
        self.AdvanceAll(bit_width * 10);
        
        // Process TX data first
        for (U8 byte : tx_data) {
            // Using the same UART byte format
            if (with_start_bit) {
                uart_tx->Transition(); // Transition to LOW
                uart_tx->Advance(bit_width);
            }
            
            U8 parity = 0;
            for (int i = 0; i < 8; i++) {
                bool bit_value = (byte >> i) & 0x01;
                parity ^= bit_value;
                
                BitState bit_state = bit_value ? BIT_HIGH : BIT_LOW;
                uart_tx->TransitionIfNeeded(bit_state);
                uart_tx->Advance(bit_width);
            }
            
            if (with_parity_bit) {
                bool parity_value = even_parity ? (parity == 1) : (parity == 0);
                BitState parity_state = parity_value ? BIT_HIGH : BIT_LOW;
                
                uart_tx->TransitionIfNeeded(parity_state);
                uart_tx->Advance(bit_width);
            }
            
            if (with_stop_bit) {
                uart_tx->TransitionIfNeeded(BIT_HIGH);
                uart_tx->Advance(bit_width);
            }
            
            uart_tx->Advance(bit_width);
        }
        
        // Add a gap between TX and RX (20 bit widths)
        self.AdvanceAll(bit_width * 20);
        
        // Process RX data
        for (U8 byte : rx_data) {
            if (with_start_bit) {
                uart_rx->Transition(); // Transition to LOW
                uart_rx->Advance(bit_width);
            }
            
            U8 parity = 0;
            for (int i = 0; i < 8; i++) {
                bool bit_value = (byte >> i) & 0x01;
                parity ^= bit_value;
                
                BitState bit_state = bit_value ? BIT_HIGH : BIT_LOW;
                uart_rx->TransitionIfNeeded(bit_state);
                uart_rx->Advance(bit_width);
            }
            
            if (with_parity_bit) {
                bool parity_value = even_parity ? (parity == 1) : (parity == 0);
                BitState parity_state = parity_value ? BIT_HIGH : BIT_LOW;
                
                uart_rx->TransitionIfNeeded(parity_state);
                uart_rx->Advance(bit_width);
            }
            
            if (with_stop_bit) {
                uart_rx->TransitionIfNeeded(BIT_HIGH);
                uart_rx->Advance(bit_width);
            }
            
            uart_rx->Advance(bit_width);
        }
        
        // Final idle time
        self.AdvanceAll(bit_width * 10);
        
        return std::make_tuple(uart_tx, uart_rx);
    }, R"pbdoc(
        Add a bidirectional UART simulation with both TX and RX channels.
        
        Creates a complete UART simulation with both transmit and receive channels.
        
        Args:
            uart_tx_channel (Channel): Channel for UART TX
            uart_rx_channel (Channel): Channel for UART RX
            tx_data (list[U8]): List of bytes to transmit
            rx_data (list[U8]): List of bytes to receive
            baud_rate (U32): UART baud rate in bits per second
            sample_rate (U32): Sample rate in Hz
            with_start_bit (bool, optional): Include start bit. Defaults to True.
            with_stop_bit (bool, optional): Include stop bit. Defaults to True.
            with_parity_bit (bool, optional): Include parity bit. Defaults to False.
            even_parity (bool, optional): Use even parity (if with_parity_bit is True).
                                         Defaults to True.
            
        Returns:
            tuple: (tx_channel, rx_channel) pointers to created channel descriptors
            
        Note:
            The returned pointers are owned by the group and should not be deleted.
            TX data is transmitted first, followed by a gap, then RX data.
    )pbdoc", 
    py::arg("uart_tx_channel"), 
    py::arg("uart_rx_channel"),
    py::arg("tx_data"), 
    py::arg("rx_data"),
    py::arg("baud_rate"), 
    py::arg("sample_rate"), 
    py::arg("with_start_bit") = true, 
    py::arg("with_stop_bit") = true, 
    py::arg("with_parity_bit") = false, 
    py::arg("even_parity") = true,
    py::return_value_policy::reference);

    // I2C simulation helper
    sim_group.def("add_i2c_simulation", [](SimulationChannelDescriptorGroup &self,
                                          Channel &scl_channel,
                                          Channel &sda_channel,
                                          U8 address,
                                          bool is_read,
                                          const std::vector<U8> &data,
                                          U32 clock_rate,
                                          U32 sample_rate) {
        // Calculate bit width based on clock rate and sample rate
        U32 bit_width = sample_rate / clock_rate;
        
        // Add I2C channels to group
        SimulationChannelDescriptor *scl = self.Add(scl_channel, sample_rate, BIT_HIGH);
        SimulationChannelDescriptor *sda = self.Add(sda_channel, sample_rate, BIT_HIGH);
        
        // Start with some idle time (both lines HIGH)
        self.AdvanceAll(bit_width * 4);
        
        // Start condition
        // SDA goes LOW while SCL is HIGH
        sda->Transition(); // HIGH to LOW
        sda->Advance(bit_width / 2);
        scl->Advance(bit_width / 2);
        
        // Clock goes LOW
        scl->Transition();
        scl->Advance(bit_width / 2);
        sda->Advance(bit_width / 2);
        
        // Address byte (7-bit address + R/W bit)
        U8 address_byte = (address << 1) | (is_read ? 1 : 0);
        
        // Send address byte MSB first
        for (int i = 7; i >= 0; i--) {
            bool bit_value = (address_byte >> i) & 0x01;
            
            // Set SDA
            sda->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
            sda->Advance(bit_width / 4);
            scl->Advance(bit_width / 4);
            
            // SCL goes HIGH
            scl->Transition();
            sda->Advance(bit_width / 2);
            scl->Advance(bit_width / 2);
            
            // SCL goes LOW
            scl->Transition();
            sda->Advance(bit_width / 4);
            scl->Advance(bit_width / 4);
        }
        
        // ACK from slave (SDA pulled LOW)
        sda->TransitionIfNeeded(BIT_LOW);
        sda->Advance(bit_width / 4);
        scl->Advance(bit_width / 4);
        
        // SCL goes HIGH
        scl->Transition();
        sda->Advance(bit_width / 2);
        scl->Advance(bit_width / 2);
        
        // SCL goes LOW
        scl->Transition();
        sda->Advance(bit_width / 4);
        scl->Advance(bit_width / 4);
        
        // Data transfer
        for (U8 byte : data) {
            // If reading, SDA control transfers to slave
            BitState data_source = is_read ? BIT_LOW : BIT_HIGH;
            
            // Send data byte MSB first
            for (int i = 7; i >= 0; i--) {
                bool bit_value = (byte >> i) & 0x01;
                
                // Set SDA
                sda->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
                sda->Advance(bit_width / 4);
                scl->Advance(bit_width / 4);
                
                // SCL goes HIGH
                scl->Transition();
                sda->Advance(bit_width / 2);
                scl->Advance(bit_width / 2);
                
                // SCL goes LOW
                scl->Transition();
                sda->Advance(bit_width / 4);
                scl->Advance(bit_width / 4);
            }
            
            // ACK/NACK
            // For write: slave sends ACK (SDA LOW)
            // For last read byte: master sends NACK (SDA HIGH)
            // For non-last read byte: master sends ACK (SDA LOW)
            bool is_last_byte = (&byte == &data.back());
            bool send_nack = is_read && is_last_byte;
            
            sda->TransitionIfNeeded(send_nack ? BIT_HIGH : BIT_LOW);
            sda->Advance(bit_width / 4);
            scl->Advance(bit_width / 4);
            
            // SCL goes HIGH
            scl->Transition();
            sda->Advance(bit_width / 2);
            scl->Advance(bit_width / 2);
            
            // SCL goes LOW
            scl->Transition();
            sda->Advance(bit_width / 4);
            scl->Advance(bit_width / 4);
        }
        
        // Stop condition
        // First ensure SDA is LOW
        sda->TransitionIfNeeded(BIT_LOW);
        sda->Advance(bit_width / 4);
        scl->Advance(bit_width / 4);
        
        // SCL goes HIGH
        scl->Transition();
        sda->Advance(bit_width / 2);
        scl->Advance(bit_width / 2);
        
        // SDA goes HIGH while SCL is HIGH (STOP)
        sda->Transition();
        sda->Advance(bit_width / 2);
        scl->Advance(bit_width / 2);
        
        // Final idle time
        self.AdvanceAll(bit_width * 4);
        
        return std::make_tuple(scl, sda);
    }, R"pbdoc(
        Add an I2C transaction simulation.
        
        Creates a complete I2C transaction including START, address, data bytes, and STOP.
        
        Args:
            scl_channel (Channel): Channel for I2C clock (SCL)
            sda_channel (Channel): Channel for I2C data (SDA)
            address (U8): 7-bit I2C device address
            is_read (bool): True for read operation, False for write operation
            data (list[U8]): Data bytes to read or write
            clock_rate (U32): I2C clock rate in Hz
            sample_rate (U32): Sample rate in Hz
            
        Returns:
            tuple: (scl, sda) pointers to the created channel descriptors
            
        Note:
            The returned pointers are owned by the group and should not be deleted.
            For write operations, the data is sent from master to slave.
            For read operations, the data represents the slave's response.
    )pbdoc", 
    py::arg("scl_channel"), 
    py::arg("sda_channel"), 
    py::arg("address"), 
    py::arg("is_read"), 
    py::arg("data"), 
    py::arg("clock_rate"), 
    py::arg("sample_rate"),
    py::return_value_policy::reference);

    // SPI simulation helper
    sim_group.def("add_spi_simulation", [](SimulationChannelDescriptorGroup &self,
                                          Channel &sck_channel,
                                          Channel &mosi_channel,
                                          Channel &miso_channel,
                                          Channel &cs_channel,
                                          const std::vector<U8> &mosi_data,
                                          const std::vector<U8> &miso_data,
                                          U32 clock_rate,
                                          U32 sample_rate,
                                          bool cpol,
                                          bool cpha,
                                          bool msb_first,
                                          bool cs_active_low) {
        // Calculate bit width based on clock rate and sample rate
        U32 bit_width = sample_rate / clock_rate;
        
        // Add SPI channels to group
        SimulationChannelDescriptor *sck = self.Add(sck_channel, sample_rate, cpol ? BIT_HIGH : BIT_LOW);
        SimulationChannelDescriptor *mosi = self.Add(mosi_channel, sample_rate, BIT_LOW);
        SimulationChannelDescriptor *miso = self.Add(miso_channel, sample_rate, BIT_LOW);
        SimulationChannelDescriptor *cs = self.Add(cs_channel, sample_rate, cs_active_low ? BIT_HIGH : BIT_LOW);
        
        // Start with some idle time
        self.AdvanceAll(bit_width * 4);
        
        // Activate chip select
        cs->TransitionIfNeeded(cs_active_low ? BIT_LOW : BIT_HIGH);
        self.AdvanceAll(bit_width / 2);
        
        // Data size validation
        size_t max_bytes = std::max(mosi_data.size(), miso_data.size());
        
        // Process each byte
        for (size_t byte_idx = 0; byte_idx < max_bytes; byte_idx++) {
            U8 mosi_byte = byte_idx < mosi_data.size() ? mosi_data[byte_idx] : 0;
            U8 miso_byte = byte_idx < miso_data.size() ? miso_data[byte_idx] : 0;
            
            // Process each bit
            for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
                // Determine which bit in the byte we're processing
                int actual_bit_idx = msb_first ? (7 - bit_idx) : bit_idx;
                
                // Get the bit values
                bool mosi_bit = (mosi_byte >> actual_bit_idx) & 0x01;
                bool miso_bit = (miso_byte >> actual_bit_idx) & 0x01;
                
                // Set data lines based on CPHA
                if (cpha == 0) {
                    // For CPHA=0, data is valid on leading edge of clock
                    // Set data before clock toggles
                    mosi->TransitionIfNeeded(mosi_bit ? BIT_HIGH : BIT_LOW);
                    miso->TransitionIfNeeded(miso_bit ? BIT_HIGH : BIT_LOW);
                    self.AdvanceAll(bit_width / 4);
                    
                    // First clock edge (leading)
                    sck->Transition();
                    self.AdvanceAll(bit_width / 2);
                    
                    // Second clock edge (trailing)
                    sck->Transition();
                    self.AdvanceAll(bit_width / 4);
                } else {
                    // For CPHA=1, data is valid on trailing edge of clock
                    // First clock edge
                    sck->Transition();
                    self.AdvanceAll(bit_width / 4);
                    
                    // Set data after first clock edge
                    mosi->TransitionIfNeeded(mosi_bit ? BIT_HIGH : BIT_LOW);
                    miso->TransitionIfNeeded(miso_bit ? BIT_HIGH : BIT_LOW);
                    self.AdvanceAll(bit_width / 2);
                    
                    // Second clock edge
                    sck->Transition();
                    self.AdvanceAll(bit_width / 4);
                }
            }
            
            // Inter-byte gap (optional, depending on protocol)
            self.AdvanceAll(bit_width / 2);
        }
        
        // Deactivate chip select
        cs->TransitionIfNeeded(cs_active_low ? BIT_HIGH : BIT_LOW);
        self.AdvanceAll(bit_width * 4);
        
        return std::make_tuple(sck, mosi, miso, cs);
    }, R"pbdoc(
        Add an SPI transaction simulation.
        
        Creates a complete SPI transaction with configurable clock polarity, phase,
        and bit order.
        
        Args:
            sck_channel (Channel): Channel for SPI clock (SCK)
            mosi_channel (Channel): Channel for Master Out Slave In (MOSI)
            miso_channel (Channel): Channel for Master In Slave Out (MISO)
            cs_channel (Channel): Channel for Chip Select (CS)
            mosi_data (list[U8]): Data bytes to send from master to slave
            miso_data (list[U8]): Data bytes to send from slave to master
            clock_rate (U32): SPI clock rate in Hz
            sample_rate (U32): Sample rate in Hz
            cpol (bool, optional): Clock polarity (0 = idle low, 1 = idle high). Defaults to False.
            cpha (bool, optional): Clock phase (0 = sample on first edge, 1 = sample on second edge). 
                                  Defaults to False.
            msb_first (bool, optional): True for MSB first, False for LSB first. Defaults to True.
            cs_active_low (bool, optional): True if chip select is active low. Defaults to True.
            
        Returns:
            tuple: (sck, mosi, miso, cs) pointers to the created channel descriptors
            
        Note:
            The returned pointers are owned by the group and should not be deleted.
            If mosi_data and miso_data have different lengths, the longer one determines
            the transaction length.
    )pbdoc", 
    py::arg("sck_channel"), 
    py::arg("mosi_channel"), 
    py::arg("miso_channel"), 
    py::arg("cs_channel"), 
    py::arg("mosi_data"), 
    py::arg("miso_data"), 
    py::arg("clock_rate"), 
    py::arg("sample_rate"),
    py::arg("cpol") = false,
    py::arg("cpha") = false,
    py::arg("msb_first") = true,
    py::arg("cs_active_low") = true,
    py::return_value_policy::reference);

    // CAN simulation helper
    sim_group.def("add_can_simulation", [](SimulationChannelDescriptorGroup &self,
                                          Channel &can_channel,
                                          U32 identifier,
                                          bool extended_id,
                                          const std::vector<U8> &data,
                                          bool remote_frame,
                                          U32 bit_rate,
                                          U32 sample_rate) {
        // Calculate bit width based on bit rate and sample rate
        U32 bit_width = sample_rate / bit_rate;
        
        // Add CAN channel to group
        SimulationChannelDescriptor *can = self.Add(can_channel, sample_rate, BIT_HIGH); // CAN idle is HIGH (recessive)
        
        // Start with some idle time
        can->Advance(bit_width * 10);
        
        // Generate CAN frame
        
        // Start of Frame (SOF) - dominant bit (LOW)
        can->TransitionIfNeeded(BIT_LOW);
        can->Advance(bit_width);
        
        // Identifier Field (11-bit standard or 29-bit extended)
        if (extended_id) {
            // 29-bit Extended Identifier
            // First 11 bits
            for (int i = 10; i >= 0; i--) {
                bool bit_value = (identifier >> (i + 18)) & 0x01;
                can->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
                can->Advance(bit_width);
            }
            
            // SRR (Substitute Remote Request) - recessive bit (HIGH)
            can->TransitionIfNeeded(BIT_HIGH);
            can->Advance(bit_width);
            
            // IDE (Identifier Extension) - recessive bit (HIGH) for extended frame
            can->TransitionIfNeeded(BIT_HIGH);
            can->Advance(bit_width);
            
            // Extended 18 bits of identifier
            for (int i = 17; i >= 0; i--) {
                bool bit_value = (identifier >> i) & 0x01;
                can->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
                can->Advance(bit_width);
            }
        } else {
            // 11-bit Standard Identifier
            for (int i = 10; i >= 0; i--) {
                bool bit_value = (identifier >> i) & 0x01;
                can->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
                can->Advance(bit_width);
            }
            
            // RTR (Remote Transmission Request)
            can->TransitionIfNeeded(remote_frame ? BIT_HIGH : BIT_LOW);
            can->Advance(bit_width);
            
            // IDE (Identifier Extension) - dominant bit (LOW) for standard frame
            can->TransitionIfNeeded(BIT_LOW);
            can->Advance(bit_width);
        }
        
        // r0 Reserved bit - dominant (LOW)
        can->TransitionIfNeeded(BIT_LOW);
        can->Advance(bit_width);
        
        // DLC (Data Length Code) - 4 bits
        U8 dlc = remote_frame ? 0 : static_cast<U8>(std::min(static_cast<size_t>(8), data.size()));
        for (int i = 3; i >= 0; i--) {
            bool bit_value = (dlc >> i) & 0x01;
            can->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
            can->Advance(bit_width);
        }
        
        // Data Field (if not a remote frame)
        if (!remote_frame) {
            for (size_t byte_idx = 0; byte_idx < dlc; byte_idx++) {
                U8 byte = byte_idx < data.size() ? data[byte_idx] : 0;
                
                // Send 8 bits MSB first
                for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
                    bool bit_value = (byte >> bit_idx) & 0x01;
                    can->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
                    can->Advance(bit_width);
                }
            }
        }
        
        // Simple CRC calculation (15-bit)
        // NOTE: This is not the actual CAN CRC algorithm, just a placeholder
        U16 crc = 0x5555; // Placeholder CRC value
        for (int i = 14; i >= 0; i--) {
            bool bit_value = (crc >> i) & 0x01;
            can->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
            can->Advance(bit_width);
        }
        
        // CRC Delimiter - recessive bit (HIGH)
        can->TransitionIfNeeded(BIT_HIGH);
        can->Advance(bit_width);
        
        // ACK Slot - recessive bit (HIGH) sent by transmitter, should be overwritten
        // by a dominant bit (LOW) by at least one receiver
        can->TransitionIfNeeded(BIT_LOW); // Simulating an ACK from a receiver
        can->Advance(bit_width);
        
        // ACK Delimiter - recessive bit (HIGH)
        can->TransitionIfNeeded(BIT_HIGH);
        can->Advance(bit_width);
        
        // End of Frame (EOF) - 7 recessive bits (HIGH)
        for (int i = 0; i < 7; i++) {
            can->TransitionIfNeeded(BIT_HIGH);
            can->Advance(bit_width);
        }
        
        // Inter-Frame Space (IFS) - 3 recessive bits (HIGH)
        for (int i = 0; i < 3; i++) {
            can->TransitionIfNeeded(BIT_HIGH);
            can->Advance(bit_width);
        }
        
        // Final idle time
        can->Advance(bit_width * 10);
        
        return can;
    }, R"pbdoc(
        Add a CAN frame simulation.
        
        Creates a complete CAN frame including start bit, identifier, control, data, CRC, and EOF.
        
        Args:
            can_channel (Channel): Channel for CAN signal
            identifier (U32): CAN identifier (11-bit standard or 29-bit extended)
            extended_id (bool): Whether to use extended identifier (29-bit)
            data (list[U8]): Data bytes to include in frame
            remote_frame (bool, optional): Whether this is a remote frame request. Defaults to False.
            bit_rate (U32): CAN bit rate in bits per second
            sample_rate (U32): Sample rate in Hz
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created CAN channel descriptor
            
        Note:
            The returned pointer is owned by the group and should not be deleted.
            This simulation includes a simplified CRC calculation for demonstration purposes.
    )pbdoc", 
    py::arg("can_channel"), 
    py::arg("identifier"), 
    py::arg("extended_id"), 
    py::arg("data"), 
    py::arg("remote_frame") = false,
    py::arg("bit_rate"), 
    py::arg("sample_rate"),
    py::return_value_policy::reference);

    // 1-Wire simulation helper
    sim_group.def("add_1wire_simulation", [](SimulationChannelDescriptorGroup &self,
                                           Channel &onewire_channel,
                                           const std::vector<U8> &data,
                                           bool include_reset,
                                           bool standard_speed,
                                           U32 sample_rate) {
        // 1-Wire timing constants (in microseconds)
        // Standard speed
        const float RESET_LOW_TIME_STD = 480.0f;  // Reset pulse low time
        const float RESET_HIGH_TIME_STD = 480.0f; // Reset pulse high time
        const float PRESENCE_LOW_TIME_STD = 240.0f; // Presence pulse low time
        const float PRESENCE_HIGH_TIME_STD = 240.0f; // Time after presence pulse
        const float WRITE_1_LOW_TIME_STD = 6.0f;   // Write 1 low time
        const float WRITE_1_HIGH_TIME_STD = 64.0f; // Write 1 high time
        const float WRITE_0_LOW_TIME_STD = 60.0f;  // Write 0 low time
        const float WRITE_0_HIGH_TIME_STD = 10.0f; // Write 0 high time
        const float READ_LOW_TIME_STD = 6.0f;     // Read time slot low time
        const float READ_HIGH_TIME_STD = 64.0f;   // Read time slot high time
        const float RECOVERY_TIME_STD = 10.0f;    // Recovery time between slots
        
        // Overdrive speed (10x faster)
        const float RESET_LOW_TIME_OD = 48.0f;
        const float RESET_HIGH_TIME_OD = 48.0f;
        const float PRESENCE_LOW_TIME_OD = 24.0f;
        const float PRESENCE_HIGH_TIME_OD = 24.0f;
        const float WRITE_1_LOW_TIME_OD = 1.0f;
        const float WRITE_1_HIGH_TIME_OD = 7.5f;
        const float WRITE_0_LOW_TIME_OD = 7.5f;
        const float WRITE_0_HIGH_TIME_OD = 1.0f;
        const float READ_LOW_TIME_OD = 1.0f;
        const float READ_HIGH_TIME_OD = 7.5f;
        const float RECOVERY_TIME_OD = 2.0f;
        
        // Select timing based on speed parameter
        const float RESET_LOW_TIME = standard_speed ? RESET_LOW_TIME_STD : RESET_LOW_TIME_OD;
        const float RESET_HIGH_TIME = standard_speed ? RESET_HIGH_TIME_STD : RESET_HIGH_TIME_OD;
        const float PRESENCE_LOW_TIME = standard_speed ? PRESENCE_LOW_TIME_STD : PRESENCE_LOW_TIME_OD;
        const float PRESENCE_HIGH_TIME = standard_speed ? PRESENCE_HIGH_TIME_STD : PRESENCE_HIGH_TIME_OD;
        const float WRITE_1_LOW_TIME = standard_speed ? WRITE_1_LOW_TIME_STD : WRITE_1_LOW_TIME_OD;
        const float WRITE_1_HIGH_TIME = standard_speed ? WRITE_1_HIGH_TIME_STD : WRITE_1_HIGH_TIME_OD;
        const float WRITE_0_LOW_TIME = standard_speed ? WRITE_0_LOW_TIME_STD : WRITE_0_LOW_TIME_OD;
        const float WRITE_0_HIGH_TIME = standard_speed ? WRITE_0_HIGH_TIME_STD : WRITE_0_HIGH_TIME_OD;
        const float READ_LOW_TIME = standard_speed ? READ_LOW_TIME_STD : READ_LOW_TIME_OD;
        const float READ_HIGH_TIME = standard_speed ? READ_HIGH_TIME_STD : READ_HIGH_TIME_OD;
        const float RECOVERY_TIME = standard_speed ? RECOVERY_TIME_STD : RECOVERY_TIME_OD;
        
        // Convert microsecond times to sample counts
        const U32 samples_per_us = sample_rate / 1000000;
        const U32 reset_low_samples = static_cast<U32>(RESET_LOW_TIME * samples_per_us);
        const U32 reset_high_samples = static_cast<U32>(RESET_HIGH_TIME * samples_per_us);
        const U32 presence_low_samples = static_cast<U32>(PRESENCE_LOW_TIME * samples_per_us);
        const U32 presence_high_samples = static_cast<U32>(PRESENCE_HIGH_TIME * samples_per_us);
        const U32 write_1_low_samples = static_cast<U32>(WRITE_1_LOW_TIME * samples_per_us);
        const U32 write_1_high_samples = static_cast<U32>(WRITE_1_HIGH_TIME * samples_per_us);
        const U32 write_0_low_samples = static_cast<U32>(WRITE_0_LOW_TIME * samples_per_us);
        const U32 write_0_high_samples = static_cast<U32>(WRITE_0_HIGH_TIME * samples_per_us);
        const U32 read_low_samples = static_cast<U32>(READ_LOW_TIME * samples_per_us);
        const U32 read_high_samples = static_cast<U32>(READ_HIGH_TIME * samples_per_us);
        const U32 recovery_samples = static_cast<U32>(RECOVERY_TIME * samples_per_us);
        
        // Add 1-Wire channel to group
        SimulationChannelDescriptor *onewire = self.Add(onewire_channel, sample_rate, BIT_HIGH); // 1-Wire idle is HIGH (pulled up)
        
        // Start with some idle time
        onewire->Advance(reset_high_samples * 2);
        
        // Reset and Presence pulse (if requested)
        if (include_reset) {
            // Reset pulse (master pulls low)
            onewire->TransitionIfNeeded(BIT_LOW);
            onewire->Advance(reset_low_samples);
            
            // Release (pull-up resistor pulls high)
            onewire->TransitionIfNeeded(BIT_HIGH);
            onewire->Advance(reset_high_samples / 4); // Delay before presence pulse
            
            // Presence pulse (device pulls low)
            onewire->TransitionIfNeeded(BIT_LOW);
            onewire->Advance(presence_low_samples);
            
            // Release (pull-up resistor pulls high)
            onewire->TransitionIfNeeded(BIT_HIGH);
            onewire->Advance(presence_high_samples);
        }
        
        // Send each byte
        for (U8 byte : data) {
            // Send LSB first
            for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
                bool bit_value = (byte >> bit_idx) & 0x01;
                
                if (bit_value) {
                    // Write 1 bit
                    // Master pulls low briefly
                    onewire->TransitionIfNeeded(BIT_LOW);
                    onewire->Advance(write_1_low_samples);
                    
                    // Release (pull-up resistor pulls high)
                    onewire->TransitionIfNeeded(BIT_HIGH);
                    onewire->Advance(write_1_high_samples);
                } else {
                    // Write 0 bit
                    // Master pulls low for longer time
                    onewire->TransitionIfNeeded(BIT_LOW);
                    onewire->Advance(write_0_low_samples);
                    
                    // Release (pull-up resistor pulls high)
                    onewire->TransitionIfNeeded(BIT_HIGH);
                    onewire->Advance(write_0_high_samples);
                }
                
                // Recovery time between bits
                onewire->Advance(recovery_samples);
            }
        }
        
        // Final idle time
        onewire->TransitionIfNeeded(BIT_HIGH);
        onewire->Advance(reset_high_samples * 2);
        
        return onewire;
    }, R"pbdoc(
        Add a 1-Wire transmission simulation.
        
        Creates a complete 1-Wire transaction including optional reset/presence pulse and data bytes.
        
        Args:
            onewire_channel (Channel): Channel for 1-Wire signal
            data (list[U8]): Data bytes to transmit
            include_reset (bool, optional): Whether to include reset/presence pulse. Defaults to True.
            standard_speed (bool, optional): Whether to use standard speed (True) or overdrive (False). Defaults to True.
            sample_rate (U32): Sample rate in Hz
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created 1-Wire channel descriptor
            
        Note:
            The returned pointer is owned by the group and should not be deleted.
            1-Wire data is transmitted LSB first.
    )pbdoc", 
    py::arg("onewire_channel"), 
    py::arg("data"), 
    py::arg("include_reset") = true, 
    py::arg("standard_speed") = true, 
    py::arg("sample_rate"),
    py::return_value_policy::reference);

    // PWM simulation helper
    sim_group.def("add_pwm_simulation", [](SimulationChannelDescriptorGroup &self,
                                         Channel &pwm_channel,
                                         U32 frequency_hz,
                                         double duty_cycle_percent,
                                         U32 count,
                                         bool start_high,
                                         U32 sample_rate) {
        // Validate inputs
        if (duty_cycle_percent < 0) duty_cycle_percent = 0;
        if (duty_cycle_percent > 100) duty_cycle_percent = 100;
        
        // Calculate period and duty cycle in samples
        U32 period_samples = sample_rate / frequency_hz;
        U32 high_time_samples = static_cast<U32>(period_samples * duty_cycle_percent / 100.0);
        U32 low_time_samples = period_samples - high_time_samples;
        
        // Add PWM channel to group
        SimulationChannelDescriptor *pwm = self.Add(pwm_channel, sample_rate, start_high ? BIT_HIGH : BIT_LOW);
        
        // Generate cycles
        for (U32 i = 0; i < count; i++) {
            if (start_high) {
                // Start with HIGH
                pwm->TransitionIfNeeded(BIT_HIGH);
                pwm->Advance(high_time_samples);
                
                // Then LOW
                pwm->TransitionIfNeeded(BIT_LOW);
                pwm->Advance(low_time_samples);
            } else {
                // Start with LOW
                pwm->TransitionIfNeeded(BIT_LOW);
                pwm->Advance(low_time_samples);
                
                // Then HIGH
                pwm->TransitionIfNeeded(BIT_HIGH);
                pwm->Advance(high_time_samples);
            }
        }
        
        return pwm;
    }, R"pbdoc(
        Add a PWM signal simulation.
        
        Creates a pulse width modulation signal with specified frequency and duty cycle.
        
        Args:
            pwm_channel (Channel): Channel for PWM signal
            frequency_hz (U32): PWM frequency in Hz
            duty_cycle_percent (double): Duty cycle (0-100%)
            count (U32): Number of PWM cycles to generate
            start_high (bool, optional): Whether to start with high pulse. Defaults to True.
            sample_rate (U32): Sample rate in Hz
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created PWM channel descriptor
            
        Note:
            The returned pointer is owned by the group and should not be deleted.
    )pbdoc", 
    py::arg("pwm_channel"), 
    py::arg("frequency_hz"), 
    py::arg("duty_cycle_percent"), 
    py::arg("count"), 
    py::arg("start_high") = true, 
    py::arg("sample_rate"),
    py::return_value_policy::reference);

    // Manchester encoding simulation
    sim_group.def("add_manchester_simulation", [](SimulationChannelDescriptorGroup &self,
                                               Channel &manchester_channel,
                                               const std::vector<U8> &data,
                                               U32 bit_rate,
                                               bool standard_manchester,
                                               bool msb_first,
                                               U32 sample_rate) {
        // Calculate samples per bit
        U32 half_bit_samples = sample_rate / (bit_rate * 2); // Half bit period (transition occurs in middle)
        
        // Add Manchester channel to group
        SimulationChannelDescriptor *manchester = self.Add(manchester_channel, sample_rate, BIT_HIGH);
        
        // Start with some idle time
        manchester->Advance(half_bit_samples * 4);
        
        // Process each byte
        for (U8 byte : data) {
            // Process each bit
            for (int bit_idx = msb_first ? 7 : 0; 
                 msb_first ? bit_idx >= 0 : bit_idx < 8; 
                 msb_first ? bit_idx-- : bit_idx++) {
                
                bool bit_value = (byte >> bit_idx) & 0x01;
                
                // In standard IEEE 802.3 Manchester:
                // 0 = high-to-low transition in middle of bit
                // 1 = low-to-high transition in middle of bit
                //
                // In Thomas/G.E. Manchester:
                // 0 = low-to-high transition in middle of bit
                // 1 = high-to-low transition in middle of bit
                
                if (standard_manchester) {
                    // IEEE 802.3 Manchester
                    if (bit_value) {
                        // Bit 1: low->high transition
                        manchester->TransitionIfNeeded(BIT_LOW);
                        manchester->Advance(half_bit_samples);
                        manchester->TransitionIfNeeded(BIT_HIGH);
                        manchester->Advance(half_bit_samples);
                    } else {
                        // Bit 0: high->low transition
                        manchester->TransitionIfNeeded(BIT_HIGH);
                        manchester->Advance(half_bit_samples);
                        manchester->TransitionIfNeeded(BIT_LOW);
                        manchester->Advance(half_bit_samples);
                    }
                } else {
                    // Thomas/G.E. Manchester (inverted)
                    if (bit_value) {
                        // Bit 1: high->low transition
                        manchester->TransitionIfNeeded(BIT_HIGH);
                        manchester->Advance(half_bit_samples);
                        manchester->TransitionIfNeeded(BIT_LOW);
                        manchester->Advance(half_bit_samples);
                    } else {
                        // Bit 0: low->high transition
                        manchester->TransitionIfNeeded(BIT_LOW);
                        manchester->Advance(half_bit_samples);
                        manchester->TransitionIfNeeded(BIT_HIGH);
                        manchester->Advance(half_bit_samples);
                    }
                }
            }
        }
        
        // Final idle time
        manchester->TransitionIfNeeded(BIT_HIGH);
        manchester->Advance(half_bit_samples * 4);
        
        return manchester;
    }, R"pbdoc(
        Add a Manchester-encoded data simulation.
        
        Creates Manchester-encoded data with specified bit rate and encoding standard.
        
        Args:
            manchester_channel (Channel): Channel for Manchester-encoded signal
            data (list[U8]): Data bytes to encode
            bit_rate (U32): Bit rate in bits per second
            standard_manchester (bool, optional): Whether to use IEEE 802.3 (True) or Thomas/G.E. (False) encoding.
                                                 Defaults to True.
            msb_first (bool, optional): Whether to transmit most significant bit first. Defaults to True.
            sample_rate (U32): Sample rate in Hz
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created Manchester channel descriptor
            
        Note:
            The returned pointer is owned by the group and should not be deleted.
            
            IEEE 802.3 Manchester:
            - 0 = high-to-low transition in middle of bit
            - 1 = low-to-high transition in middle of bit
            
            Thomas/G.E. Manchester:
            - 0 = low-to-high transition in middle of bit
            - 1 = high-to-low transition in middle of bit
    )pbdoc", 
    py::arg("manchester_channel"), 
    py::arg("data"), 
    py::arg("bit_rate"), 
    py::arg("standard_manchester") = true, 
    py::arg("msb_first") = true, 
    py::arg("sample_rate"),
    py::return_value_policy::reference);

    // Modbus RTU simulation helper
    sim_group.def("add_modbus_rtu_simulation", [](SimulationChannelDescriptorGroup &self,
                                                Channel &tx_channel,
                                                Channel &rx_channel,
                                                U8 slave_address,
                                                U8 function_code,
                                                const std::vector<U8> &data,
                                                U32 baud_rate,
                                                U32 sample_rate) {
        // Calculate bit width based on baud rate and sample rate
        U32 bit_width = sample_rate / baud_rate;
        
        // Add Modbus channels to group
        SimulationChannelDescriptor *tx = self.Add(tx_channel, sample_rate, BIT_HIGH);
        SimulationChannelDescriptor *rx = self.Add(rx_channel, sample_rate, BIT_HIGH);
        
        // Start with some idle time
        self.AdvanceAll(bit_width * 10);
        
        // Create request frame
        std::vector<U8> request_frame;
        request_frame.push_back(slave_address);
        request_frame.push_back(function_code);
        
        // Add data to request
        for (U8 byte : data) {
            request_frame.push_back(byte);
        }
        
        // Calculate CRC-16 (Modbus RTU uses CRC-16)
        U16 crc = 0xFFFF;
        for (U8 byte : request_frame) {
            crc ^= byte;
            for (int i = 0; i < 8; i++) {
                if (crc & 0x0001) {
                    crc >>= 1;
                    crc ^= 0xA001; // Polynomial 0x8005 with reversed bits
                } else {
                    crc >>= 1;
                }
            }
        }
        
        // Add CRC to request (low byte first, then high byte)
        request_frame.push_back(crc & 0xFF);
        request_frame.push_back((crc >> 8) & 0xFF);
        
        // Send request frame (master to slave)
        for (U8 byte : request_frame) {
            // Start bit (always LOW)
            tx->TransitionIfNeeded(BIT_LOW);
            tx->Advance(bit_width);
            
            // Data bits (LSB first)
            for (int i = 0; i < 8; i++) {
                bool bit_value = (byte >> i) & 0x01;
                tx->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
                tx->Advance(bit_width);
            }
            
            // Stop bit(s) (always HIGH, using 1 stop bit)
            tx->TransitionIfNeeded(BIT_HIGH);
            tx->Advance(bit_width);
            
            // Inter-byte gap (at least 3.5 character times)
            tx->Advance(bit_width * 4);
        }
        
        // Gap between request and response (slave processing time)
        self.AdvanceAll(bit_width * 20);
        
        // Create response frame (simple echo response)
        std::vector<U8> response_frame;
        response_frame.push_back(slave_address);
        response_frame.push_back(function_code);
        
        // Add echo data to response (or modify based on function code)
        for (U8 byte : data) {
            response_frame.push_back(byte);
        }
        
        // Calculate CRC-16 for response
        crc = 0xFFFF;
        for (U8 byte : response_frame) {
            crc ^= byte;
            for (int i = 0; i < 8; i++) {
                if (crc & 0x0001) {
                    crc >>= 1;
                    crc ^= 0xA001;
                } else {
                    crc >>= 1;
                }
            }
        }
        
        // Add CRC to response (low byte first, then high byte)
        response_frame.push_back(crc & 0xFF);
        response_frame.push_back((crc >> 8) & 0xFF);
        
        // Send response frame (slave to master)
        for (U8 byte : response_frame) {
            // Start bit (always LOW)
            rx->TransitionIfNeeded(BIT_LOW);
            rx->Advance(bit_width);
            
            // Data bits (LSB first)
            for (int i = 0; i < 8; i++) {
                bool bit_value = (byte >> i) & 0x01;
                rx->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
                rx->Advance(bit_width);
            }
            
            // Stop bit(s) (always HIGH, using 1 stop bit)
            rx->TransitionIfNeeded(BIT_HIGH);
            rx->Advance(bit_width);
            
            // Inter-byte gap (at least 3.5 character times)
            rx->Advance(bit_width * 4);
        }
        
        // Final idle time
        self.AdvanceAll(bit_width * 10);
        
        return std::make_tuple(tx, rx);
    }, R"pbdoc(
        Add a Modbus RTU transaction simulation.
        
        Creates a complete Modbus RTU request and response transaction.
        
        Args:
            tx_channel (Channel): Channel for master transmission (request)
            rx_channel (Channel): Channel for slave transmission (response)
            slave_address (U8): Modbus slave address
            function_code (U8): Modbus function code
            data (list[U8]): Data bytes for the request
            baud_rate (U32): UART baud rate in bits per second
            sample_rate (U32): Sample rate in Hz
            
        Returns:
            tuple: (tx, rx) pointers to the created channel descriptors
            
        Note:
            The returned pointers are owned by the group and should not be deleted.
            This simulation creates a simple echo response. For more complex Modbus
            interactions, use the lower-level UART simulation functions.
    )pbdoc", 
    py::arg("tx_channel"), 
    py::arg("rx_channel"), 
    py::arg("slave_address"), 
    py::arg("function_code"), 
    py::arg("data"), 
    py::arg("baud_rate"), 
    py::arg("sample_rate"),
    py::return_value_policy::reference);

    // LIN simulation helper
    sim_group.def("add_lin_simulation", [](SimulationChannelDescriptorGroup &self,
                                         Channel &lin_channel,
                                         U8 identifier,
                                         const std::vector<U8> &data,
                                         U32 baud_rate,
                                         U32 sample_rate) {
        // Calculate bit width based on baud rate and sample rate
        U32 bit_width = sample_rate / baud_rate;
        
        // Add LIN channel to group
        SimulationChannelDescriptor *lin = self.Add(lin_channel, sample_rate, BIT_HIGH); // LIN idle is HIGH
        
        // Start with some idle time
        lin->Advance(bit_width * 10);
        
        // Break field (13 dominant bits + 1 recessive bit)
        lin->TransitionIfNeeded(BIT_LOW);
        lin->Advance(bit_width * 13);
        lin->TransitionIfNeeded(BIT_HIGH);
        lin->Advance(bit_width);
        
        // Sync field (0x55)
        U8 sync = 0x55;
        for (int i = 0; i < 8; i++) {
            bool bit_value = (sync >> i) & 0x01;
            lin->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
            lin->Advance(bit_width);
        }
        
        // Protected Identifier Field (PID)
        // Calculate parity bits P0 and P1
        U8 p0 = ((identifier >> 0) & 1) ^ ((identifier >> 1) & 1) ^ ((identifier >> 2) & 1) ^ ((identifier >> 4) & 1);
        U8 p1 = ~(((identifier >> 1) & 1) ^ ((identifier >> 3) & 1) ^ ((identifier >> 4) & 1) ^ ((identifier >> 5) & 1)) & 1;
        U8 pid = identifier | (p0 << 6) | (p1 << 7);
        
        // Send PID byte LSB first (LIN is LSB first)
        for (int i = 0; i < 8; i++) {
            bool bit_value = (pid >> i) & 0x01;
            lin->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
            lin->Advance(bit_width);
        }
        
        // Data Field
        for (size_t byte_idx = 0; byte_idx < data.size(); byte_idx++) {
            // Send data bytes LSB first
            for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
                bool bit_value = (data[byte_idx] >> bit_idx) & 0x01;
                lin->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
                lin->Advance(bit_width);
            }
        }
        
        // Calculate checksum (classic method - simple sum of data bytes)
        U8 checksum = 0;
        for (U8 byte : data) {
            checksum += byte;
        }
        checksum = ~checksum; // Inverted sum
        
        // Send checksum byte LSB first
        for (int i = 0; i < 8; i++) {
            bool bit_value = (checksum >> i) & 0x01;
            lin->TransitionIfNeeded(bit_value ? BIT_HIGH : BIT_LOW);
            lin->Advance(bit_width);
        }
        
        // Final idle time
        lin->TransitionIfNeeded(BIT_HIGH);
        lin->Advance(bit_width * 10);
        
        return lin;
    }, R"pbdoc(
        Add a LIN frame simulation.
        
        Creates a complete LIN frame including break field, sync field, identifier, data, and checksum.
        
        Args:
            lin_channel (Channel): Channel for LIN signal
            identifier (U8): LIN identifier (6-bit)
            data (list[U8]): Data bytes to include in frame
            baud_rate (U32): LIN baud rate in bits per second
            sample_rate (U32): Sample rate in Hz
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created LIN channel descriptor
            
        Note:
            The returned pointer is owned by the group and should not be deleted.
            This simulation uses the classic checksum calculation method.
    )pbdoc", 
    py::arg("lin_channel"), 
    py::arg("identifier"), 
    py::arg("data"), 
    py::arg("baud_rate"), 
    py::arg("sample_rate"),
    py::return_value_policy::reference);
}