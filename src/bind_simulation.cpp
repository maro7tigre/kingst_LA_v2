// bind_simulation.cpp
//
// Python bindings for the SimulationChannelDescriptor classes

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "SimulationChannelDescriptor.h"
#include "LogicPublicTypes.h"

namespace py = pybind11;

void init_simulation(py::module_ &m) {
    // SimulationChannelDescriptor class
    py::class_<SimulationChannelDescriptor> sim_channel(m, "SimulationChannelDescriptor", R"pbdoc(
        Represents a channel in simulation mode.
        
        SimulationChannelDescriptor is used to create simulated data for testing
        analyzers without requiring physical hardware. It allows you to generate
        bit patterns by setting transitions at specific sample points.
    )pbdoc");

    // Transition methods
    sim_channel.def("transition", &SimulationChannelDescriptor::Transition, R"pbdoc(
        Toggle the current bit state (from HIGH to LOW or LOW to HIGH).
        
        This method creates a transition at the current sample position.
    )pbdoc");

    sim_channel.def("transition_if_needed", &SimulationChannelDescriptor::TransitionIfNeeded, R"pbdoc(
        Transition to a specific bit state if not already at that state.
        
        Args:
            bit_state: Desired bit state (HIGH or LOW)
    )pbdoc", py::arg("bit_state"));

    // Navigation methods
    sim_channel.def("advance", &SimulationChannelDescriptor::Advance, R"pbdoc(
        Move forward by a specified number of samples.
        
        This advances the current sample position without changing the bit state.
        
        Args:
            num_samples_to_advance: Number of samples to move forward
    )pbdoc", py::arg("num_samples_to_advance"));

    // State methods
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

    // Configuration methods
    sim_channel.def("set_channel", &SimulationChannelDescriptor::SetChannel, R"pbdoc(
        Set the channel that this descriptor is associated with.
        
        Args:
            channel: Channel to associate with this descriptor
    )pbdoc", py::arg("channel"));

    sim_channel.def("set_sample_rate", &SimulationChannelDescriptor::SetSampleRate, R"pbdoc(
        Set the sample rate for simulation.
        
        Args:
            sample_rate_hz: Sample rate in Hz
    )pbdoc", py::arg("sample_rate_hz"));

    sim_channel.def("set_initial_bit_state", &SimulationChannelDescriptor::SetInitialBitState, R"pbdoc(
        Set the initial bit state for the channel.
        
        Args:
            initial_bit_state: Initial bit state (HIGH or LOW)
    )pbdoc", py::arg("initial_bit_state"));

    // Access methods
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

    // Add Python-specific helper methods for common simulation patterns

    // Create a pulse
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
        
        This helper creates a series of pulses (transitions from the current state
        and back) with specified width and gaps between pulses.
        
        Args:
            width: Width of each pulse in samples
            count: Number of pulses to create
            gap: Gap between pulses in samples
    )pbdoc", py::arg("width"), py::arg("count") = 1, py::arg("gap") = 0);

    // Create a specific bit pattern
    sim_channel.def("add_bit_pattern", [](SimulationChannelDescriptor &self, 
                                         const std::vector<BitState> &pattern, 
                                         U32 samples_per_bit) {
        BitState current_state = self.GetCurrentBitState();
        
        for (BitState bit : pattern) {
            // Transition if needed to match the desired bit state
            self.TransitionIfNeeded(bit);
            
            // Advance for the duration of this bit
            self.Advance(samples_per_bit);
        }
    }, R"pbdoc(
        Add a specific bit pattern.
        
        This helper creates a sequence of bits with specified duration per bit.
        
        Args:
            pattern: List of BitStates to generate
            samples_per_bit: Number of samples for each bit
    )pbdoc", py::arg("pattern"), py::arg("samples_per_bit"));

    // Create a UART byte transmission
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
        
        This helper creates a complete UART byte transmission including optional
        start bit, stop bit, and parity bit.
        
        Args:
            byte: Data byte to transmit
            bit_width: Width of each bit in samples
            with_start_bit: Include start bit
            with_stop_bit: Include stop bit
            with_parity_bit: Include parity bit
            even_parity: Use even parity (if with_parity_bit is True)
    )pbdoc", 
    py::arg("byte"), 
    py::arg("bit_width"), 
    py::arg("with_start_bit") = true, 
    py::arg("with_stop_bit") = true, 
    py::arg("with_parity_bit") = false, 
    py::arg("even_parity") = true);

    // Create an I2C byte transmission
    sim_channel.def("add_i2c_byte", [](SimulationChannelDescriptor &self, U8 byte, bool with_ack, 
                                      U32 bit_width, bool is_scl) {
        if (is_scl) {
            // For SCL, generate 9 clock cycles (8 data bits + 1 ACK bit)
            for (int i = 0; i < 9; i++) {
                // Clock starts low
                self.TransitionIfNeeded(BIT_LOW);
                self.Advance(bit_width / 2);
                
                // Clock goes high
                self.Transition();
                self.Advance(bit_width / 2);
                
                // Clock goes low again
                self.Transition();
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
        
        This helper creates an I2C byte transmission including the ACK bit.
        
        Args:
            byte: Data byte to transmit
            with_ack: Include ACK bit (pulled low)
            bit_width: Width of each bit in samples
            is_scl: True if this is the SCL (clock) channel, False for SDA (data)
    )pbdoc", 
    py::arg("byte"), 
    py::arg("with_ack") = true, 
    py::arg("bit_width") = 100, 
    py::arg("is_scl") = false);

    // SimulationChannelDescriptorGroup class
    py::class_<SimulationChannelDescriptorGroup> sim_group(m, "SimulationChannelDescriptorGroup", R"pbdoc(
        A group of simulation channels.
        
        This class manages a collection of SimulationChannelDescriptor objects
        for convenient multichannel simulation.
    )pbdoc");

    sim_group.def(py::init<>(), "Default constructor");

    sim_group.def("add", &SimulationChannelDescriptorGroup::Add, R"pbdoc(
        Add a channel to the group.
        
        Args:
            channel: Channel to add
            sample_rate: Sample rate in Hz
            initial_bit_state: Initial bit state (HIGH or LOW)
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created descriptor
            
        Note: Do not delete the returned pointer.
    )pbdoc", py::arg("channel"), py::arg("sample_rate"), py::arg("initial_bit_state"), 
    py::return_value_policy::reference);

    sim_group.def("advance_all", &SimulationChannelDescriptorGroup::AdvanceAll, R"pbdoc(
        Advance all channels by the same number of samples.
        
        This method moves all channels forward together while maintaining
        their individual bit states.
        
        Args:
            num_samples_to_advance: Number of samples to advance
    )pbdoc", py::arg("num_samples_to_advance"));

    sim_group.def("get_array", &SimulationChannelDescriptorGroup::GetArray, R"pbdoc(
        Get the array of channel descriptors.
        
        Returns:
            SimulationChannelDescriptor*: Pointer to the array of descriptors
            
        Note: Usually not needed in Python; use list access instead.
    )pbdoc", py::return_value_policy::reference);

    sim_group.def("get_count", &SimulationChannelDescriptorGroup::GetCount, R"pbdoc(
        Get the number of channels in the group.
        
        Returns:
            U32: Number of channels
    )pbdoc");

    // Python-specific helpers for channel groups

    // Add list-like access to SimulationChannelDescriptorGroup
    sim_group.def("__len__", &SimulationChannelDescriptorGroup::GetCount);
    sim_group.def("__getitem__", [](SimulationChannelDescriptorGroup &self, U32 index) {
        if (index >= self.GetCount()) {
            throw py::index_error("Index out of range");
        }
        return &(self.GetArray()[index]);
    }, py::return_value_policy::reference);

    // Helper to add common serial protocol simulation
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
            // Add byte with UART framing
            // Using the helper function defined earlier
            if (with_start_bit) {
                // Start bit is always LOW
                uart_tx->Transition(); // Transition to LOW
                uart_tx->Advance(bit_width);
            }
            
            // Data bits (LSB first for UART)
            U8 parity = 0; // For tracking parity bit value
            for (int i = 0; i < 8; i++) {
                bool bit_value = (byte >> i) & 0x01;
                parity ^= bit_value; // XOR for parity calculation
                
                BitState bit_state = bit_value ? BIT_HIGH : BIT_LOW;
                uart_tx->TransitionIfNeeded(bit_state);
                uart_tx->Advance(bit_width);
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
        
        return uart_tx;
    }, R"pbdoc(
        Add a UART transmission simulation.
        
        This helper creates a complete UART transmission for a sequence of bytes.
        
        Args:
            uart_tx_channel: Channel for UART TX
            data: List of bytes to transmit
            baud_rate: UART baud rate in bits per second
            sample_rate: Sample rate in Hz
            with_start_bit: Include start bit
            with_stop_bit: Include stop bit
            with_parity_bit: Include parity bit
            even_parity: Use even parity (if with_parity_bit is True)
            
        Returns:
            SimulationChannelDescriptor: Pointer to the created TX channel descriptor
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

    // Helper to add SPI simulation
    sim_group.def("add_spi_simulation", [](SimulationChannelDescriptorGroup &self,
                                          Channel &sck_channel,
                                          Channel &mosi_channel,
                                          Channel &miso_channel,
                                          Channel &cs_channel,
                                          const std::vector<U8> &mosi_data,
                                          const std::vector<U8> &miso_data,
                                          U32 bit_width,
                                          U32 sample_rate,
                                          bool cpol,
                                          bool cpha) {
        // Add all channels to group
        SimulationChannelDescriptor *sck = self.Add(sck_channel, sample_rate, cpol ? BIT_HIGH : BIT_LOW);
        SimulationChannelDescriptor *mosi = self.Add(mosi_channel, sample_rate, BIT_LOW);
        SimulationChannelDescriptor *miso = self.Add(miso_channel, sample_rate, BIT_LOW);
        SimulationChannelDescriptor *cs = self.Add(cs_channel, sample_rate, BIT_HIGH);  // CS is active low
        
        // Start with some idle time
        self.AdvanceAll(bit_width * 10);
        
        // Assert CS (active low)
        cs->Transition();  // Transition to LOW
        
        // Gap after CS assertion
        self.AdvanceAll(bit_width);
        
        // Determine which data array is longer
        size_t tx_size = mosi_data.size();
        size_t rx_size = miso_data.size();
        size_t max_size = (tx_size > rx_size) ? tx_size : rx_size;
        
        // Process each byte
        for (size_t byte_idx = 0; byte_idx < max_size; byte_idx++) {
            U8 mosi_byte = (byte_idx < tx_size) ? mosi_data[byte_idx] : 0;
            U8 miso_byte = (byte_idx < rx_size) ? miso_data[byte_idx] : 0;
            
            // Transfer each bit (MSB first)
            for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
                bool mosi_bit = (mosi_byte >> bit_idx) & 0x01;
                bool miso_bit = (miso_byte >> bit_idx) & 0x01;
                
                // Setup phase
                if (cpha) {
                    // CPHA=1: Data changes on trailing edge, sampled on leading edge
                    // Setup MOSI and MISO first
                    mosi->TransitionIfNeeded(mosi_bit ? BIT_HIGH : BIT_LOW);
                    miso->TransitionIfNeeded(miso_bit ? BIT_HIGH : BIT_LOW);
                    
                    // First clock transition
                    sck->Transition();
                    self.AdvanceAll(bit_width / 2);
                    
                    // Second clock transition
                    sck->Transition();
                    self.AdvanceAll(bit_width / 2);
                } else {
                    // CPHA=0: Data changes on leading edge, sampled on trailing edge
                    // Setup MOSI and MISO first
                    mosi->TransitionIfNeeded(mosi_bit ? BIT_HIGH : BIT_LOW);
                    miso->TransitionIfNeeded(miso_bit ? BIT_HIGH : BIT_LOW);
                    
                    // Half bit delay
                    self.AdvanceAll(bit_width / 2);
                    
                    // First clock transition
                    sck->Transition();
                    self.AdvanceAll(bit_width / 2);
                    
                    // Second clock transition
                    sck->Transition();
                    self.AdvanceAll(bit_width / 2);
                    
                    // Half bit delay
                    self.AdvanceAll(bit_width / 2);
                }
            }
            
            // Inter-byte gap (1 bit width)
            self.AdvanceAll(bit_width);
        }
        
        // De-assert CS (return to HIGH)
        cs->TransitionIfNeeded(BIT_HIGH);
        
        // Final idle time
        self.AdvanceAll(bit_width * 10);
        
        return std::make_tuple(sck, mosi, miso, cs);
    }, R"pbdoc(
        Add an SPI transmission simulation.
        
        This helper creates a complete SPI transmission for sequences of bytes.
        
        Args:
            sck_channel: Channel for SCK (clock)
            mosi_channel: Channel for MOSI (Master Out Slave In)
            miso_channel: Channel for MISO (Master In Slave Out)
            cs_channel: Channel for CS (Chip Select)
            mosi_data: List of bytes to transmit on MOSI
            miso_data: List of bytes to transmit on MISO
            bit_width: Width of each bit in samples
            sample_rate: Sample rate in Hz
            cpol: Clock polarity (idle state of clock)
            cpha: Clock phase (when data is sampled)
            
        Returns:
            tuple: (sck, mosi, miso, cs) pointers to the created channel descriptors
    )pbdoc", 
    py::arg("sck_channel"), 
    py::arg("mosi_channel"), 
    py::arg("miso_channel"), 
    py::arg("cs_channel"), 
    py::arg("mosi_data"), 
    py::arg("miso_data"), 
    py::arg("bit_width"), 
    py::arg("sample_rate"),
    py::arg("cpol") = false, 
    py::arg("cpha") = false,
    py::return_value_policy::reference);
}