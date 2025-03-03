// bind_channel_data.cpp
//
// Python bindings for the AnalyzerChannelData class

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AnalyzerChannelData.h"
#include "LogicPublicTypes.h"

namespace py = pybind11;

void init_channel_data(py::module_ &m) {
    py::class_<AnalyzerChannelData> channel_data(m, "AnalyzerChannelData", R"pbdoc(
        Provides access to sampled data from a logic analyzer channel.
        
        This class allows analyzer implementations to read bit values and navigate
        through the captured data. It provides methods for moving to specific samples,
        finding edges (transitions between high and low), and examining bit states.
        
        The AnalyzerChannelData object is usually obtained from the Analyzer's
        `get_channel_data()` method and should not be created directly.
        
        Example:
            # Get channel data for data_channel
            data = analyzer.get_channel_data(data_channel)
            
            # Check current bit state
            if data.get_bit_state() == BitState.HIGH:
                # Process high state
                pass
                
            # Move to next edge
            data.advance_to_next_edge()
    )pbdoc");

    // --------------------------------------------------------------------------
    // State methods
    // --------------------------------------------------------------------------
    
    channel_data.def("get_sample_number", &AnalyzerChannelData::GetSampleNumber, R"pbdoc(
        Get the current sample position.
        
        Returns:
            int: Current sample number (64-bit unsigned integer)
            
        Usage note:
            This returns the absolute sample position in the captured data.
            Sample numbering starts at 0.
    )pbdoc");

    channel_data.def("get_bit_state", &AnalyzerChannelData::GetBitState, R"pbdoc(
        Get the bit state at the current sample position.
        
        Returns:
            BitState: Current bit state (BitState.HIGH or BitState.LOW)
            
        Example:
            if channel.get_bit_state() == BitState.HIGH:
                # Process high state
                pass
    )pbdoc");

    // --------------------------------------------------------------------------
    // Basic navigation methods
    // --------------------------------------------------------------------------
    
    channel_data.def("advance", &AnalyzerChannelData::Advance, R"pbdoc(
        Move forward by a specified number of samples.
        
        Args:
            num_samples (int): Number of samples to advance
            
        Returns:
            int: Number of bit state transitions that occurred during the move
            
        Raises:
            RuntimeError: If attempting to advance beyond the end of captured data
            
        Example:
            # Advance 10 samples and count transitions
            transitions = channel.advance(10)
            print(f"Encountered {transitions} transitions")
    )pbdoc", py::arg("num_samples"));

    channel_data.def("advance_to_abs_position", &AnalyzerChannelData::AdvanceToAbsPosition, R"pbdoc(
        Move to an absolute sample position.
        
        Args:
            sample_number (int): Sample number to move to (64-bit unsigned integer)
            
        Returns:
            int: Number of bit state transitions that occurred during the move
            
        Raises:
            RuntimeError: If the specified position is beyond the end of captured data
            
        Note:
            This can move both forward and backward in the capture.
    )pbdoc", py::arg("sample_number"));

    channel_data.def("advance_to_next_edge", &AnalyzerChannelData::AdvanceToNextEdge, R"pbdoc(
        Move forward until the bit state changes from its current state.
        
        This method advances to the next transition edge (from HIGH to LOW or 
        from LOW to HIGH).
        
        Raises:
            RuntimeError: If no more edges exist in the captured data
            
        Example:
            # Move to the next edge
            channel.advance_to_next_edge()
            
            # Get the bit state after the edge
            new_state = channel.get_bit_state()
    )pbdoc");

    // --------------------------------------------------------------------------
    // Edge detection methods
    // --------------------------------------------------------------------------
    
    channel_data.def("get_sample_of_next_edge", &AnalyzerChannelData::GetSampleOfNextEdge, R"pbdoc(
        Get the sample number of the next transition edge without moving.
        
        Returns:
            int: Sample number of the next bit state transition (64-bit unsigned integer)
            
        Raises:
            RuntimeError: If no more edges exist in the captured data
            
        Usage note:
            This method does not change the current position.
    )pbdoc");

    channel_data.def("would_advancing_cause_transition", &AnalyzerChannelData::WouldAdvancingCauseTransition, R"pbdoc(
        Check if advancing by a number of samples would cross a transition.
        
        Args:
            num_samples (int): Number of samples to hypothetically advance
            
        Returns:
            bool: True if a transition would be encountered
            
        Example:
            # Check if there's a transition in the next 100 samples
            if channel.would_advancing_cause_transition(100):
                # Handle the case with a transition
                pass
            else:
                # Handle the case without transitions
                pass
    )pbdoc", py::arg("num_samples"));

    channel_data.def("would_advancing_to_abs_position_cause_transition", 
                    &AnalyzerChannelData::WouldAdvancingToAbsPositionCauseTransition, R"pbdoc(
        Check if moving to an absolute position would cross a transition.
        
        Args:
            sample_number (int): Sample number to hypothetically move to (64-bit unsigned integer)
            
        Returns:
            bool: True if a transition would be encountered
            
        Note:
            This can check both forward and backward in the capture.
    )pbdoc", py::arg("sample_number"));

    // --------------------------------------------------------------------------
    // Pulse width tracking
    // --------------------------------------------------------------------------
    
    channel_data.def("track_minimum_pulse_width", &AnalyzerChannelData::TrackMinimumPulseWidth, R"pbdoc(
        Enable minimum pulse width tracking.
        
        Once enabled, the analyzer will track the minimum pulse width (time between
        transitions) encountered during navigation. This is useful for protocols
        with timing-dependent features, such as auto-baud detection in serial protocols.
        
        Usage note:
            This tracking is disabled by default and must be explicitly enabled.
            After enabling, all subsequent navigation methods will update the
            minimum pulse width value.
            
        Example:
            # Enable pulse width tracking for auto-baud detection
            serial_channel.track_minimum_pulse_width()
            
            # Navigate through the data to find the minimum pulse
            while serial_channel.get_sample_number() < end_sample:
                serial_channel.advance_to_next_edge()
    )pbdoc");

    channel_data.def("get_minimum_pulse_width_so_far", &AnalyzerChannelData::GetMinimumPulseWidthSoFar, R"pbdoc(
        Get the minimum pulse width detected so far.
        
        This method returns the smallest number of samples between any two transitions
        encountered since tracking was enabled.
        
        Returns:
            int: Minimum pulse width in samples (64-bit unsigned integer)
            
        Usage note:
            This will return 0 if no transitions have been encountered since
            enabling tracking or if tracking has not been enabled.
            
        Example:
            # After navigating through data with tracking enabled
            min_pulse = channel.get_minimum_pulse_width_so_far()
            
            # Calculate baud rate based on minimum pulse (for serial analyzers)
            if min_pulse > 0:
                # For serial, the minimum pulse is typically one bit time
                baud_rate = sample_rate_hz / min_pulse
    )pbdoc");

    // --------------------------------------------------------------------------
    // Additional methods
    // --------------------------------------------------------------------------
    
    channel_data.def("do_more_transitions_exist_in_current_data", 
                    &AnalyzerChannelData::DoMoreTransitionsExistInCurrentData, R"pbdoc(
        Check if more transitions exist in the current captured data.
        
        This method is useful when working with multiple channels to determine
        if a channel might not change state again in the captured data.
        
        Returns:
            bool: True if more transitions exist
            
        Example:
            # Check if we should continue processing this channel
            if not channel.do_more_transitions_exist_in_current_data():
                # No more transitions, handle end of data
                break
    )pbdoc");

    // --------------------------------------------------------------------------
    // Python-specific convenience methods
    // --------------------------------------------------------------------------
    
    channel_data.def("move_to_next_edge", [](AnalyzerChannelData &self) {
        U64 next_edge = self.GetSampleOfNextEdge();
        self.AdvanceToAbsPosition(next_edge);
        return next_edge;
    }, R"pbdoc(
        Move to the next edge and return its sample number.
        
        Returns:
            int: Sample number of the edge (64-bit unsigned integer)
            
        Raises:
            RuntimeError: If no more edges exist in the captured data
            
        Example:
            # Move to next edge and get its position
            edge_position = channel.move_to_next_edge()
    )pbdoc");

    channel_data.def("advance_by_time", [](AnalyzerChannelData &self, double time_s, U32 sample_rate_hz) {
        if (time_s < 0) {
            throw py::value_error("Time value must be non-negative");
        }
        
        U32 num_samples = static_cast<U32>(time_s * sample_rate_hz);
        return self.Advance(num_samples);
    }, R"pbdoc(
        Advance by a specified time in seconds.
        
        Args:
            time_s (float): Time to advance in seconds
            sample_rate_hz (int): Sample rate in Hz
            
        Returns:
            int: Number of transitions that occurred
            
        Raises:
            ValueError: If time_s is negative
            RuntimeError: If attempting to advance beyond the end of captured data
            
        Example:
            # Advance by 1 millisecond at 20MHz sample rate
            channel.advance_by_time(0.001, 20000000)
    )pbdoc", py::arg("time_s"), py::arg("sample_rate_hz"));

    channel_data.def("wait_for_state", [](AnalyzerChannelData &self, BitState state) {
        U64 start_sample = self.GetSampleNumber();
        BitState current_state = self.GetBitState();
        
        // If we're already at the desired state, return current position
        if (current_state == state) {
            return start_sample;
        }
        
        // Otherwise advance to next edge and check
        if (self.DoMoreTransitionsExistInCurrentData()) {
            self.AdvanceToNextEdge();
            current_state = self.GetBitState();
            
            // If we're now at the desired state, return new position
            if (current_state == state) {
                return self.GetSampleNumber();
            }
            
            // Otherwise, we need to advance again if possible
            if (self.DoMoreTransitionsExistInCurrentData()) {
                self.AdvanceToNextEdge();
                return self.GetSampleNumber();
            }
        }
        
        // If we get here, we couldn't find the state
        throw py::value_error("Desired bit state not found in remaining data");
    }, R"pbdoc(
        Advance until a specific bit state is reached.
        
        Args:
            state (BitState): BitState to wait for (BitState.HIGH or BitState.LOW)
            
        Returns:
            int: Sample number where the state was found (64-bit unsigned integer)
            
        Raises:
            ValueError: If the desired state cannot be found in the remaining data
            
        Example:
            # Wait for a HIGH state
            high_pos = channel.wait_for_state(BitState.HIGH)
    )pbdoc", py::arg("state"));

    channel_data.def("find_pattern", [](AnalyzerChannelData &self, const std::vector<BitState> &pattern, 
                                       U64 max_samples_to_search) {
        if (pattern.empty()) {
            throw py::value_error("Pattern cannot be empty");
        }
        
        // Save original position
        U64 original_sample = self.GetSampleNumber();
        BitState original_state = self.GetBitState();
        
        // Track current position in search
        U64 current_sample = original_sample;
        size_t pattern_index = 0;
        BitState current_state = original_state;
        
        U64 max_sample = original_sample + max_samples_to_search;
        U64 pattern_start_sample = 0;
        bool pattern_found = false;
        
        try {
            while (current_sample < max_sample && self.DoMoreTransitionsExistInCurrentData()) {
                // Check if current bit matches pattern at current index
                if (current_state == pattern[pattern_index]) {
                    // If this is the first match, mark it as potential start
                    if (pattern_index == 0) {
                        pattern_start_sample = current_sample;
                    }
                    
                    pattern_index++;
                    
                    // If we've matched the entire pattern
                    if (pattern_index >= pattern.size()) {
                        pattern_found = true;
                        break;
                    }
                } else {
                    // Mismatch, reset pattern search
                    pattern_index = 0;
                    // If the current bit matches the first pattern bit, start a new match
                    if (current_state == pattern[0]) {
                        pattern_start_sample = current_sample;
                        pattern_index = 1;
                    }
                }
                
                // Advance to next sample if not at end
                if (self.DoMoreTransitionsExistInCurrentData()) {
                    if (current_sample < max_sample - 1) {
                        self.Advance(1);
                        current_sample = self.GetSampleNumber();
                        current_state = self.GetBitState();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        catch (...) {
            // Restore original position on any exception
            self.AdvanceToAbsPosition(original_sample);
            throw;
        }
        
        // Restore original position
        self.AdvanceToAbsPosition(original_sample);
        
        // Return result tuple (found, sample_number)
        if (pattern_found) {
            return py::make_tuple(true, pattern_start_sample);
        } else {
            return py::make_tuple(false, 0ULL);
        }
    }, R"pbdoc(
        Find a specific bit pattern in the channel data.
        
        This helper searches for a sequence of bit states starting from the
        current position without modifying the current position.
        
        Args:
            pattern (list): List of BitStates to search for
            max_samples_to_search (int): Maximum number of samples to look through
            
        Returns:
            tuple: (found, sample_number)
                found (bool): True if pattern was found
                sample_number (int): Starting sample of the pattern if found
                
        Raises:
            ValueError: If pattern is empty
            
        Example:
            # Define SPI MOSI pattern (START = LOW, CLOCK = HIGH)
            pattern = [BitState.LOW, BitState.HIGH]
            
            # Search for pattern in the next 1000 samples
            found, position = channel.find_pattern(pattern, 1000)
            
            if found:
                # Pattern found at position
                pass
    )pbdoc", py::arg("pattern"), py::arg("max_samples_to_search"));

    // Additional utility method for measuring time between events
    channel_data.def("measure_time_between", [](AnalyzerChannelData &self, U64 start_sample, 
                                              U64 end_sample, double sample_rate_hz) {
        if (end_sample < start_sample) {
            throw py::value_error("End sample must be greater than or equal to start sample");
        }
        
        U64 samples_between = end_sample - start_sample;
        double time_s = static_cast<double>(samples_between) / sample_rate_hz;
        return time_s;
    }, R"pbdoc(
        Measure time between two sample positions.
        
        Args:
            start_sample (int): Starting sample number
            end_sample (int): Ending sample number
            sample_rate_hz (float): Sample rate in Hz
            
        Returns:
            float: Time between samples in seconds
            
        Raises:
            ValueError: If end_sample is less than start_sample
            
        Example:
            # Measure time between two edges
            first_edge = channel.get_sample_of_next_edge()
            channel.advance_to_next_edge()
            second_edge = channel.get_sample_number()
            
            # Calculate time between edges
            pulse_width = channel.measure_time_between(first_edge, second_edge, 20000000)
            print(f"Pulse width: {pulse_width * 1000} ms")
    )pbdoc", py::arg("start_sample"), py::arg("end_sample"), py::arg("sample_rate_hz"));

    // Method to count transitions in a range
    channel_data.def("count_transitions_between", [](AnalyzerChannelData &self, U64 start_sample, U64 end_sample) {
        if (end_sample < start_sample) {
            throw py::value_error("End sample must be greater than or equal to start sample");
        }
        
        // Save original position
        U64 original_sample = self.GetSampleNumber();
        
        // Move to start position
        self.AdvanceToAbsPosition(start_sample);
        
        // Count transitions
        U32 transitions = self.AdvanceToAbsPosition(end_sample);
        
        // Restore original position
        self.AdvanceToAbsPosition(original_sample);
        
        return transitions;
    }, R"pbdoc(
        Count the number of transitions between two sample positions.
        
        This method doesn't change the current position.
        
        Args:
            start_sample (int): Starting sample number
            end_sample (int): Ending sample number
            
        Returns:
            int: Number of transitions between the two positions
            
        Raises:
            ValueError: If end_sample is less than start_sample
            
        Example:
            # Count clock transitions during a byte transmission
            transitions = channel.count_transitions_between(byte_start, byte_end)
            
            # For SPI, transitions should be 16 (8 bits × 2 transitions per bit)
            if transitions != 16:
                print("Unexpected number of clock transitions")
    )pbdoc", py::arg("start_sample"), py::arg("end_sample"));

    // Method to find the nth edge from current position
    channel_data.def("find_nth_edge", [](AnalyzerChannelData &self, U32 n) {
        if (n == 0) {
            throw py::value_error("n must be at least 1");
        }
        
        // Save original position
        U64 original_sample = self.GetSampleNumber();
        U64 edge_sample = original_sample;
        
        try {
            for (U32 i = 0; i < n; i++) {
                if (!self.DoMoreTransitionsExistInCurrentData()) {
                    // Restore original position
                    self.AdvanceToAbsPosition(original_sample);
                    throw py::value_error("Not enough edges in remaining data");
                }
                self.AdvanceToNextEdge();
                edge_sample = self.GetSampleNumber();
            }
        }
        catch (...) {
            // Restore original position on any exception
            self.AdvanceToAbsPosition(original_sample);
            throw;
        }
        
        // Restore original position
        self.AdvanceToAbsPosition(original_sample);
        
        return edge_sample;
    }, R"pbdoc(
        Find the sample number of the nth edge from the current position.
        
        This method doesn't change the current position.
        
        Args:
            n (int): Which edge to find (1 for next edge, 2 for second edge, etc.)
            
        Returns:
            int: Sample number of the nth edge
            
        Raises:
            ValueError: If n is 0 or if not enough edges exist
            
        Example:
            # Find the 8th edge (for an SPI byte with 8 clock pulses)
            byte_end = channel.find_nth_edge(8)
    )pbdoc", py::arg("n"));
}