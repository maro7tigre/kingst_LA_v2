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
    )pbdoc");

    // State methods
    channel_data.def("get_sample_number", &AnalyzerChannelData::GetSampleNumber, R"pbdoc(
        Get the current sample position.
        
        Returns:
            U64: Current sample number
    )pbdoc");

    channel_data.def("get_bit_state", &AnalyzerChannelData::GetBitState, R"pbdoc(
        Get the bit state at the current sample position.
        
        Returns:
            BitState: Current bit state (HIGH or LOW)
    )pbdoc");

    // Basic navigation methods
    channel_data.def("advance", &AnalyzerChannelData::Advance, R"pbdoc(
        Move forward by a specified number of samples.
        
        Args:
            num_samples: Number of samples to advance
            
        Returns:
            U32: Number of bit state transitions that occurred during the move
    )pbdoc", py::arg("num_samples"));

    channel_data.def("advance_to_abs_position", &AnalyzerChannelData::AdvanceToAbsPosition, R"pbdoc(
        Move to an absolute sample position.
        
        Args:
            sample_number: Sample number to move to
            
        Returns:
            U32: Number of bit state transitions that occurred during the move
    )pbdoc", py::arg("sample_number"));

    channel_data.def("advance_to_next_edge", &AnalyzerChannelData::AdvanceToNextEdge, R"pbdoc(
        Move forward until the bit state changes from its current state.
        
        This method advances to the next transition edge (from HIGH to LOW or 
        from LOW to HIGH).
    )pbdoc");

    // Edge detection methods
    channel_data.def("get_sample_of_next_edge", &AnalyzerChannelData::GetSampleOfNextEdge, R"pbdoc(
        Get the sample number of the next transition edge without moving.
        
        Returns:
            U64: Sample number of the next bit state transition
    )pbdoc");

    channel_data.def("would_advancing_cause_transition", &AnalyzerChannelData::WouldAdvancingCauseTransition, R"pbdoc(
        Check if advancing by a number of samples would cross a transition.
        
        Args:
            num_samples: Number of samples to hypothetically advance
            
        Returns:
            bool: True if a transition would be encountered
    )pbdoc", py::arg("num_samples"));

    channel_data.def("would_advancing_to_abs_position_cause_transition", 
                    &AnalyzerChannelData::WouldAdvancingToAbsPositionCauseTransition, R"pbdoc(
        Check if moving to an absolute position would cross a transition.
        
        Args:
            sample_number: Sample number to hypothetically move to
            
        Returns:
            bool: True if a transition would be encountered
    )pbdoc", py::arg("sample_number"));

    // Pulse width tracking
    channel_data.def("track_minimum_pulse_width", &AnalyzerChannelData::TrackMinimumPulseWidth, R"pbdoc(
        Enable minimum pulse width tracking.
        
        Once enabled, the analyzer will track the minimum pulse width (time between
        transitions) encountered during navigation. This is useful for protocols
        with timing-dependent features, such as auto-baud detection in serial protocols.
    )pbdoc");

    channel_data.def("get_minimum_pulse_width_so_far", &AnalyzerChannelData::GetMinimumPulseWidthSoFar, R"pbdoc(
        Get the minimum pulse width detected so far.
        
        This method returns the smallest number of samples between any two transitions
        encountered since tracking was enabled.
        
        Returns:
            U64: Minimum pulse width in samples
    )pbdoc");

    // Additional methods
    channel_data.def("do_more_transitions_exist_in_current_data", 
                    &AnalyzerChannelData::DoMoreTransitionsExistInCurrentData, R"pbdoc(
        Check if more transitions exist in the current captured data.
        
        This method is useful when working with multiple channels to determine
        if a channel might not change state again in the captured data.
        
        Returns:
            bool: True if more transitions exist
    )pbdoc");

    // Navigation helpers (Python-specific convenience methods)
    channel_data.def("find_next_edge", [](AnalyzerChannelData &self) {
        U64 next_edge = self.GetSampleOfNextEdge();
        return next_edge;
    }, R"pbdoc(
        Find the next edge without moving.
        
        Returns:
            U64: Sample number of the next edge
    )pbdoc");

    channel_data.def("move_to_next_edge", [](AnalyzerChannelData &self) {
        U64 next_edge = self.GetSampleOfNextEdge();
        self.AdvanceToAbsPosition(next_edge);
        return next_edge;
    }, R"pbdoc(
        Move to the next edge and return its sample number.
        
        Returns:
            U64: Sample number of the edge
    )pbdoc");

    channel_data.def("advance_by_time", [](AnalyzerChannelData &self, double time_s, U32 sample_rate_hz) {
        U32 num_samples = static_cast<U32>(time_s * sample_rate_hz);
        return self.Advance(num_samples);
    }, R"pbdoc(
        Advance by a specified time in seconds.
        
        Args:
            time_s: Time to advance in seconds
            sample_rate_hz: Sample rate in Hz
            
        Returns:
            U32: Number of transitions that occurred
    )pbdoc", py::arg("time_s"), py::arg("sample_rate_hz"));

    // Bit pattern searching (Python-specific convenience methods)
    channel_data.def("find_pattern", [](AnalyzerChannelData &self, const std::vector<BitState> &pattern, 
                                       U64 max_samples_to_search) {
        U64 start_sample = self.GetSampleNumber();
        U64 current_sample = start_sample;
        size_t pattern_index = 0;
        BitState current_state = self.GetBitState();
        
        // Make a copy we can modify
        AnalyzerChannelData channel_copy = self;
        
        U64 max_sample = start_sample + max_samples_to_search;
        
        while (current_sample < max_sample) {
            if (current_state == pattern[pattern_index]) {
                pattern_index++;
                
                if (pattern_index >= pattern.size()) {
                    // Found complete pattern
                    return py::make_tuple(true, current_sample - pattern.size() + 1);
                }
                
                // Get next sample
                channel_copy.Advance(1);
                current_sample = channel_copy.GetSampleNumber();
                current_state = channel_copy.GetBitState();
            } else {
                // Mismatch, reset pattern search but don't go back in samples
                pattern_index = 0;
                
                // Get next sample
                channel_copy.Advance(1);
                current_sample = channel_copy.GetSampleNumber();
                current_state = channel_copy.GetBitState();
            }
        }
        
        // Pattern not found within limits
        return py::make_tuple(false, 0ULL);
    }, R"pbdoc(
        Find a specific bit pattern in the channel data.
        
        This helper searches for a sequence of bit states starting from the
        current position.
        
        Args:
            pattern: List of BitStates to search for
            max_samples_to_search: Maximum number of samples to look through
            
        Returns:
            tuple: (found, sample_number)
                found: True if pattern was found
                sample_number: Starting sample of the pattern if found
                
        Note: This method doesn't change the current position.
    )pbdoc", py::arg("pattern"), py::arg("max_samples_to_search"));

    // Wait for specific bit state (Python-specific convenience method)
    channel_data.def("wait_for_state", [](AnalyzerChannelData &self, BitState state) {
        U64 start_sample = self.GetSampleNumber();
        BitState current_state = self.GetBitState();
        
        // If we're already at the desired state, return current position
        if (current_state == state) {
            return start_sample;
        }
        
        // Otherwise, advance to next edge and check again
        self.AdvanceToNextEdge();
        current_state = self.GetBitState();
        
        // If we're now at the desired state, return new position
        if (current_state == state) {
            return self.GetSampleNumber();
        }
        
        // Otherwise, advance to the next edge
        self.AdvanceToNextEdge();
        return self.GetSampleNumber();
    }, R"pbdoc(
        Advance until a specific bit state is reached.
        
        Args:
            state: BitState to wait for (HIGH or LOW)
            
        Returns:
            U64: Sample number where the state was found
    )pbdoc", py::arg("state"));
}