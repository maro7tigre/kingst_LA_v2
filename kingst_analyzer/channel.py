"""
Channel module for the Kingst Logic Analyzer Python bindings.

This module provides Pythonic interfaces for accessing and manipulating channel data
from a Kingst Logic Analyzer. It wraps the low-level C++ bindings with higher-level
abstractions, visualization tools, and statistical functions for signal analysis.
"""

from typing import List, Tuple, Dict, Optional, Iterator, Union, Callable, Any
from enum import Enum
import warnings
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

# Import the C++ bindings - this would reference the actual binding module
# The actual import path may need to be adjusted based on the package structure
from kingst_analyzer._core import AnalyzerChannelData, BitState


class Channel:
    """
    Pythonic wrapper for logic analyzer channel data access and analysis.
    
    This class provides methods for navigating through channel data, detecting signal
    patterns, calculating timing information, and performing statistical analysis
    on digital signals. It also includes visualization helpers for displaying
    waveforms and timing diagrams.
    
    Attributes:
        name (str): User-defined name of the channel
        channel_index (int): Index of the channel in the analyzer
        sample_rate_hz (float): Sample rate in Hz
    
    Example:
        >>> channel = Channel(analyzer.get_channel_data(0), name="SPI_CLK", sample_rate_hz=20e6)
        >>> # Find all rising edges
        >>> edges = list(channel.find_all_edges(edge_type=EdgeType.RISING))
        >>> # Measure average period
        >>> avg_period = channel.measure_average_period()
        >>> print(f"Clock frequency: {1.0 / avg_period:.2f} Hz")
    """
    
    class EdgeType(Enum):
        """Types of signal edges for edge detection and analysis."""
        RISING = 0   # Transition from LOW to HIGH
        FALLING = 1  # Transition from HIGH to LOW
        BOTH = 2     # Any transition
    
    @dataclass
    class Pulse:
        """Represents a digital pulse with timing information."""
        start_sample: int
        end_sample: int
        start_time: float
        duration: float
        polarity: BitState
        
        @property
        def frequency(self) -> float:
            """Calculate frequency based on pulse duration."""
            if self.duration > 0:
                return 1.0 / self.duration
            return float('inf')
    
    @dataclass
    class Pattern:
        """Represents a detected bit pattern in the signal."""
        pattern: List[BitState]
        start_sample: int
        end_sample: int
        start_time: float
        duration: float
    
    def __init__(self, channel_data: AnalyzerChannelData, name: str = "", 
                 channel_index: int = -1, sample_rate_hz: float = 0):
        """
        Initialize a channel wrapper.
        
        Args:
            channel_data: The underlying AnalyzerChannelData instance
            name: User-defined name for this channel
            channel_index: Index of this channel in the analyzer (0-based)
            sample_rate_hz: Sample rate in Hz, needed for timing calculations
        """
        self._channel_data = channel_data
        self.name = name
        self.channel_index = channel_index
        self.sample_rate_hz = sample_rate_hz
        self._cached_data = None  # For numpy-based operations
    
    # -------------------------------------------------------------------------
    # Basic state and navigation methods (wrapping the C++ API)
    # -------------------------------------------------------------------------
    
    @property
    def sample_number(self) -> int:
        """Get the current sample position."""
        return self._channel_data.get_sample_number()
    
    @property
    def bit_state(self) -> BitState:
        """Get the current bit state (HIGH or LOW)."""
        return self._channel_data.get_bit_state()
    
    @property
    def time_s(self) -> float:
        """Get the current time in seconds from the start of capture."""
        return self.sample_number / self.sample_rate_hz if self.sample_rate_hz else 0.0
    
    def advance(self, num_samples: int) -> int:
        """
        Move forward by the specified number of samples.
        
        Args:
            num_samples: Number of samples to advance
            
        Returns:
            Number of transitions encountered
            
        Raises:
            RuntimeError: If attempting to advance beyond the data
        """
        return self._channel_data.advance(num_samples)
    
    def advance_to_position(self, sample_number: int) -> int:
        """
        Move to the specified absolute sample position.
        
        Args:
            sample_number: Sample position to move to
            
        Returns:
            Number of transitions encountered
            
        Raises:
            RuntimeError: If the position is beyond the data
        """
        return self._channel_data.advance_to_abs_position(sample_number)
    
    def advance_to_next_edge(self) -> None:
        """
        Move to the next edge (transition) in the data.
        
        Raises:
            RuntimeError: If no more edges exist
        """
        self._channel_data.advance_to_next_edge()
    
    def advance_by_time(self, time_s: float) -> int:
        """
        Move forward by the specified time in seconds.
        
        Args:
            time_s: Time to advance in seconds
            
        Returns:
            Number of transitions encountered
            
        Raises:
            ValueError: If time_s is negative
            RuntimeError: If attempting to advance beyond the data
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to use time-based navigation")
        return self._channel_data.advance_by_time(time_s, self.sample_rate_hz)
    
    def get_sample_of_next_edge(self) -> int:
        """
        Get the sample number of the next edge without moving.
        
        Returns:
            Sample number of the next edge
            
        Raises:
            RuntimeError: If no more edges exist
        """
        return self._channel_data.get_sample_of_next_edge()
    
    def wait_for_state(self, state: BitState) -> int:
        """
        Advance until a specific bit state is reached.
        
        Args:
            state: BitState to wait for (BitState.HIGH or BitState.LOW)
            
        Returns:
            Sample number where the state was found
            
        Raises:
            ValueError: If the state cannot be found
        """
        return self._channel_data.wait_for_state(state)
    
    def find_pattern(self, pattern: List[BitState], max_samples: int = 100000) -> Tuple[bool, int]:
        """
        Find a specific bit pattern starting from the current position.
        
        This method doesn't change the current position.
        
        Args:
            pattern: List of BitStates to search for
            max_samples: Maximum number of samples to search through
            
        Returns:
            Tuple of (found, sample_number)
            
        Raises:
            ValueError: If pattern is empty
        """
        return self._channel_data.find_pattern(pattern, max_samples)
    
    def measure_time_between(self, start_sample: int, end_sample: int) -> float:
        """
        Measure time between two sample positions in seconds.
        
        Args:
            start_sample: Starting sample number
            end_sample: Ending sample number
            
        Returns:
            Time between samples in seconds
            
        Raises:
            ValueError: If end_sample is less than start_sample or sample rate is not set
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to measure time")
        return self._channel_data.measure_time_between(
            start_sample, end_sample, self.sample_rate_hz)
    
    def count_transitions_between(self, start_sample: int, end_sample: int) -> int:
        """
        Count the number of transitions between two sample positions.
        
        This method doesn't change the current position.
        
        Args:
            start_sample: Starting sample number
            end_sample: Ending sample number
            
        Returns:
            Number of transitions between the positions
            
        Raises:
            ValueError: If end_sample is less than start_sample
        """
        return self._channel_data.count_transitions_between(start_sample, end_sample)
    
    def find_nth_edge(self, n: int) -> int:
        """
        Find the sample number of the nth edge from the current position.
        
        This method doesn't change the current position.
        
        Args:
            n: Which edge to find (1 for next edge, 2 for second edge, etc.)
            
        Returns:
            Sample number of the nth edge
            
        Raises:
            ValueError: If n is 0 or if not enough edges exist
        """
        return self._channel_data.find_nth_edge(n)
    
    # -------------------------------------------------------------------------
    # Extended navigation and search methods
    # -------------------------------------------------------------------------
    
    def find_edge(self, edge_type: EdgeType = EdgeType.BOTH, start_sample: Optional[int] = None) -> int:
        """
        Find the next edge of the specified type from the given position.
        
        This method doesn't change the current position.
        
        Args:
            edge_type: Type of edge to find (RISING, FALLING, or BOTH)
            start_sample: Sample to start searching from (default: current position)
            
        Returns:
            Sample number of the found edge
            
        Raises:
            RuntimeError: If no matching edge is found
        """
        if start_sample is not None:
            # Save current position
            current = self.sample_number
            self.advance_to_position(start_sample)
        
        try:
            if edge_type == self.EdgeType.BOTH:
                # Find any edge
                return self.get_sample_of_next_edge()
            
            # For specific edge types, we need to check the current state
            current_state = self.bit_state
            target_state = None
            
            if edge_type == self.EdgeType.RISING:
                # If currently LOW, next edge is rising
                # If currently HIGH, we need to find a falling edge first
                if current_state == BitState.HIGH:
                    self.get_sample_of_next_edge()  # Find falling edge
                    self.advance_to_next_edge()     # Move to falling edge
                    edge = self.get_sample_of_next_edge()  # Find rising edge
                else:
                    edge = self.get_sample_of_next_edge()  # Already at LOW, find rising
            else:  # FALLING
                # If currently HIGH, next edge is falling
                # If currently LOW, we need to find a rising edge first
                if current_state == BitState.LOW:
                    self.get_sample_of_next_edge()  # Find rising edge
                    self.advance_to_next_edge()     # Move to rising edge
                    edge = self.get_sample_of_next_edge()  # Find falling edge
                else:
                    edge = self.get_sample_of_next_edge()  # Already at HIGH, find falling
                    
            return edge
        finally:
            # Restore position if we changed it
            if start_sample is not None:
                self.advance_to_position(current)
    
    def find_all_edges(self, max_count: int = 1000, edge_type: EdgeType = EdgeType.BOTH, 
                      start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> Iterator[int]:
        """
        Find all edges of the specified type in a range.
        
        This is a generator function that yields edge positions.
        The current position is not changed.
        
        Args:
            max_count: Maximum number of edges to find
            edge_type: Type of edge to find (RISING, FALLING, or BOTH)
            start_sample: Sample to start searching from (default: current position)
            end_sample: Sample to end searching at (default: end of data)
            
        Yields:
            Sample numbers of found edges
            
        Example:
            >>> # Find first 10 rising edges
            >>> edges = list(channel.find_all_edges(max_count=10, edge_type=Channel.EdgeType.RISING))
        """
        # Save current position
        current_pos = self.sample_number
        
        try:
            # Set starting position
            if start_sample is not None:
                self.advance_to_position(start_sample)
            
            count = 0
            while count < max_count:
                try:
                    edge = self.find_edge(edge_type)
                    
                    # Stop if we've gone past the end sample
                    if end_sample is not None and edge > end_sample:
                        break
                    
                    yield edge
                    count += 1
                    
                    # Move past this edge to find the next one
                    self.advance_to_position(edge)
                    self.advance(1)
                except (RuntimeError, ValueError):
                    # No more edges found
                    break
        finally:
            # Restore original position
            self.advance_to_position(current_pos)
    
    def find_all_patterns(self, pattern: List[BitState], max_count: int = 100, 
                         start_sample: Optional[int] = None, 
                         end_sample: Optional[int] = None) -> Iterator[Pattern]:
        """
        Find all occurrences of a bit pattern in a range.
        
        This is a generator function that yields Pattern objects.
        The current position is not changed.
        
        Args:
            pattern: List of BitStates to search for
            max_count: Maximum number of patterns to find
            start_sample: Sample to start searching from (default: current position)
            end_sample: Sample to end searching at (default: end of data)
            
        Yields:
            Pattern objects with details about each found pattern
            
        Example:
            >>> # Find all START conditions (defined as LOW-HIGH-LOW sequence)
            >>> start_pattern = [BitState.LOW, BitState.HIGH, BitState.LOW]
            >>> for pattern in channel.find_all_patterns(start_pattern):
            >>>     print(f"START found at sample {pattern.start_sample}, time {pattern.start_time:.6f}s")
        """
        # Save current position
        current_pos = self.sample_number
        
        try:
            # Set starting position
            if start_sample is not None:
                self.advance_to_position(start_sample)
            current_sample = self.sample_number
            
            count = 0
            while count < max_count:
                # Find the next pattern
                found, start = self.find_pattern(pattern)
                if not found:
                    break
                    
                # Calculate end sample (start + pattern length in bits)
                # We need to advance to the start and then count edges
                self.advance_to_position(start)
                end = start
                for _ in range(len(pattern) - 1):
                    try:
                        self.advance_to_next_edge()
                        end = self.sample_number
                    except RuntimeError:
                        break
                
                # Stop if we've gone past the end sample
                if end_sample is not None and start > end_sample:
                    break
                
                # Create a Pattern object
                start_time = start / self.sample_rate_hz if self.sample_rate_hz else 0
                duration = (end - start) / self.sample_rate_hz if self.sample_rate_hz else 0
                pattern_obj = self.Pattern(pattern, start, end, start_time, duration)
                
                yield pattern_obj
                count += 1
                
                # Move past this pattern to continue search
                self.advance_to_position(end)
                self.advance(1)
                current_sample = self.sample_number
                
                # Check if we've reached the end
                if end_sample is not None and current_sample >= end_sample:
                    break
        finally:
            # Restore original position
            self.advance_to_position(current_pos)
    
    def find_all_pulses(self, polarity: Optional[BitState] = None, min_width: float = 0, 
                       max_width: Optional[float] = None, max_count: int = 1000,
                       start_sample: Optional[int] = None, 
                       end_sample: Optional[int] = None) -> Iterator[Pulse]:
        """
        Find all pulses matching the specified criteria.
        
        This is a generator function that yields Pulse objects.
        The current position is not changed.
        
        Args:
            polarity: BitState of the pulse (HIGH or LOW), or None for both
            min_width: Minimum pulse width in seconds (default: 0)
            max_width: Maximum pulse width in seconds, or None for no limit
            max_count: Maximum number of pulses to find
            start_sample: Sample to start searching from (default: current position)
            end_sample: Sample to end searching at (default: end of data)
            
        Yields:
            Pulse objects with details about each found pulse
            
        Example:
            >>> # Find all HIGH pulses between 1ms and 2ms
            >>> for pulse in channel.find_all_pulses(BitState.HIGH, min_width=0.001, max_width=0.002):
            >>>     print(f"Pulse at {pulse.start_time:.6f}s, duration {pulse.duration*1000:.3f}ms")
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to find pulses by width")
        
        # Convert time specifications to sample counts
        min_samples = int(min_width * self.sample_rate_hz) if min_width else 0
        max_samples = int(max_width * self.sample_rate_hz) if max_width else None
        
        # Save current position
        current_pos = self.sample_number
        
        try:
            # Set starting position
            if start_sample is not None:
                self.advance_to_position(start_sample)
            
            count = 0
            while count < max_count:
                try:
                    # Get current state
                    current_state = self.bit_state
                    
                    # Skip if we're looking for a specific polarity and current state doesn't match
                    if polarity is not None and current_state != polarity:
                        # Advance to the state we're looking for
                        try:
                            self.advance_to_next_edge()  # Move to next edge
                        except RuntimeError:
                            break  # No more edges
                        continue
                    
                    # Record start of pulse
                    pulse_start = self.sample_number
                    
                    # Find end of pulse (next edge)
                    try:
                        self.advance_to_next_edge()
                    except RuntimeError:
                        # No more edges, end of data
                        break
                        
                    pulse_end = self.sample_number
                    pulse_width = pulse_end - pulse_start
                    
                    # Check width constraints
                    if min_samples and pulse_width < min_samples:
                        continue
                    if max_samples and pulse_width > max_samples:
                        continue
                        
                    # Stop if we've gone past the end sample
                    if end_sample is not None and pulse_end > end_sample:
                        break
                    
                    # Create a Pulse object
                    start_time = pulse_start / self.sample_rate_hz
                    duration = pulse_width / self.sample_rate_hz
                    pulse_obj = self.Pulse(pulse_start, pulse_end, start_time, duration, current_state)
                    
                    yield pulse_obj
                    count += 1
                except (RuntimeError, ValueError):
                    # No more edges found
                    break
        finally:
            # Restore original position
            self.advance_to_position(current_pos)
    
    # -------------------------------------------------------------------------
    # Signal analysis and statistics methods
    # -------------------------------------------------------------------------
    
    def measure_frequency(self, num_cycles: int = 10) -> float:
        """
        Measure the frequency of the signal over a number of cycles.
        
        This method doesn't change the current position.
        
        Args:
            num_cycles: Number of cycles to measure over
            
        Returns:
            Frequency in Hz
            
        Raises:
            ValueError: If sample rate is not set or not enough cycles are found
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to measure frequency")
        
        # Save current position
        current_pos = self.sample_number
        
        try:
            # Count edges over num_cycles complete cycles (2 edges per cycle)
            edges_needed = num_cycles * 2
            
            # Find the first edge
            first_edge = self.get_sample_of_next_edge()
            self.advance_to_position(first_edge)
            
            # Find the last edge
            last_edge = first_edge
            for _ in range(edges_needed):
                self.advance_to_next_edge()
                last_edge = self.sample_number
            
            # Calculate frequency
            time_diff = (last_edge - first_edge) / self.sample_rate_hz
            return num_cycles / time_diff
        except RuntimeError:
            raise ValueError("Not enough edges found to measure frequency")
        finally:
            # Restore original position
            self.advance_to_position(current_pos)
    
    def measure_duty_cycle(self, num_cycles: int = 10) -> float:
        """
        Measure the duty cycle of the signal over a number of cycles.
        
        This method doesn't change the current position.
        
        Args:
            num_cycles: Number of cycles to measure over
            
        Returns:
            Duty cycle as a percentage (0-100)
            
        Raises:
            ValueError: If sample rate is not set or not enough cycles are found
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to measure duty cycle")
        
        # Save current position
        current_pos = self.sample_number
        
        try:
            high_time = 0
            total_time = 0
            
            # Ensure we start at a known state transition
            first_edge = self.get_sample_of_next_edge()
            self.advance_to_position(first_edge)
            
            # Get initial state after first edge
            self.advance_to_next_edge()
            initial_state = self.bit_state
            
            # Process cycles
            cycle_count = 0
            last_edge = self.sample_number
            state = initial_state
            
            while cycle_count < num_cycles:
                try:
                    # Record current position
                    edge_start = self.sample_number
                    
                    # Move to next edge
                    self.advance_to_next_edge()
                    edge_end = self.sample_number
                    
                    # Calculate time in this state
                    time_in_state = edge_end - edge_start
                    
                    # Add to high time if state was HIGH
                    if state == BitState.HIGH:
                        high_time += time_in_state
                    
                    # Toggle state
                    state = BitState.HIGH if state == BitState.LOW else BitState.LOW
                    
                    # Check if we completed a cycle (back to initial state)
                    if state == initial_state:
                        cycle_count += 1
                    
                    last_edge = edge_end
                except RuntimeError:
                    if cycle_count == 0:
                        raise ValueError("Not enough edges found to measure duty cycle")
                    break
            
            # Calculate total time and duty cycle
            total_time = last_edge - first_edge
            return (high_time / total_time) * 100.0
        finally:
            # Restore original position
            self.advance_to_position(current_pos)
    
    def measure_pulse_statistics(self, polarity: Optional[BitState] = None, 
                                max_pulses: int = 1000) -> Dict[str, float]:
        """
        Measure statistics about pulse widths in the signal.
        
        This method doesn't change the current position.
        
        Args:
            polarity: BitState of pulses to measure (HIGH or LOW), or None for both
            max_pulses: Maximum number of pulses to analyze
            
        Returns:
            Dictionary with statistics (min, max, mean, median, std_dev)
            
        Raises:
            ValueError: If sample rate is not set or no pulses are found
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to measure pulse statistics")
        
        # Get all pulses
        pulses = list(self.find_all_pulses(polarity=polarity, max_count=max_pulses))
        
        if not pulses:
            raise ValueError("No pulses found for analysis")
        
        # Extract durations
        durations = [pulse.duration for pulse in pulses]
        
        # Calculate statistics
        stats = {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations) / len(durations),
            "median": sorted(durations)[len(durations) // 2],
            "std_dev": np.std(durations) if np is not None else None
        }
        
        return stats
    
    def measure_average_period(self, num_periods: int = 10) -> float:
        """
        Measure the average period of the signal.
        
        This method doesn't change the current position.
        
        Args:
            num_periods: Number of periods to average over
            
        Returns:
            Average period in seconds
            
        Raises:
            ValueError: If sample rate is not set or not enough periods are found
        """
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be set to measure period")
        
        try:
            freq = self.measure_frequency(num_periods)
            return 1.0 / freq
        except ValueError as e:
            raise ValueError(f"Could not measure average period: {str(e)}")
    
    def count_pulses(self, polarity: Optional[BitState] = None, 
                    min_width: Optional[float] = None, 
                    max_width: Optional[float] = None,
                    start_sample: Optional[int] = None, 
                    end_sample: Optional[int] = None) -> int:
        """
        Count pulses matching the specified criteria.
        
        This method doesn't change the current position.
        
        Args:
            polarity: BitState of pulses to count (HIGH or LOW), or None for both
            min_width: Minimum pulse width in seconds, or None for no limit
            max_width: Maximum pulse width in seconds, or None for no limit
            start_sample: Sample to start counting from (default: current position)
            end_sample: Sample to end counting at (default: end of data)
            
        Returns:
            Number of matching pulses
        """
        pulses = self.find_all_pulses(
            polarity=polarity,
            min_width=min_width or 0,
            max_width=max_width,
            start_sample=start_sample,
            end_sample=end_sample,
            max_count=1000000  # Effectively unlimited
        )
        
        return sum(1 for _ in pulses)
    
    # -------------------------------------------------------------------------
    # Data conversion and numpy integration
    # -------------------------------------------------------------------------
    
    def to_numpy(self, start_sample: Optional[int] = None, 
                end_sample: Optional[int] = None, 
                max_samples: int = 1000000) -> np.ndarray:
        """
        Convert channel data to a numpy array for advanced analysis.
        
        This builds a structured array with sample numbers and bit states.
        
        Args:
            start_sample: First sample to include (default: current position)
            end_sample: Last sample to include (default: start + max_samples)
            max_samples: Maximum number of samples to convert
            
        Returns:
            Numpy structured array with 'sample' and 'state' fields
            
        Raises:
            ImportError: If numpy is not available
        """
        if np is None:
            raise ImportError("NumPy is required for this operation")
        
        # Save current position
        current_pos = self.sample_number
        
        try:
            # Set starting position
            if start_sample is not None:
                self.advance_to_position(start_sample)
            
            # Default end sample if not specified
            if end_sample is None:
                end_sample = self.sample_number + max_samples
            
            # Limit by max_samples
            end_sample = min(end_sample, self.sample_number + max_samples)
            
            # Extract data around edges for efficiency
            samples = []
            states = []
            
            # Record initial sample and state
            samples.append(self.sample_number)
            states.append(1 if self.bit_state == BitState.HIGH else 0)
            
            # Iterate through edges until end_sample
            while self.sample_number < end_sample:
                try:
                    # Find next edge
                    edge = self.get_sample_of_next_edge()
                    
                    # Stop if past end
                    if edge > end_sample:
                        break
                    
                    # Move to edge
                    self.advance_to_position(edge)
                    
                    # Record new sample and state
                    samples.append(self.sample_number)
                    states.append(1 if self.bit_state == BitState.HIGH else 0)
                except RuntimeError:
                    # No more edges
                    break
            
            # Create structured array
            dtype = [('sample', np.uint64), ('state', np.uint8)]
            return np.array(list(zip(samples, states)), dtype=dtype)
        finally:
            # Restore original position
            self.advance_to_position(current_pos)
    
    def to_transitions(self, start_sample: Optional[int] = None, 
                      end_sample: Optional[int] = None, 
                      max_transitions: int = 10000) -> List[Tuple[int, BitState]]:
        """
        Extract a list of transitions (edges) from the data.
        
        This is more memory-efficient than to_numpy() for sparse signals.
        
        Args:
            start_sample: First sample to include (default: current position)
            end_sample: Last sample to include (default: end of data)
            max_transitions: Maximum number of transitions to extract
            
        Returns:
            List of (sample_number, new_state) tuples
        """
        # Save current position
        current_pos = self.sample_number
        
        try:
            # Set starting position
            if start_sample is not None:
                self.advance_to_position(start_sample)
            
            transitions = []
            count = 0
            
            # Record initial state
            initial_sample = self.sample_number
            initial_state = self.bit_state
            transitions.append((initial_sample, initial_state))
            
            # Find transitions
            while count < max_transitions:
                try:
                    # Find and move to next edge
                    self.advance_to_next_edge()
                    
                    # Stop if past end
                    if end_sample is not None and self.sample_number > end_sample:
                        break
                    
                    # Record transition
                    transitions.append((self.sample_number, self.bit_state))
                    count += 1
                except RuntimeError:
                    # No more edges
                    break
            
            return transitions
        finally:
            # Restore original position
            self.advance_to_position(current_pos)
    
    # -------------------------------------------------------------------------
    # Visualization methods
    # -------------------------------------------------------------------------
    
    def plot_waveform(self, start_sample: Optional[int] = None, 
                     end_sample: Optional[int] = None,
                     title: Optional[str] = None,
                     show_grid: bool = True,
                     time_unit: str = 's',
                     figsize: Tuple[int, int] = (12, 3)) -> Any:
        """
        Plot the signal waveform using matplotlib.
        
        Args:
            start_sample: First sample to plot (default: current position)
            end_sample: Last sample to plot (default: auto-detect reasonable range)
            title: Plot title (default: channel name)
            show_grid: Whether to show grid lines
            time_unit: Time unit for x-axis ('s', 'ms', 'us', or 'ns')
            figsize: Figure size as (width, height) tuple
            
        Returns:
            The figure and axes objects for further customization
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        # Get transitions for plotting
        transitions = self.to_transitions(
            start_sample=start_sample,
            end_sample=end_sample,
            max_transitions=10000
        )
        
        if not transitions:
            warnings.warn("No transitions found in the specified range")
            # Create empty plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel(f"Time ({time_unit})")
            ax.set_ylabel("Logic Level")
            ax.set_title(title or self.name or f"Channel {self.channel_index}")
            return fig, ax
        
        # Extract samples and states
        samples, states = zip(*[(s, 1 if st == BitState.HIGH else 0) for s, st in transitions])
        
        # Convert to time
        time_divisors = {'s': 1, 'ms': 1e3, 'us': 1e6, 'ns': 1e9}
        divisor = time_divisors.get(time_unit, 1)
        
        if self.sample_rate_hz <= 0:
            # Use sample numbers if no sample rate
            times = samples
            xlabel = "Sample Number"
        else:
            # Convert to selected time unit
            times = [s / self.sample_rate_hz * divisor for s in samples]
            xlabel = f"Time ({time_unit})"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create step plot (digital waveform)
        ax.step(times, states, where='post', color='blue', linewidth=2)
        
        # Set limits and labels
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Low', 'High'])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Logic Level")
        ax.set_title(title or self.name or f"Channel {self.channel_index}")
        
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_timing_diagram(self, channels: List['Channel'], 
                           start_sample: Optional[int] = None,
                           end_sample: Optional[int] = None,
                           title: str = "Timing Diagram",
                           time_unit: str = 'ms',
                           show_grid: bool = True,
                           figsize: Tuple[int, int] = (12, 6)) -> Any:
        """
        Plot a multi-channel timing diagram including this channel.
        
        Args:
            channels: List of Channel objects to include in the diagram
            start_sample: First sample to plot (default: current position)
            end_sample: Last sample to plot (default: auto-detect)
            title: Plot title
            time_unit: Time unit for x-axis ('s', 'ms', 'us', or 'ns')
            show_grid: Whether to show grid lines
            figsize: Figure size as (width, height) tuple
            
        Returns:
            The figure and axes objects for further customization
            
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        # Include this channel if not already in the list
        if self not in channels:
            channels = [self] + channels
        
        # Create figure with one subplot per channel
        fig, axes = plt.subplots(len(channels), 1, figsize=figsize, sharex=True)
        if len(channels) == 1:
            axes = [axes]  # Make sure axes is always a list
        
        # Time unit conversion
        time_divisors = {'s': 1, 'ms': 1e3, 'us': 1e6, 'ns': 1e9}
        divisor = time_divisors.get(time_unit, 1)
        
        # Plot each channel
        for i, channel in enumerate(channels):
            # Get transitions
            transitions = channel.to_transitions(
                start_sample=start_sample,
                end_sample=end_sample,
                max_transitions=10000
            )
            
            if not transitions:
                # No transitions, plot a flat line
                if start_sample is None:
                    start_sample = 0
                if end_sample is None:
                    end_sample = start_sample + 1000
                    
                state_value = 1 if channel.bit_state == BitState.HIGH else 0
                
                if self.sample_rate_hz <= 0:
                    # Use sample numbers if no sample rate
                    times = [start_sample, end_sample]
                    states = [state_value, state_value]
                else:
                    # Convert to selected time unit
                    times = [s / self.sample_rate_hz * divisor for s in [start_sample, end_sample]]
                    states = [state_value, state_value]
            else:
                # Extract samples and states
                samples, states = zip(*[(s, 1 if st == BitState.HIGH else 0) for s, st in transitions])
                
                if self.sample_rate_hz <= 0:
                    # Use sample numbers if no sample rate
                    times = samples
                else:
                    # Convert to selected time unit
                    times = [s / self.sample_rate_hz * divisor for s in samples]
            
            # Create step plot for this channel
            axes[i].step(times, states, where='post', color='blue', linewidth=2)
            
            # Set limits and labels
            axes[i].set_ylim(-0.1, 1.1)
            axes[i].set_yticks([0, 1])
            axes[i].set_yticklabels(['L', 'H'])
            
            # Add channel name as label
            name = channel.name or f"Channel {channel.channel_index}"
            axes[i].set_ylabel(name, rotation=0, labelpad=20, ha='right')
            
            if show_grid:
                axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Set common x-axis label
        if self.sample_rate_hz <= 0:
            axes[-1].set_xlabel("Sample Number")
        else:
            axes[-1].set_xlabel(f"Time ({time_unit})")
        
        # Set title
        fig.suptitle(title)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        
        return fig, axes
    
    # -------------------------------------------------------------------------
    # Python protocols for iteration and context management
    # -------------------------------------------------------------------------
    
    def __iter__(self) -> Iterator[Tuple[int, BitState]]:
        """
        Iterate through transitions in the channel data.
        
        Each iteration returns a tuple of (sample_number, bit_state).
        The original position is restored after iteration.
        
        Example:
            >>> for sample, state in channel:
            >>>     print(f"At sample {sample}: {'HIGH' if state == BitState.HIGH else 'LOW'}")
        """
        # Save current position
        original_pos = self.sample_number
        
        try:
            # Start from current position
            current_sample = self.sample_number
            current_state = self.bit_state
            
            # Yield initial state
            yield (current_sample, current_state)
            
            # Iterate through all transitions
            while True:
                try:
                    self.advance_to_next_edge()
                    current_sample = self.sample_number
                    current_state = self.bit_state
                    yield (current_sample, current_state)
                except RuntimeError:
                    # No more edges
                    break
        finally:
            # Restore original position
            self.advance_to_position(original_pos)
    
    def __enter__(self) -> 'Channel':
        """Enter a context that saves the current position."""
        self._saved_position = self.sample_number
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and restore the saved position."""
        if hasattr(self, '_saved_position'):
            self.advance_to_position(self._saved_position)
            delattr(self, '_saved_position')
    
    # -------------------------------------------------------------------------
    # Protocol decoder integration
    # -------------------------------------------------------------------------
    
    def decode_protocol(self, protocol_type: str, **kwargs) -> Any:
        """
        Decode a standard protocol from this channel.
        
        This is a convenience method that forwards to the appropriate
        protocol decoder class.
        
        Args:
            protocol_type: Protocol type ('spi', 'i2c', 'uart', etc.)
            **kwargs: Protocol-specific configuration parameters
            
        Returns:
            Protocol decoder results
            
        Raises:
            ImportError: If the required protocol decoder module is not available
            ValueError: For unsupported protocol types
        """
        try:
            from kingst_analyzer import protocols
        except ImportError:
            raise ImportError("Protocol decoders not available - install the protocols package")
        
        # Delegate to the appropriate protocol decoder
        if protocol_type.lower() == 'spi':
            return protocols.decode_spi(self, **kwargs)
        elif protocol_type.lower() == 'i2c':
            return protocols.decode_i2c(self, **kwargs)
        elif protocol_type.lower() == 'uart':
            return protocols.decode_uart(self, **kwargs)
        else:
            raise ValueError(f"Unsupported protocol type: {protocol_type}")