"""
Kingst Logic Analyzer Results Module

This module provides Pythonic classes for accessing and working with Kingst Logic Analyzer results.
It wraps the C++ bindings with intuitive data access patterns, iterators, and filtering capabilities,
along with extensive analysis and visualization tools.

The classes and functions in this module allow you to:
- Access and manipulate analyzer frames, packets, and transactions
- Filter and search for specific data patterns
- Perform statistical analysis on decoded data
- Visualize results with customizable plots and timelines
- Export results to various formats

Typical usage:
    >>> from kingst_analyzer import AnalyzerResults
    >>> from kingst_LA.results import create_analyzer_results
    >>>
    >>> # Get Python wrapper for analyzer results
    >>> results = create_analyzer_results(analyzer.get_results())
    >>>
    >>> # Access all frames
    >>> for frame in results.frames:
    >>>     print(f"Frame at {frame.starting_sample}: data=0x{frame.data1:x}")
    >>>
    >>> # Filter frames by type
    >>> spi_data_frames = results.frames.filter_by_type(SPI_DATA_FRAME_TYPE)
    >>>
    >>> # Export to CSV
    >>> results.export("spi_data.csv", display_base=DisplayBase.HEXADECIMAL)
    >>>
    >>> # Analyze timing
    >>> stats = results.get_statistics()
    >>> print(f"Average frame duration: {stats['timing']['mean_duration']} samples")
    >>>
    >>> # Visualize results
    >>> results.visualize_timeline(title="SPI Communication Timeline")
"""

from __future__ import annotations
from typing import (
    Iterator, List, Dict, Optional, Union, Tuple, Callable, 
    Any, TypeVar, Sequence, Set, DefaultDict, cast,
    Protocol, runtime_checkable, Generic, overload
)
from enum import IntEnum, auto
import os
from pathlib import Path
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
import contextlib
import io
import csv
import datetime

# Import the C++ bindings
try:
    import kingst_analyzer._bindings as _bindings
except ImportError:
    # For development/documentation without the actual bindings
    class _bindings:
        """Mock bindings for documentation and development."""
        
        class MarkerType(IntEnum):
            DOT = 0
            ERROR_DOT = 1
            SQUARE = 2
            ERROR_SQUARE = 3
            UP_ARROW = 4
            DOWN_ARROW = 5
            X = 6
            ERROR_X = 7
            START = 8
            STOP = 9
            ONE = 10
            ZERO = 11
            
        class Frame:
            """Mock Frame class."""
            def __init__(self):
                self.starting_sample = 0
                self.ending_sample = 0
                self.data1 = 0
                self.data2 = 0
                self.type = 0
                self.flags = 0
                
            def has_flag(self, flag):
                return bool(self.flags & flag)
                
        class AnalyzerResults:
            """Mock AnalyzerResults class."""
            def __init__(self):
                pass
                
            def get_num_frames(self):
                return 0
                
            # Additional mock methods would be defined here
        
        # Mock constants
        DISPLAY_AS_ERROR_FLAG = 0x80
        DISPLAY_AS_WARNING_FLAG = 0x40
        INVALID_RESULT_INDEX = 0xFFFFFFFFFFFFFFFF

# Optional dependencies for advanced functionality
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Re-export constants
DISPLAY_AS_ERROR_FLAG = _bindings.DISPLAY_AS_ERROR_FLAG
DISPLAY_AS_WARNING_FLAG = _bindings.DISPLAY_AS_WARNING_FLAG
INVALID_RESULT_INDEX = _bindings.INVALID_RESULT_INDEX

# Type variables for generic collections
T = TypeVar('T')
F = TypeVar('F', bound='Frame')  # For Frame types


class MarkerType(IntEnum):
    """
    Types of markers that can be displayed on channels.
    
    Markers are visual indicators shown on the channel waveforms to highlight
    important points in the signal.
    
    Attributes:
        DOT: A normal dot marker (default)
        ERROR_DOT: An error dot marker (displayed in red)
        SQUARE: A square marker
        ERROR_SQUARE: An error square marker (displayed in red)
        UP_ARROW: An up arrow marker (↑)
        DOWN_ARROW: A down arrow marker (↓)
        X: An X marker (×)
        ERROR_X: An error X marker (displayed in red)
        START: A start marker (used to indicate the beginning of a transaction)
        STOP: A stop marker (used to indicate the end of a transaction)
        ONE: A one marker (1)
        ZERO: A zero marker (0)
    """
    DOT = _bindings.MarkerType.DOT
    ERROR_DOT = _bindings.MarkerType.ERROR_DOT
    SQUARE = _bindings.MarkerType.SQUARE
    ERROR_SQUARE = _bindings.MarkerType.ERROR_SQUARE
    UP_ARROW = _bindings.MarkerType.UP_ARROW
    DOWN_ARROW = _bindings.MarkerType.DOWN_ARROW
    X = _bindings.MarkerType.X
    ERROR_X = _bindings.MarkerType.ERROR_X
    START = _bindings.MarkerType.START
    STOP = _bindings.MarkerType.STOP
    ONE = _bindings.MarkerType.ONE
    ZERO = _bindings.MarkerType.ZERO


class DisplayBase(IntEnum):
    """
    Display base for numerical values in analyzer results.
    
    Determines how values are formatted in UI, exports, and textual representations.
    
    Attributes:
        ASCII: Display values as ASCII characters (where appropriate)
        DECIMAL: Display values in decimal (base-10)
        HEXADECIMAL: Display values in hexadecimal (base-16, prefixed with '0x')
        BINARY: Display values in binary (base-2, prefixed with '0b')
        OCTAL: Display values in octal (base-8, prefixed with '0o')
    """
    ASCII = 0
    DECIMAL = 1
    HEXADECIMAL = 2
    BINARY = 3
    OCTAL = 4
    
    def format_value(self, value: int, width: int = 0) -> str:
        """
        Format an integer value according to this display base.
        
        Args:
            value: The integer value to format
            width: Minimum width (in characters) for the formatted result
                  For binary and hex, this is the number of bits/nibbles
        
        Returns:
            A string representation of the value in the appropriate base
        """
        if self == DisplayBase.ASCII:
            try:
                if 32 <= value <= 126:  # Printable ASCII
                    return f"'{chr(value)}' ({value})"
                else:
                    return f"<{value}>"
            except (ValueError, OverflowError):
                return f"<{value}>"
                
        elif self == DisplayBase.DECIMAL:
            return f"{value:{width}d}"
            
        elif self == DisplayBase.HEXADECIMAL:
            hex_chars = max(1, (width + 3) // 4)  # 4 bits per hex char
            return f"0x{value:0{hex_chars}X}"
            
        elif self == DisplayBase.BINARY:
            return f"0b{value:0{width}b}"
            
        elif self == DisplayBase.OCTAL:
            return f"0o{value:0{width}o}"
            
        # Default: decimal
        return str(value)


@dataclass
class Frame:
    """
    Represents a single protocol data frame.
    
    A frame is a decoded piece of protocol data with start/end sample positions
    and up to two 64-bit data values. The meaning of the data values depends on
    the specific analyzer and frame type.
    
    Attributes:
        starting_sample (int): The first sample index (inclusive) of this frame
        ending_sample (int): The last sample index (inclusive) of this frame
        data1 (int): Primary data value (meaning depends on frame type)
        data2 (int): Secondary data value (meaning depends on frame type)
        type (int): Frame type identifier (meaning depends on analyzer)
        flags (int): Flags for this frame (error, warning, etc.)
    
    Properties:
        duration (int): Duration of the frame in samples
        is_error (bool): True if this frame has the error flag set
        is_warning (bool): True if this frame has the warning flag set
    """
    
    _frame: _bindings.Frame
    
    def __init__(self, frame: Union[_bindings.Frame, Dict]):
        """
        Initialize a Frame from a binding frame or a dictionary.
        
        Args:
            frame: Either a _bindings.Frame object or a dictionary with frame attributes
        """
        if isinstance(frame, dict):
            self._frame = _bindings.Frame()
            self._frame.starting_sample = frame.get('starting_sample', 0)
            self._frame.ending_sample = frame.get('ending_sample', 0)
            self._frame.data1 = frame.get('data1', 0)
            self._frame.data2 = frame.get('data2', 0)
            self._frame.type = frame.get('type', 0)
            self._frame.flags = frame.get('flags', 0)
        else:
            self._frame = frame
            
    @property
    def starting_sample(self) -> int:
        """Get the starting sample index of this frame."""
        return self._frame.starting_sample
    
    @starting_sample.setter
    def starting_sample(self, value: int) -> None:
        """Set the starting sample index of this frame."""
        self._frame.starting_sample = value
        
    @property
    def ending_sample(self) -> int:
        """Get the ending sample index of this frame."""
        return self._frame.ending_sample
    
    @ending_sample.setter
    def ending_sample(self, value: int) -> None:
        """Set the ending sample index of this frame."""
        self._frame.ending_sample = value
        
    @property
    def data1(self) -> int:
        """Get the primary data value."""
        return self._frame.data1
    
    @data1.setter
    def data1(self, value: int) -> None:
        """Set the primary data value."""
        self._frame.data1 = value
        
    @property
    def data2(self) -> int:
        """Get the secondary data value."""
        return self._frame.data2
    
    @data2.setter
    def data2(self, value: int) -> None:
        """Set the secondary data value."""
        self._frame.data2 = value
        
    @property
    def type(self) -> int:
        """Get the frame type."""
        return self._frame.type
    
    @type.setter
    def type(self, value: int) -> None:
        """Set the frame type."""
        self._frame.type = value
        
    @property
    def flags(self) -> int:
        """Get the frame flags."""
        return self._frame.flags
    
    @flags.setter
    def flags(self, value: int) -> None:
        """Set the frame flags."""
        self._frame.flags = value
    
    @property
    def duration(self) -> int:
        """Get the duration of this frame in samples."""
        return self.ending_sample - self.starting_sample + 1
    
    @property
    def is_error(self) -> bool:
        """Check if this frame has the error flag set."""
        return self.has_flag(DISPLAY_AS_ERROR_FLAG)
    
    @property
    def is_warning(self) -> bool:
        """Check if this frame has the warning flag set."""
        return self.has_flag(DISPLAY_AS_WARNING_FLAG)
    
    def has_flag(self, flag: int) -> bool:
        """
        Check if this frame has a specific flag set.
        
        Args:
            flag: Flag to check
            
        Returns:
            True if the flag is set, False otherwise
        """
        return self._frame.has_flag(flag)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this frame to a dictionary.
        
        Returns:
            Dictionary representation of this frame
        """
        return {
            'starting_sample': self.starting_sample,
            'ending_sample': self.ending_sample,
            'data1': self.data1,
            'data2': self.data2,
            'type': self.type,
            'flags': self.flags,
            'duration': self.duration,
            'is_error': self.is_error,
            'is_warning': self.is_warning
        }
    
    def format_data(self, data_field: str = 'data1', display_base: DisplayBase = DisplayBase.HEXADECIMAL,
                   width: int = 0) -> str:
        """
        Format a data field according to the specified display base.
        
        Args:
            data_field: Which data field to format ('data1' or 'data2')
            display_base: Display base to use
            width: Minimum width for the formatted output
            
        Returns:
            Formatted string representation of the data
            
        Raises:
            ValueError: If data_field is not 'data1' or 'data2'
        """
        if data_field == 'data1':
            value = self.data1
        elif data_field == 'data2':
            value = self.data2
        else:
            raise ValueError(f"Invalid data field: {data_field}. Must be 'data1' or 'data2'")
            
        return display_base.format_value(value, width)
        
    def _get_binding_frame(self) -> _bindings.Frame:
        """
        Get the underlying binding frame.
        
        Returns:
            The binding frame object
        """
        return self._frame
        
    def __repr__(self) -> str:
        """Get a string representation of this frame."""
        return (f"Frame(start={self.starting_sample}, end={self.ending_sample}, "
                f"data1=0x{self.data1:x}, data2=0x{self.data2:x}, "
                f"type={self.type}, flags=0x{self.flags:x})")


class FrameCollection(Sequence[Frame]):
    """
    A collection of frames with filtering and transformation capabilities.
    
    Implements Python's Sequence protocol for easy iteration, indexing, and
    integration with standard Python functionality. Provides methods for filtering,
    searching, and analyzing frames.
    
    Examples:
        >>> # Get all frames with type 1
        >>> data_frames = results.frames.filter_by_type(1)
        >>> # Get only error frames
        >>> error_frames = results.frames.errors()
        >>> # Find frames containing a specific value
        >>> address_frames = results.frames.filter(lambda f: f.data1 == 0x42)
    """
    
    def __init__(self, analyzer_results: 'AnalyzerResults', frames: Optional[List[Frame]] = None):
        """
        Initialize a FrameCollection.
        
        Args:
            analyzer_results: The AnalyzerResults object this collection is from
            frames: Optional list of Frame objects to initialize with
        """
        self._analyzer_results = analyzer_results
        self._frames = frames
        
    def __len__(self) -> int:
        """Get the number of frames in this collection."""
        if self._frames is not None:
            return len(self._frames)
        else:
            return self._analyzer_results._results.get_num_frames()
        
    def __getitem__(self, index) -> Union[Frame, 'FrameCollection']:
        """
        Get a frame or slice of frames from this collection.
        
        Args:
            index: Either an integer index or a slice
            
        Returns:
            Either a single Frame object or a new FrameCollection with the sliced frames
            
        Raises:
            IndexError: If the index is out of range
        """
        if isinstance(index, slice):
            # Handle slice
            start, stop, step = index.indices(len(self))
            frames = [self[i] for i in range(start, stop, step)]
            return FrameCollection(self._analyzer_results, frames)
        else:
            # Handle integer index
            if index < 0:
                index = len(self) + index
                
            if index < 0 or index >= len(self):
                raise IndexError(f"Frame index {index} out of range for {len(self)} frames")
            
            if self._frames is not None:
                return self._frames[index]
            else:
                binding_frame = self._analyzer_results._results.get_frame(index)
                return Frame(binding_frame)
    
    def __iter__(self) -> Iterator[Frame]:
        """
        Get an iterator over all frames in this collection.
        
        Returns:
            Iterator yielding Frame objects
        """
        for i in range(len(self)):
            yield self[i]
            
    def filter(self, predicate: Callable[[Frame], bool]) -> 'FrameCollection':
        """
        Filter frames based on a predicate function.
        
        Args:
            predicate: Function that takes a Frame and returns True to include it
            
        Returns:
            New FrameCollection with only frames that match the predicate
            
        Example:
            >>> # Get frames with data1 > 100
            >>> high_value_frames = frames.filter(lambda f: f.data1 > 100)
        """
        filtered_frames = [frame for frame in self if predicate(frame)]
        return FrameCollection(self._analyzer_results, filtered_frames)
    
    def filter_by_type(self, frame_type: int) -> 'FrameCollection':
        """
        Filter frames by type.
        
        Args:
            frame_type: Frame type to filter by
            
        Returns:
            New FrameCollection with only frames of the specified type
            
        Example:
            >>> # Get all SPI command frames (assuming type 2 is for commands)
            >>> command_frames = frames.filter_by_type(2)
        """
        return self.filter(lambda frame: frame.type == frame_type)
    
    def filter_by_types(self, frame_types: List[int]) -> 'FrameCollection':
        """
        Filter frames by multiple types.
        
        Args:
            frame_types: List of frame types to include
            
        Returns:
            New FrameCollection with only frames of the specified types
            
        Example:
            >>> # Get all SPI command and data frames (types 2 and 3)
            >>> spi_frames = frames.filter_by_types([2, 3])
        """
        return self.filter(lambda frame: frame.type in frame_types)
    
    def filter_by_time_range(self, start_sample: int, end_sample: int) -> 'FrameCollection':
        """
        Filter frames by time range.
        
        Args:
            start_sample: Starting sample (inclusive)
            end_sample: Ending sample (inclusive)
            
        Returns:
            New FrameCollection with only frames that overlap the specified range
            
        Example:
            >>> # Get frames in the first 10,000 samples
            >>> initial_frames = frames.filter_by_time_range(0, 10000)
        """
        return self.filter(
            lambda frame: (frame.starting_sample <= end_sample and 
                          frame.ending_sample >= start_sample)
        )
    
    def filter_by_data(self, data_field: str, value: int) -> 'FrameCollection':
        """
        Filter frames by data value.
        
        Args:
            data_field: Which data field to filter on ('data1' or 'data2')
            value: Value to match
            
        Returns:
            New FrameCollection with only frames that match the value
            
        Raises:
            ValueError: If data_field is not 'data1' or 'data2'
            
        Example:
            >>> # Get all frames with data1 = 0x7F
            >>> address_frames = frames.filter_by_data('data1', 0x7F)
        """
        if data_field == 'data1':
            return self.filter(lambda frame: frame.data1 == value)
        elif data_field == 'data2':
            return self.filter(lambda frame: frame.data2 == value)
        else:
            raise ValueError(f"Invalid data field: {data_field}. Must be 'data1' or 'data2'")
    
    def filter_by_data_range(self, data_field: str, min_value: int, max_value: int) -> 'FrameCollection':
        """
        Filter frames by data value range.
        
        Args:
            data_field: Which data field to filter on ('data1' or 'data2')
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            
        Returns:
            New FrameCollection with only frames that match the value range
            
        Raises:
            ValueError: If data_field is not 'data1' or 'data2'
            
        Example:
            >>> # Get all frames with data1 between 0x10 and 0x1F
            >>> command_frames = frames.filter_by_data_range('data1', 0x10, 0x1F)
        """
        if data_field == 'data1':
            return self.filter(lambda frame: min_value <= frame.data1 <= max_value)
        elif data_field == 'data2':
            return self.filter(lambda frame: min_value <= frame.data2 <= max_value)
        else:
            raise ValueError(f"Invalid data field: {data_field}. Must be 'data1' or 'data2'")
    
    def errors(self) -> 'FrameCollection':
        """
        Get frames with the error flag set.
        
        Returns:
            New FrameCollection with only error frames
            
        Example:
            >>> # Get all error frames and print details
            >>> for frame in frames.errors():
            >>>     print(f"Error at sample {frame.starting_sample}: 0x{frame.data1:x}")
        """
        return self.filter(lambda frame: frame.is_error)
    
    def warnings(self) -> 'FrameCollection':
        """
        Get frames with the warning flag set.
        
        Returns:
            New FrameCollection with only warning frames
            
        Example:
            >>> # Count warning frames
            >>> warning_count = len(frames.warnings())
        """
        return self.filter(lambda frame: frame.is_warning)
    
    def count_by_type(self) -> Dict[int, int]:
        """
        Count frames by type.
        
        Returns:
            Dictionary mapping frame type to count
            
        Example:
            >>> # Get distribution of frame types
            >>> type_counts = frames.count_by_type()
            >>> for type_id, count in type_counts.items():
            >>>     print(f"Type {type_id}: {count} frames")
        """
        counts: DefaultDict[int, int] = defaultdict(int)
        for frame in self:
            counts[frame.type] += 1
        return dict(counts)
    
    def get_unique_types(self) -> Set[int]:
        """
        Get the set of unique frame types in this collection.
        
        Returns:
            Set of frame types
            
        Example:
            >>> # Check what frame types are present
            >>> types = frames.get_unique_types()
            >>> print(f"Found frame types: {types}")
        """
        return {frame.type for frame in self}
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert all frames to a list of dictionaries.
        
        Returns:
            List of frame dictionaries
            
        Example:
            >>> # Export frames to JSON
            >>> import json
            >>> with open('frames.json', 'w') as f:
            >>>     json.dump(frames.to_dict_list(), f, indent=2)
        """
        return [frame.to_dict() for frame in self]
    
    def to_dataframe(self) -> Any:
        """
        Convert frames to a pandas DataFrame.
        
        Returns:
            DataFrame containing frame data
        
        Raises:
            ImportError: If pandas is not installed
            
        Example:
            >>> # Analyze frame durations with pandas
            >>> df = frames.to_dataframe()
            >>> print(f"Mean duration: {df['duration'].mean()} samples")
            >>> print(f"Max duration: {df['duration'].max()} samples")
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame conversion")
        
        return pd.DataFrame(self.to_dict_list())
    
    def get_frame_type_names(self, type_names: Dict[int, str]) -> Dict[int, str]:
        """
        Get descriptive names for frame types in this collection.
        
        Args:
            type_names: Dictionary mapping known frame types to descriptive names
            
        Returns:
            Dictionary mapping frame types to names (or default names for unknown types)
            
        Example:
            >>> # Define names for frame types
            >>> type_names = {1: "START", 2: "COMMAND", 3: "DATA", 4: "STOP"}
            >>> # Get names for the types in this collection
            >>> names = frames.get_frame_type_names(type_names)
        """
        result = {}
        for frame_type in self.get_unique_types():
            if frame_type in type_names:
                result[frame_type] = type_names[frame_type]
            else:
                result[frame_type] = f"Type {frame_type}"
        return result
    
    def find_frame_by_sample(self, sample: int) -> Optional[Frame]:
        """
        Find a frame containing the specified sample.
        
        Args:
            sample: Sample number to search for
            
        Returns:
            Frame containing the sample, or None if not found
            
        Example:
            >>> # Find what frame contains sample 12345
            >>> frame = frames.find_frame_by_sample(12345)
            >>> if frame:
            >>>     print(f"Sample 12345 is in {frame}")
            >>> else:
            >>>     print("No frame contains sample 12345")
        """
        for frame in self:
            if frame.starting_sample <= sample <= frame.ending_sample:
                return frame
        return None
        
    def find_nearest_frame(self, sample: int) -> Optional[Frame]:
        """
        Find the frame nearest to a specific sample.
        
        Args:
            sample: Sample number to search near
            
        Returns:
            Nearest frame, or None if the collection is empty
            
        Example:
            >>> # Find frame nearest to sample 50000
            >>> frame = frames.find_nearest_frame(50000)
            >>> if frame:
            >>>     print(f"Nearest frame is at {frame.starting_sample}")
        """
        if not len(self):
            return None
            
        # First check if any frame contains the sample
        frame = self.find_frame_by_sample(sample)
        if frame:
            return frame
            
        min_distance = float('inf')
        nearest_frame = None
        
        for frame in self:
            # Calculate distance to this frame
            if sample < frame.starting_sample:
                distance = frame.starting_sample - sample
            else:  # sample > frame.ending_sample
                distance = sample - frame.ending_sample
                
            if distance < min_distance:
                min_distance = distance
                nearest_frame = frame
                
        return nearest_frame
    
    def get_timing_statistics(self) -> Dict[str, float]:
        """
        Calculate timing statistics for frames in this collection.
        
        Returns:
            Dictionary with timing statistics:
            - min_duration: Minimum frame duration in samples
            - max_duration: Maximum frame duration in samples
            - mean_duration: Mean frame duration in samples
            - total_duration: Sum of all frame durations in samples
            - frame_rate: Average number of frames per sample
            
        Example:
            >>> # Get timing statistics for data frames
            >>> data_frames = frames.filter_by_type(3)
            >>> stats = data_frames.get_timing_statistics()
            >>> print(f"Average data frame duration: {stats['mean_duration']} samples")
        """
        if not len(self):
            return {
                'min_duration': 0,
                'max_duration': 0,
                'mean_duration': 0,
                'total_duration': 0,
                'frame_rate': 0
            }
            
        durations = [frame.duration for frame in self]
        
        if not HAS_NUMPY:
            # Calculate statistics manually
            min_duration = min(durations)
            max_duration = max(durations)
            total_duration = sum(durations)
            mean_duration = total_duration / len(durations)
            
            # Calculate total time span
            frames_list = list(self)
            total_span = frames_list[-1].ending_sample - frames_list[0].starting_sample + 1
            frame_rate = len(self) / total_span if total_span > 0 else 0
        else:
            # Use numpy for statistics
            min_duration = np.min(durations)
            max_duration = np.max(durations)
            mean_duration = np.mean(durations)
            total_duration = np.sum(durations)
            
            # Calculate total time span
            frames_list = list(self)
            total_span = frames_list[-1].ending_sample - frames_list[0].starting_sample + 1
            frame_rate = len(self) / total_span if total_span > 0 else 0
            
        return {
            'min_duration': min_duration,
            'max_duration': max_duration,
            'mean_duration': mean_duration,
            'total_duration': total_duration,
            'frame_rate': frame_rate
        }
    
    def visualize(self, title: str = "Frame Timeline", show_types: bool = True, 
                 show_errors: bool = True, type_names: Optional[Dict[int, str]] = None, 
                 figsize: Tuple[int, int] = (12, 6)) -> Any:
        """
        Create a visualization of frames in this collection.
        
        Args:
            title: Plot title
            show_types: Whether to color-code by frame type
            show_errors: Whether to highlight error frames
            type_names: Optional dictionary mapping frame types to descriptive names
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure if matplotlib is available
            
        Raises:
            ImportError: If matplotlib is not installed
            
        Example:
            >>> # Visualize different frame types with custom names
            >>> type_names = {1: "START", 2: "ADDRESS", 3: "DATA", 4: "STOP"}
            >>> frames.visualize(title="I2C Transaction", type_names=type_names)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
            
        if not len(self):
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No frames to visualize", ha='center', va='center')
            ax.set_title(title)
            return fig
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get type names for the legend
        if type_names is None:
            type_names = {}
            
        # Get all frame types in this collection
        frame_types = self.get_unique_types()
        
        # Create color map for types
        colors = plt.cm.tab10.colors
        type_colors = {t: colors[i % len(colors)] for i, t in enumerate(sorted(frame_types))}
        
        # Draw frames
        for i, frame in enumerate(self):
            # Determine color based on frame type and flags
            if show_errors and frame.is_error:
                color = 'red'
            elif frame.is_warning:
                color = 'orange'
            elif show_types:
                color = type_colors.get(frame.type, 'gray')
            else:
                color = 'blue'
                
            # Draw the frame as a rectangle
            width = frame.duration
            rect = Rectangle((frame.starting_sample, 0), width, 1, 
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add a text label for very wide frames
            if width > (ax.get_xlim()[1] - ax.get_xlim()[0]) / 50:
                # For wider frames, add a text label
                type_name = type_names.get(frame.type, f"Type {frame.type}")
                ax.text(frame.starting_sample + width/2, 0.5, type_name, 
                       ha='center', va='center', color='white', fontsize=8)
        
        # Set up the plot
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Sample Number")
        ax.set_title(title)
        
        # Add legend for frame types
        if show_types and frame_types:
            handles = []
            labels = []
            for t in sorted(frame_types):
                color = type_colors.get(t, 'gray')
                handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                labels.append(type_names.get(t, f"Type {t}"))
                
            if show_errors:
                handles.append(plt.Rectangle((0, 0), 1, 1, color='red'))
                labels.append("Error")
                
            ax.legend(handles, labels, loc='upper right')
            
        plt.tight_layout()
        return fig
    
    def export_to_csv(self, filename: str, display_base: DisplayBase = DisplayBase.HEXADECIMAL,
                    include_flags: bool = True, delimiter: str = ',') -> None:
        """
        Export frames to a CSV file.
        
        Args:
            filename: Output filename
            display_base: Display base for data values
            include_flags: Whether to include flag columns (is_error, is_warning)
            delimiter: Field delimiter character
            
        Example:
            >>> # Export all frames to CSV in hexadecimal format
            >>> frames.export_to_csv("frames.csv", display_base=DisplayBase.HEXADECIMAL)
        """
        with open(filename, 'w', newline='') as f:
            # Determine fields based on include_flags
            fields = ['starting_sample', 'ending_sample', 'duration', 'type', 'data1', 'data2']
            if include_flags:
                fields.extend(['flags', 'is_error', 'is_warning'])
                
            writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            
            for frame in self:
                row = {
                    'starting_sample': frame.starting_sample,
                    'ending_sample': frame.ending_sample,
                    'duration': frame.duration,
                    'type': frame.type,
                    'data1': frame.format_data('data1', display_base),
                    'data2': frame.format_data('data2', display_base)
                }
                
                if include_flags:
                    row.update({
                        'flags': f"0x{frame.flags:x}",
                        'is_error': 'Y' if frame.is_error else 'N',
                        'is_warning': 'Y' if frame.is_warning else 'N'
                    })
                    
                writer.writerow(row)
    
    def find_sequence(self, pattern: List[Dict[str, Any]]) -> List[List[Frame]]:
        """
        Find frame sequences matching a pattern specification.
        
        Args:
            pattern: List of dictionaries, each describing a frame to match, with optional keys:
                - type: Frame type to match
                - data1, data2: Data values to match
                - max_gap: Maximum sample gap to next frame
                
        Returns:
            List of lists, each inner list containing the frames making up one match
            
        Example:
            >>> # Find I2C START followed by ADDRESS writing to 0x42
            >>> pattern = [
            >>>     {'type': 1},  # START
            >>>     {'type': 2, 'data1': 0x42, 'max_gap': 100}  # ADDRESS
            >>> ]
            >>> matches = frames.find_sequence(pattern)
            >>> print(f"Found {len(matches)} occurrences")
        """
        if not pattern:
            return []
            
        # Find all frames that match the first pattern element
        candidates = []
        first_pattern = pattern[0]
        first_type = first_pattern.get('type')
        first_data1 = first_pattern.get('data1')
        first_data2 = first_pattern.get('data2')
        
        for frame in self:
            if ((first_type is None or frame.type == first_type) and
                (first_data1 is None or frame.data1 == first_data1) and
                (first_data2 is None or frame.data2 == first_data2)):
                candidates.append([frame])
                
        # If only one pattern element, return all candidates
        if len(pattern) == 1:
            return candidates
            
        # Extend candidates with matching frames for remaining pattern elements
        results = []
        
        for candidate in candidates:
            current_sequence = candidate
            matches_all = True
            
            for i in range(1, len(pattern)):
                current_frame = current_sequence[-1]
                pattern_element = pattern[i]
                
                # Get constraints for this pattern element
                target_type = pattern_element.get('type')
                target_data1 = pattern_element.get('data1')
                target_data2 = pattern_element.get('data2')
                max_gap = pattern_element.get('max_gap', float('inf'))
                
                # Find next frame that matches constraints
                found_match = False
                
                for frame in self:
                    # Skip frames that occur before the current frame
                    if frame.starting_sample <= current_frame.ending_sample:
                        continue
                        
                    # Check if frame is within max_gap
                    gap = frame.starting_sample - current_frame.ending_sample
                    if gap > max_gap:
                        break
                        
                    # Check if frame matches pattern constraints
                    if ((target_type is None or frame.type == target_type) and
                        (target_data1 is None or frame.data1 == target_data1) and
                        (target_data2 is None or frame.data2 == target_data2)):
                        current_sequence.append(frame)
                        found_match = True
                        break
                        
                if not found_match:
                    matches_all = False
                    break
            
            # If we found matches for all pattern elements, add to results
            if matches_all and len(current_sequence) == len(pattern):
                results.append(current_sequence)
                
        return results


@dataclass
class Marker:
    """
    Represents a marker on a channel.
    
    Markers are visual indicators shown on the channel waveforms to highlight
    important points in the signal or provide annotations.
    
    Attributes:
        sample (int): Sample number where the marker is placed
        type (MarkerType): Type of marker
        channel: Channel the marker is on
    """
    
    sample: int
    type: MarkerType
    channel: Any
        
    def __repr__(self) -> str:
        """Get a string representation of this marker."""
        return f"Marker(sample={self.sample}, type={self.type.name}, channel={self.channel})"


class MarkerCollection(Sequence[Marker]):
    """
    A collection of markers with filtering capabilities.
    
    Implements Python's Sequence protocol for easy iteration, indexing, and
    integration with standard Python functionality.
    
    Examples:
        >>> # Get all dot markers
        >>> dots = markers.filter_by_type(MarkerType.DOT)
        >>> # Get markers in a specific time range
        >>> range_markers = markers.filter_by_time_range(1000, 2000)
    """
    
    def __init__(self, analyzer_results: 'AnalyzerResults', channel, markers: Optional[List[Marker]] = None):
        """
        Initialize a MarkerCollection.
        
        Args:
            analyzer_results: The AnalyzerResults object this collection is from
            channel: Channel these markers are on
            markers: Optional list of Marker objects to initialize with
        """
        self._analyzer_results = analyzer_results
        self._channel = channel
        self._markers = markers
        
    def __len__(self) -> int:
        """Get the number of markers in this collection."""
        if self._markers is not None:
            return len(self._markers)
        else:
            return self._analyzer_results._results.get_num_markers(self._channel)
        
    def __getitem__(self, index) -> Union[Marker, 'MarkerCollection']:
        """
        Get a marker or slice of markers from this collection.
        
        Args:
            index: Either an integer index or a slice
            
        Returns:
            Either a single Marker object or a new MarkerCollection with the sliced markers
            
        Raises:
            IndexError: If the index is out of range
        """
        if isinstance(index, slice):
            # Handle slice
            start, stop, step = index.indices(len(self))
            markers = [self[i] for i in range(start, stop, step)]
            return MarkerCollection(self._analyzer_results, self._channel, markers)
        else:
            # Handle integer index
            if index < 0:
                index = len(self) + index
                
            if index < 0 or index >= len(self):
                raise IndexError(f"Marker index {index} out of range for {len(self)} markers")
            
            if self._markers is not None:
                return self._markers[index]
            else:
                marker_type, sample = self._analyzer_results._results.get_marker(
                    self._channel, index)
                return Marker(sample, MarkerType(marker_type), self._channel)
    
    def __iter__(self) -> Iterator[Marker]:
        """
        Get an iterator over all markers in this collection.
        
        Returns:
            Iterator yielding Marker objects
        """
        for i in range(len(self)):
            yield self[i]
            
    def filter(self, predicate: Callable[[Marker], bool]) -> 'MarkerCollection':
        """
        Filter markers based on a predicate function.
        
        Args:
            predicate: Function that takes a Marker and returns True to include it
            
        Returns:
            New MarkerCollection with only markers that match the predicate
            
        Example:
            >>> # Get markers at samples divisible by 1000
            >>> markers.filter(lambda m: m.sample % 1000 == 0)
        """
        filtered_markers = [marker for marker in self if predicate(marker)]
        return MarkerCollection(self._analyzer_results, self._channel, filtered_markers)
    
    def filter_by_type(self, marker_type: MarkerType) -> 'MarkerCollection':
        """
        Filter markers by type.
        
        Args:
            marker_type: Marker type to filter by
            
        Returns:
            New MarkerCollection with only markers of the specified type
            
        Example:
            >>> # Get all error markers
            >>> error_markers = markers.filter_by_type(MarkerType.ERROR_DOT)
        """
        return self.filter(lambda marker: marker.type == marker_type)
    
    def filter_by_time_range(self, start_sample: int, end_sample: int) -> 'MarkerCollection':
        """
        Filter markers by time range.
        
        Args:
            start_sample: Starting sample (inclusive)
            end_sample: Ending sample (inclusive)
            
        Returns:
            New MarkerCollection with only markers in the specified range
            
        Example:
            >>> # Get markers in the first 10,000 samples
            >>> early_markers = markers.filter_by_time_range(0, 10000)
        """
        return self.filter(
            lambda marker: start_sample <= marker.sample <= end_sample
        )
    
    def count_by_type(self) -> Dict[MarkerType, int]:
        """
        Count markers by type.
        
        Returns:
            Dictionary mapping marker type to count
            
        Example:
            >>> # Count different marker types
            >>> type_counts = markers.count_by_type()
            >>> for marker_type, count in type_counts.items():
            >>>     print(f"{marker_type.name}: {count} markers")
        """
        counts: DefaultDict[MarkerType, int] = defaultdict(int)
        for marker in self:
            counts[marker.type] += 1
        return dict(counts)
    
    def to_dataframe(self) -> Any:
        """
        Convert markers to a pandas DataFrame.
        
        Returns:
            DataFrame containing marker data
        
        Raises:
            ImportError: If pandas is not installed
            
        Example:
            >>> # Convert markers to a DataFrame for analysis
            >>> df = markers.to_dataframe()
            >>> print(df.groupby('type').count())
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame conversion")
        
        data = [
            {
                'sample': marker.sample,
                'type': marker.type.name,
                'channel': str(marker.channel)
            }
            for marker in self
        ]
        
        return pd.DataFrame(data)
    
    def visualize(self, title: str = "Markers", show_types: bool = True,
                 figsize: Tuple[int, int] = (12, 3)) -> Any:
        """
        Create a visualization of markers in this collection.
        
        Args:
            title: Plot title
            show_types: Whether to color-code by marker type
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure if matplotlib is available
            
        Raises:
            ImportError: If matplotlib is not installed
            
        Example:
            >>> # Visualize markers
            >>> markers.visualize(title="Protocol Markers")
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
            
        if not len(self):
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No markers to visualize", ha='center', va='center')
            ax.set_title(title)
            return fig
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get marker types
        marker_types = set(marker.type for marker in self)
        
        # Create mapping for marker styles
        marker_styles = {
            MarkerType.DOT: 'o',
            MarkerType.ERROR_DOT: 'o',
            MarkerType.SQUARE: 's',
            MarkerType.ERROR_SQUARE: 's',
            MarkerType.UP_ARROW: '^',
            MarkerType.DOWN_ARROW: 'v',
            MarkerType.X: 'x',
            MarkerType.ERROR_X: 'x',
            MarkerType.START: '>',
            MarkerType.STOP: '<',
            MarkerType.ONE: '1',
            MarkerType.ZERO: '0'
        }
        
        # Create mapping for marker colors
        marker_colors = {
            MarkerType.DOT: 'blue',
            MarkerType.ERROR_DOT: 'red',
            MarkerType.SQUARE: 'green',
            MarkerType.ERROR_SQUARE: 'red',
            MarkerType.UP_ARROW: 'purple',
            MarkerType.DOWN_ARROW: 'purple',
            MarkerType.X: 'orange',
            MarkerType.ERROR_X: 'red',
            MarkerType.START: 'green',
            MarkerType.STOP: 'red',
            MarkerType.ONE: 'blue',
            MarkerType.ZERO: 'gray'
        }
        
        # Group markers by type
        for marker_type in marker_types:
            markers_of_type = self.filter_by_type(marker_type)
            samples = [marker.sample for marker in markers_of_type]
            
            # Add to plot with appropriate style
            ax.scatter(samples, [0.5] * len(samples),
                     marker=marker_styles.get(marker_type, 'o'),
                     color=marker_colors.get(marker_type, 'blue'),
                     label=marker_type.name,
                     s=100, zorder=10)
                     
        # Add a line to represent the channel
        samples = [marker.sample for marker in self]
        if samples:
            min_sample = min(samples)
            max_sample = max(samples)
            ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.2, 
                      xmin=min_sample, xmax=max_sample)
        
        # Set up the plot
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Sample Number")
        ax.set_title(title)
        
        # Add legend
        if show_types and marker_types:
            ax.legend(loc='upper right')
            
        plt.tight_layout()
        return fig


class PacketCollection(Sequence):
    """
    A collection of packets with filtering capabilities.
    
    A packet is a group of frames that together form a logical unit of communication.
    This class provides methods for accessing and analyzing these packets.
    
    Examples:
        >>> # Iterate through packets
        >>> for packet_frames in results.packets:
        >>>     print(f"Packet with {len(packet_frames)} frames")
        >>> # Get specific packet
        >>> first_packet = results.packets[0]
    """
    
    def __init__(self, analyzer_results: 'AnalyzerResults'):
        """
        Initialize a PacketCollection.
        
        Args:
            analyzer_results: The AnalyzerResults object this collection is from
        """
        self._analyzer_results = analyzer_results
        
    def __len__(self) -> int:
        """Get the number of packets in this collection."""
        return self._analyzer_results._results.get_num_packets()
        
    def __getitem__(self, index) -> Union[FrameCollection, List['FrameCollection']]:
        """
        Get a packet or slice of packets from this collection.
        
        Args:
            index: Either an integer index or a slice
            
        Returns:
            For a single index: FrameCollection containing frames in the packet
            For a slice: List of FrameCollection objects, one for each packet
            
        Raises:
            IndexError: If the index is out of range
        """
        if isinstance(index, slice):
            # Handle slice - create a list of frame collections for each packet
            start, stop, step = index.indices(len(self))
            packets = []
            for i in range(start, stop, step):
                packets.append(self[i])
            return packets
        else:
            # Handle integer index
            if index < 0:
                index = len(self) + index
                
            if index < 0 or index >= len(self):
                raise IndexError(f"Packet index {index} out of range for {len(self)} packets")
                
            first_frame, last_frame = self._analyzer_results._results.get_frames_contained_in_packet(index)
            
            # Get all frames in the packet
            frames = [
                Frame(self._analyzer_results._results.get_frame(i))
                for i in range(first_frame, last_frame + 1)
            ]
            
            return FrameCollection(self._analyzer_results, frames)
    
    def __iter__(self) -> Iterator[FrameCollection]:
        """
        Get an iterator over all packets in this collection.
        
        Returns:
            Iterator yielding FrameCollection objects for each packet
            
        Example:
            >>> # Print details about each packet
            >>> for packet in results.packets:
            >>>     print(f"Packet has {len(packet)} frames, "
            >>>           f"duration: {packet[-1].ending_sample - packet[0].starting_sample} samples")
        """
        for i in range(len(self)):
            yield self[i]
    
    def get_packet_for_frame(self, frame_index: int) -> int:
        """
        Get the packet that contains a frame.
        
        Args:
            frame_index: Index of the frame
            
        Returns:
            Index of the packet containing the frame, or -1 if not found
            
        Example:
            >>> # Find which packet contains frame 42
            >>> packet_idx = results.packets.get_packet_for_frame(42)
            >>> if packet_idx >= 0:
            >>>     print(f"Frame 42 is in packet {packet_idx}")
        """
        packet = self._analyzer_results._results.get_packet_containing_frame_sequential(frame_index)
        if packet == INVALID_RESULT_INDEX:
            return -1
        return packet
    
    def filter(self, predicate: Callable[[FrameCollection], bool]) -> List[FrameCollection]:
        """
        Filter packets based on a predicate function.
        
        Args:
            predicate: Function that takes a FrameCollection and returns True to include it
            
        Returns:
            List of FrameCollection objects that satisfy the predicate
            
        Example:
            >>> # Find packets with more than 5 frames
            >>> large_packets = results.packets.filter(lambda p: len(p) > 5)
        """
        return [packet for packet in self if predicate(packet)]
    
    def filter_by_frame_type(self, frame_type: int) -> List[FrameCollection]:
        """
        Filter packets that contain frames of a specific type.
        
        Args:
            frame_type: Frame type to search for
            
        Returns:
            List of FrameCollection objects representing packets with matching frames
            
        Example:
            >>> # Find packets containing error frames (assuming type 10 is for errors)
            >>> error_packets = results.packets.filter_by_frame_type(10)
        """
        return self.filter(lambda p: any(f.type == frame_type for f in p))
    
    def to_dataframe(self) -> Any:
        """
        Convert packet information to a pandas DataFrame.
        
        Returns:
            DataFrame with packet information
            
        Raises:
            ImportError: If pandas is not installed
            
        Example:
            >>> # Analyze packet sizes
            >>> df = results.packets.to_dataframe()
            >>> print(f"Average frames per packet: {df['frame_count'].mean()}")
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame conversion")
            
        data = []
        for i, packet in enumerate(self):
            # Get frames in this packet
            frames = list(packet)
            
            if not frames:
                continue
                
            # Basic packet info
            first_frame = frames[0]
            last_frame = frames[-1]
            
            packet_info = {
                'packet_id': i,
                'frame_count': len(frames),
                'start_sample': first_frame.starting_sample,
                'end_sample': last_frame.ending_sample,
                'duration': last_frame.ending_sample - first_frame.starting_sample,
                'has_errors': any(f.is_error for f in frames),
                'has_warnings': any(f.is_warning for f in frames)
            }
            
            # Count frame types
            type_counts = packet.count_by_type()
            for frame_type, count in type_counts.items():
                packet_info[f'type_{frame_type}_count'] = count
                
            data.append(packet_info)
            
        return pd.DataFrame(data)


class TransactionCollection(Sequence):
    """
    A collection of transactions.
    
    A transaction is a group of packets that together form a logical unit of communication.
    This class provides methods for accessing and analyzing these transactions.
    
    Examples:
        >>> # Get packets in a transaction
        >>> packets_in_transaction = results.transactions[0]
        >>> print(f"Transaction 0 contains {len(packets_in_transaction)} packets")
    """
    
    def __init__(self, analyzer_results: 'AnalyzerResults'):
        """
        Initialize a TransactionCollection.
        
        Args:
            analyzer_results: The AnalyzerResults object this collection is from
        """
        self._analyzer_results = analyzer_results
        
    def __getitem__(self, transaction_id: int) -> List[int]:
        """
        Get the packets in a transaction.
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            List of packet IDs in the transaction
            
        Example:
            >>> # Get all packets in transaction 0
            >>> packet_ids = results.transactions[0]
            >>> for packet_id in packet_ids:
            >>>     packet = results.packets[packet_id]
            >>>     print(f"Packet {packet_id} has {len(packet)} frames")
        """
        return self._analyzer_results._results.get_packets_contained_in_transaction(transaction_id)
    
    def get_transaction_for_packet(self, packet_id: int) -> int:
        """
        Get the transaction that contains a packet.
        
        Args:
            packet_id: ID of the packet
            
        Returns:
            ID of the transaction containing the packet, or -1 if not found
            
        Example:
            >>> # Find which transaction contains packet 5
            >>> transaction_id = results.transactions.get_transaction_for_packet(5)
            >>> if transaction_id >= 0:
            >>>     print(f"Packet 5 is part of transaction {transaction_id}")
        """
        transaction = self._analyzer_results._results.get_transaction_containing_packet(packet_id)
        return transaction if transaction != 0 else -1
    
    def get_frames_in_transaction(self, transaction_id: int) -> FrameCollection:
        """
        Get all frames in a transaction.
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            FrameCollection containing all frames in the transaction
            
        Example:
            >>> # Get all frames in transaction 1
            >>> frames = results.transactions.get_frames_in_transaction(1)
            >>> print(f"Transaction 1 contains {len(frames)} frames")
        """
        packet_ids = self[transaction_id]
        all_frames = []
        
        for packet_id in packet_ids:
            packet_frames = list(self._analyzer_results.packets[packet_id])
            all_frames.extend(packet_frames)
            
        return FrameCollection(self._analyzer_results, all_frames)
    
    def get_transaction_statistics(self, transaction_id: int) -> Dict[str, Any]:
        """
        Get statistics about a transaction.
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            Dictionary with transaction statistics
            
        Example:
            >>> # Analyze transaction 0
            >>> stats = results.transactions.get_transaction_statistics(0)
            >>> print(f"Transaction 0: {stats['packet_count']} packets, "
            >>>       f"{stats['frame_count']} frames")
        """
        frames = self.get_frames_in_transaction(transaction_id)
        packet_ids = self[transaction_id]
        
        # Calculate basic statistics
        stats = {
            'transaction_id': transaction_id,
            'packet_count': len(packet_ids),
            'frame_count': len(frames),
            'has_errors': any(f.is_error for f in frames),
            'has_warnings': any(f.is_warning for f in frames)
        }
        
        # Add timing information if there are frames
        if len(frames) > 0:
            frames_list = list(frames)
            start_sample = frames_list[0].starting_sample
            end_sample = frames_list[-1].ending_sample
            
            stats.update({
                'start_sample': start_sample,
                'end_sample': end_sample,
                'duration': end_sample - start_sample
            })
            
            # Add frame type counts
            type_counts = frames.count_by_type()
            stats['frame_types'] = type_counts
        
        return stats
    
    def __len__(self) -> int:
        """
        Get the number of transactions.
        
        Note: This is an estimation as there's no direct method to get the count.
        This implementation checks transaction IDs for packets to estimate the count.
        
        Returns:
            Estimated number of transactions
        """
        # Since the SDK doesn't provide a direct way to get transaction count,
        # we need to scan through all packets to find the highest transaction ID
        if not len(self._analyzer_results.packets):
            return 0
            
        max_transaction_id = -1
        for packet_idx in range(len(self._analyzer_results.packets)):
            transaction_id = self.get_transaction_for_packet(packet_idx)
            if transaction_id > max_transaction_id:
                max_transaction_id = transaction_id
                
        # Transaction IDs are 0-based, so add 1 to get the count
        return max_transaction_id + 1 if max_transaction_id >= 0 else 0
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Get an iterator over all transactions.
        
        Returns:
            Iterator yielding list of packet IDs for each transaction
            
        Example:
            >>> # Iterate through all transactions
            >>> for packet_ids in results.transactions:
            >>>     print(f"Transaction with {len(packet_ids)} packets")
        """
        for i in range(len(self)):
            try:
                yield self[i]
            except (IndexError, RuntimeError):
                # Skip invalid transactions
                continue
    
    def to_dataframe(self) -> Any:
        """
        Convert transaction information to a pandas DataFrame.
        
        Returns:
            DataFrame with transaction information
            
        Raises:
            ImportError: If pandas is not installed
            
        Example:
            >>> # Analyze all transactions
            >>> df = results.transactions.to_dataframe()
            >>> print(f"Average packets per transaction: {df['packet_count'].mean()}")
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame conversion")
            
        data = []
        
        # Get statistics for each transaction
        for transaction_id in range(len(self)):
            try:
                stats = self.get_transaction_statistics(transaction_id)
                data.append(stats)
            except (IndexError, RuntimeError):
                continue
                
        if not data:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'transaction_id', 'packet_count', 'frame_count',
                'has_errors', 'has_warnings', 'start_sample',
                'end_sample', 'duration'
            ])
            
        return pd.DataFrame(data)


class ResultExporter:
    """
    Utility for exporting analyzer results to various formats.
    
    This class provides methods for exporting results to different formats,
    including CSV, HTML, and text files, with customizable formatting options.
    
    Examples:
        >>> # Export to CSV
        >>> with ResultExporter(results) as exporter:
        >>>     exporter.export("data.csv", format_type="csv")
        >>>
        >>> # Export to custom HTML
        >>> exporter = ResultExporter(results)
        >>> exporter.export_to_html("data.html", include_errors_only=True)
    """
    
    def __init__(self, analyzer_results: 'AnalyzerResults'):
        """
        Initialize a ResultExporter.
        
        Args:
            analyzer_results: The AnalyzerResults object to export
        """
        self._analyzer_results = analyzer_results
        
    def export(self, filename: str, format_type: Optional[str] = None, 
              display_base: DisplayBase = DisplayBase.HEXADECIMAL) -> None:
        """
        Export results to a file using the analyzer's built-in export functionality.
        
        Args:
            filename: Path to the output file
            format_type: Format to export as (auto-detected from extension if None)
            display_base: Display base for numerical values
            
        Raises:
            ValueError: If the format cannot be determined or is unsupported
            
        Example:
            >>> # Export to CSV in hexadecimal format
            >>> exporter.export("data.csv", format_type="csv", display_base=DisplayBase.HEXADECIMAL)
        """
        if format_type is None:
            # Auto-detect format from filename extension
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.csv':
                format_type = 'csv'
            elif ext == '.txt':
                format_type = 'text'
            elif ext == '.html':
                format_type = 'html'
            else:
                raise ValueError(f"Cannot determine export format from filename: {filename}")
                
        # Use the analyzer's export method with the appropriate format ID
        format_id = 0  # Default format ID
        
        if format_type == 'csv':
            format_id = 0
        elif format_type == 'text':
            format_id = 1
        elif format_type == 'html':
            format_id = 2
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
            
        self._analyzer_results._results.generate_export_file(filename, display_base, format_id)
        
    def export_to_csv(self, filename: str, display_base: DisplayBase = DisplayBase.HEXADECIMAL,
                     include_errors_only: bool = False, include_warnings: bool = True,
                     frame_types: Optional[List[int]] = None,
                     delimiter: str = ',') -> None:
        """
        Export frames to a custom CSV file with advanced filtering options.
        
        Args:
            filename: Path to the output file
            display_base: Display base for numerical values
            include_errors_only: Whether to include only error frames
            include_warnings: Whether to include warning flags in the output
            frame_types: List of frame types to include (None for all)
            delimiter: Field delimiter character
            
        Example:
            >>> # Export only error frames in hexadecimal format
            >>> exporter.export_to_csv("errors.csv", include_errors_only=True)
        """
        # Filter frames if needed
        frames = self._analyzer_results.frames
        
        if include_errors_only:
            frames = frames.errors()
            
        if frame_types is not None:
            frames = frames.filter_by_types(frame_types)
            
        frames.export_to_csv(filename, display_base=display_base, delimiter=delimiter)
        
    def export_to_html(self, filename: str, display_base: DisplayBase = DisplayBase.HEXADECIMAL,
                      include_errors_only: bool = False, include_packet_info: bool = True,
                      frame_types: Optional[List[int]] = None,
                      title: str = "Analyzer Results") -> None:
        """
        Export frames to a custom HTML file with advanced formatting.
        
        Args:
            filename: Path to the output file
            display_base: Display base for numerical values
            include_errors_only: Whether to include only error frames
            include_packet_info: Whether to include packet information
            frame_types: List of frame types to include (None for all)
            title: Title for the HTML document
            
        Example:
            >>> # Export to HTML with packet grouping
            >>> exporter.export_to_html("results.html", title="SPI Analysis")
        """
        # Filter frames if needed
        frames = self._analyzer_results.frames
        
        if include_errors_only:
            frames = frames.errors()
            
        if frame_types is not None:
            frames = frames.filter_by_types(frame_types)
            
        with open(filename, 'w') as f:
            # Write HTML header
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr.error {{ background-color: #ffdddd; }}
        tr.warning {{ background-color: #ffffcc; }}
        tr.packet-header {{ background-color: #e8eaf6; font-weight: bold; }}
        .time {{ font-family: monospace; }}
        .data {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
""")

            # Write summary information
            f.write(f"<h2>Summary</h2>\n")
            f.write(f"<p>Total frames: {len(frames)}</p>\n")
            f.write(f"<p>Error frames: {len(frames.errors())}</p>\n")
            f.write(f"<p>Warning frames: {len(frames.warnings())}</p>\n")
            
            # Write type distribution
            f.write(f"<h3>Frame Type Distribution</h3>\n")
            f.write(f"<table>\n")
            f.write(f"<tr><th>Type</th><th>Count</th></tr>\n")
            
            type_counts = frames.count_by_type()
            for frame_type, count in sorted(type_counts.items()):
                f.write(f"<tr><td>Type {frame_type}</td><td>{count}</td></tr>\n")
                
            f.write(f"</table>\n")
            
            # Write frame data
            f.write(f"<h2>Frame Data</h2>\n")
            
            if include_packet_info and len(self._analyzer_results.packets) > 0:
                # Group frames by packet
                f.write(f"<table>\n")
                f.write(f"<tr><th>Sample</th><th>Duration</th><th>Type</th>"
                       f"<th>Data 1</th><th>Data 2</th></tr>\n")
                
                for packet_idx in range(len(self._analyzer_results.packets)):
                    try:
                        packet = self._analyzer_results.packets[packet_idx]
                        if not len(packet):
                            continue
                            
                        # Write packet header
                        f.write(f"<tr class=\"packet-header\"><td colspan=\"5\">"
                               f"Packet {packet_idx} - {len(packet)} frames</td></tr>\n")
                        
                        # Write packet frames
                        for frame in packet:
                            # Skip if not in our filtered frames
                            if frame_types is not None and frame.type not in frame_types:
                                continue
                                
                            if include_errors_only and not frame.is_error:
                                continue
                                
                            # Determine row class
                            row_class = ""
                            if frame.is_error:
                                row_class = "error"
                            elif frame.is_warning:
                                row_class = "warning"
                                
                            # Format data values
                            data1 = frame.format_data('data1', display_base)
                            data2 = frame.format_data('data2', display_base)
                            
                            f.write(f"<tr class=\"{row_class}\">"
                                   f"<td class=\"time\">{frame.starting_sample}</td>"
                                   f"<td>{frame.duration}</td>"
                                   f"<td>Type {frame.type}</td>"
                                   f"<td class=\"data\">{data1}</td>"
                                   f"<td class=\"data\">{data2}</td>"
                                   f"</tr>\n")
                    except (IndexError, RuntimeError):
                        continue
                
                f.write(f"</table>\n")
            else:
                # Write all frames without packet grouping
                f.write(f"<table>\n")
                f.write(f"<tr><th>Sample</th><th>Duration</th><th>Type</th>"
                       f"<th>Data 1</th><th>Data 2</th></tr>\n")
                
                for frame in frames:
                    # Determine row class
                    row_class = ""
                    if frame.is_error:
                        row_class = "error"
                    elif frame.is_warning:
                        row_class = "warning"
                        
                    # Format data values
                    data1 = frame.format_data('data1', display_base)
                    data2 = frame.format_data('data2', display_base)
                    
                    f.write(f"<tr class=\"{row_class}\">"
                           f"<td class=\"time\">{frame.starting_sample}</td>"
                           f"<td>{frame.duration}</td>"
                           f"<td>Type {frame.type}</td>"
                           f"<td class=\"data\">{data1}</td>"
                           f"<td class=\"data\">{data2}</td>"
                           f"</tr>\n")
                
                f.write(f"</table>\n")
            
            # Write HTML footer
            f.write("""
</body>
</html>
""")
    
    def export_to_dataframe(self) -> Any:
        """
        Export results to a pandas DataFrame.
        
        Returns:
            DataFrame containing frame data
            
        Raises:
            ImportError: If pandas is not installed
            
        Example:
            >>> # Get results as a DataFrame for analysis
            >>> df = exporter.export_to_dataframe()
            >>> # Group by frame type and analyze durations
            >>> df.groupby('type')['duration'].describe()
        """
        return self._analyzer_results.frames.to_dataframe()
    
    def __enter__(self) -> 'ResultExporter':
        """Support for context manager protocol."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up when exiting the context manager."""
        pass


class AnalyzerResults:
    """
    Python wrapper for analyzer results.
    
    This class provides access to frames, packets, transactions, and markers,
    along with methods for filtering, exporting, and visualization. It serves as
    the main entry point for working with analyzer results.
    
    Examples:
        >>> # Assuming we have an analyzer and we've run an analysis
        >>> python_results = create_analyzer_results(analyzer.get_results())
        >>> 
        >>> # Access frames, packets, and transactions
        >>> print(f"Found {len(python_results.frames)} frames in {len(python_results.packets)} packets")
        >>> 
        >>> # Filter and analyze specific frame types
        >>> command_frames = python_results.frames.filter_by_type(COMMAND_FRAME_TYPE)
        >>> print(f"Average command duration: {command_frames.get_timing_statistics()['mean_duration']} samples")
        >>> 
        >>> # Visualize the results
        >>> python_results.visualize_timeline(title="Protocol Analysis")
        >>> 
        >>> # Export to file
        >>> python_results.export("results.csv", display_base=DisplayBase.HEXADECIMAL)
    """
    
    def __init__(self, results: '_bindings.AnalyzerResults'):
        """
        Initialize AnalyzerResults.
        
        Args:
            results: The binding AnalyzerResults object
        """
        self._results = results
        
    @property
    def frames(self) -> FrameCollection:
        """
        Get all frames in these results.
        
        Returns:
            FrameCollection containing all frames
            
        Example:
            >>> # Get all frames and filter by type
            >>> all_frames = results.frames
            >>> command_frames = all_frames.filter_by_type(COMMAND_TYPE)
        """
        return FrameCollection(self)
    
    @property
    def packets(self) -> PacketCollection:
        """
        Get all packets in these results.
        
        Returns:
            PacketCollection containing all packets
            
        Example:
            >>> # Iterate through packets
            >>> for packet in results.packets:
            >>>     print(f"Packet with {len(packet)} frames")
        """
        return PacketCollection(self)
    
    @property
    def transactions(self) -> TransactionCollection:
        """
        Get all transactions in these results.
        
        Returns:
            TransactionCollection for accessing transactions
            
        Example:
            >>> # Get packets in a transaction
            >>> packets_in_transaction = results.transactions[0]
        """
        return TransactionCollection(self)
    
    def get_markers(self, channel) -> MarkerCollection:
        """
        Get markers on a channel.
        
        Args:
            channel: Channel to get markers for
            
        Returns:
            MarkerCollection containing markers on the channel
            
        Example:
            >>> # Get markers on the clock channel
            >>> clock_markers = results.get_markers(clock_channel)
        """
        return MarkerCollection(self, channel)
    
    def add_marker(self, sample: int, marker_type: MarkerType, channel) -> None:
        """
        Add a marker to the results.
        
        Args:
            sample: Sample number to place the marker at
            marker_type: Type of marker to add
            channel: Channel to place the marker on
            
        Example:
            >>> # Mark important points in the analysis
            >>> results.add_marker(start_sample, MarkerType.START, data_channel)
            >>> results.add_marker(end_sample, MarkerType.STOP, data_channel)
        """
        self._results.add_marker(sample, marker_type, channel)
    
    def add_frame(self, frame: Frame) -> int:
        """
        Add a frame to the results.
        
        Args:
            frame: Frame to add
            
        Returns:
            Index of the added frame
            
        Example:
            >>> # Create and add a custom frame
            >>> frame = Frame({'starting_sample': 1000, 'ending_sample': 1100, 
            >>>                'data1': 0x42, 'type': 1})
            >>> frame_idx = results.add_frame(frame)
        """
        return self._results.add_frame(frame._get_binding_frame())
    
    def commit_packet(self) -> int:
        """
        Commit the current packet and start a new one.
        
        Returns:
            ID of the committed packet
            
        Example:
            >>> # Group frames into packets
            >>> for _ in range(3):  # Add 3 frames to packet 1
            >>>     results.add_frame(frame)
            >>> packet_id = results.commit_packet()
            >>> # Now new frames will go to packet 2
        """
        return self._results.commit_packet_and_start_new_packet()
    
    def cancel_packet(self) -> None:
        """
        Cancel the current packet and start a new one.
        
        Example:
            >>> # Discard frames in the current packet
            >>> results.cancel_packet()
        """
        self._results.cancel_packet_and_start_new_packet()
    
    def add_packet_to_transaction(self, transaction_id: int, packet_id: int) -> None:
        """
        Add a packet to a transaction.
        
        Args:
            transaction_id: ID of the transaction
            packet_id: ID of the packet to add
            
        Example:
            >>> # Group packets into a transaction
            >>> results.add_packet_to_transaction(0, packet_id1)
            >>> results.add_packet_to_transaction(0, packet_id2)
        """
        self._results.add_packet_to_transaction(transaction_id, packet_id)
    
    def commit_results(self) -> None:
        """
        Commit the results to make them visible in the UI.
        
        Example:
            >>> # After adding frames, commit to update the UI
            >>> results.commit_results()
        """
        self._results.commit_results()
    
    def get_frames_in_range(self, start_sample: int, end_sample: int) -> FrameCollection:
        """
        Get frames within a sample range.
        
        Args:
            start_sample: First sample in the range (inclusive)
            end_sample: Last sample in the range (inclusive)
            
        Returns:
            FrameCollection containing frames in the range
            
        Example:
            >>> # Get frames in the first 10,000 samples
            >>> early_frames = results.get_frames_in_range(0, 10000)
        """
        found, first_frame, last_frame = self._results.get_frames_in_range(
            start_sample, end_sample)
            
        if not found:
            return FrameCollection(self, [])
            
        # Get all frames in the range
        frames = [
            Frame(self._results.get_frame(i))
            for i in range(first_frame, last_frame + 1)
        ]
        
        return FrameCollection(self, frames)
    
    def export(self, filename: str, format_type: Optional[str] = None, 
              display_base: DisplayBase = DisplayBase.HEXADECIMAL) -> None:
        """
        Export results to a file.
        
        Args:
            filename: Path to the output file
            format_type: Format to export as (auto-detected from extension if None)
            display_base: Display base for numerical values
            
        Example:
            >>> # Export to CSV
            >>> results.export("data.csv")
            >>> # Export to text with decimal values
            >>> results.export("data.txt", format_type="text", display_base=DisplayBase.DECIMAL)
        """
        exporter = ResultExporter(self)
        exporter.export(filename, format_type, display_base)
    
    def to_dataframe(self) -> Any:
        """
        Convert results to a pandas DataFrame.
        
        Returns:
            DataFrame containing frame data
            
        Raises:
            ImportError: If pandas is not installed
            
        Example:
            >>> # Get results as a DataFrame for analysis
            >>> df = results.to_dataframe()
            >>> # Filter and analyze
            >>> error_df = df[df['is_error']]
        """
        return self.frames.to_dataframe()
    
    def visualize_timeline(self, frame_types: Optional[List[int]] = None, 
                         show_errors: bool = True, show_warnings: bool = True, 
                         figsize: Tuple[int, int] = (12, 6), title: str = "Logic Analyzer Timeline",
                         type_names: Optional[Dict[int, str]] = None) -> Any:
        """
        Visualize frames on a timeline.
        
        Creates a visualization showing frame positions and durations along a timeline.
        Different frame types are color-coded, and errors/warnings are highlighted.
        
        Args:
            frame_types: Optional list of frame types to include (None = all)
            show_errors: Whether to highlight error frames
            show_warnings: Whether to highlight warning frames
            figsize: Figure size (width, height) in inches
            title: Plot title
            type_names: Optional dictionary mapping frame types to descriptive names
            
        Returns:
            matplotlib Figure object if matplotlib is installed
            
        Raises:
            ImportError: If matplotlib is not installed
            
        Example:
            >>> # Visualize with custom type names
            >>> type_names = {1: "START", 2: "ADDRESS", 3: "DATA", 4: "STOP"}
            >>> results.visualize_timeline(title="I2C Bus Timeline", type_names=type_names)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
        # Filter frames if needed
        filtered_frames = self.frames
        if frame_types is not None:
            filtered_frames = filtered_frames.filter_by_types(frame_types)
        
        # Get frame types for the legend
        if type_names is None:
            type_names = {}
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if not len(filtered_frames):
            ax.text(0.5, 0.5, "No frames available", ha='center', va='center')
            fig.suptitle(title)
            return fig
        
        # Get all frame types
        frame_types = filtered_frames.get_unique_types()
        
        # Create color map
        colors = plt.cm.tab10.colors
        type_colors = {t: colors[i % len(colors)] for i, t in enumerate(sorted(frame_types))}
        
        # Add frames to the timeline
        y_pos = 0
        y_positions = {}
        
        for frame in filtered_frames:
            # Determine y-position based on frame type for better visualization
            if frame.type not in y_positions:
                y_positions[frame.type] = y_pos
                y_pos += 1
            
            y = y_positions[frame.type]
            
            # Determine color based on frame type and flags
            color = type_colors.get(frame.type, 'gray')
            if show_errors and frame.is_error:
                color = 'red'
            elif show_warnings and frame.is_warning:
                color = 'orange'
            
            # Plot the frame as a horizontal line
            ax.plot(
                [frame.starting_sample, frame.ending_sample],
                [y, y],
                linewidth=3,
                color=color,
                solid_capstyle='butt'
            )
        
        # Add legend for frame types
        handles = []
        labels = []
        for t in sorted(frame_types):
            handles.append(plt.Line2D([0], [0], color=type_colors.get(t, 'gray'), linewidth=3))
            labels.append(type_names.get(t, f"Type {t}"))
        
        if show_errors:
            handles.append(plt.Line2D([0], [0], color='red', linewidth=3))
            labels.append("Error")
        
        if show_warnings:
            handles.append(plt.Line2D([0], [0], color='orange', linewidth=3))
            labels.append("Warning")
        
        ax.legend(handles, labels)
        
        # Set labels and title
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([type_names.get(t, f"Type {t}") for t in y_positions.keys()])
        ax.set_xlabel("Sample Number")
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def visualize_data(self, data_field: str = 'data1', frame_types: Optional[List[int]] = None, 
                      figsize: Tuple[int, int] = (12, 6), 
                      title: Optional[str] = None, 
                      display_base: DisplayBase = DisplayBase.DECIMAL,
                      type_names: Optional[Dict[int, str]] = None) -> Any:
        """
        Visualize frame data values.
        
        Creates a visualization showing data values from frames as a scatter plot,
        with optional color-coding by frame type.
        
        Args:
            data_field: Which data field to visualize ('data1' or 'data2')
            frame_types: Optional list of frame types to include (None = all)
            figsize: Figure size (width, height) in inches
            title: Plot title (defaults to data field name)
            display_base: Display base for axis labels
            type_names: Optional dictionary mapping frame types to descriptive names
            
        Returns:
            matplotlib Figure object if matplotlib is installed
            
        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If data_field is not 'data1' or 'data2'
            
        Example:
            >>> # Visualize data1 values for all frames
            >>> results.visualize_data(data_field='data1', title="Command Values")
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
        if data_field not in ('data1', 'data2'):
            raise ValueError(f"Invalid data field: {data_field}. Must be 'data1' or 'data2'")
        
        # Filter frames if needed
        filtered_frames = self.frames
        if frame_types is not None:
            filtered_frames = filtered_frames.filter_by_types(frame_types)
        
        # Set default title if none provided
        if title is None:
            title = f"{data_field.capitalize()} Values by Frame"
        
        # Get frame types for the legend
        if type_names is None:
            type_names = {}
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if not len(filtered_frames):
            ax.text(0.5, 0.5, "No frames available", ha='center', va='center')
            fig.suptitle(title)
            return fig
        
        # Get data from frames
        samples = []
        values = []
        types = []
        error_markers = []
        
        for frame in filtered_frames:
            # Get middle sample position for this frame
            sample = (frame.starting_sample + frame.ending_sample) // 2
            samples.append(sample)
            
            # Get the requested data value
            if data_field == 'data1':
                values.append(frame.data1)
            else:
                values.append(frame.data2)
                
            types.append(frame.type)
            error_markers.append(frame.is_error)
        
        # Get type to color mapping
        frame_types = set(types)
        colors = plt.cm.tab10.colors
        type_colors = {t: colors[i % len(colors)] for i, t in enumerate(sorted(frame_types))}
        
        # Determine point colors based on frame type
        point_colors = [type_colors.get(t, 'gray') for t in types]
        
        # Create a scatter plot for non-error frames
        non_error_indices = [i for i, is_error in enumerate(error_markers) if not is_error]
        if non_error_indices:
            ax.scatter(
                [samples[i] for i in non_error_indices], 
                [values[i] for i in non_error_indices],
                c=[point_colors[i] for i in non_error_indices], 
                alpha=0.7, s=50
            )
        
        # Create a separate scatter plot for error frames (with 'x' marker)
        error_indices = [i for i, is_error in enumerate(error_markers) if is_error]
        if error_indices:
            ax.scatter(
                [samples[i] for i in error_indices], 
                [values[i] for i in error_indices],
                c='red', marker='x', s=80, 
                label='Error', zorder=10
            )
        
        # Add legend for frame types
        handles = []
        labels = []
        for t in sorted(frame_types):
            handles.append(plt.Line2D([0], [0], marker='o', color=type_colors.get(t, 'gray'), 
                                      linestyle='None', markersize=8))
            labels.append(type_names.get(t, f"Type {t}"))
        
        if error_indices:
            handles.append(plt.Line2D([0], [0], marker='x', color='red', 
                                     linestyle='None', markersize=8))
            labels.append("Error")
        
        ax.legend(handles, labels)
        
        # Format y-axis based on display base
        if display_base == DisplayBase.HEXADECIMAL:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, pos: f"0x{int(x):X}" if x == int(x) else f"{x:.2f}"
            ))
        elif display_base == DisplayBase.BINARY:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, pos: f"0b{int(x):b}" if x == int(x) else f"{x:.2f}"
            ))
        
        # Set labels and title
        ax.set_xlabel("Sample Number")
        ax.set_ylabel(f"Frame {data_field} Value")
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def visualize_type_distribution(self, figsize: Tuple[int, int] = (10, 6), 
                                  title: str = "Frame Type Distribution",
                                  type_names: Optional[Dict[int, str]] = None) -> Any:
        """
        Visualize the distribution of frame types.
        
        Creates a pie chart and bar chart showing the distribution of frame types.
        
        Args:
            figsize: Figure size (width, height) in inches
            title: Plot title
            type_names: Optional dictionary mapping frame types to descriptive names
            
        Returns:
            matplotlib Figure object if matplotlib is installed
            
        Raises:
            ImportError: If matplotlib is not installed
            
        Example:
            >>> # Visualize distribution with custom type names
            >>> type_names = {1: "START", 2: "ADDRESS", 3: "DATA", 4: "STOP"}
            >>> results.visualize_type_distribution(type_names=type_names)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Count frames by type
        type_counts = self.frames.count_by_type()
        
        if not type_names:
            type_names = {}
        
        if not type_counts:
            ax1.text(0.5, 0.5, "No frames available", ha='center', va='center')
            ax2.text(0.5, 0.5, "No frames available", ha='center', va='center')
            fig.suptitle(title)
            return fig
        
        # Prepare data for plotting with labels
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        labels = [type_names.get(t, f"Type {t}") for t in types]
        
        # Create color map
        colors = plt.cm.tab10.colors
        type_colors = [colors[i % len(colors)] for i in range(len(types))]
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(
            counts, 
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=type_colors
        )
        
        # Make percentage labels more readable
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
        
        # Create bar chart
        bars = ax2.bar(labels, counts, color=type_colors)
        
        # Add count labels on the bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.1,
                str(count),
                ha='center',
                va='bottom'
            )
        
        # Set titles
        ax1.set_title("Frame Type Distribution (Percentage)")
        ax2.set_title("Frame Type Counts")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis='x', rotation=45)
        
        fig.suptitle(title)
        plt.tight_layout()
        return fig
    
    def visualize_timing(self, frame_type: Optional[int] = None, metric: str = 'duration', 
                       figsize: Tuple[int, int] = (10, 6), title: Optional[str] = None,
                       bins: int = 30) -> Any:
        """
        Visualize timing metrics for frames.
        
        Creates histograms and statistics for timing-related metrics like
        frame duration or intervals between frames.
        
        Args:
            frame_type: Optional frame type to filter by (None = all types)
            metric: Timing metric to visualize ('duration' or 'interval')
            figsize: Figure size (width, height) in inches
            title: Plot title (defaults to metric name and frame type)
            bins: Number of bins for the histogram
            
        Returns:
            matplotlib Figure object if matplotlib is installed
            
        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If metric is not 'duration' or 'interval'
            
        Example:
            >>> # Visualize durations of data frames (assuming type 3 is for data)
            >>> results.visualize_timing(frame_type=3, metric='duration')
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
            
        if metric not in ('duration', 'interval'):
            raise ValueError(f"Invalid timing metric: {metric}. Must be 'duration' or 'interval'")
        
        # Filter frames if needed
        filtered_frames = self.frames
        if frame_type is not None:
            filtered_frames = filtered_frames.filter_by_type(frame_type)
            type_str = f" (Type {frame_type})"
        else:
            type_str = " (All Types)"
        
        # Get timing data
        if metric == 'duration':
            # Frame durations
            data = [frame.duration for frame in filtered_frames]
            metric_name = "Frame Duration"
        elif metric == 'interval':
            # Intervals between consecutive frames
            frames_list = list(filtered_frames)
            if len(frames_list) < 2:
                data = []
            else:
                data = [
                    frames_list[i+1].starting_sample - frames_list[i].ending_sample
                    for i in range(len(frames_list)-1)
                ]
            metric_name = "Frame Interval"
        
        # Set default title if none provided
        if title is None:
            title = f"{metric_name} Distribution{type_str}"
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                      gridspec_kw={'width_ratios': [2, 1]})
        
        if not data:
            ax1.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax2.text(0.5, 0.5, "No data available", ha='center', va='center')
            fig.suptitle(title)
            return fig
        
        # Determine number of bins (at least 5, at most 50)
        actual_bins = max(5, min(bins, len(data) // 5 + 1, 50))
        
        # Histogram
        n, bins, patches = ax1.hist(data, bins=actual_bins, alpha=0.7, color='steelblue')
        
        # Add a kernel density estimate
        if HAS_NUMPY and len(data) > 5:
            try:
                from scipy import stats
                density = stats.gaussian_kde(data)
                x = np.linspace(min(data), max(data), 100)
                ax1.plot(x, density(x) * len(data) * (bins[1] - bins[0]), 
                       color='darkred', linewidth=2)
            except (ImportError, ValueError):
                # Scipy not available or KDE failed
                pass
        
        ax1.set_xlabel(f"{metric_name} (samples)")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"{metric_name} Histogram")
        
        # Calculate statistics
        if HAS_NUMPY:
            stats = {
                'Min': np.min(data),
                'Max': np.max(data),
                'Mean': np.mean(data),
                'Median': np.median(data),
                'Std Dev': np.std(data)
            }
        else:
            # Calculate statistics manually
            data.sort()
            stats = {
                'Min': min(data),
                'Max': max(data),
                'Mean': sum(data) / len(data),
                'Median': data[len(data) // 2] if len(data) % 2 == 1 
                        else (data[len(data) // 2 - 1] + data[len(data) // 2]) / 2,
                'Std Dev': (sum((x - sum(data) / len(data)) ** 2 for x in data) / len(data)) ** 0.5
            }
        
        # Display statistics as a table
        cell_text = [[f"{v:.2f}"] for v in stats.values()]
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(
            cellText=cell_text,
            rowLabels=list(stats.keys()),
            colLabels=["Value"],
            loc='center'
        )
        
        # Improve table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax2.set_title("Statistics")
        
        fig.suptitle(title)
        plt.tight_layout()
        return fig
    
    def get_statistics(self, frame_types: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for frames.
        
        Args:
            frame_types: Optional list of frame types to include (None = all)
            
        Returns:
            Dictionary with detailed statistics including:
            - Frame counts by type
            - Error and warning counts
            - Timing statistics
            - Data value statistics
            
        Example:
            >>> # Get statistics for all frames
            >>> stats = results.get_statistics()
            >>> print(f"Total frames: {stats['total_frames']}")
            >>> print(f"Error frames: {stats['error_count']}")
            >>> print(f"Average duration: {stats['timing']['mean_duration']} samples")
        """
        # Filter frames if needed
        filtered_frames = self.frames
        if frame_types is not None:
            filtered_frames = filtered_frames.filter_by_types(frame_types)
        
        # Count frames by type
        type_counts = filtered_frames.count_by_type()
        
        # Calculate timing statistics
        frames_list = list(filtered_frames)
        
        # Initialize statistics dictionary
        stats = {
            'total_frames': len(filtered_frames),
            'error_count': 0,
            'warning_count': 0,
            'type_counts': type_counts,
            'timing': {},
            'data_values': {
                'data1': {},
                'data2': {}
            }
        }
        
        if not frames_list:
            return stats
        
        # Count error and warning frames
        stats['error_count'] = len(filtered_frames.errors())
        stats['warning_count'] = len(filtered_frames.warnings())
        
        # Calculate duration statistics
        durations = [frame.duration for frame in frames_list]
        
        if HAS_NUMPY:
            # Use numpy for statistics
            timing_stats = {
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'mean_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'stddev_duration': np.std(durations)
            }
        else:
            # Calculate statistics manually
            durations.sort()
            timing_stats = {
                'min_duration': min(durations),
                'max_duration': max(durations),
                'mean_duration': sum(durations) / len(durations),
                'median_duration': durations[len(durations) // 2] if len(durations) % 2 == 1 
                                else (durations[len(durations) // 2 - 1] + durations[len(durations) // 2]) / 2,
                'stddev_duration': (sum((x - sum(durations) / len(durations)) ** 2 
                                    for x in durations) / len(durations)) ** 0.5
            }
        
        # Calculate interval statistics if there are multiple frames
        if len(frames_list) > 1:
            intervals = [
                frames_list[i+1].starting_sample - frames_list[i].ending_sample
                for i in range(len(frames_list)-1)
            ]
            
            if HAS_NUMPY:
                timing_stats.update({
                    'min_interval': np.min(intervals),
                    'max_interval': np.max(intervals),
                    'mean_interval': np.mean(intervals),
                    'median_interval': np.median(intervals),
                    'stddev_interval': np.std(intervals)
                })
            else:
                intervals.sort()
                timing_stats.update({
                    'min_interval': min(intervals),
                    'max_interval': max(intervals),
                    'mean_interval': sum(intervals) / len(intervals),
                    'median_interval': intervals[len(intervals) // 2] if len(intervals) % 2 == 1 
                                    else (intervals[len(intervals) // 2 - 1] + intervals[len(intervals) // 2]) / 2,
                    'stddev_interval': (sum((x - sum(intervals) / len(intervals)) ** 2 
                                       for x in intervals) / len(intervals)) ** 0.5
                })
        
        stats['timing'] = timing_stats
        
        # Calculate data value statistics
        for data_field in ('data1', 'data2'):
            values = [getattr(frame, data_field) for frame in frames_list]
            
            if HAS_NUMPY:
                data_stats = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'stddev': np.std(values)
                }
                
                # Add distribution information
                unique_values, counts = np.unique(values, return_counts=True)
                most_common_idx = np.argmax(counts)
                data_stats.update({
                    'most_common_value': int(unique_values[most_common_idx]),
                    'most_common_count': int(counts[most_common_idx]),
                    'unique_values': len(unique_values)
                })
            else:
                # Basic statistics without numpy
                values.sort()
                data_stats = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'median': values[len(values) // 2] if len(values) % 2 == 1 
                            else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
                }
                
                # Count frequency of each value
                value_counts = {}
                for v in values:
                    value_counts[v] = value_counts.get(v, 0) + 1
                
                most_common_value = max(value_counts.items(), key=lambda x: x[1])
                data_stats.update({
                    'most_common_value': most_common_value[0],
                    'most_common_count': most_common_value[1],
                    'unique_values': len(value_counts)
                })
            
            stats['data_values'][data_field] = data_stats
        
        return stats
    
    def find_frame_nearest_sample(self, sample: int) -> Optional[Frame]:
        """
        Find the frame nearest to a sample position.
        
        Args:
            sample: Sample position to find frame near
            
        Returns:
            Nearest Frame object, or None if no frames
            
        Example:
            >>> # Find the frame nearest to sample 5000
            >>> frame = results.find_frame_nearest_sample(5000)
            >>> if frame:
            >>>     print(f"Nearest frame at {frame.starting_sample}, type {frame.type}")
        """
        return self.frames.find_nearest_frame(sample)
    
    def find_frames_by_data(self, data_value: int, data_field: str = 'data1') -> FrameCollection:
        """
        Find frames with a specific data value.
        
        Args:
            data_value: Data value to search for
            data_field: Which data field to search ('data1' or 'data2')
            
        Returns:
            FrameCollection containing matching frames
            
        Example:
            >>> # Find all frames with data1 = 0x7F
            >>> address_frames = results.find_frames_by_data(0x7F, 'data1')
        """
        return self.frames.filter_by_data(data_field, data_value)
    
    def find_frames_by_predicate(self, predicate: Callable[[Frame], bool]) -> FrameCollection:
        """
        Find frames that match a predicate function.
        
        Args:
            predicate: Function that takes a Frame and returns True for matches
            
        Returns:
            FrameCollection containing matching frames
            
        Example:
            >>> # Find frames with specific duration
            >>> long_frames = results.find_frames_by_predicate(
            >>>     lambda f: f.duration > 100 and f.type == 3
            >>> )
        """
        return self.frames.filter(predicate)
    
    def find_patterns(self, pattern: List[Dict[str, Any]]) -> List[List[Frame]]:
        """
        Find sequences of frames that match a pattern.
        
        Args:
            pattern: List of dictionaries describing frames to match
                Each dictionary can contain:
                - type: Frame type to match
                - data1, data2: Data values to match
                - max_gap: Maximum samples between this frame and next
                
        Returns:
            List of frame sequences that match the pattern
            
        Example:
            >>> # Find I2C START followed by ADDRESS writing to 0x42
            >>> pattern = [
            >>>     {'type': 1},  # START
            >>>     {'type': 2, 'data1': 0x42, 'max_gap': 100}  # ADDRESS
            >>> ]
            >>> matches = results.find_patterns(pattern)
            >>> print(f"Found {len(matches)} occurrences")
        """
        return self.frames.find_sequence(pattern)
    
    def create_report(self, filename: str, title: str = "Logic Analyzer Results Report",
                    include_visualizations: bool = True, 
                    include_statistics: bool = True) -> None:
        """
        Create a comprehensive HTML report of analysis results.
        
        This report includes statistics, visualizations, and frame data in
        an easy-to-read HTML format.
        
        Args:
            filename: Path to the output HTML file
            title: Report title
            include_visualizations: Whether to include visualizations
            include_statistics: Whether to include statistics
            
        Raises:
            ImportError: If matplotlib is not available and visualizations are requested
            
        Example:
            >>> # Create a full report
            >>> results.create_report("spi_analysis.html", title="SPI Protocol Analysis")
        """
        if include_visualizations and not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualizations")
        
        with open(filename, 'w') as f:
            # Write HTML header with embedded CSS for styling
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr.error {{ background-color: #ffdddd; }}
        tr.warning {{ background-color: #ffffcc; }}
        tr.packet-header {{ background-color: #e8eaf6; font-weight: bold; }}
        .time {{ font-family: monospace; }}
        .data {{ font-family: monospace; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        .stat-box {{ display: inline-block; background-color: #f5f5f5; 
                   border-radius: 5px; margin: 10px; padding: 15px; width: calc(25% - 30px); 
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .stat-box h3 {{ margin-top: 0; }}
        .stat-box .value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .stat-row {{ display: flex; flex-wrap: wrap; margin: 0 -10px; }}
        .error {{ color: #d32f2f; }}
        .warning {{ color: #f57c00; }}
        .summary {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; 
                  margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
""")

            # Add summary statistics in attractive boxes
            if include_statistics:
                stats = self.get_statistics()
                
                f.write(f'<div class="summary">\n')
                f.write(f'<h2>Summary</h2>\n')
                f.write(f'<div class="stat-row">\n')
                
                # Total frames
                f.write(f'<div class="stat-box">\n')
                f.write(f'<h3>Total Frames</h3>\n')
                f.write(f'<div class="value">{stats["total_frames"]}</div>\n')
                f.write(f'</div>\n')
                
                # Frame types
                f.write(f'<div class="stat-box">\n')
                f.write(f'<h3>Frame Types</h3>\n')
                f.write(f'<div class="value">{len(stats["type_counts"])}</div>\n')
                f.write(f'</div>\n')
                
                # Error frames
                error_class = " error" if stats["error_count"] > 0 else ""
                f.write(f'<div class="stat-box">\n')
                f.write(f'<h3>Error Frames</h3>\n')
                f.write(f'<div class="value{error_class}">{stats["error_count"]}</div>\n')
                f.write(f'</div>\n')
                
                # Warning frames
                warning_class = " warning" if stats["warning_count"] > 0 else ""
                f.write(f'<div class="stat-box">\n')
                f.write(f'<h3>Warning Frames</h3>\n')
                f.write(f'<div class="value{warning_class}">{stats["warning_count"]}</div>\n')
                f.write(f'</div>\n')
                
                f.write(f'</div>\n')  # End stat-row
                
                # Add timing statistics if available
                if stats["timing"]:
                    f.write(f'<h3>Timing Statistics</h3>\n')
                    f.write(f'<div class="stat-row">\n')
                    
                    # Duration statistics
                    f.write(f'<div class="stat-box">\n')
                    f.write(f'<h3>Avg Duration</h3>\n')
                    f.write(f'<div class="value">{stats["timing"].get("mean_duration", 0):.2f}</div>\n')
                    f.write(f'<div>samples</div>\n')
                    f.write(f'</div>\n')
                    
                    # Min/Max duration
                    f.write(f'<div class="stat-box">\n')
                    f.write(f'<h3>Min/Max Duration</h3>\n')
                    min_dur = stats["timing"].get("min_duration", 0)
                    max_dur = stats["timing"].get("max_duration", 0)
                    f.write(f'<div class="value">{min_dur} / {max_dur}</div>\n')
                    f.write(f'<div>samples</div>\n')
                    f.write(f'</div>\n')
                    
                    # Add interval statistics if available
                    if "mean_interval" in stats["timing"]:
                        f.write(f'<div class="stat-box">\n')
                        f.write(f'<h3>Avg Interval</h3>\n')
                        f.write(f'<div class="value">{stats["timing"].get("mean_interval", 0):.2f}</div>\n')
                        f.write(f'<div>samples</div>\n')
                        f.write(f'</div>\n')
                        
                        # Min/Max interval
                        f.write(f'<div class="stat-box">\n')
                        f.write(f'<h3>Min/Max Interval</h3>\n')
                        min_int = stats["timing"].get("min_interval", 0)
                        max_int = stats["timing"].get("max_interval", 0)
                        f.write(f'<div class="value">{min_int} / {max_int}</div>\n')
                        f.write(f'<div>samples</div>\n')
                        f.write(f'</div>\n')
                    
                    f.write(f'</div>\n')  # End stat-row
                
                # Add frame type distribution
                if stats["type_counts"]:
                    f.write(f'<h3>Frame Type Distribution</h3>\n')
                    f.write(f'<table>\n')
                    f.write(f'<tr><th>Type</th><th>Count</th><th>Percentage</th></tr>\n')
                    
                    for frame_type, count in sorted(stats["type_counts"].items()):
                        percentage = (count / stats["total_frames"]) * 100 if stats["total_frames"] > 0 else 0
                        f.write(f'<tr><td>Type {frame_type}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>\n')
                        
                    f.write(f'</table>\n')
                
                f.write(f'</div>\n')  # End summary
            
            # Add visualizations if requested
            if include_visualizations and HAS_MATPLOTLIB:
                f.write(f'<h2>Visualizations</h2>\n')
                
                # Timeline visualization
                f.write(f'<div class="chart">\n')
                f.write(f'<h3>Frame Timeline</h3>\n')
                
                # Create the plot and save it to a temporary file
                import tempfile
                import base64
                
                # Frame timeline
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    timeline_fig = self.visualize_timeline(title="Frame Timeline")
                    timeline_fig.savefig(tmp.name, format='png', dpi=100, bbox_inches='tight')
                    plt.close(timeline_fig)
                    
                    # Embed the image directly in the HTML
                    with open(tmp.name, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        f.write(f'<img src="data:image/png;base64,{img_data}" alt="Frame Timeline" style="max-width:100%">\n')
                    
                    # Clean up temporary file
                    import os
                    os.unlink(tmp.name)
                
                f.write(f'</div>\n')
                
                # Frame type distribution
                f.write(f'<div class="chart">\n')
                f.write(f'<h3>Frame Type Distribution</h3>\n')
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    dist_fig = self.visualize_type_distribution(title="Frame Type Distribution")
                    dist_fig.savefig(tmp.name, format='png', dpi=100, bbox_inches='tight')
                    plt.close(dist_fig)
                    
                    # Embed the image
                    with open(tmp.name, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        f.write(f'<img src="data:image/png;base64,{img_data}" alt="Frame Type Distribution" style="max-width:100%">\n')
                    
                    os.unlink(tmp.name)
                
                f.write(f'</div>\n')
                
                # Duration histogram
                f.write(f'<div class="chart">\n')
                f.write(f'<h3>Frame Duration Distribution</h3>\n')
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    dur_fig = self.visualize_timing(metric='duration', title="Frame Duration Distribution")
                    dur_fig.savefig(tmp.name, format='png', dpi=100, bbox_inches='tight')
                    plt.close(dur_fig)
                    
                    # Embed the image
                    with open(tmp.name, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        f.write(f'<img src="data:image/png;base64,{img_data}" alt="Frame Duration Distribution" style="max-width:100%">\n')
                    
                    os.unlink(tmp.name)
                
                f.write(f'</div>\n')
                
                # Data visualization (for data1 only if there are frames)
                if len(self.frames) > 0:
                    f.write(f'<div class="chart">\n')
                    f.write(f'<h3>Data Values</h3>\n')
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        data_fig = self.visualize_data(data_field='data1', title="Data Values by Frame")
                        data_fig.savefig(tmp.name, format='png', dpi=100, bbox_inches='tight')
                        plt.close(data_fig)
                        
                        # Embed the image
                        with open(tmp.name, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            f.write(f'<img src="data:image/png;base64,{img_data}" alt="Data Values" style="max-width:100%">\n')
                        
                        os.unlink(tmp.name)
                    
                    f.write(f'</div>\n')
            
            # Write frame data table
            f.write(f'<h2>Frame Data</h2>\n')
            
            if len(self.frames) > 0:
                f.write(f'<table>\n')
                f.write(f'<tr><th>Index</th><th>Sample Range</th><th>Duration</th><th>Type</th>'
                       f'<th>Data 1</th><th>Data 2</th><th>Flags</th></tr>\n')
                
                for i, frame in enumerate(self.frames):
                    # Determine row class
                    row_class = ""
                    if frame.is_error:
                        row_class = "error"
                    elif frame.is_warning:
                        row_class = "warning"
                        
                    # Format data values in hexadecimal
                    data1 = frame.format_data('data1', DisplayBase.HEXADECIMAL)
                    data2 = frame.format_data('data2', DisplayBase.HEXADECIMAL)
                    
                    # Format flags
                    flag_str = ""
                    if frame.is_error:
                        flag_str += "ERROR "
                    if frame.is_warning:
                        flag_str += "WARNING "
                    if not flag_str:
                        flag_str = "None"
                    
                    f.write(f'<tr class="{row_class}">'
                           f'<td>{i}</td>'
                           f'<td class="time">{frame.starting_sample} - {frame.ending_sample}</td>'
                           f'<td>{frame.duration}</td>'
                           f'<td>Type {frame.type}</td>'
                           f'<td class="data">{data1}</td>'
                           f'<td class="data">{data2}</td>'
                           f'<td>{flag_str}</td>'
                           f'</tr>\n')
                
                f.write(f'</table>\n')
            else:
                f.write(f'<p>No frames available.</p>\n')
            
            # Write HTML footer
            f.write("""
</body>
</html>
""")
    
    def __str__(self) -> str:
        """Get a string representation of these results."""
        frame_count = len(self.frames)
        packet_count = self._results.get_num_packets()
        return f"AnalyzerResults(frames={frame_count}, packets={packet_count})"
    
    def __repr__(self) -> str:
        """Get a detailed string representation of these results."""
        frame_count = len(self.frames)
        packet_count = self._results.get_num_packets()
        error_count = len(self.frames.errors())
        frame_type_counts = self.frames.count_by_type()
        return (f"AnalyzerResults(frames={frame_count}, packets={packet_count}, "
                f"errors={error_count}, frame_types={frame_type_counts})")


class PyAnalyzerResults(_bindings.AnalyzerResults):
    """
    Base class for implementing custom analyzer results in Python.
    
    Inherit from this class and override the required methods to create
    your own analyzer results implementation. This provides a way to create
    custom analyzers that decode specific protocols.
    
    Example:
        >>> class SPIAnalyzerResults(PyAnalyzerResults):
        >>>     # Frame types
        >>>     FRAME_CHIP_SELECT = 0
        >>>     FRAME_DATA = 1
        >>>     
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>     
        >>>     def generate_bubble_text(self, frame_index, channel, display_base):
        >>>         frame = self.get_frame(frame_index)
        >>>         
        >>>         if frame.type == self.FRAME_CHIP_SELECT:
        >>>             if frame.data1 == 0:  # Active
        >>>                 self.add_result_string("CS Active")
        >>>             else:
        >>>                 self.add_result_string("CS Inactive")
        >>>         elif frame.type == self.FRAME_DATA:
        >>>             value = frame.data1
        >>>             if display_base == DisplayBase.ASCII:
        >>>                 self.add_result_string(chr(value))
        >>>             else:
        >>>                 self.add_result_string(hex(value))
    """
    
    def __init__(self):
        """Initialize PyAnalyzerResults."""
        super().__init__()
        self._wrapped = None
    
    def get_wrapped(self) -> AnalyzerResults:
        """
        Get the wrapped AnalyzerResults object.
        
        Returns:
            AnalyzerResults object wrapping this instance
        """
        if self._wrapped is None:
            self._wrapped = AnalyzerResults(self)
        return self._wrapped
    
    # Pure virtual methods that must be implemented
    
    def generate_bubble_text(self, frame_index, channel, display_base):
        """
        Generate bubble text (tooltip) for a frame.
        
        Args:
            frame_index: Index of the frame
            channel: Channel the frame is on
            display_base: Display base to use
            
        MUST be implemented by derived classes.
        """
        raise NotImplementedError("Must implement generate_bubble_text")
    
    def generate_export_file(self, file, display_base, export_type_user_id):
        """
        Generate an export file with the analysis results.
        
        Args:
            file: Path to the output file
            display_base: Display base to use
            export_type_user_id: ID of the export type
            
        MUST be implemented by derived classes.
        """
        raise NotImplementedError("Must implement generate_export_file")
    
    def generate_frame_tabular_text(self, frame_index, display_base):
        """
        Generate tabular text for a frame.
        
        Args:
            frame_index: Index of the frame
            display_base: Display base to use
            
        MUST be implemented by derived classes.
        """
        raise NotImplementedError("Must implement generate_frame_tabular_text")
    
    def generate_packet_tabular_text(self, packet_id, display_base):
        """
        Generate tabular text for a packet.
        
        Args:
            packet_id: ID of the packet
            display_base: Display base to use
            
        MUST be implemented by derived classes.
        """
        raise NotImplementedError("Must implement generate_packet_tabular_text")
    
    def generate_transaction_tabular_text(self, transaction_id, display_base):
        """
        Generate tabular text for a transaction.
        
        Args:
            transaction_id: ID of the transaction
            display_base: Display base to use
            
        MUST be implemented by derived classes.
        """
        raise NotImplementedError("Must implement generate_transaction_tabular_text")


# Helper functions

def create_analyzer_results(results: '_bindings.AnalyzerResults') -> AnalyzerResults:
    """
    Create a Python AnalyzerResults wrapper for a C++ bindings object.
    
    This function is the primary way to convert from the low-level C++ bindings
    to the high-level Python wrapper class.
    
    Args:
        results: C++ bindings AnalyzerResults object
        
    Returns:
        Python AnalyzerResults wrapper
        
    Example:
        >>> from kingst_analyzer import Analyzer
        >>> from kingst_LA.results import create_analyzer_results
        >>> 
        >>> # Run the analyzer
        >>> analyzer = Analyzer(...)
        >>> analyzer.analyze(...)
        >>> 
        >>> # Get the results and create Python wrapper
        >>> cpp_results = analyzer.get_results()
        >>> python_results = create_analyzer_results(cpp_results)
        >>> 
        >>> # Now use the high-level Python API
        >>> for frame in python_results.frames:
        >>>     print(f"Frame: {frame}")
    """
    if isinstance(results, PyAnalyzerResults):
        return results.get_wrapped()
    else:
        return AnalyzerResults(results)