"""
Kingst Logic Analyzer Results Module

This module provides Pythonic classes for accessing and working with Kingst Logic Analyzer results.
It wraps the C++ bindings with intuitive data access patterns, iterators, and filtering capabilities.
"""

from __future__ import annotations
from typing import Iterator, List, Dict, Optional, Union, Tuple, Callable, Any, TypeVar, Sequence
from enum import IntEnum
import os
from pathlib import Path
import warnings

# Import the C++ bindings
import kingst_analyzer._bindings as _bindings

# Optional dependencies for advanced functionality
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Re-export constants
DISPLAY_AS_ERROR_FLAG = _bindings.DISPLAY_AS_ERROR_FLAG
DISPLAY_AS_WARNING_FLAG = _bindings.DISPLAY_AS_WARNING_FLAG
INVALID_RESULT_INDEX = _bindings.INVALID_RESULT_INDEX


class MarkerType(IntEnum):
    """
    Types of markers that can be displayed on channels.
    
    Markers are visual indicators shown on the channel waveforms.
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
    
    Determines how values are formatted in UI and exports.
    """
    ASCII = 0
    DECIMAL = 1
    HEXADECIMAL = 2
    BINARY = 3
    OCTAL = 4


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
    """
    
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
    integration with standard Python functionality.
    """
    
    def __init__(self, analyzer_results, frames=None):
        """
        Initialize a FrameCollection.
        
        Args:
            analyzer_results: The AnalyzerResults object this collection is from
            frames: Optional list of Frame objects to initialize with
        """
        self._analyzer_results = analyzer_results
        self._frames = frames or []
        
    def __len__(self) -> int:
        """Get the number of frames in this collection."""
        if self._frames:
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
            
            if self._frames:
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
        """
        return self.filter(lambda frame: frame.type == frame_type)
    
    def filter_by_time_range(self, start_sample: int, end_sample: int) -> 'FrameCollection':
        """
        Filter frames by time range.
        
        Args:
            start_sample: Starting sample (inclusive)
            end_sample: Ending sample (inclusive)
            
        Returns:
            New FrameCollection with only frames that overlap the specified range
        """
        return self.filter(
            lambda frame: (frame.starting_sample <= end_sample and 
                          frame.ending_sample >= start_sample)
        )
    
    def errors(self) -> 'FrameCollection':
        """
        Get frames with the error flag set.
        
        Returns:
            New FrameCollection with only error frames
        """
        return self.filter(lambda frame: frame.is_error)
    
    def warnings(self) -> 'FrameCollection':
        """
        Get frames with the warning flag set.
        
        Returns:
            New FrameCollection with only warning frames
        """
        return self.filter(lambda frame: frame.is_warning)
    
    def count_by_type(self) -> Dict[int, int]:
        """
        Count frames by type.
        
        Returns:
            Dictionary mapping frame type to count
        """
        counts = {}
        for frame in self:
            counts[frame.type] = counts.get(frame.type, 0) + 1
        return counts
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert all frames to a list of dictionaries.
        
        Returns:
            List of frame dictionaries
        """
        return [frame.to_dict() for frame in self]
    
    def to_dataframe(self) -> Any:
        """
        Convert frames to a pandas DataFrame.
        
        Returns:
            DataFrame containing frame data
        
        Raises:
            ImportError: If pandas is not installed
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for DataFrame conversion")
        
        return pd.DataFrame(self.to_dict_list())


class Marker:
    """
    Represents a marker on a channel.
    
    Markers are visual indicators shown on the channel waveforms.
    
    Attributes:
        sample (int): Sample number where the marker is placed
        type (MarkerType): Type of marker
        channel (Channel): Channel the marker is on
    """
    
    def __init__(self, sample: int, marker_type: MarkerType, channel):
        """
        Initialize a Marker.
        
        Args:
            sample: Sample number where the marker is placed
            marker_type: Type of marker
            channel: Channel the marker is on
        """
        self.sample = sample
        self.type = marker_type
        self.channel = channel
        
    def __repr__(self) -> str:
        """Get a string representation of this marker."""
        return f"Marker(sample={self.sample}, type={self.type.name}, channel={self.channel})"


class MarkerCollection(Sequence[Marker]):
    """
    A collection of markers with filtering capabilities.
    
    Implements Python's Sequence protocol for easy iteration, indexing, and
    integration with standard Python functionality.
    """
    
    def __init__(self, analyzer_results, channel, markers=None):
        """
        Initialize a MarkerCollection.
        
        Args:
            analyzer_results: The AnalyzerResults object this collection is from
            channel: Channel these markers are on
            markers: Optional list of Marker objects to initialize with
        """
        self._analyzer_results = analyzer_results
        self._channel = channel
        self._markers = markers or []
        
    def __len__(self) -> int:
        """Get the number of markers in this collection."""
        if self._markers:
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
            
            if self._markers:
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
        """
        return self.filter(
            lambda marker: start_sample <= marker.sample <= end_sample
        )
    
    def to_dataframe(self) -> Any:
        """
        Convert markers to a pandas DataFrame.
        
        Returns:
            DataFrame containing marker data
        
        Raises:
            ImportError: If pandas is not installed
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


class PacketCollection(Sequence):
    """
    A collection of packets with filtering capabilities.
    
    A packet is a group of frames that together form a logical unit.
    """
    
    def __init__(self, analyzer_results):
        """
        Initialize a PacketCollection.
        
        Args:
            analyzer_results: The AnalyzerResults object this collection is from
        """
        self._analyzer_results = analyzer_results
        
    def __len__(self) -> int:
        """Get the number of packets in this collection."""
        return self._analyzer_results._results.get_num_packets()
        
    def __getitem__(self, index) -> Union[FrameCollection, 'PacketCollection']:
        """
        Get a packet or slice of packets from this collection.
        
        Args:
            index: Either an integer index or a slice
            
        Returns:
            For a single index: FrameCollection containing frames in the packet
            For a slice: New PacketCollection with the sliced packets
        """
        if isinstance(index, slice):
            # Handle slice - create a new PacketCollection with frame collections for each packet
            start, stop, step = index.indices(len(self))
            packets = []
            for i in range(start, stop, step):
                packets.append(self[i])
            return packets
        else:
            # Handle integer index
            if index < 0:
                index = len(self) + index
                
            if index >= len(self):
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
        """
        packet = self._analyzer_results._results.get_packet_containing_frame_sequential(frame_index)
        if packet == INVALID_RESULT_INDEX:
            return -1
        return packet


class TransactionCollection(Sequence):
    """
    A collection of transactions.
    
    A transaction is a group of packets that together form a logical unit.
    """
    
    def __init__(self, analyzer_results):
        """
        Initialize a TransactionCollection.
        
        Args:
            analyzer_results: The AnalyzerResults object this collection is from
        """
        self._analyzer_results = analyzer_results
        
    def __getitem__(self, transaction_id) -> List[int]:
        """
        Get the packets in a transaction.
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            List of packet IDs in the transaction
        """
        return self._analyzer_results._results.get_packets_contained_in_transaction(transaction_id)
    
    def get_transaction_for_packet(self, packet_id: int) -> int:
        """
        Get the transaction that contains a packet.
        
        Args:
            packet_id: ID of the packet
            
        Returns:
            ID of the transaction containing the packet, or -1 if not found
        """
        transaction = self._analyzer_results._results.get_transaction_containing_packet(packet_id)
        return transaction if transaction != 0 else -1


class ResultExporter:
    """
    Utility for exporting analyzer results to various formats.
    
    This class provides methods for exporting results to different formats,
    including CSV, HTML, and text files.
    """
    
    def __init__(self, analyzer_results):
        """
        Initialize a ResultExporter.
        
        Args:
            analyzer_results: The AnalyzerResults object to export
        """
        self._analyzer_results = analyzer_results
        
    def export(self, filename: str, format_type: str = None, display_base: DisplayBase = DisplayBase.HEXADECIMAL):
        """
        Export results to a file.
        
        Args:
            filename: Path to the output file
            format_type: Format to export as (auto-detected from extension if None)
            display_base: Display base for numerical values
            
        Raises:
            ValueError: If the format cannot be determined or is unsupported
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
        
    def export_to_dataframe(self) -> Any:
        """
        Export results to a pandas DataFrame.
        
        Returns:
            DataFrame containing frame data
            
        Raises:
            ImportError: If pandas is not installed
        """
        return self._analyzer_results.frames.to_dataframe()
    
    def __enter__(self):
        """Support for context manager protocol."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting the context manager."""
        pass


class AnalyzerResults:
    """
    Python wrapper for analyzer results.
    
    This class provides access to frames, packets, transactions, and markers,
    along with methods for filtering, exporting, and visualization.
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
        """
        return FrameCollection(self)
    
    @property
    def packets(self) -> PacketCollection:
        """
        Get all packets in these results.
        
        Returns:
            PacketCollection containing all packets
        """
        return PacketCollection(self)
    
    @property
    def transactions(self) -> TransactionCollection:
        """
        Get all transactions in these results.
        
        Returns:
            TransactionCollection for accessing transactions
        """
        return TransactionCollection(self)
    
    def get_markers(self, channel) -> MarkerCollection:
        """
        Get markers on a channel.
        
        Args:
            channel: Channel to get markers for
            
        Returns:
            MarkerCollection containing markers on the channel
        """
        return MarkerCollection(self, channel)
    
    def add_marker(self, sample: int, marker_type: MarkerType, channel) -> None:
        """
        Add a marker to the results.
        
        Args:
            sample: Sample number to place the marker at
            marker_type: Type of marker to add
            channel: Channel to place the marker on
        """
        self._results.add_marker(sample, marker_type, channel)
    
    def add_frame(self, frame: Frame) -> int:
        """
        Add a frame to the results.
        
        Args:
            frame: Frame to add
            
        Returns:
            Index of the added frame
        """
        return self._results.add_frame(frame._get_binding_frame())
    
    def commit_packet(self) -> int:
        """
        Commit the current packet and start a new one.
        
        Returns:
            ID of the committed packet
        """
        return self._results.commit_packet_and_start_new_packet()
    
    def cancel_packet(self) -> None:
        """
        Cancel the current packet and start a new one.
        """
        self._results.cancel_packet_and_start_new_packet()
    
    def add_packet_to_transaction(self, transaction_id: int, packet_id: int) -> None:
        """
        Add a packet to a transaction.
        
        Args:
            transaction_id: ID of the transaction
            packet_id: ID of the packet to add
        """
        self._results.add_packet_to_transaction(transaction_id, packet_id)
    
    def commit_results(self) -> None:
        """
        Commit the results to make them visible in the UI.
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
    
    def export(self, filename: str, format_type: str = None, display_base: DisplayBase = DisplayBase.HEXADECIMAL) -> None:
        """
        Export results to a file.
        
        Args:
            filename: Path to the output file
            format_type: Format to export as (auto-detected from extension if None)
            display_base: Display base for numerical values
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
        """
        return self.frames.to_dataframe()
    
    def visualize_timeline(self, frame_types=None, show_errors=True, show_warnings=True, 
                         figsize=(12, 6), title="Logic Analyzer Timeline"):
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
            
        Returns:
            matplotlib Figure object if matplotlib is installed
            
        Raises:
            ImportError: If matplotlib is not installed
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter frames if needed
        filtered_frames = self.frames
        if frame_types is not None:
            filtered_frames = filtered_frames.filter(
                lambda frame: frame.type in frame_types)
        
        # Get type to color mapping
        frame_types = set(frame.type for frame in filtered_frames)
        colors = plt.cm.tab10.colors
        type_colors = {t: colors[i % len(colors)] for i, t in enumerate(frame_types)}
        
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
        for t, color in type_colors.items():
            handles.append(plt.Line2D([0], [0], color=color, linewidth=3))
            labels.append(f"Type {t}")
        
        if show_errors:
            handles.append(plt.Line2D([0], [0], color='red', linewidth=3))
            labels.append("Error")
        
        if show_warnings:
            handles.append(plt.Line2D([0], [0], color='orange', linewidth=3))
            labels.append("Warning")
        
        ax.legend(handles, labels)
        
        # Set labels and title
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([f"Type {t}" for t in y_positions.keys()])
        ax.set_xlabel("Sample Number")
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def visualize_data(self, data_field='data1', frame_types=None, figsize=(12, 6), 
                      title="Data Values by Frame", display_base=DisplayBase.DECIMAL):
        """
        Visualize frame data values.
        
        Creates a visualization showing data values from frames.
        
        Args:
            data_field: Which data field to visualize ('data1' or 'data2')
            frame_types: Optional list of frame types to include (None = all)
            figsize: Figure size (width, height) in inches
            title: Plot title
            display_base: Display base for data values
            
        Returns:
            matplotlib Figure object if matplotlib is installed
            
        Raises:
            ImportError: If matplotlib is not installed
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter frames if needed
        filtered_frames = self.frames
        if frame_types is not None:
            filtered_frames = filtered_frames.filter(
                lambda frame: frame.type in frame_types)
        
        # Get data from frames
        samples = []
        values = []
        types = []
        
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
        
        # Get type to color mapping
        frame_types = set(types)
        colors = plt.cm.tab10.colors
        type_colors = {t: colors[i % len(colors)] for i, t in enumerate(frame_types)}
        
        # Determine point colors based on frame type
        point_colors = [type_colors.get(t, 'gray') for t in types]
        
        # Plot the data points
        ax.scatter(samples, values, c=point_colors, alpha=0.7)
        
        # Add legend for frame types
        handles = []
        labels = []
        for t, color in type_colors.items():
            handles.append(plt.Line2D([0], [0], marker='o', color=color, 
                                      linestyle='None', markersize=8))
            labels.append(f"Type {t}")
        
        ax.legend(handles, labels)
        
        # Format y-axis based on display base
        if display_base == DisplayBase.HEXADECIMAL:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, pos: f"0x{int(x):X}"))
        elif display_base == DisplayBase.BINARY:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, pos: f"0b{int(x):b}"))
        
        # Set labels and title
        ax.set_xlabel("Sample Number")
        ax.set_ylabel(f"Frame {data_field} Value")
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def visualize_type_distribution(self, figsize=(10, 6), title="Frame Type Distribution"):
        """
        Visualize the distribution of frame types.
        
        Creates a pie chart or bar chart showing the distribution of frame types.
        
        Args:
            figsize: Figure size (width, height) in inches
            title: Plot title
            
        Returns:
            matplotlib Figure object if matplotlib is installed
            
        Raises:
            ImportError: If matplotlib is not installed
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Count frames by type
        type_counts = self.frames.count_by_type()
        
        if not type_counts:
            ax1.text(0.5, 0.5, "No frames available", ha='center', va='center')
            ax2.text(0.5, 0.5, "No frames available", ha='center', va='center')
            return fig
        
        # Prepare data for plotting
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(
            counts, 
            labels=[f"Type {t}" for t in types],
            autopct='%1.1f%%',
            startangle=90
        )
        
        # Create bar chart
        bars = ax2.bar(
            [f"Type {t}" for t in types],
            counts,
            color=plt.cm.tab10.colors[:len(types)]
        )
        
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
    
    def visualize_timing(self, frame_type=None, metric='duration', 
                       figsize=(10, 6), title=None):
        """
        Visualize timing metrics for frames.
        
        Creates histograms and statistics for timing-related metrics like
        frame duration or intervals between frames.
        
        Args:
            frame_type: Optional frame type to filter by (None = all types)
            metric: Timing metric to visualize ('duration' or 'interval')
            figsize: Figure size (width, height) in inches
            title: Plot title (defaults to metric name)
            
        Returns:
            matplotlib Figure object if matplotlib is installed
            
        Raises:
            ImportError: If matplotlib is not installed
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
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
        else:
            raise ValueError(f"Unknown timing metric: {metric}")
        
        # Set default title if none provided
        if title is None:
            title = f"{metric_name} Distribution{type_str}"
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                        gridspec_kw={'width_ratios': [2, 1]})
        
        if not data:
            ax1.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax2.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Histogram
        ax1.hist(data, bins=min(30, len(data)//5 + 1), alpha=0.7, color='steelblue')
        ax1.set_xlabel(f"{metric_name} (samples)")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"{metric_name} Histogram")
        
        # Calculate statistics
        import numpy as np
        stats = {
            'Min': np.min(data),
            'Max': np.max(data),
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Std Dev': np.std(data)
        }
        
        # Display statistics as a table
        cell_text = [[f"{v:.2f}"] for v in stats.values()]
        ax2.axis('tight')
        ax2.axis('off')
        ax2.table(
            cellText=cell_text,
            rowLabels=list(stats.keys()),
            colLabels=["Value"],
            loc='center'
        )
        ax2.set_title("Statistics")
        
        fig.suptitle(title)
        plt.tight_layout()
        return fig
    
    def get_statistics(self, frame_types=None):
        """
        Calculate statistics for frames.
        
        Args:
            frame_types: Optional list of frame types to include (None = all)
            
        Returns:
            Dictionary of statistics
        """
        # Filter frames if needed
        filtered_frames = self.frames
        if frame_types is not None:
            filtered_frames = filtered_frames.filter(
                lambda frame: frame.type in frame_types)
        
        # Count frames by type
        type_counts = filtered_frames.count_by_type()
        
        # Calculate timing statistics
        import numpy as np
        durations = [frame.duration for frame in filtered_frames]
        
        if not durations:
            return {
                'total_frames': 0,
                'type_counts': {},
                'timing': {}
            }
        
        timing_stats = {
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'mean_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'stddev_duration': np.std(durations)
        }
        
        # Calculate interval statistics if there are multiple frames
        frames_list = list(filtered_frames)
        if len(frames_list) > 1:
            intervals = [
                frames_list[i+1].starting_sample - frames_list[i].ending_sample
                for i in range(len(frames_list)-1)
            ]
            
            timing_stats.update({
                'min_interval': np.min(intervals),
                'max_interval': np.max(intervals),
                'mean_interval': np.mean(intervals),
                'median_interval': np.median(intervals),
                'stddev_interval': np.std(intervals)
            })
        
        # Count error and warning frames
        error_count = sum(1 for frame in filtered_frames if frame.is_error)
        warning_count = sum(1 for frame in filtered_frames if frame.is_warning)
        
        return {
            'total_frames': len(filtered_frames),
            'error_count': error_count,
            'warning_count': warning_count,
            'type_counts': type_counts,
            'timing': timing_stats
        }
    
    def find_frame_nearest_sample(self, sample: int) -> Optional[Frame]:
        """
        Find the frame nearest to a sample position.
        
        Args:
            sample: Sample position to find frame near
            
        Returns:
            Nearest Frame object, or None if no frames
        """
        if not len(self.frames):
            return None
        
        min_distance = float('inf')
        nearest_frame = None
        
        for frame in self.frames:
            # Check if the sample is within the frame
            if frame.starting_sample <= sample <= frame.ending_sample:
                return frame
            
            # Calculate distance to the frame
            if sample < frame.starting_sample:
                distance = frame.starting_sample - sample
            else:  # sample > frame.ending_sample
                distance = sample - frame.ending_sample
                
            if distance < min_distance:
                min_distance = distance
                nearest_frame = frame
                
        return nearest_frame
    
    def find_frames_by_data(self, data_value: int, data_field: str = 'data1') -> FrameCollection:
        """
        Find frames with a specific data value.
        
        Args:
            data_value: Data value to search for
            data_field: Which data field to search ('data1' or 'data2')
            
        Returns:
            FrameCollection containing matching frames
        """
        if data_field == 'data1':
            return self.frames.filter(lambda frame: frame.data1 == data_value)
        else:
            return self.frames.filter(lambda frame: frame.data2 == data_value)
    
    def find_frames_by_predicate(self, predicate: Callable[[Frame], bool]) -> FrameCollection:
        """
        Find frames that match a predicate function.
        
        Args:
            predicate: Function that takes a Frame and returns True for matches
            
        Returns:
            FrameCollection containing matching frames
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
        """
        if not pattern:
            return []
            
        # Find all frames that match the first pattern element
        candidates = []
        first_pattern = pattern[0]
        first_type = first_pattern.get('type')
        first_data1 = first_pattern.get('data1')
        first_data2 = first_pattern.get('data2')
        
        for frame in self.frames:
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
                
                for frame in self.frames:
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
                    break
            
            # If we found matches for all pattern elements, add to results
            if len(current_sequence) == len(pattern):
                results.append(current_sequence)
                
        return results
    
    def __str__(self) -> str:
        """Get a string representation of these results."""
        frame_count = len(self.frames)
        packet_count = self._results.get_num_packets()
        return f"AnalyzerResults(frames={frame_count}, packets={packet_count})"
    
    def __repr__(self) -> str:
        """Get a detailed string representation of these results."""
        frame_count = len(self.frames)
        packet_count = self._results.get_num_packets()
        frame_type_counts = self.frames.count_by_type()
        return (f"AnalyzerResults(frames={frame_count}, packets={packet_count}, "
                f"frame_types={frame_type_counts})")


class PyAnalyzerResults(_bindings.AnalyzerResults):
    """
    Base class for implementing custom analyzer results in Python.
    
    Inherit from this class and override the required methods to create
    your own analyzer results implementation.
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
    
    Args:
        results: C++ bindings AnalyzerResults object
        
    Returns:
        Python AnalyzerResults wrapper
    """
    if isinstance(results, PyAnalyzerResults):
        return results.get_wrapped()
    else:
        return AnalyzerResults(results)