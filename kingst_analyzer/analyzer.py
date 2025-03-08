"""
Analyzer Module

This module provides a high-level Pythonic API for the Kingst Logic Analyzer's core Analyzer functionality.
It wraps the lower-level C++ bindings with a more intuitive interface and proper Python idioms.

The Analyzer class represents the main entry point for implementing custom protocol analyzers.
Developers will typically subclass Analyzer to create protocol-specific analyzers (SPI, I2C, UART, etc.).

Example:
    ```python
    from kingst_analyzer.analyzer import Analyzer
    from kingst_analyzer.settings import SPIAnalyzerSettings
    
    class SPIAnalyzer(Analyzer):
        def __init__(self):
            super().__init__()
            self._settings = SPIAnalyzerSettings()
            
        def _get_analyzer_name(self):
            return "SPI Analyzer"
            
        def _get_minimum_sample_rate_hz(self):
            return 10000000  # 10 MHz
            
        def worker_thread(self):
            # Get channel data
            mosi = self.get_channel_data(self._settings.mosi_channel)
            miso = self.get_channel_data(self._settings.miso_channel)
            clock = self.get_channel_data(self._settings.clock_channel)
            enable = self.get_channel_data(self._settings.enable_channel)
            
            # Main analysis loop
            while True:
                # Check if we should exit
                if self.should_abort():
                    break
                    
                # Process one SPI frame
                # ...
                
                # Report progress
                self.report_progress(clock.sample_number)
            
        def needs_rerun(self):
            return False
    ```
"""

from typing import Optional, List, Tuple, Union, Dict, Any, Callable, Type, Iterator, Set, Generic, TypeVar
import abc
from enum import Enum, auto
import contextlib
import warnings
import threading
import time
import weakref

# Import the low-level bindings
from kingst_analyzer import _core as _ka


class AnalyzerState(Enum):
    """
    Enumeration of possible analyzer states.
    
    Attributes:
        IDLE: The analyzer is not running
        INITIALIZING: The analyzer is initializing
        RUNNING: The analyzer is currently running
        PAUSED: The analyzer is paused
        COMPLETED: The analyzer has completed successfully
        STOPPED: The analyzer was stopped by the user
        ERROR: The analyzer encountered an error
    """
    IDLE = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    STOPPED = auto()
    ERROR = auto()


class AnalysisError(Exception):
    """Base exception class for analyzer errors."""
    pass


class AnalyzerInitError(AnalysisError):
    """Exception raised when analyzer initialization fails."""
    pass


class AnalysisAbortedError(AnalysisError):
    """Exception raised when analysis is aborted."""
    pass


class BaseAnalyzer(abc.ABC):
    """
    Abstract base class for all Kingst Logic Analyzer protocol analyzers.
    
    This class defines the interface that all protocol analyzers must implement.
    It provides methods to control the analyzer lifecycle, access channel data,
    and generate analysis results.
    
    Attributes:
        name (str): The name of the analyzer
        version (str): The version of the analyzer
        min_sample_rate (int): Minimum required sample rate in Hz
        settings (Any): The analyzer settings object
        state (AnalyzerState): Current state of the analyzer
        progress (float): Current analysis progress (0.0 to 1.0)
        sample_rate (int): Current sample rate in Hz
        trigger_sample (int): Trigger sample number
    """
    
    def __init__(self):
        """Initialize a new Analyzer instance."""
        self._analyzer = None  # Will hold the C++ analyzer instance
        self._settings = None  # Will hold the settings object
        self._results = None   # Will hold the analyzer results
        self._state = AnalyzerState.IDLE
        self._analysis_thread = None
        self._analysis_exception = None
        self._abort_requested = False
        self._progress_callbacks = set()
        self._state_callbacks = set()
    
    # -------------------------------------------------------------------------
    # Properties and status accessors
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """
        Get the name of the analyzer.
        
        Returns:
            str: The analyzer name
        """
        if self._analyzer:
            return self._analyzer.get_analyzer_name()
        return self._get_analyzer_name()  # Fall back to our implementation
    
    @property
    def version(self) -> str:
        """
        Get the version of the analyzer.
        
        Returns:
            str: The analyzer version
        """
        if self._analyzer:
            return self._analyzer.get_analyzer_version()
        return "1.0.0"  # Default version if not specified
    
    @property
    def min_sample_rate(self) -> int:
        """
        Get the minimum sample rate required for this analyzer in Hz.
        
        Returns:
            int: Minimum sample rate in Hz
        """
        if self._analyzer:
            return self._analyzer.get_minimum_sample_rate_hz()
        return self._get_minimum_sample_rate_hz()  # Fall back to our implementation
    
    @property
    def settings(self) -> Any:
        """
        Get the analyzer settings object.
        
        Returns:
            Any: The settings object
        """
        return self._settings
    
    @settings.setter
    def settings(self, settings: Any) -> None:
        """
        Set the analyzer settings object.
        
        Args:
            settings: The settings object to use
            
        Raises:
            ValueError: If settings is not a valid analyzer settings object
        """
        if not isinstance(settings, _ka.AnalyzerSettings):
            raise ValueError("Settings must be an instance of AnalyzerSettings")
            
        self._settings = settings
        if self._analyzer:
            self._analyzer.set_analyzer_settings(settings)
    
    @property
    def state(self) -> AnalyzerState:
        """
        Get the current state of the analyzer.
        
        Returns:
            AnalyzerState: Current analyzer state
        """
        return self._state
    
    @property
    def sample_rate(self) -> int:
        """
        Get the current sample rate in Hz.
        
        Returns:
            int: Sample rate in Hz, or 0 if not available
        """
        if self._analyzer:
            return self._analyzer.get_sample_rate()
        return 0
    
    @property
    def trigger_sample(self) -> int:
        """
        Get the trigger sample number.
        
        Returns:
            int: Trigger sample number, or 0 if not available
        """
        if self._analyzer:
            return self._analyzer.get_trigger_sample()
        return 0
    
    @property
    def progress(self) -> float:
        """
        Get the current analyzer progress as a value between 0.0 and 1.0.
        
        Returns:
            float: Analysis progress (0.0 to 1.0)
        """
        if self._analyzer:
            return self._analyzer.get_analyzer_progress()
        return 0.0
    
    @property
    def results(self) -> Optional["_ka.AnalyzerResults"]:
        """
        Get the analyzer results.
        
        Returns:
            AnalyzerResults: Results object, or None if not available
        """
        if self._analyzer:
            success, results = self._analyzer._get_analyzer_results()
            if success:
                return results
        return None
    
    # -------------------------------------------------------------------------
    # Analysis control methods
    # -------------------------------------------------------------------------
    
    def start_analysis(self, starting_sample: int = 0, async_mode: bool = False) -> None:
        """
        Start the analysis process.
        
        Args:
            starting_sample: Sample number to start processing from (default: 0)
            async_mode: Whether to run analysis asynchronously (default: False)
            
        Raises:
            RuntimeError: If the analyzer is already running or not properly initialized
            
        Note:
            If async_mode is True, this method will return immediately and analysis
            will run in a background thread. You can check the status using the 
            state property or register callbacks using add_progress_callback() and
            add_state_callback().
            
            If async_mode is False, this method will block until analysis is complete.
        """
        if self._state == AnalyzerState.RUNNING:
            raise RuntimeError("Analyzer is already running")
        
        if not self._analyzer:
            self._initialize_analyzer()
        
        # Reset state
        self._abort_requested = False
        self._analysis_exception = None
        self._set_state(AnalyzerState.INITIALIZING)
        
        if async_mode:
            # Start analysis in a separate thread
            self._analysis_thread = threading.Thread(
                target=self._run_analysis_thread,
                args=(starting_sample,),
                daemon=True
            )
            self._analysis_thread.start()
        else:
            # Run analysis in the current thread
            try:
                self._run_analysis(starting_sample)
            except Exception as e:
                self._set_state(AnalyzerState.ERROR)
                raise e
    
    def _run_analysis_thread(self, starting_sample: int) -> None:
        """
        Run analysis in a separate thread.
        
        Args:
            starting_sample: Sample number to start processing from
        """
        try:
            self._run_analysis(starting_sample)
        except Exception as e:
            self._analysis_exception = e
            self._set_state(AnalyzerState.ERROR)
    
    def _run_analysis(self, starting_sample: int) -> None:
        """
        Run the analysis process.
        
        Args:
            starting_sample: Sample number to start processing from
        """
        try:
            self.setup()
            
            if starting_sample > 0:
                self._analyzer.start_processing_from(starting_sample)
            else:
                self._analyzer.start_processing()
                
            self._set_state(AnalyzerState.RUNNING)
            
            # This will block until analysis is complete
            self.worker_thread()
            
            # Check if we need to rerun
            if self.needs_rerun():
                self._run_analysis(starting_sample)
            else:
                self._set_state(AnalyzerState.COMPLETED)
        except AnalysisAbortedError:
            self._set_state(AnalyzerState.STOPPED)
        except Exception as e:
            self._set_state(AnalyzerState.ERROR)
            raise e
    
    def stop_analysis(self) -> None:
        """
        Stop the analysis process.
        
        Raises:
            RuntimeError: If the analyzer is not running
        """
        if self._state != AnalyzerState.RUNNING:
            raise RuntimeError("Analyzer is not running")
        
        self._abort_requested = True
        
        if self._analyzer:
            self._analyzer.stop_worker_thread()
            
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=1.0)
            
        self._set_state(AnalyzerState.STOPPED)
    
    def pause_analysis(self) -> None:
        """
        Pause the analysis process.
        
        This method is not yet implemented.
        
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Pause functionality is not yet implemented")
    
    def resume_analysis(self) -> None:
        """
        Resume a paused analysis process.
        
        This method is not yet implemented.
        
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Resume functionality is not yet implemented")
    
    def setup(self) -> None:
        """
        Set up the analyzer before running.
        
        This is called automatically before starting the analysis.
        It initializes the results object and performs any other needed setup.
        """
        if self._analyzer:
            self._analyzer.setup_results()
    
    def should_abort(self) -> bool:
        """
        Check if the analysis should be aborted.
        
        Returns:
            bool: True if the analysis should be aborted
            
        Raises:
            AnalysisAbortedError: If abort is requested
        """
        if self._abort_requested:
            raise AnalysisAbortedError("Analysis aborted by user")
            
        if not self._analyzer:
            return False
            
        try:
            self._analyzer.check_if_thread_should_exit()
            return False
        except Exception:
            raise AnalysisAbortedError("Analysis aborted by system")
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for analysis to complete.
        
        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            
        Returns:
            bool: True if analysis completed, False if timeout occurred
            
        Raises:
            RuntimeError: If analysis was never started
            Exception: Any exception raised during analysis
        """
        if not self._analysis_thread:
            raise RuntimeError("Analysis was never started")
            
        self._analysis_thread.join(timeout=timeout)
        completed = not self._analysis_thread.is_alive()
        
        if completed and self._analysis_exception:
            raise self._analysis_exception
            
        return completed
    
    def add_progress_callback(self, callback: Callable[[float], None]) -> None:
        """
        Add a callback to be called when progress is reported.
        
        Args:
            callback: Function taking a progress value (0.0 to 1.0)
            
        Example:
            ```python
            def on_progress(progress):
                print(f"Analysis progress: {progress:.1%}")
                
            analyzer.add_progress_callback(on_progress)
            ```
        """
        self._progress_callbacks.add(callback)
    
    def remove_progress_callback(self, callback: Callable[[float], None]) -> None:
        """
        Remove a previously added progress callback.
        
        Args:
            callback: The callback function to remove
        """
        self._progress_callbacks.discard(callback)
    
    def add_state_callback(self, callback: Callable[[AnalyzerState], None]) -> None:
        """
        Add a callback to be called when analyzer state changes.
        
        Args:
            callback: Function taking an AnalyzerState value
            
        Example:
            ```python
            def on_state_change(state):
                print(f"Analyzer state changed to: {state.name}")
                
            analyzer.add_state_callback(on_state_change)
            ```
        """
        self._state_callbacks.add(callback)
    
    def remove_state_callback(self, callback: Callable[[AnalyzerState], None]) -> None:
        """
        Remove a previously added state callback.
        
        Args:
            callback: The callback function to remove
        """
        self._state_callbacks.discard(callback)
    
    # -------------------------------------------------------------------------
    # Analysis utility methods
    # -------------------------------------------------------------------------
    
    def get_channel_data(self, channel: "_ka.Channel") -> "_ka.AnalyzerChannelData":
        """
        Get the analyzer channel data for a specific channel.
        
        Args:
            channel: The channel to get data for
            
        Returns:
            AnalyzerChannelData: The channel data object
            
        Raises:
            RuntimeError: If the analyzer is not properly initialized
        """
        if not self._analyzer:
            raise RuntimeError("Analyzer not initialized")
            
        return self._analyzer.get_analyzer_channel_data(channel)
    
    def report_progress(self, sample_number: int) -> None:
        """
        Report progress to the UI and any registered callbacks.
        
        Call this periodically during analysis to update the progress indicator.
        
        Args:
            sample_number: Current sample number being processed
        """
        if self._analyzer:
            self._analyzer.report_progress(sample_number)
            
            # Calculate progress and notify callbacks
            progress = self.progress
            for callback in list(self._progress_callbacks):
                try:
                    callback(progress)
                except Exception as e:
                    warnings.warn(f"Progress callback raised exception: {e}")
    
    def reset(self) -> None:
        """
        Reset the analyzer to its initial state.
        
        This method clears all results and resets the analyzer state.
        """
        if self._analyzer:
            # There's no direct "reset" method in the C++ API, so we'll
            # re-initialize the analyzer
            self._initialize_analyzer()
            
        self._set_state(AnalyzerState.IDLE)
        self._abort_requested = False
        self._analysis_exception = None
        self._analysis_thread = None
    
    # -------------------------------------------------------------------------
    # Context management and simulation methods
    # -------------------------------------------------------------------------
    
    @contextlib.contextmanager
    def analysis_session(self, starting_sample: int = 0):
        """
        Context manager for handling analyzer lifecycle.
        
        This provides a convenient way to ensure proper cleanup of analyzer resources.
        
        Args:
            starting_sample: Sample number to start processing from (default: 0)
            
        Yields:
            The analyzer instance
            
        Example:
            ```python
            with analyzer.analysis_session() as session:
                # Analysis is running
                # Do something with the session...
            # Analysis is automatically stopped when the context exits
            ```
        """
        try:
            self.start_analysis(starting_sample, async_mode=True)
            yield self
        finally:
            if self._state == AnalyzerState.RUNNING:
                self.stop_analysis()
    
    def generate_simulation_data(self, sample_rate: int = 0) -> None:
        """
        Generate simulation data for testing.
        
        This is a convenience method for generating simulation data.
        Derived classes should implement _generate_simulation_data to
        provide custom simulation data.
        
        Args:
            sample_rate: Sample rate to use for simulation (default: min_sample_rate)
            
        Raises:
            NotImplementedError: If _generate_simulation_data is not implemented
        """
        if sample_rate <= 0:
            sample_rate = self.min_sample_rate
            
        if not self._analyzer:
            self._initialize_analyzer()
            
        # Delegate to the implementation method
        self._generate_simulation_data(sample_rate)
    
    # -------------------------------------------------------------------------
    # Batch analysis methods
    # -------------------------------------------------------------------------
    
    def analyze_file(self, file_path: str, **options) -> "_ka.AnalyzerResults":
        """
        Analyze data from a file.
        
        This is a convenience method for quickly analyzing data from a file.
        
        Args:
            file_path: Path to the data file
            **options: Additional options for analysis
            
        Returns:
            AnalyzerResults: Results of the analysis
            
        Raises:
            NotImplementedError: This method is not yet implemented
            
        Note: 
            This functionality depends on the Kingst SDK's file loading capabilities.
        """
        raise NotImplementedError("File analysis is not yet implemented")
    
    def analyze_batch(self, file_paths: List[str], **options) -> List["_ka.AnalyzerResults"]:
        """
        Batch analyze multiple files.
        
        This is a convenience method for analyzing multiple files in sequence.
        
        Args:
            file_paths: List of paths to data files
            **options: Additional options for analysis
            
        Returns:
            List[AnalyzerResults]: Results for each file
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Batch analysis is not yet implemented")
    
    # -------------------------------------------------------------------------
    # Internal implementation methods
    # -------------------------------------------------------------------------
    
    def _initialize_analyzer(self) -> None:
        """
        Initialize the C++ analyzer instance.
        
        This is called automatically when needed, but can also be called
        manually to pre-initialize the analyzer.
        
        Raises:
            AnalyzerInitError: If analyzer initialization fails
        """
        # This will be implemented in derived classes
        pass
    
    def _set_state(self, state: AnalyzerState) -> None:
        """
        Set the analyzer state and notify any registered callbacks.
        
        Args:
            state: New analyzer state
        """
        if self._state != state:
            self._state = state
            
            # Notify callbacks
            for callback in list(self._state_callbacks):
                try:
                    callback(state)
                except Exception as e:
                    warnings.warn(f"State callback raised exception: {e}")
    
    # -------------------------------------------------------------------------
    # Abstract methods that must be implemented by derived classes
    # -------------------------------------------------------------------------
    
    @abc.abstractmethod
    def _get_analyzer_name(self) -> str:
        """
        Get the name of the analyzer (implementation method).
        
        Returns:
            str: Analyzer name
            
        This method must be implemented by derived classes.
        """
        pass
    
    @abc.abstractmethod
    def _get_minimum_sample_rate_hz(self) -> int:
        """
        Get the minimum sample rate required for this analyzer in Hz (implementation method).
        
        Returns:
            int: Minimum sample rate in Hz
            
        This method must be implemented by derived classes.
        """
        pass
    
    @abc.abstractmethod
    def worker_thread(self) -> None:
        """
        Main worker thread for the analyzer.
        
        This method contains the main analysis logic and is called when the analyzer is run.
        It should process the channel data, detect protocol features, and add frames to the results.
        
        This method must be implemented by derived classes.
        
        Raises:
            AnalysisAbortedError: If analysis is aborted
        """
        pass
    
    @abc.abstractmethod
    def needs_rerun(self) -> bool:
        """
        Check if the analyzer needs to be rerun.
        
        Returns:
            bool: True if the analyzer needs to be rerun
            
        This method must be implemented by derived classes.
        """
        pass
    
    def _generate_simulation_data(self, sample_rate: int) -> None:
        """
        Generate simulation data for testing (implementation method).
        
        Args:
            sample_rate: Sample rate to use for simulation
            
        This method can be implemented by derived classes to provide custom
        simulation data. The default implementation raises NotImplementedError.
        
        Raises:
            NotImplementedError: If not implemented by derived class
        """
        raise NotImplementedError(
            "Simulation data generation is not implemented. "
            "Override _generate_simulation_data in your analyzer class."
        )


class Analyzer(BaseAnalyzer):
    """
    Concrete implementation of a Kingst Logic Analyzer protocol analyzer.
    
    This class extends BaseAnalyzer to provide a concrete implementation
    that directly wraps the C++ Analyzer class.
    
    Example:
        ```python
        from kingst_analyzer.analyzer import Analyzer
        
        class SPIAnalyzer(Analyzer):
            def __init__(self):
                super().__init__()
                # Configure channels and settings
                
            def _get_analyzer_name(self):
                return "SPI Analyzer"
                
            def _get_minimum_sample_rate_hz(self):
                return 10000000  # 10 MHz
                
            def worker_thread(self):
                # Get channel data
                mosi = self.get_channel_data(self.settings.mosi_channel)
                miso = self.get_channel_data(self.settings.miso_channel)
                clock = self.get_channel_data(self.settings.clock_channel)
                
                # Process the data
                while True:
                    if self.should_abort():
                        break
                        
                    # Process one SPI transaction
                    # ...
                    
                    # Report progress
                    self.report_progress(clock.sample_number)
                
            def needs_rerun(self):
                return False
        ```
    """
    
    class PyAnalyzerImpl(_ka.Analyzer):
        """Inner class that implements the C++ Analyzer interface."""
        
        def __init__(self, outer):
            """
            Initialize a PyAnalyzerImpl instance.
            
            Args:
                outer: The outer Analyzer instance
            """
            super().__init__()
            self.outer = weakref.ref(outer)  # Use weakref to avoid circular references
        
        def worker_thread(self):
            """
            Delegate to the outer class's worker_thread method.
            
            This method is called by the C++ code when the analyzer is run.
            """
            outer = self.outer()
            if outer is None:
                warnings.warn("Outer analyzer object has been garbage collected")
                return
            outer.worker_thread()
        
        def generate_simulation_data(self, newest_sample_requested, sample_rate, simulation_channels):
            """
            Implementation of the C++ GenerateSimulationData method.
            
            This method is not meant to be called directly by Python code.
            Python code should use _generate_simulation_data instead.
            """
            # Return sample rate as a simple implementation 
            # that doesn't try to use the simulation_channels parameter
            return sample_rate
        
        def get_analyzer_name(self):
            """
            Delegate to the outer class's _get_analyzer_name method.
            
            Returns:
                str: Analyzer name
            """
            outer = self.outer()
            if outer is None:
                return "Unknown Analyzer (Outer object deleted)"
            return outer._get_analyzer_name()
        
        def get_minimum_sample_rate_hz(self):
            """
            Delegate to the outer class's _get_minimum_sample_rate_hz method.
            
            Returns:
                int: Minimum sample rate in Hz
            """
            outer = self.outer()
            if outer is None:
                return 1000000  # Default to 1 MHz if outer is gone
            return outer._get_minimum_sample_rate_hz()
        
        def needs_rerun(self):
            """
            Delegate to the outer class's needs_rerun method.
            
            Returns:
                bool: True if the analyzer needs to be rerun
            """
            outer = self.outer()
            if outer is None:
                return False
            return outer.needs_rerun()
        
        def setup_results(self):
            """
            Delegate to the outer class's setup method.
            
            This allows Python code to override the default setup_results method.
            """
            outer = self.outer()
            if outer is None:
                # Call the parent implementation if outer is gone
                super().setup_results()
                return
                
            # Call the outer's setup method
            outer.setup()
    
    def __init__(self):
        """Initialize a new Analyzer instance."""
        super().__init__()
        self._initialize_analyzer()
    
    def _initialize_analyzer(self) -> None:
        """
        Initialize the C++ analyzer instance.
        
        This creates a new PyAnalyzerImpl instance that delegates to this
        Analyzer instance. This allows Python subclasses to override the
        virtual methods of the C++ Analyzer class.
        
        Raises:
            AnalyzerInitError: If analyzer initialization fails
        """
        try:
            self._analyzer = self.PyAnalyzerImpl(self)
            
            # Set up settings if available
            if self._settings:
                self._analyzer.set_analyzer_settings(self._settings)
        except Exception as e:
            raise AnalyzerInitError(f"Failed to initialize analyzer: {e}")

class ProtocolAnalyzer(Analyzer):
    """
    Base class for protocol-specific analyzers.
    
    This class adds common functionality for protocol analyzers,
    such as frame decoding and visualization.
    
    Attributes:
        frame_types (Dict[int, str]): Mapping of frame type values to names
    """
    
    def __init__(self):
        """Initialize a new ProtocolAnalyzer instance."""
        super().__init__()
        self.frame_types = {}  # Subclasses should override this
    
    def get_frame_type_name(self, frame_type: int) -> str:
        """
        Get the name of a frame type.
        
        Args:
            frame_type: Frame type value
            
        Returns:
            str: Frame type name, or "Unknown" if not found
        """
        return self.frame_types.get(frame_type, f"Unknown ({frame_type})")
    
    def decode_frame(self, frame: "_ka.Frame") -> Dict[str, Any]:
        """
        Decode a frame into a dictionary of values.
        
        Args:
            frame: Frame to decode
            
        Returns:
            Dict[str, Any]: Decoded values
            
        Raises:
            NotImplementedError: This method must be implemented by derived classes
        """
        raise NotImplementedError(
            "Frame decoding is not implemented. "
            "Override decode_frame in your analyzer class."
        )
    
    def find_frames(self, frame_type: Optional[int] = None, 
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None,
                    max_frames: int = 1000) -> Iterator["_ka.Frame"]:
        """
        Find frames matching the specified criteria.
        
        Args:
            frame_type: Specific frame type to find, or None for all types
            start_time: Start time in seconds, or None for beginning of data
            end_time: End time in seconds, or None for end of data
            max_frames: Maximum number of frames to find
            
        Yields:
            Frame objects matching the criteria
        """
        # Convert time to samples if provided
        start_sample = int(start_time * self.sample_rate) if start_time is not None else None
        end_sample = int(end_time * self.sample_rate) if end_time is not None else None
        
        # Get results
        results = self.results
        if not results:
            return
            
        # Use frame_index_iterator if available, otherwise fallback to manual iteration
        try:
            # Try to use the optimized iterator
            iterator = results.get_frames(frame_type, start_sample, end_sample, max_frames)
            yield from iterator
        except (AttributeError, NotImplementedError):
            # Fall back to manual iteration
            frame_count = 0
            frame = results.get_first_frame()
            
            while frame and frame_count < max_frames:
                # Check frame type if specified
                if frame_type is not None and frame.type != frame_type:
                    frame = results.get_next_frame()
                    continue
                    
                # Check sample range if specified
                sample = frame.sample
                if start_sample is not None and sample < start_sample:
                    frame = results.get_next_frame()
                    continue
                    
                if end_sample is not None and sample > end_sample:
                    break
                
                yield frame
                frame_count += 1
                
                frame = results.get_next_frame()
    
    def decode_all_frames(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Decode all frames matching the specified criteria.
        
        Args:
            **kwargs: Arguments to pass to find_frames()
            
        Yields:
            Dict[str, Any]: Decoded frame data
        """
        for frame in self.find_frames(**kwargs):
            yield self.decode_frame(frame)
    
    def get_protocol_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the protocol analysis.
        
        Returns:
            Dict[str, Any]: Summary information including frame counts by type,
                            timing statistics, and protocol-specific metrics
        """
        results = self.results
        if not results:
            return {"error": "No analysis results available"}
        
        # Count frames by type
        frame_counts = {}
        for frame in self.find_frames():
            frame_type = frame.type
            frame_type_name = self.get_frame_type_name(frame_type)
            
            if frame_type_name not in frame_counts:
                frame_counts[frame_type_name] = 0
                
            frame_counts[frame_type_name] += 1
        
        # Calculate basic timing
        sample_rate = self.sample_rate
        first_frame = results.get_first_frame()
        last_frame = None
        
        # Find the last frame (inefficient but necessary without direct access)
        frame = results.get_first_frame()
        while frame:
            last_frame = frame
            frame = results.get_next_frame()
        
        # Build summary
        summary = {
            "analyzer_name": self.name,
            "analyzer_version": self.version,
            "sample_rate_hz": sample_rate,
            "frame_counts": frame_counts,
            "total_frames": sum(frame_counts.values()),
        }
        
        # Add timing info if we have frames
        if first_frame and last_frame:
            analysis_duration_samples = last_frame.sample - first_frame.sample
            analysis_duration_seconds = analysis_duration_samples / sample_rate if sample_rate else 0
            
            summary.update({
                "first_frame_sample": first_frame.sample,
                "last_frame_sample": last_frame.sample,
                "analysis_duration_seconds": analysis_duration_seconds,
            })
        
        return summary
    
    # -------------------------------------------------------------------------
    # Visualization and export methods
    # -------------------------------------------------------------------------
    
    def export_results(self, file_path: str, format: str = "csv") -> None:
        """
        Export analysis results to a file.
        
        Args:
            file_path: Path to save the file
            format: Export format ("csv", "json", or "txt")
            
        Raises:
            ValueError: If the format is not supported
            RuntimeError: If no results are available
        """
        if not self.results:
            raise RuntimeError("No analysis results available")
            
        if format.lower() == "csv":
            self._export_to_csv(file_path)
        elif format.lower() == "json":
            self._export_to_json(file_path)
        elif format.lower() == "txt":
            self._export_to_txt(file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_csv(self, file_path: str) -> None:
        """
        Export results to CSV format.
        
        Args:
            file_path: Path to save the CSV file
        """
        import csv
        
        with open(file_path, 'w', newline='') as csvfile:
            # Get a sample decoded frame to determine columns
            for decoded_frame in self.decode_all_frames(max_frames=1):
                # Add standard fields
                fieldnames = ["frame_type", "sample", "timestamp"]
                
                # Add protocol-specific fields
                fieldnames.extend(decoded_frame.keys())
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                break
                
            # No frames found
            if 'writer' not in locals():
                writer = csv.writer(csvfile)
                writer.writerow(["No frames found"])
                return
            
            # Write all frames
            for frame in self.find_frames():
                # Decode the frame
                decoded = self.decode_frame(frame)
                
                # Add standard fields
                row = {
                    "frame_type": self.get_frame_type_name(frame.type),
                    "sample": frame.sample,
                    "timestamp": frame.sample / self.sample_rate if self.sample_rate else 0,
                }
                
                # Add decoded fields
                row.update(decoded)
                
                writer.writerow(row)
    
    def _export_to_json(self, file_path: str) -> None:
        """
        Export results to JSON format.
        
        Args:
            file_path: Path to save the JSON file
        """
        import json
        
        # Collect all decoded frames
        frames = []
        for frame in self.find_frames():
            # Decode the frame
            decoded = self.decode_frame(frame)
            
            # Add standard fields
            row = {
                "frame_type": self.get_frame_type_name(frame.type),
                "sample": frame.sample,
                "timestamp": frame.sample / self.sample_rate if self.sample_rate else 0,
            }
            
            # Add decoded fields
            row.update(decoded)
            
            frames.append(row)
        
        # Create the full JSON structure
        data = {
            "analyzer": self.name,
            "version": self.version,
            "sample_rate_hz": self.sample_rate,
            "trigger_sample": self.trigger_sample,
            "summary": self.get_protocol_summary(),
            "frames": frames,
        }
        
        # Write to file
        with open(file_path, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2)
    
    def _export_to_txt(self, file_path: str) -> None:
        """
        Export results to plain text format.
        
        Args:
            file_path: Path to save the text file
        """
        with open(file_path, 'w') as txtfile:
            # Write header
            txtfile.write(f"Analysis Results: {self.name} v{self.version}\n")
            txtfile.write(f"Sample Rate: {self.sample_rate} Hz\n")
            txtfile.write("-" * 80 + "\n\n")
            
            # Write summary
            summary = self.get_protocol_summary()
            txtfile.write("Summary:\n")
            for key, value in summary.items():
                if key == "frame_counts":
                    txtfile.write("Frame Counts:\n")
                    for frame_type, count in value.items():
                        txtfile.write(f"  {frame_type}: {count}\n")
                else:
                    txtfile.write(f"{key}: {value}\n")
            
            txtfile.write("\n" + "-" * 80 + "\n\n")
            
            # Write frames
            txtfile.write("Frames:\n\n")
            
            for frame in self.find_frames():
                # Decode the frame
                decoded = self.decode_frame(frame)
                
                # Write standard fields
                frame_type = self.get_frame_type_name(frame.type)
                sample = frame.sample
                timestamp = sample / self.sample_rate if self.sample_rate else 0
                
                txtfile.write(f"Frame Type: {frame_type}\n")
                txtfile.write(f"Sample: {sample}\n")
                txtfile.write(f"Timestamp: {timestamp:.6f} s\n")
                
                # Write decoded fields
                for key, value in decoded.items():
                    txtfile.write(f"{key}: {value}\n")
                
                txtfile.write("\n")
    
    def plot_protocol_overview(self, figsize=(12, 6)) -> Any:
        """
        Plot an overview of the protocol analysis.
        
        Args:
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
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot frame type distribution (top)
        summary = self.get_protocol_summary()
        frame_counts = summary.get("frame_counts", {})
        
        if frame_counts:
            axes[0].bar(frame_counts.keys(), frame_counts.values())
            axes[0].set_title("Frame Type Distribution")
            axes[0].set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
        else:
            axes[0].text(0.5, 0.5, "No frames available", ha='center', va='center')
            axes[0].set_title("Frame Type Distribution")
        
        # Plot frame timing (bottom)
        frames = list(self.find_frames())
        
        if frames:
            # Extract sample numbers and types
            samples = [frame.sample for frame in frames]
            frame_types = [frame.type for frame in frames]
            
            # Convert to timestamps if sample rate is available
            if self.sample_rate:
                times = [sample / self.sample_rate for sample in samples]
                x_values = times
                x_label = "Time (s)"
            else:
                x_values = samples
                x_label = "Sample"
            
            # Create a scatter plot with different colors for frame types
            unique_types = set(frame_types)
            colors = plt.cm.tab10(range(len(unique_types)))
            color_map = dict(zip(unique_types, colors))
            
            for frame_type in unique_types:
                # Get indices where frame_type matches
                indices = [i for i, t in enumerate(frame_types) if t == frame_type]
                
                # Plot points for this frame type
                axes[1].scatter(
                    [x_values[i] for i in indices],
                    [1 for _ in indices],  # All at height 1
                    color=color_map[frame_type],
                    label=self.get_frame_type_name(frame_type),
                    s=50  # Point size
                )
            
            axes[1].set_title("Frame Timing")
            axes[1].set_xlabel(x_label)
            axes[1].set_yticks([])  # Hide Y axis
            axes[1].legend()
            axes[1].grid(True, axis='x')
        else:
            axes[1].text(0.5, 0.5, "No frames available", ha='center', va='center')
            axes[1].set_title("Frame Timing")
        
        plt.tight_layout()
        return fig, axes


# Import common protocol analyzers for convenience
try:
    from kingst_analyzer.protocols import (
        SPIAnalyzer,
        I2CAnalyzer,
        UARTAnalyzer,
        CANAnalyzer
    )
except ImportError:
    # Protocol analyzers not available
    pass