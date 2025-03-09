"""
Analyzer Module

This module provides a high-level Pythonic API for the Kingst Logic Analyzer's core Analyzer functionality.
It wraps the lower-level C++ bindings with a more intuitive interface and proper Python idioms.

The Analyzer class represents the main entry point for implementing custom protocol analyzers.
Developers will typically subclass Analyzer to create protocol-specific analyzers (SPI, I2C, UART, etc.).
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


class HardwareNotConnectedError(AnalysisError):
    """Exception raised when hardware is required but not connected."""
    pass


class BaseAnalyzer(abc.ABC):
    """
    Abstract base class for all Kingst Logic Analyzer protocol analyzers.
    
    This class defines the interface that all protocol analyzers must implement.
    It provides methods to control the analyzer lifecycle, access channel data,
    and generate analysis results.
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
        self._hardware_connected = False
        self._device_handle = None
    
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
    
    @property
    def is_hardware_connected(self) -> bool:
        """
        Check if the analyzer is connected to hardware.
        
        Returns:
            bool: True if the analyzer is connected to hardware
        """
        return self._hardware_connected
    
    # -------------------------------------------------------------------------
    # Hardware connection methods
    # -------------------------------------------------------------------------
    
    def connect_to_hardware(self, device_handle=None) -> bool:
        """
        Connect this analyzer to actual hardware.
        
        Args:
            device_handle: Optional handle to a specific device
                
        Returns:
            bool: True if connection was successful
                
        Raises:
            RuntimeError: If hardware connection fails
        """
        if not self._analyzer:
            self._initialize_analyzer()
            
        try:
            # Store the device handle
            self._device_handle = device_handle
            
            # In a real implementation, you would connect to the hardware here
            # For example: self._analyzer.connect_to_device(device_handle)
            
            # For now, just assume it succeeded if we have a device handle
            self._hardware_connected = device_handle is not None
            return self._hardware_connected
        except Exception as e:
            self._hardware_connected = False
            raise RuntimeError(f"Failed to connect to hardware: {e}")
    
    def disconnect_from_hardware(self) -> None:
        """
        Disconnect from hardware.
        
        This method should be called when you're done with the hardware to 
        release any resources.
        """
        if self._hardware_connected and self._analyzer:
            # In a real implementation, you would disconnect from hardware here
            # For example: self._analyzer.disconnect_from_device()
            
            self._hardware_connected = False
            self._device_handle = None
    
    # -------------------------------------------------------------------------
    # Analysis control methods
    # -------------------------------------------------------------------------
    
    def start_analysis(self, starting_sample: int = 0, async_mode: bool = False,
                       require_hardware: bool = False) -> None:
        """
        Start the analysis process.
        
        Args:
            starting_sample: Sample number to start processing from (default: 0)
            async_mode: Whether to run analysis asynchronously (default: False)
            require_hardware: Whether hardware connection is required (default: False)
            
        Raises:
            RuntimeError: If the analyzer is already running or not properly initialized
            HardwareNotConnectedError: If require_hardware is True but no hardware is connected
        """
        if self._state == AnalyzerState.RUNNING:
            raise RuntimeError("Analyzer is already running")
        
        if require_hardware and not self._hardware_connected:
            raise HardwareNotConnectedError("Hardware connection required but not available")
        
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
            if not self._analyzer:
                raise AnalyzerInitError("Analyzer not initialized")

            # Setup the analyzer before running
            self.setup()

            # Safely call start_processing with exception handling
            try:
                if starting_sample > 0:
                    self._analyzer.start_processing_from(starting_sample)
                else:
                    self._analyzer.start_processing()
            except Exception as e:
                # Convert generic exceptions to a more specific analyzer error
                raise AnalyzerInitError(f"Failed to start processing: {e}")

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
        
        This method is called before the worker thread and can be overridden
        to perform custom setup operations.
        """
        pass
    
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
            
        Note:
            This method may raise exceptions if the analyzer is not properly
            initialized with hardware in a test environment. Use 
            setup_test_mode() for testing without hardware.
        """
        if self._analyzer:
            try:
                self._analyzer.report_progress(sample_number)
                
                # Calculate progress and notify callbacks
                progress = self.progress
                for callback in list(self._progress_callbacks):
                    try:
                        callback(progress)
                    except Exception as e:
                        warnings.warn(f"Progress callback raised exception: {e}")
            except Exception as e:
                # Handle exceptions in report_progress more gracefully
                warnings.warn(f"Error in report_progress: {e}")
                # Still notify callbacks even if the C++ call failed
                for callback in list(self._progress_callbacks):
                    try:
                        callback(0.0)  # Use a default value
                    except Exception as e:
                        warnings.warn(f"Progress callback raised exception: {e}")
    
    def reset(self) -> None:
        """
        Reset the analyzer to its initial state.
        
        This method clears all results and resets the analyzer state.
        """
        if self._analyzer:
            # Re-initialize the analyzer
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
    # Testing methods
    # -------------------------------------------------------------------------
    
    def setup_test_mode(self) -> None:
        """
        Set up the analyzer for testing without requiring hardware access.
        
        This method replaces problematic C++ methods with test-safe versions
        to allow testing without real hardware.
        
        Example:
            ```python
            analyzer = TestAnalyzer()
            analyzer._initialize_analyzer()
            analyzer.setup_test_mode()
            analyzer.start_analysis(async_mode=False)  # Won't crash now
            ```
        """
        if not self._analyzer:
            self._initialize_analyzer()
            
        # Replace problematic methods with test-safe implementations
        self._analyzer.start_processing = lambda: None
        self._analyzer.start_processing_from = lambda sample: None
        self._analyzer.stop_worker_thread = lambda: None
        self._analyzer.check_if_thread_should_exit = lambda: None
        self._analyzer.get_analyzer_progress = lambda: 0.0
        
        # Critical fix: Safely mock report_progress to prevent access violations
        self._analyzer.report_progress = lambda sample_number: None
        
        # Mock get_analyzer_channel_data to return a mock object if needed
        original_get_channel_data = self._analyzer.get_analyzer_channel_data
        
        def mock_get_channel_data(channel):
            try:
                return original_get_channel_data(channel)
            except Exception:
                from kingst_analyzer.testing import MockChannelData
                return MockChannelData()
                
        self._analyzer.get_analyzer_channel_data = mock_get_channel_data
    
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
        try:
            # Create the Python implementation of the C++ Analyzer
            self._analyzer = self.PyAnalyzerImpl(self)

            # Set up settings if available
            if self._settings:
                self._analyzer.set_analyzer_settings(self._settings)
        except Exception as e:
            self._analyzer = None
            raise AnalyzerInitError(f"Failed to initialize analyzer: {e}")
    
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

    # -------------------------------------------------------------------------
    # Inner class for C++ binding
    # -------------------------------------------------------------------------

    class PyAnalyzerImpl(_ka.Analyzer):
        """Inner class that implements the C++ Analyzer interface."""
    
        def __init__(self, outer):
            """
            Initialize a PyAnalyzerImpl instance.
    
            Args:
                outer: The outer Analyzer instance
            """
            # Important: Call the C++ constructor first
            super().__init__()
            # Store a strong reference to avoid premature garbage collection
            self.outer = outer
    
        def worker_thread(self):
            """
            Implementation of the C++ WorkerThread method.
            This is called by the C++ code when the analyzer is run.
            """
            try:
                self.outer.worker_thread()
            except Exception as e:
                # Let the exception propagate so the C++ side can handle it properly
                raise
            
        def generate_simulation_data(self, newest_sample_requested, sample_rate, simulation_channels):
            """
            Implementation of the C++ GenerateSimulationData method.
            
            This is a simplified implementation that delegates to the Python implementation
            without trying to deal with the simulation_channels parameter directly.
            """
            # This is a minimal implementation to satisfy the virtual method
            # A proper implementation would need to handle simulation_channels correctly
            try:
                # Call the Python implementation if it exists
                if hasattr(self.outer, "_generate_simulation_data"):
                    return self.outer._generate_simulation_data(newest_sample_requested, sample_rate)
                return sample_rate
            except Exception as e:
                warnings.warn(f"Error in generate_simulation_data: {str(e)}")
                return sample_rate
    
        def get_minimum_sample_rate_hz(self):
            """Implementation of the C++ GetMinimumSampleRateHz method."""
            return self.outer._get_minimum_sample_rate_hz()
    
        def get_analyzer_name(self):
            """Implementation of the C++ GetAnalyzerName method."""
            return self.outer._get_analyzer_name()
    
        def needs_rerun(self):
            """Implementation of the C++ NeedsRerun method."""
            return self.outer.needs_rerun()
    
        def setup_results(self):
            """
            Implementation of the C++ SetupResults method.
            
            This calls the outer Python setup method
            without calling the base class implementation to avoid recursion.
            """
            # Call the outer Python setup method
            self.outer.setup()


class Analyzer(BaseAnalyzer):
    """
    Concrete implementation of a Kingst Logic Analyzer protocol analyzer.
    
    This class extends BaseAnalyzer to provide a concrete implementation
    that directly wraps the C++ Analyzer class.
    """
    pass


# Simplified mock class for testing
class MockChannelData:
    """A simple mock for AnalyzerChannelData that can be used in tests."""
    
    def __init__(self):
        """Initialize a new MockChannelData instance."""
        from kingst_analyzer.types import BitState
        self._sample_number = 0
        self._bit_state = BitState.LOW
    
    def get_sample_number(self):
        """Get the current sample number."""
        return self._sample_number
    
    def get_bit_state(self):
        """Get the current bit state."""
        return self._bit_state
    
    def advance_to_next_edge(self):
        """Advance to the next edge in the data."""
        self._sample_number += 100
        from kingst_analyzer.types import BitState
        self._bit_state = BitState.HIGH if self._bit_state == BitState.LOW else BitState.LOW
    
    def advance_to_next_transition(self):
        """Advance to the next transition in the data."""
        self.advance_to_next_edge()
        
    def advance_to_absolute_sample_number(self, sample):
        """Advance to an absolute sample number."""
        self._sample_number = sample


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