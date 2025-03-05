"""
Analyzer Module

This module provides a high-level Pythonic API for the Kingst Logic Analyzer's core Analyzer functionality.
It wraps the lower-level C++ bindings with a more intuitive interface and proper Python idioms.
"""

from typing import Optional, List, Tuple, Union, Dict, Any, Callable
import abc
from enum import Enum
import contextlib

# Import the low-level bindings
import kingst_analyzer as ka


class AnalyzerState(Enum):
    """Enumeration of possible analyzer states."""
    IDLE = 0
    RUNNING = 1
    COMPLETED = 2
    ERROR = 3


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
    """
    
    def __init__(self):
        """Initialize a new Analyzer instance."""
        self._analyzer = None  # Will hold the C++ analyzer instance
        self._settings = None  # Will hold the settings object
        self._results = None   # Will hold the analyzer results
        self._state = AnalyzerState.IDLE
    
    @property
    def name(self) -> str:
        """Get the name of the analyzer."""
        if self._analyzer:
            return self._analyzer.get_analyzer_name()
        return self._get_analyzer_name()  # Fall back to our implementation
    
    @property
    def version(self) -> str:
        """Get the version of the analyzer."""
        if self._analyzer:
            return self._analyzer.get_analyzer_version()
        return "1.0.0"  # Default version if not specified
    
    @property
    def min_sample_rate(self) -> int:
        """Get the minimum sample rate required for this analyzer in Hz."""
        if self._analyzer:
            return self._analyzer.get_minimum_sample_rate_hz()
        return self._get_minimum_sample_rate_hz()  # Fall back to our implementation
    
    @property
    def settings(self) -> Any:
        """Get the analyzer settings object."""
        return self._settings
    
    @settings.setter
    def settings(self, settings: Any) -> None:
        """Set the analyzer settings object."""
        self._settings = settings
        if self._analyzer:
            self._analyzer.set_analyzer_settings(settings)
    
    @property
    def state(self) -> AnalyzerState:
        """Get the current state of the analyzer."""
        return self._state
    
    @property
    def sample_rate(self) -> int:
        """Get the current sample rate in Hz."""
        if self._analyzer:
            return self._analyzer.get_sample_rate()
        return 0
    
    @property
    def trigger_sample(self) -> int:
        """Get the trigger sample number."""
        if self._analyzer:
            return self._analyzer.get_trigger_sample()
        return 0
    
    @property
    def progress(self) -> float:
        """Get the current analyzer progress as a value between 0.0 and 1.0."""
        if self._analyzer:
            return self._analyzer.get_analyzer_progress()
        return 0.0
    
    def start_analysis(self, starting_sample: int = 0) -> None:
        """
        Start the analysis process.
        
        Args:
            starting_sample: Sample number to start processing from (default: 0)
            
        Raises:
            RuntimeError: If the analyzer is already running or not properly initialized
        """
        if self._state == AnalyzerState.RUNNING:
            raise RuntimeError("Analyzer is already running")
        
        if not self._analyzer:
            self._initialize_analyzer()
        
        if starting_sample > 0:
            self._analyzer.start_processing_from(starting_sample)
        else:
            self._analyzer.start_processing()
            
        self._state = AnalyzerState.RUNNING
    
    def stop_analysis(self) -> None:
        """
        Stop the analysis process.
        
        Raises:
            RuntimeError: If the analyzer is not running
        """
        if self._state != AnalyzerState.RUNNING:
            raise RuntimeError("Analyzer is not running")
        
        if self._analyzer:
            self._analyzer.stop_worker_thread()
            
        self._state = AnalyzerState.IDLE
    
    def get_channel_data(self, channel: "ka.Channel") -> "ka.AnalyzerChannelData":
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
        Report progress to the UI.
        
        Call this periodically during analysis to update the progress indicator.
        
        Args:
            sample_number: Current sample number being processed
        """
        if self._analyzer:
            self._analyzer.report_progress(sample_number)
    
    def check_should_abort(self) -> bool:
        """
        Check if the analysis should be aborted.
        
        Returns:
            bool: True if the analysis should be aborted
        """
        if not self._analyzer:
            return False
            
        try:
            self._analyzer.check_if_thread_should_exit()
            return False
        except Exception:
            return True
    
    def setup(self) -> None:
        """
        Set up the analyzer before running.
        
        This is called automatically before starting the analysis.
        """
        if self._analyzer:
            self._analyzer.setup_results()
    
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
            self.start_analysis(starting_sample)
            yield self
        finally:
            self.stop_analysis()
    
    def _initialize_analyzer(self) -> None:
        """Initialize the C++ analyzer instance."""
        # This will be implemented in derived classes
        pass
    
    # Abstract methods that must be implemented by derived classes
    
    @abc.abstractmethod
    def _get_analyzer_name(self) -> str:
        """Get the name of the analyzer (implementation method)."""
        pass
    
    @abc.abstractmethod
    def _get_minimum_sample_rate_hz(self) -> int:
        """Get the minimum sample rate required for this analyzer in Hz (implementation method)."""
        pass
    
    @abc.abstractmethod
    def worker_thread(self) -> None:
        """
        Main worker thread for the analyzer.
        
        This method contains the main analysis logic and is called when the analyzer is run.
        It should process the channel data, detect protocol features, and add frames to the results.
        """
        pass
    
    @abc.abstractmethod
    def needs_rerun(self) -> bool:
        """
        Check if the analyzer needs to be rerun.
        
        Returns:
            bool: True if the analyzer needs to be rerun
        """
        pass


class Analyzer(BaseAnalyzer):
    """
    Concrete implementation of a Kingst Logic Analyzer protocol analyzer.
    
    This class extends BaseAnalyzer to provide a concrete implementation
    that directly wraps the C++ Analyzer class.
    
    Example:
        ```python
        from kingst_LA.analyzer import Analyzer
        
        class SPIAnalyzer(Analyzer):
            def __init__(self):
                super().__init__()
                # Configure channels and settings
                
            def _get_analyzer_name(self):
                return "SPI Analyzer"
                
            def _get_minimum_sample_rate_hz(self):
                return 10000  # 10 kHz
                
            def worker_thread(self):
                # Get channel data
                mosi = self.get_channel_data(self.settings.mosi_channel)
                miso = self.get_channel_data(self.settings.miso_channel)
                clock = self.get_channel_data(self.settings.clock_channel)
                
                # Process the data
                while clock.get_sample_number() < end_sample:
                    if self.check_should_abort():
                        break
                        
                    # Process one SPI transaction
                    # ...
                    
                    # Report progress
                    self.report_progress(clock.get_sample_number())
                
            def needs_rerun(self):
                return False
        ```
    """
    
    class PyAnalyzerImpl(ka.Analyzer):
        """Inner class that implements the C++ Analyzer interface."""
        
        def __init__(self, outer):
            super().__init__()
            self.outer = outer
        
        def worker_thread(self):
            """Delegate to the outer class's worker_thread method."""
            self.outer.worker_thread()
        
        def get_analyzer_name(self):
            """Delegate to the outer class's _get_analyzer_name method."""
            return self.outer._get_analyzer_name()
        
        def get_minimum_sample_rate_hz(self):
            """Delegate to the outer class's _get_minimum_sample_rate_hz method."""
            return self.outer._get_minimum_sample_rate_hz()
        
        def needs_rerun(self):
            """Delegate to the outer class's needs_rerun method."""
            return self.outer.needs_rerun()
    
    def __init__(self):
        """Initialize a new Analyzer instance."""
        super().__init__()
        self._initialize_analyzer()
    
    def _initialize_analyzer(self) -> None:
        """Initialize the C++ analyzer instance."""
        self._analyzer = self.PyAnalyzerImpl(self)
        
        # Set up results if settings are available
        if self._settings:
            self._analyzer.set_analyzer_settings(self._settings)
            
        # Set up results object
        # Note: This normally happens when calling setup(), but we can do it here too
        self._analyzer.setup_results()