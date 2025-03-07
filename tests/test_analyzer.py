"""
Tests for the Analyzer class from the kingst_analyzer module.

These tests validate the core functionality of the Analyzer class,
which is the primary interface for implementing custom protocol analyzers.
"""

import os
import sys
import time
from pathlib import Path
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kingst_analyzer.analyzer import (
    Analyzer, 
    BaseAnalyzer, 
    AnalyzerState, 
    AnalysisError, 
    AnalyzerInitError, 
    AnalysisAbortedError
)
from kingst_analyzer.types import BitState, Channel, DisplayBase
from kingst_analyzer.settings import AnalyzerSettings


class TestAnalyzerInitialization:
    """Tests for Analyzer initialization and basic properties."""

    def test_analyzer_abstract_base_class(self):
        """Test that BaseAnalyzer is an abstract base class that cannot be instantiated directly."""
        with pytest.raises(TypeError) as excinfo:
            analyzer = BaseAnalyzer()
        
        assert "abstract" in str(excinfo.value).lower(), "BaseAnalyzer should be an abstract class"

    def test_analyzer_minimal_implementation(self):
        """Test instantiating a minimal Analyzer subclass with required methods implemented."""
        
        class MinimalAnalyzer(Analyzer):
            def _get_analyzer_name(self):
                return "Minimal Test Analyzer"
                
            def _get_minimum_sample_rate_hz(self):
                return 1000000  # 1 MHz
                
            def worker_thread(self):
                pass  # Do nothing
                
            def needs_rerun(self):
                return False

        # Should not raise an exception
        analyzer = MinimalAnalyzer()
        
        # Check basic properties
        assert analyzer.name == "Minimal Test Analyzer"
        assert analyzer.min_sample_rate == 1000000
        assert analyzer.state == AnalyzerState.IDLE

    def test_analyzer_initialization_error(self):
        """Test that errors during analyzer initialization are properly handled."""
        
        class BrokenAnalyzer(Analyzer):
            def __init__(self):
                # Intentionally break initialization
                raise RuntimeError("Simulated initialization error")
                
            def _get_analyzer_name(self):
                return "Broken Analyzer"
                
            def _get_minimum_sample_rate_hz(self):
                return 1000000
                
            def worker_thread(self):
                pass
                
            def needs_rerun(self):
                return False
        
        with pytest.raises(RuntimeError) as excinfo:
            analyzer = BrokenAnalyzer()
            
        assert "Simulated initialization error" in str(excinfo.value)

    def test_analyzer_default_properties(self):
        """Test that default properties of Analyzer are correctly set."""
        
        class SimpleAnalyzer(Analyzer):
            def _get_analyzer_name(self):
                return "Simple Analyzer"
                
            def _get_minimum_sample_rate_hz(self):
                return 8000000  # 8 MHz
                
            def worker_thread(self):
                pass
                
            def needs_rerun(self):
                return False
        
        analyzer = SimpleAnalyzer()
        
        # Check default properties
        assert analyzer.name == "Simple Analyzer"
        assert analyzer.version.startswith("1.0.0")  # Default version
        assert analyzer.min_sample_rate == 8000000
        assert analyzer.settings is None
        assert analyzer.state == AnalyzerState.IDLE
        assert analyzer.sample_rate == 0  # No hardware connected
        assert analyzer.trigger_sample == 0  # No trigger defined
        assert analyzer.progress == 0.0  # No analysis running
        assert analyzer.results is None  # No results yet

    def test_analyzer_with_custom_settings(self):
        """Test that custom settings can be properly set and retrieved."""
        
        # Create a simple settings class
        class SimpleSettings(AnalyzerSettings):
            def __init__(self):
                super().__init__()
                self.channel = Channel(0, 0)
                self.sample_rate = 10000000
            
            def get_required_channels(self):
                return [self.channel]
            
            def get_name(self):
                return "Simple Analyzer Settings"
        
        class SimpleAnalyzer(Analyzer):
            def __init__(self):
                super().__init__()
                self._settings = SimpleSettings()
                
            def _get_analyzer_name(self):
                return "Simple Analyzer"
                
            def _get_minimum_sample_rate_hz(self):
                return 8000000
                
            def worker_thread(self):
                pass
                
            def needs_rerun(self):
                return False
        
        analyzer = SimpleAnalyzer()
        
        # Check that settings are properly set
        assert analyzer.settings is not None
        assert analyzer.settings.get_name() == "Simple Analyzer Settings"


class TestAnalyzerLifecycle:
    """Tests for the analyzer lifecycle (start, stop, etc.)."""
    
    class BasicAnalyzer(Analyzer):
        """A basic analyzer implementation for testing."""
        
        def __init__(self):
            super().__init__()
            self.worker_called = False
            self.worker_completed = False
            self.worker_aborted = False
            self.worker_exception = None
            self.progress_samples = []
        
        def _get_analyzer_name(self):
            return "Basic Test Analyzer"
            
        def _get_minimum_sample_rate_hz(self):
            return 1000000
            
        def worker_thread(self):
            self.worker_called = True
            try:
                # Simulate some work and report progress
                for i in range(10):
                    # Check for abort request
                    self.should_abort()
                    
                    # Report progress
                    self.report_progress(i * 100)
                    self.progress_samples.append(i * 100)
                    
                    # Simulate work
                    time.sleep(0.01)
                
                self.worker_completed = True
            except AnalysisAbortedError:
                self.worker_aborted = True
            except Exception as e:
                self.worker_exception = e
                raise
                
        def needs_rerun(self):
            return False
    
    def test_start_analysis_sync(self):
        """Test starting analysis in synchronous mode."""
        analyzer = self.BasicAnalyzer()
        
        # Start analysis synchronously (blocks until complete)
        analyzer.start_analysis(async_mode=False)
        
        # Check that worker was called and completed
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.worker_completed, "Worker thread should have completed"
        assert not analyzer.worker_aborted, "Worker thread should not have been aborted"
        assert analyzer.worker_exception is None, "Worker thread should not have raised an exception"
        
        # Check progress reporting
        assert len(analyzer.progress_samples) == 10, "Progress should have been reported 10 times"
        assert analyzer.state == AnalyzerState.COMPLETED, "Analyzer state should be COMPLETED"

    def test_start_analysis_async(self):
        """Test starting analysis in asynchronous mode."""
        analyzer = self.BasicAnalyzer()
        
        # Start analysis asynchronously
        analyzer.start_analysis(async_mode=True)
        
        # Wait for completion
        analyzer.wait_for_completion(timeout=1.0)
        
        # Check that worker was called and completed
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.worker_completed, "Worker thread should have completed"
        assert not analyzer.worker_aborted, "Worker thread should not have been aborted"
        assert analyzer.worker_exception is None, "Worker thread should not have raised an exception"
        
        # Check progress reporting
        assert len(analyzer.progress_samples) == 10, "Progress should have been reported 10 times"
        assert analyzer.state == AnalyzerState.COMPLETED, "Analyzer state should be COMPLETED"

    def test_stop_analysis(self):
        """Test stopping a running analysis."""
        
        class SlowAnalyzer(self.BasicAnalyzer):
            def worker_thread(self):
                self.worker_called = True
                try:
                    # Simulate slow work that can be interrupted
                    for i in range(100):
                        self.should_abort()
                        self.report_progress(i * 100)
                        self.progress_samples.append(i * 100)
                        time.sleep(0.1)  # Slow enough to be interrupted
                    
                    self.worker_completed = True
                except AnalysisAbortedError:
                    self.worker_aborted = True
                except Exception as e:
                    self.worker_exception = e
                    raise
        
        analyzer = SlowAnalyzer()
        
        # Start analysis asynchronously
        analyzer.start_analysis(async_mode=True)
        
        # Give it a moment to start
        time.sleep(0.2)
        
        # Stop the analysis
        analyzer.stop_analysis()
        
        # Wait a bit to ensure it's stopped
        time.sleep(0.2)
        
        # Check that worker was called and aborted
        assert analyzer.worker_called, "Worker thread should have been called"
        assert not analyzer.worker_completed, "Worker thread should not have completed"
        assert analyzer.worker_aborted, "Worker thread should have been aborted"
        assert analyzer.worker_exception is None, "Worker thread should not have raised an exception"
        assert analyzer.state == AnalyzerState.STOPPED, "Analyzer state should be STOPPED"

    def test_start_analysis_error(self):
        """Test error handling when analysis fails."""
        
        class ErrorAnalyzer(self.BasicAnalyzer):
            def worker_thread(self):
                self.worker_called = True
                try:
                    # Report progress once
                    self.report_progress(100)
                    self.progress_samples.append(100)
                    
                    # Simulate an error
                    raise ValueError("Simulated analysis error")
                except AnalysisAbortedError:
                    self.worker_aborted = True
                except Exception as e:
                    self.worker_exception = e
                    raise
        
        analyzer = ErrorAnalyzer()
        
        # Start analysis asynchronously
        analyzer.start_analysis(async_mode=True)
        
        # Wait a moment for the error to occur
        time.sleep(0.2)
        
        # Check that worker was called and an error occurred
        assert analyzer.worker_called, "Worker thread should have been called"
        assert not analyzer.worker_completed, "Worker thread should not have completed"
        assert not analyzer.worker_aborted, "Worker thread should not have been aborted"
        assert analyzer.worker_exception is not None, "Worker thread should have raised an exception"
        assert isinstance(analyzer.worker_exception, ValueError), "Worker thread should have raised ValueError"
        assert "Simulated analysis error" in str(analyzer.worker_exception)
        assert analyzer.state == AnalyzerState.ERROR, "Analyzer state should be ERROR"
        
        # Verify that wait_for_completion re-raises the exception
        with pytest.raises(ValueError) as excinfo:
            analyzer.wait_for_completion()
        assert "Simulated analysis error" in str(excinfo.value)

    def test_analyzer_context_manager(self):
        """Test using the analyzer as a context manager."""
        analyzer = self.BasicAnalyzer()
        
        # Use context manager
        with analyzer.analysis_session() as session:
            # Check that analysis is running
            assert analyzer.state == AnalyzerState.RUNNING, "Analyzer state should be RUNNING in context"
            assert session is analyzer, "Context manager should return the analyzer"
            
            # Let it run for a moment
            time.sleep(0.2)
        
        # Context should have closed, stopping the analysis if needed
        assert analyzer.state in (AnalyzerState.COMPLETED, AnalyzerState.STOPPED), \
            "Analyzer state should be COMPLETED or STOPPED after context"

    def test_analyzer_context_manager_with_error(self):
        """Test error handling in analyzer context manager."""
        
        class ErrorAnalyzer(self.BasicAnalyzer):
            def worker_thread(self):
                self.worker_called = True
                # Sleep a bit to make sure we're still running when the error is raised
                time.sleep(0.2)
                self.worker_completed = True
        
        analyzer = ErrorAnalyzer()
        
        # Use context manager with an error
        try:
            with analyzer.analysis_session():
                # Check that analysis is running
                assert analyzer.state == AnalyzerState.RUNNING, "Analyzer state should be RUNNING in context"
                
                # Raise an error
                raise RuntimeError("Simulated error in context")
        except RuntimeError as e:
            assert "Simulated error in context" in str(e), "Context manager should not suppress errors"
        
        # Context should have closed, stopping the analysis if needed
        assert analyzer.state in (AnalyzerState.COMPLETED, AnalyzerState.STOPPED), \
            "Analyzer state should be COMPLETED or STOPPED after context with error"


class TestAnalyzerCallbacks:
    """Tests for analyzer callbacks (progress, state changes)."""
    
    class CallbackTestAnalyzer(Analyzer):
        """Analyzer implementation for testing callbacks."""
        
        def __init__(self):
            super().__init__()
        
        def _get_analyzer_name(self):
            return "Callback Test Analyzer"
            
        def _get_minimum_sample_rate_hz(self):
            return 1000000
            
        def worker_thread(self):
            # Simulate work and trigger callbacks
            for i in range(5):
                self.report_progress(i * 1000)
                time.sleep(0.05)
                
        def needs_rerun(self):
            return False
    
    def test_progress_callback(self):
        """Test that progress callbacks are properly called."""
        analyzer = self.CallbackTestAnalyzer()
        
        # Set up progress callback
        progress_values = []
        def progress_callback(progress):
            progress_values.append(progress)
        
        analyzer.add_progress_callback(progress_callback)
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        
        # Check that callback was called
        assert len(progress_values) > 0, "Progress callback should have been called"

    def test_state_callback(self):
        """Test that state callbacks are properly called."""
        analyzer = self.CallbackTestAnalyzer()
        
        # Set up state callback
        state_changes = []
        def state_callback(state):
            state_changes.append(state)
        
        analyzer.add_state_callback(state_callback)
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        
        # Check that callback was called for state transitions
        assert len(state_changes) >= 2, "State callback should have been called at least twice"
        assert AnalyzerState.INITIALIZING in state_changes, "Should have seen INITIALIZING state"
        assert AnalyzerState.RUNNING in state_changes, "Should have seen RUNNING state"
        assert AnalyzerState.COMPLETED in state_changes, "Should have seen COMPLETED state"

    def test_remove_callbacks(self):
        """Test that callbacks can be removed."""
        analyzer = self.CallbackTestAnalyzer()
        
        # Set up callbacks
        progress_values1 = []
        progress_values2 = []
        
        def progress_callback1(progress):
            progress_values1.append(progress)
            
        def progress_callback2(progress):
            progress_values2.append(progress)
        
        # Add both callbacks
        analyzer.add_progress_callback(progress_callback1)
        analyzer.add_progress_callback(progress_callback2)
        
        # Remove one callback
        analyzer.remove_progress_callback(progress_callback2)
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        
        # Check that only callback1 was called
        assert len(progress_values1) > 0, "Progress callback1 should have been called"
        assert len(progress_values2) == 0, "Progress callback2 should not have been called"

    def test_callback_error_handling(self):
        """Test that errors in callbacks are handled gracefully."""
        analyzer = self.CallbackTestAnalyzer()
        
        # Set up a callback that raises an exception
        def error_callback(progress):
            raise ValueError("Simulated callback error")
        
        # Set up a normal callback to verify things continue
        normal_called = False
        def normal_callback(progress):
            nonlocal normal_called
            normal_called = True
        
        # Add callbacks (error callback first)
        analyzer.add_progress_callback(error_callback)
        analyzer.add_progress_callback(normal_callback)
        
        # Run analysis - this should not raise an exception despite the callback error
        with pytest.warns(UserWarning):  # Expect a warning
            analyzer.start_analysis(async_mode=False)
        
        # Check that normal callback was still called
        assert normal_called, "Normal callback should still have been called despite error in another callback"


class TestChannelDataAccess:
    """Tests for accessing channel data."""
    
    class ChannelDataAnalyzer(Analyzer):
        """Analyzer implementation for testing channel data access."""
        
        def __init__(self, mock_data=None):
            super().__init__()
            self.channel = Channel(0, 0)
            self.mock_data = mock_data
            
            # Mock the analyzer's _analyzer object
            if mock_data is not None:
                self._analyzer = MagicMock()
                self._analyzer.get_analyzer_channel_data.return_value = mock_data
        
        def _get_analyzer_name(self):
            return "Channel Data Test Analyzer"
            
        def _get_minimum_sample_rate_hz(self):
            return 1000000
            
        def worker_thread(self):
            # This would normally access channel data
            pass
                
        def needs_rerun(self):
            return False
    
    def test_get_channel_data(self):
        """Test getting channel data from the analyzer."""
        # Create mock channel data
        mock_channel_data = MagicMock()
        mock_channel_data.sample_number = 0
        mock_channel_data.get_next_transition.return_value = (100, BitState.HIGH)
        
        analyzer = self.ChannelDataAnalyzer(mock_data=mock_channel_data)
        
        # Get channel data
        channel_data = analyzer.get_channel_data(analyzer.channel)
        
        # Check that we got the mock data
        assert channel_data is mock_channel_data, "Should have returned the mock channel data"
        
        # Check that we can use the channel data
        transition_sample, bit_state = channel_data.get_next_transition()
        assert transition_sample == 100, "Should have returned the mock transition sample"
        assert bit_state == BitState.HIGH, "Should have returned the mock bit state"

    @pytest.mark.hardware
    def test_get_channel_data_hardware(self):
        """Test getting channel data from actual hardware (if available)."""
        # This test requires hardware
        pytest.importorskip("kingst_hardware", reason="Hardware support not available")
        
        # Create a hardware-connected analyzer
        analyzer = self.ChannelDataAnalyzer()
        
        # Try to get channel data (this may fail if no hardware is connected)
        try:
            channel_data = analyzer.get_channel_data(analyzer.channel)
            assert channel_data is not None, "Channel data should not be None with hardware connected"
        except RuntimeError:
            pytest.skip("No hardware connected or channel data not available")

    def test_get_channel_data_error(self):
        """Test error handling when getting channel data fails."""
        analyzer = self.ChannelDataAnalyzer()
        
        # Analyzer not properly initialized (no mock data)
        with pytest.raises(RuntimeError) as excinfo:
            channel_data = analyzer.get_channel_data(analyzer.channel)
        
        assert "not initialized" in str(excinfo.value).lower(), "Should indicate analyzer not initialized"


class TestAnalyzerReset:
    """Tests for resetting the analyzer."""
    
    class ResettableAnalyzer(Analyzer):
        """Analyzer implementation for testing reset functionality."""
        
        def __init__(self):
            super().__init__()
            self.reset_count = 0
            self.init_count = 1  # Already initialized once in __init__
        
        def _get_analyzer_name(self):
            return "Resettable Test Analyzer"
            
        def _get_minimum_sample_rate_hz(self):
            return 1000000
            
        def worker_thread(self):
            # Simulate work
            time.sleep(0.1)
                
        def needs_rerun(self):
            return False
            
        def _initialize_analyzer(self):
            """Override to track initialization."""
            super()._initialize_analyzer()
            self.init_count += 1
    
    def test_reset(self):
        """Test resetting the analyzer."""
        analyzer = self.ResettableAnalyzer()
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        assert analyzer.state == AnalyzerState.COMPLETED, "Analyzer should have completed"
        
        # Reset
        analyzer.reset()
        
        # Check that state was reset
        assert analyzer.state == AnalyzerState.IDLE, "Analyzer state should be IDLE after reset"
        assert analyzer.init_count == 2, "Analyzer should have been reinitialized"
        
        # Run analysis again
        analyzer.start_analysis(async_mode=False)
        assert analyzer.state == AnalyzerState.COMPLETED, "Analyzer should have completed after reset"


class TestNestedAnalyzers:
    """Tests for analyzers that use other analyzers."""
    
    class FrameGenerator(Analyzer):
        """Simple analyzer that generates frames."""
        
        def __init__(self):
            super().__init__()
            self.frames_generated = 0
        
        def _get_analyzer_name(self):
            return "Frame Generator"
            
        def _get_minimum_sample_rate_hz(self):
            return 1000000
            
        def worker_thread(self):
            # Simulate generating some frames (in a real analyzer, these would be added to results)
            for i in range(10):
                self.should_abort()  # Check for abort requests
                self.frames_generated += 1
                self.report_progress(i * 100)
                time.sleep(0.01)
                
        def needs_rerun(self):
            return False
    
    class MetaAnalyzer(Analyzer):
        """Analyzer that uses another analyzer."""
        
        def __init__(self):
            super().__init__()
            self.inner_analyzer = None
            self.processed_frames = 0
        
        def _get_analyzer_name(self):
            return "Meta Analyzer"
            
        def _get_minimum_sample_rate_hz(self):
            return 1000000
            
        def worker_thread(self):
            # Create and run an inner analyzer
            self.inner_analyzer = self.FrameGenerator()
            self.inner_analyzer.start_analysis(async_mode=False)
            
            # Process the frames from the inner analyzer
            self.processed_frames = self.inner_analyzer.frames_generated
            
            # Report our own progress
            self.report_progress(1000)
                
        def needs_rerun(self):
            return False
    
    def test_nested_analyzer(self):
        """Test an analyzer that uses another analyzer internally."""
        analyzer = self.MetaAnalyzer()
        
        # Run the meta-analyzer
        analyzer.start_analysis(async_mode=False)
        
        # Check that both analyzers ran
        assert analyzer.inner_analyzer is not None, "Inner analyzer should have been created"
        assert analyzer.inner_analyzer.frames_generated == 10, "Inner analyzer should have generated 10 frames"
        assert analyzer.processed_frames == 10, "Meta analyzer should have processed 10 frames"
        assert analyzer.state == AnalyzerState.COMPLETED, "Meta analyzer should have completed"
        assert analyzer.inner_analyzer.state == AnalyzerState.COMPLETED, "Inner analyzer should have completed"


@pytest.mark.hardware
class TestHardwareAnalyzer:
    """Hardware-dependent tests for the Analyzer class."""
    
    @pytest.fixture
    def hardware_check(self):
        """Check if hardware is available and skip if not."""
        try:
            # Import hardware module
            pytest.importorskip("kingst_hardware", reason="Hardware support not available")
            
            # Check if device is connected
            from kingst_hardware import DeviceManager
            manager = DeviceManager()
            devices = manager.get_connected_devices()
            
            if not devices:
                pytest.skip("No Kingst devices connected")
                
            return devices[0]  # Return the first device
        except ImportError:
            pytest.skip("Hardware support not available")
    
    class RealHardwareAnalyzer(Analyzer):
        """Analyzer implementation for testing with real hardware."""
        
        def __init__(self, device):
            super().__init__()
            self.device = device
            self.channel = Channel(device.id, 0)  # Use first channel
            
            # Create basic settings
            class SimpleSettings(AnalyzerSettings):
                def __init__(self, channel):
                    super().__init__()
                    self.channel = channel
                
                def get_required_channels(self):
                    return [self.channel]
                
                def get_name(self):
                    return "Simple Hardware Settings"
            
            self._settings = SimpleSettings(self.channel)
        
        def _get_analyzer_name(self):
            return "Hardware Test Analyzer"
            
        def _get_minimum_sample_rate_hz(self):
            return 1000000
            
        def worker_thread(self):
            # Get channel data
            channel_data = self.get_channel_data(self.channel)
            
            # Process a few transitions
            transitions = []
            for _ in range(5):
                try:
                    # Get next transition with a timeout (avoid infinite wait)
                    sample, state = channel_data.get_next_transition()
                    transitions.append((sample, state))
                    self.report_progress(sample)
                except:
                    # No more transitions or error
                    break
                
        def needs_rerun(self):
            return False
    
    def test_hardware_analyzer_init(self, hardware_check):
        """Test initializing an analyzer with real hardware."""
        device = hardware_check
        analyzer = self.RealHardwareAnalyzer(device)
        
        # Check that analyzer was initialized with hardware
        assert analyzer.device is device, "Analyzer should be associated with the device"
        assert analyzer.sample_rate > 0, "Sample rate should be non-zero with hardware"

    def test_hardware_analyzer_run(self, hardware_check):
        """Test running an analyzer with real hardware."""
        device = hardware_check
        analyzer = self.RealHardwareAnalyzer(device)
        
        # Start the analyzer (this may involve hardware interaction)
        analyzer.start_analysis(async_mode=True)
        
        # Wait for completion (with timeout to avoid hanging)
        completed = analyzer.wait_for_completion(timeout=5.0)
        
        # Check that analysis completed or is in a valid state
        if not completed:
            analyzer.stop_analysis()
        
        assert analyzer.state in (AnalyzerState.COMPLETED, AnalyzerState.STOPPED), \
            f"Analyzer should be completed or stopped, not {analyzer.state}"


if __name__ == "__main__":
    # Set this to True to run hardware tests, False to skip them
    RUN_HARDWARE_TESTS = False
    
    if RUN_HARDWARE_TESTS:
        # Run all tests including hardware tests
        pytest.main(["-v", __file__])
    else:
        # Skip hardware tests
        pytest.main(["-v", "-m", "not hardware", __file__])