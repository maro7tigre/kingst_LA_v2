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
import warnings
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kingst_analyzer.analyzer import (
    Analyzer, 
    BaseAnalyzer, 
    AnalyzerState, 
    AnalysisError, 
    AnalyzerInitError, 
    AnalysisAbortedError,
    HardwareNotConnectedError,
    MockChannelData
)
from kingst_analyzer.types import BitState, Channel, DisplayBase
from kingst_analyzer.settings import AnalyzerSettings
from kingst_analyzer.simulation import SimulationManager, SimulationChannelDescriptor


# Test-specific analyzer implementations
class TestAnalyzer(Analyzer):
    """A test analyzer implementation with tracking variables."""
    
    def __init__(self):
        # Initialize tracking variables first
        self.worker_called = False
        self.worker_completed = False
        self.worker_exception = None
        self.progress_values = []
        
        # Call parent init
        super().__init__()
    
    def _get_analyzer_name(self):
        return "Test Analyzer"
    
    def _get_minimum_sample_rate_hz(self):
        return 1000000  # 1 MHz
    
    def worker_thread(self):
        self.worker_called = True
        try:
            # Simulate work
            for i in range(10):
                # Check for abort
                if self._abort_requested:
                    raise AnalysisAbortedError("Analysis aborted by user")
                
                # Report progress
                self.report_progress(i * 100)
                
                # Short sleep
                time.sleep(0.01)
            
            self.worker_completed = True
        except AnalysisAbortedError:
            raise
        except Exception as e:
            self.worker_exception = e
            raise
    
    def needs_rerun(self):
        return False
    
    # Override report_progress to track values
    def report_progress(self, sample_number):
        # Call the real implementation but catch any errors
        try:
            super().report_progress(sample_number)
        except Exception as e:
            warnings.warn(f"Error in report_progress: {e}")
        
        # Track the value
        self.progress_values.append(sample_number)


class FailingTestAnalyzer(TestAnalyzer):
    """A test analyzer that fails in worker_thread."""
    
    def worker_thread(self):
        self.worker_called = True
        # Just fail immediately
        raise ValueError("Simulated worker error")


class SlowTestAnalyzer(TestAnalyzer):
    """A test analyzer with a slow worker thread."""
    
    def worker_thread(self):
        self.worker_called = True
        try:
            # Simulate slow work
            for i in range(5):  # Reduced for faster tests
                if self._abort_requested:
                    raise AnalysisAbortedError("Analysis aborted by user")
                    
                self.report_progress(i * 100)
                time.sleep(0.1)  # Slow enough to interrupt
            
            self.worker_completed = True
        except AnalysisAbortedError:
            # We expect to be aborted
            raise
        except Exception as e:
            self.worker_exception = e
            raise


class RerunTestAnalyzer(TestAnalyzer):
    """A test analyzer that requests a rerun once."""
    
    def __init__(self):
        super().__init__()
        self.rerun_count = 0
        self.rerun_requested = False
    
    def needs_rerun(self):
        if self.rerun_count < 1:
            self.rerun_requested = True
            self.rerun_count += 1
            return True
        return False


@pytest.fixture
def simulation_setup():
    """Create a simulation setup with test data for analyzers to process."""
    # Create a simulation manager
    manager = SimulationManager(sample_rate=10_000_000)  # 10MHz
    
    # Add a clock channel (Channel 0)
    try:
        clock_channel = manager.add_clock(
            channel=Channel(0, 0),
            frequency_hz=1_000_000,  # 1MHz
            duty_cycle_percent=50.0, 
            count=100
        )
        
        # Add a data channel with random pattern (Channel 1)
        data_channel = manager.add_pattern(
            channel=Channel(1, 1),
            pattern_type='walking_ones',
            bit_width=10,
            bits=8
        )
        
        # Return the simulation manager and channels
        return {
            'manager': manager,
            'clock_channel': clock_channel,
            'data_channel': data_channel
        }
    except Exception as e:
        pytest.skip(f"Could not set up simulation: {e}")
        return None


class SpiAnalyzerSettings(AnalyzerSettings):
    """Settings for the SPI Analyzer."""
    
    def __init__(self):
        super().__init__()
        
        # Default channels
        self.clock_channel = Channel(0, 0)  # Device 0, Channel 0
        self.mosi_channel = Channel(0, 1)   # Device 0, Channel 1
        self.miso_channel = Channel(0, 2)   # Device 0, Channel 2
        self.enable_channel = Channel(0, 3) # Device 0, Channel 3
    
    def get_name(self):
        return "SPI Analyzer Settings"
    
    # Add required method from abstract class
    def _create_cpp_settings(self):
        # This is a stub implementation for testing
        pass


class SpiAnalyzer(Analyzer):
    """A simple SPI analyzer for testing with real data."""
    
    def __init__(self):
        super().__init__()
        
        # Create settings
        self._settings = SpiAnalyzerSettings()
        
        # Initialize for testing
        self._initialize_analyzer()
        
        # Results tracking
        self.frames_found = 0
    
    def _get_analyzer_name(self):
        return "SPI Analyzer"
    
    def _get_minimum_sample_rate_hz(self):
        return 1_000_000  # 1 MHz
    
    def worker_thread(self):
        # Get channel data
        # In real usage, we'd process this data and add frames to results
        try:
            clock = self.get_channel_data(self._settings.clock_channel)
            mosi = self.get_channel_data(self._settings.mosi_channel)
            miso = self.get_channel_data(self._settings.miso_channel)
            enable = self.get_channel_data(self._settings.enable_channel)
            
            # Simple processing loop - just count transitions on clock
            # In a real analyzer, we'd actually decode SPI data here
            if clock is not None:
                start_sample = clock.get_sample_number()
                
                # Find some transitions
                for i in range(10):
                    if self._abort_requested:
                        break
                        
                    # Try to advance - might not work in test mode
                    try:
                        clock.advance_to_next_edge()
                        self.frames_found += 1
                    except Exception as e:
                        warnings.warn(f"Error advancing: {e}")
                        break
                    
                    # Report progress
                    current_sample = clock.get_sample_number()
                    self.report_progress(current_sample)
        except Exception as e:
            warnings.warn(f"Exception in worker_thread: {e}")
            # In a real analyzer, we'd add decoded frames to self._results here
    
    def needs_rerun(self):
        return False


class SimpleSettings(AnalyzerSettings):
    """Simple settings class for testing."""
    
    def __init__(self):
        super().__init__()
        self.channel = Channel(0, 0)
        
    def get_name(self):
        return "Simple Test Settings"
        
    def _create_cpp_settings(self):
        # This is a stub implementation for testing
        pass


class SimpleAnalyzer(Analyzer):
    """Simple analyzer implementation for basic tests."""
    
    def __init__(self):
        super().__init__()
        self._settings = SimpleSettings()
        
    def _get_analyzer_name(self):
        return "Simple Analyzer"
        
    def _get_minimum_sample_rate_hz(self):
        return 1000000
        
    def worker_thread(self):
        pass
        
    def needs_rerun(self):
        return False


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
        
        # Create analyzer and set up test mode
        analyzer = MinimalAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Check basic properties
        assert analyzer.name == "Minimal Test Analyzer"
        assert analyzer.min_sample_rate == 1000000
        assert analyzer.state == AnalyzerState.IDLE

    def test_analyzer_with_settings(self):
        """Test that settings can be properly initialized and accessed."""
        analyzer = SimpleAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Check that settings are properly set
        assert analyzer.settings is not None
        assert analyzer.settings.get_name() == "Simple Test Settings"


class TestAnalyzerLifecycle:
    """Tests for the analyzer lifecycle (start, stop, etc.)."""
    
    def test_start_analysis_sync(self):
        """Test starting analysis in synchronous mode."""
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Start analysis synchronously
        analyzer.start_analysis(async_mode=False)
        
        # Check that worker thread was called and completed
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.worker_completed, "Worker thread should have completed successfully"
        assert analyzer.worker_exception is None, "No exceptions should have been raised"
        assert analyzer.state == AnalyzerState.COMPLETED, "Final state should be COMPLETED"
    
    def test_start_analysis_async(self):
        """Test starting analysis in asynchronous mode."""
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Start analysis asynchronously
        analyzer.start_analysis(async_mode=True)
        
        # Wait for completion
        analyzer.wait_for_completion(timeout=1.0)
        
        # Check results
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.worker_completed, "Worker thread should have completed successfully"
        assert analyzer.worker_exception is None, "No exceptions should have been raised"
        assert analyzer.state == AnalyzerState.COMPLETED, "Final state should be COMPLETED"
    
    def test_stop_analysis(self):
        """Test stopping a running analysis."""
        analyzer = SlowTestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Start analysis asynchronously
        analyzer.start_analysis(async_mode=True)
        
        # Give it a moment to start
        time.sleep(0.2)
        
        # Stop the analysis
        analyzer.stop_analysis()
        
        # Wait for it to fully stop
        analyzer.wait_for_completion(timeout=1.0)
        
        # Check results
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.state == AnalyzerState.STOPPED, "Final state should be STOPPED"
    
    def test_analysis_error(self):
        """Test error handling during analysis."""
        analyzer = FailingTestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Start analysis asynchronously to catch the error
        analyzer.start_analysis(async_mode=True)
        
        # Wait for completion or error
        with pytest.raises(ValueError) as excinfo:
            analyzer.wait_for_completion(timeout=1.0)
        
        # Check that the correct error was raised
        assert "Simulated worker error" in str(excinfo.value)
        
        # Check results
        assert analyzer.worker_called, "Worker thread should have been called"
        assert not analyzer.worker_completed, "Worker thread should not have completed successfully"
        assert analyzer.state == AnalyzerState.ERROR, "Final state should be ERROR"
    
    def test_analysis_needs_rerun(self):
        """Test handling analyzers that need to be rerun."""
        analyzer = RerunTestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Start analysis
        analyzer.start_analysis(async_mode=False)
        
        # Check results
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.worker_completed, "Worker thread should have completed successfully"
        assert analyzer.rerun_requested, "Rerun should have been requested"
        assert analyzer.rerun_count == 1, "Rerun count should be 1"
        assert analyzer.state == AnalyzerState.COMPLETED, "Final state should be COMPLETED"
    
    def test_context_manager(self):
        """Test using the analyzer as a context manager."""
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Use the analyzer as a context manager
        with analyzer.analysis_session() as session:
            # Check that analysis is running
            assert analyzer.state == AnalyzerState.RUNNING, "Analyzer should be RUNNING in context"
            assert session is analyzer, "Context should return the analyzer instance"
        
        # Check final state
        assert analyzer.state == AnalyzerState.COMPLETED, "Analyzer should be COMPLETED after context"
    
    def test_context_manager_with_error(self):
        """Test error handling in context manager."""
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Use the analyzer as a context manager with an error
        try:
            with analyzer.analysis_session() as session:
                raise RuntimeError("Simulated context error")
        except RuntimeError as e:
            # The error should propagate
            assert "Simulated context error" in str(e)
        
        # Check that analysis was properly stopped
        assert analyzer.state == AnalyzerState.STOPPED, "Analyzer should be STOPPED after context with error"
    
    def test_hardware_required_error(self):
        """Test that requiring hardware raises appropriate error."""
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()
        # Don't connect to hardware
        
        # Attempting to analyze with require_hardware=True should fail
        with pytest.raises(HardwareNotConnectedError):
            analyzer.start_analysis(async_mode=False, require_hardware=True)


class TestAnalyzerCallbacks:
    """Tests for analyzer callback mechanisms."""
    
    def test_progress_callback(self):
        """Test that progress callbacks are properly called."""
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Define a progress callback
        callback_values = []
        def progress_callback(progress):
            callback_values.append(progress)
        
        # Register the callback
        analyzer.add_progress_callback(progress_callback)
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        
        # Check that callback was called
        assert len(callback_values) > 0, "Progress callback should have been called"
        assert len(callback_values) == len(analyzer.progress_values), "Callback should be called for each progress update"
    
    def test_state_callback(self):
        """Test that state callbacks are properly called."""
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Define a state callback
        state_changes = []
        def state_callback(state):
            state_changes.append(state)
        
        # Register the callback
        analyzer.add_state_callback(state_callback)
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        
        # Check that callback was called for each state transition
        assert len(state_changes) >= 3, "State callback should be called at least 3 times"
        assert AnalyzerState.INITIALIZING in state_changes, "INITIALIZING state should be reported"
        assert AnalyzerState.RUNNING in state_changes, "RUNNING state should be reported"
        assert AnalyzerState.COMPLETED in state_changes, "COMPLETED state should be reported"
    
    def test_remove_callback(self):
        """Test that callbacks can be removed."""
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Define two callbacks
        callback1_values = []
        callback2_values = []
        
        def callback1(progress):
            callback1_values.append(progress)
        
        def callback2(progress):
            callback2_values.append(progress)
        
        # Register both callbacks
        analyzer.add_progress_callback(callback1)
        analyzer.add_progress_callback(callback2)
        
        # Remove one callback
        analyzer.remove_progress_callback(callback2)
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        
        # Check that only callback1 was called
        assert len(callback1_values) > 0, "Callback1 should have been called"
        assert len(callback2_values) == 0, "Callback2 should not have been called after removal"


class TestChannelData:
    """Tests for channel data access."""
    
    def test_get_channel_data(self):
        """Test accessing channel data from the analyzer."""
        # Create an SPI analyzer that will access channel data
        analyzer = SpiAnalyzer()
        analyzer.setup_test_mode()
        
        # Update analyzer settings to use our simulation channels
        analyzer._settings.clock_channel = Channel(0, 0)  # Clock channel
        analyzer._settings.mosi_channel = Channel(0, 1)   # Data channel
        
        # Run the analyzer
        analyzer.start_analysis(async_mode=False)
        
        # In test mode without real data, we might not find any frames
        # but the analyzer should run to completion without errors
        assert analyzer.state == AnalyzerState.COMPLETED
    
    def test_get_channel_data_error(self):
        """Test error handling when get_channel_data is called before initialization."""
        # Create an analyzer but don't initialize it
        class MinimalAnalyzer(Analyzer):
            def _get_analyzer_name(self):
                return "Test"
                
            def _get_minimum_sample_rate_hz(self):
                return 1000000
                
            def worker_thread(self):
                pass
                
            def needs_rerun(self):
                return False
        
        analyzer = MinimalAnalyzer()
        
        # Attempt to get channel data should raise error
        with pytest.raises(RuntimeError) as excinfo:
            analyzer.get_channel_data(Channel(0, 0))
        
        assert "not initialized" in str(excinfo.value).lower()


class TestAnalyzerReset:
    """Tests for resetting the analyzer."""
    
    def test_reset(self):
        """Test resetting the analyzer."""
        class ResetTrackingAnalyzer(TestAnalyzer):
            def __init__(self):
                self.init_count = 0
                super().__init__()
                
            def _initialize_analyzer(self):
                super()._initialize_analyzer()
                self.init_count += 1
                
            # Override setup_test_mode to count calls
            def setup_test_mode(self):
                super().setup_test_mode()
                # Track that we've called this
                self.test_mode_setup = True
        
        analyzer = ResetTrackingAnalyzer()
        analyzer._initialize_analyzer()
        analyzer.setup_test_mode()
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        assert analyzer.state == AnalyzerState.COMPLETED
        
        # Get current init count
        initial_init_count = analyzer.init_count
        
        # Reset
        analyzer.reset()
        
        # After reset, we need to set up test mode again
        analyzer.setup_test_mode()
        
        # Check state
        assert analyzer.state == AnalyzerState.IDLE
        assert analyzer.init_count == initial_init_count + 1, "Initialization should be called again during reset"
        
        # Run analysis again after reset
        analyzer.start_analysis(async_mode=False)
        assert analyzer.state == AnalyzerState.COMPLETED, "Analysis should run successfully after reset"


@pytest.mark.integration
class TestIntegrationWithHardware:
    """Integration tests that require actual hardware."""
    
    def detect_hardware(self):
        """Helper method to detect if hardware is connected."""
        # This would be implemented according to your SDK
        # Return True if hardware is detected, False otherwise
        try:
            # Example hardware detection
            return False  # Replace with actual detection code
        except Exception:
            return False
    
    @pytest.mark.skipif(True, reason="Hardware tests are not automated")
    def test_analyzer_with_hardware(self):
        """Test running an analyzer with actual hardware."""
        # Skip if no hardware
        if not self.detect_hardware():
            pytest.skip("No hardware detected")
            
        # Create an analyzer for testing with hardware
        analyzer = SpiAnalyzer()
        
        # Connect to hardware
        connected = analyzer.connect_to_hardware()
        if not connected:
            pytest.skip("Could not connect to hardware")
        
        try:
            # Start analysis
            analyzer.start_analysis(async_mode=True, require_hardware=True)
            
            # Wait for completion or timeout
            completed = analyzer.wait_for_completion(timeout=5.0)
            
            # Check results
            assert completed, "Analysis should complete within timeout"
            assert analyzer.state == AnalyzerState.COMPLETED, "Analysis should complete successfully"
            assert analyzer.frames_found > 0, "Should have found some frames"
            
        finally:
            # Clean up
            if analyzer.state == AnalyzerState.RUNNING:
                analyzer.stop_analysis()
            analyzer.disconnect_from_hardware()


if __name__ == "__main__":
    pytest.main(["-v", __file__])