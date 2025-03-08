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
                self.should_abort()
                
                # Report progress
                self.report_progress(i * 100)
                
                # Short sleep
                time.sleep(0.01)
            
            self.worker_completed = True
        except Exception as e:
            self.worker_exception = e
            raise
    
    def needs_rerun(self):
        return False
    
    # Override report_progress to track values
    def report_progress(self, sample_number):
        # Call the real implementation
        super().report_progress(sample_number)
        
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
            for i in range(20):
                self.should_abort()
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
def mock_cpp_backend():
    """Create a patch for the C++ backend to avoid actual C++ calls."""
    with patch('kingst_analyzer._core.Analyzer') as mock_analyzer:
        # Set up the mock to track method calls
        instance = mock_analyzer.return_value
        instance.get_analyzer_name.return_value = "Test Mock Analyzer"
        instance.get_minimum_sample_rate_hz.return_value = 1000000
        instance.get_analyzer_progress.return_value = 0.0
        instance.get_sample_rate.return_value = 10000000
        instance.get_trigger_sample.return_value = 0
        
        yield mock_analyzer


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

    def test_analyzer_initialization_error(self, mock_cpp_backend):
        """Test that errors during analyzer initialization are properly handled."""
        
        # Make the C++ initialization fail
        mock_cpp_backend.side_effect = RuntimeError("Simulated C++ initialization error")
        
        class SimpleAnalyzer(Analyzer):
            def _get_analyzer_name(self):
                return "Simple Analyzer"
                
            def _get_minimum_sample_rate_hz(self):
                return 1000000
                
            def worker_thread(self):
                pass
                
            def needs_rerun(self):
                return False
        
        # Should raise an AnalyzerInitError
        with pytest.raises(AnalyzerInitError) as excinfo:
            analyzer = SimpleAnalyzer()
            analyzer._initialize_analyzer()  # Force initialization
            
        assert "initialization" in str(excinfo.value).lower()
        assert "error" in str(excinfo.value).lower()

    def test_analyzer_with_settings(self, mock_cpp_backend):
        """Test that settings can be properly initialized and accessed."""
        
        # Create a simple settings class
        class SimpleSettings(AnalyzerSettings):
            def __init__(self):
                super().__init__()
                self.channel = Channel(0, 0)
                
            def get_name(self):
                return "Simple Test Settings"
        
        class SimpleAnalyzer(Analyzer):
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
        
        analyzer = SimpleAnalyzer()
        
        # Check that settings are properly set
        assert analyzer.settings is not None
        assert analyzer.settings.get_name() == "Simple Test Settings"
        
        # Check that settings are passed to the C++ analyzer
        analyzer._initialize_analyzer()
        mock_cpp_backend.return_value.set_analyzer_settings.assert_called_once()


class TestAnalyzerLifecycle:
    """Tests for the analyzer lifecycle (start, stop, etc.)."""
    
    def test_start_analysis_sync(self, mock_cpp_backend):
        """Test starting analysis in synchronous mode."""
        analyzer = TestAnalyzer()
        
        # Start analysis synchronously
        analyzer.start_analysis(async_mode=False)
        
        # Check that worker thread was called and completed
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.worker_completed, "Worker thread should have completed successfully"
        assert analyzer.worker_exception is None, "No exceptions should have been raised"
        assert analyzer.state == AnalyzerState.COMPLETED, "Final state should be COMPLETED"
        
        # Check C++ interaction
        mock_cpp_backend.return_value.start_processing.assert_called_once()
    
    def test_start_analysis_async(self, mock_cpp_backend):
        """Test starting analysis in asynchronous mode."""
        analyzer = TestAnalyzer()
        
        # Start analysis asynchronously
        analyzer.start_analysis(async_mode=True)
        
        # Wait for completion
        analyzer.wait_for_completion(timeout=1.0)
        
        # Check results
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.worker_completed, "Worker thread should have completed successfully"
        assert analyzer.worker_exception is None, "No exceptions should have been raised"
        assert analyzer.state == AnalyzerState.COMPLETED, "Final state should be COMPLETED"
        
        # Check C++ interaction
        mock_cpp_backend.return_value.start_processing.assert_called_once()
    
    def test_stop_analysis(self, mock_cpp_backend):
        """Test stopping a running analysis."""
        analyzer = SlowTestAnalyzer()
        
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
        assert not analyzer.worker_completed, "Worker thread should not have completed successfully"
        assert analyzer.state == AnalyzerState.STOPPED, "Final state should be STOPPED"
        
        # Check C++ interaction
        mock_cpp_backend.return_value.start_processing.assert_called_once()
        mock_cpp_backend.return_value.stop_worker_thread.assert_called_once()
    
    def test_analysis_error(self, mock_cpp_backend):
        """Test error handling during analysis."""
        analyzer = FailingTestAnalyzer()
        
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
    
    def test_analysis_needs_rerun(self, mock_cpp_backend):
        """Test handling analyzers that need to be rerun."""
        analyzer = RerunTestAnalyzer()
        
        # Start analysis
        analyzer.start_analysis(async_mode=False)
        
        # Check results
        assert analyzer.worker_called, "Worker thread should have been called"
        assert analyzer.worker_completed, "Worker thread should have completed successfully"
        assert analyzer.rerun_requested, "Rerun should have been requested"
        assert analyzer.rerun_count == 1, "Rerun count should be 1"
        assert analyzer.state == AnalyzerState.COMPLETED, "Final state should be COMPLETED"
        
        # Check C++ interaction - startProcessing should have been called twice
        assert mock_cpp_backend.return_value.start_processing.call_count == 2, "StartProcessing should be called twice for a rerun"
    
    def test_context_manager(self, mock_cpp_backend):
        """Test using the analyzer as a context manager."""
        analyzer = TestAnalyzer()
        
        # Use the analyzer as a context manager
        with analyzer.analysis_session() as session:
            # Check that analysis is running
            assert analyzer.state == AnalyzerState.RUNNING, "Analyzer should be RUNNING in context"
            assert session is analyzer, "Context should return the analyzer instance"
        
        # Check final state
        assert analyzer.state == AnalyzerState.COMPLETED, "Analyzer should be COMPLETED after context"
        
        # Check C++ interaction
        mock_cpp_backend.return_value.start_processing.assert_called_once()
    
    def test_context_manager_with_error(self, mock_cpp_backend):
        """Test error handling in context manager."""
        analyzer = TestAnalyzer()
        
        # Use the analyzer as a context manager with an error
        try:
            with analyzer.analysis_session() as session:
                raise RuntimeError("Simulated context error")
        except RuntimeError as e:
            # The error should propagate
            assert "Simulated context error" in str(e)
        
        # Check that analysis was properly stopped
        assert analyzer.state == AnalyzerState.STOPPED, "Analyzer should be STOPPED after context with error"
        
        # Check C++ interaction
        mock_cpp_backend.return_value.start_processing.assert_called_once()
        mock_cpp_backend.return_value.stop_worker_thread.assert_called_once()


class TestAnalyzerCallbacks:
    """Tests for analyzer callback mechanisms."""
    
    def test_progress_callback(self, mock_cpp_backend):
        """Test that progress callbacks are properly called."""
        analyzer = TestAnalyzer()
        
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
    
    def test_state_callback(self, mock_cpp_backend):
        """Test that state callbacks are properly called."""
        analyzer = TestAnalyzer()
        
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
    
    def test_remove_callback(self, mock_cpp_backend):
        """Test that callbacks can be removed."""
        analyzer = TestAnalyzer()
        
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
    
    def test_callback_error_handling(self, mock_cpp_backend):
        """Test that errors in callbacks are handled gracefully."""
        analyzer = TestAnalyzer()
        
        # Define a problematic callback
        def error_callback(progress):
            raise ValueError("Simulated callback error")
        
        # Define a normal callback
        normal_called = False
        def normal_callback(progress):
            nonlocal normal_called
            normal_called = True
        
        # Register both callbacks
        analyzer.add_progress_callback(error_callback)
        analyzer.add_progress_callback(normal_callback)
        
        # Run analysis - should issue warning but not crash
        with pytest.warns(UserWarning):
            analyzer.start_analysis(async_mode=False)
        
        # Check that normal callback still ran despite error in other callback
        assert normal_called, "Normal callback should still run despite error in another callback"
        assert analyzer.state == AnalyzerState.COMPLETED, "Analysis should complete despite callback error"


class TestChannelData:
    """Tests for channel data access."""
    
    def test_get_channel_data(self, mock_cpp_backend):
        """Test accessing channel data from the analyzer."""
        # Set up mock channel data
        mock_channel_data = MagicMock()
        mock_channel_data.get_bit_state.return_value = BitState.HIGH
        mock_channel_data.get_sample_number.return_value = 0
        
        # Configure mock to return our mock channel data
        mock_cpp_backend.return_value.get_analyzer_channel_data.return_value = mock_channel_data
        
        # Create analyzer and test
        analyzer = TestAnalyzer()
        analyzer._initialize_analyzer()  # Ensure initialization
        
        # Get channel data for a test channel
        channel = Channel(0, 0)
        channel_data = analyzer.get_channel_data(channel)
        
        # Verify it's the mock we created
        assert channel_data is mock_channel_data, "Should return the mocked channel data"
        
        # Check that the C++ method was called correctly
        mock_cpp_backend.return_value.get_analyzer_channel_data.assert_called_once_with(channel)
    
    def test_get_channel_data_error(self):
        """Test error handling when get_channel_data is called before initialization."""
        analyzer = TestAnalyzer()
        # Don't initialize analyzer._analyzer
        
        # Attempt to get channel data should raise error
        with pytest.raises(RuntimeError) as excinfo:
            analyzer.get_channel_data(Channel(0, 0))
        
        assert "not initialized" in str(excinfo.value).lower()


class TestAnalyzerReset:
    """Tests for resetting the analyzer."""
    
    def test_reset(self, mock_cpp_backend):
        """Test resetting the analyzer."""
        class ResetTrackingAnalyzer(TestAnalyzer):
            def __init__(self):
                self.init_count = 0
                super().__init__()
                
            def _initialize_analyzer(self):
                super()._initialize_analyzer()
                self.init_count += 1
        
        analyzer = ResetTrackingAnalyzer()
        
        # Run analysis
        analyzer.start_analysis(async_mode=False)
        assert analyzer.state == AnalyzerState.COMPLETED
        
        # Get current init count
        initial_init_count = analyzer.init_count
        
        # Reset
        analyzer.reset()
        
        # Check state
        assert analyzer.state == AnalyzerState.IDLE
        assert analyzer.init_count == initial_init_count + 1, "Initialization should be called again during reset"
        
        # Run analysis again after reset
        analyzer.start_analysis(async_mode=False)
        assert analyzer.state == AnalyzerState.COMPLETED, "Analysis should run successfully after reset"


class TestNestedAnalyzers:
    """Tests for analyzers that use other analyzers."""
    
    def test_nested_analyzers(self, mock_cpp_backend):
        """Test using analyzers within other analyzers."""
        
        class InnerAnalyzer(TestAnalyzer):
            def __init__(self):
                super().__init__()
                self.data_processed = 0
                
            def worker_thread(self):
                super().worker_thread()
                # Simulate processing data
                self.data_processed = 100
        
        class OuterAnalyzer(TestAnalyzer):
            def __init__(self):
                super().__init__()
                self.inner = None
                self.combined_result = 0
                
            def worker_thread(self):
                self.worker_called = True
                
                # Create and run an inner analyzer
                self.inner = InnerAnalyzer()
                self.inner.start_analysis(async_mode=False)
                
                # Use results from inner analyzer
                self.combined_result = self.inner.data_processed * 2
                
                # Mark as completed
                self.worker_completed = True
        
        # Create and run outer analyzer
        analyzer = OuterAnalyzer()
        analyzer.start_analysis(async_mode=False)
        
        # Check results
        assert analyzer.worker_completed, "Outer analyzer should complete"
        assert analyzer.inner is not None, "Inner analyzer should be created"
        assert analyzer.inner.worker_completed, "Inner analyzer should complete"
        assert analyzer.inner.data_processed == 100, "Inner analyzer should process data"
        assert analyzer.combined_result == 200, "Outer analyzer should use inner analyzer's results"


@pytest.mark.integration
class TestRealCPPIntegration:
    """
    Tests that actually use the real C++ implementation.
    These are integration tests and should be run when testing full integration.
    """
    
    def test_real_cpp_initialization(self):
        """Test initializing the analyzer with the real C++ backend."""
        try:
            # Import the real C++ module
            import kingst_analyzer._core
            
            class RealAnalyzer(TestAnalyzer):
                pass
            
            # Create the analyzer - this should use the real C++ backend
            analyzer = RealAnalyzer()
            
            # Initialize the analyzer
            analyzer._initialize_analyzer()
            
            # Check that we got a real C++ analyzer object
            assert analyzer._analyzer is not None
            assert not isinstance(analyzer._analyzer, MagicMock)
            
            # Try calling a method on the C++ analyzer
            version = analyzer._analyzer.get_analyzer_version()
            
            # Basic sanity check - version should be a string
            assert isinstance(version, str)
            
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Real C++ backend not available: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])