# In tests/conftest.py
import os
import sys
import pytest
from pathlib import Path

# Add the package root to the Python path if running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kingst_analyzer
    from kingst_analyzer.types import BitState, Channel, DisplayBase
    from kingst_analyzer.analyzer import Analyzer
    from kingst_analyzer.settings import AnalyzerSettings
    from kingst_analyzer.simulation import SimulationManager
except ImportError as e:
    pytest.skip(f"Could not import kingst_analyzer: {e}", allow_module_level=True)

# Configure constants for testing
SAMPLE_RATE_HZ = 10_000_000  # 10MHz

@pytest.fixture
def simulation_manager():
    """Return a SimulationManager with a default sample rate."""
    return SimulationManager(sample_rate=SAMPLE_RATE_HZ)

@pytest.fixture
def sample_channel():
    """Return a sample Channel for testing."""
    return Channel(device_id=0, channel_index=0)

@pytest.fixture
def bit_sequence():
    """Return a sample sequence of bits for testing."""
    return [BitState.HIGH, BitState.LOW, BitState.HIGH, BitState.LOW, BitState.HIGH]

@pytest.fixture
def byte_sequence():
    """Return a sample sequence of bytes for testing."""
    return b'\x55\xAA\x00\xFF'  # 01010101 10101010 00000000 11111111

# Hardware detection and setup functions
def is_hardware_connected():
    """Check if Kingst Logic Analyzer hardware is connected."""
    try:
        from kingst_analyzer import _core
        # This would need to be implemented based on your SDK's device detection
        # For now just return False to avoid hardware dependency in tests
        return False
    except Exception:
        return False

@pytest.fixture(scope="session")
def hardware_device():
    """Fixture to provide access to hardware device if available."""
    if not is_hardware_connected():
        pytest.skip("No Kingst hardware connected")
    
    try:
        # Initialize hardware and return device handle
        # The exact implementation depends on your C++ SDK
        from kingst_analyzer import _core
        # Example: device = _core.DeviceManager().get_device(0)
        device = None  # Replace with actual device initialization
        return device
    except Exception as e:
        pytest.skip(f"Failed to initialize hardware: {e}")
        
@pytest.fixture
def initialize_with_hardware(hardware_device):
    """Fixture to initialize analyzer with actual hardware."""
    def _initialize(analyzer):
        """Initialize the analyzer with real hardware."""
        if not hasattr(analyzer, '_analyzer') or not analyzer._analyzer:
            analyzer._initialize_analyzer()
            
        # Connect the analyzer to the hardware
        return analyzer.connect_to_hardware(hardware_device)
    
    return _initialize

# Test execution markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark tests that require hardware integration"
    )