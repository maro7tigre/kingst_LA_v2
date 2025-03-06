"""
Shared test fixtures for Kingst Logic Analyzer Python binding tests.
"""

import pytest
import os
import sys
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

# You can add more fixtures as needed for specific test modules