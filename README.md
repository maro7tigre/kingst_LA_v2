# Kingst Logic Analyzer Python Bindings

Python bindings for the Kingst Logic Analyzer SDK using pybind11.

## Overview

This library provides Python bindings for the Kingst Logic Analyzer, allowing you to control and analyze digital signals directly from Python. The bindings expose the full functionality of the C++ SDK with a Pythonic interface.


## Project  Structure

├── include/ # Kingst SDK headers
├── lib/ # Kingst SDK libraries
├── src/ # C++ binding code
│ ├── main.cpp                    # Main module definition
│ ├── bind_basic_types.cpp        # Bindings for Channel, BitState, DisplayBase, etc.
│ ├── bind_analyzer.cpp           # Bindings for Analyzer class
│ ├── bind_analyzer_settings.cpp  # Bindings for AnalyzerSettings and setting interfaces
│ ├── bind_analyzer_results.cpp   # Bindings for AnalyzerResults and Frame
│ ├── bind_channel_data.cpp       # Bindings for AnalyzerChannelData
│ ├── bind_simulation.cpp         # Bindings for SimulationChannelDescriptor
│ └── bind_helpers.cpp            # Bindings for AnalyzerHelpers
├── kingst_analyzer/ # Python package
│ ├── init.py
│ ├── utils.py # Helper functions
│ └── ...
├── examples/ # Example scripts
│ ├── basic_capture.py
│ ├── spi_analyzer.py
│ └── ...
├── tests/ # Test cases
├── setup.py # Build configuration
├── pyproject.toml # Project metadata