# Kingst Logic Analyzer Python Bindings

Python bindings for the Kingst Logic Analyzer SDK using pybind11.

## Overview

This library provides Python bindings for the Kingst Logic Analyzer, allowing you to control and analyze digital signals directly from Python. The bindings expose the full functionality of the C++ SDK with a Pythonic interface.


## Project  Structure
  
├── include/ # Kingst SDK headers Download from : https://www.qdkingst.com/en  
│ ├── Analyzer.h  
│ ├── AnalyzerChannelData.h  
│ ├── AnalyzerHelpers.h  
│ ├── AnalyzerResults.h  
│ ├── AnalyzerSettingInterface.h  
│ ├── AnalyzerSettings.h  
│ ├── AnalyzerTypes.h  
│ ├── LogicPublicTypes.h  
│ └── SimulationChannelDescriptor.h  
├── lib/ # Kingst SDK libraries Download from : https://www.qdkingst.com/en  
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
  
## Pybind11 Binding Style Guide for Kingst LA

1. Method Organization:
   - Group by type: pure virtuals → virtuals → utilities → process control → internal
   - Mark internal methods with underscore prefix (_method_name)
2. Naming:
   - Use snake_case for Python-facing methods
   - Keep original CamelCase in C++ implementation calls
   - Use static_cast<> for overloaded methods
3. Documentation:
   - Use R"pbdoc(...)" for all docstrings
   - Document parameter types, return values, and ownership
   - Include usage notes for complex methods
4. Memory Management:
   - Specify return_value_policy for all pointer returns
   - Use reference policy for borrowed pointers (don't delete)
   - Use take_ownership for transferred ownership
5. Parameters:
   - Always use py::arg("name") for all parameters
   - Document default values in docstrings
6. Trampolines:
   - Implement all virtual methods in trampoline classes
   - Use PYBIND11_OVERRIDE_PURE for pure virtuals
   - Use PYBIND11_OVERRIDE for virtuals with defaults
