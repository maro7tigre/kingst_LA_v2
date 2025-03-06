# Kingst Logic Analyzer Python Bindings

Python bindings for the Kingst Logic Analyzer SDK using pybind11.

## Overview

This library provides Python bindings for the Kingst Logic Analyzer, allowing you to control and analyze digital signals directly from Python. The bindings expose the full functionality of the C++ SDK with a Pythonic interface.


## Project  Structure
KINGST_LA_V2
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
│ └── Win64  
│   ├── Analyzer.dll  
│   └── Analyzer.lib  
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
│ ├── __init__.py              # Package exports and version info  
│ ├── analyzer.py              # Main Analyzer class interface  
│ ├── settings.py              # Settings classes and interfaces  
│ ├── results.py               # Results and frame handling  
│ ├── channel.py               # Channel data and management  
│ ├── simulation.py            # Simulation functionality  
│ ├── types.py                 # Enums, constants, and basic types  
│ ├── helpers.py               # Utility functions  
│ └── exceptions.py            # Custom exception classes  
├── tests/  
│ ├── __init__.py  
│ ├── conftest.py                  # Shared fixtures and test configuration  
│ ├── test_types.py                # Tests for types.py (BitState, Channel, etc.)  
│ ├── test_analyzer.py             # Tests for analyzer.py (Analyzer class)  
│ ├── test_settings.py             # Tests for settings.py (AnalyzerSettings classes)  
│ ├── test_results.py              # Tests for results.py (AnalyzerResults, Frame)  
│ ├── test_channel.py              # Tests for channel.py (Channel data functionality)  
│ ├── test_simulation.py           # Tests for simulation.py (SimulationChannelDescriptor)  
│ ├── test_helpers.py              # Tests for helpers.py (Utility functions)  
│ └── integration/                 # Integration tests combining multiple components  
│   ├── __init__.py  
│   ├── test_uart_analysis.py    # Complete UART analyzer test  
│   ├── test_spi_analysis.py     # Complete SPI analyzer test  
│   └── test_i2c_analysis.py     # Complete I2C analyzer test  
├── examples/ # Example scripts  
├── _core_.cp313-win_amd64.pyd  
  
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

## Python API Style Guide for Kingst Logic Analyzer

This style guide covers conventions for the Python API layer that wraps the lower-level C++ bindings for the Kingst Logic Analyzer SDK.

### 1. Module Organization

- **Core Types Module**: Define base types, enums, and constants in a dedicated `types.py`
- **Feature Modules**: Create separate modules for logical functionality groups (e.g., `channel.py`, `analyzer.py`) 
- **Import Hierarchy**: Core modules should not depend on feature modules
- **Internal vs. Public**: Internal implementation modules should be prefixed with underscore (`_core.py`)

### 2. Class Design

- **Wrapper Pattern**: Wrap C++ classes with Pythonic interfaces that hold references to C++ objects
- **Property Access**: Use properties instead of direct attribute access for C++ object fields
- **Class Hierarchy**: Follow the original C++ class hierarchy while adding Pythonic improvements
- **Extension Methods**: Add convenience methods not in the C++ API to improve usability
- **Data Classes**: Use `@dataclass` for simple data container classes (e.g., `Pulse`, `Pattern`)

### 3. Naming Conventions

- **Method Names**: Use `snake_case` for all methods and functions
- **Internal Methods**: Prefix internal methods with underscore (e.g., `_internal_helper`)
- **Constant Names**: Use `UPPER_CASE` for constants
- **Enum Values**: Use `PascalCase` for enum values to match Python conventions
- **Boolean Methods**: Prefix boolean methods with `is_`, `has_`, or `can_`

### 4. Type Hints

- **Complete Coverage**: Include type hints for all function parameters and return types
- **Union Types**: Use `Union[Type1, Type2]` for parameters accepting multiple types
- **Optional Parameters**: Use `Optional[Type]` for parameters that may be None
- **Collections**: Specify collection content types (e.g., `List[BitState]`, `Dict[str, float]`)
- **TypeVar**: Use for generic type specifications where appropriate

### 5. Documentation

- **Module Docstrings**: Begin each module with a comprehensive docstring explaining its purpose
- **Class Docstrings**: Describe the class's purpose, main features, and typical usage
- **Method Docstrings**: Include purpose, parameters, return values, exceptions, and example usage
- **Examples**: Provide concrete examples for complex methods
- **Cross-References**: Reference related classes/methods where helpful

### 6. Error Handling

- **Input Validation**: Validate method inputs with descriptive error messages
- **Specific Exceptions**: Raise specific exception types (ValueError, TypeError, etc.)
- **Context Preservation**: When wrapping C++ exceptions, preserve context information
- **Error Messages**: Write clear, actionable error messages
- **Default Values**: Use sensible defaults to reduce errors where appropriate

### 7. Method Parameters

- **Default Values**: Provide reasonable defaults for optional parameters
- **Named Parameters**: Design APIs to encourage named parameters for clarity
- **Parameter Order**: Place required parameters first, followed by optional ones
- **Boolean Flags**: Use keyword arguments for boolean flags, not positional
- **Validation**: Validate parameters at the API boundary before passing to C++

### 8. Enums and Constants

- **IntEnum**: Use IntEnum for C++ enum wrappers to support both symbolic and numeric comparison
- **String Conversion**: Implement `__str__` for human-readable output and `__repr__` for debugging
- **Helper Methods**: Add utility methods to enums to enhance functionality
- **Documentation**: Document each enum value with examples

### 9. Context Managers

- **Resource Management**: Implement context managers (`__enter__`/`__exit__`) for resource-holding objects
- **State Preservation**: Use context managers for operations that should restore previous state
- **Cleanup**: Ensure proper cleanup in the `__exit__` method

### 10. Integration Features

- **NumPy Integration**: Provide methods to convert to/from NumPy arrays
- **Matplotlib Integration**: Add visualization methods where appropriate
- **Iterator Protocol**: Implement `__iter__` for collections and sequence-like objects
- **Container Types**: Implement appropriate container protocols where it makes sense