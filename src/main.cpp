/**
 * main.cpp
 * 
 * Main entry point for the Kingst Logic Analyzer Python bindings.
 * This file defines the Python module structure and initializes
 * all the binding components.
 */

// pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // Support for C++ STL types
#include <pybind11/functional.h>    // Support for C++ functions/lambdas
#include <pybind11/operators.h>     // Support for operator overloading

// Kingst SDK headers
#include "LogicPublicTypes.h"
#include "AnalyzerTypes.h"
#include "Analyzer.h"
#include "AnalyzerSettings.h"
#include "AnalyzerSettingInterface.h"
#include "AnalyzerChannelData.h"
#include "AnalyzerResults.h"
#include "SimulationChannelDescriptor.h"
#include "AnalyzerHelpers.h"

namespace py = pybind11;

// Forward declarations for module initialization functions
void init_basic_types(py::module_ &m);      // Binds fundamental types and enums 
void init_analyzer(py::module_ &m);         // Binds the Analyzer base class
void init_analyzer_settings(py::module_ &m); // Binds settings classes
void init_analyzer_results(py::module_ &m); // Binds results and frame classes
void init_channel_data(py::module_ &m);     // Binds channel data access
void init_simulation(py::module_ &m);       // Binds simulation data generation
void init_helpers(py::module_ &m);          // Binds helper utilities

/**
 * Define the Python module
 */
PYBIND11_MODULE(kingst_analyzer, m) {
    m.doc() = R"pbdoc(
        Kingst Logic Analyzer Python Bindings
        -------------------------------------

        This module provides Python bindings for the Kingst Logic Analyzer SDK,
        allowing developers to create custom protocol analyzers in Python.
        
        Key components:
        - Channel: Represents a physical channel on the logic analyzer
        - BitState: Represents logic states (HIGH/LOW)
        - DisplayBase: Controls how numbers are displayed (Binary, Decimal, Hex, etc.)
        - Analyzer: Base class for implementing protocol analyzers
        - AnalyzerSettings: Configuration for analyzers
        - AnalyzerResults: Storing and displaying analyzer results
        - AnalyzerChannelData: Accessing sampled data from channels
        - SimulationChannelDescriptor: Creating simulation data for testing
        - AnalyzerHelpers: Utility functions for analyzers
        
        Typical workflow:
        1. Create a custom analyzer class that inherits from Analyzer
        2. Implement required methods (WorkerThread, GetAnalyzerName, etc.)
        3. Create settings class for your analyzer
        4. Process channel data in the WorkerThread method
        5. Generate results and add frames to display in the UI

        Example:
            import kingst_analyzer as ka
            
            # Define channels
            data_channel = ka.Channel(0, 0)  # Device 0, Channel 0
            clock_channel = ka.Channel(0, 1) # Device 0, Channel 1
            
            # Access data
            data = analyzer.get_channel_data(data_channel)
            while data.get_sample_number() < end_sample:
                data.advance_to_next_edge()
                # Process data...
    )pbdoc";
    
    // Initialize all binding submodules
    init_basic_types(m);
    init_analyzer(m);
    init_analyzer_settings(m);
    init_analyzer_results(m);
    init_channel_data(m);  
    init_simulation(m);
    init_helpers(m);

    // Module version information
    m.attr("__version__") = "0.2.0";
    m.attr("__author__") = "maro7tiger";
    m.attr("__license__") = "GNU General Public License v3 (GPLv3)";
    
    // Add module-level constants
    m.attr("UNDEFINED_CHANNEL") = UNDEFINED_CHANNEL;
    m.attr("DISPLAY_AS_ERROR_FLAG") = DISPLAY_AS_ERROR_FLAG;
    m.attr("DISPLAY_AS_WARNING_FLAG") = DISPLAY_AS_WARNING_FLAG;
    m.attr("INVALID_RESULT_INDEX") = INVALID_RESULT_INDEX;

}