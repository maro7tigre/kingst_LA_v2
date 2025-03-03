#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

// Include all Kingst SDK headers
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

PYBIND11_MODULE(kingst_analyzer, m) {
    m.doc() = R"pbdoc(
        Kingst Logic Analyzer Python Bindings
        -------------------------------------

        This module provides Python bindings for the Kingst Logic Analyzer SDK,
        allowing developers to create custom protocol analyzers in Python.
        
        Key components:
        - Channel: Represents a physical channel on the logic analyzer
        - Analyzer: Base class for implementing protocol analyzers
        - AnalyzerSettings: Configuration for analyzers
        - AnalyzerResults: Storing and displaying analyzer results
        - AnalyzerChannelData: Accessing sampled data from channels
    )pbdoc";
    
    // Initialize all binding submodules
    init_basic_types(m);
    init_analyzer(m);
    init_analyzer_settings(m);
    init_analyzer_results(m);
    init_channel_data(m);  
    init_simulation(m);
    init_helpers(m);

    // Module version
    m.attr("__version__") = "0.2.0";
}
