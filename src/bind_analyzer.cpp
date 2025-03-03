// bind_analyzer.cpp
//
// Python bindings for the Analyzer base class

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "Analyzer.h"
#include "AnalyzerResults.h"
#include "AnalyzerSettings.h"
#include "AnalyzerChannelData.h"
#include "SimulationChannelDescriptor.h"

namespace py = pybind11;

// Trampoline class for Analyzer
// This allows Python classes to inherit from and override the virtual methods
class PyAnalyzer : public Analyzer {
public:
    // Default constructor
    using Analyzer::Analyzer;

    // Virtual destructor
    ~PyAnalyzer() override = default;

    // Override all pure virtual methods with trampolines to Python
    void WorkerThread() override {
        PYBIND11_OVERRIDE_PURE(
            void,                 // Return type
            Analyzer,             // Parent class
            WorkerThread          // Method name
        );
    }

    U32 GenerateSimulationData(U64 newest_sample_requested, U32 sample_rate, 
                              SimulationChannelDescriptor **simulation_channels) override {
        PYBIND11_OVERRIDE_PURE(
            U32,                  // Return type
            Analyzer,             // Parent class
            GenerateSimulationData, // Method name
            newest_sample_requested, sample_rate, simulation_channels // Arguments
        );
    }

    U32 GetMinimumSampleRateHz() override {
        PYBIND11_OVERRIDE_PURE(
            U32,                  // Return type
            Analyzer,             // Parent class
            GetMinimumSampleRateHz // Method name
        );
    }

    const char* GetAnalyzerName() const override {
        PYBIND11_OVERRIDE_PURE(
            const char*,          // Return type
            Analyzer,             // Parent class
            GetAnalyzerName       // Method name
        );
    }

    bool NeedsRerun() override {
        PYBIND11_OVERRIDE_PURE(
            bool,                 // Return type
            Analyzer,             // Parent class
            NeedsRerun            // Method name
        );
    }

    // Override non-pure virtual methods
    void SetupResults() override {
        PYBIND11_OVERRIDE(
            void,                 // Return type
            Analyzer,             // Parent class
            SetupResults          // Method name
        );
    }

    const char* GetAnalyzerVersion() const override {
        PYBIND11_OVERRIDE(
            const char*,          // Return type
            Analyzer,             // Parent class
            GetAnalyzerVersion    // Method name
        );
    }
};

void init_analyzer(py::module_ &m) {
    // Define the Analyzer class
    py::class_<Analyzer, PyAnalyzer> analyzer(m, "Analyzer", R"pbdoc(
        Base class for implementing protocol analyzers.
        
        This is an abstract base class that must be subclassed to create custom protocol analyzers.
        The key methods that must be implemented are:
        
        - worker_thread(): Contains the main analysis logic
        - generate_simulation_data(): Creates simulated data for testing
        - get_minimum_sample_rate_hz(): Returns minimum required sample rate
        - get_analyzer_name(): Returns analyzer name
        - needs_rerun(): Checks if analysis needs to be rerun
    )pbdoc");

    // Add constructors
    analyzer.def(py::init<>(), "Default constructor");

    // Add pure virtual methods that must be implemented by derived classes
    analyzer.def("worker_thread", &Analyzer::WorkerThread, R"pbdoc(
        Main worker thread for the analyzer.
        
        This method contains the main analysis logic and is called when the analyzer is run.
        It should process the channel data, detect protocol features, and add frames to the results.
        
        Must be implemented by derived classes.
    )pbdoc");

    analyzer.def("generate_simulation_data", &Analyzer::GenerateSimulationData, R"pbdoc(
        Generate simulated data for testing the analyzer.
        
        Args:
            newest_sample_requested: The sample number up to which to generate data
            sample_rate: The sample rate in Hz
            simulation_channels: Array of simulation channel descriptors
            
        Returns:
            U32: The sample rate used for simulation
            
        Must be implemented by derived classes.
    )pbdoc", 
    py::arg("newest_sample_requested"), 
    py::arg("sample_rate"), 
    py::arg("simulation_channels"));

    analyzer.def("get_minimum_sample_rate_hz", &Analyzer::GetMinimumSampleRateHz, R"pbdoc(
        Get the minimum sample rate required for this analyzer.
        
        Returns:
            U32: Minimum sample rate in Hz
            
        Must be implemented by derived classes.
    )pbdoc");

    analyzer.def("get_analyzer_name", &Analyzer::GetAnalyzerName, R"pbdoc(
        Get the name of the analyzer.
        
        Returns:
            str: Analyzer name
            
        Must be implemented by derived classes.
    )pbdoc");

    analyzer.def("needs_rerun", &Analyzer::NeedsRerun, R"pbdoc(
        Check if the analyzer needs to be rerun.
        
        Returns:
            bool: True if the analyzer needs to be rerun
            
        Must be implemented by derived classes.
    )pbdoc");

    // Add virtual methods with default implementations
    analyzer.def("setup_results", &Analyzer::SetupResults, R"pbdoc(
        Set up the analyzer results.
        
        This method is called before the analyzer is run to set up the results.
        The default implementation does nothing.
    )pbdoc");

    analyzer.def("get_analyzer_version", &Analyzer::GetAnalyzerVersion, R"pbdoc(
        Get the analyzer version.
        
        Returns:
            str: Analyzer version string
            
        The default implementation returns a version based on the build date.
    )pbdoc");

    // Add non-virtual utility methods
    analyzer.def("set_analyzer_settings", &Analyzer::SetAnalyzerSettings, R"pbdoc(
        Set the analyzer settings.
        
        Args:
            settings: Analyzer settings object
    )pbdoc", py::arg("settings"));

    analyzer.def("get_analyzer_channel_data", &Analyzer::GetAnalyzerChannelData, R"pbdoc(
        Get channel data for analysis.
        
        Args:
            channel: Channel to get data for
            
        Returns:
            AnalyzerChannelData: Channel data object
            
        Note: Do not delete the returned pointer.
    )pbdoc", py::arg("channel"), py::return_value_policy::reference);

    analyzer.def("report_progress", &Analyzer::ReportProgress, R"pbdoc(
        Report progress to the UI.
        
        Args:
            sample_number: Current sample number being processed
    )pbdoc", py::arg("sample_number"));

    analyzer.def("set_analyzer_results", &Analyzer::SetAnalyzerResults, R"pbdoc(
        Set the analyzer results object.
        
        Args:
            results: Analyzer results object
    )pbdoc", py::arg("results"));

    analyzer.def("get_simulation_sample_rate", &Analyzer::GetSimulationSampleRate, R"pbdoc(
        Get the sample rate used for simulation.
        
        Returns:
            U32: Simulation sample rate in Hz
    )pbdoc");

    analyzer.def("get_sample_rate", &Analyzer::GetSampleRate, R"pbdoc(
        Get the current sample rate.
        
        Returns:
            U32: Sample rate in Hz
    )pbdoc");

    analyzer.def("get_trigger_sample", &Analyzer::GetTriggerSample, R"pbdoc(
        Get the trigger sample number.
        
        Returns:
            U64: Trigger sample number
    )pbdoc");

    analyzer.def("check_if_thread_should_exit", &Analyzer::CheckIfThreadShouldExit, R"pbdoc(
        Check if the worker thread should exit.
        
        Call this periodically in long-running operations to check if the
        thread should exit (e.g., if the user cancels the analysis).
    )pbdoc");

    analyzer.def("get_analyzer_progress", &Analyzer::GetAnalyzerProgress, R"pbdoc(
        Get the current analyzer progress.
        
        Returns:
            double: Progress as a value between 0.0 and 1.0
    )pbdoc");
    
    // Process control methods
    analyzer.def("start_processing", static_cast<void (Analyzer::*)()>(&Analyzer::StartProcessing), R"pbdoc(
        Start processing from the beginning.
        
        This method starts the analyzer worker thread to begin processing data.
    )pbdoc");
    
    analyzer.def("start_processing_from", static_cast<void (Analyzer::*)(U64)>(&Analyzer::StartProcessing), R"pbdoc(
        Start processing from a specific sample.
        
        Args:
            starting_sample: Sample number to start processing from
    )pbdoc", py::arg("starting_sample"));
    
    analyzer.def("stop_worker_thread", &Analyzer::StopWorkerThread, R"pbdoc(
        Stop the worker thread.
        
        This method stops the analyzer worker thread.
    )pbdoc");
    
    // Internal methods (exposed for completeness but marked with underscore)
    analyzer.def("_kill_thread", &Analyzer::KillThread, "Internal: Kill the worker thread");
    analyzer.def("_set_thread_must_exit", &Analyzer::SetThreadMustExit, "Internal: Set flag that thread must exit");
    analyzer.def("_get_analyzer_settings", &Analyzer::GetAnalyzerSettings, "Internal: Get analyzer settings", 
                py::return_value_policy::reference);
    analyzer.def("_does_analyzer_use_device", &Analyzer::DoesAnalyzerUseDevice, "Internal: Check if analyzer uses device", 
                py::arg("device_id"));
    analyzer.def("_is_valid", &Analyzer::IsValid, "Internal: Check if channels are valid", 
                py::arg("channel_array"), py::arg("count"));
    analyzer.def("_initial_worker_thread", &Analyzer::InitialWorkerThread, "Internal: Initialize worker thread");
    analyzer.def("_get_analyzer_results", &Analyzer::GetAnalyzerResults, "Internal: Get analyzer results", 
                py::arg("analyzer_results"));
    
    // Add attribute access for mData if needed
    // Since mData is protected and a struct AnalyzerData, consider exposing needed properties
    // through dedicated Python properties or methods if required
}