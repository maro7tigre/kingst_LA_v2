// bind_helpers.cpp
//
// Python bindings for the AnalyzerHelpers class and related utility classes

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "AnalyzerHelpers.h"
#include "LogicPublicTypes.h"

namespace py = pybind11;

void init_helpers(py::module_ &m) {
    // AnalyzerHelpers static class
    py::class_<AnalyzerHelpers> analyzer_helpers(m, "AnalyzerHelpers", R"pbdoc(
        Static utility class with helper functions for analyzers.
        
        This class provides various utility functions for common tasks in analyzers,
        such as number formatting, bit manipulation, and file operations.
    )pbdoc");

    // Make the class non-constructible from Python
    analyzer_helpers.def(py::init<>(), "Constructor not available");

    // Bit manipulation utilities
    m.def("is_even", &AnalyzerHelpers::IsEven, R"pbdoc(
        Check if a number is even.
        
        Args:
            value: Number to check
            
        Returns:
            bool: True if the number is even
    )pbdoc", py::arg("value"));

    m.def("is_odd", &AnalyzerHelpers::IsOdd, R"pbdoc(
        Check if a number is odd.
        
        Args:
            value: Number to check
            
        Returns:
            bool: True if the number is odd
    )pbdoc", py::arg("value"));

    m.def("get_ones_count", &AnalyzerHelpers::GetOnesCount, R"pbdoc(
        Count the number of bits set to 1 in a value.
        
        Args:
            value: Value to check
            
        Returns:
            U32: Number of bits set to 1
    )pbdoc", py::arg("value"));

    m.def("diff32", &AnalyzerHelpers::Diff32, R"pbdoc(
        Calculate the absolute difference between two 32-bit values.
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            U32: Absolute difference |a - b|
    )pbdoc", py::arg("a"), py::arg("b"));

    // String formatting
    m.def("get_number_string", [](U64 number, DisplayBase display_base, U32 num_data_bits) {
        char result_string[128];
        AnalyzerHelpers::GetNumberString(number, display_base, num_data_bits, result_string, sizeof(result_string));
        return std::string(result_string);
    }, R"pbdoc(
        Convert a number to a string using the specified display base.
        
        Args:
            number: Number to convert
            display_base: Base to display in (Binary, Decimal, Hexadecimal, etc.)
            num_data_bits: Number of data bits in the value
            
        Returns:
            str: Formatted string representation
    )pbdoc", py::arg("number"), py::arg("display_base"), py::arg("num_data_bits"));

    m.def("get_time_string", [](U64 sample, U64 trigger_sample, U32 sample_rate_hz) {
        char result_string[128];
        AnalyzerHelpers::GetTimeString(sample, trigger_sample, sample_rate_hz, result_string, sizeof(result_string));
        return std::string(result_string);
    }, R"pbdoc(
        Convert a sample number to a time string.
        
        Args:
            sample: Sample number to convert
            trigger_sample: Trigger sample number (reference point)
            sample_rate_hz: Sample rate in Hz
            
        Returns:
            str: Formatted time string
    )pbdoc", py::arg("sample"), py::arg("trigger_sample"), py::arg("sample_rate_hz"));

    // Debugging
    m.def("assert_true", [](bool condition, const char *message) {
        if (!condition) {
            AnalyzerHelpers::Assert(message);
        }
    }, R"pbdoc(
        Assert that a condition is true.
        
        Args:
            condition: Condition to check
            message: Error message if condition is false
    )pbdoc", py::arg("condition"), py::arg("message"));

    // Simulation helpers
    m.def("adjust_simulation_target_sample", &AnalyzerHelpers::AdjustSimulationTargetSample, R"pbdoc(
        Adjust a target sample number for simulation.
        
        This function adjusts a target sample number based on different sample rates
        between simulation and normal operation.
        
        Args:
            target_sample: Target sample number to adjust
            sample_rate: Normal sample rate
            simulation_sample_rate: Simulation sample rate
            
        Returns:
            U64: Adjusted sample number
    )pbdoc", py::arg("target_sample"), py::arg("sample_rate"), py::arg("simulation_sample_rate"));

    // Channel manipulation
    m.def("do_channels_overlap", [](const std::vector<Channel> &channels) {
        return AnalyzerHelpers::DoChannelsOverlap(channels.data(), channels.size());
    }, R"pbdoc(
        Check if any channels overlap.
        
        Args:
            channels: List of channels to check
            
        Returns:
            bool: True if any channels overlap
    )pbdoc", py::arg("channels"));

    // File operations
    m.def("save_file", [](const char *file_name, const std::vector<U8> &data, bool is_binary) {
        AnalyzerHelpers::SaveFile(file_name, data.data(), data.size(), is_binary);
    }, R"pbdoc(
        Save data to a file.
        
        Args:
            file_name: Name of the file to save
            data: Data to save
            is_binary: True for binary file, False for text file
    )pbdoc", py::arg("file_name"), py::arg("data"), py::arg("is_binary") = false);

    // File streaming operations
    m.def("start_file", &AnalyzerHelpers::StartFile, R"pbdoc(
        Start writing to a file.
        
        Args:
            file_name: Name of the file to write
            is_binary: True for binary file, False for text file
            
        Returns:
            object: File handle to use with append_to_file and end_file
    )pbdoc", py::arg("file_name"), py::arg("is_binary") = false);

    m.def("append_to_file", [](const std::vector<U8> &data, void *file) {
        AnalyzerHelpers::AppendToFile(data.data(), data.size(), file);
    }, R"pbdoc(
        Append data to an open file.
        
        Args:
            data: Data to append
            file: File handle from start_file
    )pbdoc", py::arg("data"), py::arg("file"));

    m.def("end_file", &AnalyzerHelpers::EndFile, R"pbdoc(
        Finish writing to a file and close it.
        
        Args:
            file: File handle from start_file
    )pbdoc", py::arg("file"));

    // Number conversion
    m.def("convert_to_signed_number", &AnalyzerHelpers::ConvertToSignedNumber, R"pbdoc(
        Convert an unsigned number to a signed number with the specified bit width.
        
        Args:
            number: Unsigned number to convert
            num_bits: Number of bits in the value
            
        Returns:
            S64: Signed representation of the number
    )pbdoc", py::arg("number"), py::arg("num_bits"));

    // Additional convenience functions for Python
    m.def("format_number", [](U64 number, DisplayBase display_base, U32 num_data_bits) {
        char result_string[128];
        AnalyzerHelpers::GetNumberString(number, display_base, num_data_bits, result_string, sizeof(result_string));
        return std::string(result_string);
    }, R"pbdoc(
        Format a number as a string using the specified display base.
        
        Args:
            number: Number to format
            display_base: Base to display in (Binary, Decimal, Hexadecimal, etc.)
            num_data_bits: Number of data bits in the value
            
        Returns:
            str: Formatted string representation
    )pbdoc", py::arg("number"), py::arg("display_base"), py::arg("num_data_bits"));

    m.def("format_time", [](U64 sample, U64 trigger_sample, U32 sample_rate_hz) {
        char result_string[128];
        AnalyzerHelpers::GetTimeString(sample, trigger_sample, sample_rate_hz, result_string, sizeof(result_string));
        return std::string(result_string);
    }, R"pbdoc(
        Format a sample number as a time string.
        
        Args:
            sample: Sample number to format
            trigger_sample: Trigger sample number (reference point)
            sample_rate_hz: Sample rate in Hz
            
        Returns:
            str: Formatted time string
    )pbdoc", py::arg("sample"), py::arg("trigger_sample"), py::arg("sample_rate_hz"));

    // ClockGenerator class
    py::class_<ClockGenerator> clock_generator(m, "ClockGenerator", R"pbdoc(
        Helper class for generating clock signals in simulation.
        
        This class helps generate evenly-spaced clock signals at a specified
        frequency, accounting for sample rate limitations.
    )pbdoc");

    clock_generator.def(py::init<>(), "Default constructor");

    clock_generator.def("init", &ClockGenerator::Init, R"pbdoc(
        Initialize the clock generator.
        
        Args:
            target_frequency: Target clock frequency in Hz
            sample_rate_hz: Sample rate in Hz
    )pbdoc", py::arg("target_frequency"), py::arg("sample_rate_hz"));

    clock_generator.def("advance_by_half_period", &ClockGenerator::AdvanceByHalfPeriod, R"pbdoc(
        Advance by half a clock period.
        
        This is useful for generating clock transitions (rising/falling edges).
        
        Args:
            multiple: Multiplier for the half period (default: 1.0)
            
        Returns:
            U32: Number of samples to advance
    )pbdoc", py::arg("multiple") = 1.0);

    clock_generator.def("advance_by_time_s", &ClockGenerator::AdvanceByTimeS, R"pbdoc(
        Advance by a specified time in seconds.
        
        Args:
            time_s: Time to advance in seconds
            
        Returns:
            U32: Number of samples to advance
    )pbdoc", py::arg("time_s"));

    // BitExtractor class
    py::class_<BitExtractor> bit_extractor(m, "BitExtractor", R"pbdoc(
        Helper class for extracting individual bits from a data value.
        
        This class helps extract bits from a value one at a time in the specified order
        (MSB first or LSB first).
    )pbdoc");

    bit_extractor.def(py::init<U64, AnalyzerEnums::ShiftOrder, U32>(), R"pbdoc(
        Initialize the bit extractor.
        
        Args:
            data: Data value to extract bits from
            shift_order: Order to extract bits (MSB first or LSB first)
            num_bits: Number of bits to extract
    )pbdoc", py::arg("data"), py::arg("shift_order"), py::arg("num_bits"));

    bit_extractor.def("get_next_bit", &BitExtractor::GetNextBit, R"pbdoc(
        Get the next bit from the data value.
        
        Returns:
            BitState: Next bit (HIGH or LOW)
    )pbdoc");

    // Helper method to extract all bits at once
    bit_extractor.def("get_all_bits", [](BitExtractor &self, U32 num_bits) {
        std::vector<BitState> bits;
        bits.reserve(num_bits);
        
        for (U32 i = 0; i < num_bits; i++) {
            bits.push_back(self.GetNextBit());
        }
        
        return bits;
    }, R"pbdoc(
        Get all bits from the data value.
        
        Args:
            num_bits: Number of bits to extract
            
        Returns:
            list: List of BitState values
    )pbdoc", py::arg("num_bits"));

    // DataBuilder class
    py::class_<DataBuilder> data_builder(m, "DataBuilder", R"pbdoc(
        Helper class for building a data value from individual bits.
        
        This class helps construct a data value by adding bits one at a time
        in the specified order (MSB first or LSB first).
    )pbdoc");

    data_builder.def(py::init<>(), "Default constructor");

    data_builder.def("reset", [](DataBuilder &self, AnalyzerEnums::ShiftOrder shift_order, U32 num_bits) {
        U64 data = 0;
        self.Reset(&data, shift_order, num_bits);
        return data;
    }, R"pbdoc(
        Reset the data builder for a new value.
        
        Args:
            shift_order: Order to add bits (MSB first or LSB first)
            num_bits: Number of bits in the value
            
        Returns:
            U64: Initial data value (0)
    )pbdoc", py::arg("shift_order"), py::arg("num_bits"));

    data_builder.def("add_bit", &DataBuilder::AddBit, R"pbdoc(
        Add a bit to the data value.
        
        Args:
            bit: Bit to add (HIGH or LOW)
    )pbdoc", py::arg("bit"));

    // Helper method to build from a list of bits
    data_builder.def("build_from_bits", [](DataBuilder &self, const std::vector<BitState> &bits, 
                                          AnalyzerEnums::ShiftOrder shift_order) {
        U64 data = 0;
        self.Reset(&data, shift_order, bits.size());
        
        for (BitState bit : bits) {
            self.AddBit(bit);
        }
        
        return data;
    }, R"pbdoc(
        Build a data value from a list of bits.
        
        Args:
            bits: List of bits to build from
            shift_order: Order to add bits (MSB first or LSB first)
            
        Returns:
            U64: Constructed data value
    )pbdoc", py::arg("bits"), py::arg("shift_order"));

    // SimpleArchive class
    py::class_<SimpleArchive> simple_archive(m, "SimpleArchive", R"pbdoc(
        Helper class for serializing/deserializing data.
        
        This class helps save and load settings data for analyzers by serializing
        and deserializing values to/from a string representation.
    )pbdoc");

    simple_archive.def(py::init<>(), "Default constructor");

    simple_archive.def("set_string", &SimpleArchive::SetString, R"pbdoc(
        Set the archive string for deserializing.
        
        Args:
            archive_string: String to deserialize from
    )pbdoc", py::arg("archive_string"));

    simple_archive.def("get_string", &SimpleArchive::GetString, R"pbdoc(
        Get the current archive string after serializing.
        
        Returns:
            str: Serialized data string
    )pbdoc");

    // Operator overloads for serializing data
    simple_archive.def("serialize_u64", [](SimpleArchive &self, U64 data) {
        return self << data;
    }, R"pbdoc(
        Serialize a U64 value.
        
        Args:
            data: Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_u32", [](SimpleArchive &self, U32 data) {
        return self << data;
    }, R"pbdoc(
        Serialize a U32 value.
        
        Args:
            data: Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_s64", [](SimpleArchive &self, S64 data) {
        return self << data;
    }, R"pbdoc(
        Serialize an S64 value.
        
        Args:
            data: Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_s32", [](SimpleArchive &self, S32 data) {
        return self << data;
    }, R"pbdoc(
        Serialize an S32 value.
        
        Args:
            data: Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_double", [](SimpleArchive &self, double data) {
        return self << data;
    }, R"pbdoc(
        Serialize a double value.
        
        Args:
            data: Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_bool", [](SimpleArchive &self, bool data) {
        return self << data;
    }, R"pbdoc(
        Serialize a boolean value.
        
        Args:
            data: Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_string", [](SimpleArchive &self, const char *data) {
        return self << data;
    }, R"pbdoc(
        Serialize a string value.
        
        Args:
            data: String to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_channel", [](SimpleArchive &self, Channel &data) {
        return self << data;
    }, R"pbdoc(
        Serialize a Channel value.
        
        Args:
            data: Channel to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    // Operator overloads for deserializing data
    simple_archive.def("deserialize_u64", [](SimpleArchive &self) {
        U64 data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a U64 value.
        
        Returns:
            tuple: (success, value)
                success: True if successful
                value: Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_u32", [](SimpleArchive &self) {
        U32 data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a U32 value.
        
        Returns:
            tuple: (success, value)
                success: True if successful
                value: Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_s64", [](SimpleArchive &self) {
        S64 data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize an S64 value.
        
        Returns:
            tuple: (success, value)
                success: True if successful
                value: Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_s32", [](SimpleArchive &self) {
        S32 data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize an S32 value.
        
        Returns:
            tuple: (success, value)
                success: True if successful
                value: Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_double", [](SimpleArchive &self) {
        double data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a double value.
        
        Returns:
            tuple: (success, value)
                success: True if successful
                value: Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_bool", [](SimpleArchive &self) {
        bool data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a boolean value.
        
        Returns:
            tuple: (success, value)
                success: True if successful
                value: Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_string", [](SimpleArchive &self) {
        const char *data;
        bool success = self >> data;
        return py::make_tuple(success, success ? std::string(data) : std::string());
    }, R"pbdoc(
        Deserialize a string value.
        
        Returns:
            tuple: (success, value)
                success: True if successful
                value: Deserialized string
    )pbdoc");

    simple_archive.def("deserialize_channel", [](SimpleArchive &self) {
        Channel data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a Channel value.
        
        Returns:
            tuple: (success, value)
                success: True if successful
                value: Deserialized Channel
    )pbdoc");

    // Helper function for serializing multiple values
    simple_archive.def("serialize_values", [](SimpleArchive &self, const py::args &args) {
        bool success = true;
        
        for (auto arg : args) {
            if (py::isinstance<py::int_>(arg)) {
                U64 value = arg.cast<U64>();
                success = success && (self << value);
            } else if (py::isinstance<py::float_>(arg)) {
                double value = arg.cast<double>();
                success = success && (self << value);
            } else if (py::isinstance<py::bool_>(arg)) {
                bool value = arg.cast<bool>();
                success = success && (self << value);
            } else if (py::isinstance<py::str>(arg)) {
                std::string value = arg.cast<std::string>();
                success = success && (self << value.c_str());
            } else if (py::isinstance<Channel>(arg)) {
                Channel value = arg.cast<Channel>();
                success = success && (self << value);
            } else {
                throw py::type_error("Unsupported type for serialization");
            }
        }
        
        return success;
    }, R"pbdoc(
        Serialize multiple values of different types.
        
        Args:
            *args: Values to serialize (int, float, bool, str, Channel)
            
        Returns:
            bool: True if all values were serialized successfully
    )pbdoc");
}