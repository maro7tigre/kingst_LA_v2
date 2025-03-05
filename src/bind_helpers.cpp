// bind_helpers.cpp
//
// Python bindings for the AnalyzerHelpers class and related utility classes
// Following the Kingst LA Pybind11 Binding Style Guide

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>

#include "AnalyzerHelpers.h"
#include "LogicPublicTypes.h"

namespace py = pybind11;

void init_helpers(py::module_ &m) {
    // ================================================================
    // AnalyzerHelpers static class
    // ================================================================
    py::class_<AnalyzerHelpers> analyzer_helpers(m, "AnalyzerHelpers", R"pbdoc(
        Static utility class with helper functions for analyzers.
        
        This class provides various utility functions for common tasks in protocol analyzers,
        such as number formatting, bit manipulation, and file operations.
    )pbdoc");

    // Make the class non-constructible from Python
    analyzer_helpers.def(py::init<>(), R"pbdoc(
        Constructor not available for static utility class.
    )pbdoc");

    // ----------------------------------------------------------------
    // Bit manipulation utilities
    // ----------------------------------------------------------------
    m.def("is_even", &AnalyzerHelpers::IsEven, R"pbdoc(
        Check if a number is even.
        
        Args:
            value (int): Number to check
            
        Returns:
            bool: True if the number is even
    )pbdoc", py::arg("value"));

    m.def("is_odd", &AnalyzerHelpers::IsOdd, R"pbdoc(
        Check if a number is odd.
        
        Args:
            value (int): Number to check
            
        Returns:
            bool: True if the number is odd
    )pbdoc", py::arg("value"));

    m.def("get_ones_count", &AnalyzerHelpers::GetOnesCount, R"pbdoc(
        Count the number of bits set to 1 in a value.
        
        Args:
            value (int): Value to check
            
        Returns:
            int: Number of bits set to 1
    )pbdoc", py::arg("value"));

    m.def("diff32", &AnalyzerHelpers::Diff32, R"pbdoc(
        Calculate the absolute difference between two 32-bit values.
        
        Args:
            a (int): First value
            b (int): Second value
            
        Returns:
            int: Absolute difference |a - b|
    )pbdoc", py::arg("a"), py::arg("b"));

    m.def("convert_to_signed_number", &AnalyzerHelpers::ConvertToSignedNumber, R"pbdoc(
        Convert an unsigned number to a signed number with the specified bit width.
        
        Args:
            number (int): Unsigned number to convert
            num_bits (int): Number of bits in the value
            
        Returns:
            int: Signed representation of the number
    )pbdoc", py::arg("number"), py::arg("num_bits"));

    // ----------------------------------------------------------------
    // String formatting
    // ----------------------------------------------------------------
    m.def("get_number_string", [](U64 number, DisplayBase display_base, U32 num_data_bits) {
        char result_string[128];
        AnalyzerHelpers::GetNumberString(number, display_base, num_data_bits, result_string, sizeof(result_string));
        return std::string(result_string);
    }, R"pbdoc(
        Convert a number to a string using the specified display base.
        
        Args:
            number (int): Number to convert
            display_base (DisplayBase): Base to display in (Binary, Decimal, Hexadecimal, etc.)
            num_data_bits (int): Number of data bits in the value
            
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
            sample (int): Sample number to convert
            trigger_sample (int): Trigger sample number (reference point)
            sample_rate_hz (int): Sample rate in Hz
            
        Returns:
            str: Formatted time string
    )pbdoc", py::arg("sample"), py::arg("trigger_sample"), py::arg("sample_rate_hz"));

    // Python-friendly convenience aliases
    m.def("format_number", [](U64 number, DisplayBase display_base, U32 num_data_bits) {
        char result_string[128];
        AnalyzerHelpers::GetNumberString(number, display_base, num_data_bits, result_string, sizeof(result_string));
        return std::string(result_string);
    }, R"pbdoc(
        Format a number as a string using the specified display base.
        
        This is an alias for get_number_string with a more pythonic name.
        
        Args:
            number (int): Number to format
            display_base (DisplayBase): Base to display in (Binary, Decimal, Hexadecimal, etc.)
            num_data_bits (int): Number of data bits in the value
            
        Returns:
            str: Formatted string representation
    )pbdoc", py::arg("number"), py::arg("display_base"), py::arg("num_data_bits"));

    m.def("format_time", [](U64 sample, U64 trigger_sample, U32 sample_rate_hz) {
        char result_string[128];
        AnalyzerHelpers::GetTimeString(sample, trigger_sample, sample_rate_hz, result_string, sizeof(result_string));
        return std::string(result_string);
    }, R"pbdoc(
        Format a sample number as a time string.
        
        This is an alias for get_time_string with a more pythonic name.
        
        Args:
            sample (int): Sample number to format
            trigger_sample (int): Trigger sample number (reference point)
            sample_rate_hz (int): Sample rate in Hz
            
        Returns:
            str: Formatted time string
    )pbdoc", py::arg("sample"), py::arg("trigger_sample"), py::arg("sample_rate_hz"));

    // ----------------------------------------------------------------
    // Debugging utilities
    // ----------------------------------------------------------------
    m.def("assert_true", [](bool condition, const std::string& message) {
        if (!condition) {
            AnalyzerHelpers::Assert(message.c_str());
        }
    }, R"pbdoc(
        Assert that a condition is true.
        
        Args:
            condition (bool): Condition to check
            message (str): Error message if condition is false
            
        Raises:
            AssertionError: If the condition is false
            
        Note:
            This is different from Python's assert in that it will always be checked
            regardless of optimization settings.
    )pbdoc", py::arg("condition"), py::arg("message"));

    // ----------------------------------------------------------------
    // Simulation utilities
    // ----------------------------------------------------------------
    m.def("adjust_simulation_target_sample", &AnalyzerHelpers::AdjustSimulationTargetSample, R"pbdoc(
        Adjust a target sample number for simulation.
        
        This function adjusts a target sample number based on different sample rates
        between simulation and normal operation.
        
        Args:
            target_sample (int): Target sample number to adjust
            sample_rate (int): Normal sample rate
            simulation_sample_rate (int): Simulation sample rate
            
        Returns:
            int: Adjusted sample number
    )pbdoc", 
        py::arg("target_sample"), 
        py::arg("sample_rate"), 
        py::arg("simulation_sample_rate")
    );

    // ----------------------------------------------------------------
    // Channel utilities
    // ----------------------------------------------------------------
    m.def("do_channels_overlap", [](const std::vector<Channel>& channels) {
        if (channels.empty()) {
            return false;
        }
        return AnalyzerHelpers::DoChannelsOverlap(channels.data(), static_cast<U32>(channels.size()));
    }, R"pbdoc(
        Check if any channels in the list overlap.
        
        Args:
            channels (list[Channel]): List of channels to check
            
        Returns:
            bool: True if any channels overlap
            
        Note:
            Overlapping channels cannot be used in the same analyzer.
    )pbdoc", py::arg("channels"));

    // ----------------------------------------------------------------
    // File operations
    // ----------------------------------------------------------------
    m.def("save_file", [](const std::string& file_name, const std::vector<U8>& data, bool is_binary) {
        AnalyzerHelpers::SaveFile(file_name.c_str(), data.data(), static_cast<U32>(data.size()), is_binary);
    }, R"pbdoc(
        Save data to a file.
        
        Args:
            file_name (str): Name of the file to save
            data (bytes or list[int]): Data to save
            is_binary (bool, optional): True for binary file, False for text file. Defaults to False.
            
        Note:
            This method loads all data into memory. For large files, use the
            start_file/append_to_file/end_file sequence instead.
    )pbdoc", 
        py::arg("file_name"), 
        py::arg("data"), 
        py::arg("is_binary") = false
    );

    // File streaming operations with better memory management
    m.def("start_file", [](const std::string& file_name, bool is_binary) {
        return AnalyzerHelpers::StartFile(file_name.c_str(), is_binary);
    }, R"pbdoc(
        Start writing to a file.
        
        Args:
            file_name (str): Name of the file to write
            is_binary (bool, optional): True for binary file, False for text file. Defaults to False.
            
        Returns:
            object: File handle to use with append_to_file and end_file
            
        Note:
            Must call end_file when finished to avoid resource leaks.
    )pbdoc", 
        py::arg("file_name"), 
        py::arg("is_binary") = false,
        py::return_value_policy::reference  // File handle is borrowed, closed by end_file
    );

    m.def("append_to_file", [](const std::vector<U8>& data, void* file) {
        AnalyzerHelpers::AppendToFile(data.data(), static_cast<U32>(data.size()), file);
    }, R"pbdoc(
        Append data to an open file.
        
        Args:
            data (bytes or list[int]): Data to append
            file (object): File handle from start_file
            
        Note:
            The file handle must be obtained from start_file.
    )pbdoc", py::arg("data"), py::arg("file"));

    m.def("end_file", &AnalyzerHelpers::EndFile, R"pbdoc(
        Finish writing to a file and close it.
        
        Args:
            file (object): File handle from start_file
            
        Note:
            This must be called to properly close the file and free resources.
    )pbdoc", py::arg("file"));

    // ================================================================
    // ClockGenerator class
    // ================================================================
    py::class_<ClockGenerator> clock_generator(m, "ClockGenerator", R"pbdoc(
        Helper class for generating clock signals in simulation.
        
        This class helps generate evenly-spaced clock signals at a specified
        frequency, accounting for sample rate limitations.
        
        Example:
            # Create a 1MHz clock with 100MHz sample rate
            clock_gen = ClockGenerator()
            clock_gen.init(1_000_000, 100_000_000)
            
            # Advance to create clock transitions
            samples_to_first_edge = clock_gen.advance_by_half_period()
            samples_to_second_edge = clock_gen.advance_by_half_period()
    )pbdoc");

    clock_generator.def(py::init<>(), R"pbdoc(
        Create a new uninitialized ClockGenerator.
        
        Note:
            Must call init() before using the generator.
    )pbdoc");

    clock_generator.def("init", &ClockGenerator::Init, R"pbdoc(
        Initialize the clock generator.
        
        Args:
            target_frequency (float): Target clock frequency in Hz
            sample_rate_hz (int): Sample rate in Hz
            
        Note:
            The actual generated frequency may differ slightly from the target
            due to sample rate resolution limitations.
    )pbdoc", py::arg("target_frequency"), py::arg("sample_rate_hz"));

    clock_generator.def("advance_by_half_period", &ClockGenerator::AdvanceByHalfPeriod, R"pbdoc(
        Advance by half a clock period.
        
        This is useful for generating clock transitions (rising/falling edges).
        
        Args:
            multiple (float, optional): Multiplier for the half period. Defaults to 1.0.
            
        Returns:
            int: Number of samples to advance
            
        Note:
            Calling this repeatedly alternates between rising and falling edges.
    )pbdoc", py::arg("multiple") = 1.0);

    clock_generator.def("advance_by_time_s", &ClockGenerator::AdvanceByTimeS, R"pbdoc(
        Advance by a specified time in seconds.
        
        Args:
            time_s (float): Time to advance in seconds
            
        Returns:
            int: Number of samples to advance
    )pbdoc", py::arg("time_s"));

    // ================================================================
    // BitExtractor class
    // ================================================================
    py::class_<BitExtractor> bit_extractor(m, "BitExtractor", R"pbdoc(
        Helper class for extracting individual bits from a data value.
        
        This class helps extract bits from a value one at a time in the specified order
        (MSB first or LSB first).
        
        Example:
            # Extract bits from 0xA5 (10100101), MSB first
            extractor = BitExtractor(0xA5, ShiftOrder.MSBFirst, 8)
            
            # Get the first bit (1)
            first_bit = extractor.get_next_bit()
    )pbdoc");

    bit_extractor.def(py::init<U64, AnalyzerEnums::ShiftOrder, U32>(), R"pbdoc(
        Initialize the bit extractor.
        
        Args:
            data (int): Data value to extract bits from
            shift_order (ShiftOrder): Order to extract bits (MSBFirst or LSBFirst)
            num_bits (int): Number of bits to extract
    )pbdoc", 
        py::arg("data"), 
        py::arg("shift_order"), 
        py::arg("num_bits")
    );

    bit_extractor.def("get_next_bit", &BitExtractor::GetNextBit, R"pbdoc(
        Get the next bit from the data value.
        
        Returns:
            BitState: Next bit (HIGH or LOW)
            
        Note:
            Returns bits in the order specified by the shift_order parameter.
    )pbdoc");

    // Helper method to extract all bits at once
    bit_extractor.def("get_all_bits", [](BitExtractor& self, U32 num_bits) {
        std::vector<BitState> bits;
        bits.reserve(num_bits);
        
        for (U32 i = 0; i < num_bits; i++) {
            bits.push_back(self.GetNextBit());
        }
        
        return bits;
    }, R"pbdoc(
        Get all bits from the data value as a list.
        
        Args:
            num_bits (int): Number of bits to extract
            
        Returns:
            list[BitState]: List of bit states in the specified order
            
        Note:
            This is a convenience method not present in the C++ API.
    )pbdoc", py::arg("num_bits"));

    // ================================================================
    // DataBuilder class
    // ================================================================
    py::class_<DataBuilder> data_builder(m, "DataBuilder", R"pbdoc(
        Helper class for building a data value from individual bits.
        
        This class helps construct a data value by adding bits one at a time
        in the specified order (MSB first or LSB first).
        
        Example:
            # Build a value by adding bits LSB first
            builder = DataBuilder()
            data = builder.reset(ShiftOrder.LSBFirst, 8)
            
            # Add bits for 0xA5 (10100101)
            builder.add_bit(BitState.HIGH)  # 1
            builder.add_bit(BitState.LOW)   # 0
            builder.add_bit(BitState.HIGH)  # 1
            # ...
    )pbdoc");

    data_builder.def(py::init<>(), R"pbdoc(
        Create a new uninitialized DataBuilder.
        
        Note:
            Must call reset() before adding bits.
    )pbdoc");

    data_builder.def("reset", [](DataBuilder& self, AnalyzerEnums::ShiftOrder shift_order, U32 num_bits) {
        U64 data = 0;
        self.Reset(&data, shift_order, num_bits);
        return data;
    }, R"pbdoc(
        Reset the data builder for a new value.
        
        Args:
            shift_order (ShiftOrder): Order to add bits (MSBFirst or LSBFirst)
            num_bits (int): Number of bits in the value
            
        Returns:
            int: Initial data value (0)
            
        Note:
            The return value is a reference to the internal data value that
            will be modified by add_bit() calls.
    )pbdoc", py::arg("shift_order"), py::arg("num_bits"));

    data_builder.def("add_bit", &DataBuilder::AddBit, R"pbdoc(
        Add a bit to the data value.
        
        Args:
            bit (BitState): Bit to add (HIGH or LOW)
            
        Note:
            Bits are added in the order specified by shift_order in reset().
    )pbdoc", py::arg("bit"));

    // Helper method to build from a list of bits
    data_builder.def("build_from_bits", [](DataBuilder& self, const std::vector<BitState>& bits, 
                                          AnalyzerEnums::ShiftOrder shift_order) {
        U64 data = 0;
        self.Reset(&data, shift_order, static_cast<U32>(bits.size()));
        
        for (BitState bit : bits) {
            self.AddBit(bit);
        }
        
        return data;
    }, R"pbdoc(
        Build a data value from a list of bits.
        
        Args:
            bits (list[BitState]): List of bits to build from
            shift_order (ShiftOrder): Order to add bits (MSBFirst or LSBFirst)
            
        Returns:
            int: Constructed data value
            
        Note:
            This is a convenience method not present in the C++ API.
    )pbdoc", py::arg("bits"), py::arg("shift_order"));

    // ================================================================
    // SimpleArchive class
    // ================================================================
    py::class_<SimpleArchive> simple_archive(m, "SimpleArchive", R"pbdoc(
        Helper class for serializing/deserializing data.
        
        This class helps save and load settings data for analyzers by serializing
        and deserializing values to/from a string representation.
        
        Example:
            # Serialize values
            archive = SimpleArchive()
            archive.serialize_u32(123)
            archive.serialize_string("test")
            
            # Get the serialized string
            serialized = archive.get_string()
            
            # Later, deserialize
            archive2 = SimpleArchive()
            archive2.set_string(serialized)
            success, value = archive2.deserialize_u32()
    )pbdoc");

    simple_archive.def(py::init<>(), R"pbdoc(
        Create a new empty SimpleArchive.
    )pbdoc");

    simple_archive.def("set_string", &SimpleArchive::SetString, R"pbdoc(
        Set the archive string for deserializing.
        
        Args:
            archive_string (str): String to deserialize from
            
        Note:
            This prepares the archive for deserializing values.
    )pbdoc", py::arg("archive_string"));

    simple_archive.def("get_string", &SimpleArchive::GetString, R"pbdoc(
        Get the current archive string after serializing.
        
        Returns:
            str: Serialized data string
            
        Note:
            Use this to retrieve the serialized data after calling serialize methods.
    )pbdoc");

    // ----------------------------------------------------------------
    // Serialization methods
    // ----------------------------------------------------------------
    
    simple_archive.def("serialize_u64", [](SimpleArchive& self, U64 data) -> bool {
        return self << data;
    }, R"pbdoc(
        Serialize a 64-bit unsigned integer.
        
        Args:
            data (int): Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_u32", [](SimpleArchive& self, U32 data) -> bool {
        return self << data;
    }, R"pbdoc(
        Serialize a 32-bit unsigned integer.
        
        Args:
            data (int): Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_s64", [](SimpleArchive& self, S64 data) -> bool {
        return self << data;
    }, R"pbdoc(
        Serialize a 64-bit signed integer.
        
        Args:
            data (int): Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_s32", [](SimpleArchive& self, S32 data) -> bool {
        return self << data;
    }, R"pbdoc(
        Serialize a 32-bit signed integer.
        
        Args:
            data (int): Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_double", [](SimpleArchive& self, double data) -> bool {
        return self << data;
    }, R"pbdoc(
        Serialize a double-precision floating point number.
        
        Args:
            data (float): Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_bool", [](SimpleArchive& self, bool data) -> bool {
        return self << data;
    }, R"pbdoc(
        Serialize a boolean value.
        
        Args:
            data (bool): Value to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_string", [](SimpleArchive& self, const std::string& data) -> bool {
        return self << data.c_str();
    }, R"pbdoc(
        Serialize a string value.
        
        Args:
            data (str): String to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    simple_archive.def("serialize_channel", [](SimpleArchive& self, Channel& data) -> bool {
        return self << data;
    }, R"pbdoc(
        Serialize a Channel value.
        
        Args:
            data (Channel): Channel to serialize
            
        Returns:
            bool: True if successful
    )pbdoc", py::arg("data"));

    // ----------------------------------------------------------------
    // Deserialization methods
    // ----------------------------------------------------------------
    
    simple_archive.def("deserialize_u64", [](SimpleArchive& self) {
        U64 data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a 64-bit unsigned integer.
        
        Returns:
            tuple: (success, value)
                success (bool): True if successful
                value (int): Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_u32", [](SimpleArchive& self) {
        U32 data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a 32-bit unsigned integer.
        
        Returns:
            tuple: (success, value)
                success (bool): True if successful
                value (int): Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_s64", [](SimpleArchive& self) {
        S64 data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a 64-bit signed integer.
        
        Returns:
            tuple: (success, value)
                success (bool): True if successful
                value (int): Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_s32", [](SimpleArchive& self) {
        S32 data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a 32-bit signed integer.
        
        Returns:
            tuple: (success, value)
                success (bool): True if successful
                value (int): Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_double", [](SimpleArchive& self) {
        double data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a double-precision floating point number.
        
        Returns:
            tuple: (success, value)
                success (bool): True if successful
                value (float): Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_bool", [](SimpleArchive& self) {
        bool data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a boolean value.
        
        Returns:
            tuple: (success, value)
                success (bool): True if successful
                value (bool): Deserialized value
    )pbdoc");

    simple_archive.def("deserialize_string", [](SimpleArchive& self) {
        const char* data = nullptr;  // Use const char* to match the operator>> signature
        bool success = self >> &data; // Pass address of pointer
        // Return the string value if successful, empty string otherwise
        return py::make_tuple(success, success && data ? std::string(data) : std::string());
    }, R"pbdoc(
        Deserialize a string value.

        Returns:
            tuple: (success, value)
                success (bool): True if successful
                value (str): Deserialized string
    )pbdoc");

    simple_archive.def("deserialize_channel", [](SimpleArchive& self) {
        Channel data;
        bool success = self >> data;
        return py::make_tuple(success, data);
    }, R"pbdoc(
        Deserialize a Channel value.
        
        Returns:
            tuple: (success, value)
                success (bool): True if successful
                value (Channel): Deserialized Channel
    )pbdoc");

    // ----------------------------------------------------------------
    // Convenience methods
    // ----------------------------------------------------------------
    
    simple_archive.def("serialize_values", [](SimpleArchive& self, const py::args& args) {
        bool success = true;
        
        for (auto arg : args) {
            if (py::isinstance<py::int_>(arg)) {
                // Try to use the smallest integer type that fits
                py::int_ py_int = py::cast<py::int_>(arg);
                py::int_ max_u32 = py::cast(static_cast<U64>(std::numeric_limits<U32>::max()));
                
                try {
                    // Convert to C++ long long to do the comparison
                    long long value = py::cast<long long>(py_int);
                    if (value < 0) {
                        // Negative number - use signed type
                        // Get the absolute value and compare to max
                        long long abs_val = value < 0 ? -value : value;
                        if (abs_val <= static_cast<long long>(std::numeric_limits<S32>::max())) {
                            success = success && (self << static_cast<S32>(value));
                        } else {
                            success = success && (self << static_cast<S64>(value));
                        }
                    } else {
                        // Positive number - use unsigned type if it fits
                        if (value <= static_cast<long long>(std::numeric_limits<U32>::max())) {
                            success = success && (self << static_cast<U32>(value));
                        } else {
                            success = success && (self << static_cast<U64>(value));
                        }
                    }
                } catch (const std::exception& e) {
                    py::print("Error serializing int:", e.what());
                    success = false;
                }
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
            
        Raises:
            TypeError: If an unsupported type is provided
            
        Note:
            This is a convenience method not present in the C++ API.
            Integer types are automatically serialized using the smallest
            appropriate type based on their value.
    )pbdoc");
    
    // Add a simple method to deserialize multiple values in sequence
    simple_archive.def("deserialize_multiple", [](SimpleArchive& self, const std::vector<std::string>& types) {
        py::list results;
        bool overall_success = true;
        
        for (const auto& type : types) {
            bool success = false;
            py::object value;
            
            if (type == "u64" || type == "uint64") {
                U64 data;
                success = self >> data;
                value = py::cast(data);
            } else if (type == "u32" || type == "uint32") {
                U32 data;
                success = self >> data;
                value = py::cast(data);
            } else if (type == "s64" || type == "int64") {
                S64 data;
                success = self >> data;
                value = py::cast(data);
            } else if (type == "s32" || type == "int32") {
                S32 data;
                success = self >> data;
                value = py::cast(data);
            } else if (type == "double" || type == "float") {
                double data;
                success = self >> data;
                value = py::cast(data);
            } else if (type == "bool" || type == "boolean") {
                bool data;
                success = self >> data;
                value = py::cast(data);
            } else if (type == "str" || type == "string") {
                const char* data = nullptr;
                success = self >> &data;
                value = (success && data) ? py::cast(std::string(data)) : py::cast(std::string());
            } else if (type == "channel") {
                Channel data;
                success = self >> data;
                value = py::cast(data);
            } else {
                throw py::type_error("Unsupported deserialization type: " + type);
            }
            
            overall_success = overall_success && success;
            results.append(py::make_tuple(success, value));
        }
        
        return py::make_tuple(overall_success, results);
    }, R"pbdoc(
        Deserialize multiple values in sequence.
        
        Args:
            types (list[str]): List of type names to deserialize.
                               Valid types: "u64", "u32", "s64", "s32", "double", 
                               "bool", "str", "channel"
            
        Returns:
            tuple: (overall_success, results)
                overall_success (bool): True if all deserialization operations succeeded
                results (list): List of (success, value) tuples for each type
                
        Raises:
            TypeError: If an unsupported type name is provided
            
        Note:
            This is a convenience method not present in the C++ API.
            Type names are case-insensitive and have aliases:
            - "u64"/"uint64": 64-bit unsigned int
            - "u32"/"uint32": 32-bit unsigned int
            - "s64"/"int64": 64-bit signed int
            - "s32"/"int32": 32-bit signed int
            - "double"/"float": double-precision float
            - "bool"/"boolean": boolean
            - "str"/"string": string
            - "channel": Channel
    )pbdoc", py::arg("types"));
}