/**
 * bind_basic_types.cpp
 * Comprehensive implementation of bindings for LogicPublicTypes.h and AnalyzerTypes.h
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "LogicPublicTypes.h"
#include "AnalyzerTypes.h"

namespace py = pybind11;

/**
 * Helper functions for bit manipulation
 */
BitState toggle_bit(BitState bit) {
    return bit == BIT_LOW ? BIT_HIGH : BIT_LOW;
}

BitState invert_bit(BitState bit) {
    return bit == BIT_LOW ? BIT_HIGH : BIT_LOW;
}

void init_basic_types(py::module_ &m) {
    // ---- Integer Type Definitions and Bounds ----
    
    // 8-bit integers
    m.attr("S8_MIN") = (int)(-128);
    m.attr("S8_MAX") = (int)127;
    m.attr("U8_MAX") = (int)255;
    
    // 16-bit integers
    m.attr("S16_MIN") = (int)(-32768);
    m.attr("S16_MAX") = (int)32767;
    m.attr("U16_MAX") = (int)65535;
    
    // 32-bit integers
    m.attr("S32_MIN") = (int)(-2147483648);
    m.attr("S32_MAX") = (int)2147483647;
    m.attr("U32_MAX") = (unsigned long)4294967295;
    
    // 64-bit integers (Python int has arbitrary precision)
    m.attr("S64_MIN") = py::int("-9223372036854775808");
    m.attr("S64_MAX") = py::int("9223372036854775807");
    m.attr("U64_MAX") = py::int("18446744073709551615");
    
    // ---- LogicPublicTypes.h - Basic types and enums ----
    
    // DisplayBase enum
    py::enum_<DisplayBase>(m, "DisplayBase", R"pbdoc(
        Controls how numbers are displayed in the UI.

        Members:
            Binary: Display as binary (e.g., "10110")
            Decimal: Display as decimal (e.g., "22")
            Hexadecimal: Display as hex (e.g., "0x16")
            ASCII: Display as ASCII characters
            AsciiHex: Display as ASCII with hex for non-printable characters
    )pbdoc")
        .value("Binary", Binary, "Display numbers in binary format")
        .value("Decimal", Decimal, "Display numbers in decimal format")
        .value("Hexadecimal", Hexadecimal, "Display numbers in hexadecimal format")
        .value("ASCII", ASCII, "Display numbers as ASCII characters")
        .value("AsciiHex", AsciiHex, "Display numbers as ASCII with hex for non-printable characters")
        .export_values();

    // BitState enum
    py::enum_<BitState>(m, "BitState", R"pbdoc(
        Represents the logical state of a digital signal.

        Members:
            LOW: Logic low (0)
            HIGH: Logic high (1)
    )pbdoc")
        .value("LOW", BIT_LOW, "Logic low (0)")
        .value("HIGH", BIT_HIGH, "Logic high (1)")
        .export_values();
    
    // Add utility functions for bit manipulation
    m.def("toggle_bit", &toggle_bit, 
          R"pbdoc(
          Toggle a bit state (equivalent to C++ Toggle macro).

          Args:
              bit (BitState): The bit state to toggle

          Returns:
              BitState: The toggled bit state (LOW->HIGH or HIGH->LOW)
          )pbdoc",
          py::arg("bit"));
    
    m.def("invert_bit", &invert_bit, 
          R"pbdoc(
          Invert a bit state (equivalent to C++ Invert macro).

          Args:
              bit (BitState): The bit state to invert

          Returns:
              BitState: The inverted bit state (LOW->HIGH or HIGH->LOW)
          )pbdoc",
          py::arg("bit"));

    // Channel class
    py::class_<Channel>(m, "Channel", R"pbdoc(
        Represents a physical channel on the logic analyzer.
        
        Each channel is identified by a device ID and a channel index.
        The device ID identifies the specific logic analyzer hardware,
        and the channel index identifies the specific channel on that device.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Default constructor. Creates an uninitialized channel.
        )pbdoc")
        
        .def(py::init<const Channel &>(), R"pbdoc(
            Copy constructor.

            Args:
                channel (Channel): The channel to copy
        )pbdoc",
             py::arg("channel"))
             
        .def(py::init<U64, U32>(), R"pbdoc(
            Construct a channel with specified device ID and channel index.

            Args:
                device_id (int): Identifier for the logic analyzer device
                channel_index (int): Channel number on the device (zero-based)
        )pbdoc",
             py::arg("device_id"), py::arg("channel_index"))
             
        .def_readwrite("device_id", &Channel::mDeviceId,
                      R"pbdoc(
                      Identifier for the logic analyzer device.
                      
                      Each physical logic analyzer has a unique device ID.
                      )pbdoc")
                      
        .def_readwrite("channel_index", &Channel::mChannelIndex,
                      R"pbdoc(
                      Channel number on the device (zero-based).
                      
                      Identifies which specific channel on the device this
                      Channel object represents.
                      )pbdoc")
                      
        // Comparison operators
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        
        // String representation
        .def("__repr__", [](const Channel &c) {
            return "<Channel device_id=" + std::to_string(c.mDeviceId) + 
                   ", channel_index=" + std::to_string(c.mChannelIndex) + ">";
        });
    
    // Define UNDEFINED_CHANNEL as a module-level constant
    m.attr("UNDEFINED_CHANNEL") = Channel(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFF);

    // ---- AnalyzerTypes.h - Enums for analyzer configuration ----

    // Create a submodule for AnalyzerEnums namespace
    py::module_ analyzer_enums = m.def_submodule(
        "AnalyzerEnums", 
        R"pbdoc(
        Enumerations for analyzer configuration.
        
        This submodule contains enumerations used to configure
        various aspects of protocol analyzers.
        )pbdoc");
    
    // ShiftOrder enum
    py::enum_<AnalyzerEnums::ShiftOrder>(analyzer_enums, "ShiftOrder", R"pbdoc(
        Specifies the bit order for serial protocols.

        Members:
            MsbFirst: Most significant bit first (e.g., standard SPI)
            LsbFirst: Least significant bit first
    )pbdoc")
        .value("MsbFirst", AnalyzerEnums::MsbFirst, "Most significant bit first")
        .value("LsbFirst", AnalyzerEnums::LsbFirst, "Least significant bit first")
        .export_values();

    // EdgeDirection enum
    py::enum_<AnalyzerEnums::EdgeDirection>(analyzer_enums, "EdgeDirection", R"pbdoc(
        Defines edge directions for triggering or sampling.

        Members:
            PosEdge: Rising edge (low to high transition)
            NegEdge: Falling edge (high to low transition)
    )pbdoc")
        .value("PosEdge", AnalyzerEnums::PosEdge, "Rising edge (low to high)")
        .value("NegEdge", AnalyzerEnums::NegEdge, "Falling edge (high to low)")
        .export_values();

    // Edge enum
    py::enum_<AnalyzerEnums::Edge>(analyzer_enums, "Edge", R"pbdoc(
        Defines specific edges for analysis.

        Members:
            LeadingEdge: First edge of a sequence
            TrailingEdge: Last edge of a sequence
    )pbdoc")
        .value("LeadingEdge", AnalyzerEnums::LeadingEdge, "First edge of a sequence")
        .value("TrailingEdge", AnalyzerEnums::TrailingEdge, "Last edge of a sequence")
        .export_values();
    
    // Parity enum
    py::enum_<AnalyzerEnums::Parity>(analyzer_enums, "Parity", R"pbdoc(
        Specifies parity settings for serial protocols.

        Members:
            None: No parity bit
            Even: Even parity (bit ensures even number of 1s)
            Odd: Odd parity (bit ensures odd number of 1s)
    )pbdoc")
        .value("None", AnalyzerEnums::None, "No parity bit")
        .value("Even", AnalyzerEnums::Even, "Even parity (even number of 1s)")
        .value("Odd", AnalyzerEnums::Odd, "Odd parity (odd number of 1s)")
        .export_values();
    
    // Acknowledge enum
    py::enum_<AnalyzerEnums::Acknowledge>(analyzer_enums, "Acknowledge", R"pbdoc(
        Represents acknowledgment states in protocols.

        Members:
            Ack: Acknowledged (typically low for I2C)
            Nak: Not acknowledged (typically high for I2C)
    )pbdoc")
        .value("Ack", AnalyzerEnums::Ack, "Acknowledged")
        .value("Nak", AnalyzerEnums::Nak, "Not acknowledged")
        .export_values();
    
    // Sign enum
    py::enum_<AnalyzerEnums::Sign>(analyzer_enums, "Sign", R"pbdoc(
        Specifies if values should be interpreted as signed or unsigned.

        Members:
            UnsignedInteger: Treat as unsigned value (all bits are magnitude)
            SignedInteger: Treat as signed value (MSB is sign bit)
    )pbdoc")
        .value("UnsignedInteger", AnalyzerEnums::UnsignedInteger, "Unsigned integer interpretation")
        .value("SignedInteger", AnalyzerEnums::SignedInteger, "Signed integer interpretation")
        .export_values();
    
    // For convenience, also expose the enums directly in the main module
    // This makes them accessible both as analyzer_enums.X and directly as X
    m.attr("ShiftOrder") = analyzer_enums.attr("ShiftOrder");
    m.attr("EdgeDirection") = analyzer_enums.attr("EdgeDirection");
    m.attr("Edge") = analyzer_enums.attr("Edge");
    m.attr("Parity") = analyzer_enums.attr("Parity");
    m.attr("Acknowledge") = analyzer_enums.attr("Acknowledge");
    m.attr("Sign") = analyzer_enums.attr("Sign");
    
    // Add module-level constants for marker values instead of an enum
    m.attr("NO_SHIFT_ORDER_SPECIFIED") = py::none();
    m.attr("NO_EDGE_DIRECTION_SPECIFIED") = py::none();
    m.attr("NO_PARITY_SPECIFIED") = py::none();
}