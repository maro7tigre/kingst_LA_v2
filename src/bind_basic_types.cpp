// init_basic_types.cpp
// Complete implementation of bindings for LogicPublicTypes.h and AnalyzerTypes.h

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "LogicPublicTypes.h"
#include "AnalyzerTypes.h"

namespace py = pybind11;

// Helper function equivalent to Toggle macro from LogicPublicTypes.h
BitState toggle_bit(BitState bit) {
    return bit == BIT_LOW ? BIT_HIGH : BIT_LOW;
}

// Helper function equivalent to Invert macro from LogicPublicTypes.h
BitState invert_bit(BitState bit) {
    return bit == BIT_LOW ? BIT_HIGH : BIT_LOW;
}

void init_basic_types(py::module_ &m) {
    // ---- Expose integer typedefs as attributes ----
    // Note: These are for reference only, Python will use its own numeric types
    m.attr("S8_MIN") = (int)(-128);
    m.attr("S8_MAX") = (int)127;
    m.attr("U8_MAX") = (int)255;
    
    m.attr("S16_MIN") = (int)(-32768);
    m.attr("S16_MAX") = (int)32767;
    m.attr("U16_MAX") = (int)65535;
    
    m.attr("S32_MIN") = (int)(-2147483648);
    m.attr("S32_MAX") = (int)2147483647;
    m.attr("U32_MAX") = (unsigned long)4294967295;
    
    // Note: Using Python's int for 64-bit integers as they have arbitrary precision
    
    // ---- LogicPublicTypes.h - Basic types and enums ----
    
    /*
     * DisplayBase enum:
     * Controls how numbers are displayed in the UI
     * - Binary: Display as binary (e.g., "10110")
     * - Decimal: Display as decimal (e.g., "22")
     * - Hexadecimal: Display as hex (e.g., "0x16")
     * - ASCII: Display as ASCII characters
     * - AsciiHex: Display as ASCII with hex for non-printable characters
     */
    py::enum_<DisplayBase>(m, "DisplayBase")
        .value("Binary", Binary)
        .value("Decimal", Decimal)
        .value("Hexadecimal", Hexadecimal)
        .value("ASCII", ASCII)
        .value("AsciiHex", AsciiHex);

    /*
     * BitState enum:
     * Represents the logical state of a digital signal
     * - BIT_LOW: Logic low (0)
     * - BIT_HIGH: Logic high (1)
     */
    py::enum_<BitState>(m, "BitState")
        .value("LOW", BIT_LOW)
        .value("HIGH", BIT_HIGH);
    
    // Add utility functions equivalent to the Toggle and Invert macros
    m.def("toggle_bit", &toggle_bit, "Toggle a bit state (equivalent to C++ Toggle macro)",
          py::arg("bit"));
    m.def("invert_bit", &invert_bit, "Invert a bit state (equivalent to C++ Invert macro)",
          py::arg("bit"));

    /*
     * Channel class:
     * Represents a physical channel on the logic analyzer
     * - mDeviceId: Identifies the logic analyzer device
     * - mChannelIndex: Identifies the specific channel on the device
     */
    py::class_<Channel>(m, "Channel")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const Channel &>(), "Copy constructor", 
             py::arg("channel"))
        .def(py::init<U64, U32>(), "Construct with device_id and channel_index",
             py::arg("device_id"), py::arg("channel_index"))
        .def_readwrite("device_id", &Channel::mDeviceId,
                       "Identifier for the logic analyzer device")
        .def_readwrite("channel_index", &Channel::mChannelIndex,
                       "Channel number on the device")
        .def("__eq__", &Channel::operator==, py::is_operator())
        .def("__ne__", &Channel::operator!=, py::is_operator())
        .def("__lt__", &Channel::operator<, py::is_operator())
        .def("__gt__", &Channel::operator>, py::is_operator());
    
    // Define UNDEFINED_CHANNEL as a module-level constant
    m.attr("UNDEFINED_CHANNEL") = Channel(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFF);

    // ---- AnalyzerTypes.h - Enums for analyzer configuration ----

    // Create a submodule for AnalyzerEnums
    py::module_ analyzer_enums = m.def_submodule("AnalyzerEnums", "Enumerations for analyzer configuration");
    
    /*
     * ShiftOrder enum:
     * Specifies the bit order for serial protocols
     * - MsbFirst: Most significant bit first
     * - LsbFirst: Least significant bit first
     */
    py::enum_<AnalyzerEnums::ShiftOrder>(analyzer_enums, "ShiftOrder")
        .value("MsbFirst", AnalyzerEnums::MsbFirst)
        .value("LsbFirst", AnalyzerEnums::LsbFirst);

    /*
     * EdgeDirection enum:
     * Defines edge directions for triggering or sampling
     * - PosEdge: Rising edge (low to high)
     * - NegEdge: Falling edge (high to low)
     */
    py::enum_<AnalyzerEnums::EdgeDirection>(analyzer_enums, "EdgeDirection")
        .value("PosEdge", AnalyzerEnums::PosEdge)
        .value("NegEdge", AnalyzerEnums::NegEdge);

    /*
     * Edge enum:
     * Defines specific edges for analysis
     * - LeadingEdge: First edge
     * - TrailingEdge: Last edge
     */
    py::enum_<AnalyzerEnums::Edge>(analyzer_enums, "Edge")
        .value("LeadingEdge", AnalyzerEnums::LeadingEdge)
        .value("TrailingEdge", AnalyzerEnums::TrailingEdge);
    
    /*
     * Parity enum:
     * Specifies parity settings for serial protocols
     * - None: No parity
     * - Even: Even parity
     * - Odd: Odd parity
     */
    py::enum_<AnalyzerEnums::Parity>(analyzer_enums, "Parity")
        .value("None", AnalyzerEnums::None)
        .value("Even", AnalyzerEnums::Even)
        .value("Odd", AnalyzerEnums::Odd);
    
    /*
     * Acknowledge enum:
     * Represents acknowledgment states
     * - Ack: Acknowledged
     * - Nak: Not acknowledged
     */
    py::enum_<AnalyzerEnums::Acknowledge>(analyzer_enums, "Acknowledge")
        .value("Ack", AnalyzerEnums::Ack)
        .value("Nak", AnalyzerEnums::Nak);
    
    /*
     * Sign enum:
     * Specifies if values should be interpreted as signed or unsigned
     * - UnsignedInteger: Treat as unsigned value
     * - SignedInteger: Treat as signed value
     */
    py::enum_<AnalyzerEnums::Sign>(analyzer_enums, "Sign")
        .value("UnsignedInteger", AnalyzerEnums::UnsignedInteger)
        .value("SignedInteger", AnalyzerEnums::SignedInteger);
    
    // For convenience, also expose the enums directly in the main module
    m.attr("ShiftOrder") = analyzer_enums.attr("ShiftOrder");
    m.attr("EdgeDirection") = analyzer_enums.attr("EdgeDirection");
    m.attr("Edge") = analyzer_enums.attr("Edge");
    m.attr("Parity") = analyzer_enums.attr("Parity");
    m.attr("Acknowledge") = analyzer_enums.attr("Acknowledge");
    m.attr("Sign") = analyzer_enums.attr("Sign");
}