// bind_basic_types.cpp

#include <pybind11/pybind11.h>
#include "LogicPublicTypes.h"
#include "AnalyzerTypes.h"

namespace py = pybind11;

void init_basic_types(py::module_ &m) {
    // LogicPublicTypes.h - Basic types and enums
    
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

    /*
     * Channel class:
     * Represents a physical channel on the logic analyzer
     * - mDeviceId: Identifies the logic analyzer device
     * - mChannelIndex: Identifies the specific channel on the device
     */
    py::class_<Channel>(m, "Channel")
        .def(py::init<>(), "Default constructor")
        .def(py::init<U64, U32>(), "Construct with device_id and channel_index",
             py::arg("device_id"), py::arg("channel_index"))
        .def_readwrite("device_id", &Channel::mDeviceId,
                       "Identifier for the logic analyzer device")
        .def_readwrite("channel_index", &Channel::mChannelIndex,
                       "Channel number on the device")
        .def("__eq__", &Channel::operator==)
        .def("__ne__", &Channel::operator!=)
        .def("__lt__", &Channel::operator<)
        .def("__gt__", &Channel::operator>);

    // AnalyzerTypes.h - Enums for analyzer configuration

    /*
     * ShiftOrder enum:
     * Specifies the bit order for serial protocols
     * - MsbFirst: Most significant bit first
     * - LsbFirst: Least significant bit first
     */
    py::enum_<AnalyzerEnums::ShiftOrder>(m, "ShiftOrder")
        .value("MsbFirst", AnalyzerEnums::MsbFirst)
        .value("LsbFirst", AnalyzerEnums::LsbFirst);

    /*
     * EdgeDirection enum:
     * Defines edge directions for triggering or sampling
     * - PosEdge: Rising edge (low to high)
     * - NegEdge: Falling edge (high to low)
     */
    py::enum_<AnalyzerEnums::EdgeDirection>(m, "EdgeDirection")
        .value("PosEdge", AnalyzerEnums::PosEdge)
        .value("NegEdge", AnalyzerEnums::NegEdge);

    /*
     * Other AnalyzerEnums types
     * Edge, Parity, Acknowledge, Sign
     */
    py::enum_<AnalyzerEnums::Edge>(m, "Edge")
        .value("LeadingEdge", AnalyzerEnums::LeadingEdge)
        .value("TrailingEdge", AnalyzerEnums::TrailingEdge);
    
    py::enum_<AnalyzerEnums::Parity>(m, "Parity")
        .value("None", AnalyzerEnums::None)
        .value("Even", AnalyzerEnums::Even)
        .value("Odd", AnalyzerEnums::Odd);
    
    py::enum_<AnalyzerEnums::Acknowledge>(m, "Acknowledge")
        .value("Ack", AnalyzerEnums::Ack)
        .value("Nak", AnalyzerEnums::Nak);
    
    py::enum_<AnalyzerEnums::Sign>(m, "Sign")
        .value("UnsignedInteger", AnalyzerEnums::UnsignedInteger)
        .value("SignedInteger", AnalyzerEnums::SignedInteger);
}