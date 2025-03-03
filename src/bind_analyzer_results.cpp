// bind_analyzer_results.cpp
//
// Python bindings for the AnalyzerResults class and Frame class

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "AnalyzerResults.h"
#include "LogicPublicTypes.h"

namespace py = pybind11;

// Trampoline class for AnalyzerResults
// This allows Python classes to inherit from and override the virtual methods
class PyAnalyzerResults : public AnalyzerResults {
public:
    // Default constructor
    using AnalyzerResults::AnalyzerResults;

    // Override pure virtual methods with trampolines to Python
    void GenerateBubbleText(U64 frame_index, Channel &channel, DisplayBase display_base) override {
        PYBIND11_OVERRIDE_PURE(
            void,               // Return type
            AnalyzerResults,    // Parent class
            GenerateBubbleText, // Method name
            frame_index, channel, display_base // Arguments
        );
    }

    void GenerateExportFile(const char *file, DisplayBase display_base, U32 export_type_user_id) override {
        PYBIND11_OVERRIDE_PURE(
            void,               // Return type
            AnalyzerResults,    // Parent class
            GenerateExportFile, // Method name
            file, display_base, export_type_user_id // Arguments
        );
    }

    void GenerateFrameTabularText(U64 frame_index, DisplayBase display_base) override {
        PYBIND11_OVERRIDE_PURE(
            void,                    // Return type
            AnalyzerResults,         // Parent class
            GenerateFrameTabularText, // Method name
            frame_index, display_base // Arguments
        );
    }

    void GeneratePacketTabularText(U64 packet_id, DisplayBase display_base) override {
        PYBIND11_OVERRIDE_PURE(
            void,                     // Return type
            AnalyzerResults,          // Parent class
            GeneratePacketTabularText, // Method name
            packet_id, display_base   // Arguments
        );
    }

    void GenerateTransactionTabularText(U64 transaction_id, DisplayBase display_base) override {
        PYBIND11_OVERRIDE_PURE(
            void,                          // Return type
            AnalyzerResults,               // Parent class
            GenerateTransactionTabularText, // Method name
            transaction_id, display_base   // Arguments
        );
    }
};

void init_analyzer_results(py::module_ &m) {
    // Frame class
    py::class_<Frame> frame(m, "Frame", R"pbdoc(
        Represents a single protocol data frame.
        
        A frame is a decoded piece of protocol data with start/end sample positions
        and up to two 64-bit data values. The meaning of the data values depends on
        the specific analyzer and frame type.
    )pbdoc");

    frame.def(py::init<>(), "Default constructor");
    frame.def(py::init<const Frame &>(), "Copy constructor", py::arg("frame"));

    frame.def_readwrite("starting_sample", &Frame::mStartingSampleInclusive, R"pbdoc(
        The first sample index (inclusive) of this frame.
    )pbdoc");

    frame.def_readwrite("ending_sample", &Frame::mEndingSampleInclusive, R"pbdoc(
        The last sample index (inclusive) of this frame.
    )pbdoc");

    frame.def_readwrite("data1", &Frame::mData1, R"pbdoc(
        Primary data value. Meaning depends on the analyzer and frame type.
    )pbdoc");

    frame.def_readwrite("data2", &Frame::mData2, R"pbdoc(
        Secondary data value. Meaning depends on the analyzer and frame type.
    )pbdoc");

    frame.def_readwrite("type", &Frame::mType, R"pbdoc(
        Frame type identifier. Meaning depends on the specific analyzer.
    )pbdoc");

    frame.def_readwrite("flags", &Frame::mFlags, R"pbdoc(
        Flags for this frame. Can include error/warning flags.
    )pbdoc");

    frame.def("has_flag", &Frame::HasFlag, R"pbdoc(
        Check if this frame has a specific flag set.
        
        Args:
            flag: Flag to check
            
        Returns:
            bool: True if the flag is set
    )pbdoc", py::arg("flag"));

    // Define MarkerType enum
    py::enum_<AnalyzerResults::MarkerType>(m, "MarkerType", R"pbdoc(
        Types of markers that can be displayed on channels.
    )pbdoc")
        .value("Dot", AnalyzerResults::Dot, "A dot marker")
        .value("ErrorDot", AnalyzerResults::ErrorDot, "An error dot marker")
        .value("Square", AnalyzerResults::Square, "A square marker")
        .value("ErrorSquare", AnalyzerResults::ErrorSquare, "An error square marker")
        .value("UpArrow", AnalyzerResults::UpArrow, "An up arrow marker")
        .value("DownArrow", AnalyzerResults::DownArrow, "A down arrow marker")
        .value("X", AnalyzerResults::X, "An X marker")
        .value("ErrorX", AnalyzerResults::ErrorX, "An error X marker")
        .value("Start", AnalyzerResults::Start, "A start marker")
        .value("Stop", AnalyzerResults::Stop, "A stop marker")
        .value("One", AnalyzerResults::One, "A one (1) marker")
        .value("Zero", AnalyzerResults::Zero, "A zero (0) marker");

    // Add module-level constants
    m.attr("DISPLAY_AS_ERROR_FLAG") = DISPLAY_AS_ERROR_FLAG;
    m.attr("DISPLAY_AS_WARNING_FLAG") = DISPLAY_AS_WARNING_FLAG;
    m.attr("INVALID_RESULT_INDEX") = INVALID_RESULT_INDEX;

    // AnalyzerResults class
    py::class_<AnalyzerResults, PyAnalyzerResults> analyzer_results(m, "AnalyzerResults", R"pbdoc(
        Base class for analyzer results.
        
        This class stores and manages the results of an analysis, including frames,
        packets, transactions, and markers. It also handles generating text for
        display in the UI and exporting results to files.
        
        Must be subclassed to implement specific analyzer results.
    )pbdoc");

    analyzer_results.def(py::init<>(), "Default constructor");

    // Pure virtual methods
    analyzer_results.def("generate_bubble_text", &AnalyzerResults::GenerateBubbleText, R"pbdoc(
        Generate bubble text (tooltip) for a frame.
        
        This method is called when the user hovers over a frame in the UI.
        It should set the text to display using AddResultString().
        
        Args:
            frame_index: Index of the frame
            channel: Channel the frame is on
            display_base: Display base to use (binary, decimal, hex, etc.)
            
        Must be implemented by derived classes.
    )pbdoc", py::arg("frame_index"), py::arg("channel"), py::arg("display_base"));

    analyzer_results.def("generate_export_file", &AnalyzerResults::GenerateExportFile, R"pbdoc(
        Generate an export file with the analysis results.
        
        Args:
            file: Path to the output file
            display_base: Display base to use (binary, decimal, hex, etc.)
            export_type_user_id: ID of the export type selected by the user
            
        Must be implemented by derived classes.
    )pbdoc", py::arg("file"), py::arg("display_base"), py::arg("export_type_user_id"));

    analyzer_results.def("generate_frame_tabular_text", &AnalyzerResults::GenerateFrameTabularText, R"pbdoc(
        Generate tabular text for a frame.
        
        This method is called when displaying frames in the tabular view.
        It should set the text to display using AddTabularText().
        
        Args:
            frame_index: Index of the frame
            display_base: Display base to use (binary, decimal, hex, etc.)
            
        Must be implemented by derived classes.
    )pbdoc", py::arg("frame_index"), py::arg("display_base"));

    analyzer_results.def("generate_packet_tabular_text", &AnalyzerResults::GeneratePacketTabularText, R"pbdoc(
        Generate tabular text for a packet.
        
        This method is called when displaying packets in the tabular view.
        It should set the text to display using AddTabularText().
        
        Args:
            packet_id: ID of the packet
            display_base: Display base to use (binary, decimal, hex, etc.)
            
        Must be implemented by derived classes.
    )pbdoc", py::arg("packet_id"), py::arg("display_base"));

    analyzer_results.def("generate_transaction_tabular_text", &AnalyzerResults::GenerateTransactionTabularText, R"pbdoc(
        Generate tabular text for a transaction.
        
        This method is called when displaying transactions in the tabular view.
        It should set the text to display using AddTabularText().
        
        Args:
            transaction_id: ID of the transaction
            display_base: Display base to use (binary, decimal, hex, etc.)
            
        Must be implemented by derived classes.
    )pbdoc", py::arg("transaction_id"), py::arg("display_base"));

    // Methods for adding data
    analyzer_results.def("add_marker", &AnalyzerResults::AddMarker, R"pbdoc(
        Add a marker to the results.
        
        Args:
            sample_number: Sample number to place the marker at
            marker_type: Type of marker to add
            channel: Channel to place the marker on
    )pbdoc", py::arg("sample_number"), py::arg("marker_type"), py::arg("channel"));

    analyzer_results.def("add_frame", &AnalyzerResults::AddFrame, R"pbdoc(
        Add a frame to the results.
        
        Args:
            frame: Frame to add
            
        Returns:
            U64: Index of the added frame
    )pbdoc", py::arg("frame"));

    analyzer_results.def("commit_packet_and_start_new_packet", &AnalyzerResults::CommitPacketAndStartNewPacket, R"pbdoc(
        Commit the current packet and start a new one.
        
        Returns:
            U64: ID of the committed packet
    )pbdoc");

    analyzer_results.def("cancel_packet_and_start_new_packet", &AnalyzerResults::CancelPacketAndStartNewPacket, R"pbdoc(
        Cancel the current packet and start a new one.
    )pbdoc");

    analyzer_results.def("add_packet_to_transaction", &AnalyzerResults::AddPacketToTransaction, R"pbdoc(
        Add a packet to a transaction.
        
        Args:
            transaction_id: ID of the transaction
            packet_id: ID of the packet to add
    )pbdoc", py::arg("transaction_id"), py::arg("packet_id"));

    analyzer_results.def("add_channel_bubbles_will_appear_on", &AnalyzerResults::AddChannelBubblesWillAppearOn, R"pbdoc(
        Mark a channel as having bubble text.
        
        Args:
            channel: Channel that will have bubble text
    )pbdoc", py::arg("channel"));

    analyzer_results.def("commit_results", &AnalyzerResults::CommitResults, R"pbdoc(
        Commit the results to make them visible in the UI.
    )pbdoc");

    // Data access methods
    analyzer_results.def("get_num_frames", &AnalyzerResults::GetNumFrames, R"pbdoc(
        Get the number of frames in the results.
        
        Returns:
            U64: Number of frames
    )pbdoc");

    analyzer_results.def("get_num_packets", &AnalyzerResults::GetNumPackets, R"pbdoc(
        Get the number of packets in the results.
        
        Returns:
            U64: Number of packets
    )pbdoc");

    analyzer_results.def("get_frame", &AnalyzerResults::GetFrame, R"pbdoc(
        Get a frame by index.
        
        Args:
            frame_id: Index of the frame
            
        Returns:
            Frame: The requested frame
    )pbdoc", py::arg("frame_id"));

    analyzer_results.def("get_packet_containing_frame", &AnalyzerResults::GetPacketContainingFrame, R"pbdoc(
        Get the packet that contains a frame.
        
        Args:
            frame_id: Index of the frame
            
        Returns:
            U64: ID of the packet containing the frame, or INVALID_RESULT_INDEX if not found
    )pbdoc", py::arg("frame_id"));

    analyzer_results.def("get_packet_containing_frame_sequential", &AnalyzerResults::GetPacketContainingFrameSequential, R"pbdoc(
        Get the packet that contains a frame, using sequential search.
        
        This is slower but more reliable than get_packet_containing_frame.
        
        Args:
            frame_id: Index of the frame
            
        Returns:
            U64: ID of the packet containing the frame, or INVALID_RESULT_INDEX if not found
    )pbdoc", py::arg("frame_id"));

    analyzer_results.def("get_frames_contained_in_packet", [](AnalyzerResults &self, U64 packet_id) {
        U64 first_frame_id, last_frame_id;
        self.GetFramesContainedInPacket(packet_id, &first_frame_id, &last_frame_id);
        return py::make_tuple(first_frame_id, last_frame_id);
    }, R"pbdoc(
        Get the range of frames contained in a packet.
        
        Args:
            packet_id: ID of the packet
            
        Returns:
            tuple: (first_frame_id, last_frame_id)
    )pbdoc", py::arg("packet_id"));

    analyzer_results.def("get_transaction_containing_packet", &AnalyzerResults::GetTransactionContainingPacket, R"pbdoc(
        Get the transaction that contains a packet.
        
        Args:
            packet_id: ID of the packet
            
        Returns:
            U32: ID of the transaction containing the packet
    )pbdoc", py::arg("packet_id"));

    analyzer_results.def("get_packets_contained_in_transaction", [](AnalyzerResults &self, U64 transaction_id) {
        U64 *packet_id_array;
        U64 packet_id_count;
        self.GetPacketsContainedInTransaction(transaction_id, &packet_id_array, &packet_id_count);
        
        py::list result;
        for (U64 i = 0; i < packet_id_count; i++) {
            result.append(packet_id_array[i]);
        }
        return result;
    }, R"pbdoc(
        Get the packets contained in a transaction.
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            list: List of packet IDs
    )pbdoc", py::arg("transaction_id"));

    // Text result methods
    analyzer_results.def("clear_result_strings", &AnalyzerResults::ClearResultStrings, R"pbdoc(
        Clear all result strings.
    )pbdoc");

    analyzer_results.def("add_result_string", [](AnalyzerResults &self, const char *str1, 
                                                const char *str2, const char *str3,
                                                const char *str4, const char *str5,
                                                const char *str6) {
        self.AddResultString(str1, str2, str3, str4, str5, str6);
    }, R"pbdoc(
        Add a result string for display in bubble text.
        
        Multiple strings will be concatenated.
        
        Args:
            str1: First string
            str2: Second string (optional)
            str3: Third string (optional)
            str4: Fourth string (optional)
            str5: Fifth string (optional)
            str6: Sixth string (optional)
    )pbdoc", 
    py::arg("str1"), 
    py::arg("str2") = nullptr, 
    py::arg("str3") = nullptr,
    py::arg("str4") = nullptr,
    py::arg("str5") = nullptr,
    py::arg("str6") = nullptr);

    analyzer_results.def("get_result_strings", [](AnalyzerResults &self) {
        char const **result_string_array;
        U32 num_strings;
        self.GetResultStrings(&result_string_array, &num_strings);
        
        py::list result;
        for (U32 i = 0; i < num_strings; i++) {
            result.append(result_string_array[i]);
        }
        return result;
    }, R"pbdoc(
        Get all result strings.
        
        Returns:
            list: List of result strings
    )pbdoc");

    // Tabular text methods
    analyzer_results.def("clear_tabular_text", &AnalyzerResults::ClearTabularText, R"pbdoc(
        Clear all tabular text.
    )pbdoc");

    analyzer_results.def("add_tabular_text", [](AnalyzerResults &self, const char *str1, 
                                               const char *str2, const char *str3,
                                               const char *str4, const char *str5,
                                               const char *str6) {
        self.AddTabularText(str1, str2, str3, str4, str5, str6);
    }, R"pbdoc(
        Add text for display in the tabular view.
        
        Multiple strings will be concatenated.
        
        Args:
            str1: First string
            str2: Second string (optional)
            str3: Third string (optional)
            str4: Fourth string (optional)
            str5: Fifth string (optional)
            str6: Sixth string (optional)
    )pbdoc", 
    py::arg("str1"), 
    py::arg("str2") = nullptr, 
    py::arg("str3") = nullptr,
    py::arg("str4") = nullptr,
    py::arg("str5") = nullptr,
    py::arg("str6") = nullptr);

    analyzer_results.def("get_tabular_text_string", &AnalyzerResults::GetTabularTextString, R"pbdoc(
        Get the complete tabular text string.
        
        Returns:
            str: Tabular text string
    )pbdoc");

    // Utility methods
    analyzer_results.def("get_string_for_display_base", &AnalyzerResults::GetStringForDisplayBase, R"pbdoc(
        Get a string representation of a frame value using the specified display base.
        
        Args:
            frame_id: Index of the frame
            channel: Channel to get the value for
            disp_base: Display base to use
            
        Returns:
            str: String representation of the value
    )pbdoc", py::arg("frame_id"), py::arg("channel"), py::arg("disp_base"));

    analyzer_results.def("update_export_progress_and_check_for_cancel", &AnalyzerResults::UpdateExportProgressAndCheckForCancel, R"pbdoc(
        Update export progress and check if the user has canceled.
        
        Args:
            completed_frames: Number of frames processed so far
            total_frames: Total number of frames to process
            
        Returns:
            bool: True if export should continue, False if canceled
    )pbdoc", py::arg("completed_frames"), py::arg("total_frames"));

    analyzer_results.def("do_bubbles_appear_on_channel", &AnalyzerResults::DoBubblesAppearOnChannel, R"pbdoc(
        Check if bubble text appears on a channel.
        
        Args:
            channel: Channel to check
            
        Returns:
            bool: True if bubble text appears on the channel
    )pbdoc", py::arg("channel"));

    analyzer_results.def("do_markers_appear_on_channel", &AnalyzerResults::DoMarkersAppearOnChannel, R"pbdoc(
        Check if markers appear on a channel.
        
        Args:
            channel: Channel to check
            
        Returns:
            bool: True if markers appear on the channel
    )pbdoc", py::arg("channel"));

    analyzer_results.def("get_frames_in_range", [](AnalyzerResults &self, S64 starting_sample_inclusive, 
                                                  S64 ending_sample_inclusive) {
        U64 first_frame_index, last_frame_index;
        bool result = self.GetFramesInRange(starting_sample_inclusive, ending_sample_inclusive, 
                                           &first_frame_index, &last_frame_index);
        
        if (result) {
            return py::make_tuple(true, first_frame_index, last_frame_index);
        } else {
            return py::make_tuple(false, 0, 0);
        }
    }, R"pbdoc(
        Get frames within a sample range.
        
        Args:
            starting_sample_inclusive: First sample in the range
            ending_sample_inclusive: Last sample in the range
            
        Returns:
            tuple: (found, first_frame_index, last_frame_index)
                found: True if frames were found in the range
                first_frame_index: Index of the first frame in the range
                last_frame_index: Index of the last frame in the range
    )pbdoc", py::arg("starting_sample_inclusive"), py::arg("ending_sample_inclusive"));

    analyzer_results.def("get_markers_in_range", [](AnalyzerResults &self, Channel &channel, 
                                                   S64 starting_sample_inclusive, 
                                                   S64 ending_sample_inclusive) {
        U64 first_marker_index, last_marker_index;
        bool result = self.GetMarkersInRange(channel, starting_sample_inclusive, 
                                            ending_sample_inclusive, 
                                            &first_marker_index, &last_marker_index);
        
        if (result) {
            return py::make_tuple(true, first_marker_index, last_marker_index);
        } else {
            return py::make_tuple(false, 0, 0);
        }
    }, R"pbdoc(
        Get markers within a sample range.
        
        Args:
            channel: Channel to get markers for
            starting_sample_inclusive: First sample in the range
            ending_sample_inclusive: Last sample in the range
            
        Returns:
            tuple: (found, first_marker_index, last_marker_index)
                found: True if markers were found in the range
                first_marker_index: Index of the first marker in the range
                last_marker_index: Index of the last marker in the range
    )pbdoc", py::arg("channel"), py::arg("starting_sample_inclusive"), py::arg("ending_sample_inclusive"));

    analyzer_results.def("get_marker", [](AnalyzerResults &self, Channel &channel, U64 marker_index) {
        AnalyzerResults::MarkerType marker_type;
        U64 marker_sample;
        self.GetMarker(channel, marker_index, &marker_type, &marker_sample);
        return py::make_tuple(marker_type, marker_sample);
    }, R"pbdoc(
        Get information about a marker.
        
        Args:
            channel: Channel the marker is on
            marker_index: Index of the marker
            
        Returns:
            tuple: (marker_type, marker_sample)
                marker_type: Type of the marker
                marker_sample: Sample number of the marker
    )pbdoc", py::arg("channel"), py::arg("marker_index"));

    analyzer_results.def("get_num_markers", &AnalyzerResults::GetNumMarkers, R"pbdoc(
        Get the number of markers on a channel.
        
        Args:
            channel: Channel to count markers for
            
        Returns:
            U64: Number of markers
    )pbdoc", py::arg("channel"));

    analyzer_results.def("cancel_export", &AnalyzerResults::CancelExport, R"pbdoc(
        Cancel an in-progress export operation.
    )pbdoc");

    analyzer_results.def("get_progress", &AnalyzerResults::GetProgress, R"pbdoc(
        Get the current export progress.
        
        Returns:
            double: Progress as a value between 0.0 and 1.0
    )pbdoc");
}