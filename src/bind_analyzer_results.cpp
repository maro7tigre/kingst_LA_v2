// bind_analyzer_results.cpp
//
// Python bindings for the AnalyzerResults class and Frame class
// Following the Pybind11 Binding Style Guide for Kingst LA

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "AnalyzerResults.h"
#include "LogicPublicTypes.h"

namespace py = pybind11;

// Wrapper class to access protected methods
class AnalyzerResultsWrapper : public AnalyzerResults {
public:
    // Don't inherit constructors - define manually
    AnalyzerResultsWrapper() : AnalyzerResults() {}

    // Add implementations for pure virtual methods
    void GenerateBubbleText(U64 frame_index, Channel &channel, DisplayBase display_base) override {}
    void GenerateExportFile(const char *file, DisplayBase display_base, U32 export_type_user_id) override {}
    void GenerateFrameTabularText(U64 frame_index, DisplayBase display_base) override {}
    void GeneratePacketTabularText(U64 packet_id, DisplayBase display_base) override {}
    void GenerateTransactionTabularText(U64 transaction_id, DisplayBase display_base) override {}

    // Method to expose protected UpdateExportProgressAndCheckForCancel
    bool PublicUpdateExportProgressAndCheckForCancel(U64 completed_frames, U64 total_frames) {
        return UpdateExportProgressAndCheckForCancel(completed_frames, total_frames);
    }
};

// Trampoline class for AnalyzerResults
// This allows Python classes to inherit from and override the virtual methods
class PyAnalyzerResults : public AnalyzerResults {
public:
    // Default constructor
    PyAnalyzerResults() : AnalyzerResults() {}

    // =============== Pure Virtual Methods ===============
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
    // =============== Module-level constants ===============
    m.attr("DISPLAY_AS_ERROR_FLAG") = DISPLAY_AS_ERROR_FLAG;
    m.attr("DISPLAY_AS_WARNING_FLAG") = DISPLAY_AS_WARNING_FLAG;
    m.attr("INVALID_RESULT_INDEX") = INVALID_RESULT_INDEX;
    
    // =============== Frame class ===============
    py::class_<Frame> frame(m, "Frame", R"pbdoc(
        Represents a single protocol data frame.
        
        A frame is a decoded piece of protocol data with start/end sample positions
        and up to two 64-bit data values. The meaning of the data values depends on
        the specific analyzer and frame type.
    )pbdoc");

    frame.def(py::init<>(), R"pbdoc(
        Default constructor.
        
        Creates an empty frame with all fields initialized to zero.
    )pbdoc");
    
    frame.def(py::init<const Frame &>(), R"pbdoc(
        Copy constructor.
        
        Creates a new frame that is a copy of an existing frame.
        
        Args:
            frame: Source frame to copy from
    )pbdoc", py::arg("frame"));

    frame.def_readwrite("starting_sample", &Frame::mStartingSampleInclusive, R"pbdoc(
        The first sample index (inclusive) of this frame.
        
        Type: S64 (signed 64-bit integer)
    )pbdoc");

    frame.def_readwrite("ending_sample", &Frame::mEndingSampleInclusive, R"pbdoc(
        The last sample index (inclusive) of this frame.
        
        Type: S64 (signed 64-bit integer)
    )pbdoc");

    frame.def_readwrite("data1", &Frame::mData1, R"pbdoc(
        Primary data value. Meaning depends on the analyzer and frame type.
        
        Type: U64 (unsigned 64-bit integer)
    )pbdoc");

    frame.def_readwrite("data2", &Frame::mData2, R"pbdoc(
        Secondary data value. Meaning depends on the analyzer and frame type.
        
        Type: U64 (unsigned 64-bit integer)
    )pbdoc");

    frame.def_readwrite("type", &Frame::mType, R"pbdoc(
        Frame type identifier. Meaning depends on the specific analyzer.
        
        Type: U8 (unsigned 8-bit integer)
    )pbdoc");

    frame.def_readwrite("flags", &Frame::mFlags, R"pbdoc(
        Flags for this frame. Can include error/warning flags.
        
        Common flags:
        - DISPLAY_AS_ERROR_FLAG (0x80): Display this frame as an error
        - DISPLAY_AS_WARNING_FLAG (0x40): Display this frame as a warning
        
        Type: U8 (unsigned 8-bit integer)
    )pbdoc");

    frame.def("has_flag", &Frame::HasFlag, R"pbdoc(
        Check if this frame has a specific flag set.
        
        Args:
            flag (U8): Flag to check
            
        Returns:
            bool: True if the flag is set, False otherwise
    )pbdoc", py::arg("flag"));

    // =============== MarkerType enum ===============
    py::enum_<AnalyzerResults::MarkerType>(m, "MarkerType", R"pbdoc(
        Types of markers that can be displayed on channels.
        
        Markers are visual indicators shown on the channel waveforms.
    )pbdoc")
        .value("DOT", AnalyzerResults::Dot, R"pbdoc(A normal dot marker)pbdoc")
        .value("ERROR_DOT", AnalyzerResults::ErrorDot, R"pbdoc(An error dot marker (red))pbdoc")
        .value("SQUARE", AnalyzerResults::Square, R"pbdoc(A square marker)pbdoc")
        .value("ERROR_SQUARE", AnalyzerResults::ErrorSquare, R"pbdoc(An error square marker (red))pbdoc")
        .value("UP_ARROW", AnalyzerResults::UpArrow, R"pbdoc(An up arrow marker)pbdoc")
        .value("DOWN_ARROW", AnalyzerResults::DownArrow, R"pbdoc(A down arrow marker)pbdoc")
        .value("X", AnalyzerResults::X, R"pbdoc(An X marker)pbdoc")
        .value("ERROR_X", AnalyzerResults::ErrorX, R"pbdoc(An error X marker (red))pbdoc")
        .value("START", AnalyzerResults::Start, R"pbdoc(A start marker)pbdoc")
        .value("STOP", AnalyzerResults::Stop, R"pbdoc(A stop marker)pbdoc")
        .value("ONE", AnalyzerResults::One, R"pbdoc(A one (1) marker)pbdoc")
        .value("ZERO", AnalyzerResults::Zero, R"pbdoc(A zero (0) marker)pbdoc");

    // =============== AnalyzerResults class ===============
    py::class_<AnalyzerResults, PyAnalyzerResults> analyzer_results(m, "AnalyzerResults", R"pbdoc(
        Base class for analyzer results.
        
        This class stores and manages the results of an analysis, including frames,
        packets, transactions, and markers. It also handles generating text for
        display in the UI and exporting results to files.
        
        Must be subclassed to implement specific analyzer results. The subclass 
        must implement all the pure virtual methods.
    )pbdoc");

    analyzer_results.def(py::init<>(), R"pbdoc(
        Default constructor.
        
        Creates an empty set of analyzer results.
    )pbdoc");

    // =============== 1. Pure Virtual Methods ===============
    analyzer_results.def("generate_bubble_text", &AnalyzerResults::GenerateBubbleText, R"pbdoc(
        Generate bubble text (tooltip) for a frame.
        
        This method is called when the user hovers over a frame in the UI.
        It should set the text to display using add_result_string().
        
        Args:
            frame_index (U64): Index of the frame
            channel (Channel): Channel the frame is on
            display_base (DisplayBase): Display base to use (binary, decimal, hex, etc.)
            
        MUST be implemented by derived classes.
    )pbdoc", py::arg("frame_index"), py::arg("channel"), py::arg("display_base"));

    analyzer_results.def("generate_export_file", &AnalyzerResults::GenerateExportFile, R"pbdoc(
        Generate an export file with the analysis results.
        
        Args:
            file (str): Path to the output file
            display_base (DisplayBase): Display base to use (binary, decimal, hex, etc.)
            export_type_user_id (U32): ID of the export type selected by the user
            
        MUST be implemented by derived classes.
        
        Implementation Notes:
        - Can call update_export_progress_and_check_for_cancel() to update progress
        - Should respect the selected display_base format for numerical output
    )pbdoc", py::arg("file"), py::arg("display_base"), py::arg("export_type_user_id"));

    analyzer_results.def("generate_frame_tabular_text", &AnalyzerResults::GenerateFrameTabularText, R"pbdoc(
        Generate tabular text for a frame.
        
        This method is called when displaying frames in the tabular view.
        It should set the text to display using add_tabular_text().
        
        Args:
            frame_index (U64): Index of the frame
            display_base (DisplayBase): Display base to use (binary, decimal, hex, etc.)
            
        MUST be implemented by derived classes.
    )pbdoc", py::arg("frame_index"), py::arg("display_base"));

    analyzer_results.def("generate_packet_tabular_text", &AnalyzerResults::GeneratePacketTabularText, R"pbdoc(
        Generate tabular text for a packet.
        
        This method is called when displaying packets in the tabular view.
        It should set the text to display using add_tabular_text().
        
        Args:
            packet_id (U64): ID of the packet
            display_base (DisplayBase): Display base to use (binary, decimal, hex, etc.)
            
        MUST be implemented by derived classes.
    )pbdoc", py::arg("packet_id"), py::arg("display_base"));

    analyzer_results.def("generate_transaction_tabular_text", &AnalyzerResults::GenerateTransactionTabularText, R"pbdoc(
        Generate tabular text for a transaction.
        
        This method is called when displaying transactions in the tabular view.
        It should set the text to display using add_tabular_text().
        
        Args:
            transaction_id (U64): ID of the transaction
            display_base (DisplayBase): Display base to use (binary, decimal, hex, etc.)
            
        MUST be implemented by derived classes.
    )pbdoc", py::arg("transaction_id"), py::arg("display_base"));

    // =============== 2. Data Addition Methods ===============
    analyzer_results.def("add_marker", &AnalyzerResults::AddMarker, R"pbdoc(
        Add a marker to the results.
        
        Markers appear as visual indicators on the waveform display.
        
        Args:
            sample_number (U64): Sample number to place the marker at
            marker_type (MarkerType): Type of marker to add
            channel (Channel): Channel to place the marker on
    )pbdoc", py::arg("sample_number"), py::arg("marker_type"), py::arg("channel"));

    analyzer_results.def("add_frame", &AnalyzerResults::AddFrame, R"pbdoc(
        Add a frame to the results.
        
        Frames are the basic units of decoded data. Most analyzers
        will continuously add frames during the analysis process.
        
        Args:
            frame (Frame): Frame to add
            
        Returns:
            U64: Index of the added frame
    )pbdoc", py::arg("frame"));

    analyzer_results.def("commit_packet_and_start_new_packet", &AnalyzerResults::CommitPacketAndStartNewPacket, R"pbdoc(
        Commit the current packet and start a new one.
        
        A packet is a group of frames that together form a logical unit.
        Frames added after calling this function will belong to the new packet.
        
        Returns:
            U64: ID of the committed packet
    )pbdoc");

    analyzer_results.def("cancel_packet_and_start_new_packet", &AnalyzerResults::CancelPacketAndStartNewPacket, R"pbdoc(
        Cancel the current packet and start a new one.
        
        All frames added since the last commit will be removed from any packet association.
        Frames added after calling this function will belong to the new packet.
    )pbdoc");

    analyzer_results.def("add_packet_to_transaction", &AnalyzerResults::AddPacketToTransaction, R"pbdoc(
        Add a packet to a transaction.
        
        A transaction is a group of packets that together form a logical unit.
        
        Args:
            transaction_id (U64): ID of the transaction
            packet_id (U64): ID of the packet to add
    )pbdoc", py::arg("transaction_id"), py::arg("packet_id"));

    analyzer_results.def("add_channel_bubbles_will_appear_on", &AnalyzerResults::AddChannelBubblesWillAppearOn, R"pbdoc(
        Mark a channel as having bubble text.
        
        This must be called for any channel that will have bubble text displayed.
        
        Args:
            channel (Channel): Channel that will have bubble text
    )pbdoc", py::arg("channel"));

    analyzer_results.def("commit_results", &AnalyzerResults::CommitResults, R"pbdoc(
        Commit the results to make them visible in the UI.
        
        This should be called periodically during analysis to update the UI.
        It's particularly important for long-running analyzers to show progress.
    )pbdoc");

    // =============== 3. Data Access Methods ===============
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
            frame_id (U64): Index of the frame
            
        Returns:
            Frame: The requested frame
    )pbdoc", py::arg("frame_id"));

    analyzer_results.def("get_packet_containing_frame", &AnalyzerResults::GetPacketContainingFrame, R"pbdoc(
        Get the packet that contains a frame.
        
        Args:
            frame_id (U64): Index of the frame
            
        Returns:
            U64: ID of the packet containing the frame, or INVALID_RESULT_INDEX if not found
            
        Note: This uses an index-based lookup which is fast but may not always find
        the packet if frames were added in a non-sequential order.
    )pbdoc", py::arg("frame_id"));

    analyzer_results.def("get_packet_containing_frame_sequential", &AnalyzerResults::GetPacketContainingFrameSequential, R"pbdoc(
        Get the packet that contains a frame, using sequential search.
        
        This is slower but more reliable than get_packet_containing_frame.
        
        Args:
            frame_id (U64): Index of the frame
            
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
            packet_id (U64): ID of the packet
            
        Returns:
            tuple: (first_frame_id, last_frame_id)
                - first_frame_id (U64): Index of the first frame in the packet
                - last_frame_id (U64): Index of the last frame in the packet
    )pbdoc", py::arg("packet_id"));

    analyzer_results.def("get_transaction_containing_packet", &AnalyzerResults::GetTransactionContainingPacket, R"pbdoc(
        Get the transaction that contains a packet.
        
        Args:
            packet_id (U64): ID of the packet
            
        Returns:
            U32: ID of the transaction containing the packet, or 0 if not found
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
            transaction_id (U64): ID of the transaction
            
        Returns:
            list: List of packet IDs (U64)
    )pbdoc", py::arg("transaction_id"));

    // =============== 4. Marker Methods ===============
    analyzer_results.def("get_num_markers", &AnalyzerResults::GetNumMarkers, R"pbdoc(
        Get the number of markers on a channel.
        
        Args:
            channel (Channel): Channel to count markers for
            
        Returns:
            U64: Number of markers
    )pbdoc", py::arg("channel"));

    analyzer_results.def("get_marker", [](AnalyzerResults &self, Channel &channel, U64 marker_index) {
        AnalyzerResults::MarkerType marker_type;
        U64 marker_sample;
        self.GetMarker(channel, marker_index, &marker_type, &marker_sample);
        return py::make_tuple(marker_type, marker_sample);
    }, R"pbdoc(
        Get information about a marker.
        
        Args:
            channel (Channel): Channel the marker is on
            marker_index (U64): Index of the marker
            
        Returns:
            tuple: (marker_type, marker_sample)
                - marker_type (MarkerType): Type of the marker
                - marker_sample (U64): Sample number of the marker
    )pbdoc", py::arg("channel"), py::arg("marker_index"));

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
            channel (Channel): Channel to get markers for
            starting_sample_inclusive (S64): First sample in the range
            ending_sample_inclusive (S64): Last sample in the range
            
        Returns:
            tuple: (found, first_marker_index, last_marker_index)
                - found (bool): True if markers were found in the range
                - first_marker_index (U64): Index of the first marker in the range
                - last_marker_index (U64): Index of the last marker in the range
    )pbdoc", py::arg("channel"), py::arg("starting_sample_inclusive"), py::arg("ending_sample_inclusive"));

    analyzer_results.def("do_markers_appear_on_channel", &AnalyzerResults::DoMarkersAppearOnChannel, R"pbdoc(
        Check if markers appear on a channel.
        
        Args:
            channel (Channel): Channel to check
            
        Returns:
            bool: True if markers appear on the channel
    )pbdoc", py::arg("channel"));

    // =============== 5. Result Text Methods ===============
    analyzer_results.def("clear_result_strings", &AnalyzerResults::ClearResultStrings, R"pbdoc(
        Clear all result strings.
        
        This should typically be called at the start of generate_bubble_text().
    )pbdoc");

    analyzer_results.def("add_result_string", [](AnalyzerResults &self, const char *str1, 
                                               const char *str2, const char *str3,
                                               const char *str4, const char *str5,
                                               const char *str6) {
        self.AddResultString(str1, str2, str3, str4, str5, str6);
    }, R"pbdoc(
        Add a result string for display in bubble text.
        
        Multiple strings will be concatenated. This is typically called from
        generate_bubble_text() to set text that appears when hovering over a frame.
        
        Args:
            str1 (str): First string (required)
            str2 (str, optional): Second string. Defaults to None.
            str3 (str, optional): Third string. Defaults to None.
            str4 (str, optional): Fourth string. Defaults to None.
            str5 (str, optional): Fifth string. Defaults to None.
            str6 (str, optional): Sixth string. Defaults to None.
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
            list: List of result strings (str)
    )pbdoc");

    // =============== 6. Tabular Text Methods ===============
    analyzer_results.def("clear_tabular_text", &AnalyzerResults::ClearTabularText, R"pbdoc(
        Clear all tabular text.
        
        This should typically be called at the start of generate_frame_tabular_text(),
        generate_packet_tabular_text(), or generate_transaction_tabular_text().
    )pbdoc");

    analyzer_results.def("add_tabular_text", [](AnalyzerResults &self, const char *str1, 
                                              const char *str2, const char *str3,
                                              const char *str4, const char *str5,
                                              const char *str6) {
        self.AddTabularText(str1, str2, str3, str4, str5, str6);
    }, R"pbdoc(
        Add text for display in the tabular view.
        
        Multiple strings will be concatenated. This is typically called from
        the generate_*_tabular_text() methods.
        
        Args:
            str1 (str): First string (required)
            str2 (str, optional): Second string. Defaults to None.
            str3 (str, optional): Third string. Defaults to None.
            str4 (str, optional): Fourth string. Defaults to None.
            str5 (str, optional): Fifth string. Defaults to None.
            str6 (str, optional): Sixth string. Defaults to None.
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

    // =============== 7. Utility Methods ===============
    analyzer_results.def("get_string_for_display_base", &AnalyzerResults::GetStringForDisplayBase, R"pbdoc(
        Get a string representation of a frame value using the specified display base.
        
        Args:
            frame_id (U64): Index of the frame
            channel (Channel): Channel to get the value for
            disp_base (DisplayBase): Display base to use
            
        Returns:
            str: String representation of the value
    )pbdoc", py::arg("frame_id"), py::arg("channel"), py::arg("disp_base"));

    analyzer_results.def("do_bubbles_appear_on_channel", &AnalyzerResults::DoBubblesAppearOnChannel, R"pbdoc(
        Check if bubble text appears on a channel.
        
        Args:
            channel (Channel): Channel to check
            
        Returns:
            bool: True if bubble text appears on the channel
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
            starting_sample_inclusive (S64): First sample in the range
            ending_sample_inclusive (S64): Last sample in the range
            
        Returns:
            tuple: (found, first_frame_index, last_frame_index)
                - found (bool): True if frames were found in the range
                - first_frame_index (U64): Index of the first frame in the range
                - last_frame_index (U64): Index of the last frame in the range
    )pbdoc", py::arg("starting_sample_inclusive"), py::arg("ending_sample_inclusive"));

    // =============== 8. Search Methods ===============
    analyzer_results.def("build_search_data", [](AnalyzerResults &self, U64 frame_id, DisplayBase disp_base, int channel_list_index) {
        char result[1024]; // Buffer for the result
        const char* search_str = self.BuildSearchData(frame_id, disp_base, channel_list_index, result);
        return std::string(search_str);
    }, R"pbdoc(
        Build search data for a frame.
        
        This is used by the search feature in the UI.
        
        Args:
            frame_id (U64): Index of the frame
            disp_base (DisplayBase): Display base to use
            channel_list_index (int): Index of the channel in the channel list
            
        Returns:
            str: Search data string
    )pbdoc", py::arg("frame_id"), py::arg("disp_base"), py::arg("channel_list_index"));

    // =============== 9. Export Control Methods ===============
    analyzer_results.def("start_export_thread", &AnalyzerResults::StartExportThread, R"pbdoc(
        Start an export operation in a background thread.
        
        This method starts a thread that calls generate_export_file().
        
        Args:
            file (str): Path to the output file
            display_base (DisplayBase): Display base to use
            export_type_user_id (U32): ID of the export type selected by the user
    )pbdoc", py::arg("file"), py::arg("display_base"), py::arg("export_type_user_id"));

    // Create a wrapper instance to access protected method
    AnalyzerResultsWrapper wrapper;
    analyzer_results.def("update_export_progress_and_check_for_cancel", [&wrapper](AnalyzerResults& self, U64 completed_frames, U64 total_frames) {
        // Using the wrapper to call the protected method
        return wrapper.PublicUpdateExportProgressAndCheckForCancel(completed_frames, total_frames);
    }, R"pbdoc(
        Update export progress and check if the user has canceled.
        
        This should be called periodically during generate_export_file() to update
        the progress indicator and check if the user has canceled the export.
        
        Args:
            completed_frames (U64): Number of frames processed so far
            total_frames (U64): Total number of frames to process
            
        Returns:
            bool: True if export should continue, False if canceled
    )pbdoc", py::arg("completed_frames"), py::arg("total_frames"));

    analyzer_results.def("cancel_export", &AnalyzerResults::CancelExport, R"pbdoc(
        Cancel an in-progress export operation.
        
        This can be called to cancel an export operation started with start_export_thread().
    )pbdoc");

    analyzer_results.def("get_progress", &AnalyzerResults::GetProgress, R"pbdoc(
        Get the current export progress.
        
        Returns:
            double: Progress as a value between 0.0 and 1.0
    )pbdoc");
}