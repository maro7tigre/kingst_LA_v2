// bind_analyzer_settings.cpp
//
// Python bindings for the AnalyzerSettings base class and setting interfaces

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "AnalyzerSettings.h"
#include "AnalyzerSettingInterface.h"
#include "LogicPublicTypes.h"

namespace py = pybind11;

// Trampoline class for AnalyzerSettings
// This allows Python classes to inherit from and override the virtual methods
class PyAnalyzerSettings : public AnalyzerSettings {
public:
    // Default constructor
    using AnalyzerSettings::AnalyzerSettings;

    // Override pure virtual methods with trampolines to Python
    bool SetSettingsFromInterfaces() override {
        PYBIND11_OVERRIDE_PURE(
            bool,                      // Return type
            AnalyzerSettings,          // Parent class
            SetSettingsFromInterfaces  // Method name
        );
    }

    void LoadSettings(const char *settings) override {
        PYBIND11_OVERRIDE_PURE(
            void,                // Return type
            AnalyzerSettings,    // Parent class
            LoadSettings,        // Method name
            settings             // Arguments
        );
    }

    const char* SaveSettings() override {
        PYBIND11_OVERRIDE_PURE(
            const char*,         // Return type
            AnalyzerSettings,    // Parent class
            SaveSettings         // Method name
        );
    }

    // Override non-pure virtual methods
    const char* GetSettingBrief() override {
        PYBIND11_OVERRIDE(
            const char*,         // Return type
            AnalyzerSettings,    // Parent class
            GetSettingBrief      // Method name
        );
    }
};

void init_analyzer_settings(py::module_ &m) {
    // AnalyzerSettingInterface - Base class for all setting interfaces
    py::class_<AnalyzerSettingInterface> setting_interface(m, "AnalyzerSettingInterface", R"pbdoc(
        Base class for all analyzer setting interfaces.
        
        Setting interfaces are used to create UI controls for configuring analyzers.
        This is the base class that all specific interface types inherit from.
    )pbdoc");

    setting_interface.def("get_tool_tip", &AnalyzerSettingInterface::GetToolTip, R"pbdoc(
        Get the tooltip text for this setting.
        
        Returns:
            str: Tooltip text
    )pbdoc");

    setting_interface.def("get_title", &AnalyzerSettingInterface::GetTitle, R"pbdoc(
        Get the title/label for this setting.
        
        Returns:
            str: Title text
    )pbdoc");

    setting_interface.def("is_disabled", &AnalyzerSettingInterface::IsDisabled, R"pbdoc(
        Check if this setting is disabled.
        
        Returns:
            bool: True if the setting is disabled
    )pbdoc");

    setting_interface.def("set_title_and_tooltip", &AnalyzerSettingInterface::SetTitleAndTooltip, R"pbdoc(
        Set the title and tooltip for this setting.
        
        Args:
            title: Title/label text
            tooltip: Tooltip text
    )pbdoc", py::arg("title"), py::arg("tooltip"));

    // AnalyzerSettingInterfaceChannel - Channel selection setting
    py::class_<AnalyzerSettingInterfaceChannel, AnalyzerSettingInterface> channel_interface(
        m, "AnalyzerSettingInterfaceChannel", R"pbdoc(
        Interface for channel selection settings.
        
        This interface allows the user to select a channel from the available channels.
    )pbdoc");

    channel_interface.def(py::init<>(), "Default constructor");

    channel_interface.def("get_channel", &AnalyzerSettingInterfaceChannel::GetChannel, R"pbdoc(
        Get the currently selected channel.
        
        Returns:
            Channel: The selected channel
    )pbdoc");

    channel_interface.def("set_channel", &AnalyzerSettingInterfaceChannel::SetChannel, R"pbdoc(
        Set the selected channel.
        
        Args:
            channel: The channel to select
    )pbdoc", py::arg("channel"));

    channel_interface.def("get_selection_of_none_is_allowed", 
                         &AnalyzerSettingInterfaceChannel::GetSelectionOfNoneIsAllowed, R"pbdoc(
        Check if selecting "None" is allowed for this channel setting.
        
        Returns:
            bool: True if "None" selection is allowed
    )pbdoc");

    channel_interface.def("set_selection_of_none_is_allowed", 
                         &AnalyzerSettingInterfaceChannel::SetSelectionOfNoneIsAllowed, R"pbdoc(
        Set whether selecting "None" is allowed for this channel setting.
        
        Args:
            is_allowed: True to allow "None" selection
    )pbdoc", py::arg("is_allowed"));

    // AnalyzerSettingInterfaceNumberList - Dropdown list of numeric values
    py::class_<AnalyzerSettingInterfaceNumberList, AnalyzerSettingInterface> number_list_interface(
        m, "AnalyzerSettingInterfaceNumberList", R"pbdoc(
        Interface for dropdown lists with numeric values.
        
        This interface is used for settings where the user selects from a list of options,
        each with an associated numeric value (e.g., baud rates, bit counts, etc.)
    )pbdoc");

    number_list_interface.def(py::init<>(), "Default constructor");

    number_list_interface.def("get_number", &AnalyzerSettingInterfaceNumberList::GetNumber, R"pbdoc(
        Get the currently selected numeric value.
        
        Returns:
            float: The selected value
    )pbdoc");

    number_list_interface.def("set_number", &AnalyzerSettingInterfaceNumberList::SetNumber, R"pbdoc(
        Set the selected numeric value.
        
        Args:
            number: The value to select
    )pbdoc", py::arg("number"));

    number_list_interface.def("get_listbox_numbers_count", 
                             &AnalyzerSettingInterfaceNumberList::GetListboxNumbersCount, R"pbdoc(
        Get the number of items in the list.
        
        Returns:
            int: Number of items
    )pbdoc");

    number_list_interface.def("get_listbox_number", 
                             &AnalyzerSettingInterfaceNumberList::GetListboxNumber, R"pbdoc(
        Get the numeric value for an item at the specified index.
        
        Args:
            index: Index of the item
            
        Returns:
            float: Numeric value of the item
    )pbdoc", py::arg("index"));

    number_list_interface.def("get_listbox_strings_count", 
                             &AnalyzerSettingInterfaceNumberList::GetListboxStringsCount, R"pbdoc(
        Get the number of string labels in the list.
        
        Returns:
            int: Number of string labels
    )pbdoc");

    number_list_interface.def("get_listbox_string", 
                             &AnalyzerSettingInterfaceNumberList::GetListboxString, R"pbdoc(
        Get the string label for an item at the specified index.
        
        Args:
            index: Index of the item
            
        Returns:
            str: String label of the item
    )pbdoc", py::arg("index"));

    number_list_interface.def("add_number", &AnalyzerSettingInterfaceNumberList::AddNumber, R"pbdoc(
        Add a numeric option to the list.
        
        Args:
            number: Numeric value
            str: Display string
            tooltip: Tooltip text for this option
    )pbdoc", py::arg("number"), py::arg("str"), py::arg("tooltip"));

    number_list_interface.def("clear_numbers", &AnalyzerSettingInterfaceNumberList::ClearNumbers, R"pbdoc(
        Clear all options from the list.
    )pbdoc");

    // AnalyzerSettingInterfaceInteger - Integer input setting
    py::class_<AnalyzerSettingInterfaceInteger, AnalyzerSettingInterface> integer_interface(
        m, "AnalyzerSettingInterfaceInteger", R"pbdoc(
        Interface for integer input settings.
        
        This interface allows the user to enter an integer value within a specified range.
    )pbdoc");

    integer_interface.def(py::init<>(), "Default constructor");

    integer_interface.def("get_integer", &AnalyzerSettingInterfaceInteger::GetInteger, R"pbdoc(
        Get the current integer value.
        
        Returns:
            int: Current value
    )pbdoc");

    integer_interface.def("set_integer", &AnalyzerSettingInterfaceInteger::SetInteger, R"pbdoc(
        Set the integer value.
        
        Args:
            integer: Value to set
    )pbdoc", py::arg("integer"));

    integer_interface.def("get_max", &AnalyzerSettingInterfaceInteger::GetMax, R"pbdoc(
        Get the maximum allowed value.
        
        Returns:
            int: Maximum value
    )pbdoc");

    integer_interface.def("get_min", &AnalyzerSettingInterfaceInteger::GetMin, R"pbdoc(
        Get the minimum allowed value.
        
        Returns:
            int: Minimum value
    )pbdoc");

    integer_interface.def("set_max", &AnalyzerSettingInterfaceInteger::SetMax, R"pbdoc(
        Set the maximum allowed value.
        
        Args:
            max: Maximum value
    )pbdoc", py::arg("max"));

    integer_interface.def("set_min", &AnalyzerSettingInterfaceInteger::SetMin, R"pbdoc(
        Set the minimum allowed value.
        
        Args:
            min: Minimum value
    )pbdoc", py::arg("min"));

    // AnalyzerSettingInterfaceText - Text input setting
    py::class_<AnalyzerSettingInterfaceText, AnalyzerSettingInterface> text_interface(
        m, "AnalyzerSettingInterfaceText", R"pbdoc(
        Interface for text input settings.
        
        This interface allows the user to enter text, which can be used for various purposes
        such as naming, entering file paths, etc.
    )pbdoc");

    text_interface.def(py::init<>(), "Default constructor");

    text_interface.def("get_text", &AnalyzerSettingInterfaceText::GetText, R"pbdoc(
        Get the current text value.
        
        Returns:
            str: Current text
    )pbdoc");

    text_interface.def("set_text", &AnalyzerSettingInterfaceText::SetText, R"pbdoc(
        Set the text value.
        
        Args:
            text: Text to set
    )pbdoc", py::arg("text"));

    // Enum for TextType
    py::enum_<AnalyzerSettingInterfaceText::TextType>(text_interface, "TextType", R"pbdoc(
        Types of text input fields.
    )pbdoc")
        .value("NormalText", AnalyzerSettingInterfaceText::NormalText, "Regular text input")
        .value("FilePath", AnalyzerSettingInterfaceText::FilePath, "File path input with browse button")
        .value("FolderPath", AnalyzerSettingInterfaceText::FolderPath, "Folder path input with browse button");

    text_interface.def("get_text_type", &AnalyzerSettingInterfaceText::GetTextType, R"pbdoc(
        Get the type of text input.
        
        Returns:
            TextType: The text field type
    )pbdoc");

    text_interface.def("set_text_type", &AnalyzerSettingInterfaceText::SetTextType, R"pbdoc(
        Set the type of text input.
        
        Args:
            text_type: The text field type
    )pbdoc", py::arg("text_type"));

    // AnalyzerSettingInterfaceBool - Boolean (checkbox) setting
    py::class_<AnalyzerSettingInterfaceBool, AnalyzerSettingInterface> bool_interface(
        m, "AnalyzerSettingInterfaceBool", R"pbdoc(
        Interface for boolean (checkbox) settings.
        
        This interface presents a checkbox with a label for toggling boolean options.
    )pbdoc");

    bool_interface.def(py::init<>(), "Default constructor");

    bool_interface.def("get_value", &AnalyzerSettingInterfaceBool::GetValue, R"pbdoc(
        Get the current boolean value.
        
        Returns:
            bool: Current value
    )pbdoc");

    bool_interface.def("set_value", &AnalyzerSettingInterfaceBool::SetValue, R"pbdoc(
        Set the boolean value.
        
        Args:
            value: Value to set
    )pbdoc", py::arg("value"));

    bool_interface.def("get_check_box_text", &AnalyzerSettingInterfaceBool::GetCheckBoxText, R"pbdoc(
        Get the text displayed next to the checkbox.
        
        Returns:
            str: Checkbox text
    )pbdoc");

    bool_interface.def("set_check_box_text", &AnalyzerSettingInterfaceBool::SetCheckBoxText, R"pbdoc(
        Set the text displayed next to the checkbox.
        
        Args:
            text: Checkbox text
    )pbdoc", py::arg("text"));

    // AnalyzerSettings - Base class for analyzer settings
    py::class_<AnalyzerSettings, PyAnalyzerSettings> analyzer_settings(m, "AnalyzerSettings", R"pbdoc(
        Base class for analyzer settings.
        
        This class handles the configuration settings for an analyzer. It manages
        the settings interfaces, load/save functionality, and validation.
        
        Must be subclassed to implement specific analyzer settings.
    )pbdoc");

    analyzer_settings.def(py::init<>(), "Default constructor");

    // Pure virtual methods
    analyzer_settings.def("set_settings_from_interfaces", &AnalyzerSettings::SetSettingsFromInterfaces, R"pbdoc(
        Apply settings from interfaces to the analyzer.
        
        Validates and applies settings from the UI interfaces to the analyzer configuration.
        
        Returns:
            bool: True if settings are valid and were applied successfully
            
        Must be implemented by derived classes.
    )pbdoc");

    analyzer_settings.def("load_settings", &AnalyzerSettings::LoadSettings, R"pbdoc(
        Load settings from a string.
        
        Args:
            settings: String containing serialized settings
            
        Must be implemented by derived classes.
    )pbdoc", py::arg("settings"));

    analyzer_settings.def("save_settings", &AnalyzerSettings::SaveSettings, R"pbdoc(
        Save settings to a string.
        
        Returns:
            str: Serialized settings string
            
        Must be implemented by derived classes.
    )pbdoc");

    // Virtual methods with default implementations
    analyzer_settings.def("get_setting_brief", &AnalyzerSettings::GetSettingBrief, R"pbdoc(
        Get a brief description of the current settings.
        
        Returns:
            str: Brief description string
    )pbdoc");

    // Protected methods exposed for use in derived classes
    analyzer_settings.def("clear_channels", &AnalyzerSettings::ClearChannels, R"pbdoc(
        Clear all reported channels.
        
        Call this before adding channels if the channel configuration changes.
    )pbdoc");

    analyzer_settings.def("add_channel", &AnalyzerSettings::AddChannel, R"pbdoc(
        Add a channel to the analyzer.
        
        Args:
            channel: Channel to add
            channel_label: Label for the channel
            is_used: Whether the channel is used by the analyzer
    )pbdoc", py::arg("channel"), py::arg("channel_label"), py::arg("is_used"));

    analyzer_settings.def("set_error_text", &AnalyzerSettings::SetErrorText, R"pbdoc(
        Set error text for invalid settings.
        
        Args:
            error_text: Error message to display
    )pbdoc", py::arg("error_text"));

    analyzer_settings.def("add_interface", &AnalyzerSettings::AddInterface, R"pbdoc(
        Add a setting interface.
        
        Args:
            analyzer_setting_interface: Interface to add
    )pbdoc", py::arg("analyzer_setting_interface"));

    analyzer_settings.def("add_export_option", &AnalyzerSettings::AddExportOption, R"pbdoc(
        Add an export option to the analyzer.
        
        Args:
            user_id: User-defined ID for the export option
            menu_text: Text to display in the export menu
    )pbdoc", py::arg("user_id"), py::arg("menu_text"));

    analyzer_settings.def("add_export_extension", &AnalyzerSettings::AddExportExtension, R"pbdoc(
        Add a file extension for an export option.
        
        Args:
            user_id: User-defined ID for the export option
            extension_description: Description of the file type
            extension: File extension (e.g., "csv")
    )pbdoc", py::arg("user_id"), py::arg("extension_description"), py::arg("extension"));

    analyzer_settings.def("set_return_string", &AnalyzerSettings::SetReturnString, R"pbdoc(
        Set the string to return from SaveSettings.
        
        Args:
            str: Settings string
            
        Returns:
            str: The same string that was passed in
    )pbdoc", py::arg("str"));

    // Public utility methods
    analyzer_settings.def("get_use_system_display_base", &AnalyzerSettings::GetUseSystemDisplayBase, R"pbdoc(
        Check if the analyzer uses the system display base.
        
        Returns:
            bool: True if using system display base
    )pbdoc");

    analyzer_settings.def("set_use_system_display_base", &AnalyzerSettings::SetUseSystemDisplayBase, R"pbdoc(
        Set whether to use the system display base.
        
        Args:
            use_system_display_base: True to use system display base
    )pbdoc", py::arg("use_system_display_base"));

    analyzer_settings.def("get_analyzer_display_base", &AnalyzerSettings::GetAnalyzerDisplayBase, R"pbdoc(
        Get the display base used by the analyzer.
        
        Returns:
            DisplayBase: Current display base
    )pbdoc");

    analyzer_settings.def("set_analyzer_display_base", &AnalyzerSettings::SetAnalyzerDisplayBase, R"pbdoc(
        Set the display base used by the analyzer.
        
        Args:
            analyzer_display_base: Display base to use
    )pbdoc", py::arg("analyzer_display_base"));
}