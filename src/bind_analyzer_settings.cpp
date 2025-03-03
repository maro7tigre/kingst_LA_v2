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

    // Pure virtual methods with trampolines to Python
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

    // Virtual methods with default implementations
    const char* GetSettingBrief() override {
        PYBIND11_OVERRIDE(
            const char*,         // Return type
            AnalyzerSettings,    // Parent class
            GetSettingBrief      // Method name
        );
    }
};

void init_analyzer_settings(py::module_ &m) {
    // Bind the AnalyzerInterfaceTypeId enum
    py::enum_<AnalyzerInterfaceTypeId>(m, "AnalyzerInterfaceTypeId", R"pbdoc(
        Enum defining the types of analyzer setting interfaces.
        
        Used internally to identify the specific type of a setting interface.
    )pbdoc")
        .value("INTERFACE_BASE", INTERFACE_BASE, R"pbdoc(Base interface type.)pbdoc")
        .value("INTERFACE_CHANNEL", INTERFACE_CHANNEL, R"pbdoc(Channel selection interface.)pbdoc")
        .value("INTERFACE_NUMBER_LIST", INTERFACE_NUMBER_LIST, R"pbdoc(Dropdown with numeric values.)pbdoc")
        .value("INTERFACE_INTEGER", INTERFACE_INTEGER, R"pbdoc(Integer input interface.)pbdoc")
        .value("INTERFACE_TEXT", INTERFACE_TEXT, R"pbdoc(Text input interface.)pbdoc")
        .value("INTERFACE_BOOL", INTERFACE_BOOL, R"pbdoc(Boolean checkbox interface.)pbdoc");

    // AnalyzerSettingInterface - Base class for all setting interfaces
    py::class_<AnalyzerSettingInterface> setting_interface(m, "AnalyzerSettingInterface", R"pbdoc(
        Base class for all analyzer setting interfaces.
        
        Setting interfaces are used to create UI controls for configuring analyzers.
        This is the base class that all specific interface types inherit from.
    )pbdoc");

    // Virtual methods
    setting_interface.def("get_type", &AnalyzerSettingInterface::GetType, R"pbdoc(
        Get the type of this setting interface.
        
        Returns:
            AnalyzerInterfaceTypeId: The type identifier
    )pbdoc");

    // Utility methods
    setting_interface.def("get_tool_tip", &AnalyzerSettingInterface::GetToolTip, R"pbdoc(
        Get the tooltip text for this setting.
        
        Returns:
            str: Tooltip text
    )pbdoc", py::return_value_policy::reference);

    setting_interface.def("get_title", &AnalyzerSettingInterface::GetTitle, R"pbdoc(
        Get the title/label for this setting.
        
        Returns:
            str: Title text
    )pbdoc", py::return_value_policy::reference);

    setting_interface.def("is_disabled", &AnalyzerSettingInterface::IsDisabled, R"pbdoc(
        Check if this setting is disabled.
        
        Returns:
            bool: True if the setting is disabled
    )pbdoc");

    setting_interface.def("set_title_and_tooltip", &AnalyzerSettingInterface::SetTitleAndTooltip, R"pbdoc(
        Set the title and tooltip for this setting.
        
        Args:
            title (str): Title/label text
            tooltip (str): Tooltip text
    )pbdoc", py::arg("title"), py::arg("tooltip"));

    // AnalyzerSettingInterfaceChannel - Channel selection setting
    py::class_<AnalyzerSettingInterfaceChannel, AnalyzerSettingInterface> channel_interface(
        m, "AnalyzerSettingInterfaceChannel", R"pbdoc(
        Interface for channel selection settings.
        
        This interface allows the user to select a channel from the available channels.
    )pbdoc");

    // Constructor
    channel_interface.def(py::init<>(), R"pbdoc(
        Default constructor.
        
        Creates a new channel interface with default settings.
    )pbdoc");

    // Virtual methods
    channel_interface.def("get_type", &AnalyzerSettingInterfaceChannel::GetType, R"pbdoc(
        Get the type of this setting interface.
        
        Returns:
            AnalyzerInterfaceTypeId: Always returns INTERFACE_CHANNEL
    )pbdoc");

    // Utility methods
    channel_interface.def("get_channel", &AnalyzerSettingInterfaceChannel::GetChannel, R"pbdoc(
        Get the currently selected channel.
        
        Returns:
            Channel: The selected channel
    )pbdoc");

    channel_interface.def("set_channel", &AnalyzerSettingInterfaceChannel::SetChannel, R"pbdoc(
        Set the selected channel.
        
        Args:
            channel (Channel): The channel to select
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
            is_allowed (bool): True to allow "None" selection
    )pbdoc", py::arg("is_allowed"));

    // AnalyzerSettingInterfaceNumberList - Dropdown list of numeric values
    py::class_<AnalyzerSettingInterfaceNumberList, AnalyzerSettingInterface> number_list_interface(
        m, "AnalyzerSettingInterfaceNumberList", R"pbdoc(
        Interface for dropdown lists with numeric values.
        
        This interface is used for settings where the user selects from a list of options,
        each with an associated numeric value (e.g., baud rates, bit counts, etc.)
    )pbdoc");

    // Constructor
    number_list_interface.def(py::init<>(), R"pbdoc(
        Default constructor.
        
        Creates a new number list interface with an empty list of options.
    )pbdoc");

    // Virtual methods
    number_list_interface.def("get_type", &AnalyzerSettingInterfaceNumberList::GetType, R"pbdoc(
        Get the type of this setting interface.
        
        Returns:
            AnalyzerInterfaceTypeId: Always returns INTERFACE_NUMBER_LIST
    )pbdoc");

    // Utility methods
    number_list_interface.def("get_number", &AnalyzerSettingInterfaceNumberList::GetNumber, R"pbdoc(
        Get the currently selected numeric value.
        
        Returns:
            float: The selected value
    )pbdoc");

    number_list_interface.def("set_number", &AnalyzerSettingInterfaceNumberList::SetNumber, R"pbdoc(
        Set the selected numeric value.
        
        Args:
            number (float): The value to select
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
            index (int): Index of the item (0-based)
            
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
            index (int): Index of the item (0-based)
            
        Returns:
            str: String label of the item
    )pbdoc", py::arg("index"), py::return_value_policy::reference);

    number_list_interface.def("get_listbox_tooltips_count", 
                              &AnalyzerSettingInterfaceNumberList::GetListboxTooltipsCount, R"pbdoc(
        Get the number of tooltips in the list.
        
        Returns:
            int: Number of tooltips
    )pbdoc");

    number_list_interface.def("get_listbox_tooltip", 
                              &AnalyzerSettingInterfaceNumberList::GetListboxTooltip, R"pbdoc(
        Get the tooltip for an item at the specified index.
        
        Args:
            index (int): Index of the item (0-based)
            
        Returns:
            str: Tooltip text for the item
    )pbdoc", py::arg("index"), py::return_value_policy::reference);

    number_list_interface.def("add_number", &AnalyzerSettingInterfaceNumberList::AddNumber, R"pbdoc(
        Add a numeric option to the list.
        
        Args:
            number (float): Numeric value
            str (str): Display string
            tooltip (str): Tooltip text for this option
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

    // Constructor
    integer_interface.def(py::init<>(), R"pbdoc(
        Default constructor.
        
        Creates a new integer interface with default range.
    )pbdoc");

    // Virtual methods
    integer_interface.def("get_type", &AnalyzerSettingInterfaceInteger::GetType, R"pbdoc(
        Get the type of this setting interface.
        
        Returns:
            AnalyzerInterfaceTypeId: Always returns INTERFACE_INTEGER
    )pbdoc");

    // Utility methods
    integer_interface.def("get_integer", &AnalyzerSettingInterfaceInteger::GetInteger, R"pbdoc(
        Get the current integer value.
        
        Returns:
            int: Current value
    )pbdoc");

    integer_interface.def("set_integer", &AnalyzerSettingInterfaceInteger::SetInteger, R"pbdoc(
        Set the integer value.
        
        Args:
            integer (int): Value to set
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
            max (int): Maximum value
    )pbdoc", py::arg("max"));

    integer_interface.def("set_min", &AnalyzerSettingInterfaceInteger::SetMin, R"pbdoc(
        Set the minimum allowed value.
        
        Args:
            min (int): Minimum value
    )pbdoc", py::arg("min"));

    // AnalyzerSettingInterfaceText - Text input setting
    py::class_<AnalyzerSettingInterfaceText, AnalyzerSettingInterface> text_interface(
        m, "AnalyzerSettingInterfaceText", R"pbdoc(
        Interface for text input settings.
        
        This interface allows the user to enter text, which can be used for various purposes
        such as naming, entering file paths, etc.
    )pbdoc");

    // Constructor
    text_interface.def(py::init<>(), R"pbdoc(
        Default constructor.
        
        Creates a new text interface with empty text.
    )pbdoc");

    // Virtual methods
    text_interface.def("get_type", &AnalyzerSettingInterfaceText::GetType, R"pbdoc(
        Get the type of this setting interface.
        
        Returns:
            AnalyzerInterfaceTypeId: Always returns INTERFACE_TEXT
    )pbdoc");

    // Utility methods
    text_interface.def("get_text", &AnalyzerSettingInterfaceText::GetText, R"pbdoc(
        Get the current text value.
        
        Returns:
            str: Current text
    )pbdoc", py::return_value_policy::reference);

    text_interface.def("set_text", &AnalyzerSettingInterfaceText::SetText, R"pbdoc(
        Set the text value.
        
        Args:
            text (str): Text to set
    )pbdoc", py::arg("text"));

    // Enum for TextType
    py::enum_<AnalyzerSettingInterfaceText::TextType>(text_interface, "TextType", R"pbdoc(
        Types of text input fields.
    )pbdoc")
        .value("NormalText", AnalyzerSettingInterfaceText::NormalText, R"pbdoc(Regular text input)pbdoc")
        .value("FilePath", AnalyzerSettingInterfaceText::FilePath, R"pbdoc(File path input with browse button)pbdoc")
        .value("FolderPath", AnalyzerSettingInterfaceText::FolderPath, R"pbdoc(Folder path input with browse button)pbdoc");

    text_interface.def("get_text_type", &AnalyzerSettingInterfaceText::GetTextType, R"pbdoc(
        Get the type of text input.
        
        Returns:
            TextType: The text field type
    )pbdoc");

    text_interface.def("set_text_type", &AnalyzerSettingInterfaceText::SetTextType, R"pbdoc(
        Set the type of text input.
        
        Args:
            text_type (TextType): The text field type
    )pbdoc", py::arg("text_type"));

    // AnalyzerSettingInterfaceBool - Boolean (checkbox) setting
    py::class_<AnalyzerSettingInterfaceBool, AnalyzerSettingInterface> bool_interface(
        m, "AnalyzerSettingInterfaceBool", R"pbdoc(
        Interface for boolean (checkbox) settings.
        
        This interface presents a checkbox with a label for toggling boolean options.
    )pbdoc");

    // Constructor
    bool_interface.def(py::init<>(), R"pbdoc(
        Default constructor.
        
        Creates a new boolean interface with default value of false.
    )pbdoc");

    // Virtual methods
    bool_interface.def("get_type", &AnalyzerSettingInterfaceBool::GetType, R"pbdoc(
        Get the type of this setting interface.
        
        Returns:
            AnalyzerInterfaceTypeId: Always returns INTERFACE_BOOL
    )pbdoc");

    // Utility methods
    bool_interface.def("get_value", &AnalyzerSettingInterfaceBool::GetValue, R"pbdoc(
        Get the current boolean value.
        
        Returns:
            bool: Current value
    )pbdoc");

    bool_interface.def("set_value", &AnalyzerSettingInterfaceBool::SetValue, R"pbdoc(
        Set the boolean value.
        
        Args:
            value (bool): Value to set
    )pbdoc", py::arg("value"));

    bool_interface.def("get_check_box_text", &AnalyzerSettingInterfaceBool::GetCheckBoxText, R"pbdoc(
        Get the text displayed next to the checkbox.
        
        Returns:
            str: Checkbox text
    )pbdoc", py::return_value_policy::reference);

    bool_interface.def("set_check_box_text", &AnalyzerSettingInterfaceBool::SetCheckBoxText, R"pbdoc(
        Set the text displayed next to the checkbox.
        
        Args:
            text (str): Checkbox text
    )pbdoc", py::arg("text"));

    // AnalyzerSettings - Base class for analyzer settings
    py::class_<AnalyzerSettings, PyAnalyzerSettings> analyzer_settings(m, "AnalyzerSettings", R"pbdoc(
        Base class for analyzer settings.
        
        This class handles the configuration settings for an analyzer. It manages
        the settings interfaces, load/save functionality, and validation.
        
        Must be subclassed to implement specific analyzer settings.
    )pbdoc");

    // Constructor
    analyzer_settings.def(py::init<>(), R"pbdoc(
        Default constructor.
        
        Creates a new analyzer settings object with no channels or interfaces.
    )pbdoc");

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
            settings (str): String containing serialized settings
            
        Must be implemented by derived classes.
    )pbdoc", py::arg("settings"));

    analyzer_settings.def("save_settings", &AnalyzerSettings::SaveSettings, R"pbdoc(
        Save settings to a string.
        
        Returns:
            str: Serialized settings string
            
        Must be implemented by derived classes.
    )pbdoc", py::return_value_policy::reference);

    // Virtual methods with default implementations
    analyzer_settings.def("get_setting_brief", &AnalyzerSettings::GetSettingBrief, R"pbdoc(
        Get a brief description of the current settings.
        
        Returns:
            str: Brief description string
    )pbdoc", py::return_value_policy::reference);

    // Protected methods exposed for use in derived classes
    analyzer_settings.def("clear_channels", &AnalyzerSettings::ClearChannels, R"pbdoc(
        Clear all reported channels.
        
        Call this before adding channels if the channel configuration changes.
    )pbdoc");

    analyzer_settings.def("add_channel", &AnalyzerSettings::AddChannel, R"pbdoc(
        Add a channel to the analyzer.
        
        Args:
            channel (Channel): Channel to add
            channel_label (str): Label for the channel
            is_used (bool): Whether the channel is used by the analyzer
    )pbdoc", py::arg("channel"), py::arg("channel_label"), py::arg("is_used"));

    analyzer_settings.def("set_error_text", &AnalyzerSettings::SetErrorText, R"pbdoc(
        Set error text for invalid settings.
        
        Args:
            error_text (str): Error message to display
    )pbdoc", py::arg("error_text"));

    analyzer_settings.def("add_interface", &AnalyzerSettings::AddInterface, R"pbdoc(
        Add a setting interface.
        
        Args:
            analyzer_setting_interface (AnalyzerSettingInterface): Interface to add
            
        Note:
            The caller must maintain ownership of the interface object.
    )pbdoc", py::arg("analyzer_setting_interface"));

    analyzer_settings.def("add_export_option", &AnalyzerSettings::AddExportOption, R"pbdoc(
        Add an export option to the analyzer.
        
        Args:
            user_id (int): User-defined ID for the export option
            menu_text (str): Text to display in the export menu
    )pbdoc", py::arg("user_id"), py::arg("menu_text"));

    analyzer_settings.def("add_export_extension", &AnalyzerSettings::AddExportExtension, R"pbdoc(
        Add a file extension for an export option.
        
        Args:
            user_id (int): User-defined ID for the export option
            extension_description (str): Description of the file type
            extension (str): File extension (e.g., "csv")
    )pbdoc", py::arg("user_id"), py::arg("extension_description"), py::arg("extension"));

    analyzer_settings.def("set_return_string", &AnalyzerSettings::SetReturnString, R"pbdoc(
        Set the string to return from SaveSettings.
        
        Args:
            str (str): Settings string
            
        Returns:
            str: The same string that was passed in
            
        Note:
            This method should be used in the SaveSettings implementation.
    )pbdoc", py::arg("str"), py::return_value_policy::reference);

    // Public utility methods
    analyzer_settings.def("get_settings_interfaces_count", &AnalyzerSettings::GetSettingsInterfacesCount, R"pbdoc(
        Get the number of setting interfaces.
        
        Returns:
            int: Number of interfaces
    )pbdoc");

    analyzer_settings.def("get_settings_interface", 
                          &AnalyzerSettings::GetSettingsInterface, R"pbdoc(
        Get a setting interface by index.
        
        Args:
            index (int): Index of the interface (0-based)
            
        Returns:
            AnalyzerSettingInterface: The interface at the specified index
            
        Note:
            The returned interface is owned by the settings object.
    )pbdoc", py::arg("index"), py::return_value_policy::reference);

    analyzer_settings.def("get_file_extension_count", &AnalyzerSettings::GetFileExtensionCount, R"pbdoc(
        Get the number of file extensions for an export option.
        
        Args:
            index_id (int): Export option ID
            
        Returns:
            int: Number of file extensions
    )pbdoc", py::arg("index_id"));

    analyzer_settings.def("get_file_extension", [](AnalyzerSettings& self, U32 index_id, U32 extension_id) {
        const char* extension_description = nullptr;
        const char* extension = nullptr;
        self.GetFileExtension(index_id, extension_id, &extension_description, &extension);
        return py::make_tuple(
            std::string(extension_description ? extension_description : ""),
            std::string(extension ? extension : "")
        );
    }, R"pbdoc(
        Get file extension information for an export option.
        
        Args:
            index_id (int): Export option ID
            extension_id (int): Extension index
            
        Returns:
            tuple: (extension_description, extension)
    )pbdoc", py::arg("index_id"), py::arg("extension_id"));

    analyzer_settings.def("get_channels_count", &AnalyzerSettings::GetChannelsCount, R"pbdoc(
        Get the number of channels used by the analyzer.
        
        Returns:
            int: Number of channels
    )pbdoc");

    analyzer_settings.def("get_channel", [](AnalyzerSettings& self, U32 index) {
        const char* channel_label = nullptr;
        bool channel_is_used = false;
        Channel channel = self.GetChannel(index, &channel_label, &channel_is_used);
        return py::make_tuple(
            channel,
            std::string(channel_label ? channel_label : ""),
            channel_is_used
        );
    }, R"pbdoc(
        Get information about a channel by index.
        
        Args:
            index (int): Channel index (0-based)
            
        Returns:
            tuple: (Channel, label, is_used)
    )pbdoc", py::arg("index"));

    analyzer_settings.def("get_export_options_count", &AnalyzerSettings::GetExportOptionsCount, R"pbdoc(
        Get the number of export options.
        
        Returns:
            int: Number of export options
    )pbdoc");

    analyzer_settings.def("get_export_option", [](AnalyzerSettings& self, U32 index) {
        U32 user_id = 0;
        const char* menu_text = nullptr;
        self.GetExportOption(index, &user_id, &menu_text);
        return py::make_tuple(
            user_id,
            std::string(menu_text ? menu_text : "")
        );
    }, R"pbdoc(
        Get information about an export option by index.
        
        Args:
            index (int): Export option index (0-based)
            
        Returns:
            tuple: (user_id, menu_text)
    )pbdoc", py::arg("index"));

    analyzer_settings.def("get_save_error_message", &AnalyzerSettings::GetSaveErrorMessage, R"pbdoc(
        Get the error message from the last save operation.
        
        Returns:
            str: Error message or empty string if no error
    )pbdoc", py::return_value_policy::reference);

    analyzer_settings.def("get_use_system_display_base", &AnalyzerSettings::GetUseSystemDisplayBase, R"pbdoc(
        Check if the analyzer uses the system display base.
        
        Returns:
            bool: True if using system display base
    )pbdoc");

    analyzer_settings.def("set_use_system_display_base", &AnalyzerSettings::SetUseSystemDisplayBase, R"pbdoc(
        Set whether to use the system display base.
        
        Args:
            use_system_display_base (bool): True to use system display base
    )pbdoc", py::arg("use_system_display_base"));

    analyzer_settings.def("get_analyzer_display_base", &AnalyzerSettings::GetAnalyzerDisplayBase, R"pbdoc(
        Get the display base used by the analyzer.
        
        Returns:
            DisplayBase: Current display base
    )pbdoc");

    analyzer_settings.def("set_analyzer_display_base", &AnalyzerSettings::SetAnalyzerDisplayBase, R"pbdoc(
        Set the display base used by the analyzer.
        
        Args:
            analyzer_display_base (DisplayBase): Display base to use
    )pbdoc", py::arg("analyzer_display_base"));
}